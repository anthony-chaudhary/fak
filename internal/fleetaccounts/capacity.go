package fleetaccounts

import (
	"fmt"
	"sort"
	"strings"

	configaccounts "github.com/anthony-chaudhary/fak/internal/accounts"
	"github.com/anthony-chaudhary/fak/internal/strmatch"
)

const (
	CapacityPreflightSchema = "fleet-capacity-preflight/1"

	CapacityFresh        = "fresh"
	CapacityStale        = "stale"
	CapacityBlockedUntil = "blocked-until"

	// The recoverability split over a non-offerable seat (#3580). A walled seat is not
	// uniformly unavailable: CapacityRecoverable names the seats one operator action away
	// from serving right now (a `login`, or a credential re-read for the #3216 mislabel
	// where a re-logged-in seat still reads as logged-out), while CapacityHardWalled names
	// the seats only a cap reset, a credit top-up, or an upstream access-restore clears.
	// Collapsing both into stale/blocked-until left recoverable supply on the floor: at
	// 0/24 live, the cheapest new seat is often a walled one the fleet already owns.
	CapacityRecoverable = "recoverable"
	CapacityHardWalled  = "hard"
)

// The closed suggested-action vocabulary that goes with the split. It mirrors the words
// `fak accounts doctor` already prints on its recovery worklist, so the detection layer and
// the operator view name the same action for the same seat.
const (
	capacityActionLogin         = "login"
	capacityActionReReadCred    = "re-read credential"
	capacityActionWaitReset     = "wait for reset"
	capacityActionTopUp         = "top up credit"
	capacityActionRestoreAccess = "restore access"
)

// CapacityAccount is the per-seat auth/capacity state exposed before dispatch.
type CapacityAccount struct {
	Seat           string  `json:"seat"`
	Account        string  `json:"account"`
	Tag            string  `json:"tag"`
	Product        string  `json:"product"`
	Model          *string `json:"model,omitempty"`
	ModelTier      *int    `json:"model_tier,omitempty"`
	State          string  `json:"state"`
	StateLabel     string  `json:"state_label"`
	BlockedUntil   string  `json:"blocked_until,omitempty"`
	Reason         string  `json:"reason,omitempty"`
	LoginStatus    *string `json:"login_status,omitempty"`
	CanServe       *bool   `json:"can_serve,omitempty"`
	SessionCap     int     `json:"session_cap,omitempty"`
	ActiveSessions int     `json:"active_sessions"`
	LiveSessions   int     `json:"live_sessions"`
	StatusSource   string  `json:"status_source,omitempty"`
	// Recovery is CapacityRecoverable or CapacityHardWalled on a non-offerable seat, and
	// RecoveryAction is the operator action that reclaims it. Both stay empty on a fresh
	// seat, so a fully-offerable roster renders byte-identically to before (#3580).
	Recovery       string `json:"recovery,omitempty"`
	RecoveryAction string `json:"recovery_action,omitempty"`
}

// CapacityPreflight is the dispatch-sizing report: true_concurrent_ceiling is the
// number of fresh distinct seats a dispatcher may safely size a wave against.
type CapacityPreflight struct {
	Schema                string `json:"schema"`
	Product               string `json:"product"`
	Required              int    `json:"required,omitempty"`
	OK                    bool   `json:"ok"`
	Verdict               string `json:"verdict"`
	TrueConcurrentCeiling int    `json:"true_concurrent_ceiling"`
	FreshSeats            int    `json:"fresh_seats"`
	StaleSeats            int    `json:"stale_seats"`
	BlockedSeats          int    `json:"blocked_seats"`
	TotalSeats            int    `json:"total_seats"`
	// RecoverableSeats is the servable-seat gain available right now from seats the fleet
	// already owns: the session slots behind CapacityRecoverable seats, which one operator
	// action each would return to the ceiling. HardWalledSeats is the remainder that only
	// time/billing/admin clears. Both omit when zero, so a fully-offerable roster is
	// byte-identical to the pre-#3580 report.
	RecoverableSeats int               `json:"recoverable_seats,omitempty"`
	HardWalledSeats  int               `json:"hard_walled_seats,omitempty"`
	Reason           string            `json:"reason"`
	Accounts         []CapacityAccount `json:"accounts"`
}

// BuildCapacityPreflight folds annotated account rows into the proactive seat
// ceiling. product filters by product family; "" or "all" includes every product.
func BuildCapacityPreflight(rows []Account, product string, required int) CapacityPreflight {
	product = strings.ToLower(strings.TrimSpace(product))
	if product == "" {
		product = "all"
	}
	rep := CapacityPreflight{
		Schema:   CapacityPreflightSchema,
		Product:  product,
		Required: required,
		OK:       true,
		Verdict:  "OK",
	}
	workers := []Account{}
	for _, row := range rows {
		if !RoutableWorker(row) {
			continue
		}
		rowProduct := strings.ToLower(productOf(row))
		if product != "all" && rowProduct != product {
			continue
		}
		workers = append(workers, row)
	}
	for _, row := range uniquePoolAccounts(workers) {
		acct := capacityAccount(row)
		rep.Accounts = append(rep.Accounts, acct)
		capacity := AccountSessionCap(row)
		if capacity <= 0 {
			continue
		}
		rep.TotalSeats += capacity
		switch acct.State {
		case CapacityFresh:
			rep.FreshSeats += capacity
		case CapacityBlockedUntil:
			rep.BlockedSeats += capacity
		default:
			rep.StaleSeats += capacity
		}
		switch acct.Recovery {
		case CapacityRecoverable:
			rep.RecoverableSeats += capacity
		case CapacityHardWalled:
			rep.HardWalledSeats += capacity
		}
	}
	sort.SliceStable(rep.Accounts, func(i, j int) bool {
		ri, rj := capacityRank(rep.Accounts[i].State), capacityRank(rep.Accounts[j].State)
		if ri != rj {
			return ri < rj
		}
		if rep.Accounts[i].Product != rep.Accounts[j].Product {
			return rep.Accounts[i].Product < rep.Accounts[j].Product
		}
		return rep.Accounts[i].Tag < rep.Accounts[j].Tag
	})
	rep.TrueConcurrentCeiling = rep.FreshSeats
	rep.Reason = fmt.Sprintf("%d fresh session slot(s), %d stale, %d blocked", rep.FreshSeats, rep.StaleSeats, rep.BlockedSeats)
	if required > 0 && rep.TrueConcurrentCeiling < required {
		rep.OK = false
		rep.Verdict = "UNDER_CAPACITY"
		rep.Reason = fmt.Sprintf("requires %d fresh session slot(s), only %d available; dispatch must downsize before spawning", required, rep.TrueConcurrentCeiling)
	}
	return rep
}

func capacityAccount(row Account) CapacityAccount {
	state, until, reason := capacityState(row)
	label := state
	if state == CapacityBlockedUntil {
		label = "blocked-until-" + capacityFirstNonEmpty(until, "unknown")
	}
	recovery, action := capacityRecovery(row, state)
	return CapacityAccount{
		Seat:           PoolKey(row),
		Account:        row.Account,
		Tag:            row.Tag,
		Product:        productOf(row),
		Model:          row.Model,
		ModelTier:      row.ModelTier,
		State:          state,
		StateLabel:     label,
		BlockedUntil:   until,
		Reason:         reason,
		LoginStatus:    row.LoginStatus,
		CanServe:       row.CanServe,
		SessionCap:     AccountSessionCap(row),
		ActiveSessions: derefInt(row.ActiveSessions),
		LiveSessions:   derefInt(row.LiveSessions),
		StatusSource:   derefStr(row.StatusSource),
		Recovery:       recovery,
		RecoveryAction: action,
	}
}

// capacityRecovery splits a non-offerable seat by whether an operator action reclaims it
// now, and names that action (#3580). A fresh seat carries no class — it is already serving,
// so reclaiming it grows nothing.
//
// Precedence is deliberate. A usage/weekly wall outranks any login wall: re-logging in
// re-auths the SAME capped account and hits the SAME ceiling, so a blocked-until seat is
// hard no matter what its login status reads. Only below that does the login status decide,
// and an identity_mismatch is reclaimed by re-reading the credential rather than a fresh
// login — that is the #3216 mislabel, where a seat that HAS been re-logged-in is still read
// as logged out. A seat whose wall we cannot name stays hard with no action rather than
// being advertised as reclaimable: over-claiming recoverable supply would send an operator
// to a dead end.
func capacityRecovery(row Account, state string) (class, action string) {
	switch state {
	case CapacityFresh:
		return "", ""
	case CapacityBlockedUntil:
		return CapacityHardWalled, capacityActionWaitReset
	}
	switch configaccounts.LoginStatus(derefStr(row.LoginStatus)) {
	case configaccounts.LoginIdentityMismatch:
		return CapacityRecoverable, capacityActionReReadCred
	case configaccounts.LoginNeedsLogin:
		return CapacityRecoverable, capacityActionLogin
	}
	switch strings.ToLower(derefStr(row.BlockKind)) {
	case "auth":
		return CapacityRecoverable, capacityActionLogin
	case "usage":
		return CapacityHardWalled, capacityActionWaitReset
	case "credit":
		return CapacityHardWalled, capacityActionTopUp
	case "access":
		return CapacityHardWalled, capacityActionRestoreAccess
	}
	return CapacityHardWalled, ""
}

func capacityState(row Account) (state, until, reason string) {
	if accountCanBeOffered(row) {
		return CapacityFresh, "", "ready to serve"
	}
	reason = capacityFirstNonEmpty(derefStr(row.BlockReason), row.Reason, "not currently offerable")
	blockKind := strings.ToLower(derefStr(row.BlockKind))
	// weekly-first "when does it free up", shared with the cap-disambiguation core so the
	// capacity view and the runtime fold agree on precedence.
	reset := CapState{Weekly: derefStr(row.Weekly), Reset: derefStr(row.Reset)}.EffectiveFreeUp()
	if derefBool(row.Blocked) && (blockKind == "usage" || derefBool(row.Throttled)) {
		return CapacityBlockedUntil, reset, reason
	}
	return CapacityStale, "", reason
}

func capacityRank(state string) int {
	switch state {
	case CapacityFresh:
		return 0
	case CapacityBlockedUntil:
		return 1
	default:
		return 2
	}
}

func productOf(row Account) string {
	if row.Product != "" {
		return row.Product
	}
	return AccountProduct(row.Account)
}

// capacityFirstNonEmpty is strmatch.FirstTrimmed: first value with non-whitespace text,
// returned trimmed. Kept as a named local so capstate.go's cross-reference still resolves,
// but the rule itself now has exactly one definition instead of a private copy here.
func capacityFirstNonEmpty(vals ...string) string { return strmatch.FirstTrimmed(vals...) }
