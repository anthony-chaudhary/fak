package fleetaccounts

import (
	"fmt"
	"sort"
	"strings"
)

const (
	CapacityPreflightSchema = "fleet-capacity-preflight/1"

	CapacityFresh        = "fresh"
	CapacityStale        = "stale"
	CapacityBlockedUntil = "blocked-until"
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
	ActiveSessions int     `json:"active_sessions"`
	LiveSessions   int     `json:"live_sessions"`
	StatusSource   string  `json:"status_source,omitempty"`
}

// CapacityPreflight is the dispatch-sizing report: true_concurrent_ceiling is the
// number of fresh distinct seats a dispatcher may safely size a wave against.
type CapacityPreflight struct {
	Schema                string            `json:"schema"`
	Product               string            `json:"product"`
	Required              int               `json:"required,omitempty"`
	OK                    bool              `json:"ok"`
	Verdict               string            `json:"verdict"`
	TrueConcurrentCeiling int               `json:"true_concurrent_ceiling"`
	FreshSeats            int               `json:"fresh_seats"`
	StaleSeats            int               `json:"stale_seats"`
	BlockedSeats          int               `json:"blocked_seats"`
	TotalSeats            int               `json:"total_seats"`
	Reason                string            `json:"reason"`
	Accounts              []CapacityAccount `json:"accounts"`
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
	for _, row := range rows {
		if !RoutableWorker(row) {
			continue
		}
		rowProduct := strings.ToLower(productOf(row))
		if product != "all" && rowProduct != product {
			continue
		}
		acct := capacityAccount(row)
		rep.Accounts = append(rep.Accounts, acct)
		rep.TotalSeats++
		switch acct.State {
		case CapacityFresh:
			rep.FreshSeats++
		case CapacityBlockedUntil:
			rep.BlockedSeats++
		default:
			rep.StaleSeats++
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
	rep.Reason = fmt.Sprintf("%d fresh seat(s), %d stale, %d blocked", rep.FreshSeats, rep.StaleSeats, rep.BlockedSeats)
	if required > 0 && rep.TrueConcurrentCeiling < required {
		rep.OK = false
		rep.Verdict = "UNDER_CAPACITY"
		rep.Reason = fmt.Sprintf("requires %d fresh seat(s), only %d available; dispatch must downsize before spawning", required, rep.TrueConcurrentCeiling)
	}
	return rep
}

func capacityAccount(row Account) CapacityAccount {
	state, until, reason := capacityState(row)
	label := state
	if state == CapacityBlockedUntil {
		label = "blocked-until-" + capacityFirstNonEmpty(until, "unknown")
	}
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
		ActiveSessions: derefInt(row.ActiveSessions),
		LiveSessions:   derefInt(row.LiveSessions),
		StatusSource:   derefStr(row.StatusSource),
	}
}

func capacityState(row Account) (state, until, reason string) {
	if accountCanBeOffered(row) {
		return CapacityFresh, "", "ready to serve"
	}
	reason = capacityFirstNonEmpty(derefStr(row.BlockReason), row.Reason, "not currently offerable")
	blockKind := strings.ToLower(derefStr(row.BlockKind))
	reset := capacityFirstNonEmpty(derefStr(row.Weekly), derefStr(row.Reset))
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

func capacityFirstNonEmpty(vals ...string) string {
	for _, v := range vals {
		if strings.TrimSpace(v) != "" {
			return strings.TrimSpace(v)
		}
	}
	return ""
}
