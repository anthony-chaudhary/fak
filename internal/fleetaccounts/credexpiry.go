package fleetaccounts

import (
	"fmt"
	"path/filepath"
	"sort"
	"strings"
	"time"

	configaccounts "github.com/anthony-chaudhary/fak/internal/accounts"
)

// credexpiry.go — credential honesty for the seat pool (#2059, #2075).
//
// The runtime-status fold historically derived availability from probe / throttle /
// auth-block signals only and never read the account dir's .credentials.json expiry.
// A seat whose OAuth token was already expired but had no recorded auth-blocked
// session reported available=true, so the pool / preflight / the wave offered it and
// every spawn died STALE_CRED at the guard — one wasted spawn per tick, and a seat
// silently dropped from the fleet's true concurrency without anything surfacing the
// re-login need. This file reads the one fact that fold was missing and folds the
// resulting pool shrink into an operator-visible dropped-seat report.

// credExpiredReason is the block reason stamped on a seat whose OAuth credential is
// provably expired with no setup-token fallback (issue #2059's done-condition text).
const credExpiredReason = "credential expired — needs login"

// CredExpiry is the observable credential-expiry state of one Claude config dir. It
// carries instants and booleans only — never a token value.
type CredExpiry struct {
	HasCredFile   bool      // .credentials.json exists and parses as an object
	HasExpiry     bool      // claudeAiOauth.expiresAt was present and positive
	ExpiresAt     time.Time // the parsed expiry instant (UTC); zero unless HasExpiry
	HasSetupToken bool      // a non-empty .oauth-token setup-token fallback exists
}

// ReadCredExpiry reads the credential-expiry facts for one Claude config dir. It
// never raises: a missing/malformed .credentials.json or an absent expiresAt just
// reads as "no provable expiry", which the gate treats as serveable (fail-open — an
// unknown expiry must not tombstone a working seat).
func ReadCredExpiry(acctDir string) CredExpiry {
	var out CredExpiry
	if acctDir == "" {
		return out
	}
	_, out.HasSetupToken = ReadOAuthToken(acctDir)
	doc, ok := readJSONObject(filepath.Join(acctDir, ".credentials.json"))
	if !ok {
		return out
	}
	out.HasCredFile = true
	oauth, ok := doc["claudeAiOauth"].(map[string]any)
	if !ok {
		return out
	}
	raw, ok := oauth["expiresAt"].(float64)
	if !ok || raw <= 0 {
		return out
	}
	// Claude Code writes expiresAt as a Unix-epoch in MILLISECONDS; accept a
	// seconds-epoch too (values below ~1e11 are past year 5000 as ms, so the
	// magnitude disambiguates the two encodings).
	ms := int64(raw)
	if raw < 1e11 {
		ms = int64(raw * 1000)
	}
	out.HasExpiry = true
	out.ExpiresAt = time.UnixMilli(ms).UTC()
	return out
}

// NeedsLogin reports whether the seat's credential is provably expired at now with
// no setup-token fallback — the #2059 needs-login condition. A dir with no readable
// expiry is never needs-login from here (presence-based classification in
// internal/accounts already covers the missing-credential case).
func (c CredExpiry) NeedsLogin(now time.Time) bool {
	return c.HasExpiry && !c.HasSetupToken && now.After(c.ExpiresAt)
}

// applyCredExpiryGate annotates a Claude worker seat whose OAuth credential is
// provably expired as needs-login — never free (#2059). A fresh active-probe verdict
// (status_source probe / probe-ledger) is more authoritative than the on-disk expiry
// instant: the refresh token may have silently re-minted the access token, so a seat
// a probe just confirmed is left alone (the guard against tombstoning a live seat).
// A setup-token fallback also keeps the seat serveable headlessly. Runs after
// applyLoginGate so its honest reason wins over the generic no-credentials text.
func applyCredExpiryGate(r *Account, now time.Time) {
	if r.Product != "claude" || r.Kind != KindWorker {
		return
	}
	switch derefStr(r.StatusSource) {
	case "probe", "probe-ledger":
		return
	}
	if !ReadCredExpiry(r.Dir).NeedsLogin(now) {
		return
	}
	r.Available = boolp(false)
	r.Blocked = boolp(true)
	r.BlockKind = strp("auth")
	r.BlockReason = strp(credExpiredReason)
	r.Throttled = boolp(false)
	r.LoginStatus = strp(string(configaccounts.LoginNeedsLogin))
	r.CanServe = boolp(false)
}

// DroppedSeat is a routable Claude worker seat that left the offerable pool for an
// AUTH-class reason needing interactive attention (a stale/expired credential), as
// opposed to a usage throttle, which heals itself at reset. It is the explicit
// "the pool silently shrank" record issue #2075 asks the status fold to surface.
type DroppedSeat struct {
	Tag        string `json:"tag"`
	Account    string `json:"account"`
	Dir        string `json:"dir"`
	Reason     string `json:"reason"`
	NextAction string `json:"next_action"`
}

// DroppedSeats folds an annotated roster into the seats that need a re-login before
// they count toward dispatch fan-out again. Throttled (usage-walled) seats are not
// dropped — they return on their own; only auth-class blocks that need a human at an
// interactive prompt qualify.
func DroppedSeats(rows []Account) []DroppedSeat {
	var out []DroppedSeat
	for _, r := range rows {
		if r.Product != "claude" || !RoutableWorker(r) || accountCanBeOffered(r) {
			continue
		}
		if derefBool(r.Throttled) {
			continue
		}
		if strings.ToLower(derefStr(r.BlockKind)) != "auth" && !accountLoginBlocked(r) {
			continue
		}
		reason := firstNonEmpty(derefStr(r.BlockReason), r.Reason)
		if reason == "" {
			reason = "auth-blocked"
		}
		out = append(out, DroppedSeat{
			Tag:        r.Tag,
			Account:    r.Account,
			Dir:        r.Dir,
			Reason:     reason,
			NextAction: fmt.Sprintf("CLAUDE_CONFIG_DIR=%s claude /login", r.Dir),
		})
	}
	sort.SliceStable(out, func(i, j int) bool { return out[i].Tag < out[j].Tag })
	return out
}
