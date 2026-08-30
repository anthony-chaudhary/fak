package main

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/anthony-chaudhary/fak/internal/accounts"
	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/gateway"
)

// guard_account_failover.go — the guard's ACCOUNT-FAILOVER closure: the on-box roster walk behind
// the planner's AccountFailoverFunc hook. When a live turn 403s with an ACCOUNT-SCOPED wall (the
// org has OAuth/subscription inference disabled upstream, or a region/billing wall — see
// agent.classifyUpstream -> RemedyFailoverAccount), no retry or re-login on THIS credential can
// clear it: every login mints another token for the same walled org. The only remedy is a
// DIFFERENT account whose org still permits the request. This file finds that account among the
// sibling config homes under ~ and hands the planner its live token, then STICKILY redirects the
// per-request token source so the swap persists across turns (the session heals in place, no
// restart).
//
// It is deliberately isolated from guard.go (a peer-hot file) and built on a PURE core
// (pickFailoverAccount / readLiveAccessToken) so the roster logic is unit-testable without a live
// gateway or a real `claude`. The guard wires it in one line; everything else lives here.

// accountFailover holds the guard's session-scoped failover state: which account keys are already
// known-walled (so the picker never re-selects them), which keys the operator deliberately rehomed
// off (excluded from auto-reselection but NOT walled — the seat may be perfectly healthy), and the
// config dir currently in force (so the per-request token source follows the adopted account
// across turns). It is safe for concurrent use — the planner's failover hook, the per-request
// apiKeyFunc, and the gateway's operator-rehome route can run on different goroutines.
type accountFailover struct {
	homeRoot   string
	mu         sync.Mutex
	walled     map[string]bool // account keys (uuid:/tok:) proven walled this session
	moved      map[string]bool // account keys the operator rehomed OFF this session (never auto-reselect)
	currentDir string          // config dir the live token is read from; advances on each adopted swap
	// lastNoTarget is the typed reason the most recent failover/rehome found no sibling to adopt
	// (FailoverFoundTarget when the last attempt succeeded or none has run). The auto arm cannot
	// print on the hot path, so it stashes the reason here for the operator-facing surfaces (the
	// status endpoint, the terminal-403 message) to render the account-level fix — the single
	// reason both the auto and operator paths share instead of each guessing or discarding it.
	lastNoTarget failoverNoTargetReason
	now          func() time.Time
}

// newAccountFailover seeds the state with the initially-pinned config dir (the account the guard
// launched with) and the home root under which siblings are discovered. now is injectable for
// tests; nil uses time.Now.
func newAccountFailover(homeRoot, pinnedDir string, now func() time.Time) *accountFailover {
	if now == nil {
		now = time.Now
	}
	return &accountFailover{
		homeRoot:   homeRoot,
		walled:     map[string]bool{},
		moved:      map[string]bool{},
		currentDir: pinnedDir,
		now:        now,
	}
}

// currentConfigDir returns the config dir the per-request token source should read from — the
// initially-pinned dir until a failover adopts a sibling, then that sibling's dir. This is what
// makes the swap sticky across turns: after the first heal, apiKeyFunc reads the permitted
// account's rotating token, not the walled one.
func (a *accountFailover) currentConfigDir() string {
	a.mu.Lock()
	defer a.mu.Unlock()
	return a.currentDir
}

// walledKeys returns a snapshot of the account keys (uuid:/tok:) proven walled this
// session — the seats an account-scoped 403 forced a failover to skip. It is read by the
// live accounts+nodes endpoints provider (guard_endpoints.go) to mark those seats in the
// status area. A copy is returned so the caller never races the failover mutator. nil
// receiver (no failover armed) yields nil, which the provider reads as "nothing walled".
func (a *accountFailover) walledKeys() map[string]bool {
	if a == nil {
		return nil
	}
	a.mu.Lock()
	defer a.mu.Unlock()
	if len(a.walled) == 0 {
		return nil
	}
	out := make(map[string]bool, len(a.walled))
	for k, v := range a.walled {
		out[k] = v
	}
	return out
}

// failover is the AccountFailoverFunc body: mark the current account walled, then pick a permitted
// sibling and adopt it. reason is the classified remedy label (never a raw upstream body). It
// returns the sibling's live token and ok=true when a target was found — advancing currentDir so
// future turns follow it — or ("", false) when every sibling is walled/absent (the caller surfaces
// the account-scoped 403 terminally). The current account is added to the walled set FIRST so the
// picker cannot hand back the very credential that just failed.
//
// When the wall is a LIVE 429 account cap (reason == agent.RehomedSeat), it ALSO persists a
// self-recovering cooldown for the walled account to the fleet-shared store — so the cap outlives
// this process and the next `fak guard`/`fak accounts launch` (via guardrotate.Plan / the login
// overlay) automatically avoids the just-capped account instead of re-selecting it. The 403
// org/region/billing wall (reason == "failover_account") is NOT a timed cap and stays in-memory
// only: a default cooldown window would wrongly re-admit a durably-blocked org after it elapses.
// The persist is done AFTER releasing the lock (disk I/O off the mutex) and is best-effort.
func (a *accountFailover) failover(reason string) (string, bool) {
	token, walledKey, ok := a.failoverLocked()
	// Persist a durable cooldown for a live 429 account cap so the wall is visible fleet-wide and
	// to the next launch — not just to this session's in-memory walled set. Gated on the cap reason
	// and done off the lock; fail-open (a store error never affects the in-process swap above).
	isAccountCap := reason == agent.RehomedSeat
	if walledKey != "" && isAccountCap {
		_, _ = recordRehomeCooldown(os.Stderr, walledKey, reason, isAccountCap, a.now())
	}
	return token, ok
}

// transientTarget selects and adopts a live sibling without marking the current account
// walled. A transient 5xx/529 proves only that this target is unhealthy now, not that the
// account is permanently unusable.
func (a *accountFailover) transientTarget(_ int) (string, bool) {
	a.mu.Lock()
	defer a.mu.Unlock()

	homes, err := accounts.Discover(a.homeRoot)
	if err != nil {
		a.lastNoTarget = FailoverNoSiblings
		return "", false
	}
	excluded := a.excludedLocked()
	if current := accountKeyForDir(a.currentDir); current != "" {
		excluded[current] = true
	}
	dir, tok, noTarget, ok := pickFailoverAccount(homes, excluded, a.now())
	if !ok {
		a.lastNoTarget = noTarget
		return "", false
	}
	a.lastNoTarget = FailoverFoundTarget
	a.currentDir = dir
	return tok, true
}

// failoverLocked is failover's mutex-held core: it walls the current account in-memory, picks a
// permitted sibling, and advances the sticky dir. It returns the adopted token, the account KEY it
// just walled (so the caller can persist a cooldown off the lock), and ok. walledKey is "" when the
// current dir had no derivable identity.
func (a *accountFailover) failoverLocked() (token, walledKey string, ok bool) {
	a.mu.Lock()
	defer a.mu.Unlock()

	// Mark the account currently in force as walled so it is never re-selected this session.
	walledKey = accountKeyForDir(a.currentDir)
	if walledKey != "" {
		a.walled[walledKey] = true
	}

	homes, err := accounts.Discover(a.homeRoot)
	if err != nil {
		a.lastNoTarget = FailoverNoSiblings // roster unreadable — treat as nothing to fail over to
		return "", walledKey, false
	}
	dir, tok, noTarget, ok := pickFailoverAccount(homes, a.excludedLocked(), a.now())
	if !ok {
		// Record WHY no sibling qualified so the terminal 403 surface (and the status endpoint) can
		// name the account-level fix instead of a bare "no failover target". The auto arm cannot
		// print here (it is on the hot path), so it stashes the reason for the operator-facing
		// surfaces to read — the same single reason forceRehome renders inline.
		a.lastNoTarget = noTarget
		return "", walledKey, false
	}
	a.lastNoTarget = FailoverFoundTarget
	// Adopt it: advance the sticky dir so the per-request token source follows the permitted
	// account on every subsequent turn, and return the live token for THIS re-send.
	a.currentDir = dir
	return tok, walledKey, true
}

// lastNoTargetReason returns the reason the most recent failover/rehome found no sibling to adopt
// (FailoverFoundTarget when the last attempt succeeded or none has run). It lets the operator-facing
// surfaces — the status endpoint and the terminal-403 message — report the account-level fix the
// auto arm could not print on the hot path. Safe for concurrent use; a nil receiver reports
// FailoverFoundTarget (nothing to report).
func (a *accountFailover) lastNoTargetReason() failoverNoTargetReason {
	if a == nil {
		return FailoverFoundTarget
	}
	a.mu.Lock()
	defer a.mu.Unlock()
	return a.lastNoTarget
}

// excludedLocked returns the union of every account key the picker must skip: the seats an
// account-scoped 403 proved walled AND the seats the operator deliberately rehomed off. Caller
// must hold a.mu. The two sets stay separate so the status area can keep labeling only the
// genuinely walled seats as walled.
func (a *accountFailover) excludedLocked() map[string]bool {
	out := make(map[string]bool, len(a.walled)+len(a.moved))
	for k := range a.walled {
		out[k] = true
	}
	for k := range a.moved {
		out[k] = true
	}
	return out
}

// forceRehome is the OPERATOR "switch seat now" body behind POST /v1/fak/account/rehome — the
// on-demand form of failover. It picks the next available sibling seat (enabled, logged in, live
// token, not walled/rehomed-off) and adopts it exactly the way failover does, so the per-request
// token source follows on the session's next upstream turn. The seat moved off is recorded in the
// moved set — a later automatic failover must never bounce the session back onto a seat the
// operator deliberately left — but is NOT marked walled: it was not proven bad, and the status
// area must not claim it was. When no sibling qualifies the state is left untouched (the session
// keeps its current seat) and the error names the real fix. The returned metadata is seat display
// identity only — never a token.
func (a *accountFailover) forceRehome(reason string) (gateway.AccountRehome, error) {
	a.mu.Lock()
	defer a.mu.Unlock()
	if strings.TrimSpace(reason) == "" {
		reason = "operator_rehome"
	}
	homes, err := accounts.Discover(a.homeRoot)
	if err != nil {
		return gateway.AccountRehome{}, fmt.Errorf("account roster unreadable under %s: %v", a.homeRoot, err)
	}
	fromDir := a.currentDir
	fromKey := accountKeyForDir(fromDir)
	// "Next available" means a DIFFERENT seat: exclude the current account by key (which also
	// covers a second dir name for the same account), and — for a seat with no derivable
	// identity — by dir, so the picker can never hand back the seat being left.
	excluded := a.excludedLocked()
	if fromKey != "" {
		excluded[fromKey] = true
	}
	candidates := make([]accounts.Home, 0, len(homes))
	for _, h := range homes {
		if guardCleanDir(h.Dir) == guardCleanDir(fromDir) {
			continue
		}
		candidates = append(candidates, h)
	}
	dir, _, noTarget, ok := pickFailoverAccount(candidates, excluded, a.now())
	if !ok {
		a.lastNoTarget = noTarget
		// Name the ACTUAL closest miss (walled / needs-login / disabled / none) and its one fix,
		// derived from the picker's typed reason — instead of the old OR-of-every-possible-cause
		// string that the picker's own knowledge could already narrow.
		return gateway.AccountRehome{}, fmt.Errorf("no available sibling seat: %s — %s", noTarget.describe(), noTarget.fix())
	}
	// Adopt: record the seat being left as operator-moved, then advance the sticky dir so the
	// per-request token source reads the adopted seat from the next turn on.
	if fromKey != "" {
		a.moved[fromKey] = true
	}
	a.currentDir = dir
	res := gateway.AccountRehome{Reason: reason}
	res.From, res.FromEmail = seatDisplayIdentity(homes, fromDir)
	res.To, res.ToEmail = seatDisplayIdentity(homes, dir)
	return res, nil
}

// seatDisplayIdentity resolves a config dir to its roster display identity (seat name + email).
// A dir no discovered home matches falls back to the dir's base name so the operator still sees
// WHICH directory moved, never an empty field.
func seatDisplayIdentity(homes []accounts.Home, dir string) (name, email string) {
	for _, h := range homes {
		if guardCleanDir(h.Dir) == guardCleanDir(dir) {
			return h.Name, h.Identity.Email
		}
	}
	return filepath.Base(guardCleanDir(dir)), ""
}

// failoverNoTargetReason is the CLOSED set of reasons pickFailoverAccount found no sibling to fail
// over to. Like rotation's RotationNoCandidateReason, it lets the picker explain itself ONCE at the
// point of decision, so the auto-failover terminal path and the operator forceRehome message derive
// the SAME account-level diagnosis instead of each hand-writing (or discarding) it. The tiers are
// ordered by how close a seat came to qualifying, so the reason names the CLOSEST miss — the most
// actionable thing the operator can fix: a seat that only needs a login beats one that is disabled.
type failoverNoTargetReason string

const (
	// FailoverFoundTarget — a sibling qualified (returned with ok=true).
	FailoverFoundTarget failoverNoTargetReason = ""
	// FailoverNoSiblings — the roster has no OTHER seat at all besides the ones excluded this
	// session. Enroll one (`fak accounts add`).
	FailoverNoSiblings failoverNoTargetReason = "no_siblings"
	// FailoverAllWalled — every candidate sibling is on an account already proven walled (or
	// operator-rehomed off) this session. Nothing to do but wait for a reset or enroll another.
	FailoverAllWalled failoverNoTargetReason = "all_walled"
	// FailoverNeedsLogin — a sibling exists and is not walled, but no seat holds a live token
	// (creds missing/expired/torn). The fix is a login (`claude /login` under its CLAUDE_CONFIG_DIR).
	FailoverNeedsLogin failoverNoTargetReason = "needs_login"
	// FailoverAllDisabled — siblings exist but every one is disabled/tombstoned/identity-mismatched
	// (CanServe false for a non-credential reason). Re-enable or remove them.
	FailoverAllDisabled failoverNoTargetReason = "all_disabled"
)

// describe renders the closest-miss STATE for a no-target reason — what is true of the sibling
// seats — so a message reads "<state> — <fix>". Empty for FailoverFoundTarget.
func (r failoverNoTargetReason) describe() string {
	switch r {
	case FailoverNoSiblings:
		return "no other seat is enrolled"
	case FailoverAllWalled:
		return "every other seat is on an account already walled or rehomed off this session"
	case FailoverNeedsLogin:
		return "a seat is available but none holds a live login token"
	case FailoverAllDisabled:
		return "every other seat is disabled, tombstoned, or has a mismatched identity"
	default:
		return ""
	}
}

// fix renders the one actionable next step for a no-target reason, so both the auto and operator
// paths speak with one voice. Empty for FailoverFoundTarget (there is nothing to fix).
func (r failoverNoTargetReason) fix() string {
	switch r {
	case FailoverNoSiblings:
		return "enroll another account (`fak accounts add`)"
	case FailoverAllWalled:
		return "wait for a reset, or enroll another account (`fak accounts add`)"
	case FailoverNeedsLogin:
		return "log in on another seat (`claude /login` under its CLAUDE_CONFIG_DIR)"
	case FailoverAllDisabled:
		return "re-enable or remove the disabled seats (`fak accounts`), then log one in"
	default:
		return ""
	}
}

// pickFailoverAccount is the PURE selection core: among discovered homes, choose one that (a) is
// not on an already-walled account, (b) CanServe (exists, enabled, has creds, no identity lie),
// and (c) holds a live, non-expired access token right now. It returns that home's dir and live
// token with reason FailoverFoundTarget, or ok=false with the CLOSED reason the closest sibling
// missed by — so a caller can tell the operator WHY failover could not help, not just that it
// couldn't. Deterministic given (homes, walled, now): homes arrive sorted by name from Discover,
// and the first qualifying one wins, so the choice is stable.
func pickFailoverAccount(homes []accounts.Home, walled map[string]bool, now time.Time) (dir, token string, reason failoverNoTargetReason, ok bool) {
	// Track the closest miss so the reason names the most-actionable gap. A seat that only needs a
	// login is a better thing to tell the operator than a walled or disabled one, so the miss tiers
	// are ranked by recoverability (needs-login beats disabled beats walled) and the CLOSEST one wins.
	var sawSibling, sawNonWalled, sawNeedsLogin bool
	for _, h := range homes {
		sawSibling = true
		key := h.Identity.AccountKey()
		if key != "" && walled[key] {
			continue // this account is known-walled this session
		}
		sawNonWalled = true
		if h.CanServe() {
			tok, live := readLiveAccessToken(h.Dir, now)
			if live {
				return h.Dir, tok, FailoverFoundTarget, true
			}
			// Serveable per the login report but the access token is expired/torn right now — the
			// fix is a fresh login, same actionable class as a NeedsLogin seat.
			sawNeedsLogin = true
			continue
		}
		// Not launch-ready: a NeedsLogin seat (creds absent) is one login away, so it counts as the
		// recoverable miss; a truly disabled/tombstoned/identity-mismatched seat does not.
		if h.LoginStatus() == accounts.LoginNeedsLogin || h.LoginStatus() == accounts.LoginIdentityMismatch {
			sawNeedsLogin = true
		}
	}
	switch {
	case !sawSibling:
		return "", "", FailoverNoSiblings, false
	case !sawNonWalled:
		return "", "", FailoverAllWalled, false
	case sawNeedsLogin:
		// At least one non-walled sibling is one login away — the most actionable miss.
		return "", "", FailoverNeedsLogin, false
	default:
		// Non-walled siblings exist but every one is disabled/tombstoned (no login can fix them here).
		return "", "", FailoverAllDisabled, false
	}
}

// accountKeyForDir returns the AccountKey (uuid:/tok:) for a config dir, or "" when the dir is
// empty or has no derivable identity. Used to key the walled set on the ACCOUNT, not the dir name,
// so two dir-names for one account are treated as the same (walling one walls both).
func accountKeyForDir(dir string) string {
	if strings.TrimSpace(dir) == "" {
		return ""
	}
	return accounts.DeriveIdentity(dir).AccountKey()
}

// credentialsDoc is the slice of a home's .credentials.json this file reads: the live OAuth access
// token and its expiry. It mirrors accounts.credExpiry's shape (kept local so this cmd file needs
// no new export from the accounts leaf).
type credentialsDoc struct {
	ClaudeAIOauth struct {
		AccessToken string `json:"accessToken"`
		ExpiresAt   int64  `json:"expiresAt"` // epoch milliseconds
	} `json:"claudeAiOauth"`
}

// readLiveAccessToken reads the live subscription OAuth access token from <dir>/.credentials.json
// and reports whether it is usable right now: present, non-empty, and not past its expiry at now.
// A missing/torn/expired credential returns ("", false) so the picker skips it — fak must never
// fail over TO a dead token. Only the access token itself is returned; the refresh token and other
// fields never leave this function. A miss on disk falls back to the macOS Keychain (#5363) —
// where darwin's live login actually lives — under the SAME strict contract (positive expiry,
// strictly after now), so a Mac seat is failover-eligible without ever loosening the rule.
func readLiveAccessToken(dir string, now time.Time) (string, bool) {
	if tok, ok := readLiveFileAccessToken(dir, now); ok {
		return tok, true
	}
	cred, ok := guardKeychainCred(dir)
	if !ok || cred.AccessToken == "" || cred.ExpiresAt <= 0 {
		return "", false
	}
	if !time.UnixMilli(cred.ExpiresAt).After(now) {
		return "", false // already expired at now
	}
	return cred.AccessToken, true
}

func readLiveFileAccessToken(dir string, now time.Time) (string, bool) {
	b, err := os.ReadFile(filepath.Join(dir, ".credentials.json"))
	if err != nil {
		return "", false
	}
	var doc credentialsDoc
	if json.Unmarshal(b, &doc) != nil {
		return "", false
	}
	tok := strings.TrimSpace(doc.ClaudeAIOauth.AccessToken)
	if tok == "" {
		return "", false
	}
	if doc.ClaudeAIOauth.ExpiresAt <= 0 {
		return "", false // no positive expiry recorded — treat as unusable rather than guess
	}
	if !time.UnixMilli(doc.ClaudeAIOauth.ExpiresAt).After(now) {
		return "", false // already expired at now
	}
	return tok, true
}
