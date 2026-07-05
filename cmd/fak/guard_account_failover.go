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
	now        func() time.Time
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
func (a *accountFailover) failover(reason string) (string, bool) {
	a.mu.Lock()
	defer a.mu.Unlock()

	// Mark the account currently in force as walled so it is never re-selected this session.
	if key := accountKeyForDir(a.currentDir); key != "" {
		a.walled[key] = true
	}

	homes, err := accounts.Discover(a.homeRoot)
	if err != nil {
		return "", false
	}
	dir, token, ok := pickFailoverAccount(homes, a.excludedLocked(), a.now())
	if !ok {
		return "", false
	}
	// Adopt it: advance the sticky dir so the per-request token source follows the permitted
	// account on every subsequent turn, and return the live token for THIS re-send.
	a.currentDir = dir
	return token, true
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
	dir, _, ok := pickFailoverAccount(candidates, excluded, a.now())
	if !ok {
		return gateway.AccountRehome{}, fmt.Errorf("no available sibling seat: every other seat is walled, already rehomed off, disabled, or has no live token — log in on another seat (`claude /login` under its CLAUDE_CONFIG_DIR) or enroll one (`fak accounts add`)")
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

// pickFailoverAccount is the PURE selection core: among discovered homes, choose one that (a) is
// not on an already-walled account, (b) CanServe (exists, enabled, has creds, no identity lie),
// and (c) holds a live, non-expired access token right now. It returns that home's dir and live
// token, or ok=false when none qualifies. Deterministic given (homes, walled, now): homes arrive
// sorted by name from Discover, and the first qualifying one wins, so the choice is stable.
func pickFailoverAccount(homes []accounts.Home, walled map[string]bool, now time.Time) (dir, token string, ok bool) {
	for _, h := range homes {
		key := h.Identity.AccountKey()
		if key != "" && walled[key] {
			continue // this account is known-walled this session
		}
		if !h.CanServe() {
			continue // not launch-ready (missing dir/creds, disabled, tombstoned, identity lie)
		}
		tok, live := readLiveAccessToken(h.Dir, now)
		if !live {
			continue // creds present but the access token is expired/torn — not usable right now
		}
		return h.Dir, tok, true
	}
	return "", "", false
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
// fields never leave this function.
func readLiveAccessToken(dir string, now time.Time) (string, bool) {
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
