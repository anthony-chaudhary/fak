package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/anthony-chaudhary/fak/internal/accounts"
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
// known-walled (so the picker never re-selects them), and the config dir currently in force (so
// the per-request token source follows the adopted account across turns). It is safe for
// concurrent use — the planner's failover hook and the per-request apiKeyFunc can run on different
// turns' goroutines.
type accountFailover struct {
	homeRoot   string
	mu         sync.Mutex
	walled     map[string]bool // account keys (uuid:/tok:) proven walled this session
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
	dir, token, ok := pickFailoverAccount(homes, a.walled, a.now())
	if !ok {
		return "", false
	}
	// Adopt it: advance the sticky dir so the per-request token source follows the permitted
	// account on every subsequent turn, and return the live token for THIS re-send.
	a.currentDir = dir
	return token, true
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
