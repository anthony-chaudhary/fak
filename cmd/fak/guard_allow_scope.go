package main

import (
	"os"
	"path/filepath"
	"strings"
)

// guard_allow_scope.go — ticket #5180 (epic #5170 "Policy Amendment Classes"):
// NAMED SCOPES for the guard allow-overlay layers, plus a SESSION-scope layer.
//
// Every allow-overlay layer (guard_allow.go) now carries a scope name, and a new
// session-scoped layer joins the stack. The scopes, and their PRECEDENCE, are:
//
//	scope     rank  file                                            durability
//	-------   ----  ----------------------------------------------  -----------------------------
//	repo        0   <repo>/.fak/guard/allow.json                    checked-out tree; broadest
//	user        1   ~/.fak/guard/allow.json                         operator-local, host-wide
//	env         1   $FAK_GUARD_ALLOW_OVERLAY (explicit path)        operator-controlled override
//	session     2   <repo>/.fak/guard/sessions/<id>.allow.json      THIS guard session only
//
// PRECEDENCE RULE: a HIGHER rank is the NARROWER, more-ephemeral scope, and it layers
// ON TOP of every lower rank — session over user/env over repo. Layers are applied in
// ascending-precedence order (broadest first, session last), so the narrowest scope is
// always the last word. Because the allow overlay is additive-only (every layer only
// WIDENS Allow/AllowPrefix; the union merge in loadGuardAllowOverlayLayers cannot drop
// an entry), a more-specific scope can never be silently overridden by a broader one:
// a session grant survives the merge no matter what the repo or user layers say, and
// guardAllowWinningScope attributes an entry present in several scopes to the
// NARROWEST scope that names it, never a broader one.
//
// The session layer is keyed to the guard session id the guard already resolves at
// launch (guardTraceID — see resolveGuardSessionID): the guard boot, or a test, hands
// it to setGuardAllowSessionScopeID, and $FAK_GUARD_SESSION_ID serves the same id to
// out-of-process callers (`fak guard allow`-style tooling run beside a session). When
// NO session id is available the layer falls back to the DOCUMENTED session-scoped
// path <repo>/.fak/guard/sessions/current.allow.json — "the current session on this
// checkout" — so a session-scoped widening always has a well-known home. A missing
// session file is the common case and yields an empty layer (fail-open, exactly like
// the other layers); the session path is fed into protectGuardPolicyConfig with the
// rest of the layer paths, so a wrapped agent can no more widen its own session scope
// than the repo or user ones.

// guardAllowScopeSession names the session-scope layer. The repo/user/env layer names
// in guardAllowOverlayPaths double as their scope names.
const guardAllowScopeSession = "session"

// guardAllowSessionIDEnv carries the guard session id to processes that did not
// resolve it themselves (e.g. operator tooling running beside a guarded session).
const guardAllowSessionIDEnv = "FAK_GUARD_SESSION_ID"

// guardAllowSessionScopeID is the programmatic session-id override — the reuse point
// for the id the guard already knows (guardTraceID), and the deterministic test hook.
var guardAllowSessionScopeID string

func setGuardAllowSessionScopeID(id string) { guardAllowSessionScopeID = strings.TrimSpace(id) }

// guardAllowSessionID resolves the session id keying the session-scope layer:
// programmatic override first (the guard's own resolved id), then the documented env
// var, then "" (→ the documented fallback path).
func guardAllowSessionID() string {
	if guardAllowSessionScopeID != "" {
		return guardAllowSessionScopeID
	}
	return strings.TrimSpace(os.Getenv(guardAllowSessionIDEnv))
}

// guardAllowSessionPathComponent makes a session id safe as a single file-name
// component: every byte outside [A-Za-z0-9._-] becomes '-', so an id can never
// escape the sessions dir or produce an invalid Windows path.
func guardAllowSessionPathComponent(id string) string {
	var b strings.Builder
	for _, r := range id {
		switch {
		case r >= 'a' && r <= 'z', r >= 'A' && r <= 'Z', r >= '0' && r <= '9', r == '.', r == '_', r == '-':
			b.WriteRune(r)
		default:
			b.WriteRune('-')
		}
	}
	return b.String()
}

// guardAllowSessionOverlayPath is the session-scope layer's file, under the guard
// config dir (.fak/guard) beside the repo layer. With a known session id the file is
// keyed to it; without one it is the documented current-session fallback.
func guardAllowSessionOverlayPath() string {
	dir := filepath.Join(findRepoRoot("."), ".fak", "guard", "sessions")
	if id := guardAllowSessionID(); id != "" {
		return filepath.Join(dir, guardAllowSessionPathComponent(id)+".allow.json")
	}
	return filepath.Join(dir, "current.allow.json")
}

func guardAllowSessionLayer() guardAllowOverlayLayer {
	return guardAllowOverlayLayer{Name: guardAllowScopeSession, Path: guardAllowSessionOverlayPath()}
}

// guardAllowLayersWithSessionScope appends the session-scope layer AFTER the base
// layers, keeping the base order untouched (backward-compatible, including the
// env-override-as-sole-base case). Last position = applied last = layered on top,
// which is exactly the session scope's precedence.
func guardAllowLayersWithSessionScope(base []guardAllowOverlayLayer) []guardAllowOverlayLayer {
	out := make([]guardAllowOverlayLayer, 0, len(base)+1)
	out = append(out, base...)
	return append(out, guardAllowSessionLayer())
}

// guardAllowScopeRank is the precedence order defined in the header comment: higher
// rank = narrower, more-ephemeral scope = layers on top. An unknown scope ranks below
// everything (conservative: it can never claim to shadow a known scope).
func guardAllowScopeRank(scope string) int {
	switch scope {
	case "repo":
		return 0
	case "user", "env":
		return 1
	case guardAllowScopeSession:
		return 2
	default:
		return -1
	}
}

// guardAllowWinningScope reports which scope OWNS an entry when it appears in more
// than one: the highest-precedence (narrowest) layer that names it. This is the
// enforcement seam for "a more-specific scope must not be silently overridden by a
// broader one" — an entry present in both repo and session scope is the SESSION's,
// so removing it from the broader file cannot silently strip the session grant, and
// provenance rendering can never attribute it to the broader scope. prefix selects
// AllowPrefix instead of Allow. Returns "" (no error) when no scope names the entry.
func guardAllowWinningScope(entry string, prefix bool) (string, error) {
	winner, best := "", -1
	for _, layer := range guardAllowLayersWithSessionScope(guardAllowOverlayPaths()) {
		ov, err := loadGuardAllowOverlay(layer.Path)
		if err != nil {
			return "", err
		}
		list := ov.Allow
		if prefix {
			list = ov.AllowPrefix
		}
		for _, e := range list {
			if e == entry && guardAllowScopeRank(layer.Name) >= best {
				winner, best = layer.Name, guardAllowScopeRank(layer.Name)
			}
		}
	}
	return winner, nil
}
