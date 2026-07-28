package main

import (
	"errors"
	"fmt"
	"io"
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
//	session     2   <repo>/.fak/guard/sessions/<id>.allow.json      THIS guard session only —
//	                                                                meant to be dropped at session end
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
//
// EPHEMERALITY is what the session scope is FOR: `fak guard allow --session <tool>`
// records a widening into the session file, every read path unions that layer last, and
// dropping the file at session end leaves no permanent hole in the floor. All three
// halves are wired (#5180):
//
//   - WRITE and READ. cmdGuardAllow routes --session to guardAllowSessionOverlayPath via
//     guardAllowWritePathForScope, and every reader (guardAllowOverlayLayerPaths,
//     guardAllowWinningScope, the --list rendering) unions the session layer through
//     guardAllowLayersWithSessionScope.
//   - DROP. cmdGuard calls armGuardAllowSessionScopeTeardown immediately BEFORE it loads
//     the capability floor (loadGuardCapabilityFloor), and finishGuardChildAndReport calls
//     dropGuardAllowSessionScopeAtSessionEnd once the session is over.
//
// TWO ORDERING CONSTRAINTS make that pair of call sites the correct ones, and both are
// easy to get wrong:
//
//   - ARM BEFORE THE FLOOR READ, and do NOT re-key the layer to guardTraceID. The guard
//     finalizes guardTraceID (resolveGuardSessionID) ~370 lines AFTER the floor loads, so
//     calling setGuardAllowSessionScopeID at that point would retarget the layer to a file
//     the floor never read: the widening the session actually HONORED (under the id already
//     in scope, or the documented current.allow.json fallback) would then survive the drop
//     while an unrelated, probably nonexistent path got removed. Arming first captures the
//     exact path the very next read resolves, which is the one that must be dropped.
//   - DROP IN THE TERMINAL FUNNEL, NOT VIA `defer` IN cmdGuard. finishGuardChildAndReport
//     ends a nonzero child exit, a non-ExitError launch failure, and a mid-session gateway
//     error with os.Exit — which does not run cmdGuard's defers. A deferred drop would
//     therefore fire only on a CLEAN exit and leak the widening on exactly the crash paths
//     an escalated session is most likely to take. The drop sits beside recordGuardUsage in
//     that function's terminal region, which every exit path reaches first.
//
// Only the session file is ever removed — repo/user/env are durable operator decisions.
// A SIGKILL'd or power-cut guard still cannot run the drop; that residue is bounded to the
// session file and the next launch's arm/drop cycle reclaims it.

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

// guardAllowScopeDurabilityNote is the one-clause durability legend for a scope. Every
// operator-facing listing carries it so the question the scope table answers — does a
// widening recorded HERE survive the next launch? — never has to be inferred from a path.
// The session line says what is TRUE TODAY rather than what the scope is for: the drop is
// armed at guard boot and runs on every session-end path (see the header), so the legend
// may now promise ephemerality — but only for a guard that reaches its own teardown.
func guardAllowScopeDurabilityNote(scope string) string {
	switch scope {
	case guardAllowScopeSession:
		return " (ephemeral — dropped when this guard session ends; survives only a killed guard)"
	case "user":
		return " (durable — host-wide, every repo)"
	case "env":
		return " (durable — $" + guardAllowOverlayEnv + " override)"
	case "repo":
		return " (durable — this checkout)"
	}
	return ""
}

// guardAllowSessionScopeTeardownPath is the session-scope overlay path CAPTURED AT
// ARM TIME. Teardown must remove the file this session actually read, not whatever the
// id resolves to at exit: the guard finalizes guardTraceID (resolveGuardSessionID)
// AFTER the capability floor loads, so re-resolving at teardown could name a different
// file and silently leak the widening into the next launch. Empty = nothing armed — the
// state in any process that never boots a guard session (`fak guard policy explain`, a
// test) — and an unarmed dropGuardAllowSessionScope is a no-op.
var guardAllowSessionScopeTeardownPath string

// armGuardAllowSessionScopeTeardown records the session-scope layer path this launch
// resolved and arms the drop; it returns the armed path so a caller can report it.
// Called from cmdGuard immediately BEFORE loadGuardCapabilityFloor, so the armed path is
// by construction the one the floor is about to read (see the header's ordering note).
func armGuardAllowSessionScopeTeardown() string {
	guardAllowSessionScopeTeardownPath = guardAllowSessionOverlayPath()
	return guardAllowSessionScopeTeardownPath
}

// dropGuardAllowSessionScope deletes the ARMED session-scope overlay. It is the half of
// the scope contract that makes a session widening EPHEMERAL — honored for this session,
// absent on the next launch — and it is deliberately narrow: it removes ONLY the armed
// session file, never the repo/user/env layers, which are durable operator decisions. A
// missing file is the common case (no session-scoped widening was ever added) and is not
// an error. Disarms itself, so a double teardown cannot delete a path a later launch has
// since armed. Production callers go through dropGuardAllowSessionScopeAtSessionEnd.
func dropGuardAllowSessionScope() error {
	path := strings.TrimSpace(guardAllowSessionScopeTeardownPath)
	guardAllowSessionScopeTeardownPath = ""
	if path == "" {
		return nil
	}
	if err := os.Remove(path); err != nil && !errors.Is(err, os.ErrNotExist) {
		return fmt.Errorf("guard allow session scope: drop %s: %w", path, err)
	}
	return nil
}

// dropGuardAllowSessionScopeAtSessionEnd is the SESSION-END call site of the drop — the
// production half of the ephemerality contract. It lives in finishGuardChildAndReport's
// terminal region rather than behind a `defer` in cmdGuard because that function ends
// three of its four exit paths with os.Exit, which runs no defers; see the header.
//
// A failed drop is REPORTED, never fatal, and never changes the exit code: the session is
// already over, and the operator's own exit status must stay the child's. But it is not
// swallowed either — a session widening that survived its session is a floor the next
// launch inherits without anyone deciding to, so the operator is told which file to remove.
// Suppressed under --quiet, matching the other teardown notices around it.
func dropGuardAllowSessionScopeAtSessionEnd(quiet bool, stderr io.Writer) {
	if err := dropGuardAllowSessionScope(); err != nil && !quiet && stderr != nil {
		fmt.Fprintf(stderr, "fak guard: %v (session-scoped widening will persist into the next launch; remove the file to clear it)\n", err)
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
