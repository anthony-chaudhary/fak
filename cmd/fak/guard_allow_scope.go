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
// dropping the file leaves no permanent hole in the floor. All halves are wired (#5180):
//
//   - WRITE and READ. cmdGuardAllow routes --session to guardAllowSessionOverlayPath via
//     guardAllowWritePathForScope, and every reader (guardAllowOverlayLayerPaths,
//     guardAllowWinningScope, the --list rendering) unions the session layer through
//     guardAllowLayersWithSessionScope.
//   - DROP AT SESSION END. cmdGuard calls armGuardAllowSessionScopeTeardown immediately
//     BEFORE it loads the capability floor (loadGuardCapabilityFloor), and
//     finishGuardChildAndReport calls dropGuardAllowSessionScopeAtSessionEnd once the
//     session is over.
//   - DROP AT BOOT. The same arm call ALSO reclaims whatever session file is already there
//     (reclaimStaleGuardAllowSessionScope), because a session-end drop alone cannot make the
//     scope ephemeral. See the next paragraph — this is the load-bearing half.
//
// WHY A BOOT DROP IS REQUIRED, not merely tidy (#5417 §2). Nothing in production supplies a
// session id: setGuardAllowSessionScopeID has no production caller and nothing exports
// $FAK_GUARD_SESSION_ID, so guardAllowSessionID() returns "" and EVERY launch on a checkout
// resolves the SAME fallback file, sessions/current.allow.json. With only a session-end drop
// that shared file is a durable grant wearing an ephemeral label, in two ways, and both fail
// in the dangerous direction — authority OUTLIVING the consent that created it:
//
//   - A guard that never reaches its teardown (SIGKILL, power cut, host reboot) leaves the
//     file behind. The NEXT launch resolves that same path as its own session layer, so a
//     dead session's widening is HONORED for the whole next session and only dropped at
//     THAT session's end — one full session of authority nobody granted, repeating for as
//     long as launches keep being killed.
//   - Two concurrent guards on one checkout share the file, so a --session grant made in
//     one session is honored by the other, which never consented to it.
//
// The boot reclaim closes both: at the instant a guard boots it has consented to nothing, so
// any session-scoped grant already on disk provably belongs to a DIFFERENT session and can
// never be proved in scope for this one. The rule is therefore mechanical rather than
// heuristic — a session grant is honored only if THIS session wrote it after its own boot —
// and the failure direction is to drop (the operator re-grants), never to retain.
//
// QUARANTINE. If the boot reclaim cannot clear the file, the grant is both unprovable and
// undroppable, so it must not be honored either: guardAllowEffectiveReadLayers omits the
// session layer for the rest of the process. Note that guardAllowLayersWithSessionScope
// stays UNCONDITIONAL, because its other consumer is guardAllowOverlayLayerPaths, which
// feeds protectGuardPolicyConfig — the set of files the wrapped agent may not write. Hiding
// a quarantined path from THAT list would let the agent author its own session overlay, so
// quarantine narrows what is read and never what is protected.
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
// A SIGKILL'd or power-cut guard still cannot run its own drop; the boot reclaim above is
// what bounds that residue, by clearing it BEFORE the next launch's floor can read it.

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

// guardAllowSessionScopeQuarantined records that the boot reclaim could NOT clear a
// pre-existing session overlay (see armGuardAllowSessionScopeTeardown). The file then holds
// a widening this session cannot prove it consented to AND cannot delete, so the only
// fail-safe left is to stop honoring it. Cleared by every arm and by every drop, so the flag
// never outlives the launch that set it.
var guardAllowSessionScopeQuarantined bool

// guardAllowEffectiveReadLayers is the layer list the ENFORCEMENT readers use: the full
// stack, minus a quarantined session layer. It is deliberately NOT the list
// guardAllowOverlayLayerPaths hands to protectGuardPolicyConfig — that one must keep the
// session path even while quarantined, or the wrapped agent gains write access to a file the
// guard has stopped watching. Narrowing what is READ is fail-safe; narrowing what is
// PROTECTED would be the opposite.
func guardAllowEffectiveReadLayers() []guardAllowOverlayLayer {
	layers := guardAllowLayersWithSessionScope(guardAllowOverlayPaths())
	if !guardAllowSessionScopeQuarantined {
		return layers
	}
	out := make([]guardAllowOverlayLayer, 0, len(layers))
	for _, layer := range layers {
		if layer.Name == guardAllowScopeSession {
			continue
		}
		out = append(out, layer)
	}
	return out
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
// The session line says what is TRUE TODAY rather than what the scope is for: the drop runs
// at guard BOOT as well as on every session-end path (see the header), so the legend may
// promise ephemerality without the "unless the guard was killed" hedge the session-end drop
// alone needed — a killed guard's residue is cleared by the next boot before it is read.
func guardAllowScopeDurabilityNote(scope string) string {
	switch scope {
	case guardAllowScopeSession:
		return " (ephemeral — dropped at guard boot and when this guard session ends)"
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

// reclaimStaleGuardAllowSessionScope clears a session-scope overlay left at path by SOME
// OTHER session — the boot half of the ephemerality contract (#5417). A booting guard has
// consented to nothing yet, so a session file already on disk belongs to a session that
// ended, was killed, or is running concurrently; none of those is this one, and there is no
// evidence that could make it this one. So the file goes, unconditionally, before the floor
// reads it. Absent is the common case and counts as clear.
//
// Reports whether the path is now PROVABLY clear. A false return is the uncertain case the
// caller must fail closed on: the widening is still readable and could not be removed, so it
// is reported to the operator (who is told exactly which file to delete) and quarantined out
// of the read path rather than silently added to this session's floor.
func reclaimStaleGuardAllowSessionScope(path string, stderr io.Writer) bool {
	if strings.TrimSpace(path) == "" {
		return true
	}
	err := os.Remove(path)
	if err == nil || errors.Is(err, os.ErrNotExist) {
		return true
	}
	if stderr != nil {
		fmt.Fprintf(stderr, "fak guard: could not reclaim the stale session-scope allow overlay %s: %v (it is being IGNORED for this session; remove the file to re-enable --session widenings)\n", path, err)
	}
	return false
}

// armGuardAllowSessionScopeTeardown reclaims any session-scope overlay left behind by an
// earlier or concurrent session, records the path this launch resolved, and arms the
// session-end drop; it returns the armed path so a caller can report it. Called from cmdGuard
// immediately BEFORE loadGuardCapabilityFloor, so the reclaim lands before the floor's very
// first read and the armed path is by construction the file that read resolves (see the
// header's ordering note).
//
// The reclaim is what makes --session ephemeral. The session-end drop alone cannot: it never
// runs on a killed guard, and because no launch supplies a session id every launch shares one
// file, so a skipped drop is inherited and honored by the next session in full.
func armGuardAllowSessionScopeTeardown() string {
	path := guardAllowSessionOverlayPath()
	guardAllowSessionScopeQuarantined = !reclaimStaleGuardAllowSessionScope(path, os.Stderr)
	guardAllowSessionScopeTeardownPath = path
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
	// Disarming also ends the quarantine: it was scoped to the launch whose boot reclaim
	// failed, and leaving it set would silently suppress the session layer for a later one.
	guardAllowSessionScopeQuarantined = false
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
	for _, layer := range guardAllowEffectiveReadLayers() {
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
