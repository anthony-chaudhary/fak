package main

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// guard_allow_scope_launch_id_test.go — ticket #5417 §2, the PER-LAUNCH KEYING half.
//
// The boot reclaim (guard_allow_scope_wiring_test.go) makes a session widening ephemeral
// against a DEAD session. It cannot help against a LIVE one: while nothing supplied a session
// id, every guard on a checkout resolved sessions/current.allow.json, so two concurrent guards
// took turns reclaiming each other's layer and the operator re-granted forever. These tests
// pin the fix — every launch resolves its OWN file — plus the two properties that must not be
// traded away for it: the path the floor reads is the path it write-protects, and the child
// resolves the same file its guard does.

// TestGuardAllowSessionScopeLaunchIDGivesConcurrentLaunchesDistinctLayerPaths is the witness
// the ticket turns on. Two launches on one checkout, given IDENTICAL inputs — first the
// ordinary attended case with no --session-id, then two concurrent resumes naming the SAME
// --session-id — must resolve DIFFERENT session-layer files, and neither may be the shared
// fallback every pre-fix launch collapsed onto.
//
// The second half is why the id cannot simply be guardTraceID: resolveGuardSessionID returns
// the constant "guard" for an ordinary launch and the --session-id verbatim otherwise, so both
// halves below would collide under it.
func TestGuardAllowSessionScopeLaunchIDGivesConcurrentLaunchesDistinctLayerPaths(t *testing.T) {
	scopeTestRepo(t, "")

	// Launch A and launch B: plain `fak guard -- claude`, no --session-id, same checkout.
	setGuardAllowSessionScopeID(guardAllowSessionScopeLaunchID(""))
	a := guardAllowSessionOverlayPath()
	setGuardAllowSessionScopeID(guardAllowSessionScopeLaunchID(""))
	b := guardAllowSessionOverlayPath()

	if a == b {
		t.Fatalf("two concurrent launches resolved the SAME session layer %q; each would reclaim the other's live widening", a)
	}
	for _, p := range []string{a, b} {
		if filepath.Base(p) == "current.allow.json" {
			t.Fatalf("a keyed launch fell back to the shared session file %q", p)
		}
		if !strings.HasPrefix(filepath.Base(p), "guard-") {
			t.Fatalf("session file %q lost the legible launch base", filepath.Base(p))
		}
	}

	// Two concurrent guards resuming the SAME operator-named identity still get their own
	// ephemeral layer: --session-id names a resumable trace, not a shared consent.
	setGuardAllowSessionScopeID(guardAllowSessionScopeLaunchID("worker-a"))
	c := guardAllowSessionOverlayPath()
	setGuardAllowSessionScopeID(guardAllowSessionScopeLaunchID("worker-a"))
	d := guardAllowSessionOverlayPath()
	if c == d {
		t.Fatalf("two launches under one --session-id shared the session layer %q", c)
	}
	if !strings.HasPrefix(filepath.Base(c), "worker-a-") {
		t.Fatalf("session file %q must stay joinable to the operator's --session-id", filepath.Base(c))
	}
}

// TestGuardAllowSessionScopeKeyedLayerIsReadArmedAndProtectedTogether pins the invariant the
// boot ordering exists to produce: under a real per-launch id, the file the floor READS, the
// file the teardown ARMED, and the file protectGuardPolicyConfig WRITE-PROTECTS are one path.
// A keying that moved the read without moving the protection would hand the wrapped agent a
// writable overlay over its own permissions — strictly worse than the shared file it replaced.
func TestGuardAllowSessionScopeKeyedLayerIsReadArmedAndProtectedTogether(t *testing.T) {
	scopeTestRepo(t, "")
	guardAllowSessionScopeTeardownPath = ""
	t.Cleanup(func() { guardAllowSessionScopeTeardownPath = "" })

	// Boot, in cmdGuard's order: key, then arm, then the floor would read.
	setGuardAllowSessionScopeID(guardAllowSessionScopeLaunchID(""))
	armed := armGuardAllowSessionScopeTeardown()
	if filepath.Base(armed) == "current.allow.json" {
		t.Fatalf("the arm captured the shared fallback %q, so keying ran too late to matter", armed)
	}

	read := ""
	for _, layer := range guardAllowEffectiveReadLayers() {
		if layer.Name == guardAllowScopeSession {
			read = layer.Path
		}
	}
	if read != armed {
		t.Fatalf("the floor reads %q but the teardown armed %q; the drop would leak the honored widening", read, armed)
	}

	protected := guardAllowOverlayLayerPaths()
	found := false
	for _, p := range protected {
		if p == armed {
			found = true
		}
	}
	if !found {
		t.Fatalf("guardAllowOverlayLayerPaths() = %v, missing the keyed session path %q — the agent could widen its own scope", protected, armed)
	}
}

// TestGuardAllowSessionScopeKeyedLayerStaysProtectedWhileQuarantined re-proves the landed
// read/protect asymmetry under a per-launch id rather than the shared fallback it was written
// against: quarantine must narrow what is READ and never what is PROTECTED.
func TestGuardAllowSessionScopeKeyedLayerStaysProtectedWhileQuarantined(t *testing.T) {
	scopeTestRepo(t, "")
	guardAllowSessionScopeTeardownPath = ""
	t.Cleanup(func() { guardAllowSessionScopeTeardownPath = "" })

	setGuardAllowSessionScopeID(guardAllowSessionScopeLaunchID(""))
	sessionPath := guardAllowSessionOverlayPath()

	// A NON-EMPTY DIRECTORY at the keyed path: os.Remove refuses one on every OS, the portable
	// stand-in for "the widening is there and this launch cannot clear it".
	if err := os.MkdirAll(sessionPath, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(sessionPath, "pin.txt"), []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}

	armGuardAllowSessionScopeTeardown()
	if !guardAllowSessionScopeQuarantined {
		t.Fatal("a boot reclaim that failed must quarantine the keyed session layer")
	}
	for _, layer := range guardAllowEffectiveReadLayers() {
		if layer.Name == guardAllowScopeSession {
			t.Fatalf("a quarantined session layer must not be read: %+v", layer)
		}
	}
	found := false
	for _, p := range guardAllowOverlayLayerPaths() {
		if p == sessionPath {
			found = true
		}
	}
	if !found {
		t.Fatalf("guardAllowOverlayLayerPaths() = %v, must still protect the quarantined keyed path %q", guardAllowOverlayLayerPaths(), sessionPath)
	}
}

// TestGuardAllowSessionScopeChildEnvResolvesTheGuardsOwnLayer is the export half end to end:
// build the wrapped child's environment with an OUTER guard's id already ambient, then let a
// would-be child resolve from what it received. It must land on this guard's file, not the
// inherited one — a child writing into a live peer session's overlay is the cross-session
// authority the whole scope exists to deny.
func TestGuardAllowSessionScopeChildEnvResolvesTheGuardsOwnLayer(t *testing.T) {
	repo := scopeTestRepo(t, "")

	// An outer guard's id sitting in the ambient environment the child inherits.
	t.Setenv(guardAllowSessionIDEnv, "guard-outer-deadbeef")
	setGuardAllowSessionScopeID(guardAllowSessionScopeLaunchID(""))
	mine := guardAllowSessionOverlayPath()

	_, env := guardChildCommandEnv([]string{"claude"}, [][2]string{guardAllowSessionScopeChildEnv()}, false)
	exported, seen := "", false
	for _, kv := range env {
		if rest, ok := strings.CutPrefix(kv, guardAllowSessionIDEnv+"="); ok {
			exported, seen = rest, true // last binding wins on every spawn path
		}
	}
	if !seen {
		t.Fatalf("child env carries no %s; an in-session `fak guard allow --session` cannot find this guard's layer", guardAllowSessionIDEnv)
	}
	if exported == "guard-outer-deadbeef" {
		t.Fatal("the child inherited an OUTER guard's session id; its widenings would land in a live peer session's overlay")
	}

	// Resolve the way the child process would: env only, no programmatic override.
	setGuardAllowSessionScopeID("")
	t.Setenv(guardAllowSessionIDEnv, exported)
	if child := guardAllowSessionOverlayPath(); child != mine {
		t.Fatalf("child resolves %q but its guard reads %q (repo %s); the widening would never be honored", child, mine, repo)
	}
}

// TestGuardAllowSessionScopeKeyingIsWiredIntoTheGuardBoot pins the production call sites in
// source. Neither is reachable from a unit test — cmdGuard execs an agent — and an unwired
// keying looks exactly like a wired one until two guards run at once, which is the state this
// ticket found the scope in. Order is checked too: keying after the arm arms the wrong file,
// and keying after the floor load protects a different file than it reads.
func TestGuardAllowSessionScopeKeyingIsWiredIntoTheGuardBoot(t *testing.T) {
	boot, err := os.ReadFile("guard.go")
	if err != nil {
		t.Fatal(err)
	}
	keyAt := bytes.Index(boot, []byte("setGuardAllowSessionScopeID(guardAllowSessionScopeLaunchID("))
	if keyAt < 0 {
		t.Fatal("cmd/fak/guard.go no longer keys the session-scope layer to this launch; every concurrent guard would share one session file")
	}
	armAt := bytes.Index(boot, []byte("armGuardAllowSessionScopeTeardown()"))
	if armAt < 0 {
		t.Fatal("cmd/fak/guard.go no longer arms the session-scope teardown")
	}
	if keyAt > armAt {
		t.Fatal("the session-scope id must be minted BEFORE the arm, else the arm captures the shared fallback path")
	}
	floorAt := bytes.Index(boot, []byte("loadGuardCapabilityFloor("))
	if floorAt < 0 {
		t.Fatal("cmd/fak/guard.go no longer loads the capability floor")
	}
	if keyAt > floorAt {
		t.Fatal("the session-scope id must be minted BEFORE the floor load, else protectGuardPolicyConfig locks a different file than the floor reads")
	}
	if !bytes.Contains(boot, []byte("guardAllowSessionScopeChildEnv()")) {
		t.Fatalf("cmd/fak/guard.go no longer exports $%s to the child; an in-session `fak guard allow --session` would write a file no guard reads", guardAllowSessionIDEnv)
	}
}
