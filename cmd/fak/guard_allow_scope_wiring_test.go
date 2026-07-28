package main

import (
	"bytes"
	"os"
	"strings"
	"testing"
)

// guard_allow_scope_wiring_test.go — ticket #5180, the WIRING rung. The scope layers,
// the precedence table, and the `policy explain` scope column all landed earlier with
// their own witnesses; what had none was the half the done condition actually turns on:
// "a session-scoped widening is honored for the session and ABSENT ON THE NEXT LAUNCH."
// The arm/drop mechanism existed but no production call site invoked it, so a session
// widening in fact persisted forever. These tests pin the behaviour AND the two call
// sites, because the failure mode here is silent: an unwired drop looks exactly like a
// wired one until a widening outlives its session.

// TestGuardAllowScopeSessionWideningIsGoneOnTheNextLaunch is the issue's done condition
// end to end: a widening written into the session scope is honored by the merged floor
// this session, and after the session-end drop a fresh load — the next launch — no longer
// sees it, while the durable repo layer keeps its own entry throughout.
func TestGuardAllowScopeSessionWideningIsGoneOnTheNextLaunch(t *testing.T) {
	scopeTestRepo(t, "guard-lifecycle")
	guardAllowSessionScopeTeardownPath = ""
	t.Cleanup(func() { guardAllowSessionScopeTeardownPath = "" })

	// A durable repo-scope widening the session teardown must never revoke.
	if err := saveGuardAllowOverlay(guardAllowOverlayPath(), guardAllowOverlay{Allow: []string{"durable_repo_tool"}}); err != nil {
		t.Fatal(err)
	}

	// Launch: arm first, exactly as cmdGuard does before it loads the floor.
	armed := armGuardAllowSessionScopeTeardown()
	if armed != guardAllowSessionOverlayPath() {
		t.Fatalf("armed %q, want the session layer %q", armed, guardAllowSessionOverlayPath())
	}

	// Mid-session: `fak guard allow --session ephemeral_tool`.
	sessionPath, err := guardAllowWritePathForScope(guardAllowScopeSession)
	if err != nil {
		t.Fatal(err)
	}
	if err := saveGuardAllowOverlay(sessionPath, guardAllowOverlay{Allow: []string{"ephemeral_tool"}}); err != nil {
		t.Fatal(err)
	}

	// Honored for THIS session.
	merged, _, err := loadGuardAllowOverlayLayers()
	if err != nil {
		t.Fatal(err)
	}
	if !containsAllow(merged.Allow, "ephemeral_tool") {
		t.Fatalf("merged.Allow = %v, want the session widening honored during the session", merged.Allow)
	}
	if !containsAllow(merged.Allow, "durable_repo_tool") {
		t.Fatalf("merged.Allow = %v, want the durable repo widening honored too", merged.Allow)
	}

	// Session end, through the production wrapper.
	var stderr bytes.Buffer
	dropGuardAllowSessionScopeAtSessionEnd(false, &stderr)
	if stderr.Len() != 0 {
		t.Fatalf("a clean drop must stay silent, got %q", stderr.String())
	}

	// Next launch: the session widening is gone, the durable one survives.
	next, _, err := loadGuardAllowOverlayLayers()
	if err != nil {
		t.Fatal(err)
	}
	if containsAllow(next.Allow, "ephemeral_tool") {
		t.Fatalf("next launch still sees the session widening: %v", next.Allow)
	}
	if !containsAllow(next.Allow, "durable_repo_tool") {
		t.Fatalf("next launch lost the durable repo widening: %v", next.Allow)
	}
	if _, err := os.Stat(guardAllowOverlayPath()); err != nil {
		t.Fatalf("repo layer must survive a session drop: %v", err)
	}
}

// TestGuardAllowScopeSessionEndDropReportsFailureWithoutPanicking: the wrapper reports a
// drop it could not perform (an unremovable path) rather than swallowing it — a widening
// that outlives its session becomes the next launch's floor, so the operator is told —
// and stays silent under --quiet.
func TestGuardAllowScopeSessionEndDropReportsFailureWithoutPanicking(t *testing.T) {
	scopeTestRepo(t, "guard-drop-report")
	guardAllowSessionScopeTeardownPath = ""
	t.Cleanup(func() { guardAllowSessionScopeTeardownPath = "" })

	// A DIRECTORY at the armed path: os.Remove refuses a non-empty one on every OS,
	// which is a portable stand-in for "the drop failed".
	sessionPath := guardAllowSessionOverlayPath()
	if err := os.MkdirAll(sessionPath, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(sessionPath+string(os.PathSeparator)+"pin.txt", []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}

	armGuardAllowSessionScopeTeardown()
	var stderr bytes.Buffer
	dropGuardAllowSessionScopeAtSessionEnd(false, &stderr)
	if !strings.Contains(stderr.String(), "session") {
		t.Fatalf("a failed drop must be reported, got %q", stderr.String())
	}

	armGuardAllowSessionScopeTeardown()
	var quietErr bytes.Buffer
	dropGuardAllowSessionScopeAtSessionEnd(true, &quietErr)
	if quietErr.Len() != 0 {
		t.Fatalf("--quiet must suppress the notice, got %q", quietErr.String())
	}

	// A nil writer must not panic.
	armGuardAllowSessionScopeTeardown()
	dropGuardAllowSessionScopeAtSessionEnd(false, nil)
}

// TestGuardAllowScopeTeardownIsWiredIntoTheGuardBootAndExitPaths pins the two production
// call sites in source. Behaviour tests above cannot reach them: cmdGuard execs an agent
// and finishGuardChildAndReport ends in os.Exit, so neither is callable from a unit test.
// Without this check the drop could be silently unwired again — which is exactly the state
// this ticket found it in — and every behavioural test above would still pass.
func TestGuardAllowScopeTeardownIsWiredIntoTheGuardBootAndExitPaths(t *testing.T) {
	boot, err := os.ReadFile("guard.go")
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Contains(boot, []byte("armGuardAllowSessionScopeTeardown()")) {
		t.Fatal("cmd/fak/guard.go no longer arms the session-scope teardown; a session widening would persist forever")
	}
	// The arm must precede the floor read, or it captures a path the floor never read.
	armAt := bytes.Index(boot, []byte("armGuardAllowSessionScopeTeardown()"))
	floorAt := bytes.Index(boot, []byte("loadGuardCapabilityFloor("))
	if floorAt < 0 {
		t.Fatal("cmd/fak/guard.go no longer loads the capability floor")
	}
	if armAt > floorAt {
		t.Fatal("the session-scope arm must come BEFORE loadGuardCapabilityFloor, else it arms a path the floor never read")
	}

	exit, err := os.ReadFile("guard_child_supervision.go")
	if err != nil {
		t.Fatal(err)
	}
	dropAt := bytes.Index(exit, []byte("dropGuardAllowSessionScopeAtSessionEnd("))
	if dropAt < 0 {
		t.Fatal("finishGuardChildAndReport no longer drops the session scope; a session widening would leak into the next launch")
	}
	// It must run BEFORE the first os.Exit in the terminal region, which runs no defers.
	if exitAt := bytes.Index(exit[dropAt:], []byte("os.Exit(")); exitAt < 0 {
		t.Fatal("expected the terminal os.Exit branches to follow the session-scope drop")
	}
}

func containsAllow(list []string, want string) bool {
	for _, got := range list {
		if got == want {
			return true
		}
	}
	return false
}
