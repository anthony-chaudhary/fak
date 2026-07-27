package main

import (
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// guard_allow_session_scope_teardown_test.go — ticket #5180 follow-on. Two things the
// session scope added to guard_allow.go / guard_allow_scope.go had no witness for:
//
//   - the WRITE-SCOPE router behind `fak guard allow --session`
//     (guardAllowWritePathForScope), which must land a widening in the session layer and
//     NOT in the durable repo-local or per-user files; and
//   - the arm/drop teardown mechanism, which is defined but deliberately NOT called from
//     the guard boot yet (see guard_allow_scope.go's header). Pinning its contract here
//     means the wiring rung lands against witnessed behaviour rather than a fresh claim.
//
// These reuse scopeTestRepo (guard_allow_scope_test.go), which chdirs into a fresh temp
// repo with an empty HOME and no env override, so they must not run in parallel.

// TestGuardAllowWritePathForScopeRoutesSessionAwayFromDurableLayers: a --session write
// resolves to the session layer, does not create the repo/user files, and is still seen
// by the read path — attributed to the SESSION scope, not the broader ones.
func TestGuardAllowWritePathForScopeRoutesSessionAwayFromDurableLayers(t *testing.T) {
	scopeTestRepo(t, "guard-write-scope")

	sessionPath, err := guardAllowWritePathForScope(guardAllowScopeSession)
	if err != nil {
		t.Fatal(err)
	}
	repoPath, err := guardAllowWritePathForScope("repo")
	if err != nil {
		t.Fatal(err)
	}
	userPath, err := guardAllowWritePathForScope("user")
	if err != nil {
		t.Fatal(err)
	}
	if sessionPath == repoPath || sessionPath == userPath {
		t.Fatalf("session write path %q must differ from repo %q and user %q", sessionPath, repoPath, userPath)
	}
	if base := filepath.Base(sessionPath); base != "guard-write-scope.allow.json" {
		t.Fatalf("session write path = %q, want keyed to the session id", base)
	}
	// An unrecognized scope must fall back to the DEFAULT repo target, never to the
	// narrower session/user ones: a typo'd scope may not silently pick a different file.
	fallback, err := guardAllowWritePathForScope("not-a-scope")
	if err != nil {
		t.Fatal(err)
	}
	if fallback != repoPath {
		t.Fatalf("unknown scope wrote to %q, want the repo default %q", fallback, repoPath)
	}

	if err := saveGuardAllowOverlay(sessionPath, guardAllowOverlay{Allow: []string{"session_scoped_tool"}}); err != nil {
		t.Fatal(err)
	}
	for name, p := range map[string]string{"repo": repoPath, "user": userPath} {
		if _, statErr := os.Stat(p); !errors.Is(statErr, os.ErrNotExist) {
			t.Fatalf("a --session write must not create the %s layer %q (stat err = %v)", name, p, statErr)
		}
	}

	merged, _, err := loadGuardAllowOverlayLayers()
	if err != nil {
		t.Fatal(err)
	}
	found := false
	for _, a := range merged.Allow {
		if a == "session_scoped_tool" {
			found = true
		}
	}
	if !found {
		t.Fatalf("merged.Allow = %v, want the session-scoped widening honored", merged.Allow)
	}
	scope, err := guardAllowWinningScope("session_scoped_tool", false)
	if err != nil {
		t.Fatal(err)
	}
	if scope != guardAllowScopeSession {
		t.Fatalf("winning scope = %q, want %q", scope, guardAllowScopeSession)
	}
}

// TestGuardAllowScopeDurabilityNoteSeparatesEphemeralFromDurable: the legend `fak guard
// allow --list` prints beside every layer must actually distinguish the session scope
// from the durable ones, otherwise the listing answers nothing.
func TestGuardAllowScopeDurabilityNoteSeparatesEphemeralFromDurable(t *testing.T) {
	if note := guardAllowScopeDurabilityNote(guardAllowScopeSession); !strings.Contains(note, "ephemeral") {
		t.Fatalf("session durability note = %q, want it to say ephemeral", note)
	}
	for _, scope := range []string{"repo", "user", "env"} {
		note := guardAllowScopeDurabilityNote(scope)
		if !strings.Contains(note, "durable") {
			t.Fatalf("%s durability note = %q, want it to say durable", scope, note)
		}
		if strings.Contains(note, "ephemeral") {
			t.Fatalf("%s durability note = %q, must not claim ephemerality", scope, note)
		}
	}
	if note := guardAllowScopeDurabilityNote("not-a-scope"); note != "" {
		t.Fatalf("unknown scope note = %q, want empty", note)
	}
}

// TestGuardAllowSessionScopeTeardownDropsOnlyTheArmedSessionFile: the drop removes the
// session layer it armed and leaves the repo/user layers — those are durable operator
// decisions and a session teardown may never revoke them.
func TestGuardAllowSessionScopeTeardownDropsOnlyTheArmedSessionFile(t *testing.T) {
	scopeTestRepo(t, "guard-teardown")
	guardAllowSessionScopeTeardownPath = ""
	t.Cleanup(func() { guardAllowSessionScopeTeardownPath = "" })

	repoPath := guardAllowOverlayPath()
	userPath := guardAllowUserOverlayPath()
	sessionPath := guardAllowSessionOverlayPath()
	for _, p := range []string{repoPath, userPath, sessionPath} {
		if err := saveGuardAllowOverlay(p, guardAllowOverlay{Allow: []string{"a_tool"}}); err != nil {
			t.Fatal(err)
		}
	}

	if armed := armGuardAllowSessionScopeTeardown(); armed != sessionPath {
		t.Fatalf("armed path = %q, want the session layer %q", armed, sessionPath)
	}
	if err := dropGuardAllowSessionScope(); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(sessionPath); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("session overlay %q survived the drop (stat err = %v)", sessionPath, err)
	}
	for name, p := range map[string]string{"repo": repoPath, "user": userPath} {
		if _, err := os.Stat(p); err != nil {
			t.Fatalf("durable %s layer %q must survive a session drop: %v", name, p, err)
		}
	}
}

// TestGuardAllowSessionScopeTeardownNoOpUnarmedAndDisarmsAfterDrop: an unarmed drop
// touches nothing, a missing armed file is not an error, and the drop disarms itself so
// a second call cannot delete a file the NEXT launch put at the same path.
func TestGuardAllowSessionScopeTeardownNoOpUnarmedAndDisarmsAfterDrop(t *testing.T) {
	scopeTestRepo(t, "guard-teardown-twice")
	guardAllowSessionScopeTeardownPath = ""
	t.Cleanup(func() { guardAllowSessionScopeTeardownPath = "" })

	sessionPath := guardAllowSessionOverlayPath()
	if err := saveGuardAllowOverlay(sessionPath, guardAllowOverlay{Allow: []string{"a_tool"}}); err != nil {
		t.Fatal(err)
	}
	if err := dropGuardAllowSessionScope(); err != nil {
		t.Fatalf("unarmed drop must be a calm no-op, got %v", err)
	}
	if _, err := os.Stat(sessionPath); err != nil {
		t.Fatalf("unarmed drop removed the session overlay: %v", err)
	}

	armGuardAllowSessionScopeTeardown()
	if err := dropGuardAllowSessionScope(); err != nil {
		t.Fatal(err)
	}
	if err := dropGuardAllowSessionScope(); err != nil {
		t.Fatalf("dropping an already-dropped scope must not error, got %v", err)
	}

	if err := saveGuardAllowOverlay(sessionPath, guardAllowOverlay{Allow: []string{"next_launch_tool"}}); err != nil {
		t.Fatal(err)
	}
	if err := dropGuardAllowSessionScope(); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(sessionPath); err != nil {
		t.Fatalf("a disarmed drop must not delete the next launch's session overlay: %v", err)
	}
}
