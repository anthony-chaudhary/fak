package main

import (
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
)

// guard_allow_scope_test.go — ticket #5180: session-scope allow layer + per-scope
// widening precedence. See guard_allow_scope.go for the scope table.

// scopeTestRepo pins the process into a fresh temp repo root with a clean scope
// environment: no env overlay override, an empty per-user home, and a known session id.
func scopeTestRepo(t *testing.T, sessionID string) string {
	t.Helper()
	home := t.TempDir()
	repo := t.TempDir()
	t.Setenv("HOME", home)
	t.Setenv("USERPROFILE", home)
	t.Setenv(guardAllowOverlayEnv, "")
	t.Setenv(guardAllowSessionIDEnv, "")
	setGuardAllowSessionScopeID(sessionID)
	t.Cleanup(func() { setGuardAllowSessionScopeID("") })
	// The boot-reclaim quarantine (guard_allow_scope.go) is process-global, so a test that
	// arms over an unclearable path could otherwise suppress the session layer for whatever
	// runs next. Clear it on both sides rather than trusting test order.
	guardAllowSessionScopeQuarantined = false
	t.Cleanup(func() { guardAllowSessionScopeQuarantined = false })
	old, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	if err := os.Chdir(repo); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = os.Chdir(old) })
	return repo
}

// TestGuardAllowSessionScopeLayerPresent: a session-scope entry is loaded into the
// merged overlay, and the session layer is keyed to the guard session id.
func TestGuardAllowSessionScopeLayerPresent(t *testing.T) {
	scopeTestRepo(t, "guard-abc123")

	sessionPath := guardAllowSessionOverlayPath()
	if base := filepath.Base(sessionPath); base != "guard-abc123.allow.json" {
		t.Fatalf("session overlay file = %q, want keyed to session id", base)
	}
	if dir := filepath.Dir(sessionPath); filepath.Base(dir) != "sessions" || filepath.Base(filepath.Dir(dir)) != "guard" {
		t.Fatalf("session overlay path %q not under the guard config dir's sessions/", sessionPath)
	}
	if err := saveGuardAllowOverlay(sessionPath, guardAllowOverlay{Allow: []string{"session_only_tool"}}); err != nil {
		t.Fatal(err)
	}

	merged, layers, err := loadGuardAllowOverlayLayers()
	if err != nil {
		t.Fatal(err)
	}
	last := layers[len(layers)-1]
	if last.Name != guardAllowScopeSession || last.Path != sessionPath {
		t.Fatalf("last (highest-precedence) layer = %+v, want session layer at %q", last, sessionPath)
	}
	found := false
	for _, a := range merged.Allow {
		if a == "session_only_tool" {
			found = true
		}
	}
	if !found {
		t.Fatalf("merged.Allow = %v, want session_only_tool present", merged.Allow)
	}
}

// TestGuardAllowSessionScopeFallbackPath: with NO session id available anywhere, the
// session layer falls back to the documented current-session path under the guard
// config dir instead of disappearing.
func TestGuardAllowSessionScopeFallbackPath(t *testing.T) {
	scopeTestRepo(t, "")

	p := guardAllowSessionOverlayPath()
	want := filepath.Join(".fak", "guard", "sessions", "current.allow.json")
	if !strings.HasSuffix(p, want) {
		t.Fatalf("fallback session path = %q, want suffix %q", p, want)
	}
}

// TestGuardAllowSessionIDEnvFallback: without a programmatic id the documented env
// var keys the session layer, and hostile ids are path-sanitized.
func TestGuardAllowSessionIDEnvFallback(t *testing.T) {
	scopeTestRepo(t, "")
	t.Setenv(guardAllowSessionIDEnv, "guard/evil:id")

	p := guardAllowSessionOverlayPath()
	if base := filepath.Base(p); base != "guard-evil-id.allow.json" {
		t.Fatalf("env-keyed session file = %q, want sanitized guard-evil-id.allow.json", base)
	}
}

// TestGuardAllowScopePrecedenceSessionWinsOverBroader: the SAME entry in repo scope
// and session scope resolves to the SESSION scope — the narrower scope is never
// silently overridden by (or attributed to) the broader one — and the merged allow
// set still admits it exactly once.
func TestGuardAllowScopePrecedenceSessionWinsOverBroader(t *testing.T) {
	scopeTestRepo(t, "guard-prec")

	repoPath := filepath.Join(findRepoRoot("."), ".fak", "guard", "allow.json")
	if err := saveGuardAllowOverlay(repoPath, guardAllowOverlay{Allow: []string{"shared_tool"}}); err != nil {
		t.Fatal(err)
	}
	if err := saveGuardAllowOverlay(guardAllowSessionOverlayPath(), guardAllowOverlay{Allow: []string{"shared_tool"}}); err != nil {
		t.Fatal(err)
	}

	scope, err := guardAllowWinningScope("shared_tool", false)
	if err != nil {
		t.Fatal(err)
	}
	if scope != guardAllowScopeSession {
		t.Fatalf("winning scope for entry in repo+session = %q, want %q", scope, guardAllowScopeSession)
	}

	merged, _, err := loadGuardAllowOverlayLayers()
	if err != nil {
		t.Fatal(err)
	}
	n := 0
	for _, a := range merged.Allow {
		if a == "shared_tool" {
			n++
		}
	}
	if n != 1 {
		t.Fatalf("merged.Allow = %v, want shared_tool exactly once", merged.Allow)
	}
}

// TestGuardAllowScopePrecedenceUserWinsOverRepo: the same rule one rung down — an
// entry in repo and user scope belongs to the narrower user scope.
func TestGuardAllowScopePrecedenceUserWinsOverRepo(t *testing.T) {
	scopeTestRepo(t, "guard-prec2")

	repoPath := filepath.Join(findRepoRoot("."), ".fak", "guard", "allow.json")
	if err := saveGuardAllowOverlay(repoPath, guardAllowOverlay{AllowPrefix: []string{"mcp__x__"}}); err != nil {
		t.Fatal(err)
	}
	if err := saveGuardAllowOverlay(guardAllowUserOverlayPath(), guardAllowOverlay{AllowPrefix: []string{"mcp__x__"}}); err != nil {
		t.Fatal(err)
	}

	scope, err := guardAllowWinningScope("mcp__x__", true)
	if err != nil {
		t.Fatal(err)
	}
	if scope != "user" {
		t.Fatalf("winning scope for prefix in repo+user = %q, want user", scope)
	}
}

// TestGuardAllowScopeMergedUnionAcrossThreeScopes: one entry per scope — the merged
// result is the exact sorted union, so no scope's widening is lost in the merge.
func TestGuardAllowScopeMergedUnionAcrossThreeScopes(t *testing.T) {
	scopeTestRepo(t, "guard-union")

	repoPath := filepath.Join(findRepoRoot("."), ".fak", "guard", "allow.json")
	if err := saveGuardAllowOverlay(repoPath, guardAllowOverlay{Allow: []string{"repo_tool"}}); err != nil {
		t.Fatal(err)
	}
	if err := saveGuardAllowOverlay(guardAllowUserOverlayPath(), guardAllowOverlay{Allow: []string{"user_tool"}}); err != nil {
		t.Fatal(err)
	}
	if err := saveGuardAllowOverlay(guardAllowSessionOverlayPath(), guardAllowOverlay{Allow: []string{"session_tool"}}); err != nil {
		t.Fatal(err)
	}

	merged, layers, err := loadGuardAllowOverlayLayers()
	if err != nil {
		t.Fatal(err)
	}
	if got, want := merged.Allow, []string{"repo_tool", "session_tool", "user_tool"}; !reflect.DeepEqual(got, want) {
		t.Fatalf("merged.Allow = %v, want %v", got, want)
	}
	if len(layers) != 3 {
		t.Fatalf("layers = %+v, want user, repo, session", layers)
	}
	// Ascending precedence must hold from each broader base layer to the session
	// layer on top (the base user/repo order is preserved for compatibility).
	if rTop, rRepo := guardAllowScopeRank(layers[2].Name), guardAllowScopeRank(layers[1].Name); rTop <= rRepo {
		t.Fatalf("session layer rank %d not above base layer rank %d", rTop, rRepo)
	}
}

// TestGuardAllowScopeRankOrdering pins the documented precedence table.
func TestGuardAllowScopeRankOrdering(t *testing.T) {
	if !(guardAllowScopeRank("repo") < guardAllowScopeRank("user") &&
		guardAllowScopeRank("user") == guardAllowScopeRank("env") &&
		guardAllowScopeRank("env") < guardAllowScopeRank(guardAllowScopeSession)) {
		t.Fatalf("scope ranks violate documented precedence: repo=%d user=%d env=%d session=%d",
			guardAllowScopeRank("repo"), guardAllowScopeRank("user"),
			guardAllowScopeRank("env"), guardAllowScopeRank(guardAllowScopeSession))
	}
	if guardAllowScopeRank("mystery") >= guardAllowScopeRank("repo") {
		t.Fatalf("unknown scope must rank below every known scope")
	}
}

// TestGuardAllowSessionLayerPathIsProtected: the session overlay file joins the
// write-protected layer path set, so a wrapped agent cannot widen its own session
// scope by editing the file.
func TestGuardAllowSessionLayerPathIsProtected(t *testing.T) {
	scopeTestRepo(t, "guard-protect")

	want := guardAllowSessionOverlayPath()
	for _, p := range guardAllowOverlayLayerPaths() {
		if p == want {
			return
		}
	}
	t.Fatalf("guardAllowOverlayLayerPaths() = %v, missing session path %q", guardAllowOverlayLayerPaths(), want)
}
