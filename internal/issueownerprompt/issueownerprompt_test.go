package issueownerprompt

import (
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

func TestTrackedResolversComposeCanonicalLifecycle(t *testing.T) {
	_, file, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("runtime.Caller failed")
	}
	root := filepath.Clean(filepath.Join(filepath.Dir(file), "..", ".."))
	if err := ValidateDir(filepath.Join(root, ".claude", "goal-prompts")); err != nil {
		t.Fatal(err)
	}
}

func TestValidateDirRejectsCopiedObsoleteLifecycle(t *testing.T) {
	dir := fixtureDir(t)
	path := filepath.Join(dir, "resolve-bad-issue-witnessed.md")
	body := includeLine + "\n\n## Domain delta: bad\n\n" +
		"Apply no lifecycle override: " + strings.Join(deltaInvariants, ", ") + ".\n" +
		"claim ONE menu item, resolve it, ship it WITNESSED, then stop.\n"
	if err := os.WriteFile(path, []byte(body), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := ValidateDir(dir); err == nil || !strings.Contains(err.Error(), "obsolete lifecycle") {
		t.Fatalf("ValidateDir error = %v, want obsolete lifecycle refusal", err)
	}
}

func TestValidateDirRejectsLifecycleDrift(t *testing.T) {
	dir := fixtureDir(t)
	path := filepath.Join(dir, "resolve-bad-issue-witnessed.md")
	body := includeLine + "\n\n## Domain delta: bad\n\nApply no lifecycle override: root implementation.\n"
	if err := os.WriteFile(path, []byte(body), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := ValidateDir(dir); err == nil || !strings.Contains(err.Error(), "missing lifecycle drift assertion") {
		t.Fatalf("ValidateDir error = %v, want missing invariant refusal", err)
	}
}

func fixtureDir(t *testing.T) string {
	t.Helper()
	dir := t.TempDir()
	canonical := strings.Join(lifecycleInvariants, "\n")
	if err := os.WriteFile(filepath.Join(dir, LifecycleFile), []byte(canonical), 0o600); err != nil {
		t.Fatal(err)
	}
	return dir
}
