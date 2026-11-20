package devcmd

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// repoOwnershipFixturePaths are the paths the draft body declares. They exist
// only under the private root in the fixture tree, so the owning repository
// is definitively the private companion. Neither path carries the literal
// "fak-private", so the text-based private-boundary audit cannot refuse the
// draft before the repository-ownership check runs.
var repoOwnershipFixturePaths = []string{
	"platform/worktree/root_index_guard.go",
	"cmd/fak-flow/precommit.go",
}

// repoOwnershipRoots builds a (public, private) root pair under t.TempDir
// where the fixture paths exist only under the private root.
func repoOwnershipRoots(t *testing.T) (string, string) {
	t.Helper()
	base := t.TempDir()
	pubRoot := filepath.Join(base, "fak")
	privRoot := filepath.Join(base, "fak-private")
	if err := os.MkdirAll(pubRoot, 0o755); err != nil {
		t.Fatal(err)
	}
	for _, p := range repoOwnershipFixturePaths {
		full := filepath.Join(privRoot, p)
		if err := os.MkdirAll(filepath.Dir(full), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(full, []byte("// fixture\n"), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	return pubRoot, privRoot
}

// withCompanionRoots pins ResolveCompanionRoots to the fixture roots for the
// duration of the test, restoring the previous override on cleanup.
func withCompanionRoots(t *testing.T, pubRoot, privRoot string) {
	t.Helper()
	previous := overrideCompanionRoots
	overrideCompanionRoots = func() (string, string) { return pubRoot, privRoot }
	t.Cleanup(func() { overrideCompanionRoots = previous })
}

func repoOwnershipBody() string {
	return validDispatchableIssueBody("devcmd", repoOwnershipFixturePaths, 3)
}

// TestIssueCreateRepoOwnershipRefusesCrossRepoMismatch proves the mis-file
// guard: a draft whose declared paths belong only to the private companion is
// refused (exit 3, runner never called) when --repo explicitly targets the
// public repository, and the refusal names the correct --repo.
func TestIssueCreateRepoOwnershipRefusesCrossRepoMismatch(t *testing.T) {
	pubRoot, privRoot := repoOwnershipRoots(t)
	withCompanionRoots(t, pubRoot, privRoot)
	called := false
	runner := func(args []string) (string, string, bool) {
		called = true
		return "https://example.test/issues/1", "", true
	}
	var out, errb bytes.Buffer
	code := runIssueCreateWithCleanScrub(&out, &errb, []string{
		"--title", "feat(devcmd): derive owning repository from declared paths",
		"--body", repoOwnershipBody(),
		"--repo", "anthony-chaudhary/fak",
		"--estimate-points", "1",
		"--parent-baseline-points", "1",
		"--target-envelope", "- paths: >= 1 command",
		"--witnessed-envelope", "- paths: 1 command",
	}, runner)
	if code != 3 {
		t.Fatalf("exit = %d, want 3; stderr:\n%s", code, errb.String())
	}
	if called {
		t.Fatalf("runner must not be called on a cross-repository mismatch; stderr:\n%s", errb.String())
	}
	if !strings.Contains(errb.String(), "--repo anthony-chaudhary/fak-private") {
		t.Fatalf("refusal must name the correct --repo; stderr:\n%s", errb.String())
	}
}

// TestIssueCreateRepoOwnershipDerivesPrivateTarget proves the derivation:
// with --repo absent, the owning repository is derived from the declared
// paths and the runner is invoked with --repo anthony-chaudhary/fak-private.
func TestIssueCreateRepoOwnershipDerivesPrivateTarget(t *testing.T) {
	pubRoot, privRoot := repoOwnershipRoots(t)
	withCompanionRoots(t, pubRoot, privRoot)
	var captured []string
	runner := func(args []string) (string, string, bool) {
		captured = append([]string(nil), args...)
		return "https://example.test/issues/2", "", true
	}
	var out, errb bytes.Buffer
	code := runIssueCreateWithCleanScrub(&out, &errb, []string{
		"--title", "feat(devcmd): derive owning repository from declared paths",
		"--body", repoOwnershipBody(),
		"--estimate-points", "1",
		"--parent-baseline-points", "1",
		"--target-envelope", "- paths: >= 1 command",
		"--witnessed-envelope", "- paths: 1 command",
	}, runner)
	if code != 0 {
		t.Fatalf("exit = %d, want 0; stderr:\n%s", code, errb.String())
	}
	if len(captured) == 0 {
		t.Fatalf("runner was not called; stderr:\n%s", errb.String())
	}
	repo := ""
	for i, arg := range captured {
		if arg == "--repo" && i+1 < len(captured) {
			repo = captured[i+1]
		}
	}
	if repo != "anthony-chaudhary/fak-private" {
		t.Fatalf("derived --repo = %q, want anthony-chaudhary/fak-private; args=%q", repo, captured)
	}
}
