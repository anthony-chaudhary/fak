package main

import (
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

const ledgerPathHelperEnv = "FAK_TEST_NIGHTRUN_LEDGER_REL"

func TestNightrunLedgerRootPrimaryCheckoutUnchanged(t *testing.T) {
	root := initLedgerTestRepo(t)
	t.Setenv(ledgerRootEnv, "")
	if got := nightrunLedgerRoot(root); !sameLedgerPath(got, root) {
		t.Fatalf("primary checkout root changed: got %q want %q", got, root)
	}
}

func TestNightrunLedgerPathInsideWorktreeUsesPrimaryCheckout(t *testing.T) {
	root := initLedgerTestRepo(t)
	wt := filepath.Join(t.TempDir(), "fak-worker-wt-cmd-ledger")
	runLedgerGit(t, root, "worktree", "add", "--detach", wt, "HEAD")
	t.Cleanup(func() { _ = exec.Command("git", "-C", root, "worktree", "remove", "--force", wt).Run() })

	rel := filepath.FromSlash(".fak/nightrun/worktree-ledger-witness.jsonl")
	cmd := exec.Command(os.Args[0], "-test.run=^TestNightrunLedgerPathHelper$")
	cmd.Dir = wt
	cmd.Env = append(os.Environ(), ledgerRootEnv+"=", ledgerPathHelperEnv+"="+rel)
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("worktree ledger helper: %v\n%s", err, out)
	}
	firstLine := strings.SplitN(string(out), "\n", 2)[0]
	got := strings.TrimSpace(strings.TrimSuffix(strings.TrimSpace(firstLine), "PASS"))
	want := filepath.Join(root, rel)
	if !sameLedgerPath(got, want) {
		t.Fatalf("ledger forked from primary checkout: got %q want %q", got, want)
	}
	if sameLedgerPath(got, filepath.Join(wt, rel)) {
		t.Fatalf("ledger path must not live in worker worktree: %q", got)
	}
	if _, err := os.Stat(filepath.Dir(filepath.Join(wt, rel))); !os.IsNotExist(err) {
		t.Fatalf("worker-local nightrun directory was created: err=%v", err)
	}
}

func TestNightrunLedgerPathHelper(t *testing.T) {
	t.Helper()
	rel := os.Getenv(ledgerPathHelperEnv)
	if rel == "" {
		return
	}
	_, _ = os.Stdout.WriteString(nightrunLedgerPath(rel))
}

func TestNightrunLedgerRootExplicitAbsoluteOverride(t *testing.T) {
	want := filepath.Join(t.TempDir(), "shared-ledgers")
	t.Setenv(ledgerRootEnv, want)
	if got := nightrunLedgerRoot(t.TempDir()); !sameLedgerPath(got, want) {
		t.Fatalf("explicit ledger root: got %q want %q", got, want)
	}
}

func initLedgerTestRepo(t *testing.T) string {
	t.Helper()
	root := t.TempDir()
	runLedgerGit(t, root, "init")
	runLedgerGit(t, root, "config", "user.email", "ledger-test@example.invalid")
	runLedgerGit(t, root, "config", "user.name", "Ledger Test")
	if err := os.WriteFile(filepath.Join(root, "go.mod"), []byte("module example.invalid/ledger-test\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	runLedgerGit(t, root, "add", "go.mod")
	runLedgerGit(t, root, "commit", "-m", "fixture")
	return root
}

func runLedgerGit(t *testing.T, dir string, args ...string) {
	t.Helper()
	cmd := exec.Command("git", append([]string{"-C", dir}, args...)...)
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("git %s: %v\n%s", strings.Join(args, " "), err, out)
	}
}

func sameLedgerPath(a, b string) bool {
	a, b = filepath.Clean(a), filepath.Clean(b)
	if runtime.GOOS == "windows" {
		return strings.EqualFold(a, b)
	}
	return a == b
}
