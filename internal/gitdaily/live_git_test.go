package gitdaily

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// TestRealGitTickReapsStaleLocksAndFoldsLooseObjects is the captured live witness for
// the user-visible promise: one applied tick clears proven orphan locks before running
// real Git maintenance, and the loose-object count falls in that same tick. The unit
// tests in gitdaily_test.go pin every refusal and ordering branch with an injected
// runner; this test crosses the process boundary so an argv that only looked plausible
// to the fake cannot satisfy the acceptance criterion.
func TestRealGitTickReapsStaleLocksAndFoldsLooseObjects(t *testing.T) {
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git is not available")
	}

	root := t.TempDir()
	mustGit(t, root, "init", "-b", "main")
	mustGit(t, root, "config", "user.name", "git-daily live witness")
	mustGit(t, root, "config", "user.email", "gitdaily-witness@example.invalid")
	mustGit(t, root, "config", "gc.auto", "0")
	mustGit(t, root, "config", "maintenance.auto", "false")
	mustGit(t, root, "config", "core.untrackedCache", "true")
	mustGit(t, root, "config", "core.fsmonitor", "false")
	// Make the test independent of Git's distribution-default loose-object threshold.
	mustGit(t, root, "config", "maintenance.loose-objects.auto", "1")
	mustGit(t, root, "commit", "--allow-empty", "-m", "base")

	// Create a reachable loose-object backlog in one hash-object process, then hang the
	// resulting tree and commit from a private test ref so maintenance must preserve it.
	sourceDir := filepath.Join(root, "proof-objects")
	if err := os.Mkdir(sourceDir, 0o755); err != nil {
		t.Fatal(err)
	}
	var paths, treeLines []string
	for i := 0; i < 130; i++ {
		name := fmt.Sprintf("object-%03d", i)
		if err := os.WriteFile(filepath.Join(sourceDir, name), []byte("git-daily-live-proof-"+name+"\n"), 0o644); err != nil {
			t.Fatal(err)
		}
		paths = append(paths, filepath.ToSlash(filepath.Join("proof-objects", name)))
	}
	oids := strings.Fields(mustGitInput(t, root, strings.Join(paths, "\n")+"\n", "hash-object", "-w", "--stdin-paths"))
	if len(oids) != len(paths) {
		t.Fatalf("hash-object returned %d object IDs, want %d", len(oids), len(paths))
	}
	for i, oid := range oids {
		treeLines = append(treeLines, fmt.Sprintf("100644 blob %s\tobject-%03d", oid, i))
	}
	tree := strings.TrimSpace(mustGitInput(t, root, strings.Join(treeLines, "\n")+"\n", "mktree"))
	parent := strings.TrimSpace(mustGit(t, root, "rev-parse", "HEAD"))
	commit := strings.TrimSpace(mustGit(t, root, "commit-tree", tree, "-p", parent, "-m", "reachable loose-object witness"))
	mustGit(t, root, "update-ref", "refs/fak/gitdaily-live-witness", commit)

	common := filepath.Join(root, ".git")
	// Keep the same-day dedupe witness independent of the wall-clock hour at
	// which CI happens to run. A live time near midnight made now.Add(time.Hour)
	// truthfully name the next day and invalidated the fixture's premise.
	now := time.Date(2026, 8, 29, 12, 0, 0, 0, time.Local)
	leaseDir := filepath.Join(common, "refs", "fak", "locks")
	if err := os.MkdirAll(leaseDir, 0o755); err != nil {
		t.Fatal(err)
	}
	leaseLock := filepath.Join(leaseDir, "session-live-witness.lock")
	residueLock := filepath.Join(common, "index.lock.stale-20260701-000000")
	for _, path := range []string{leaseLock, residueLock} {
		if err := os.WriteFile(path, []byte("orphan\n"), 0o644); err != nil {
			t.Fatal(err)
		}
		stamp := now.Add(-30 * 24 * time.Hour)
		if err := os.Chtimes(path, stamp, stamp); err != nil {
			t.Fatal(err)
		}
	}

	opts := Options{
		RepoRoot: root, GitCommonDir: common,
		Ledger: filepath.Join(t.TempDir(), "git-daily.jsonl"),
		Now:    now, Apply: true, Force: true,
	}
	res := Run(context.Background(), realGitRunner, opts)
	if res.Skipped != "" || res.Incident || res.LedgerErr != "" {
		t.Fatalf("applied tick did not complete cleanly: %+v", res)
	}
	if got := res.Locks.LeaseReaped; len(got) != 1 || got[0] != filepath.Base(leaseLock) {
		t.Fatalf("LeaseReaped = %v, want [%s]", got, filepath.Base(leaseLock))
	}
	if !containsAction(res.Locks.Actions, "swept orphan lock residue ") {
		t.Fatalf("lock actions did not witness the residue reap: %v", res.Locks.Actions)
	}
	for _, path := range []string{leaseLock, residueLock} {
		if _, err := os.Stat(path); !errors.Is(err, os.ErrNotExist) {
			t.Fatalf("orphan lock survived the applied tick: %s (%v)", path, err)
		}
	}
	if res.Maint.Before.Count <= res.Maint.After.Count {
		t.Fatalf("loose objects did not fall in one tick: before=%d after=%d", res.Maint.Before.Count, res.Maint.After.Count)
	}
	if res.Maint.After.Count != 0 {
		t.Fatalf("live fold left %d loose objects; before=%d", res.Maint.After.Count, res.Maint.Before.Count)
	}
	for _, step := range res.Maint.Steps {
		if len(step.Args) > 0 && (step.Args[0] == "worktree" || step.Args[0] == "prune") && step.Ran {
			t.Fatalf("default unattended tick ran a destructive opt-in step: %v", step.Args)
		}
	}

	rows := Status(opts.Ledger, 0)
	if len(rows) != 1 || rows[0].LooseBefore <= rows[0].LooseAfter || rows[0].LockActions != 1 || rows[0].LeaseLocksReaped != 1 {
		t.Fatalf("ledger did not preserve the live witness: %+v", rows)
	}
	repeat := opts
	repeat.Force = false
	repeat.Now = now.Add(time.Hour)
	if got := Run(context.Background(), realGitRunner, repeat).Skipped; got != SkipAlreadyRanToday {
		t.Fatalf("same-day repeat = %q, want %q", got, SkipAlreadyRanToday)
	}
}

func containsAction(actions []string, prefix string) bool {
	for _, action := range actions {
		if strings.HasPrefix(action, prefix) {
			return true
		}
	}
	return false
}

func realGitRunner(ctx context.Context, dir string, args ...string) (string, int, error) {
	cmd := exec.CommandContext(ctx, "git", args...)
	cmd.Dir = dir
	cmd.Env = append(os.Environ(), "GIT_TERMINAL_PROMPT=0")
	out, err := cmd.CombinedOutput()
	if err == nil {
		return string(out), 0, nil
	}
	var exitErr *exec.ExitError
	if errors.As(err, &exitErr) {
		return string(out), exitErr.ExitCode(), nil
	}
	return string(out), -1, err
}

func mustGit(t *testing.T, dir string, args ...string) string {
	t.Helper()
	return mustGitInput(t, dir, "", args...)
}

func mustGitInput(t *testing.T, dir, stdin string, args ...string) string {
	t.Helper()
	cmd := exec.Command("git", args...)
	cmd.Dir = dir
	cmd.Env = append(os.Environ(), "GIT_TERMINAL_PROMPT=0")
	cmd.Stdin = bytes.NewBufferString(stdin)
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("git %s: %v\n%s", strings.Join(args, " "), err, out)
	}
	return string(out)
}
