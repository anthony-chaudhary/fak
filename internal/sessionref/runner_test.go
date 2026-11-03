package sessionref

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// makeRepo initializes a throwaway git repository in dir and fails the test if
// the real git binary cannot be executed at all (the err channel of the Runner
// contract), which would make every assertion below meaningless.
func makeRepo(t *testing.T, dir string) {
	t.Helper()
	_, code, err := GitRunner(context.Background(), dir, "init")
	if err != nil {
		t.Fatalf("git init could not execute: %v", err)
	}
	if code != 0 {
		t.Fatalf("git init exited %d", code)
	}
}

func TestGitRunnerExitCodeContract(t *testing.T) {
	t.Parallel()
	dir := t.TempDir()
	makeRepo(t, dir)
	ctx := context.Background()

	out, code, err := GitRunner(ctx, dir, "status", "--porcelain")
	if err != nil {
		t.Fatalf("git status err = %v, want nil", err)
	}
	if code != 0 {
		t.Errorf("git status code = %d, want 0", code)
	}
	if out != "" {
		t.Errorf("git status out = %q, want empty on a fresh repo", out)
	}

	// A missing ref is a non-zero EXIT, not an execution failure: err must stay
	// nil and stdout must be empty because stderr is discarded by design.
	out, code, err = GitRunner(ctx, dir, "rev-parse", "--verify", "refs/no/such")
	if err != nil {
		t.Errorf("git rev-parse err = %v, want nil (non-zero exit rides code)", err)
	}
	if code == 0 {
		t.Error("git rev-parse code = 0, want non-zero for a missing ref")
	}
	if out != "" {
		t.Errorf("git rev-parse out = %q, want empty (stderr discarded)", out)
	}
}

func TestGitStdinRunnerBatchUpdateRef(t *testing.T) {
	t.Parallel()
	dir := t.TempDir()
	makeRepo(t, dir)
	ctx := context.Background()

	path := filepath.Join(dir, "blob.txt")
	if err := os.WriteFile(path, []byte("session blob\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	out, code, err := GitRunner(ctx, dir, "hash-object", "-w", path)
	if err != nil || code != 0 {
		t.Fatalf("hash-object: out=%q code=%d err=%v", out, code, err)
	}
	sha := strings.TrimSpace(out)

	// The --stdin protocol is command-based ("update <ref> <new-oid>"); the
	// batched-delete caller (reap.go) relies on exactly this shape.
	out, code, err = GitStdinRunner(ctx, dir, "update refs/batch/a "+sha+"\n", "update-ref", "--stdin")
	if err != nil {
		t.Fatalf("update-ref --stdin err = %v, want nil", err)
	}
	if code != 0 {
		t.Fatalf("update-ref --stdin code = %d, want 0 (out=%q)", code, out)
	}

	out, code, err = GitRunner(ctx, dir, "rev-parse", "refs/batch/a")
	if err != nil || code != 0 {
		t.Fatalf("rev-parse refs/batch/a: out=%q code=%d err=%v", out, code, err)
	}
	if got := strings.TrimSpace(out); got != sha {
		t.Errorf("refs/batch/a = %q, want %q", got, sha)
	}

	// A line that matches no command must fail as an exit, not an execution error.
	out, code, err = GitStdinRunner(ctx, dir, "not a ref\n", "update-ref", "--stdin")
	if err != nil {
		t.Errorf("malformed stdin err = %v, want nil", err)
	}
	if code == 0 {
		t.Errorf("malformed stdin code = 0, want non-zero (out=%q)", out)
	}
}

func TestRunnerWaitDelayBound(t *testing.T) {
	t.Parallel()
	// Design guardrail: the backstop must stay strictly shorter than the
	// callers' own 10s outer bounds; a delay approaching its parent budget
	// stops being a backstop and becomes most of the wait.
	if RunnerWaitDelay > 2*time.Second {
		t.Errorf("RunnerWaitDelay = %v, want <= 2s", RunnerWaitDelay)
	}
}
