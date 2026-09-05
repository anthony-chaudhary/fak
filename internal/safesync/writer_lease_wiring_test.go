package safesync_test

// writer_lease_wiring_test.go — the #4611 pairing witness for the #4240 cooperative
// worktree writer lease: a real temp-repo concurrency test proving BOTH directions of
// the contract between safesync.Apply and the fak-managed commit writer
// (safecommit.CommitWith, the shared seam `fak commit` and `fak sweep --apply` route
// through):
//
//   - a managed commit is refused (WRITER_LEASE_HELD) while sync apply holds the lease
//     mid-window, and
//   - sync apply is refused (Assessment.Lease) while a managed commit holds it
//     mid-window.
//
// It lives in package safesync_test (not safesync) so it may import safecommit without
// a production import cycle — safecommit already imports safesync for SafePush and the
// lease itself. Mid-window interception is deterministic, not sleep-based: each side's
// injected Runner fires the counterpart writer synchronously on its first git call made
// while the lease is held, so there is no goroutine choreography to flake.

import (
	"context"
	"errors"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/safecommit"
	"github.com/anthony-chaudhary/fak/internal/safesync"
)

// wiringGit runs a real git command in dir, failing the test on any error.
func wiringGit(t *testing.T, dir string, args ...string) string {
	t.Helper()
	cmd := exec.Command("git", append([]string{"-C", dir}, args...)...)
	cmd.Env = append(os.Environ(), "GIT_OPTIONAL_LOCKS=0")
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("git %v: %v\n%s", args, err, out)
	}
	return string(out)
}

// wiringCommitRunner is a real-git safecommit.Runner mirroring realRunner's contract:
// merged output, a non-zero git exit in code (not err), err only when git cannot start.
func wiringCommitRunner(ctx context.Context, dir string, args ...string) (string, int, error) {
	cmd := exec.CommandContext(ctx, "git", args...)
	cmd.Env = append(os.Environ(), "GIT_OPTIONAL_LOCKS=0")
	if len(args) > 0 && args[0] == "commit" {
		cmd.Env = append(cmd.Env, "FAK_SAFECOMMIT_VETTED=1")
	}
	if dir != "" {
		cmd.Dir = dir
	}
	var buf strings.Builder
	cmd.Stdout = &buf
	cmd.Stderr = &buf
	err := cmd.Run()
	if err == nil {
		return buf.String(), 0, nil
	}
	var ee *exec.ExitError
	if errors.As(err, &ee) {
		return buf.String(), ee.ExitCode(), nil
	}
	return "", -1, err
}

// wiringNoopLock stands in for the advisory commit lock: it is not what is under test
// here (the writer lease is), so it always admits.
func wiringNoopLock(safecommit.LockOptions) (func(), error) { return func() {}, nil }

// wiringClone builds a temp repo on branch main with one pushed commit, so
// safesync.Assess sees a clean in-sync state and safecommit's trunk gate passes.
func wiringClone(t *testing.T) string {
	t.Helper()
	root := t.TempDir()
	origin := filepath.Join(root, "origin.git")
	clone := filepath.Join(root, "clone")
	if out, err := exec.Command("git", "init", "--bare", origin).CombinedOutput(); err != nil {
		t.Fatalf("git init --bare: %v\n%s", err, out)
	}
	if out, err := exec.Command("git", "init", "-b", "main", clone).CombinedOutput(); err != nil {
		t.Fatalf("git init: %v\n%s", err, out)
	}
	wiringGit(t, clone, "config", "user.name", "wiring test")
	wiringGit(t, clone, "config", "user.email", "wiring@test.invalid")
	wiringGit(t, clone, "config", "commit.gpgsign", "false")
	if err := os.WriteFile(filepath.Join(clone, "a.txt"), []byte("v1\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	wiringGit(t, clone, "add", "--", "a.txt")
	wiringGit(t, clone, "commit", "-m", "v1")
	wiringGit(t, clone, "remote", "add", "origin", origin)
	wiringGit(t, clone, "push", "origin", "main")
	return clone
}

func wiringCommitOptions(clone string) safecommit.Options {
	return safecommit.Options{
		Dir:     clone,
		Paths:   []string{"w.txt"},
		Message: "test: managed write under the lease pairing witness",
	}
}

// TestManagedCommitRefusedWhileSyncApplyHoldsWriterLease is the #4611 witness for the
// first direction: while safesync.Apply holds the worktree writer lease (intercepted on
// its first git call, which happens strictly inside the lease window), a fak-managed
// commit must refuse with WRITER_LEASE_HELD naming the holder — and the same commit must
// succeed once the apply window has closed and released the lease.
func TestManagedCommitRefusedWhileSyncApplyHoldsWriterLease(t *testing.T) {
	clone := wiringClone(t)
	if err := os.WriteFile(filepath.Join(clone, "w.txt"), []byte("mine\n"), 0o644); err != nil {
		t.Fatal(err)
	}

	fired := false
	var midRes safecommit.Result
	var midErr error
	runner := func(ctx context.Context, repo string, args ...string) safesync.RunResult {
		if !fired {
			fired = true
			// Apply holds the lease RIGHT NOW: the managed commit must be refused
			// before it mutates anything.
			midRes, midErr = safecommit.CommitWith(ctx, wiringCommitRunner, wiringNoopLock, wiringCommitOptions(clone))
		}
		return safesync.RealRunner(ctx, repo, args...)
	}

	info, err := safesync.Apply(context.Background(), safesync.Options{
		Repo: clone, Remote: "origin", Branch: "main", Runner: runner, LeaseOwner: "sync-apply",
	})
	if err != nil {
		t.Fatalf("apply: %v", err)
	}
	if !fired {
		t.Fatal("interception runner never fired; the mid-window commit was not exercised")
	}
	if info.State != safesync.StateInSync {
		t.Fatalf("apply state = %q, want %q (the clone is pushed and clean)", info.State, safesync.StateInSync)
	}

	if midErr != nil {
		t.Fatalf("mid-window commit must refuse as a value, not an error: %v", midErr)
	}
	if midRes.Reason != safecommit.ReasonWriterLeaseHeld {
		t.Fatalf("mid-window commit Reason = %q, want %q", midRes.Reason, safecommit.ReasonWriterLeaseHeld)
	}
	if midRes.Committed {
		t.Fatal("mid-window commit reported Committed despite the writer-lease refusal")
	}
	if !strings.Contains(midRes.Detail, "sync-apply") {
		t.Fatalf("refusal Detail %q does not name the holding owner sync-apply", midRes.Detail)
	}
	if code, ok := safecommit.RefusalExitCode(midRes.Reason); !ok || code != safecommit.ExitLockBusy {
		t.Fatalf("WRITER_LEASE_HELD classifies as (%d, %v), want the retryable contention exit %d", code, ok, safecommit.ExitLockBusy)
	}

	// The window is closed: the identical managed commit now lands cleanly, proving the
	// refusal was the lease, not the repo state.
	res, err := safecommit.CommitWith(context.Background(), wiringCommitRunner, wiringNoopLock, wiringCommitOptions(clone))
	if err != nil {
		t.Fatalf("post-window commit: %v", err)
	}
	if !res.Committed || !res.Verified || res.Reason != "" {
		t.Fatalf("post-window commit did not land cleanly: %+v", res)
	}
}

// TestSyncApplyRefusedWhileManagedCommitHoldsWriterLease is the #4611 witness for the
// second direction: while a fak-managed commit holds the worktree writer lease
// (intercepted on its lock-riding `git add`, which runs strictly inside the lease
// window), safesync.Apply must refuse to enter its assess+apply window, naming the
// fak-commit holder — while the commit itself still lands and releases the lease.
func TestSyncApplyRefusedWhileManagedCommitHoldsWriterLease(t *testing.T) {
	clone := wiringClone(t)
	if err := os.WriteFile(filepath.Join(clone, "w.txt"), []byte("mine\n"), 0o644); err != nil {
		t.Fatal(err)
	}

	fired := false
	var midInfo safesync.Assessment
	var midErr error
	runner := func(ctx context.Context, dir string, args ...string) (string, int, error) {
		if len(args) > 0 && args[0] == "add" && !fired {
			fired = true
			// The commit holds the lease RIGHT NOW (its first tree mutation is about
			// to run): sync apply must refuse without touching the tree.
			midInfo, midErr = safesync.Apply(ctx, safesync.Options{
				Repo: clone, Remote: "origin", Branch: "main", LeaseOwner: "sync-apply",
			})
		}
		return wiringCommitRunner(ctx, dir, args...)
	}

	res, err := safecommit.CommitWith(context.Background(), runner, wiringNoopLock, wiringCommitOptions(clone))
	if err != nil {
		t.Fatalf("commit: %v", err)
	}
	if !fired {
		t.Fatal("interception runner never saw the lock-riding `git add`; the mid-window apply was not exercised")
	}
	if !res.Committed || !res.Verified || res.Reason != "" {
		t.Fatalf("managed commit did not land cleanly while correctly holding the lease: %+v", res)
	}

	if midErr != nil {
		t.Fatalf("mid-window apply must refuse as a value, not an error: %v", midErr)
	}
	if midInfo.Applied || midInfo.OK {
		t.Fatalf("mid-window apply was not refused: %+v", midInfo)
	}
	if midInfo.Lease == nil || midInfo.Lease.Owner != "fak-commit" {
		t.Fatalf("mid-window apply refusal does not name the fak-commit lease holder: %+v", midInfo.Lease)
	}

	// The commit released the lease on return: a fresh writer can take it immediately.
	l, err := safesync.AcquireWriterLease(clone, "after", nil, 0)
	if err != nil {
		t.Fatalf("lease not released after the managed commit returned: %v", err)
	}
	_ = l.Release()
}

// TestManagedCommitWithTimeoutWaitsForSyncApply proves that when safecommit is configured
// with a Lock.Timeout > 0, it queues for the worktree writer lease and lands cleanly after
// the sync apply window closes, rather than immediately failing with WRITER_LEASE_HELD (#11234/#11616).
func TestManagedCommitWithTimeoutWaitsForSyncApply(t *testing.T) {
	clone := wiringClone(t)
	if err := os.WriteFile(filepath.Join(clone, "w.txt"), []byte("mine\n"), 0o644); err != nil {
		t.Fatal(err)
	}

	applyLease, err := safesync.AcquireWriterLease(clone, "sync-apply", nil, time.Minute)
	if err != nil {
		t.Fatalf("acquire writer lease: %v", err)
	}

	commitDone := make(chan struct{})
	var commitRes safecommit.Result
	var commitErr error

	opts := wiringCommitOptions(clone)
	opts.Lock = safecommit.LockOptions{
		Timeout: 2 * time.Second,
	}

	go func() {
		defer close(commitDone)
		commitRes, commitErr = safecommit.CommitWith(context.Background(), wiringCommitRunner, wiringNoopLock, opts)
	}()

	time.Sleep(50 * time.Millisecond)
	_ = applyLease.Release()

	<-commitDone
	if commitErr != nil {
		t.Fatalf("commit error: %v", commitErr)
	}
	if !commitRes.Committed || !commitRes.Verified || commitRes.Reason != "" {
		t.Fatalf("commit failed to land after queue wait: %+v", commitRes)
	}
}
