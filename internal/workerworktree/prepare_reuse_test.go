package workerworktree

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// deadOwnerPID stands in for the `fak worktree worker prepare` process that
// stamped itself as the owner (the --owner-pid default) and then exited.
const deadOwnerPID = 99999999

// assertPreparedTreeLive fails unless wt still exists with its .git link and git
// still registers it as a worktree it would not prune.
func assertPreparedTreeLive(t *testing.T, repo, wt string) {
	t.Helper()
	if fi, err := os.Stat(wt); err != nil || !fi.IsDir() {
		t.Fatalf("prepared worktree %s vanished: %v", wt, err)
	}
	if _, err := os.Lstat(filepath.Join(wt, ".git")); err != nil {
		t.Fatalf("prepared worktree %s lost its .git link: %v", wt, err)
	}
	listed, prunable, inBlock := false, false, false
	for _, line := range strings.Split(reapProofGit(t, repo, "worktree", "list", "--porcelain"), "\n") {
		line = strings.TrimSpace(line)
		if strings.HasPrefix(line, "worktree ") {
			inBlock = samePath(strings.TrimPrefix(line, "worktree "), wt)
			listed = listed || inBlock
			continue
		}
		if inBlock && strings.HasPrefix(line, "prunable") {
			prunable = true
		}
	}
	if !listed || prunable {
		t.Fatalf("prepared worktree %s registration: listed=%v prunable=%v", wt, listed, prunable)
	}
}

// A prepare whose caller failed after materialization (the CLI refusing its own
// --message/--path pairing) leaves a tree owned by an exited process; the
// same-key retry adopts it with reused=true, owned by another exited process.
// The next prepare anywhere on the host sweeps dead worktrees, and it must not
// delete the tree the retry just handed out (2026-09-26, cand-leak-0926).
func TestPrepareRetryAfterFailedPrepareStaysRegistered(t *testing.T) {
	fixture := newReapProofFixture(t)
	wtRoot := t.TempDir()
	const lane, key = "workerworktree", "cand-leak-0926"
	owner := OwnerStamp{PID: deadOwnerPID, LeaseID: "resolve-workerworktree"}

	first := PrepareOwnedBounded(fixture.repo, lane, key, fixture.base, wtRoot, owner, time.Minute)
	if !first.OK {
		t.Fatalf("first prepare: %+v", first)
	}
	retry := PrepareOwnedBounded(fixture.repo, lane, key, fixture.base, wtRoot, owner, time.Minute)
	if !retry.OK || !retry.Reused || !samePath(retry.Path, first.Path) {
		t.Fatalf("same-key retry must reuse the first tree: first=%+v retry=%+v", first, retry)
	}

	ResetSweepGateForTest()
	t.Cleanup(ResetSweepGateForTest)
	if report := SweepDeadWorktrees(fixture.repo, wtRoot, nil); report.Pruned != 0 {
		t.Fatalf("sweep pruned a just-adopted worktree: %+v", report)
	}
	assertPreparedTreeLive(t, fixture.repo, retry.Path)
}

// timeoutAfterAddBackend materializes the real worktree, runs a concurrent
// same-key prepare while it is notionally still checking out, and then reports
// the PREPARE_TIMEOUT an over-budget checkout returns.
type timeoutAfterAddBackend struct{ during func() }

func (b timeoutAfterAddBackend) Materialize(root, lane, key, baseSHA, wtRoot string, git GitRunner) Result {
	res := gitWorktree{}.Materialize(root, lane, key, baseSHA, wtRoot, git)
	if !res.OK {
		return res
	}
	b.during()
	return Result{OK: false, Code: "PREPARE_TIMEOUT", Path: res.Path, BaseSHA: res.BaseSHA,
		Reason: "checkout outlived the prepare budget", Detail: "context deadline exceeded"}
}

func (timeoutAfterAddBackend) Release(root, wtPath string, git GitRunner) Result {
	return gitWorktree{}.Release(root, wtPath, git)
}

// A retry that starts while an earlier attempt for the same key is still inside
// its budget must never be handed a tree that the earlier attempt's timeout
// cleanup then removes (2026-09-27, cand-leak-0927-fix).
func TestTimedOutPrepareCleanupCannotDeleteAdoptedWorktree(t *testing.T) {
	t.Setenv("FAK_WORKER_PREPARE_LOCK_WAIT", "200ms")
	fixture := newReapProofFixture(t)
	wtRoot := t.TempDir()
	const lane, key = "workerworktree", "cand-leak-0927-fix"
	owner := OwnerStamp{PID: os.Getpid(), LeaseID: "resolve-workerworktree"}

	var retry Result
	first := prepareOwnedWithBackend(fixture.repo, lane, key, fixture.base, wtRoot, defaultGit,
		timeoutAfterAddBackend{during: func() {
			retry = prepareOwnedWithBackend(fixture.repo, lane, key, fixture.base, wtRoot, defaultGit, gitWorktree{}, owner, true)
		}}, owner, true)
	if first.OK || first.Code != "PREPARE_TIMEOUT" {
		t.Fatalf("first attempt must time out: %+v", first)
	}
	if retry.OK {
		assertPreparedTreeLive(t, fixture.repo, retry.Path)
	} else if retry.Code != "PREPARE_BUSY" || !retry.Preserved {
		t.Fatalf("a retry racing an in-flight attempt must refuse without touching it: %+v", retry)
	}

	again := PrepareOwnedBounded(fixture.repo, lane, key, fixture.base, wtRoot, owner, time.Minute)
	if !again.OK {
		t.Fatalf("prepare after the timed-out attempt reported: %+v", again)
	}
	assertPreparedTreeLive(t, fixture.repo, again.Path)
}

// Reuse is reported only for a tree git still registers as live. A registration
// whose checkout lost its .git link (a deleter caught half-way) is listed as
// prunable and must not come back as reused=true.
func TestPrepareDoesNotReuseGuttedRegistration(t *testing.T) {
	fixture := newReapProofFixture(t)
	wtRoot := t.TempDir()
	const lane, key = "workerworktree", "gutted-reuse"
	owner := OwnerStamp{PID: os.Getpid(), LeaseID: "resolve-workerworktree"}

	first := PrepareOwned(fixture.repo, lane, key, fixture.base, wtRoot, defaultGit, owner)
	if !first.OK {
		t.Fatalf("first prepare: %+v", first)
	}
	if err := os.Remove(filepath.Join(first.Path, ".git")); err != nil {
		t.Fatal(err)
	}
	retry := PrepareOwned(fixture.repo, lane, key, fixture.base, wtRoot, defaultGit, owner)
	if retry.OK || retry.Reused {
		t.Fatalf("gutted registration reported as a reusable worktree: %+v", retry)
	}
	if !retry.Preserved {
		t.Fatalf("refusal must preserve the unadjudicated target: %+v", retry)
	}
}

// The sweep's status probe shares one short budget across every worktree it
// inspects and times out on a loaded host. A probe that cannot answer must keep
// the tree: reading a timed-out probe as clean deleted a reused worktree along
// with the edit a worker had just made in it.
func TestSweepKeepsDeadOwnerTreeWhenStatusProbeFails(t *testing.T) {
	fixture := newReapProofFixture(t)
	wtRoot := t.TempDir()
	old := time.Now().Add(-2 * time.Hour)
	prep := PrepareOwned(fixture.repo, "workerworktree", "dirty-probe-timeout", fixture.base, wtRoot, defaultGit,
		OwnerStamp{PID: deadOwnerPID, LeaseID: "resolve-workerworktree", CreatedAt: old})
	if !prep.OK {
		t.Fatalf("prepare: %+v", prep)
	}
	if err := WriteWorkerLease(prep.Path, WorkerLease{PID: deadOwnerPID, SessionID: "exited-cli", CreatedAt: old, HeartbeatTS: old}); err != nil {
		t.Fatal(err)
	}
	edit := filepath.Join(prep.Path, "target.txt")
	if err := os.WriteFile(edit, []byte("worker edit\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	statusTimesOut := func(dir string, args []string) (int, string) {
		if len(args) > 0 && args[0] == "status" {
			return ReapTimeoutExitCode, "context deadline exceeded"
		}
		return defaultGit(dir, args)
	}

	if report := SweepDeadWorktrees(fixture.repo, wtRoot, statusTimesOut); report.Pruned != 0 {
		t.Fatalf("sweep deleted a tree whose status it could not read: %+v", report)
	}
	if got, err := os.ReadFile(edit); err != nil || string(got) != "worker edit\n" {
		t.Fatalf("worker edit lost: %q, %v", got, err)
	}
	assertPreparedTreeLive(t, fixture.repo, prep.Path)
}
