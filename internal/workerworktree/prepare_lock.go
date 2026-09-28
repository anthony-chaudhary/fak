package workerworktree

import (
	"errors"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/flock"
)

// THE PREPARE TARGET LOCK
//
// Every Prepare attempt for one (lane, key) derives the same worktree path. With
// no exclusion, a retry can adopt ("reused") the tree an earlier attempt is still
// materializing, and that attempt's fail-open cleanup (`worktree remove --force`
// after PREPARE_TIMEOUT) then deletes the tree the retry just handed to its
// worker. The prepare-time dead sweep has the same check-then-delete shape
// against an adoption. One advisory file lock per target path serializes the
// materialize → verify → stamp → cleanup sequence of a Prepare and the sweep's
// final re-check-and-remove, so a target is only deleted by whoever holds it.
//
// Lock files live in a sidecar directory beside the worktrees and are never
// unlinked: removing a lock file another process holds open would let a third
// process lock a fresh file and break the exclusion. They are empty, one per
// target ever prepared.

const (
	prepareLockDir = ".fak-worker-prepare-locks"
	prepareLockExt = ".lock"

	// PrepareLockWaitEnv bounds how long an unbounded Prepare waits for another
	// attempt on the same target (a Go duration such as "30s"). A bounded prepare
	// waits at most its remaining budget instead.
	PrepareLockWaitEnv = "FAK_WORKER_PREPARE_LOCK_WAIT"
	// defaultPrepareLockWait outlasts one bounded CLI prepare (2m budget plus the
	// 30s partial cleanup), so a retry queued behind it observes its final state.
	defaultPrepareLockWait = 2*time.Minute + 30*time.Second
	prepareLockPoll        = 25 * time.Millisecond

	// PrepareCodeBusy reports that another prepare attempt holds this target. The
	// refused attempt touched nothing; retry once that attempt has reported.
	PrepareCodeBusy = "PREPARE_BUSY"
	// PrepareCodeReuseLost reports a same-key reuse whose checkout or git
	// registration disappeared before prepare could emit its receipt.
	PrepareCodeReuseLost = "PREPARE_REUSE_LOST"
)

var errPrepareTargetBusy = errors.New("prepare target is held by another attempt")

func prepareLockPath(wtPath string) string {
	clean := filepath.Clean(wtPath)
	return filepath.Join(filepath.Dir(clean), prepareLockDir, filepath.Base(clean)+prepareLockExt)
}

// withPrepareTargetLock runs fn while holding wtPath's target lock, polling for
// at most wait (wait <= 0 is one non-blocking attempt). errPrepareTargetBusy
// reports contention; any other error means the lock could not be established,
// and callers fail toward not touching the target.
func withPrepareTargetLock(wtPath string, wait time.Duration, fn func()) error {
	lockPath := prepareLockPath(wtPath)
	if err := os.MkdirAll(filepath.Dir(lockPath), poolStateDirPerm); err != nil {
		return err
	}
	f, err := os.OpenFile(lockPath, os.O_CREATE|os.O_RDWR, poolLockFilePerm)
	if err != nil {
		return err
	}
	defer f.Close()
	deadline := time.Now().Add(wait)
	for {
		err = flock.TryLock(f)
		if err == nil {
			break
		}
		if !errors.Is(err, flock.ErrLockBusy) {
			return err
		}
		if !time.Now().Before(deadline) {
			return errPrepareTargetBusy
		}
		time.Sleep(prepareLockPoll)
	}
	defer func() { _ = flock.Unlock(f) }()
	fn()
	return nil
}

func prepareLockWait() time.Duration {
	if v := strings.TrimSpace(os.Getenv(PrepareLockWaitEnv)); v != "" {
		if d, err := time.ParseDuration(v); err == nil && d >= 0 {
			return d
		}
	}
	return defaultPrepareLockWait
}

// worktreeListing is one block of `git worktree list --porcelain`.
type worktreeListing struct {
	Path     string
	Prunable bool
}

func parseWorktreeListing(porcelain string) []worktreeListing {
	var out []worktreeListing
	for _, line := range strings.Split(porcelain, "\n") {
		line = strings.TrimRight(line, "\r")
		switch {
		case strings.HasPrefix(line, "worktree "):
			out = append(out, worktreeListing{Path: strings.TrimSpace(line[len("worktree "):])})
		case line == "prunable" || strings.HasPrefix(line, "prunable "):
			if len(out) > 0 {
				out[len(out)-1].Prunable = true
			}
		}
	}
	return out
}

// liveWorktreeRegistration reports whether git registers wt as a linked
// worktree it would keep: listed, not flagged prunable, and still carrying its
// `.git` link. A registration whose checkout was gutted by a concurrent deleter
// is listed but prunable, and must never be reported as a reusable worktree.
// listOK is false when git could not produce the listing at all.
func liveWorktreeRegistration(root, wt string, git GitRunner) (live, listOK bool) {
	rc, out := run(git, root, []string{"worktree", "list", "--porcelain"})
	if rc != 0 {
		return false, false
	}
	if _, err := os.Lstat(filepath.Join(wt, ".git")); err != nil {
		return false, true
	}
	for _, entry := range parseWorktreeListing(out) {
		if samePath(entry.Path, wt) {
			return !entry.Prunable, true
		}
	}
	return false, true
}
