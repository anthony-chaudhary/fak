package safecommit

// killedcommit.go — rolling back the index locks of a `git commit` safecommit killed.
//
// When the post-validation deadline fires while `git commit -- <paths>` is still inside the
// repository's hooks, the runner kills git's process tree. A killed git runs none of its
// own cleanup, so it leaves exactly the two lockfiles a partial commit holds through its
// hooks: <gitdir>/index.lock and <gitdir>/next-index-<pid>.lock (see partialcommit.go).
// Left behind, they wedge the shared index for every peer: git refuses to take index.lock,
// the lane observer reports it busy, and a peer's land that finds it skips its working-tree
// sync (the 2026-09-26 incident that left pkg/deploykit/* missing from the shared tree).
//
// rollbackKilledCommit performs git's own failure rollback (rollback_index_files: index.lock
// first, then the temporary index) on git's behalf, and ONLY for locks this runner can
// prove belong to the commit it just killed. The proof deliberately does not use the pid
// safecommit started: on the Windows fleet `git` resolves to the Git-for-Windows launcher
// (Git\cmd\git.exe), and it is the launcher's CHILD, mingw64 git.exe, whose pid names
// next-index-<pid>.lock. Instead a lock is ours when a next-index-<pid>.lock was created
// during this run, its named pid is dead, and it pairs with index.lock by mtime (index.lock
// not newer — see PartialCommitOwner). A live pairing writer, residue older than the run,
// or an index.lock newer than every dead partner of this run is left for the
// evidence-gated reapers. Neither removal touches .git/index itself: a partial commit only
// ever wrote its new index into index.lock, so discarding the lock is a true rollback,
// never the loss of a committed index.
//
// One environment voids the pairing: a `fak commit` running inside an OUTER git's hook
// inherits GIT_INDEX_FILE (git exports <gitdir>/index.lock there for a normal commit), so
// our git locked <that file>.lock while the outer, still-live git holds <gitdir>/index.lock
// — which our dead next-index file would then "pair" with. Under a redirected index (asked
// of git itself, see indexRedirected; an unanswered query counts as redirected) the rollback
// therefore never touches index.lock; it only removes this run's dead next-index files,
// which also keeps them from misleading the age-free reapers later.

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"time"
)

// killedCommitRollback is the outcome of one post-kill rollback attempt.
type killedCommitRollback struct {
	GitDir        string
	IndexLockPath string
	// OwnerPID is the dead partial-commit writer whose next-index file paired with
	// index.lock (0 when none of this run's residue paired).
	OwnerPID int
	// RemovedIndexLock: the paired index.lock was present and is gone.
	RemovedIndexLock bool
	// IndexLockUnpaired: an index.lock is present but no dead next-index file of this run
	// proves it ours, so it was kept.
	IndexLockUnpaired bool
	// IndexRedirected: git resolves the index somewhere other than <gitdir>/index (an
	// inherited GIT_INDEX_FILE), or could not say, so <gitdir>/index.lock was never
	// considered ours.
	IndexRedirected bool
	// RemovedNextIndex lists this run's dead-owner next-index files that were removed.
	RemovedNextIndex []string
	Err              string
}

// Seams for the post-kill rollback: the git-dir resolver (the commit's own context is
// already dead, so it runs on a fresh bounded one), the residue lister, the liveness
// probe, and the remover.
var (
	killedCommitGitDir = resolveKilledCommitGitDir
	killedCommitGlob   = filepath.Glob
	killedCommitAlive  = processAlive
	killedCommitRemove = os.Remove
	killedCommitSleep  = time.Sleep
)

const (
	killedCommitResolveTimeout = 15 * time.Second
	// killedCommitClockSlack widens "created during this run" backwards. File times come
	// from a coarser clock than time.Now (the Windows system tick, whole seconds on FAT or
	// SMB), so a lock git created just after Start can carry a stamp slightly before it.
	// Widening only admits more DEAD-owner residue, which is safe to discard anyway.
	killedCommitClockSlack = 2 * time.Second
	// A descendant the tree kill terminated may still be releasing an inherited handle
	// for a moment after git's own exit is observed (TerminateProcess is asynchronous on
	// Windows), so a sharing-violation remove is retried briefly before it is reported.
	killedCommitRemoveAttempts = 10
	killedCommitRemoveBackoff  = 100 * time.Millisecond
)

// resolveKilledCommitGitDir resolves dir's absolute git directory (the per-worktree one,
// where git keeps index.lock and next-index-<pid>.lock) and whether git's index is
// redirected away from it, on a fresh bounded context. An unanswered redirect query fails
// CLOSED (redirected), so the rollback then never touches index.lock. It builds commands
// with newGitCmd directly rather than through realRunner, which calls back into the
// rollback.
func resolveKilledCommitGitDir(dir string) (gitDir string, redirected bool) {
	ctx, cancel := context.WithTimeout(context.Background(), killedCommitResolveTimeout)
	defer cancel()
	run := func(ctx context.Context, dir string, args ...string) (string, int, error) {
		out, err := newGitCmd(ctx, dir, args...).Output()
		if err != nil {
			return "", -1, err
		}
		return string(out), 0, nil
	}
	gitDir = resolveGitDir(ctx, run, dir)
	if gitDir == "" {
		return "", true
	}
	red, known := indexRedirected(ctx, run, dir, gitDir)
	return gitDir, red || !known
}

// rollbackKilledCommit resolves dir's git directory and rolls back the index locks left by
// the killed `git commit` that started at started. It is best-effort and fail-safe: an
// unresolvable git dir, no dead next-index file from this run, a live pairing writer, or an
// index.lock that does not pair all leave the lock for the evidence-gated reapers instead
// of guessing. Every outcome that touched or kept a lock is announced through reapEventf.
func rollbackKilledCommit(dir string, started time.Time) killedCommitRollback {
	gitDir, redirected := killedCommitGitDir(dir)
	if gitDir == "" {
		reapEventf("INDEX_LOCK_ROLLBACK_FAILED killed_commit detail=%q", "git dir unresolved")
		return killedCommitRollback{Err: "git dir unresolved"}
	}
	res := rollbackKilledCommitIn(gitDir, started, redirected)
	switch {
	case res.RemovedIndexLock:
		reapEventf("INDEX_LOCK_ROLLED_BACK killed_commit owner_pid=%d index_lock=%s next_index=%s",
			res.OwnerPID, res.IndexLockPath, strings.Join(res.RemovedNextIndex, ","))
	case res.IndexLockUnpaired:
		reapEventf("INDEX_LOCK_KEPT killed_commit reason=unpaired index_lock=%s removed_next_index=%s",
			res.IndexLockPath, strings.Join(res.RemovedNextIndex, ","))
	case res.IndexRedirected && len(res.RemovedNextIndex) > 0:
		reapEventf("INDEX_LOCK_KEPT killed_commit reason=git_index_file_redirected removed_next_index=%s",
			strings.Join(res.RemovedNextIndex, ","))
	case len(res.RemovedNextIndex) > 0:
		reapEventf("NEXT_INDEX_ROLLED_BACK killed_commit next_index=%s", strings.Join(res.RemovedNextIndex, ","))
	}
	if res.Err != "" {
		reapEventf("INDEX_LOCK_ROLLBACK_FAILED killed_commit detail=%q", res.Err)
	}
	return res
}

// rollbackKilledCommitIn is the filesystem half of rollbackKilledCommit for a resolved
// gitDir, so tests can drive it against a temp directory. redirected is the resolver's
// verdict that <gitDir>/index.lock is not the lock the killed commit held.
func rollbackKilledCommitIn(gitDir string, started time.Time, redirected bool) killedCommitRollback {
	res := killedCommitRollback{GitDir: gitDir, IndexLockPath: filepath.Join(gitDir, "index.lock")}
	floor := started.Add(-killedCommitClockSlack)
	matches, err := killedCommitGlob(filepath.Join(gitDir, NextIndexLockGlob))
	if err != nil {
		res.Err = "next-index scan: " + err.Error()
		return res
	}
	var cands []NextIndexCandidate
	paths := map[int]string{}
	for _, m := range matches {
		pid := NextIndexLockPID(filepath.Base(m))
		if pid <= 0 {
			continue
		}
		fi, serr := staleIndexStat(m)
		if serr != nil || fi.ModTime().Before(floor) {
			continue // vanished, or residue that predates this run: not ours to judge
		}
		cands = append(cands, NextIndexCandidate{PID: pid, ModTime: fi.ModTime(), Alive: killedCommitAlive(pid)})
		paths[pid] = m
	}
	if len(cands) == 0 {
		// Nothing from this run names an owner: git was killed outside the window in which
		// it holds both locks, so any index.lock here is unattributed. Leave it.
		return res
	}
	var errs []string
	res.IndexRedirected = redirected
	if idx, ierr := staleIndexStat(res.IndexLockPath); ierr == nil && !res.IndexRedirected {
		pid, dead, ok := PartialCommitOwner(idx.ModTime(), cands)
		res.OwnerPID = pid
		if ok && dead && !idx.ModTime().Before(floor) {
			if rerr := removeKilledCommitFile(res.IndexLockPath); rerr != nil {
				errs = append(errs, "index.lock: "+rerr.Error())
			} else {
				res.RemovedIndexLock = true
			}
		} else {
			res.IndexLockUnpaired = true
		}
	}
	for _, c := range cands {
		if c.Alive {
			continue // a live writer's temporary index is never ours to discard
		}
		if rerr := removeKilledCommitFile(paths[c.PID]); rerr != nil {
			errs = append(errs, filepath.Base(paths[c.PID])+": "+rerr.Error())
			continue
		}
		res.RemovedNextIndex = append(res.RemovedNextIndex, paths[c.PID])
	}
	res.Err = strings.Join(errs, "; ")
	return res
}

// removeKilledCommitFile removes path, treating an already-absent file as success and
// retrying a transient refusal for a bounded moment.
func removeKilledCommitFile(path string) error {
	var err error
	for attempt := 0; attempt < killedCommitRemoveAttempts; attempt++ {
		err = killedCommitRemove(path)
		if err == nil || errors.Is(err, os.ErrNotExist) {
			return nil
		}
		killedCommitSleep(killedCommitRemoveBackoff)
	}
	return err
}
