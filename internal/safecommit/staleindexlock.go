package safecommit

// staleindexlock.go — auto-recovering a CRASHED writer's abandoned .git/index.lock.
//
// lockcontention.go rides out a PEER'S index.lock while a live git briefly holds
// it: the retries land the moment the holder clears. But a git process that
// CRASHED or was KILLED mid-write leaves index.lock behind with no owner, and it
// never clears — so the riding retries exhaust and the commit surfaces as
// LOCK_BUSY carrying git's own text: "a git process may have crashed in this
// repository earlier: remove the file manually to continue". On a shared
// multi-session tree that turned every stale lock into manual `rm` toil for the
// next committer (issue #3915).
//
// The remedy, gated to the INDEX lock only:
//
//  1. STALE ⇒ REAP + RETRY: index.lock records no owning PID and git holds it only
//     for the millisecond an index write takes, so a lock older than the staleness
//     threshold is an unambiguous crash artifact. It is removed and the mutation is
//     run once more — the automatic, auditable form of the manual `rm`.
//  2. FRESH ⇒ CLEAR MESSAGE: a lock under the threshold may be held by a live git,
//     so it is left untouched and the caller reports "another git process is
//     active" instead of git's generic crash text — still the retryable LOCK_BUSY.
//
//  3. NAMED DEAD OWNER ⇒ REAP SOONER: a partial commit (`git commit -- <paths>`, the
//     shape safecommit issues) holds index.lock AND next-index-<pid>.lock through its
//     hooks. When a next-index file pairs with the lock by mtime and its pid is dead
//     (partialcommit.go), the lock's creator is named and proven gone, so the lock is
//     reaped once frozen for DefaultPartialCommitOwnerDeadAge instead of the long
//     no-owner window, and the dead writer's next-index file goes with it.
//
// Only the index lock is auto-reaped. A ref lock (refs/heads/*.lock, packed-refs)
// can be held by a concurrent push whose window is legitimately long, and it
// carries no comparable age guarantee, so ref-lock contention keeps the plain
// ride-then-LOCK_BUSY behavior from lockcontention.go.

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"
)

// DefaultStaleIndexLockAge is how old .git/index.lock must be before the commit
// path treats it as ABANDONED by a crashed writer and reaps it automatically. git
// holds index.lock only for the instant an index write takes (milliseconds), so a
// lock older than this was left by a process that crashed or was killed — never a
// live committer. The threshold is deliberately conservative (minutes, not
// seconds) so a merely-slow live index write is never mistaken for a crash:
// waiting out a live writer costs a retry, but reaping a live lock corrupts a
// peer's in-flight index. It mirrors commitlane.DefaultStaleIndexAge — the age at
// which `fak commit status` already labels the lock "stale" — duplicated here
// because commitlane imports safecommit, so the shared value must live in this
// lower layer.
const DefaultStaleIndexLockAge = 15 * time.Minute

// DefaultPartialCommitOwnerDeadAge is the short frozen window after which an index.lock
// whose partial-commit creator is NAMED (a paired next-index-<pid>.lock) and DEAD is
// reaped. The long DefaultStaleIndexLockAge only ever stood in for "the holder is gone";
// once a dead pid proves it, one frozen minute suffices. It mirrors
// commitlane.DefaultOwnerDeadIndexAge for the same reason DefaultStaleIndexLockAge mirrors
// commitlane.DefaultStaleIndexAge.
const DefaultPartialCommitOwnerDeadAge = 60 * time.Second

// Injection points for tests: the reaper's clock, stat, and remove. Production
// uses the real filesystem; tests substitute these to exercise stale/fresh/absent
// without racing a real crash.
var (
	staleIndexNow    = time.Now
	staleIndexStat   = os.Stat
	staleIndexRemove = os.Remove
	staleIndexGlob   = filepath.Glob
	staleIndexAlive  = processAlive
)

// isIndexLockContention reports whether git's failure names the INDEX lock
// specifically. It is deliberately narrower than isGitLockContention: only the
// index lock is auto-reaped (see the file header), so a ref-lock or packed-refs
// failure must not enter the reap path even though it is also lock contention.
func isIndexLockContention(out string) bool {
	return strings.Contains(strings.ToLower(out), "index.lock")
}

// staleIndexLockReap is the outcome of one probe-and-maybe-remove of index.lock.
type staleIndexLockReap struct {
	Reaped     bool   // the lock was present, stale, and successfully removed
	Present    bool   // an index.lock existed at probe time
	AgeSeconds int64  // its age when probed (for the event and the message)
	Path       string // the lock path probed
	// OwnerPID is the partial-commit creator named by a paired next-index-<pid>.lock
	// (0 when no residue pairs), and OwnerDead whether every pairing writer is gone.
	OwnerPID  int
	OwnerDead bool
	// ReapedByOwner marks a reap proven by the dead named owner rather than by age;
	// PartnerPath is that owner's next-index file, removed alongside the lock.
	ReapedByOwner bool
	PartnerPath   string
}

// reapStaleIndexLock removes <gitDir>/index.lock when it is provably ABANDONED:
// present AND older than threshold, or present, frozen for
// DefaultPartialCommitOwnerDeadAge, and paired with a next-index-<pid>.lock whose pid is
// dead (the lock's creator is named and gone). Without a named owner age is the whole
// signal — index.lock records no owning PID, and git never legitimately holds it for
// minutes, so an over-threshold mtime is a crash artifact with no live owner to disturb.
// A fresh, unowned lock (under threshold, or threshold <= 0) is left untouched, and a
// non-positive threshold disables both reaps. Best-effort and fail-safe: an absent or
// unreadable lock reaps nothing, and a remove failure leaves the caller's existing
// LOCK_BUSY path as the backstop.
func reapStaleIndexLock(gitDir string, threshold time.Duration) staleIndexLockReap {
	path := filepath.Join(gitDir, "index.lock")
	res := staleIndexLockReap{Path: path}
	fi, err := staleIndexStat(path)
	if err != nil {
		return res // absent or unreadable => nothing to reap
	}
	res.Present = true
	age := staleIndexNow().Sub(fi.ModTime())
	if age <= 0 {
		return res
	}
	res.AgeSeconds = int64(age / time.Second)
	if threshold <= 0 {
		return res
	}
	if age >= threshold {
		if staleIndexRemove(path) == nil {
			res.Reaped = true
		}
		return res
	}
	res.OwnerPID, res.OwnerDead, _ = partialCommitOwnerOf(gitDir, fi.ModTime())
	if res.OwnerPID > 0 && res.OwnerDead && age >= DefaultPartialCommitOwnerDeadAge {
		if staleIndexRemove(path) == nil {
			res.Reaped, res.ReapedByOwner = true, true
			partner := filepath.Join(gitDir, NextIndexLockName(res.OwnerPID))
			if rerr := staleIndexRemove(partner); rerr == nil || errors.Is(rerr, os.ErrNotExist) {
				res.PartnerPath = partner
			}
		}
	}
	return res
}

// partialCommitOwnerOf scans gitDir's next-index-<pid>.lock residue and names the
// partial-commit creator of an index.lock last modified at indexMod (see
// PartialCommitOwner). Liveness is probed only for files that pair by mtime, so a pile of
// old residue costs a stat each, not a process probe each.
func partialCommitOwnerOf(gitDir string, indexMod time.Time) (pid int, dead bool, ok bool) {
	matches, err := staleIndexGlob(filepath.Join(gitDir, NextIndexLockGlob))
	if err != nil {
		return 0, false, false
	}
	var cands []NextIndexCandidate
	for _, m := range matches {
		cpid := NextIndexLockPID(filepath.Base(m))
		if cpid <= 0 {
			continue
		}
		fi, serr := staleIndexStat(m)
		if serr != nil || !NextIndexPairsIndexLock(indexMod, fi.ModTime()) {
			continue
		}
		cands = append(cands, NextIndexCandidate{PID: cpid, ModTime: fi.ModTime(), Alive: staleIndexAlive(cpid)})
	}
	return PartialCommitOwner(indexMod, cands)
}

// resolveGitDir returns the absolute .git directory for dir via the injected
// Runner, or "" when it cannot be resolved — in which case the caller declines to
// reap and falls back to the plain LOCK_BUSY classification. It reuses the commit
// Runner so it is exercised through the same fake in tests.
func resolveGitDir(ctx context.Context, run Runner, dir string) string {
	out, code, err := run(ctx, dir, "rev-parse", "--absolute-git-dir")
	if err != nil || code != 0 {
		return ""
	}
	return strings.TrimSpace(out)
}

// recoverStaleIndexLock attempts the automatic form of `rm .git/index.lock` +
// retry after the riding loop has exhausted on an index-lock failure. It returns
// handled=true when it produced a terminal outcome for the mutation:
//   - the lock was stale: it was reaped and the mutation re-run, and (reason,
//     detail, err) is that re-run's classification;
//   - the lock was fresh (a live git likely holds it): (LOCK_BUSY, "another git
//     process is active …", nil) — git's generic crash text replaced by a precise,
//     action-shaped message.
//
// It returns handled=false when it declines — the git dir could not be resolved,
// or the lock vanished between the riding loop and the probe — leaving the caller
// to classify the original failure unchanged.
func recoverStaleIndexLock(ctx context.Context, run Runner, dir string, args []string) (reason, detail string, err error, handled bool) {
	gitDir := resolveGitDir(ctx, run, dir)
	if gitDir == "" {
		return "", "", nil, false
	}
	if redirected, _ := indexRedirected(ctx, run, dir, gitDir); redirected {
		// Inside another git's hook GIT_INDEX_FILE is inherited: the lock git just refused
		// on is <that file>.lock, and <gitdir>/index.lock is the OUTER git's live lock (its
		// name still matches isIndexLockContention). Neither reap may touch it. An
		// unanswered query keeps the long-standing behavior rather than disabling the reap.
		return "", "", nil, false
	}
	reap := reapStaleIndexLock(gitDir, DefaultStaleIndexLockAge)
	switch {
	case reap.Reaped:
		if reap.ReapedByOwner {
			reapEventf("INDEX_LOCK_REAPED partial_commit_owner_dead pid=%d age=%ds path=%s partner=%s",
				reap.OwnerPID, reap.AgeSeconds, reap.Path, reap.PartnerPath)
		} else {
			reapEventf("INDEX_LOCK_REAPED age=%ds path=%s", reap.AgeSeconds, reap.Path)
		}
		out, code, rerr := runRidingLockContention(ctx, run, dir, args...)
		r, d, e := classifyMutation(out, code, rerr)
		return r, d, e, true
	case reap.Present && reap.OwnerPID > 0 && reap.OwnerDead:
		// Named dead owner, not yet frozen for the short window: say so, so the operator
		// sees the lock will clear on its own instead of hunting a phantom live writer.
		return ReasonLockBusy,
			fmt.Sprintf("index.lock (held %ds) was left by dead partial commit pid %d; it is reaped once frozen %s; retry shortly",
				reap.AgeSeconds, reap.OwnerPID, DefaultPartialCommitOwnerDeadAge),
			nil, true
	case reap.Present && reap.OwnerPID > 0:
		return ReasonLockBusy,
			fmt.Sprintf("another git process is active (partial commit pid %d holds index.lock %ds); retry shortly",
				reap.OwnerPID, reap.AgeSeconds),
			nil, true
	case reap.Present:
		// Fresh index.lock: a live git may hold it. Do not reap; replace git's
		// generic crash text with a precise, retryable message.
		return ReasonLockBusy,
			fmt.Sprintf("another git process is active (index.lock held %ds); retry shortly", reap.AgeSeconds),
			nil, true
	default:
		// The lock vanished between the riding loop and this probe (a peer released
		// it): nothing stale to reap. Let the caller classify the original failure.
		return "", "", nil, false
	}
}
