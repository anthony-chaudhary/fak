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
// Only the index lock is auto-reaped. A ref lock (refs/heads/*.lock, packed-refs)
// can be held by a concurrent push whose window is legitimately long, and it
// carries no comparable age guarantee, so ref-lock contention keeps the plain
// ride-then-LOCK_BUSY behavior from lockcontention.go.

import (
	"context"
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

// Injection points for tests: the reaper's clock, stat, and remove. Production
// uses the real filesystem; tests substitute these to exercise stale/fresh/absent
// without racing a real crash.
var (
	staleIndexNow    = time.Now
	staleIndexStat   = os.Stat
	staleIndexRemove = os.Remove
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
}

// reapStaleIndexLock removes <gitDir>/index.lock when it is provably ABANDONED:
// present AND older than threshold. Age is the whole signal — index.lock records
// no owning PID, and git never legitimately holds it for minutes, so an
// over-threshold mtime is a crash artifact with no live owner to disturb. A fresh
// lock (under threshold, or threshold <= 0) is left untouched. Best-effort and
// fail-safe: an absent or unreadable lock reaps nothing, and a remove failure
// leaves the caller's existing LOCK_BUSY path as the backstop.
func reapStaleIndexLock(gitDir string, threshold time.Duration) staleIndexLockReap {
	path := filepath.Join(gitDir, "index.lock")
	res := staleIndexLockReap{Path: path}
	fi, err := staleIndexStat(path)
	if err != nil {
		return res // absent or unreadable => nothing to reap
	}
	res.Present = true
	if age := staleIndexNow().Sub(fi.ModTime()); age > 0 {
		res.AgeSeconds = int64(age / time.Second)
		if threshold > 0 && age >= threshold {
			if staleIndexRemove(path) == nil {
				res.Reaped = true
			}
		}
	}
	return res
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
	reap := reapStaleIndexLock(gitDir, DefaultStaleIndexLockAge)
	switch {
	case reap.Reaped:
		reapEventf("INDEX_LOCK_REAPED age=%ds path=%s", reap.AgeSeconds, reap.Path)
		out, code, rerr := runRidingLockContention(ctx, run, dir, args...)
		r, d, e := classifyMutation(out, code, rerr)
		return r, d, e, true
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
