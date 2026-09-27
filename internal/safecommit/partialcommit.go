package safecommit

// partialcommit.go — naming the OWNER of a .git/index.lock left by a partial commit.
//
// A bare .git/index.lock records no owner, which is why every reaper here gates on age.
// But the lock safecommit itself orphans most often has a named owner sitting next to it:
// `git commit -- <paths>` (a partial commit, the only commit shape safecommit issues) holds
// TWO lockfiles for its whole run, hooks included —
//
//	(1) lock the real index        → <gitdir>/index.lock          (written, then left open)
//	(4) lock the temporary index   → <gitdir>/next-index-<pid>.lock
//
// and rolls both back (index.lock first) on any failure. A writer killed mid-hook, as a
// commit-phase timeout does, leaves both behind, and the second one's filename names the
// dead writer. Because git writes index.lock BEFORE it creates next-index-<pid>.lock, the
// partner of an index.lock can never be older than it; conversely a new index.lock taken by
// a live peer after the orphan was cleared is always newer than any residue the dead writer
// left. That ordering is what lets a next-index-<pid>.lock with a dead pid PROVE that the
// index.lock beside it is orphaned, instead of waiting out a fifteen-minute age window.
//
// The pairing is shared by the two places that act on it: safecommit's in-commit recovery
// (staleindexlock.go, and the post-kill rollback in killedcommit.go) and commitlane's
// `fak commit status --reclaim-stale-index-lock` decision.

import (
	"context"
	"fmt"
	"path/filepath"
	"regexp"
	"runtime"
	"strconv"
	"strings"
	"time"
)

// indexRedirected asks git — run in dir with this process's environment, which every git
// safecommit starts inherits — where it resolves the index, and reports whether that is
// anywhere but <gitDir>/index. It is the one environment that voids the partial-commit
// pairing: a `fak commit` running inside an OUTER git's hook inherits GIT_INDEX_FILE (a
// normal commit exports <gitdir>/index.lock there), so its own git locks <that file>.lock
// while <gitDir>/index.lock is the outer git's LIVE lock — which a dead inner next-index
// file would otherwise "pair" with. Asking git rather than reading the variable keeps
// git's own resolution authoritative. known is false when git could not answer (for
// example a git older than 2.31, which lacks --path-format).
func indexRedirected(ctx context.Context, run Runner, dir, gitDir string) (redirected, known bool) {
	out, code, err := run(ctx, dir, "rev-parse", "--path-format=absolute", "--git-path", "index")
	if err != nil || code != 0 {
		return false, false
	}
	idx := strings.TrimSpace(out)
	if idx == "" {
		return false, false
	}
	return !sameGitPath(idx, filepath.Join(gitDir, "index")), true
}

// sameGitPath compares two paths git printed, tolerating git's forward slashes on Windows
// and that filesystem's case-insensitivity.
func sameGitPath(a, b string) bool {
	a, b = filepath.Clean(filepath.FromSlash(a)), filepath.Clean(filepath.FromSlash(b))
	if runtime.GOOS == "windows" {
		return strings.EqualFold(a, b)
	}
	return a == b
}

// NextIndexLockGlob matches a partial commit's temporary index locks inside a git dir.
const NextIndexLockGlob = "next-index-*.lock"

// NextIndexLockName is the basename git gives a partial commit's temporary index lock:
// next-index-<pid>.lock, where pid is the git process writing it.
func NextIndexLockName(pid int) string {
	return fmt.Sprintf("next-index-%d.lock", pid)
}

var nextIndexLockPIDRe = regexp.MustCompile(`(?i)^next-index-(\d+)\.lock$`)

// NextIndexLockPID extracts the writer pid from a next-index-<pid>.lock basename, or 0
// when the name does not have that exact shape (for example the nested
// next-index-<pid>.lock.lock a hook's own `git add` creates).
func NextIndexLockPID(base string) int {
	m := nextIndexLockPIDRe.FindStringSubmatch(base)
	if m == nil {
		return 0
	}
	pid, err := strconv.Atoi(m[1])
	if err != nil || pid <= 0 {
		return 0
	}
	return pid
}

// NextIndexPairsIndexLock reports whether a next-index-<pid>.lock last modified at nextMod
// can be the partial-commit partner of an index.lock last modified at indexMod. git writes
// index.lock before it creates the temporary index, so the partner is never older than the
// lock; residue from an earlier, dead writer is always older than a lock a live peer took
// afterwards. A zero time on either side never pairs.
func NextIndexPairsIndexLock(indexMod, nextMod time.Time) bool {
	if indexMod.IsZero() || nextMod.IsZero() {
		return false
	}
	return !nextMod.Before(indexMod)
}

// NextIndexCandidate is one observed <gitdir>/next-index-<pid>.lock, as the pairing needs
// it: the pid its filename names, its mtime, and whether that pid is still running.
type NextIndexCandidate struct {
	PID     int
	ModTime time.Time
	Alive   bool
}

// PartialCommitOwner names the creator of an index.lock (mtime indexMod) from the observed
// next-index residue. ok is false when no candidate pairs with the lock — the lock then has
// no provable owner and only the age gates apply. When candidates pair, pid is the earliest
// one (the temporary index the lock's owner created right after writing it), and dead is
// true only when EVERY pairing candidate's owner is gone: a live pairing writer is a live
// partial commit that may own this lock, so it always wins. Candidates without a pid never
// pair, because a filename that names nobody proves nothing.
func PartialCommitOwner(indexMod time.Time, cands []NextIndexCandidate) (pid int, dead bool, ok bool) {
	var earliest time.Time
	anyAlive := false
	for _, c := range cands {
		if c.PID <= 0 || !NextIndexPairsIndexLock(indexMod, c.ModTime) {
			continue
		}
		if c.Alive {
			anyAlive = true
		}
		if !ok || c.ModTime.Before(earliest) {
			pid, earliest, ok = c.PID, c.ModTime, true
		}
	}
	if !ok {
		return 0, false, false
	}
	return pid, !anyAlive, true
}
