package safecommit

import (
	"context"
	"sort"
	"strings"
	"time"
)

// indexrestore.go — a refused or failed commit restores the index it changed.
//
// CommitWith stages the requested pathspec (`git add --all -- <paths>`) and only then runs
// `git commit`, whose pre-commit hook may refuse (COMMITTED_RED, BUILD_CHECK_TIMEOUT under a
// loaded host, PUBLIC_LEAK, ...). Before this guard the refusal returned with those paths
// still staged, so every retry tripped PRESTAGED_PATH_OVERLAP on the command's OWN earlier
// staging and the only way past was to skip the gate. Now the paths this invocation staged
// are unstaged again (worktree bytes untouched); entries that were already staged before
// the command ran are left exactly as they were.

// indexRestoreTimeout bounds the restore. It deliberately does not share the commit's
// post-validation context: the commonest trigger is that context expiring.
const indexRestoreTimeout = 30 * time.Second

// stagedPathSet lists the index entries that differ from HEAD under pathspec. ok=false means
// the probe could not answer, in which case no restore is attempted (never guess at the index).
func stagedPathSet(ctx context.Context, run Runner, dir string, paths []string) (set map[string]bool, ok bool) {
	args := append([]string{"diff", "--cached", "--name-only", "--no-renames", "-z", "--"}, paths...)
	out, code, err := run(ctx, dir, args...)
	if err != nil || code != 0 {
		return nil, false
	}
	set = map[string]bool{}
	for _, p := range strings.Split(out, "\x00") {
		if p = strings.TrimSpace(p); p != "" {
			set[p] = true
		}
	}
	return set, true
}

// restoreIndexAfterRefusal unstages the paths this commit staged when HEAD did not move.
// It returns the unstaged paths (sorted); nil when nothing needed restoring or the restore
// could not be proven safe.
func restoreIndexAfterRefusal(ctx context.Context, run Runner, dir, headBefore string, paths []string, before map[string]bool) []string {
	rctx, cancel := context.WithTimeout(context.WithoutCancel(ctx), indexRestoreTimeout)
	defer cancel()
	if head, err := headSHA(rctx, run, dir); err != nil || head != headBefore {
		return nil // the commit landed (or HEAD is unreadable): the index is HEAD's now
	}
	after, ok := stagedPathSet(rctx, run, dir, paths)
	if !ok {
		return nil
	}
	var ours []string
	for p := range after {
		if !before[p] {
			ours = append(ours, p)
		}
	}
	if len(ours) == 0 {
		return nil
	}
	sort.Strings(ours)
	args := append([]string{"reset", "-q", "--"}, ours...)
	if headBefore == "" { // unborn branch: there is no HEAD to reset to
		args = append([]string{"rm", "--cached", "-q", "--ignore-unmatch", "--"}, ours...)
	}
	if reason, _, err := runLockRidingMutation(rctx, run, dir, args); err != nil || reason != "" {
		return nil
	}
	return ours
}
