package gitdaily

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"time"

	"github.com/anthony-chaudhary/fak/internal/commitlane"
)

// IndexLockSweep is the evidence-backed outcome for git's index lock and the
// next-index-<pid>.lock temp files left by interrupted index writers. Reaped carries the
// files removed, or in a dry run the files that would be removed. Paths are relative to
// the repo's git directory and slash-normalized for stable cross-platform output.
type IndexLockSweep struct {
	Reaped []string `json:"reaped,omitempty"`
	Err    string   `json:"err,omitempty"`
}

// IndexLockSweepFunc is the injectable seam for the index-lock half of a daily tick.
type IndexLockSweepFunc func(ctx context.Context, repoRoot string, now time.Time, apply bool) IndexLockSweep

// SweepIndexLocks removes only index lock files that commitlane's existing decision
// layer proves abandoned. It deliberately reuses the same evidence as `fak commit
// status --reclaim-stale-index-lock`: an advancing index.lock is always kept; a frozen
// index.lock reaps only after its stale/dead-owner bar; and a next-index file additionally
// requires its PID-named owner to be dead. This closes the otherwise-fatal ordering gap
// in a daily maintenance tick: gitgate defers object folding while index.lock exists, so
// cleaning every other lock but leaving an orphaned index.lock still folds nothing.
func SweepIndexLocks(ctx context.Context, repoRoot string, now time.Time, apply bool) IndexLockSweep {
	return sweepIndexLocksWith(ctx, repoRoot, now, apply, commitlane.Status, os.Remove)
}

type commitLaneStatusFunc func(context.Context, commitlane.Options) (commitlane.Report, error)
type removeFileFunc func(string) error

func sweepIndexLocksWith(
	ctx context.Context,
	repoRoot string,
	now time.Time,
	apply bool,
	status commitLaneStatusFunc,
	remove removeFileFunc,
) IndexLockSweep {
	rep, err := status(ctx, commitlane.Options{
		Dir:       repoRoot,
		Now:       func() time.Time { return now },
		LocksOnly: true,
	})
	if err != nil {
		return IndexLockSweep{Err: err.Error()}
	}
	if rep.GitDir == "" {
		detail := rep.Reason
		if detail == "" {
			detail = "could not resolve git directory"
		}
		return IndexLockSweep{Err: detail}
	}

	var candidates []string
	if d := commitlane.DecideIndexLockReclaim(rep); d.Reap {
		candidates = append(candidates, d.Path)
	}
	for _, d := range commitlane.DecideNextIndexReclaim(rep) {
		if d.Reap {
			candidates = append(candidates, d.Path)
		}
	}

	var out IndexLockSweep
	for _, path := range candidates {
		rel := relativeLockPath(rep.GitDir, path)
		if !apply {
			out.Reaped = append(out.Reaped, rel)
			continue
		}
		if err := remove(path); err != nil && !errors.Is(err, os.ErrNotExist) {
			err = fmt.Errorf("remove %s: %w", rel, err)
			if out.Err == "" {
				out.Err = err.Error()
			} else {
				out.Err = errors.Join(errors.New(out.Err), err).Error()
			}
			continue
		}
		out.Reaped = append(out.Reaped, rel)
	}
	return out
}

func relativeLockPath(gitDir, path string) string {
	rel, err := filepath.Rel(gitDir, path)
	if err != nil || rel == "." || filepath.IsAbs(rel) || rel == ".." ||
		len(rel) > 3 && rel[:3] == ".."+string(filepath.Separator) {
		return filepath.ToSlash(path)
	}
	return filepath.ToSlash(rel)
}
