package treedoctor

// Gitignored-scratch reaper (#3211) — the one destructive verb tree-doctor is allowed to run
// unattended, and the reason it is safe: `git clean -X` removes ONLY files git already ignores.
//
// In an always-on fleet, agents and tools drop scratch into the repo — a `_scratch/` note, a
// throwaway `.tmp`, a stray build artifact. Left to accumulate it slows every `git status` and
// every tree walk. The doctrine treedoctor enforces — NEVER delete a peer's live work — usually
// forbids removing an untracked file, because a load-bearing-but-unlanded file is byte-for-byte
// indistinguishable from cruft (see wipaudit.go). The `-X` flag is what makes THIS sweep the
// exception that keeps the doctrine intact: `-X` restricts removal to files matched by a
// .gitignore rule, so by construction it can touch neither a TRACKED file nor a REAL untracked
// WIP file — only scratch a human already declared disposable by gitignoring it. `-d` recurses
// into ignored directories; `-f` performs the delete; `-n` (dry-run) previews without deleting.

import (
	"context"
	"fmt"
	"strings"
)

// ScratchSweepTimeout bounds the `git clean` invocation so a wedged filesystem walk (a huge
// ignored tree, a slow network mount) can never stall the doctor. The caller wraps its context
// with this deadline; the invocation is a bounded exec.CommandContext, never an unbounded one.
const ScratchSweepTimeout = 60 // seconds — kept as a bare int so callers pick the time unit

// ScratchSweepResult reports what a gitignored-scratch reap removed (or, in dry-run, WOULD
// remove). Removed holds the repo-relative paths git clean reported, in git's own order.
type ScratchSweepResult struct {
	DryRun  bool     `json:"dry_run"`
	Removed []string `json:"removed"`
}

// SweepScratch reaps gitignored scratch under repoRoot via `git clean -Xdf`. The `-X` flag is
// the load-bearing safety guarantee: it removes ONLY git-ignored files, so it can never delete
// a tracked file or a real (non-ignored) untracked WIP file — only scratch a .gitignore rule
// already marked disposable. With dryRun it runs `git clean -Xdn`, listing what WOULD be reaped
// while deleting nothing. The git effect goes through the injected Runner so the whole thing is
// testable against a real temp repo (the runner supplies the bounded exec.CommandContext).
func SweepScratch(ctx context.Context, run Runner, repoRoot string, dryRun bool) (ScratchSweepResult, error) {
	res := ScratchSweepResult{DryRun: dryRun}
	// -X: ignored-only (never tracked, never real untracked WIP). -d: recurse ignored dirs.
	// -f: force the delete (git refuses without -f or -n). -n: dry-run preview, delete nothing.
	mode := "-Xdf"
	if dryRun {
		mode = "-Xdn"
	}
	out, code, err := run(ctx, repoRoot, "clean", mode)
	if err != nil {
		return res, fmt.Errorf("git clean %s: %w", mode, err)
	}
	if code != 0 {
		return res, fmt.Errorf("git clean %s exited %d: %s", mode, code, strings.TrimSpace(out))
	}
	res.Removed = parseScratchCleanOutput(out)
	return res, nil
}

// parseScratchCleanOutput extracts the removed paths from `git clean` output. Real removals are
// reported as "Removing <path>"; the -n dry-run reports "Would remove <path>". Any other line
// (progress, warnings) is ignored, so the result is exactly the set of paths git acted on.
func parseScratchCleanOutput(out string) []string {
	var removed []string
	for _, line := range strings.Split(out, "\n") {
		line = strings.TrimRight(line, "\r")
		switch {
		case strings.HasPrefix(line, "Removing "):
			removed = append(removed, strings.TrimSpace(strings.TrimPrefix(line, "Removing ")))
		case strings.HasPrefix(line, "Would remove "):
			removed = append(removed, strings.TrimSpace(strings.TrimPrefix(line, "Would remove ")))
		}
	}
	return removed
}
