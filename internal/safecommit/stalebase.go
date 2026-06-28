package safecommit

import (
	"context"
	"fmt"
	"os"
	"strings"
)

// ReasonStaleBaseDeletion is the content-level stale-base guard's refusal token. It is
// emitted when committing a requested path by pathspec WOULD silently drop a contiguous
// block a peer already pushed to origin/<trunk>: the working-tree copy of the path predates
// the peer's push (it was never fetched+merged), so the whole-file blob that pathspec commit
// lands deletes the peer's lines even though the local edit never touched them. This is the
// #1073 incident class — a pure-add commit that `git show HEAD --stat` reports with
// deletions. The token is DELIBERATELY distinct from internal/sharedtask's STALE_BASE (an
// unrelated A2A task-record rev conflict over a JSON sha256, no git/origin/file content).
const ReasonStaleBaseDeletion = "STALE_BASE_DELETION"

// staleBaseMinRun is the minimum length (in non-trivial lines) of a contiguous peer-added
// run that must be entirely absent from the working tree before the guard refuses. A small
// threshold suppresses brace-only / blank-line coincidences (a lone `}` that happens to be
// peer-added and locally absent is not a clobbered block) while still catching a real
// dropped block, which is many lines.
const staleBaseMinRun = 2

// staleBaseEnvVar gates the guard, mirroring the FLEET_*_GUARD knob family.
const staleBaseEnvVar = "FAK_STALE_BASE_GUARD"

// staleBaseMode is the parsed value of FAK_STALE_BASE_GUARD.
type staleBaseMode int

const (
	staleBaseBlock staleBaseMode = iota // default: refuse with ReasonStaleBaseDeletion
	staleBaseWarn                       // record the would-be refusal in Detail, still commit
	staleBaseOff                        // skip the guard entirely
)

// staleBaseGuardMode reads FAK_STALE_BASE_GUARD. Default (unset / unrecognized) is block —
// the safe posture on a shared no-amend trunk. `warn` and `off` are documented one-shot
// escapes for a genuinely-intended cross-base deletion.
func staleBaseGuardMode() staleBaseMode {
	switch strings.ToLower(strings.TrimSpace(os.Getenv(staleBaseEnvVar))) {
	case "off", "0", "false":
		return staleBaseOff
	case "warn", "advisory":
		return staleBaseWarn
	default:
		return staleBaseBlock
	}
}

// checkStaleBaseDeletion is the content-level merge-base guard (step 4b). For each requested
// path P it asks: would committing the working-tree copy of P drop a contiguous block a peer
// pushed to the ALREADY-PRESENT-LOCALLY origin/<trunk> ref since this clone's fork point?
//
// It reads ONLY the local origin/<trunk> ref via the injected run — NO network fetch — so it
// closes the fetched-but-not-merged window (the common case on a busy shared clone), not the
// never-fetched window. The boundary is named in the refusal Detail, never hidden.
//
// Returns (detail, fired):
//   - fired == false, detail == "": no peer block would be dropped, OR the guard could not
//     run (fail-open: no origin/<trunk> ref, a fresh base, an unreadable diff). CommitWith
//     proceeds exactly as before.
//   - fired == true, detail != "": a peer-added run would be dropped. The caller refuses
//     (block) or records the detail and proceeds (warn). detail names P, the dropped count,
//     the fork point, and the fetch+merge remedy.
//
// The algorithm is whitespace-insensitive over non-trivial lines so a peer gofmt-reformat
// (which changes only layout) never fires: a reformatted line has a trimmed twin already in
// the working tree, so it is not a genuinely-absent peer-added run.
func checkStaleBaseDeletion(ctx context.Context, run Runner, dir, trunk string, paths []string) (detail string, fired bool) {
	originRef := "origin/" + trunk

	// (1) Resolve the trunk tip. Fail-open if the remote-tracking ref is absent (fresh clone,
	// installer never ran) — the same posture every existing hook takes on such a clone.
	originSHA, code, err := run(ctx, dir, "rev-parse", "--verify", "--quiet", originRef)
	if err != nil || code != 0 || strings.TrimSpace(originSHA) == "" {
		return "", false
	}
	originSHA = strings.TrimSpace(originSHA)

	// (2) Find the fork point. If HEAD already descends from the trunk tip (mb == origin tip),
	// the working tree is fresh relative to origin — nothing to drop. Skip.
	mb, code, err := run(ctx, dir, "merge-base", "HEAD", originRef)
	if err != nil || code != 0 || strings.TrimSpace(mb) == "" {
		return "", false
	}
	mb = strings.TrimSpace(mb)
	if mb == originSHA {
		return "", false
	}

	// (3) Per path, intersect the two diffs:
	//   peerAdded   = lines origin gained since the fork point  (git diff mb origin -- P, '+')
	//   wtRemoved   = lines on origin absent from the WORKING TREE (git diff origin -- P, '-')
	// A dropped peer block is a contiguous run in wtRemoved whose every (non-trivial) line is
	// also peerAdded. Both diffs go through the injected run, so the whole guard is testable
	// with canned evidence and no real repo.
	for _, p := range paths {
		peerDiff, code, err := run(ctx, dir, "diff", mb, originRef, "--", p)
		if err != nil || code != 0 {
			continue // cannot read the peer diff for P — fail-open for this path
		}
		peerAdded := normalizedAddedLines(peerDiff)
		if len(peerAdded) == 0 {
			continue // origin added nothing to P since the fork — nothing to drop
		}

		wtDiff, code, err := run(ctx, dir, "diff", originRef, "--", p)
		if err != nil || code != 0 {
			continue // cannot read the working-tree diff for P — fail-open for this path
		}
		// Lines the working tree still HAS (its side of the origin-vs-wt diff). A peer line
		// re-added here under a different layout (a reformat) is NOT dropped — its trimmed
		// twin is present — so it must not anchor a run.
		wtPresent := normalizedAddedLines(wtDiff)
		dropped := droppedPeerRun(wtDiff, peerAdded, wtPresent, staleBaseMinRun)
		if dropped > 0 {
			short := mb
			if len(short) > 12 {
				short = short[:12]
			}
			return fmt.Sprintf(
				"would drop %d line(s) peer-added to %s on %s since your base %s; "+
					"git fetch origin %s && git merge %s (never --autostash), then re-commit. "+
					"(guard reads the local %s ref only — it closes the fetched-but-not-merged window, not the never-fetched one)",
				dropped, p, originRef, short, trunk, originRef, originRef,
			), true
		}
	}
	return "", false
}

// normalizedAddedLines collects the added ('+', not '+++') lines of a unified diff as a set
// of whitespace-trimmed, non-trivial line contents. Trimming makes the comparison insensitive
// to a peer's reformat; dropping trivial lines (blank / brace- or paren-only) keeps a lone
// structural line from anchoring a false run.
func normalizedAddedLines(diff string) map[string]bool {
	set := map[string]bool{}
	for _, line := range strings.Split(diff, "\n") {
		if !strings.HasPrefix(line, "+") || strings.HasPrefix(line, "+++") {
			continue
		}
		content := strings.TrimSpace(line[1:])
		if isTrivialLine(content) {
			continue
		}
		set[content] = true
	}
	return set
}

// droppedPeerRun walks the removed ('-', not '---') lines of a working-tree-vs-origin diff
// and returns the length of the LONGEST contiguous run (counting only non-trivial lines)
// whose every non-trivial line is peer-added since the fork AND genuinely absent from the
// working tree — i.e. content origin holds, the peer added, and the working tree does NOT
// still carry under any layout. A run shorter than minRun is ignored (brace/blank
// coincidence). Returns 0 when no qualifying run reaches minRun.
//
// wtPresent is the set of trimmed lines the working tree DOES still hold (the '+' side of the
// same origin-vs-wt diff). A removed line whose twin is in wtPresent is a peer REFORMAT, not a
// drop — the logic survives under different whitespace — so it breaks the run rather than
// extends it. Contiguity is also broken by any removed line that is not peer-added (an
// author-intended deletion of pre-fork content) and by leaving the '-' region. Trivial removed
// lines are skipped, so a peer block separated by a blank line still counts as one run.
func droppedPeerRun(diff string, peerAdded, wtPresent map[string]bool, minRun int) int {
	best, cur := 0, 0
	flush := func() {
		if cur > best {
			best = cur
		}
		cur = 0
	}
	for _, line := range strings.Split(diff, "\n") {
		if strings.HasPrefix(line, "---") || !strings.HasPrefix(line, "-") {
			// Left the removed region (context, '+', header) — the run ends.
			flush()
			continue
		}
		content := strings.TrimSpace(line[1:])
		if isTrivialLine(content) {
			continue // a blank/brace-only removed line neither extends nor breaks the run
		}
		if peerAdded[content] && !wtPresent[content] {
			cur++
			continue
		}
		// Either a non-peer line (author-intended pre-fork deletion) or a peer line the
		// working tree re-added under a different layout (a reformat). Both break the run.
		flush()
	}
	flush()
	if best >= minRun {
		return best
	}
	return 0
}

// isTrivialLine reports whether a trimmed line is structural noise that must not anchor a
// run: empty, or made only of braces/parens/brackets/semicolons (a lone `}` etc.).
func isTrivialLine(s string) bool {
	if s == "" {
		return true
	}
	for _, r := range s {
		switch r {
		case '{', '}', '(', ')', '[', ']', ';', ',':
		default:
			return false
		}
	}
	return true
}
