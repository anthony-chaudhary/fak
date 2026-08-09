package hooks

import (
	"fmt"

	"github.com/anthony-chaudhary/fak/internal/conceptcatalog"
)

func checkConceptFreshness(d *StagedDiff) ([]Finding, error) {
	// Counted rather than short-circuited: the loop no longer breaks on the first hit, because
	// "one concept path staged" and "forty" are different runs and this gate could not tell them
	// apart (#5602). The cost is one pass over an already-in-memory slice, no extra git read.
	relevant := 0
	for _, p := range d.StagedPaths {
		if conceptcatalog.RelevantPath(p) {
			relevant++
		}
	}
	d.NoteCandidates("CONCEPT_FRESHNESS", relevant, "staged concept-corpus path(s)")
	if relevant == 0 {
		return nil, nil
	}
	res, err := conceptcatalog.CheckGitTree(d.Root, "")
	if err != nil {
		return nil, fmt.Errorf("concept freshness: %w", err)
	}
	if res.Fresh {
		return nil, nil
	}
	// Name the artifact that actually drifted: the scorecard and the reverse name
	// index age independently, and pointing at the README when the INDEX is the stale
	// one sends the reader to a file that is already correct.
	file := conceptcatalog.GeneratedReadme
	if len(res.StalePaths) > 0 {
		file = res.StalePaths[0]
	}
	return []Finding{{Gate: "CONCEPT_FRESHNESS", File: file, Detail: conceptFreshnessDetail(res)}}, nil
}

// conceptFreshnessDetail names the cure that can actually clear this refusal.
//
// The check scored a git tree, so only a regeneration from that same tree can answer it
// (#5829). res.Regenerate carries the tree-scoped command because CheckGitTree produced
// the result; the worktree command is quoted only to say, out loud, that it is the wrong
// one -- it walks the whole checkout, so it folds every peer's unsaved edit into numbers
// this gate never looked at, and running it leaves this finding unchanged. That is the
// same index-versus-worktree sentence CONCEPT_ADMISSION prints on the sibling gate
// (#5534), applied to the half that needed a command rather than a longer sentence.
func conceptFreshnessDetail(res conceptcatalog.FreshnessResult) string {
	return fmt.Sprintf("generated concept artifacts are stale: %v; run `%s` and stage the result -- this gate scores the STAGED GIT TREE (HEAD plus your pathspec), not the worktree, so the worktree regeneration `%s` answers a different tree and reports this finding unchanged", res.StalePaths, res.Regenerate, conceptcatalog.RegenerateCommand)
}
