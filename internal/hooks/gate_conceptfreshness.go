package hooks

import (
	"fmt"

	"github.com/anthony-chaudhary/fak/internal/conceptcatalog"
)

func checkConceptFreshness(d *StagedDiff) ([]Finding, error) {
	relevant := false
	for _, p := range d.StagedPaths {
		if conceptcatalog.RelevantPath(p) {
			relevant = true
			break
		}
	}
	if !relevant {
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
	return []Finding{{Gate: "CONCEPT_FRESHNESS", File: file, Detail: fmt.Sprintf("generated concept artifacts are stale: %v; run `%s` and stage the result", res.StalePaths, res.Regenerate)}}, nil
}
