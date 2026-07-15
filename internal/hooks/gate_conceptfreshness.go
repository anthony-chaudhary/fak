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
	return []Finding{{Gate: "CONCEPT_FRESHNESS", File: "docs/concept-disambiguation-scorecard/README.md", Detail: fmt.Sprintf("generated concept artifacts are stale: %v; run `%s` and stage the result", res.StalePaths, res.Regenerate)}}, nil
}
