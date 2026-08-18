package disambiguation

import "fmt"

const CacheSourceSelfTestSchemaVersion = "fak-disambiguation-cache-source-self-test/1"

// CacheSourceResolution captures the four axes that distinguish one cache.
type CacheSourceResolution struct {
	Input         string `json:"input"`
	CanonicalTerm string `json:"canonical_term"`
	OwnerLeaf     string `json:"owner_leaf"`
	Scope         string `json:"scope"`
	SourcePath    string `json:"source_path"`
	ContrastCount int    `json:"contrast_count"`
}

// CacheSourceSelfTestReport proves all four cache concepts resolve through the
// public index and carry complete pairwise distinctions.
type CacheSourceSelfTestReport struct {
	Schema       string                  `json:"schema"`
	IndexVersion string                  `json:"index_version"`
	Resolutions  []CacheSourceResolution `json:"resolutions"`
	Pairwise     bool                    `json:"pairwise_contrasts_complete"`
}

// RunCacheSourceSelfTest resolves four overloaded cache names and verifies each
// record contrasts with every sibling using public provenance.
func RunCacheSourceSelfTest() (CacheSourceSelfTestReport, error) {
	terms := []string{"vDSO cache", "KV cache", "radix cache", "provider cache"}
	report := CacheSourceSelfTestReport{Schema: CacheSourceSelfTestSchemaVersion, IndexVersion: PublicIndexVersion, Resolutions: make([]CacheSourceResolution, 0, len(terms))}
	canonical := make([]string, 0, len(terms))
	for _, term := range terms {
		resolved, err := Resolve(term)
		if err != nil {
			return CacheSourceSelfTestReport{}, fmt.Errorf("resolve %q: %w", term, err)
		}
		entry := resolved.Entry
		if len(entry.Sources) == 0 {
			return CacheSourceSelfTestReport{}, fmt.Errorf("resolve %q returned no public source", term)
		}
		canonical = append(canonical, entry.Identity.CanonicalTerm)
		report.Resolutions = append(report.Resolutions, CacheSourceResolution{Input: term, CanonicalTerm: entry.Identity.CanonicalTerm, OwnerLeaf: entry.Owner.Leaf, Scope: entry.Scope.Value, SourcePath: entry.Sources[0].Locator, ContrastCount: len(entry.Contrasts)})
	}
	for _, leftTerm := range canonical {
		left, err := Query(leftTerm)
		if err != nil {
			return CacheSourceSelfTestReport{}, err
		}
		for _, rightTerm := range canonical {
			if leftTerm == rightTerm {
				continue
			}
			contrast, ok := contrastTo(left.Entry, rightTerm)
			if !ok || contrast.ForbiddenConflation == nil || !*contrast.ForbiddenConflation || contrast.Explanation == "" {
				return CacheSourceSelfTestReport{}, fmt.Errorf("%q lacks forbidden contrast with %q", leftTerm, rightTerm)
			}
		}
	}
	report.Pairwise = true
	return report, nil
}
