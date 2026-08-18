package disambiguation

import (
	"errors"
	"fmt"
)

// ReverseSelfTestCase records one supported locator kind proved through the
// immutable public index.
type ReverseSelfTestCase struct {
	Kind          ReverseLocatorKind `json:"kind"`
	Input         string             `json:"input"`
	CanonicalTerm string             `json:"canonical_term"`
	MatchCount    int                `json:"match_count"`
}

// ReverseSelfTestReport is the captured public-seam witness for reverse lookup.
type ReverseSelfTestReport struct {
	Schema          string                `json:"schema"`
	IndexVersion    string                `json:"index_version"`
	Cases           []ReverseSelfTestCase `json:"cases"`
	UnknownRejected bool                  `json:"unknown_rejected"`
}

// RunReverseSelfTest resolves every supported locator kind and proves unknown
// evidence does not produce a guessed identity.
func RunReverseSelfTest() (ReverseSelfTestReport, error) {
	fixtures := []struct {
		kind  ReverseLocatorKind
		input string
	}{
		{ReverseSourcePath, "internal/disambiguation/query.go"},
		{ReverseGoSymbol, "Query"},
		{ReverseCLIToken, "disambiguation"},
		{ReverseReasonCode, "SOURCE_CURRENT"},
	}
	report := ReverseSelfTestReport{Schema: ReverseLookupSchemaVersion, IndexVersion: PublicIndexVersion, Cases: make([]ReverseSelfTestCase, 0, len(fixtures))}
	for _, fixture := range fixtures {
		result, err := ReverseLookup(fixture.kind, fixture.input)
		if err != nil {
			return ReverseSelfTestReport{}, fmt.Errorf("%s %q: %w", fixture.kind, fixture.input, err)
		}
		if len(result.Matches) == 0 {
			return ReverseSelfTestReport{}, fmt.Errorf("%s %q returned no matches", fixture.kind, fixture.input)
		}
		report.Cases = append(report.Cases, ReverseSelfTestCase{Kind: fixture.kind, Input: fixture.input, CanonicalTerm: result.Matches[0].Entry.Identity.CanonicalTerm, MatchCount: len(result.Matches)})
	}
	unknown, err := ReverseLookup(ReverseGoSymbol, "DefinitelyAbsentReverseLookupSymbol")
	if !errors.Is(err, ErrReverseNotFound) || len(unknown.Matches) != 0 {
		return ReverseSelfTestReport{}, fmt.Errorf("unknown locator produced matches=%d error=%v", len(unknown.Matches), err)
	}
	report.UnknownRejected = true
	return report, nil
}
