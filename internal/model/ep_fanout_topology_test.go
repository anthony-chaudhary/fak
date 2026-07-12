package model

import (
	"errors"
	"testing"
)

// TestValidateEPFanoutCoverage is the covers-style preflight table (#4264): it feeds the
// front-rank validator well-formed and malformed fanout address sets and asserts a nil
// verdict or a typed *EPFanoutTopologyError with the specific reason — the EP analogue of
// TPPlan.Validate's fail-closed table. The malformed rows are exactly the ones the issue
// names (short by one, duplicate, non-dividing/wrong cardinality) plus the empty-binding
// and degenerate-world guards.
func TestValidateEPFanoutCoverage(t *testing.T) {
	cases := []struct {
		name       string
		worldSize  int
		followers  []string
		wantReason EPFanoutCoverReason // "" => expect nil (well-formed)
	}{
		// Well-formed: exactly world-1 distinct followers cover the ranks bijectively.
		{"two-rank world, one follower", 2, []string{"http://a/v1/chat/completions"}, ""},
		{"four-rank world, three followers", 4, []string{"a", "b", "c"}, ""},
		{"eight-rank world, seven followers", 8, []string{"1", "2", "3", "4", "5", "6", "7"}, ""},

		// Short by one: a world of 4 needs 3 followers, only 2 supplied.
		{"short by one", 4, []string{"a", "b"}, EPCoverMiscount},
		// Non-dividing / over-count: the front rank must NOT be in its own fanout set,
		// so world-many addresses over-covers (world 3 needs 2, not 3).
		{"over-count includes front rank", 3, []string{"a", "b", "c"}, EPCoverMiscount},
		// A world of 5 with a lone follower — cardinality that cannot cover 4 ranks.
		{"far short", 5, []string{"a"}, EPCoverMiscount},

		// A rank addressed twice: correct count, but not a bijection.
		{"duplicate at correct count", 3, []string{"a", "a"}, EPCoverDuplicateRank},
		// Duplicate wins over a wrong count too (more specific reason).
		{"duplicate and miscount", 4, []string{"a", "a"}, EPCoverDuplicateRank},

		// An empty binding: a follower rank with no real endpoint.
		{"blank endpoint", 3, []string{"a", "   "}, EPCoverEmptyAddress},
		{"empty string endpoint", 2, []string{""}, EPCoverEmptyAddress},

		// Degenerate world: no follower ranks exist to cover.
		{"world of one", 1, nil, EPCoverDegenerateWorld},
		{"world of zero", 0, nil, EPCoverDegenerateWorld},
		{"world of one with a stray follower", 1, []string{"a"}, EPCoverDegenerateWorld},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			err := ValidateEPFanoutCoverage(tc.worldSize, tc.followers)
			if tc.wantReason == "" {
				if err != nil {
					t.Fatalf("ValidateEPFanoutCoverage(%d, %v) = %v, want nil (well-formed)", tc.worldSize, tc.followers, err)
				}
				return
			}
			if err == nil {
				t.Fatalf("ValidateEPFanoutCoverage(%d, %v) = nil, want refusal %q", tc.worldSize, tc.followers, tc.wantReason)
			}
			var topoErr *EPFanoutTopologyError
			if !errors.As(err, &topoErr) {
				t.Fatalf("ValidateEPFanoutCoverage(%d, %v) = %T, want *EPFanoutTopologyError", tc.worldSize, tc.followers, err)
			}
			if topoErr.Reason != tc.wantReason {
				t.Fatalf("ValidateEPFanoutCoverage(%d, %v) reason = %q, want %q", tc.worldSize, tc.followers, topoErr.Reason, tc.wantReason)
			}
			// The refusal must carry the world/want/got numbers a front-rank log needs.
			if topoErr.WorldSize != tc.worldSize {
				t.Errorf("WorldSize = %d, want %d", topoErr.WorldSize, tc.worldSize)
			}
			if topoErr.Error() == "" {
				t.Errorf("typed error has an empty message")
			}
		})
	}
}

// TestValidateEPFanoutCoverageWantGot pins the reported cardinality numbers (Want=N-1,
// Got=len) so a caller's log line and any retry heuristic can trust them.
func TestValidateEPFanoutCoverageWantGot(t *testing.T) {
	err := ValidateEPFanoutCoverage(4, []string{"a", "b"})
	var topoErr *EPFanoutTopologyError
	if !errors.As(err, &topoErr) {
		t.Fatalf("got %v, want *EPFanoutTopologyError", err)
	}
	if topoErr.Want != 3 {
		t.Errorf("Want = %d, want 3 (world-1)", topoErr.Want)
	}
	if topoErr.Got != 2 {
		t.Errorf("Got = %d, want 2 (len followers)", topoErr.Got)
	}
}
