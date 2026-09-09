package dispatchtick

import (
	"reflect"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/dispatchorder"
)

// TestCandidateBlockedBy mirrors the Python CandidateBlockedByTest case-for-case
// (tools/issue_resolve_dispatch_test.py:4851) so the Go and Python pickers parse identical edges.
func TestCandidateBlockedBy(t *testing.T) {
	cases := []struct {
		name string
		body string
		want []string
	}{
		// Both marker verbs, hyphenated or spaced, colon optional.
		{"depends-on hyphen colon", "depends-on: #120", []string{"120"}},
		{"depends on spaced", "Depends on #120", []string{"120"}},
		{"blocked-by hyphen colon", "blocked-by: #120", []string{"120"}},
		{"blocked by spaced", "Blocked by #120", []string{"120"}},
		// Comma/and/&-separated refs on one marker, and across markers, deduped in first-seen order.
		{"multi-ref one marker", "blocked-by: #120, #121 and #122", []string{"120", "121", "122"}},
		{"across markers deduped", "Depends on #7.\nAlso blocked by #7 & #9.", []string{"7", "9"}},
		// Prose that merely contains the words never matches (marker must be immediately followed
		// by #N); a marker-free / body-free issue carries no prerequisite edge.
		{"prose no ref", "it depends on the weather", nil},
		{"marker no number", "no markers here #notanumber", nil},
		{"empty body", "", nil},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := CandidateBlockedBy(tc.body)
			if !reflect.DeepEqual(got, tc.want) {
				t.Fatalf("CandidateBlockedBy(%q) = %v, want %v", tc.body, got, tc.want)
			}
		})
	}
}

func TestPrereqDependencies(t *testing.T) {
	t.Run("Start blocked by: #101 blocks pickup", func(t *testing.T) {
		body := "## Scope / tree\ninternal/dispatchtick\n\nStart blocked by: #101\n"
		got := CandidateBlockedBy(body)
		want := []string{"101"}
		if !reflect.DeepEqual(got, want) {
			t.Fatalf("CandidateBlockedBy = %v, want %v", got, want)
		}
	})

	t.Run("Coordinates with: #102 is advisory and does not block pickup", func(t *testing.T) {
		body := "## Scope / tree\ninternal/dispatchtick\n\nCoordinates with: #102\n"
		got := CandidateBlockedBy(body)
		if len(got) != 0 {
			t.Fatalf("CandidateBlockedBy for advisory = %v, want empty", got)
		}
		coords := CandidateCoordinatesWith(body)
		want := []string{"102"}
		if !reflect.DeepEqual(coords, want) {
			t.Fatalf("CandidateCoordinatesWith = %v, want %v", coords, want)
		}
	})

	t.Run("Promotion requires: #103 is promotion gate and does not block pickup", func(t *testing.T) {
		body := "## Scope / tree\ninternal/dispatchtick\n\nPromotion requires: #103\n"
		got := CandidateBlockedBy(body)
		if len(got) != 0 {
			t.Fatalf("CandidateBlockedBy for promotion = %v, want empty", got)
		}
		prom := CandidatePromotionRequires(body)
		want := []string{"103"}
		if !reflect.DeepEqual(prom, want) {
			t.Fatalf("CandidatePromotionRequires = %v, want %v", prom, want)
		}
	})

	t.Run("Ambiguous free-text Depends on #104 in newly authored contract is ignored", func(t *testing.T) {
		body := "## Problem\nThis implementation depends on #104 for background.\n\n## Scope / tree\ninternal/dispatchtick\n"
		got := CandidateBlockedBy(body)
		if len(got) != 0 {
			t.Fatalf("CandidateBlockedBy for free-text depends on = %v, want empty", got)
		}
	})

	t.Run("CandidateDependencyEdges preserves all three edge types", func(t *testing.T) {
		body := `## Problem
We need to separate dependency relations. This depends on #104 for context.

## Dependencies
- Start blocked by: #101
- Coordinates with: #102
- Promotion requires: #103

## Scope / tree
internal/dispatchtick
`
		edges := CandidateDependencyEdges(body)
		if want := []string{"101"}; !reflect.DeepEqual(edges.StartBlockedBy, want) {
			t.Fatalf("edges.StartBlockedBy = %v, want %v", edges.StartBlockedBy, want)
		}
		if want := []string{"102"}; !reflect.DeepEqual(edges.CoordinatesWith, want) {
			t.Fatalf("edges.CoordinatesWith = %v, want %v", edges.CoordinatesWith, want)
		}
		if want := []string{"103"}; !reflect.DeepEqual(edges.PromotionRequires, want) {
			t.Fatalf("edges.PromotionRequires = %v, want %v", edges.PromotionRequires, want)
		}

		blocked := CandidateBlockedBy(body)
		if want := []string{"101"}; !reflect.DeepEqual(blocked, want) {
			t.Fatalf("CandidateBlockedBy = %v, want %v", blocked, want)
		}
	})

	t.Run("Open promotion evidence does not produce BLOCKED_BY_OPEN_PREREQ", func(t *testing.T) {
		body := `## Scope / tree
internal/dispatchtick

Promotion requires: #103
Coordinates with: #102
`
		blockedBy := CandidateBlockedBy(body)
		cands := []dispatchorder.Candidate{
			{ID: "103"}, // open candidate #103
			{ID: "102"}, // open candidate #102
			{ID: "200", BlockedBy: blockedBy},
		}
		held := dispatchorder.BlockedByOpenPrereq(cands)
		if len(held["200"]) != 0 {
			t.Fatalf("open promotion/coordination held candidate 200: %v, want not held", held["200"])
		}
	})
}
