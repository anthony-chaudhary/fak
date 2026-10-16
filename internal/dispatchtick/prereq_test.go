package dispatchtick

import (
	"encoding/json"
	"reflect"
	"strings"
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

// fak-test:runtime fast est=1s
func TestRepositoryQualifiedPrerequisiteParsing(t *testing.T) {
	tests := []struct {
		name, body string
		want       DependencyEdges
	}{
		{"qualified", "Start blocked by: Owner/Engine#77, owner/ops#77 AND HTTPS://GitHub.com/owner/engine/issues/77 & #77", DependencyEdges{StartBlockedBy: []string{"owner/engine#77", "owner/ops#77", "77"}}},
		{"dot_repository", "Start blocked by: Owner/.github#77, https://github.com/owner/.github/issues/77", DependencyEdges{StartBlockedBy: []string{"owner/.github#77"}}},
		{"repository_suffix", "Start blocked by: owner/repo-#77 and owner/repo_#78", DependencyEdges{StartBlockedBy: []string{"owner/repo-#77", "owner/repo_#78"}}},
		{"legacy", "Depends on owner/engine#77 and #78.\nAlso blocked by https://github.com/owner/ops/issues/77", DependencyEdges{StartBlockedBy: []string{"owner/engine#77", "78", "owner/ops#77"}}},
		{"relations", "Start blocked by: owner/engine#77\nCoordinates with: owner/ops#77\nPromotion requires: https://github.com/owner/release/issues/77", DependencyEdges{StartBlockedBy: []string{"owner/engine#77"}, CoordinatesWith: []string{"owner/ops#77"}, PromotionRequires: []string{"owner/release#77"}}},
		{"advisory_precedence", "Start blocked by: Owner/Engine#77 and #77\nCoordinates with: owner/engine#77", DependencyEdges{StartBlockedBy: []string{"77"}, CoordinatesWith: []string{"owner/engine#77"}}},
		{"modern_prose", "## Scope / tree\ninternal/gateway/**\nDepends on owner/engine#77 for background.\nCoordinates with: owner/ops#78", DependencyEdges{CoordinatesWith: []string{"owner/ops#78"}}},
		{"explicit_dependency_section", "## Dependencies\nBlocked by: https://github.com/owner/engine/issues/77\n## Witness\nOther prose", DependencyEdges{StartBlockedBy: []string{"owner/engine#77"}}},
		{"malformed_not_local", "Start blocked by: owner//engine#77", DependencyEdges{}},
		{"wrong_url_not_local", "Start blocked by: https://example.test/owner/engine/issues/77", DependencyEdges{}},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := CandidateDependencyEdges(tc.body)
			if !reflect.DeepEqual(got, tc.want) {
				t.Fatalf("edges = %+v, want %+v", got, tc.want)
			}
			wire, err := json.Marshal(got)
			if err != nil {
				t.Fatal(err)
			}
			var round DependencyEdges
			if err := json.Unmarshal(wire, &round); err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(round, got) {
				t.Fatal("wire round trip changed dependency identity")
			}
		})
	}
}

// fak-test:runtime fast est=1s
func TestRouterHoldsUnresolvedStartPrerequisites(t *testing.T) {
	tests := []struct{ name, marker, unresolved string }{
		{"qualified", "Start blocked by: owner/engine#77", "owner/engine#77"},
		{"same_repo_still_unresolved", "Start blocked by: anthony-chaudhary/fak#77", "anthony-chaudhary/fak#77"},
		{"url", "Start blocked by: https://github.com/owner/engine/issues/77", "owner/engine#77"},
		{"dot_repository_url", "Start blocked by: https://github.com/owner/.github/issues/77", "owner/.github#77"},
		{"repository_suffix", "Start blocked by: owner/repo-#77", "owner/repo-#77"},
		{"malformed_owner", "Start blocked by: owner//engine#77", "owner//engine#77"},
		{"suffix", "Start blocked by: owner/engine#77suffix", "owner/engine#77suffix"},
		{"wrong_url", "Start blocked by: https://example.test/owner/engine/issues/77", "https://example.test/owner/engine/issues/77"},
		{"missing_number", "Start blocked by: owner/engine#", "owner/engine#"},
		{"malformed_after_valid", "Start blocked by: #77, owner//engine#78", "owner//engine#78"},
		{"overflow", "Start blocked by: #99999999999999999999999999999999999999", "#99999999999999999999999999999999999999"},
		{"leading_separator", "Start blocked by: , owner/engine#77", "unexpected reference separator"},
		{"trailing_and", "Start blocked by: #77 and", "missing reference after separator"},
		{"empty", "Start blocked by:", "missing reference"},
		{"bare_local", "Start blocked by: #77", ""},
		{"none", "Start blocked by: none", ""},
		{"advisory", "Coordinates with: owner/engine#77", ""},
		{"promotion", "Promotion requires: owner/engine#77", ""},
		{"malformed_advisory", "Coordinates with: owner//engine#77", ""},
		{"unmarked", "Background reference owner/engine#77", ""},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			issue := routerIssue(200, "fix(gateway): bound the request", nil, scopedGatewayIssueBody("2")+"\n"+tc.marker)
			payload := RouteIssues(RouterInput{Injected: true, Issues: []Issue{issue}, Taxonomy: routerTestTaxonomy})
			if tc.unresolved != "" {
				if len(payload.Issues) != 0 || len(payload.SkippedHumanBlocked) != 1 {
					t.Fatalf("unresolved prerequisite remained routable: %+v", payload)
				}
				skipped := payload.SkippedHumanBlocked[0]
				if skipped.Reason != "ISSUE_PREREQUISITE_UNRESOLVED" || !strings.Contains(skipped.NextAction, tc.unresolved) {
					t.Fatalf("refusal lost unresolved identity: %+v", skipped)
				}
				return
			}
			if len(payload.Issues) != 1 || len(payload.SkippedHumanBlocked) != 0 || payload.Issues[0].Lane != "gateway" {
				t.Fatalf("supported/nonblocking control stopped routing: %+v", payload)
			}
			if tc.name == "bare_local" {
				candidates := []dispatchorder.Candidate{{ID: "77"}, {ID: "200", BlockedBy: payload.Issues[0].BlockedBy}}
				if !reflect.DeepEqual(dispatchorder.BlockedByOpenPrereq(candidates)["200"], []string{"77"}) {
					t.Fatal("bare local prerequisite no longer reaches existing hold engine")
				}
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
