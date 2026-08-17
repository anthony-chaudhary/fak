package capindex

import "testing"

func TestQueryProductOutcomesPerformanceFirst(t *testing.T) {
	tests := []struct {
		query string
		want  []string
	}{
		{"token savings", []string{"savings-observability", "context-reuse", "model-routing", "turn-savings", "context-compression", "portable-session"}},
		{"context compression", []string{"context-compression"}},
		{"session replay", []string{"portable-session"}},
		{"turn control", []string{"session-control", "turn-savings"}},
		{"model routing", []string{"model-routing"}},
	}
	for _, tc := range tests {
		t.Run(tc.query, func(t *testing.T) {
			got := QueryProductOutcomes(tc.query, 8)
			seen := map[string]bool{}
			for _, outcome := range got {
				seen[outcome.ID] = true
			}
			for _, want := range tc.want {
				if !seen[want] {
					t.Errorf("query %q missing %q; got %#v", tc.query, want, got)
				}
			}
			if len(got) == 0 || got[0].ID == "capability-floor" {
				t.Errorf("query %q led with supporting security floor: %#v", tc.query, got)
			}
		})
	}
}

func TestProductOutcomesDefaultKeepsSecuritySupporting(t *testing.T) {
	got := ProductOutcomes()
	if len(got) < 2 {
		t.Fatalf("ProductOutcomes() = %d cards, want portfolio", len(got))
	}
	if got[0].ID != "turn-savings" {
		t.Errorf("first default outcome = %q, want turn-savings", got[0].ID)
	}
	if got[len(got)-1].ID != "capability-floor" {
		t.Errorf("last default outcome = %q, want supporting capability-floor", got[len(got)-1].ID)
	}
	for _, outcome := range got {
		if len(outcome.Command) == 0 || outcome.Detail == "" || outcome.Witness == "" {
			t.Errorf("outcome %q lacks executable/discovery evidence: %#v", outcome.ID, outcome)
		}
	}
}

func TestQueryProductOutcomesFindsFleetCommitHealthCheck(t *testing.T) {
	for _, query := range []string{"commit throughput", "commits per 10 minutes", "fleet self blocking", "zero commits"} {
		got := QueryProductOutcomes(query, 3)
		if len(got) == 0 || got[0].ID != "fleet-commit-health" {
			t.Fatalf("query %q got %#v, want fleet-commit-health first", query, got)
		}
		card := got[0]
		if len(card.Command) != 4 || card.Command[0] != "fak" || card.Command[1] != "fleet" || card.Command[2] != "health" || card.Command[3] != "--json" {
			t.Fatalf("query %q command=%v", query, card.Command)
		}
	}
}
