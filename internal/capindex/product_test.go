package capindex

import (
	"reflect"
	"strings"
	"testing"
)

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

func TestQueryProductOutcomesFindsPerformanceRSIObservability(t *testing.T) {
	for _, query := range []string{"performance observability", "performance loop health", "performance debt"} {
		got := QueryProductOutcomes(query, 3)
		if len(got) == 0 || got[0].ID != "performance-rsi-health" {
			t.Fatalf("query %q got %#v, want performance-rsi-health first", query, got)
		}
	}

	got := QueryProductOutcomes("performance observability", 1)
	if len(got) != 1 {
		t.Fatalf("performance observability result count = %d, want 1", len(got))
	}
	card := got[0]
	wantCommand := []string{"fak", "score", "performance-rsi", "--input", "docs/_witnesses/issue-9768-performance-rsi-dogfood/input.json", "--json"}
	if !reflect.DeepEqual(card.Command, wantCommand) {
		t.Fatalf("performance-rsi command = %#v, want %#v", card.Command, wantCommand)
	}
	if card.Detail != "docs/notes/PERFORMANCE-RSI-DOGFOOD-2026-08-28.md" {
		t.Errorf("performance-rsi detail = %q", card.Detail)
	}
	if card.Witness != "internal/perfrsiscore.ScoreLoopTurn + docs/_witnesses/issue-9777-performance-rsi-loop-turn/loop-turn.txt" {
		t.Errorf("performance-rsi witness = %q", card.Witness)
	}
	wording := strings.ToLower(card.Name + " " + card.Summary)
	for _, want := range []string{"default loop-turn observability", "health", "debt"} {
		if !strings.Contains(wording, want) {
			t.Errorf("performance-rsi wording %q does not name %q", wording, want)
		}
	}
	for _, unsupported := range []string{"performance gain", "faster", "speedup"} {
		if strings.Contains(wording, unsupported) {
			t.Errorf("performance-rsi wording %q claims unsupported %q", wording, unsupported)
		}
	}
}

func TestQueryProductOutcomesFindsNativePerformanceStages(t *testing.T) {
	tests := []struct {
		query   string
		wantID  string
		command []string
	}{
		{"serve native model", "native-serve", []string{"fak", "serve", "--gguf", "<model.gguf>", "--metal"}},
		{"benchmark native inference", "model-benchmark", []string{"fak", "benchmarks", "describe", "modelbench"}},
		{"evaluate model quality", "model-quality", []string{"fak", "quality", "run", "--json"}},
		{"profile native bottleneck", "native-profile", []string{"fak", "native-performance", "--profile-next", "profile.json"}},
		{"performance receipt", "performance-receipt", []string{"fak", "native-performance", "--gate", "gate-request.json"}},
	}
	for _, tc := range tests {
		t.Run(tc.wantID, func(t *testing.T) {
			got := QueryProductOutcomes(tc.query, 1)
			if len(got) != 1 || got[0].ID != tc.wantID {
				t.Fatalf("QueryProductOutcomes(%q, 1) = %#v, want %q first", tc.query, got, tc.wantID)
			}
			if !reflect.DeepEqual(got[0].Command, tc.command) {
				t.Fatalf("%s command = %#v, want %#v", tc.wantID, got[0].Command, tc.command)
			}
		})
	}
}

func TestNativePerformanceStagesDoNotCaptureUnrelatedIntents(t *testing.T) {
	tests := []struct {
		query string
		want  string
	}{
		{"model routing", "model-routing"},
		{"turn control", "turn-savings"},
	}
	for _, tc := range tests {
		got := QueryProductOutcomes(tc.query, 1)
		if len(got) != 1 || got[0].ID != tc.want {
			t.Errorf("QueryProductOutcomes(%q, 1) = %#v, want %q first", tc.query, got, tc.want)
		}
	}
	for _, query := range []string{"handle customer service requests", "update my account profile"} {
		if got := QueryProductOutcomes(query, 1); len(got) != 0 {
			t.Errorf("QueryProductOutcomes(%q, 1) = %#v, want no unrelated capability", query, got)
		}
	}
}
