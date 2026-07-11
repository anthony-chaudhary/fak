package dispatchtick

import "testing"

func TestDefaultDispatchLaneScorersPreserveProfileInvariants(t *testing.T) {
	tests := []struct {
		name  string
		high  bool
		cands []DispatchLaneCandidate
		want  string
	}{
		{"throughput core leads richer docs", false, []DispatchLaneCandidate{{"docs", 1000, 100, 100, false}, {"core", 60, 1, 1, true}}, "core"},
		{"high priority urgent docs leads core", true, []DispatchLaneCandidate{{"docs", 1000, 1, 1, false}, {"core", 60, 100, 100, true}}, "docs"},
		{"high priority work leads empty", true, []DispatchLaneCandidate{{"empty", 1000, 100, 0, true}, {"work", 60, 1, 1, false}}, "work"},
		{"lexical exact tie", false, []DispatchLaneCandidate{{"z", 60, 1, 1, false}, {"a", 60, 1, 1, false}}, "a"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := DefaultDispatchLaneScorers(tt.high, 100, 100).Order(tt.cands)
			if len(got) == 0 || got[0].Lane != tt.want {
				t.Fatalf("order=%v want first %q", got, tt.want)
			}
		})
	}
}

func TestDispatchLaneScorerClampsFaultyPlugin(t *testing.T) {
	r := NewDispatchLaneScorerRegistry(
		DispatchLaneScorerFunc{"faulty", 1, func(c DispatchLaneCandidate) float64 {
			if c.Lane == "bad" {
				return 99
			}
			return -99
		}},
		DispatchLaneScorerFunc{"useful", 2, func(c DispatchLaneCandidate) float64 {
			if c.Lane == "good" {
				return 1
			}
			return 0
		}},
	)
	got := r.Order([]DispatchLaneCandidate{{Lane: "bad"}, {Lane: "good"}})
	if got[0].Lane != "good" {
		t.Fatalf("order=%v", got)
	}
}
