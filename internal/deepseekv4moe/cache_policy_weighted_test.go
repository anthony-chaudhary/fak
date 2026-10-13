package deepseekv4moe

import (
	"errors"
	"math"
	"testing"
)

// fak-test:runtime fast est=10ms
func TestWeightedScoreEMAUsesObservedWeights(t *testing.T) {
	groups := []ExpertGroup{{0, 0}, {0, 1}, {0, 2}, {0, 0}}
	policies := []CachePolicy{CachePolicyLRU, CachePolicyScoreEMA, CachePolicyOPT}
	for _, tc := range []struct {
		name   string
		scores []float64
		want   int
	}{
		{"legacy", nil, 0}, {"unit", []float64{1, 1, 1, 1}, 0}, {"weighted", []float64{1, 0.01, 0.01, 1}, 1},
	} {
		t.Run(tc.name, func(t *testing.T) {
			receipt, err := CachePolicyReplayWith(CachePolicyStream{Name: tc.name, Groups: groups, Scores: tc.scores}, 2, policies)
			if err != nil {
				t.Fatal(err)
			}
			if receipt.Rows[0].Hits != 0 || receipt.Rows[1].Hits != tc.want {
				t.Fatalf("LRU hits=%d EMA hits=%d want EMA=%d", receipt.Rows[0].Hits, receipt.Rows[1].Hits, tc.want)
			}
			for _, row := range receipt.Rows {
				if row.Hits+row.Misses != len(groups) || row.Hits > receipt.OPTHits {
					t.Fatalf("invalid receipt counts for %s", row.Policy)
				}
			}
		})
	}
}

// fak-test:runtime fast est=10ms
func TestWeightedScoreEMARejectsMalformedScores(t *testing.T) {
	for _, scores := range [][]float64{{}, {1, 2}, {-1}, {math.NaN()}, {math.Inf(1)}} {
		_, err := CachePolicyReplayWith(CachePolicyStream{Groups: []ExpertGroup{{0, 0}}, Scores: scores}, 1, []CachePolicy{CachePolicyScoreEMA})
		if !errors.Is(err, ErrInvalidTraceScores) {
			t.Fatalf("want invalid scores sentinel; got %v", err)
		}
	}
}

// fak-test:runtime fast est=10ms
func TestWeightedScoresValidateAllReplayEntrypoints(t *testing.T) {
	stream := CachePolicyStream{Groups: []ExpertGroup{{0, 0}}, Scores: []float64{}}
	if _, err := CachePolicyReplay(stream, 1); !errors.Is(err, ErrInvalidTraceScores) {
		t.Fatalf("Replay: %v", err)
	}
	if _, err := CachePolicyReplayOnline(stream, 1); !errors.Is(err, ErrInvalidTraceScores) {
		t.Fatalf("ReplayOnline: %v", err)
	}
}
