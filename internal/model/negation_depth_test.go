package model

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"sort"
	"testing"
)

type negationDepthFixture struct {
	Schema          string              `json:"schema"`
	Surface         string              `json:"surface"`
	DepthCoordinate string              `json:"depth_coordinate"`
	Pairs           []negationDepthPair `json:"pairs"`
	Aggregate       struct {
		MeanDelta   float64 `json:"mean_delta"`
		MedianDelta float64 `json:"median_delta"`
	} `json:"expected_aggregate"`
	Interpretation string `json:"interpretation"`
}

type negationDepthPair struct {
	PairID                   string      `json:"pair_id"`
	AffirmativePrompt        string      `json:"affirmative_prompt"`
	NegatedPrompt            string      `json:"negated_prompt"`
	TargetToken              int         `json:"target_token"`
	AffirmativeLayerLogits   [][]float32 `json:"affirmative_layer_logits"`
	NegatedLayerLogits       [][]float32 `json:"negated_layer_logits"`
	ExpectedAffirmativeDepth int         `json:"expected_affirmative_depth"`
	ExpectedNegatedDepth     int         `json:"expected_negated_depth"`
	ExpectedDelta            int         `json:"expected_delta"`
}

func TestNegationDepthCost(t *testing.T) {
	data, err := os.ReadFile("testdata/negation_depth_cost.json")
	if err != nil {
		t.Fatal(err)
	}
	var fixture negationDepthFixture
	if err := json.Unmarshal(data, &fixture); err != nil {
		t.Fatal(err)
	}
	if fixture.Schema != "fak-negation-depth-cost/1" {
		t.Fatalf("unexpected schema %q", fixture.Schema)
	}
	if len(fixture.Pairs) < 3 {
		t.Fatalf("need a non-degenerate matched fixture, got %d pairs", len(fixture.Pairs))
	}

	deltas := make([]int, 0, len(fixture.Pairs))
	seen := make(map[string]bool, len(fixture.Pairs))
	positive, zero, negative := 0, 0, 0
	for _, pair := range fixture.Pairs {
		if pair.PairID == "" || pair.AffirmativePrompt == "" || pair.NegatedPrompt == "" {
			t.Fatalf("incomplete matched pair: %+v", pair)
		}
		if seen[pair.PairID] {
			t.Fatalf("duplicate pair id %q", pair.PairID)
		}
		seen[pair.PairID] = true
		if len(pair.AffirmativeLayerLogits) != len(pair.NegatedLayerLogits) || len(pair.AffirmativeLayerLogits) < 2 {
			t.Fatalf("%s: unmatched or degenerate layer traces", pair.PairID)
		}

		affirmativeDepth, affirmativeOK := CrystallizationDepth(pair.AffirmativeLayerLogits, pair.TargetToken)
		negatedDepth, negatedOK := CrystallizationDepth(pair.NegatedLayerLogits, pair.TargetToken)
		if !affirmativeOK || !negatedOK {
			t.Fatalf("%s: target does not crystallize (affirmative=%t negated=%t)", pair.PairID, affirmativeOK, negatedOK)
		}
		delta := negatedDepth - affirmativeDepth
		if affirmativeDepth != pair.ExpectedAffirmativeDepth || negatedDepth != pair.ExpectedNegatedDepth || delta != pair.ExpectedDelta {
			t.Fatalf("%s: got depths affirmative=%d negated=%d delta=%+d, fixture wants %d %d %+d",
				pair.PairID, affirmativeDepth, negatedDepth, delta,
				pair.ExpectedAffirmativeDepth, pair.ExpectedNegatedDepth, pair.ExpectedDelta)
		}
		deltas = append(deltas, delta)
		switch {
		case delta > 0:
			positive++
		case delta < 0:
			negative++
		default:
			zero++
		}
		t.Logf("pair=%s affirmative_depth=%d negated_depth=%d delta=%+d", pair.PairID, affirmativeDepth, negatedDepth, delta)
	}

	mean, median := depthDeltaAggregate(deltas)
	if math.Abs(mean-fixture.Aggregate.MeanDelta) > 1e-9 || math.Abs(median-fixture.Aggregate.MedianDelta) > 1e-9 {
		t.Fatalf("aggregate got mean=%.3f median=%.3f, fixture wants mean=%.3f median=%.3f",
			mean, median, fixture.Aggregate.MeanDelta, fixture.Aggregate.MedianDelta)
	}
	// A non-degenerate fixture must preserve counterexamples rather than encode a
	// foregone positive result. The aggregate is reported, not constrained by sign.
	if positive == 0 || zero == 0 || negative == 0 {
		t.Fatalf("fixture hides sign diversity: positive=%d zero=%d negative=%d", positive, zero, negative)
	}
	t.Logf("aggregate pairs=%d mean_delta=%+.3f median_delta=%+.3f positive=%d zero=%d negative=%d",
		len(deltas), mean, median, positive, zero, negative)
	t.Logf("scope=%s interpretation=%s", fixture.Surface, fixture.Interpretation)
}

func TestCrystallizationDepth(t *testing.T) {
	tests := []struct {
		name   string
		logits [][]float32
		target int
		depth  int
		ok     bool
	}{
		{name: "stable after recovery", logits: [][]float32{{2, 1}, {1, 2}, {3, 1}, {1, 4}, {1, 5}}, target: 1, depth: 3, ok: true},
		{name: "stable from embedding", logits: [][]float32{{1, 2}, {1, 3}}, target: 1, depth: 0, ok: true},
		{name: "wrong at final", logits: [][]float32{{1, 2}, {3, 1}}, target: 1, ok: false},
		{name: "missing target", logits: [][]float32{{1, 2}, {1}}, target: 1, ok: false},
		{name: "empty trace", target: 0, ok: false},
		{name: "negative target", logits: [][]float32{{1}}, target: -1, ok: false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			depth, ok := CrystallizationDepth(tt.logits, tt.target)
			if depth != tt.depth || ok != tt.ok {
				t.Fatalf("got (%d,%t), want (%d,%t)", depth, ok, tt.depth, tt.ok)
			}
		})
	}
}

func depthDeltaAggregate(values []int) (mean, median float64) {
	if len(values) == 0 {
		return 0, 0
	}
	ordered := append([]int(nil), values...)
	sort.Ints(ordered)
	var sum int
	for _, value := range ordered {
		sum += value
	}
	mean = float64(sum) / float64(len(ordered))
	middle := len(ordered) / 2
	if len(ordered)%2 == 1 {
		median = float64(ordered[middle])
	} else {
		median = float64(ordered[middle-1]+ordered[middle]) / 2
	}
	return mean, median
}

func ExampleCrystallizationDepth() {
	layers := [][]float32{{2, 1}, {1, 3}, {4, 1}, {1, 5}, {1, 6}}
	depth, ok := CrystallizationDepth(layers, 1)
	fmt.Println(depth, ok)
	// Output: 3 true
}
