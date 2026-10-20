package agent

import (
	"math/rand"
	"reflect"
	"testing"
)

// fak-test:runtime fast est=1ms lane=default
func TestDeepInferStableTopPTieOrder(t *testing.T) {
	// top-p reaches 0.7 after the head and one member of the 0.2 tie. Stable
	// descending order must keep the lower token id from that tie.
	probs := []float64{0.5, 0.2, 0.1, 0.2}
	gotSum := nucleusTruncate(probs, 1, 0.7)
	want := []float64{0.5, 0.2, 0, 0}
	if !reflect.DeepEqual(probs, want) {
		t.Fatalf("stable nucleus = %v, want exact kept set %v", probs, want)
	}
	if gotSum != 0.7 {
		t.Fatalf("stable nucleus sum = %v, want 0.7", gotSum)
	}
}

// fak-test:runtime fast est=1ms lane=default
func TestDeepInferTopPPreservesUntiedSelection(t *testing.T) {
	probs := []float64{0.1, 0.6, 0.3}
	gotSum := nucleusTruncate(probs, 1, 0.7)
	want := []float64{0, 0.6, 0.3}
	if !reflect.DeepEqual(probs, want) {
		t.Fatalf("untied nucleus = %v, want exact kept set %v", probs, want)
	}
	wantSum := want[1] + want[2]
	if gotSum != wantSum {
		t.Fatalf("untied nucleus sum = %v, want %v", gotSum, wantSum)
	}
}

// fak-test:runtime fast est=1ms lane=default
func TestDeepInferUntiedSeededCompatibility(t *testing.T) {
	logits := []float32{-2, -0.5, 0.25, 1.75, 3}
	// Literal sequences were recorded from the pre-change sampler at 298469b8.
	for _, tc := range []struct {
		name string
		topP float64
		topK int
		want []int
	}{
		{name: "full", topP: 1, topK: 0, want: []int{4, 4, 3, 3, 4, 4, 2, 4}},
		{name: "topk_topp", topP: 0.95, topK: 3, want: []int{4, 4, 3, 3, 4, 4, 3, 4}},
	} {
		rng := rand.New(rand.NewSource(20261020))
		got := make([]int, len(tc.want))
		for i := range got {
			got[i] = sampleLogits(logits, 0.8, tc.topP, tc.topK, rng)
		}
		if !reflect.DeepEqual(got, tc.want) {
			t.Fatalf("%s seeded draws = %v, want pre-change baseline %v", tc.name, got, tc.want)
		}
	}
}

// fak-test:runtime fast est=1ms lane=default
func TestDeepInferInverseCDFSkipsZeroProbabilityBoundary(t *testing.T) {
	probs := []float64{0, 0.4, 0, 0.6}
	wantUnchanged := append([]float64(nil), probs...)

	if got := sampleProbabilityRow(probs, 1, 0); got != 1 {
		t.Fatalf("coin 0 selected token %d, want first positive-mass token 1", got)
	}
	if got := sampleProbabilityRow(probs, 1, 0.4); got != 3 {
		t.Fatalf("coin on CDF boundary selected token %d, want next positive interval token 3", got)
	}
	if !reflect.DeepEqual(probs, wantUnchanged) {
		t.Fatalf("inverse CDF mutated probabilities: got %v, want %v", probs, wantUnchanged)
	}
}
