package agent

import (
	"math"
	"math/rand"
	"testing"
)

func TestInKernelNativeReceiptMeasurementUsesActualLogits(t *testing.T) {
	logits := []float32{-1, 2, 0.5}
	lane := &decodeLane{
		logits: logits,
		rng:    newDeterministicRand(),
		stops:  map[int]bool{},
		temp:   0,
		maxNew: 1,
	}
	var token int
	var score float64
	lane.observe = func(id int, logprob float64) { token, score = id, logprob }
	if _, advance := lane.decodeOne(t.Context()); advance {
		t.Fatal("one-token lane unexpectedly requested another forward")
	}
	want := float64(logits[1]) - math.Log(math.Exp(float64(logits[0]))+math.Exp(float64(logits[1]))+math.Exp(float64(logits[2])))
	if token != 1 || math.Abs(score-want) > 1e-12 || math.IsNaN(score) || math.IsInf(score, 0) {
		t.Fatalf("observation token=%d score=%v, want token=1 score=%v", token, score, want)
	}
}

func newDeterministicRand() *rand.Rand { return rand.New(rand.NewSource(0)) }
