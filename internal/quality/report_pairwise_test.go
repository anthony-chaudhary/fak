package quality

import (
	"reflect"
	"testing"
)

func TestEvaluatePairwiseRandomizedOrderAndDeterministicAggregation(t *testing.T) {
	spec := PairwiseSpec{Dimensions: []PairwiseDimension{{Name: "grounding", Weight: 1, Criteria: []string{"source"}}, {Name: "actionability", Weight: 1, Criteria: []string{"owner", "deadline"}}}, TieMargin: .01}
	base := "source says risk"
	cand := "source says risk; owner alice; deadline friday"
	a := EvaluatePairwise(base, cand, spec, 7)
	b := EvaluatePairwise(base, cand, spec, 7)
	if !reflect.DeepEqual(a, b) {
		t.Fatalf("same seed must replay identically: %#v %#v", a, b)
	}
	if a.Winner != "candidate" || len(a.Dimensions) != 2 {
		t.Fatalf("candidate evidence should win with reasons: %+v", a)
	}
	foundSwap := false
	for seed := int64(8); seed < 64; seed++ {
		if EvaluatePairwise(base, cand, spec, seed).Order != a.Order {
			foundSwap = true
			break
		}
	}
	if !foundSwap {
		t.Fatal("seeded evaluator never randomized presentation order")
	}
}

func TestEvaluatePairwiseTieAndMissingEvidenceFailClosed(t *testing.T) {
	spec := PairwiseSpec{Dimensions: []PairwiseDimension{{Name: "grounding", Weight: 1, Criteria: []string{"source"}}}, TieMargin: .1}
	tie := EvaluatePairwise("source x", "source y", spec, 1)
	if tie.Winner != "tie" {
		t.Fatalf("expected declared tie: %+v", tie)
	}
	missing := EvaluatePairwise("", "source y", spec, 1)
	if missing.Winner != "inconclusive" {
		t.Fatalf("missing baseline must be inconclusive and localized: %+v", missing)
	}
}
