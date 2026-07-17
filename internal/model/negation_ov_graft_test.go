package model

import (
	"reflect"
	"testing"
)

func negationHeadFixtures() []NegationHeadProbe {
	return []NegationHeadProbe{
		{Layer: 3, Head: 1, NegatorMass: .9, ConceptMass: .9, OVOutput: []float64{1, 0}, ValenceAxis: []float64{1, 0}},
		{Layer: 3, Head: 2, NegatorMass: .8, ConceptMass: .7, OVOutput: []float64{-1, 0}, ValenceAxis: []float64{1, 0}},
		{Layer: 4, Head: 0, NegatorMass: .2, ConceptMass: .9, OVOutput: []float64{1, 0}, ValenceAxis: []float64{1, 0}},
	}
}

func TestNegationHeadAttribution(t *testing.T) {
	ranked, err := AttributeNegationHeads(negationHeadFixtures())
	if err != nil {
		t.Fatal(err)
	}
	if ranked[0].Layer != 3 || ranked[0].Head != 1 || ranked[0].CopyScore != .81 || ranked[0].InvertScore != -1 {
		t.Fatalf("ranking=%+v", ranked)
	}
	for _, row := range ranked {
		t.Logf("head=L%dH%d copy=%.3f invert=%.3f failure=%.3f", row.Layer, row.Head, row.CopyScore, row.InvertScore, row.Failure)
	}
}

func TestOVCircuitGraft(t *testing.T) {
	graft := OVCircuitGraft{Layer: 3, Head: 1, ValenceAxis: []float64{1, 0}}
	output := []float64{.8, .2}
	before := output[0]
	applied, err := graft.Apply(3, 1, true, output)
	if err != nil {
		t.Fatal(err)
	}
	if !applied || output[0] != -.8 || output[1] != .2 {
		t.Fatalf("graft applied=%v output=%v", applied, output)
	}
	t.Logf("head=L3H1 negated_logit_before=%.3f after=%.3f delta=%.3f", before, output[0], output[0]-before)
}

func TestNegationGraftCausal(t *testing.T) {
	graft := OVCircuitGraft{Layer: 3, Head: 1, ValenceAxis: []float64{1, 0}}
	clean := []float64{.8, .2}
	cleanWant := append([]float64(nil), clean...)
	applied, err := graft.Apply(3, 1, false, clean)
	if err != nil || applied || !reflect.DeepEqual(clean, cleanWant) {
		t.Fatalf("clean path applied=%v err=%v got=%v want=%v", applied, err, clean, cleanWant)
	}

	// Capture the causally edited head output, then inject it through the shipped
	// ActivationPatch seam on a fresh residual to prove the effect is portable.
	edited := []float64{.8, .2}
	if _, err := graft.Apply(3, 1, true, edited); err != nil {
		t.Fatal(err)
	}
	patch, err := NewActivationPatch(3)
	if err != nil {
		t.Fatal(err)
	}
	captured := []float32{float32(edited[0]), float32(edited[1])}
	if !patch.Capture(3, captured) {
		t.Fatal("patch did not capture graft output")
	}
	fresh := []float32{.8, .2}
	injected, err := patch.Inject(3, fresh)
	if err != nil || !injected || !reflect.DeepEqual(fresh, []float32{-.8, .2}) {
		t.Fatalf("causal injection=%v err=%v residual=%v", injected, err, fresh)
	}
	t.Logf("causal_patch clean_delta=0 negated_before=0.800 negated_after=%.3f", fresh[0])
}
