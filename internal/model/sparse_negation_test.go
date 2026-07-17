package model

import (
	"reflect"
	"testing"
)

func saeFixture() SparseAutoencoder {
	return SparseAutoencoder{
		Encoder:   [][]float64{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}},
		Decoder:   [][]float64{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}},
		Threshold: 0,
	}
}

func TestSAEEncodeDecode(t *testing.T) {
	sae := saeFixture()
	hidden := []float64{.2, .8, .4}
	coeff, err := sae.Encode(hidden)
	if err != nil {
		t.Fatal(err)
	}
	got, err := sae.Decode(coeff)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(got, hidden) {
		t.Fatalf("reconstruction=%v want=%v", got, hidden)
	}
	t.Logf("coefficients=%v reconstruction=%v", coeff, got)
}

func TestNegationFeature(t *testing.T) {
	scores, err := RankNegationFeatures(saeFixture(), []SparseFeaturePair{
		{Affirmative: []float64{.9, .1, .2}, Negated: []float64{.8, 1.1, .2}},
		{Affirmative: []float64{.7, .2, .3}, Negated: []float64{.7, 1.4, .3}},
		{Affirmative: []float64{.8, .1, .4}, Negated: []float64{.8, 1.2, .4}},
	})
	if err != nil {
		t.Fatal(err)
	}
	if scores[0].Feature != 1 || abs(scores[0].Delta-1.1) > 1e-12 {
		t.Fatalf("feature ranking=%+v", scores)
	}
	t.Logf("top_negation_feature=%d mean_activation_delta=%.3f ranking=%+v", scores[0].Feature, scores[0].Delta, scores)
}

func TestSuppressSubstitute(t *testing.T) {
	sae := saeFixture()
	positive := []float64{1, 0, 0}
	once, err := SuppressSubstitute(sae, []float64{.2, 1.3, .4}, 1, positive)
	if err != nil {
		t.Fatal(err)
	}
	twice, err := SuppressSubstitute(sae, once, 1, positive)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(once, []float64{1, 0, .4}) || !reflect.DeepEqual(twice, once) {
		t.Fatalf("once=%v twice=%v", once, twice)
	}
}

func TestSuppressSubstituteLogitLensReadOff(t *testing.T) {
	sae := saeFixture()
	negated := []float64{.2, 1.3, .4}
	positive := []float64{1, 0, 0}
	edited, err := SuppressSubstitute(sae, negated, 1, positive)
	if err != nil {
		t.Fatal(err)
	}
	// A frozen two-token read-off head: token 0 is the unresolved-negation axis;
	// token 1 is the resolved positive-state axis. This is the same linear projection
	// LayerLogits applies after the final norm, kept tiny for deterministic CI.
	read := func(hidden []float64) []float32 {
		return []float32{float32(hidden[1]), float32(hidden[0])}
	}
	before, after := TopK(read(negated), 2), TopK(read(edited), 2)
	if before[0].ID != 0 || after[0].ID != 1 {
		t.Fatalf("logit read-off before=%+v after=%+v", before, after)
	}
	if sparseCosine(edited, positive) <= sparseCosine(negated, positive) {
		t.Fatalf("positive direction did not improve: before=%g after=%g", sparseCosine(negated, positive), sparseCosine(edited, positive))
	}
	t.Logf("top_negation_feature=1 before_top_token=%d before_logit=%.3f after_top_token=%d after_logit=%.3f positive_cosine=%.3f->%.3f", before[0].ID, before[0].Logit, after[0].ID, after[0].Logit, sparseCosine(negated, positive), sparseCosine(edited, positive))
}
