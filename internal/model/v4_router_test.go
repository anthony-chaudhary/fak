package model

import (
	"errors"
	"math"
	"testing"
)

func TestV4ScoredRouteMatchesIndependentOracle(t *testing.T) {
	logits := []float32{-2, -0.5, 0, 1, 3, 0.25, -4, 2}
	bias := []float32{0, 0, 0, -4, 0, 3, 0, 0}
	got, err := v4ScoredRoute(logits, bias, 3, 2.5)
	if err != nil {
		t.Fatal(err)
	}

	// Independent scalar transcription of pinned Gate.forward. It deliberately
	// does not call any production scoring, sorting, or normalization helper.
	score := func(z float32) float64 { return math.Sqrt(math.Log1p(math.Exp(float64(z)))) }
	// Biased selection ranks experts 5, 4, 7. The gathered weights remain
	// unbiased, which is the important noaux_tc contract.
	wantExperts := []int{5, 4, 7}
	denom := score(logits[5]) + score(logits[4]) + score(logits[7])
	for i, expert := range wantExperts {
		if got[i].expert != expert {
			t.Fatalf("pick %d expert=%d, want %d (all=%+v)", i, got[i].expert, expert, got)
		}
		wantWeight := float32(score(logits[expert]) / denom * 2.5)
		if diff := math.Abs(float64(got[i].weight - wantWeight)); diff > 2e-6 {
			t.Fatalf("expert %d weight=%g want=%g diff=%g", expert, got[i].weight, wantWeight, diff)
		}
	}
}

func TestV4ScoredRouteTieAndExtremeFiniteInputs(t *testing.T) {
	got, err := v4ScoredRoute([]float32{1000, -100, 0, 0}, nil, 3, 2.5)
	if err != nil {
		t.Fatal(err)
	}
	want := []int{0, 2, 3} // equal scores select lower expert index first
	for i := range want {
		if got[i].expert != want[i] {
			t.Fatalf("experts=%+v, want order %v", got, want)
		}
		if math.IsNaN(float64(got[i].weight)) || math.IsInf(float64(got[i].weight), 0) {
			t.Fatalf("non-finite weight: %+v", got)
		}
	}
	var sum float32
	for _, pick := range got {
		sum += pick.weight
	}
	if math.Abs(float64(sum-2.5)) > 2e-6 {
		t.Fatalf("scaled normalized sum=%g, want 2.5", sum)
	}
}

func TestV4ScoredRouteFailsClosed(t *testing.T) {
	tests := []struct {
		name   string
		logits []float32
		bias   []float32
		k      int
		scale  float32
	}{
		{"empty", nil, nil, 1, 2.5},
		{"bias width", []float32{1, 2}, []float32{1}, 1, 2.5},
		{"zero topk", []float32{1}, nil, 0, 2.5},
		{"wide topk", []float32{1}, nil, 2, 2.5},
		{"nan logit", []float32{float32(math.NaN())}, nil, 1, 2.5},
		{"inf logit", []float32{float32(math.Inf(1))}, nil, 1, 2.5},
		{"nan bias", []float32{1}, []float32{float32(math.NaN())}, 1, 2.5},
		{"zero scale", []float32{1}, nil, 1, 0},
		{"inf scale", []float32{1}, nil, 1, float32(math.Inf(1))},
		{"zero normalization", []float32{-math.MaxFloat32}, nil, 1, 2.5},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := v4ScoredRoute(tt.logits, tt.bias, tt.k, tt.scale)
			if err == nil {
				t.Fatal("expected fail-closed error")
			}
			var typed *v4RouteError
			if !errors.As(err, &typed) {
				t.Fatalf("error %T is not *v4RouteError: %v", err, err)
			}
		})
	}
}

func TestV4HashRouteIsExplicitlyUnsupported(t *testing.T) {
	err := v4HashRouteUnsupported(2)
	var typed *v4RouteError
	if !errors.As(err, &typed) || typed.Field != "hash_layer" {
		t.Fatalf("got %T %v, want typed hash-layer refusal", err, err)
	}
}
