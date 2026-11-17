package model

// kimik31_router_test.go — parity + contract tests for the K3.1 native MoE router
// (ticket 1611-kimi-k31, port leaf sglang-kimi-k31-4 host side; goal Wave 1.4/1.5).
// The router contract is the noaux_tc one: selection ranks by sigmoid(logit)+bias,
// weights are the unbiased sigmoid(logit), renormalized and scaled. These tests pin
// the Go oracle against an independent float64 transcription and prove the two
// load-bearing properties: the correction bias CHANGES selection while leaving the
// weights unbiased, and the renormalization sums to RouteScale.
//
// fak-test:runtime fast est=5ms lane=default

import (
	"math"
	"testing"
)

func sigmoid64(z float64) float64 { return 1.0 / (1.0 + math.Exp(-z)) }

// TestKimiK31RouterBiasChangesSelectionNotWeights is the load-bearing witness: two
// experts have equal sigmoid scores, but the correction bias breaks the tie in
// SELECTION while the returned weight stays the unbiased score.
func TestKimiK31RouterBiasChangesSelectionNotWeights(t *testing.T) {
	cfg := KimiK31RouterConfig{Experts: 4, TopK: 2, RouteScale: 1.0, Renormalize: true, GroupCount: 1}
	logits := []float32{0, 0, 0, 0} // all sigmoid == 0.5, a 4-way tie
	bias := []float32{-1, 5, 0, 0}  // expert 1 is pushed to the top, expert 0 to the bottom

	picks, err := RouteKimiK31MoE(logits, bias, cfg)
	if err != nil {
		t.Fatalf("RouteKimiK31MoE: %v", err)
	}
	if len(picks) != 2 {
		t.Fatalf("picks = %d, want 2", len(picks))
	}
	// Selection ranks by score+bias: expert 1 (5.5) then one of the 0.5s by lowest id.
	if picks[0].Expert != 1 {
		t.Fatalf("top pick = expert %d, want 1 (bias 5 wins)", picks[0].Expert)
	}
	if picks[1].Expert != 2 {
		t.Fatalf("second pick = expert %d, want 2 (bias 0 ties with expert 3, lowest id wins; expert 0 is pushed to -0.5)", picks[1].Expert)
	}
	// Weights are UNBIASED sigmoid(logit) renormalized over the top-k: both are
	// 0.5, so each is 1/2 = 0.5 (the bias never enters the weight).
	for _, p := range picks {
		if math.Abs(float64(p.Weight)-0.5) > 1e-6 {
			t.Fatalf("expert %d weight = %v, want the unbiased 0.5 (bias must not enter the weight)", p.Expert, p.Weight)
		}
	}
}

// TestKimiK31RouterMatchesReference pins selection order, weights and scaling
// against an independent float64 transcription.
func TestKimiK31RouterMatchesReference(t *testing.T) {
	cfg := KimiK31RouterConfig{Experts: 6, TopK: 3, RouteScale: 2.5, Renormalize: true, GroupCount: 1}
	logits := []float32{0.3, -1.2, 2.0, 0.3, 4.1, -0.5}
	bias := []float32{0.1, 0.0, -0.2, 0.9, 0.0, 0.0}

	// Reference: rank by sigmoid(logit)+bias (ties by lower id); weights from
	// unbiased sigmoid(logit), renormalized over the top-k, then * RouteScale.
	type ref struct {
		e     int
		sel   float64
		score float64
	}
	refs := make([]ref, len(logits))
	for i, z := range logits {
		s := sigmoid64(float64(z))
		refs[i] = ref{e: i, sel: s + float64(bias[i]), score: s}
	}
	// stable sort by sel desc, id asc.
	for i := 1; i < len(refs); i++ {
		for j := i; j > 0 && (refs[j].sel > refs[j-1].sel ||
			(refs[j].sel == refs[j-1].sel && refs[j].e < refs[j-1].e)); j-- {
			refs[j], refs[j-1] = refs[j-1], refs[j]
		}
	}
	var sum float64
	for i := 0; i < cfg.TopK; i++ {
		sum += refs[i].score
	}

	picks, err := RouteKimiK31MoE(logits, bias, cfg)
	if err != nil {
		t.Fatalf("RouteKimiK31MoE: %v", err)
	}
	for i := 0; i < cfg.TopK; i++ {
		if picks[i].Expert != refs[i].e {
			t.Errorf("pick %d expert = %d, want %d", i, picks[i].Expert, refs[i].e)
		}
		want := refs[i].score / sum * float64(cfg.RouteScale)
		if math.Abs(float64(picks[i].Weight)-want) > 1e-5 {
			t.Errorf("pick %d weight = %v, want %v", i, picks[i].Weight, want)
		}
	}
}

// TestKimiK31RouterRenormalizeSumsToScale pins the renormalization invariant.
func TestKimiK31RouterRenormalizeSumsToScale(t *testing.T) {
	cfg := KimiK31RouterConfig{Experts: 8, TopK: 4, RouteScale: 1.0, Renormalize: true, GroupCount: 1}
	logits := []float32{1, 2, 3, 4, 0.5, -1, 2.5, 0}
	picks, err := RouteKimiK31MoE(logits, nil, cfg)
	if err != nil {
		t.Fatalf("RouteKimiK31MoE: %v", err)
	}
	var sum float32
	for _, p := range picks {
		sum += p.Weight
	}
	if math.Abs(float64(sum)-1.0) > 1e-5 {
		t.Fatalf("renormalized weight sum = %v, want 1.0", sum)
	}
}

// TestKimiK31RouterNoRenormalizeSkipsDivision pins the renormalize=false branch:
// weights are the raw unbiased sigmoid scores scaled by RouteScale.
func TestKimiK31RouterNoRenormalizeSkipsDivision(t *testing.T) {
	cfg := KimiK31RouterConfig{Experts: 3, TopK: 2, RouteScale: 1.0, Renormalize: false, GroupCount: 1}
	logits := []float32{0, 0, 0}
	picks, err := RouteKimiK31MoE(logits, nil, cfg)
	if err != nil {
		t.Fatalf("RouteKimiK31MoE: %v", err)
	}
	for _, p := range picks {
		if math.Abs(float64(p.Weight)-0.5) > 1e-6 {
			t.Fatalf("no-renorm weight = %v, want the raw sigmoid 0.5", p.Weight)
		}
	}
}

// TestKimiK31RouterRefusesBadContract proves the router fails closed on shape,
// domain and unimplemented-grouped inputs.
func TestKimiK31RouterRefusesBadContract(t *testing.T) {
	base := KimiK31RouterConfig{Experts: 4, TopK: 2, RouteScale: 1.0, Renormalize: true, GroupCount: 1}
	cases := []struct {
		name   string
		logits []float32
		bias   []float32
		cfg    KimiK31RouterConfig
	}{
		{"logits width", []float32{1, 2}, nil, base},
		{"bias width", []float32{1, 2, 3, 4}, []float32{1, 2}, base},
		{"non-finite logit", []float32{1, float32(math.Inf(1)), 3, 4}, nil, base},
		{"non-finite bias", []float32{1, 2, 3, 4}, []float32{1, float32(math.NaN()), 0, 0}, base},
		{"topk exceeds experts", []float32{1, 2, 3, 4}, nil, KimiK31RouterConfig{Experts: 4, TopK: 9, RouteScale: 1, GroupCount: 1}},
		{"grouped unimplemented", []float32{1, 2, 3, 4}, nil, KimiK31RouterConfig{Experts: 4, TopK: 2, RouteScale: 1, GroupCount: 2}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if _, err := RouteKimiK31MoE(tc.logits, tc.bias, tc.cfg); err == nil {
				t.Fatal("expected a refusal")
			}
		})
	}
}
