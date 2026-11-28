package compute

import (
	"errors"
	"math"
	"os"
	"sort"
	"strings"
	"testing"
)

// independentGoTopKGatingOracle provides an independent reference implementation
// using stable slice sorting rather than iterative argmax.
func independentGoTopKGatingOracle(logits []float32, numTokens, numExperts, topK int, useSigmoid, renorm bool, bias []float32, scale float64) ([]float32, []int32) {
	weights := make([]float32, numTokens*topK)
	indices := make([]int32, numTokens*topK)
	for t := 0; t < numTokens; t++ {
		row := logits[t*numExperts : (t+1)*numExperts]
		scores := make([]float64, numExperts)
		if useSigmoid {
			for e, v := range row {
				scores[e] = 1.0 / (1.0 + math.Exp(-float64(v)))
			}
		} else {
			mx := math.Inf(-1)
			for _, v := range row {
				if float64(v) > mx {
					mx = float64(v)
				}
			}
			var sum float64
			for e, v := range row {
				scores[e] = math.Exp(float64(v) - mx)
				sum += scores[e]
			}
			for e := range scores {
				scores[e] /= sum
			}
		}

		order := make([]int, numExperts)
		for i := range order {
			order[i] = i
		}
		choice := func(e int) float64 {
			if len(bias) == numExperts {
				return scores[e] + float64(bias[e])
			}
			return scores[e]
		}
		sort.SliceStable(order, func(a, b int) bool {
			ca, cb := choice(order[a]), choice(order[b])
			if ca != cb {
				return ca > cb
			}
			return order[a] < order[b]
		})
		var wsum float64
		for k := 0; k < topK; k++ {
			wsum += scores[order[k]]
		}
		for k := 0; k < topK; k++ {
			id := order[k]
			indices[t*topK+k] = int32(id)
			w := scores[id]
			if renorm && wsum > 0 {
				w /= wsum
			}
			weights[t*topK+k] = float32(w * scale)
		}
	}
	return weights, indices
}

// fak-test:runtime fast est=1s lane=default
func TestMoeTopKGatingMatchesOracle(t *testing.T) {
	t.Parallel()

	t.Run("a softmax renormalized weights and deterministic tie-break", func(t *testing.T) {
		logits := []float32{
			1.0, 3.0, 2.0, 3.0, 0.5, 1.5, // tie between expert 1 and 3 at logit 3.0
			-2.0, 0.0, 4.0, 1.0, 2.5, -1.0,
		}
		const numTokens, numExperts, topK = 2, 6, 3
		for _, renorm := range []bool{true, false} {
			got, err := MoeTopKGating(logits, numTokens, MoeTopKGatingConfig{
				NumExperts:  numExperts,
				TopK:        topK,
				Scoring:     MoeGatingSoftmax,
				Renormalize: renorm,
			})
			if err != nil {
				t.Fatalf("renorm=%v MoeTopKGating: %v", renorm, err)
			}
			wantW, wantI := independentGoTopKGatingOracle(logits, numTokens, numExperts, topK, false, renorm, nil, 1.0)
			for i := range wantI {
				if got.Indices[i] != wantI[i] {
					t.Fatalf("renorm=%v idx[%d]=%d, want %d", renorm, i, got.Indices[i], wantI[i])
				}
				if d := math.Abs(float64(got.Weights[i]) - float64(wantW[i])); d > 1e-6 {
					t.Fatalf("renorm=%v weight[%d]=%g, want %g", renorm, i, got.Weights[i], wantW[i])
				}
			}
			// Token 0 tie: expert 1 must precede expert 3.
			if got.Indices[0] != 1 || got.Indices[1] != 3 || got.Indices[2] != 2 {
				t.Fatalf("token 0 tie-break indices = %v, want [1 3 2]", got.Indices[:3])
			}
			if renorm {
				for tok := 0; tok < numTokens; tok++ {
					var s float64
					for k := 0; k < topK; k++ {
						s += float64(got.Weights[tok*topK+k])
					}
					if math.Abs(s-1.0) > 1e-6 {
						t.Fatalf("token %d renormalized sum=%g, want 1.0", tok, s)
					}
				}
			}
		}
	})

	t.Run("b sigmoid variant with optional correction bias and scaling", func(t *testing.T) {
		logits := []float32{
			0.2, -1.1, 2.3, 0.8,
			-0.5, 1.7, 1.7, -3.0, // tie between expert 1 and 2 broken by bias
		}
		bias := []float32{0.0, 0.05, 0.25, 0.0}
		const numTokens, numExperts, topK = 2, 4, 2
		const scale = 2.5

		got, err := MoeTopKGating(logits, numTokens, MoeTopKGatingConfig{
			NumExperts:          numExperts,
			TopK:                topK,
			Scoring:             MoeGatingSigmoid,
			Renormalize:         true,
			CorrectionBias:      bias,
			RoutedScalingFactor: scale,
		})
		if err != nil {
			t.Fatalf("MoeTopKGating(sigmoid): %v", err)
		}
		wantW, wantI := independentGoTopKGatingOracle(logits, numTokens, numExperts, topK, true, true, bias, scale)
		for i := range wantI {
			if got.Indices[i] != wantI[i] {
				t.Fatalf("idx[%d]=%d, want %d", i, got.Indices[i], wantI[i])
			}
			if d := math.Abs(float64(got.Weights[i]) - float64(wantW[i])); d > 1e-6 {
				t.Fatalf("weight[%d]=%g, want %g", i, got.Weights[i], wantW[i])
			}
		}
		// Token 1 had equal logits on experts 1 and 2; bias[2] > bias[1] selects expert 2 first.
		if got.Indices[2] != 2 || got.Indices[3] != 1 {
			t.Fatalf("token 1 biased indices = %v, want [2 1]", got.Indices[2:4])
		}
	})

	t.Run("c single-expert mass-concentrated row stability", func(t *testing.T) {
		// Huge positive/negative range must not overflow exp or yield NaN.
		logits := []float32{
			-1000.0, 1000.0, -500.0, 0.0,
			-500.0, -500.0, -498.0, -500.0, // all large negative logits
		}
		got, err := MoeTopKGating(logits, 2, MoeTopKGatingConfig{
			NumExperts:  4,
			TopK:        2,
			Scoring:     MoeGatingSoftmax,
			Renormalize: true,
		})
		if err != nil {
			t.Fatalf("MoeTopKGating(extreme): %v", err)
		}
		for i, w := range got.Weights {
			if math.IsNaN(float64(w)) || math.IsInf(float64(w), 0) {
				t.Fatalf("weight[%d]=%v is non-finite", i, w)
			}
		}
		if got.Indices[0] != 1 || math.Abs(float64(got.Weights[0])-1.0) > 1e-6 || got.Weights[1] != 0.0 {
			t.Fatalf("concentrated row 0 = ids=%v weights=%v, want idx0=1 w=[1 0]", got.Indices[:2], got.Weights[:2])
		}
		if got.Indices[2] != 2 {
			t.Fatalf("all-negative row 1 top idx=%d, want 2", got.Indices[2])
		}
	})

	t.Run("d geometry and non-finite refusals", func(t *testing.T) {
		if _, err := MoeTopKGating([]float32{1, 2}, 1, MoeTopKGatingConfig{NumExperts: 2, TopK: 3}); !errors.Is(err, ErrMoeTopKGatingGeometry) {
			t.Errorf("topK > numExperts: err=%v, want ErrMoeTopKGatingGeometry", err)
		}
		if _, err := MoeTopKGating([]float32{1}, 1, MoeTopKGatingConfig{NumExperts: 2, TopK: 1}); !errors.Is(err, ErrMoeTopKGatingGeometry) {
			t.Errorf("short logits: err=%v, want ErrMoeTopKGatingGeometry", err)
		}
		if _, err := MoeTopKGating([]float32{1, 2}, 1, MoeTopKGatingConfig{NumExperts: 2, TopK: 1, RoutedScalingFactor: -1}); !errors.Is(err, ErrMoeTopKGatingGeometry) {
			t.Errorf("negative scale: err=%v, want ErrMoeTopKGatingGeometry", err)
		}
		if _, err := MoeTopKGating([]float32{float32(math.NaN()), 1.0}, 1, MoeTopKGatingConfig{NumExperts: 2, TopK: 1}); !errors.Is(err, ErrMoeTopKGatingNonFinite) {
			t.Errorf("NaN logit: err=%v, want ErrMoeTopKGatingNonFinite", err)
		}
		if _, err := MoeTopKGating([]float32{float32(math.Inf(1)), 1.0}, 1, MoeTopKGatingConfig{NumExperts: 2, TopK: 1}); !errors.Is(err, ErrMoeTopKGatingNonFinite) {
			t.Errorf("Inf logit: err=%v, want ErrMoeTopKGatingNonFinite", err)
		}
	})
}

// fak-test:runtime fast est=1s lane=default
func TestMoeTopKGatingCUDASourceContract(t *testing.T) {
	t.Parallel()

	cuBytes, err := os.ReadFile("moe_topk_gating.cu")
	if err != nil {
		t.Fatalf("read moe_topk_gating.cu: %v", err)
	}
	cu := string(cuBytes)
	for _, want := range []string{
		"UNWIRED:",
		"ae065918640a198a56f6f17eb45e251c3229fe41",
		"Apache-2.0",
		"#include <stdint.h>",
		"#include <stddef.h>",
		"k_mtg_softmax_f32",
		"k_mtg_sigmoid_f32",
		"k_mtg_topk_gating_f32",
		"fcuda_moe_topk_gating_f32",
		"-INFINITY",
	} {
		if !strings.Contains(cu, want) {
			t.Errorf("moe_topk_gating.cu missing %q", want)
		}
	}
	if strings.Contains(stripCComments(cu), "-1e30f") {
		t.Error("moe_topk_gating.cu must not use finite -1e30f reduction sentinel")
	}

	// Verify standalone unwired status: cuda_kernels.cu and cuda_backend.h must not auto-link.
	for _, hostFile := range []string{"cuda_kernels.cu", "cuda_backend.h"} {
		body, err := os.ReadFile(hostFile)
		if err != nil {
			t.Fatalf("read %s: %v", hostFile, err)
		}
		txt := string(body)
		if strings.Contains(txt, "moe_topk_gating.cu") || strings.Contains(txt, "fcuda_moe_topk_gating_f32") {
			t.Fatalf("%s must not auto-link unwired moe_topk_gating.cu", hostFile)
		}
	}
}
