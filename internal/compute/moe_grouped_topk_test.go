package compute

import (
	"errors"
	"math"
	"os"
	"sort"
	"strings"
	"testing"
)

// independentGroupedTopKOracle mirrors the noaux_tc grouped routing reference
// (matching internal/model/glm5next_forward.go) via independent stable sorts.
func independentGroupedTopKOracle(
	logits []float32,
	numTokens, numExperts, numGroups, topkGroup, topK int,
	useSigmoid, renorm bool,
	bias []float32,
	scale float64,
) ([]float32, []int32) {
	weights := make([]float32, numTokens*topK)
	indices := make([]int32, numTokens*topK)
	per := numExperts / numGroups

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
			var s float64
			for e, v := range row {
				scores[e] = math.Exp(float64(v) - mx)
				s += scores[e]
			}
			for e := range scores {
				scores[e] /= s
			}
		}
		choice := make([]float64, numExperts)
		for e := range scores {
			choice[e] = scores[e]
			if len(bias) == numExperts {
				choice[e] += float64(bias[e])
			}
		}

		order := make([]int, numExperts)
		for i := range order {
			order[i] = i
		}
		if numGroups > 1 {
			gs := make([]float64, numGroups)
			for gi := 0; gi < numGroups; gi++ {
				grp := append([]float64(nil), choice[gi*per:(gi+1)*per]...)
				sort.Slice(grp, func(a, b int) bool { return grp[a] > grp[b] })
				if per >= 2 {
					gs[gi] = grp[0] + grp[1]
				} else {
					gs[gi] = grp[0]
				}
			}
			gorder := make([]int, numGroups)
			for i := range gorder {
				gorder[i] = i
			}
			sort.SliceStable(gorder, func(a, b int) bool {
				return gs[gorder[a]] > gs[gorder[b]]
			})
			keep := make([]bool, numExperts)
			for i := 0; i < topkGroup; i++ {
				g := gorder[i]
				for e := g * per; e < (g+1)*per; e++ {
					keep[e] = true
				}
			}
			filtered := order[:0]
			for _, e := range order {
				if keep[e] {
					filtered = append(filtered, e)
				}
			}
			order = filtered
		}
		sort.SliceStable(order, func(a, b int) bool {
			return choice[order[a]] > choice[order[b]]
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
func TestGroupedTopKMatchesOracle(t *testing.T) {
	t.Parallel()

	t.Run("a group top-2 filter with uneven cross-group fill", func(t *testing.T) {
		// 16 experts across 4 groups of 4, topkGroup=2, topK=4.
		// Group 0 (experts 0..3): one massive spike (8.0) + tiny tail (-5.0) => top-2 sum < Group 1 & Group 2.
		// Group 1 (experts 4..7): three strong experts (3.0, 2.9, 2.8, -5.0).
		// Group 2 (experts 8..11): two strong experts (4.0, 2.0, -5.0, -5.0).
		// Group 3 (experts 12..15): weak (-1.0...).
		// Group 1 & Group 2 win the 2 group slots; across them Group 2 fills 1 slot (expert 8)
		// and Group 1 fills 3 slots (experts 4, 5, 6), excluding Group 0's spike (expert 0).
		logits := []float32{
			8.0, -5.0, -5.0, -5.0, // group 0: sigmoid(8.0)~0.9997 + 0.0067 = ~1.0064
			3.0, 2.9, 2.8, -5.0, // group 1: top-2 sigmoid ~0.9526 + 0.9478 = ~1.9004
			4.0, 2.0, -5.0, -5.0, // group 2: top-2 sigmoid ~0.9820 + 0.8808 = ~1.8628
			-1.0, -1.0, -1.0, -1.0, // group 3: weak
		}
		bias := []float32{
			0, 0, 0, 0,
			0.02, 0.01, 0, 0,
			0.03, 0, 0, 0,
			0, 0, 0, 0,
		}
		const scale = 2.5
		got, err := GroupedTopK(logits, 1, GroupedTopKConfig{
			NumExperts:          16,
			TopK:                4,
			NumExpertGroups:     4,
			TopKGroup:           2,
			Scoring:             GroupedTopKSigmoid,
			Renormalize:         true,
			RoutedScalingFactor: scale,
			CorrectionBias:      bias,
		})
		if err != nil {
			t.Fatalf("GroupedTopK: %v", err)
		}
		wantW, wantI := independentGroupedTopKOracle(logits, 1, 16, 4, 2, 4, true, true, bias, scale)
		for k := 0; k < 4; k++ {
			if got.Indices[k] != wantI[k] {
				t.Fatalf("idx[%d]=%d, want %d (all=%v)", k, got.Indices[k], wantI[k], got.Indices)
			}
			if d := math.Abs(float64(got.Weights[k]) - float64(wantW[k])); d > 1e-6 {
				t.Fatalf("weight[%d]=%g, want %g", k, got.Weights[k], wantW[k])
			}
		}
		wantOrder := []int32{8, 4, 5, 6}
		for k, w := range wantOrder {
			if got.Indices[k] != w {
				t.Fatalf("cross-group fill got %v, want %v", got.Indices, wantOrder)
			}
		}
	})

	t.Run("b deterministic tie-break on group score and expert score", func(t *testing.T) {
		// 4 groups of 2 experts. Groups 0 and 2 tie on top-2 score (equal logits 1.0, 1.0),
		// groups 1 and 3 tie at lower score (0.0, 0.0). With topkGroup=2, groups 0 and 2 win
		// cleanly; if topkGroup=1, lower group index 0 beats tied group 2!
		// Inside winning group 0, experts 0 and 1 tie -> index 0 precedes index 1.
		logits := []float32{
			1.0, 1.0, // group 0: sum = 2 * s(1.0)
			0.0, 0.0, // group 1
			1.0, 1.0, // group 2: ties group 0
			0.5, 0.5, // group 3
		}
		got, err := GroupedTopK(logits, 1, GroupedTopKConfig{
			NumExperts:      8,
			TopK:            2,
			NumExpertGroups: 4,
			TopKGroup:       1,
			Scoring:         GroupedTopKSoftmax,
			Renormalize:     true,
		})
		if err != nil {
			t.Fatalf("GroupedTopK(tie): %v", err)
		}
		if got.Indices[0] != 0 || got.Indices[1] != 1 {
			t.Fatalf("tie-break selected %v, want [0 1] from lowest tied group 0", got.Indices)
		}

		// With topkGroup=2 and topK=4, groups 0 and 2 are kept; equal experts sort [0, 1, 4, 5].
		got2, err := GroupedTopK(logits, 1, GroupedTopKConfig{
			NumExperts:      8,
			TopK:            4,
			NumExpertGroups: 4,
			TopKGroup:       2,
			Scoring:         GroupedTopKSoftmax,
			Renormalize:     true,
		})
		if err != nil {
			t.Fatalf("GroupedTopK(tie-2groups): %v", err)
		}
		wantOrder := []int32{0, 1, 4, 5}
		for k, w := range wantOrder {
			if got2.Indices[k] != w {
				t.Fatalf("2-group tie-break got %v, want %v", got2.Indices, wantOrder)
			}
		}
	})

	t.Run("c single-group degenerates to plain top-k", func(t *testing.T) {
		logits := []float32{
			0.3, 2.2, -0.5, 1.8, 2.2, -1.1,
			-0.4, 0.9, 3.1, 0.2, 1.4, 0.9,
		}
		for _, scoring := range []struct {
			g GroupedTopKScoring
			m MoeGatingScoring
		}{
			{GroupedTopKSoftmax, MoeGatingSoftmax},
			{GroupedTopKSigmoid, MoeGatingSigmoid},
		} {
			grouped, err := GroupedTopK(logits, 2, GroupedTopKConfig{
				NumExperts:          6,
				TopK:                3,
				NumExpertGroups:     1,
				TopKGroup:           1,
				Scoring:             scoring.g,
				Renormalize:         true,
				RoutedScalingFactor: 1.5,
			})
			if err != nil {
				t.Fatalf("GroupedTopK(single-group): %v", err)
			}
			plain, err := MoeTopKGating(logits, 2, MoeTopKGatingConfig{
				NumExperts:          6,
				TopK:                3,
				Scoring:             scoring.m,
				Renormalize:         true,
				RoutedScalingFactor: 1.5,
			})
			if err != nil {
				t.Fatalf("MoeTopKGating: %v", err)
			}
			for i := range plain.Indices {
				if grouped.Indices[i] != plain.Indices[i] {
					t.Fatalf("single-group idx[%d]=%d != plain %d", i, grouped.Indices[i], plain.Indices[i])
				}
				if math.Float32bits(grouped.Weights[i]) != math.Float32bits(plain.Weights[i]) {
					t.Fatalf("single-group weight[%d]=%g != plain %g", i, grouped.Weights[i], plain.Weights[i])
				}
			}
		}
	})

	t.Run("d geometry and non-finite refusals", func(t *testing.T) {
		logits := make([]float32, 8)
		if _, err := GroupedTopK(logits, 1, GroupedTopKConfig{
			NumExperts: 8, TopK: 2, NumExpertGroups: 3, TopKGroup: 1,
		}); !errors.Is(err, ErrGroupedTopKGeometry) {
			t.Errorf("non-divisible groups: err=%v, want ErrGroupedTopKGeometry", err)
		}
		if _, err := GroupedTopK(logits, 1, GroupedTopKConfig{
			NumExperts: 8, TopK: 5, NumExpertGroups: 4, TopKGroup: 2, // 2*2=4 < 5
		}); !errors.Is(err, ErrGroupedTopKGeometry) {
			t.Errorf("topK > kept group capacity: err=%v, want ErrGroupedTopKGeometry", err)
		}
		if _, err := GroupedTopK(logits, 1, GroupedTopKConfig{
			NumExperts: 8, TopK: 2, NumExpertGroups: 2, TopKGroup: 3,
		}); !errors.Is(err, ErrGroupedTopKGeometry) {
			t.Errorf("topkGroup > numGroups: err=%v, want ErrGroupedTopKGeometry", err)
		}
		if _, err := GroupedTopK(logits, 1, GroupedTopKConfig{
			NumExperts: 8, TopK: 2, NumExpertGroups: 2, TopKGroup: 1, RoutedScalingFactor: -0.5,
		}); !errors.Is(err, ErrGroupedTopKGeometry) {
			t.Errorf("negative scale: err=%v, want ErrGroupedTopKGeometry", err)
		}
		badLogits := append([]float32(nil), logits...)
		badLogits[3] = float32(math.NaN())
		if _, err := GroupedTopK(badLogits, 1, GroupedTopKConfig{
			NumExperts: 8, TopK: 2, NumExpertGroups: 2, TopKGroup: 1,
		}); !errors.Is(err, ErrGroupedTopKNonFinite) {
			t.Errorf("NaN logit: err=%v, want ErrGroupedTopKNonFinite", err)
		}
	})
}

// fak-test:runtime fast est=1s lane=default
func TestGroupedTopKCUDASourceContract(t *testing.T) {
	t.Parallel()

	cuBytes, err := os.ReadFile("moe_grouped_topk.cu")
	if err != nil {
		t.Fatalf("read moe_grouped_topk.cu: %v", err)
	}
	cu := string(cuBytes)
	for _, want := range []string{
		"UNWIRED:",
		"ae065918640a198a56f6f17eb45e251c3229fe41",
		"Apache-2.0",
		"#include <stdint.h>",
		"#include <stddef.h>",
		"k_mgt_grouped_topk_f32",
		"fcuda_moe_grouped_topk_f32",
		"-INFINITY",
	} {
		if !strings.Contains(cu, want) {
			t.Errorf("moe_grouped_topk.cu missing %q", want)
		}
	}
	if strings.Contains(stripCComments(cu), "-1e30f") {
		t.Error("moe_grouped_topk.cu must not use finite -1e30f reduction sentinel")
	}

	// Verify standalone unwired status: cuda_kernels.cu and cuda_backend.h must not auto-link.
	for _, hostFile := range []string{"cuda_kernels.cu", "cuda_backend.h"} {
		body, err := os.ReadFile(hostFile)
		if err != nil {
			t.Fatalf("read %s: %v", hostFile, err)
		}
		txt := string(body)
		if strings.Contains(txt, "moe_grouped_topk.cu") || strings.Contains(txt, "fcuda_moe_grouped_topk_f32") {
			t.Fatalf("%s must not auto-link unwired moe_grouped_topk.cu", hostFile)
		}
	}
}
