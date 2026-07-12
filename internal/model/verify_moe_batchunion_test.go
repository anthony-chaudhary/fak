package model

import (
	"math"
	"testing"
)

// glmMoePanelModel builds a synthetic GLM-shaped MoE layer (router + E experts, optionally
// NSharedExperts shared experts) with deterministic f32 weights, plus a panel of P distinct
// post-attention-normed hidden vectors. It is the fixture for the batch-union parity proof:
// no attention/embedding tensors are needed because both the reference (glmMoeFFN.apply) and
// the prototype (verifyMoEPanelDelta) drive only the MoE FFN sublayer over Xn directly.
func glmMoePanelModel(t *testing.T, H, MI, E, K, P, nShared int) (*Model, [][]float32) {
	t.Helper()
	cfg := Config{
		HiddenSize:          H,
		NumLayers:           1,
		IntermediateSize:    MI,
		MoEIntermediateSize: MI,
		NumExperts:          E,
		NumExpertsPerTok:    K,
		NormTopKProb:        true,
		NSharedExperts:      nShared,
	}
	// Deterministic LCG so weights and inputs are arbitrary but reproducible.
	seed := uint64(0x9E3779B97F4A7C15)
	nextF := func() float32 {
		seed = seed*6364136223846793005 + 1442695040888963407
		return (float32(seed>>40)/float32(1<<24)*2 - 1) * 0.5
	}
	fill := func(n int) []float32 {
		out := make([]float32, n)
		for i := range out {
			out[i] = nextF()
		}
		return out
	}
	tensors := []NamedTensorF32{
		{Name: routerName(0), Shape: []int{E, H}, Data: fill(E * H)},
	}
	for e := 0; e < E; e++ {
		tensors = append(tensors,
			NamedTensorF32{Name: expertName(0, e, "gate_proj.weight"), Shape: []int{MI, H}, Data: fill(MI * H)},
			NamedTensorF32{Name: expertName(0, e, "up_proj.weight"), Shape: []int{MI, H}, Data: fill(MI * H)},
			NamedTensorF32{Name: expertName(0, e, "down_proj.weight"), Shape: []int{H, MI}, Data: fill(H * MI)},
		)
	}
	if nShared > 0 {
		SI := MI * nShared
		tensors = append(tensors,
			NamedTensorF32{Name: layerName(0, "mlp.shared_experts.gate_proj.weight"), Shape: []int{SI, H}, Data: fill(SI * H)},
			NamedTensorF32{Name: layerName(0, "mlp.shared_experts.up_proj.weight"), Shape: []int{SI, H}, Data: fill(SI * H)},
			NamedTensorF32{Name: layerName(0, "mlp.shared_experts.down_proj.weight"), Shape: []int{H, SI}, Data: fill(H * SI)},
		)
	}
	m, err := NewFromF32Tensors(cfg, tensors)
	if err != nil {
		t.Fatalf("NewFromF32Tensors: %v", err)
	}
	// A panel of distinct hidden vectors so different positions route to different experts.
	Xn := make([][]float32, P)
	for q := 0; q < P; q++ {
		v := make([]float32, H)
		for i := range v {
			v[i] = nextF() + float32(q)*0.13 - 0.26
		}
		Xn[q] = v
	}
	return m, Xn
}

// TestVerifyMoEBatchUnionMatchesPerPosition is the compatibility proof for #4355 (colibri
// batch-union MoE, gated prototype): the panel delta verifyMoEPanelDelta produces by reading
// each UNIQUE routed expert once and applying it to the gathered sub-batch is Float32bits-
// identical to applying glmMoeFFN per panel position — the MoE FFN sublayer of the sequential
// verify fallback. It runs with and without the GLM shared expert. It also asserts the panel
// actually exercises the union (a hot expert routed by >=2 positions and >=2 distinct experts),
// so a degenerate routing cannot make the parity vacuous.
func TestVerifyMoEBatchUnionMatchesPerPosition(t *testing.T) {
	const H, MI, E, K, P = 16, 24, 6, 2, 5

	for _, nShared := range []int{0, 1} {
		nShared := nShared
		name := "noshared"
		if nShared > 0 {
			name = "shared"
		}
		t.Run(name, func(t *testing.T) {
			m, Xn := glmMoePanelModel(t, H, MI, E, K, P, nShared)
			mat := f32Kernel{m}

			// Witness that the batch-union is non-trivial: build the union and per-expert
			// position counts from the same router the op uses.
			counts := map[int]int{}
			for q := 0; q < P; q++ {
				for _, pk := range glmRoute(m, 0, Xn[q], mat) {
					counts[pk.expert]++
				}
			}
			if len(counts) < 2 {
				t.Fatalf("panel union has %d experts, want >=2 to exercise batch-union", len(counts))
			}
			hot := false
			for _, c := range counts {
				if c >= 2 {
					hot = true
				}
			}
			if !hot {
				t.Fatalf("no expert routed by >=2 positions; batch-union gather never has n>1")
			}

			// Reference: per-position glmMoeFFN (the sequential fallback's MoE FFN sublayer).
			want := make([][]float32, P)
			for q := 0; q < P; q++ {
				want[q] = glmMoeFFN{}.apply(m, 0, Xn[q], mat)
			}

			// Prototype: batch-union panel delta.
			got := m.verifyMoEPanelDelta(0, Xn)

			if len(got) != P {
				t.Fatalf("panel delta has %d rows, want %d", len(got), P)
			}
			for q := 0; q < P; q++ {
				if len(got[q]) != H {
					t.Fatalf("pos %d delta len %d, want %d", q, len(got[q]), H)
				}
				for i := 0; i < H; i++ {
					if math.Float32bits(got[q][i]) != math.Float32bits(want[q][i]) {
						t.Fatalf("pos %d delta[%d]: batch-union %v != per-position %v (NOT token-exact)",
							q, i, got[q][i], want[q][i])
					}
				}
			}
		})
	}
}
