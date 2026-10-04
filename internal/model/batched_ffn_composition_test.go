package model

import (
	"math"
	"strings"
	"testing"
)

// batched_ffn_composition_test.go pins the migration of fuseGatedMLPPanels onto the
// shared ffn.ApplyInPlace fusion (fak#13446). The batched gated-MLP tail used to
// carry its own activation*up loop; it now delegates that loop to the ffn package.
// The migration must be numerically invisible: bit-for-bit identical gate panels,
// read-only up panels, and the same zero-row no-op.

// legacyFuseGatedMLPPanels is the PRE-refactor body, transcribed verbatim from the
// commit before the migration. It is the independent original-arithmetic oracle:
// the production fuseGatedMLPPanels is compared against it with math.Float32bits.
func legacyFuseGatedMLPPanels(m *Model, lp func(string) string, G, U []float32, N, I int, cfg Config) {
	for row := 0; row < N; row++ {
		m.addBiasIfPresent(G[row*I:(row+1)*I], lp("mlp.gate_proj.bias"))
		m.addBiasIfPresent(U[row*I:(row+1)*I], lp("mlp.up_proj.bias"))
	}
	for i := range G {
		G[i] = act(G[i], cfg) * U[i]
	}
}

// batchedFFNFixture builds a one-layer synthetic model. When withBias is set it also
// carries mlp.{gate,up}_proj.bias so the unconditional bias-add path runs.
func batchedFFNFixture(t *testing.T, cfg Config, I int, withBias bool) *Model {
	t.Helper()
	H := cfg.HiddenSize
	p := layerPrefix(0)
	tensors := []synthTensor{
		{"model.embed_tokens.weight", []int{cfg.VocabSize, H}},
		{p + "input_layernorm.weight", []int{H}},
		{p + "self_attn.q_proj.weight", []int{cfg.NumHeads * cfg.HeadDim, H}},
		{p + "self_attn.k_proj.weight", []int{cfg.NumKVHeads * cfg.HeadDim, H}},
		{p + "self_attn.v_proj.weight", []int{cfg.NumKVHeads * cfg.HeadDim, H}},
		{p + "self_attn.o_proj.weight", []int{H, cfg.NumHeads * cfg.HeadDim}},
		{p + "post_attention_layernorm.weight", []int{H}},
		{p + "mlp.gate_proj.weight", []int{I, H}},
		{p + "mlp.up_proj.weight", []int{I, H}},
		{p + "mlp.down_proj.weight", []int{H, I}},
	}
	if withBias {
		tensors = append(tensors,
			synthTensor{p + "mlp.gate_proj.bias", []int{I}},
			synthTensor{p + "mlp.up_proj.bias", []int{I}},
		)
	}
	man, raw := synthBuildRaw(tensors, func(name string, next func() float32) float32 {
		if strings.HasSuffix(name, "layernorm.weight") {
			return 1.0
		}
		return synthMatmulFill(name, next)
	})
	return &Model{Cfg: cfg, manifest: man, raw: raw}
}

// TestBatchedGatedMLPApplyInPlaceParity proves the migrated fuseGatedMLPPanels is
// bit-identical to the pre-refactor loop over a real Model's panels, for both the
// bias-free and the bias-carrying cases, and that the shared path leaves the up panel
// untouched. It exercises the real consumer code path (fuseGatedMLPPanels, called by
// batchedGatedMLP) rather than a bare helper.
func TestBatchedGatedMLPApplyInPlaceParity(t *testing.T) {
	cases := []struct {
		name     string
		withBias bool
		actGelu  bool
		N, I, H  int
	}{
		{name: "bias-free silu", N: 3, I: 8, H: 4},
		{name: "bias-carrying silu", withBias: true, N: 2, I: 6, H: 3},
		{name: "bias-free gelu-tanh", actGelu: true, N: 4, I: 5, H: 2},
		{name: "single row with bias", withBias: true, N: 1, I: 7, H: 5},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			cfg := Config{
				HiddenSize: tc.H, NumLayers: 1, NumHeads: 2, NumKVHeads: 1, HeadDim: 2,
				IntermediateSize: tc.I, VocabSize: 11, RMSNormEps: 1e-5, RopeTheta: 10000,
				TieWordEmbeddings: true, EOSTokenID: -1,
				ActGeluTanh: tc.actGelu,
			}
			m := batchedFFNFixture(t, cfg, tc.I, tc.withBias)
			lp := func(s string) string { return layerName(0, s) }

			// Two identical deterministic panels; one runs production, one the oracle.
			seed := make([]float32, tc.N*tc.I)
			up := make([]float32, tc.N*tc.I)
			for i := range seed {
				seed[i] = float32(i%19-9) / 8
				up[i] = float32(i%23-11) / 16
			}
			prodG := append([]float32(nil), seed...)
			legacyG := append([]float32(nil), seed...)
			prodU := append([]float32(nil), up...)
			legacyU := append([]float32(nil), up...)

			m.fuseGatedMLPPanels(lp, prodG, prodU, tc.N, tc.I, cfg)
			legacyFuseGatedMLPPanels(m, lp, legacyG, legacyU, tc.N, tc.I, cfg)

			for i := range prodG {
				if math.Float32bits(prodG[i]) != math.Float32bits(legacyG[i]) {
					t.Fatalf("gate[%d] bits=%#x legacy=%#x", i, math.Float32bits(prodG[i]), math.Float32bits(legacyG[i]))
				}
			}
			// The bias add intentionally rewrites U in place (pre-existing design), so the
			// migration must reproduce the legacy U bit-for-bit, not restore the seed.
			for i := range prodU {
				if math.Float32bits(prodU[i]) != math.Float32bits(legacyU[i]) {
					t.Fatalf("up[%d] bits=%#x legacy=%#x", i, math.Float32bits(prodU[i]), math.Float32bits(legacyU[i]))
				}
			}
			if !tc.withBias {
				// Bias-free: the panel must be byte-identical to the input the caller passed.
				for i := range up {
					if math.Float32bits(prodU[i]) != math.Float32bits(up[i]) {
						t.Fatalf("bias-free up[%d] mutated: bits=%#x want=%#x", i, math.Float32bits(prodU[i]), math.Float32bits(up[i]))
					}
				}
			}
		})
	}
}

// TestBatchedGatedMLPPanelsZeroRowNoOp preserves the pre-refactor empty-panel
// behavior: a zero-row panel does not panic and does not touch the slices. The
// shared ffn.ApplyInPlace refuses an empty row, so the migration must retain the
// early no-op rather than call into it.
func TestBatchedGatedMLPPanelsZeroRowNoOp(t *testing.T) {
	cfg := Config{
		HiddenSize: 4, NumLayers: 1, NumHeads: 2, NumKVHeads: 1, HeadDim: 2,
		IntermediateSize: 8, VocabSize: 11, RMSNormEps: 1e-5, RopeTheta: 10000,
		TieWordEmbeddings: true, EOSTokenID: -1,
	}
	m := batchedFFNFixture(t, cfg, cfg.IntermediateSize, false)
	lp := func(s string) string { return layerName(0, s) }

	G := []float32{}
	U := []float32{}
	m.fuseGatedMLPPanels(lp, G, U, 0, cfg.IntermediateSize, cfg)
	if len(G) != 0 || len(U) != 0 {
		t.Fatalf("zero-row panel mutated: G=%v U=%v", G, U)
	}
}
