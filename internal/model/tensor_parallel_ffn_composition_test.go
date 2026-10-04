package model

import (
	"math"
	"testing"
)

// tensor_parallel_ffn_composition_test.go pins the migration of the tensor-parallel
// shard FFN's gated-row arithmetic onto the shared ffn.ApplyInPlace component
// (fak#13451). Each rank's per-shard loop used to fuse act(g)*u inline after the
// gate/up bias add; it now delegates that loop to ffn.ApplyInPlace and keeps its
// shard bias slices, down projection and AllReduce order unchanged. The migration
// must be numerically invisible: bit-for-bit identical down-projection partials.
//
// The oracle is the PRE-refactor per-shard body, transcribed verbatim from the
// commit before the migration — an independent original-arithmetic fixture, NOT
// ffn.ApplyInPlace itself (which would be circular).

// legacyTPFFNShardRows is the pre-refactor shard body: gate/up bias add, then the
// inline fused activation*up loop, in increasing index order.
func legacyTPFFNShardRows(g, u, gBias, uBias []float32, lo, w int, cfg Config) {
	if gBias != nil {
		for i := 0; i < w; i++ {
			g[i] += gBias[lo+i]
		}
	}
	if uBias != nil {
		for i := 0; i < w; i++ {
			u[i] += uBias[lo+i]
		}
	}
	for i := 0; i < w; i++ {
		g[i] = act(g[i], cfg) * u[i]
	}
}

// TestTensorParallelFFNApplyInPlaceParity proves the migrated tpFFNLayerPartials is
// bit-identical to the pre-refactor per-shard body over the real feature matrix
// (silu / gelu-tanh, bias-free / bias-carrying) and every shard plan. It exercises
// the real consumer (tpFFNLayerPartials, called by tpFFNLayer and ForwardTP) end to
// end — shard projection, bias slices, gated rows, down projection — rather than a
// bare helper. The comparison is math.Float32bits on the final [H] partial, so any
// change to the activation, the index order, the shard bias offset or the down
// projection surfaces.
func TestTensorParallelFFNApplyInPlaceParity(t *testing.T) {
	const seq = 4
	for _, v := range tpFeatureVariants() {
		t.Run(v.name, func(t *testing.T) {
			m := tpBuildVariant(v)
			cfg := m.Cfg
			H, I := cfg.HiddenSize, cfg.IntermediateSize
			xn := tpRows(seq, H, uint64(len(v.name))*53+7)
			p := func(s string) string { return layerName(0, s) }
			gateW := m.tensor(p("mlp.gate_proj.weight"))
			upW := m.tensor(p("mlp.up_proj.weight"))
			downW := m.tensor(p("mlp.down_proj.weight"))
			gBias := m.tensorOptional(p("mlp.gate_proj.bias"))
			uBias := m.tensorOptional(p("mlp.up_proj.bias"))

			for _, ranks := range []int{1, 2, 3, 4, 6, 8} {
				if ranks > I {
					continue
				}
				plan, err := NewTPPlan(I, ranks)
				if err != nil {
					t.Fatalf("[%s] NewTPPlan(I=%d, ranks=%d): %v", v.name, I, ranks, err)
				}
				parts, err := m.tpFFNLayerPartials(0, xn, plan)
				if err != nil {
					t.Fatalf("[%s] tpFFNLayerPartials ranks=%d: %v", v.name, ranks, err)
				}
				for r, s := range plan.Shards {
					lo, w := s.Lo, s.Width()
					gSlice := shardWeightRows(gateW, H, s.Lo, s.Hi)
					uSlice := shardWeightRows(upW, H, s.Lo, s.Hi)
					downSlice := shardWeightColumns(downW, H, I, s.Lo, s.Hi)
					for pos := 0; pos < seq; pos++ {
						lg := matRows(gSlice, xn[pos], w, H)
						lu := matRows(uSlice, xn[pos], w, H)
						legacyTPFFNShardRows(lg, lu, gBias, uBias, lo, w, cfg)
						want := matRows(downSlice, lg, H, w)
						got := parts[r][pos]
						if len(got) != H {
							t.Fatalf("[%s] ranks=%d rank %d pos %d partial len=%d, want %d", v.name, ranks, r, pos, len(got), H)
						}
						for o := 0; o < H; o++ {
							if math.Float32bits(got[o]) != math.Float32bits(want[o]) {
								t.Fatalf("[%s] ranks=%d rank %d pos %d o=%d: prod=%v bits=%#x legacy=%v bits=%#x (gated-row migration not bit-exact)",
									v.name, ranks, r, pos, o, got[o], math.Float32bits(got[o]), want[o], math.Float32bits(want[o]))
							}
						}
					}
				}
			}
		})
	}
}
