package model

// v41_mhc_flatten_projection_test.go — the #13258 math witness for the published
// flattened-four-stream mHC mix projection (the forward-side rung that follows the
// admission half in v41_mhc_geometry_test.go).
//
// The reference (inference/model.py mHC; transcribed at v4_flash_oracle_test.go:168
// oracleV4FlashMHCProjection) projects the four width-H residual streams laid end
// to end (width 4H) through hc_attn_fn with a SINGLE shared rsqrt over the whole
// flattened vector, then scales the 24 outputs by that rsqrt. The artifact stores
// hc_attn_fn as [4H, 24] (input-major), the transpose of the logical [24, 4H].
// v41MHCProjectFull must reproduce that math for both admitted orientations and
// refuse a residual/weight that does not carry the flattened geometry.

import (
	"math"
	"testing"
)

// v41FlattenStreams lays the four width-H streams end to end, mirroring the
// reference's oracleV4FlashFlattenStreams.
func v41FlattenStreams(streams [][]float32) []float32 {
	out := make([]float32, 0, len(streams[0])*len(streams))
	for _, s := range streams {
		out = append(out, s...)
	}
	return out
}

// v41ScalarMHCProjection is the hand-rolled scalar transcription of
// oracleV4FlashMHCProjection: expected[m] = dot(hcFn[m], xflat) * rsqrt, where
// hcFn is the logical [24, 4H] row-major coefficient block.
func v41ScalarMHCProjection(xflat, hcFn []float32) []float32 {
	flatWidth := len(xflat)
	var ss float32
	for _, v := range xflat {
		ss += v * v
	}
	rsqrt := float32(1 / math.Sqrt(float64(ss/float32(flatWidth)+1e-6)))
	out := make([]float32, v41MHCMixWidth)
	for m := 0; m < v41MHCMixWidth; m++ {
		var s float32
		row := hcFn[m*flatWidth : (m+1)*flatWidth]
		for i := 0; i < flatWidth; i++ {
			s += row[i] * xflat[i]
		}
		out[m] = s * rsqrt
	}
	return out
}

// v41TransposeMixBlock returns the [4H, 24] transpose of a logical [24, 4H]
// row-major block.
func v41TransposeMixBlock(logical []float32, flatWidth int) []float32 {
	stored := make([]float32, flatWidth*v41MHCMixWidth)
	for m := 0; m < v41MHCMixWidth; m++ {
		for i := 0; i < flatWidth; i++ {
			stored[i*v41MHCMixWidth+m] = logical[m*flatWidth+i]
		}
	}
	return stored
}

// TestV41MHCProjectFullBothLayouts pins the projection math for the logical
// [24, 4H] and stored [4H, 24] orientations against a scalar transcription, and
// proves the stored transpose yields the SAME 24 outputs as the logical block.
func TestV41MHCProjectFullBothLayouts(t *testing.T) {
	const H = 5
	streams := [][]float32{
		{0.5, -1.25, 2, 0, 0.75},
		{-0.5, 0.25, 1.5, -2, 0.125},
		{1.0, -0.75, 0.5, 0.25, -0.5},
		{-1.5, 0.5, 0.25, 1.25, -0.125},
	}
	xflat := v41FlattenStreams(streams)
	if len(xflat) != 4*H {
		t.Fatalf("xflat width %d, want %d", len(xflat), 4*H)
	}

	logical := make([]float32, v41MHCMixWidth*4*H)
	for i := range logical {
		logical[i] = float32((i%13)-6) / 7
	}
	want := v41ScalarMHCProjection(xflat, logical)

	gotLogical, err := v41MHCProjectFull(logical, streams, H, 1e-6, false)
	if err != nil {
		t.Fatalf("logical projection error = %v, want nil", err)
	}
	gotStored, err := v41MHCProjectFull(v41TransposeMixBlock(logical, 4*H), streams, H, 1e-6, true)
	if err != nil {
		t.Fatalf("stored projection error = %v, want nil", err)
	}
	for m := 0; m < v41MHCMixWidth; m++ {
		if diff := math.Abs(float64(gotLogical[m] - want[m])); diff > 1e-4 {
			t.Fatalf("logical mixes[%d] = %v, want %v (scalar), diff %g", m, gotLogical[m], want[m], diff)
		}
		if gotStored[m] != gotLogical[m] {
			t.Fatalf("stored mixes[%d] = %v, want the logical %v", m, gotStored[m], gotLogical[m])
		}
	}
}

// TestV41MHCProjectFullRefusesBadGeometry retains the fail-closed contract: a
// residual or weight that does not carry the flattened four-stream geometry must
// refuse rather than silently projecting a wrong sub-matrix.
func TestV41MHCProjectFullRefusesBadGeometry(t *testing.T) {
	const H = 4
	good := [][]float32{{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}, {13, 14, 15, 16}}
	w := make([]float32, v41MHCMixWidth*4*H)

	cases := []struct {
		name string
		w    []float32
		ss   [][]float32
	}{
		{"three streams", w, good[:3]},
		{"wrong width", w, [][]float32{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}, {10, 11, 12}}},
		{"short weight", w[:len(w)-1], good},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if _, err := v41MHCProjectFull(tc.w, tc.ss, H, 1e-6, false); err == nil {
				t.Fatal("malformed flattened mHC geometry admitted, want a typed refusal")
			}
		})
	}
}

// TestV41MHCProjectFullFlattenedForwardFinite drives the real forward on a
// full-geometry (mhcFlat) model whose mhc.mixes.weight is stored [4H, 24]: the
// flattened projection must execute and produce finite logits instead of the
// pre-fix named refusal at the mHC stage.
func TestV41MHCProjectFullFlattenedForwardFinite(t *testing.T) {
	m := v41RawFullFlattenedMHC(t)
	if err := m.v41ForwardAdmitted(); err != nil {
		t.Fatalf("flattened full-geometry model refused at admission: %v", err)
	}
	flat, transposed, ok := m.v41MHCWeightLayout(0)
	if !ok || !flat || !transposed {
		t.Fatalf("mHC layout = (flat=%v, transposed=%v, ok=%v), want the stored [4H,24] transpose", flat, transposed, ok)
	}
	act, err := m.forwardV41([]int{0, 1}, nil)
	if err != nil {
		t.Fatalf("flattened full-geometry forward error = %v, want nil", err)
	}
	if len(act.Logits) != 2 {
		t.Fatalf("forward produced %d logit rows, want 2", len(act.Logits))
	}
	for tRow, row := range act.Logits {
		if len(row) != m.Cfg.VocabSize {
			t.Fatalf("logits[%d] width %d, want %d", tRow, len(row), m.Cfg.VocabSize)
		}
		for i, v := range row {
			if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
				t.Fatalf("logits[%d][%d] = %v, want finite", tRow, i, v)
			}
		}
	}
}

// v41RawFullFlattenedMHC builds a small, raw-backed FULL-geometry V4.1 model
// whose mhc.mixes.weight is the artifact's stored [4H, 24] orientation, so the
// real forward executes v41MHCProjectFull. Everything else is the same synthetic
// [-scale, scale] layout v41ReducedModel uses.
func v41RawFullFlattenedMHC(t *testing.T) *Model {
	t.Helper()
	cfg := v41FullGeometryConfig(t)
	// v41FullGeometryConfig narrows NumHeads to 1 and OGroups to 2 (fine for its
	// admission-only use). A real forward needs the grouped output heads divisible
	// by the groups, so make the tiny fixture self-consistent.
	cfg.OGroups = 1
	H := cfg.HiddenSize
	I := cfg.MoEIntermediateSize
	qHeadDim := cfg.NumHeads * cfg.HeadDim
	oDim := cfg.OLoraRank * cfg.OGroups

	type ts = synthTensor
	tensors := []ts{
		{"model.embed_tokens.weight", []int{cfg.VocabSize, H}},
		{"lm_head.weight", []int{cfg.VocabSize, H}},
		{"model.norm.weight", []int{H}},
		{layerName(0, "attn_norm.weight"), []int{H}},
		{layerName(0, "ffn_norm.weight"), []int{H}},
		{layerName(0, "mhc.mixes.weight"), []int{4 * H, v41MHCMixWidth}},
		{layerName(0, "mhc.base"), []int{v41MHCMixWidth}},
		{layerName(0, "mhc.scale"), []int{3}},
		{layerName(0, "attn.wq_a.weight"), []int{cfg.QLoraRank, H}},
		{layerName(0, "attn.wq_a_norm.weight"), []int{cfg.QLoraRank}},
		{layerName(0, "attn.wq_b.weight"), []int{qHeadDim, cfg.QLoraRank}},
		{layerName(0, "attn.wkv.weight"), []int{v41KVLoraRank, H}},
		{layerName(0, "attn.kv_norm.weight"), []int{v41KVLoraRank}},
		{layerName(0, "attn.wo_a.weight"), []int{cfg.OLoraRank, qHeadDim}},
		{layerName(0, "attn.wo_b.weight"), []int{H, oDim}},
		{layerName(0, "attn.sink"), []int{cfg.NumHeads}},
		{layerName(0, "ffn.gate.weight"), []int{cfg.NumExperts, H}},
		{layerName(0, "ffn.gate.e_score_correction_bias"), []int{cfg.NumExperts}},
		{layerName(0, "ffn.shared_experts.w1.weight"), []int{I, H}},
		{layerName(0, "ffn.shared_experts.w3.weight"), []int{I, H}},
		{layerName(0, "ffn.shared_experts.w2.weight"), []int{H, I}},
	}
	for e := 0; e < cfg.NumExperts; e++ {
		stem := "ffn.experts." + itoa(e)
		tensors = append(tensors,
			ts{layerName(0, stem+".w1.weight"), []int{I, H}},
			ts{layerName(0, stem+".w3.weight"), []int{I, H}},
			ts{layerName(0, stem+".w2.weight"), []int{H, I}},
		)
	}

	man, raw := synthBuildRaw(tensors, func(name string, next func() float32) float32 {
		switch {
		case name == "model.norm.weight" || hasSuffix(name, "attn_norm.weight") || hasSuffix(name, "ffn_norm.weight") || hasSuffix(name, "attn.wq_a_norm.weight") || hasSuffix(name, "attn.kv_norm.weight"):
			return 1.0
		case hasSuffix(name, "mhc.scale"):
			return 1.0
		case hasSuffix(name, "mhc.base"):
			return 0.0
		case hasSuffix(name, "attn.sink"):
			return 0.25 * next()
		default:
			return synthMatmulFill(name, next)
		}
	})
	return &Model{Cfg: cfg, manifest: man, raw: raw}
}
