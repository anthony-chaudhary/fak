package model

// v41_expert_tier_read_test.go — the #13278 witness for the DeepSeek-V4.1
// routed-expert READ from the R5 checkpoint tier.
//
// On the streamed arm a routed expert is by design ABSENT from every resident
// store (manifest/q8w/q4w/q4kw/kqw/q2w/gptqw): the tier exists to fault exactly
// one expert's stride out of a fused checkpoint slab only when it is routed.
// v41AdmitShape already falls through to m.expertCheckpoint.Has, so a tier-only
// routed expert ADMITS; the per-layer READ in v41Layer went through m.tensor,
// which resolves ONLY the f32 manifest and panics on the absent weight -- the
// physical strix3 refusal:
//
//	panic: model: missing tensor model.layers.0.ffn.experts.75.w1.weight
//	  internal/model/manifestTensor (weights.go:549)
//	  internal/model.(*Model).v41Layer (v41_forward.go:1488)
//
// This test builds a reduced V4.1 model whose routed experts live ONLY in a
// checkpoint tier and drives the real forwardV41. On the parent commit the
// m.tensor read panics with that exact message; after the fix the forward faults
// the tier, dequantizes each expert to f32, and emits finite logits.
//
// Deliberately built on pre-existing API only (no v41ExpertF32 / expertWeightF32
// helper), so the same test compiles against the parent commit and fails at
// runtime there -- the red-then-green symptom witness the land gate requires.

import (
	"bytes"
	"math"
	"testing"
)

// v41TierGeometry is the reduced forward's width when its routed experts are
// served from a k-quant checkpoint tier. Every tier representation is a 256-weight
// super-block form, so both the hidden width and the expert intermediate width
// must be multiples of qkK; the shared default fixture (H=64, I=32) cannot carry a
// k-quant slab.
const v41TierWidth = qkK

// v41TierOnlyModel builds a reduced V4.1 Model at H=I=256 whose 384 routed-expert
// w1/w3/w2 projections are carried ONLY by an attached ExpertCheckpointTier. The
// f32 manifest deliberately contains no routed-expert leaves, so the only way the
// forward's read can succeed is through the tier.
func v41TierOnlyModel(t *testing.T, quant ExpertCheckpointQuant) *Model {
	t.Helper()
	cfg := v41TestReducedConfig(t, 1, V41RouterExperts)
	cfg.NumExpertsPerTok = V41RouterTopK
	cfg.NSharedExperts = 1
	cfg.RoutedScalingFactor = 1.5
	cfg.HiddenSize = v41TierWidth
	cfg.MoEIntermediateSize = v41TierWidth
	cfg.RopeScaling = ""
	cfg.LongRope = nil
	cfg.RopeFactor = 0
	cfg.RopeOrigContext = 0
	if cfg.RopeTheta == 0 {
		cfg.RopeTheta = 10000
	}

	H := cfg.HiddenSize
	I := cfg.MoEIntermediateSize
	hd := cfg.HeadDim
	nH := cfg.NumHeads
	qHeadDim := nH * hd
	oDim := cfg.OLoraRank * cfg.OGroups

	type ts = synthTensor
	tensors := []ts{
		{"model.embed_tokens.weight", []int{cfg.VocabSize, H}},
		{"lm_head.weight", []int{cfg.VocabSize, H}},
		{"model.norm.weight", []int{H}},
	}
	for l := 0; l < cfg.NumLayers; l++ {
		tensors = append(tensors,
			ts{layerName(l, "attn_norm.weight"), []int{H}},
			ts{layerName(l, "ffn_norm.weight"), []int{H}},
			ts{layerName(l, "mhc.mixes.weight"), []int{v41MHCMixWidth, H}},
			ts{layerName(l, "mhc.base"), []int{v41MHCMixWidth}},
			ts{layerName(l, "mhc.scale"), []int{3}},
			ts{layerName(l, "attn.wq_a.weight"), []int{cfg.QLoraRank, H}},
			ts{layerName(l, "attn.wq_b.weight"), []int{qHeadDim, cfg.QLoraRank}},
			ts{layerName(l, "attn.wkv.weight"), []int{v41KVLoraRankReduced(cfg), H}},
			ts{layerName(l, "attn.wo_a.weight"), []int{cfg.OLoraRank, qHeadDim}},
			ts{layerName(l, "attn.wo_b.weight"), []int{H, oDim}},
			ts{layerName(l, "attn.sink"), []int{nH}},
			ts{layerName(l, "ffn.gate.weight"), []int{cfg.NumExperts, H}},
			ts{layerName(l, "ffn.gate.e_score_correction_bias"), []int{cfg.NumExperts}},
			ts{layerName(l, "ffn.shared_experts.w1.weight"), []int{I, H}},
			ts{layerName(l, "ffn.shared_experts.w3.weight"), []int{I, H}},
			ts{layerName(l, "ffn.shared_experts.w2.weight"), []int{H, I}},
		)
		// NOTE: no ffn.experts.<e>.* tensors -- the routed experts are tier-only.
	}

	man, raw := synthBuildRaw(tensors, func(name string, next func() float32) float32 {
		switch {
		case name == "model.norm.weight" || hasSuffix(name, "attn_norm.weight") || hasSuffix(name, "ffn_norm.weight"):
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
	m := &Model{Cfg: cfg, manifest: man, raw: raw}

	// Build one fused [Experts, Rows, Cols] slab per routed projection, exactly
	// as the loader would index the published deepseek41 artifact, and attach it
	// as the sole source of the routed experts.
	blockWeights, blockBytes, ok := quant.blockGeometry()
	if !ok {
		t.Fatalf("fixture quant %s reports no block geometry", quant)
	}
	projs := []struct {
		proj string
		rows int
		cols int
	}{
		{"gate_proj", I, H},
		{"up_proj", I, H},
		{"down_proj", H, I},
	}
	tier := NewExpertCheckpointTier(0)
	for _, p := range projs {
		if p.cols%blockWeights != 0 {
			t.Fatalf("fixture projection %s cols %d is not a multiple of %d", p.proj, p.cols, blockWeights)
		}
		perExpert := p.rows * (p.cols / blockWeights) * blockBytes
		blob := make([]byte, perExpert*V41RouterExperts)
		// Fill each expert's stride with VALID super-blocks so the tier fault
		// dequantizes to finite f32 values (an arbitrary byte pattern would
		// produce inf/nan scales and mask the read under test).
		for e := 0; e < V41RouterExperts; e++ {
			expertBytes := buildRawQ4K(t, p.rows, p.cols, 7000+e)
			if len(expertBytes) != perExpert {
				t.Fatalf("%s expert stride = %d bytes, want %d", p.proj, len(expertBytes), perExpert)
			}
			copy(blob[e*perExpert:], expertBytes)
		}
		if err := tier.AddShard(bytes.NewReader(blob), int64(len(blob)), []FusedExpertTensor{{
			Name: "blk.0.ffn_" + p.proj + "_exps.weight", Layer: 0, Proj: p.proj,
			Arch: "deepseek41", Quant: quant, Offset: 0,
			Experts: V41RouterExperts, Rows: p.rows, Cols: p.cols,
		}}); err != nil {
			t.Fatalf("AddShard over the %s slab: %v", p.proj, err)
		}
	}
	m.expertCheckpoint = tier
	return m
}

// TestV41ExpertTierRead is the #13278 acceptance witness: a model whose routed
// experts live ONLY in the R5 checkpoint tier runs the real reduced forward and
// emits finite logits. On the parent commit the routed-expert read panics
// "model: missing tensor model.layers.0.ffn.experts.<e>.w1.weight".
func TestV41ExpertTierRead(t *testing.T) {
	m := v41TierOnlyModel(t, ExpertCheckpointQ4K)

	// The routed-expert leaves must be absent from every resident store, or the
	// tier read is not exercised.
	for e := 0; e < V41RouterExperts; e++ {
		for _, leaf := range []string{"w1", "w3", "w2"} {
			name := layerName(0, "ffn.experts."+itoa(e)+"."+leaf+".weight")
			if _, ok := m.residentF32Mat(name); ok {
				t.Fatalf("fixture still resolves %s from a resident store; the tier read is not exercised", name)
			}
			if !m.expertCheckpoint.Has(name) {
				t.Fatalf("fixture tier does not index %s; the read cannot resolve", name)
			}
		}
	}

	// Admission is residency-complete and must pass (it consults the tier index).
	if err := m.v41ForwardAdmitted(); err != nil {
		t.Fatalf("tier-only admission error = %v, want nil", err)
	}

	// The real forward must run instead of panicking in m.tensor.
	act, err := m.forwardV41([]int{1, 3}, nil)
	if err != nil {
		t.Fatalf("tier-only routed-expert forward error = %v, want nil", err)
	}
	if len(act.Logits) != 2 {
		t.Fatalf("forward produced %d logit rows, want 2", len(act.Logits))
	}
	for r, row := range act.Logits {
		for i, v := range row {
			if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
				t.Fatalf("logits[%d][%d] = %v, want finite", r, i, v)
			}
		}
	}
}

// TestV41ExpertTierReadFailsClosed pins the retained fail-closed property: a
// routed expert present in NEITHER a resident store NOR the tier still refuses by
// name rather than panicking or silently producing zeros.
func TestV41ExpertTierReadFailsClosed(t *testing.T) {
	m := v41TierOnlyModel(t, ExpertCheckpointQ4K)
	// Detach the tier: the routed experts are now reachable nowhere.
	m.expertCheckpoint = nil

	if err := m.v41ForwardAdmitted(); err == nil {
		t.Fatal("admission admitted a tier-only model after the tier was detached; want a named refusal")
	}
}
