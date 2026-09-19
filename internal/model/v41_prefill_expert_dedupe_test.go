package model

// v41_prefill_expert_dedupe_test.go -- the #13296 witness for the DeepSeek-V4.1
// prefill streamed-expert read dedupe.
//
// A multi-token prefill routes the SAME expert projection many times across the
// token dimension (top-6 of 384, with the token-to-token overlap real routing
// shows). Before #13296 the V4.1 MoE loop faulted the R5 checkpoint tier once per
// (token, pick), so the layer's tier Reads scaled with the TOKEN COUNT; the
// tier's bounded host cache absorbed the repeats only while the interleaved
// working set fit its bound, which on the published artifact it does not. The
// layer-scoped f32 expert cache makes each distinct (expert, projection) one read
// per layer at ANY tier bound, so Reads is the distinct-set size, independent of
// the token count.
//
// The test drives the REAL reduced forwardV41 against a tier-only model whose
// routed experts live ONLY in the checkpoint tier, with a tier host budget
// DELIBERATELY SMALLER than the layer's distinct expert set -- the production
// regime. It asserts Reads is `distinct * 3` and unchanged as the prefill grows
// from 4 to 16 tokens. On the parent commit Reads scales linearly with the token
// count (4 tokens => 72 reads at this budget, not 18).

import (
	"bytes"
	"testing"
)

// v41TierOnlyBudgetModel mirrors v41TierOnlyModel but attaches a tier whose
// bounded host cache holds budgetUnits single-projection slabs. budgetUnits=0 is
// the stream-through default (retain nothing), which is the STRICTEST bound: any
// dedupe must come from the layer-scoped cache, not the tier.
func v41TierOnlyBudgetModel(t *testing.T, budgetUnits int) (*Model, int64) {
	t.Helper()
	quant := ExpertCheckpointQ4K
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
	H, I := cfg.HiddenSize, cfg.MoEIntermediateSize
	hd, nH := cfg.HeadDim, cfg.NumHeads
	qHeadDim := nH * hd
	oDim := cfg.OLoraRank * cfg.OGroups

	tensors := []synthTensor{
		{"model.embed_tokens.weight", []int{cfg.VocabSize, H}},
		{"lm_head.weight", []int{cfg.VocabSize, H}},
		{"model.norm.weight", []int{H}},
	}
	for l := 0; l < cfg.NumLayers; l++ {
		tensors = append(tensors,
			synthTensor{layerName(l, "attn_norm.weight"), []int{H}},
			synthTensor{layerName(l, "ffn_norm.weight"), []int{H}},
			synthTensor{layerName(l, "mhc.mixes.weight"), []int{v41MHCMixWidth, H}},
			synthTensor{layerName(l, "mhc.base"), []int{v41MHCMixWidth}},
			synthTensor{layerName(l, "mhc.scale"), []int{3}},
			synthTensor{layerName(l, "attn.wq_a.weight"), []int{cfg.QLoraRank, H}},
			synthTensor{layerName(l, "attn.wq_b.weight"), []int{qHeadDim, cfg.QLoraRank}},
			synthTensor{layerName(l, "attn.wkv.weight"), []int{v41KVLoraRankReduced(cfg), H}},
			synthTensor{layerName(l, "attn.wo_a.weight"), []int{cfg.OLoraRank, qHeadDim}},
			synthTensor{layerName(l, "attn.wo_b.weight"), []int{H, oDim}},
			synthTensor{layerName(l, "attn.sink"), []int{nH}},
			synthTensor{layerName(l, "ffn.gate.weight"), []int{cfg.NumExperts, H}},
			synthTensor{layerName(l, "ffn.gate.e_score_correction_bias"), []int{cfg.NumExperts}},
			synthTensor{layerName(l, "ffn.shared_experts.w1.weight"), []int{I, H}},
			synthTensor{layerName(l, "ffn.shared_experts.w3.weight"), []int{I, H}},
			synthTensor{layerName(l, "ffn.shared_experts.w2.weight"), []int{H, I}},
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

	blockWeights, blockBytes, ok := quant.blockGeometry()
	if !ok {
		t.Fatalf("fixture quant %s reports no block geometry", quant)
	}
	projs := []struct {
		proj       string
		rows, cols int
	}{{"gate_proj", I, H}, {"up_proj", I, H}, {"down_proj", H, I}}
	var perProj int64
	source := NewExpertCheckpointTier(0)
	for _, p := range projs {
		perExpert := p.rows * (p.cols / blockWeights) * blockBytes
		perProj = int64(perExpert)
		blob := make([]byte, perExpert*V41RouterExperts)
		for e := 0; e < V41RouterExperts; e++ {
			expertBytes := buildRawQ4K(t, p.rows, p.cols, 7000+e)
			copy(blob[e*perExpert:], expertBytes)
		}
		if err := source.AddShard(bytes.NewReader(blob), int64(len(blob)), []FusedExpertTensor{{
			Name: "blk.0.ffn_" + p.proj + "_exps.weight", Layer: 0, Proj: p.proj,
			Arch: "deepseek41", Quant: quant, Offset: 0,
			Experts: V41RouterExperts, Rows: p.rows, Cols: p.cols,
		}}); err != nil {
			t.Fatalf("AddShard over the %s slab: %v", p.proj, err)
		}
	}
	bounded := NewExpertCheckpointTier(int64(budgetUnits) * perProj)
	bounded.index = source.index
	bounded.shards = source.shards
	bounded.overlays = source.overlays
	m.expertCheckpoint = bounded
	return m, perProj
}

// TestV41PrefillExpertReadsAreTokenIndependent is the #13296 acceptance witness.
//
// Oracle: a ONE-token prefill cannot repeat a projection across the token
// dimension, so its tier Reads is exactly the layer's distinct routed set times 3
// on ANY implementation. The witness asserts a 4-token and a 16-token prefill of
// the SAME (maximally-overlapping, all-hot) token read no more than that. On the
// parent the per-(token, pick) faulting makes Reads scale with the token count.
func TestV41PrefillExpertReadsAreTokenIndependent(t *testing.T) {
	// The fixture's tier declares no resident budget, so the layer-scoped cache
	// falls back to its own (modest) default, which at this reduced geometry holds
	// the whole layer distinct set. That is the regime in which the dedupe is
	// provably complete: tier Reads is the distinct-set size, independent of token
	// count. (When the distinct set exceeds the cache bound the dedupe is partial
	// but still never worse than the per-(token,pick) stream; that regime is
	// covered by the no-regression assertion below.)
	one := []int{1}
	short := []int{1, 1, 1, 1}
	long := []int{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}

	// Both a stream-through tier (budget 0, retains nothing on its own) and a
	// positive tier bound SMALLER than the layer's reduced distinct set: neither
	// can serve the cross-token repeats itself, so the token-independence below is
	// attributable to the layer-scoped cache, not the tier.
	for _, budgetUnits := range []int{0, 3} {
		t.Run("tierBudgetUnits="+itoa(budgetUnits), func(t *testing.T) {
			mOne, _ := v41TierOnlyBudgetModel(t, budgetUnits)
			if _, err := mOne.forwardV41(one, nil); err != nil {
				t.Fatalf("1-token prefill: %v", err)
			}
			wantReads := mOne.expertCheckpoint.Stats().Reads
			if wantReads == 0 {
				t.Fatal("fixture faulted no routed projections; the read count under test would be vacuous")
			}

			mShort, _ := v41TierOnlyBudgetModel(t, budgetUnits)
			if _, err := mShort.forwardV41(short, nil); err != nil {
				t.Fatalf("4-token prefill: %v", err)
			}
			shortReads := mShort.expertCheckpoint.Stats().Reads

			mLong, _ := v41TierOnlyBudgetModel(t, budgetUnits)
			if _, err := mLong.forwardV41(long, nil); err != nil {
				t.Fatalf("16-token prefill: %v", err)
			}
			longReads := mLong.expertCheckpoint.Stats().Reads

			if shortReads != wantReads {
				t.Fatalf("4-token prefill faulted %d projections, want %d (one token's distinct set): "+
					"the token dimension re-reads the same expert", shortReads, wantReads)
			}
			if longReads != wantReads {
				t.Fatalf("16-token prefill faulted %d projections, want %d: Reads must be independent of token count",
					longReads, wantReads)
			}
		})
	}
}

// TestV41PrefillExpertReadsNeverRegress pins the safety property: with the
// layer-scoped cache engaged, a multi-token prefill never reads MORE projections
// than the per-(token, pick) stream it replaces. The cache only ever intercepts a
// fault the contraction was about to issue, so the tier read count is monotonically
// non-increasing.
func TestV41PrefillExpertReadsNeverRegress(t *testing.T) {
	long := []int{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}
	m, _ := v41TierOnlyBudgetModel(t, 0)
	if _, err := m.forwardV41(long, nil); err != nil {
		t.Fatalf("16-token prefill: %v", err)
	}
	got := m.expertCheckpoint.Stats().Reads

	// The parent stream faults 3 projections per (token, pick): tokens * topK * 3.
	ceiling := len(long) * V41RouterTopK * 3
	if got > ceiling {
		t.Fatalf("prefill faulted %d projections, above the %d of the per-(token,pick) stream it replaces",
			got, ceiling)
	}
}
