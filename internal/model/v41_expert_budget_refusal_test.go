package model

// v41_expert_budget_refusal_test.go — the #13280 witness for the bounded-residency
// guard on the DeepSeek-V4.1 streamed routed-expert read path.
//
// A physical strix3 serve of the pinned V4.1 Flash Q2_K slate under
// --cpu-offload-experts advanced into full streamed-expert staging and was then
// OOM-killed by the host kernel (anon-rss ~37 GB) before the first token: the
// routed-expert read materialized expert weights with no effective residency
// bound. The tier's declared hostBytes was a cache budget, not a ceiling, because
// a fault that Admit could not fit was handed to the caller anyway
// (`return w, nil // stream-through`).
//
// The fix makes a POSITIVE hostBytes a real ceiling: a fault whose slab cannot be
// made resident within it fails closed with a typed named refusal that wraps
// ErrExpertCheckpointBudget (fak#13280) instead of handing out bytes past the
// bound. hostBytes == 0 (the default, stream-through) is byte-for-byte unchanged.
//
// Test design. The refusal test deliberately asserts the closure through the
// PARENT-existing surface only (errors.Is(err, ErrV41ForwardStage), the #13276/
// #13278 contract, plus the error text naming the tensor and the budget) so the
// SAME file compiles against the parent production shape and fails at RUNTIME
// there: the parent stream-through returns nil and the fixture's budgeted slab
// exceeds its budget. The sentinel/typed-operand assertions live in their own
// test, which necessarily references the new symbols.

import (
	"bytes"
	"errors"
	"math"
	"strings"
	"testing"
)

// v41BudgetTierStride is one routed-expert projection's raw slab stride in the
// reduced fixture: rows * (cols/qkK) * q4kBlockBytes with rows=cols=256, so
// 256 * 1 * 144 = 36864 bytes for every one of w1/w3/w2.
const v41BudgetTierStride = v41TierWidth * (v41TierWidth / qkK) * q4kBlockBytes

// v41BudgetOnlyModel builds the reduced V4.1 fixture of v41TierOnlyModel but
// attaches a tier with an explicit non-zero hostBytes budget. Every routed-expert
// leaf is carried ONLY by that tier, so the forward's read path is forced through
// the bounded admit. budget is the tier's declared residency ceiling in bytes.
func v41BudgetOnlyModel(t *testing.T, budget int64) *Model {
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

	projs := []struct {
		proj string
		rows int
		cols int
	}{
		{"gate_proj", I, H},
		{"up_proj", I, H},
		{"down_proj", H, I},
	}
	tier := NewExpertCheckpointTier(budget)
	fillSeed := 7100
	for _, p := range projs {
		perExpert := p.rows * (p.cols / qkK) * q4kBlockBytes
		blob := make([]byte, perExpert*V41RouterExperts)
		for e := 0; e < V41RouterExperts; e++ {
			expertBytes := buildRawQ4K(t, p.rows, p.cols, fillSeed+e)
			if len(expertBytes) != perExpert {
				t.Fatalf("%s expert stride = %d bytes, want %d", p.proj, len(expertBytes), perExpert)
			}
			copy(blob[e*perExpert:], expertBytes)
		}
		if err := tier.AddShard(bytes.NewReader(blob), int64(len(blob)), []FusedExpertTensor{{
			Name: "blk.0.ffn_" + p.proj + "_exps.weight", Layer: 0, Proj: p.proj,
			Arch: "deepseek41", Quant: ExpertCheckpointQ4K, Offset: 0,
			Experts: V41RouterExperts, Rows: p.rows, Cols: p.cols,
		}}); err != nil {
			t.Fatalf("AddShard over the %s slab: %v", p.proj, err)
		}
	}
	m.expertCheckpoint = tier
	return m
}

// TestV41ExpertBudgetRefusalIsNamed is the #13280 acceptance witness: a tier whose
// declared hostBytes budget is smaller than a single routed-expert projection's
// slab cannot serve the forward's read within its bound, and the forward must
// FAIL CLOSED with a typed, named refusal rather than panicking, streaming the
// slab through anyway (parent behavior), or OOM-ing.
//
// It references only the parent-existing surface (errors.Is against
// ErrV41ForwardStage, the #13276/#13278 named-refusal contract, plus the error
// text) so it compiles against the parent production shape and fails at RUNTIME
// there.
func TestV41ExpertBudgetRefusalIsNamed(t *testing.T) {
	// Half a projection: no single routed-expert slab can ever be made resident
	// within this ceiling, so the very first fault must refuse.
	budget := int64(v41BudgetTierStride / 2)
	m := v41BudgetOnlyModel(t, budget)

	if err := m.v41ForwardAdmitted(); err != nil {
		t.Fatalf("budget-only admission error = %v, want nil", err)
	}

	_, err := m.forwardV41([]int{1, 3}, nil)
	if err == nil {
		t.Fatalf("forward over a tier with budget %d bytes (< one slab of %d bytes) returned nil error; "+
			"the unbounded stream-through that OOM-killed strix3 is still reachable", budget, v41BudgetTierStride)
	}
	if !errors.Is(err, ErrV41ForwardStage) {
		t.Fatalf("forward refusal = %v, want a %v-wrapping named stage refusal", err, ErrV41ForwardStage)
	}
	msg := err.Error()
	if !strings.Contains(msg, "bounded host residency budget") {
		t.Fatalf("refusal %q does not name the bounded-residency budget refusal", msg)
	}
	if !strings.Contains(msg, "model.layers.0.ffn.experts.") {
		t.Fatalf("refusal %q does not name the routed-expert tensor", msg)
	}
}

// TestV41ExpertBudgetRefusalSentinel pins the typed contract of the new refusal:
// it is reachable through errors.Is, it is the exported sentinel, and the typed
// error carries the offending tensor and the budget that refused it.
func TestV41ExpertBudgetRefusalSentinel(t *testing.T) {
	budget := int64(v41BudgetTierStride / 2)
	m := v41BudgetOnlyModel(t, budget)

	_, err := m.forwardV41([]int{1}, nil)
	if err == nil {
		t.Fatal("forward returned nil error; want the typed budget refusal")
	}
	if !errors.Is(err, ErrExpertCheckpointBudget) {
		t.Fatalf("refusal = %v, want errors.Is(_, ErrExpertCheckpointBudget)", err)
	}
	var be *ExpertCheckpointBudgetError
	if !errors.As(err, &be) {
		t.Fatalf("refusal = %v, want an *ExpertCheckpointBudgetError operand carrier", err)
	}
	if be.Tensor == "" || !strings.Contains(be.Tensor, "ffn.experts.") {
		t.Fatalf("typed refusal Tensor = %q, want a routed-expert tensor name", be.Tensor)
	}
	if be.Bytes != v41BudgetTierStride {
		t.Fatalf("typed refusal Bytes = %d, want the slab stride %d", be.Bytes, v41BudgetTierStride)
	}
	if be.Budget != budget {
		t.Fatalf("typed refusal Budget = %d, want the declared budget %d", be.Budget, budget)
	}
}

// TestV41ExpertBudgetBoundsResidency is the positive-side witness: a budget that
// CAN hold a handful of slabs lets the forward complete, while the tier's resident
// byte total never exceeds the declared ceiling. This is the invariant the parent
// tier claimed but did not enforce on the handed-out path.
func TestV41ExpertBudgetBoundsResidency(t *testing.T) {
	// Four projections' worth: enough to run the reduced forward with eviction
	// churn across the 6 routed experts, small enough that the ceiling bites.
	budget := int64(4 * v41BudgetTierStride)
	m := v41BudgetOnlyModel(t, budget)

	act, err := m.forwardV41([]int{1, 3}, nil)
	if err != nil {
		t.Fatalf("forward over an adequate budget %d = %v, want nil", budget, err)
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
	st := m.ExpertCheckpointStats()
	if st.BudgetBytes != budget {
		t.Fatalf("tier budget = %d, want %d", st.BudgetBytes, budget)
	}
	if st.ResidentBytes > st.BudgetBytes {
		t.Fatalf("tier resident bytes %d exceed the declared ceiling %d", st.ResidentBytes, st.BudgetBytes)
	}
	if st.PeakBytes > st.BudgetBytes {
		t.Fatalf("tier peak bytes %d exceed the declared ceiling %d", st.PeakBytes, st.BudgetBytes)
	}
}

// TestV41ExpertBudgetDefaultStreamThrough is the default-unchanged witness: the
// historical hostBytes == 0 (stream-through) arm must still run the forward to
// finite logits. The budget guard is gated entirely on budget > 0, so this path
// is byte-for-byte the pre-#13280 behavior.
func TestV41ExpertBudgetDefaultStreamThrough(t *testing.T) {
	m := v41BudgetOnlyModel(t, 0)

	act, err := m.forwardV41([]int{1, 3}, nil)
	if err != nil {
		t.Fatalf("stream-through (budget 0) forward error = %v, want nil", err)
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
	st := m.ExpertCheckpointStats()
	if st.BudgetBytes != 0 {
		t.Fatalf("stream-through tier budget = %d, want 0", st.BudgetBytes)
	}
	if st.Failures != 0 {
		t.Fatalf("stream-through tier failures = %d, want 0 (no refusal on the default path)", st.Failures)
	}
}
