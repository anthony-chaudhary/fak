package model

// v41_attn_resident_read_test.go — the #13276 witness for the DeepSeek-V4.1
// attention/shared-expert projection reads from a RESIDENT quantization store.
//
// The pinned vcruz Q2_K artifact keeps the V4.1 dense attention projections in a
// resident k-quant store rather than the f32 manifest: isQuantWeight already
// admits .attn.wq_a / .attn.wq_b / .attn.wkv / .attn.wo_a / .attn.wo_b and the
// shared-expert leaves, so the loader stores their raw k-quant blocks and drops
// the f32 copies. Two seams then interact:
//
//  1. v41ForwardAdmitted resolves presence through residentShape, which is
//     residency-complete, so the resident-only projection ADMITS.
//  2. forwardV41's per-layer reads (v41_forward.go v41Layer) went through
//     m.tensor, which resolves ONLY the f32 manifest and panics on a resident-
//     only weight. On the physical strix3 serve the forward cleared admission and
//     entered v41Layer(0), then died: "panic: model: missing tensor
//     model.layers.0.attn.wq_a.weight".
//
// This test builds a model whose attention + shared-expert projections live ONLY
// in a resident store and drives the real forwardV41. On the parent commit the
// m.tensor read panics; after the fix the forward runs and emits finite logits.
//
// Deliberately built on pre-existing API only (no v41ProjF32 / residentF32Mat
// helper), so the same test compiles against the parent commit and fails at
// runtime there — the red-then-green symptom witness the land gate requires.

import (
	"math"
	"testing"
)

// v41MoveProjToResidentKQuant removes the f32 manifest entry for a named
// per-layer projection and installs the supplied resident raw k-quant payload in
// m.kqw under the same name, so the ONLY way to read the projection is the
// resident store. shape is the model [out, in].
func v41MoveProjToResidentKQuant(m *Model, l int, leaf string, shape []int, raw []byte, kind kQuantKind) {
	name := layerName(l, leaf)
	delete(m.manifest, name)
	if m.kqw == nil {
		m.kqw = map[string]*kQuantTensor{}
	}
	m.kqw[name] = quantizeKQuantFromRaw(raw, shape[0], shape[1], kind)
}

// v41AttnLeaves are the quant-admitted per-layer projections the V4.1 forward
// reads as [out, in] matmul weights.
var v41AttnLeaves = []string{
	"attn.wq_a.weight",
	"attn.wq_b.weight",
	"attn.wkv.weight",
	"attn.wo_a.weight",
	"attn.wo_b.weight",
	"ffn.gate.weight",
	"ffn.shared_experts.w1.weight",
	"ffn.shared_experts.w3.weight",
	"ffn.shared_experts.w2.weight",
}

// v41ProjectionShape returns the reduced-fixture [out, in] of a quant-admitted
// per-layer projection leaf.
func v41ProjectionShape(t *testing.T, m *Model, leaf string) []int {
	t.Helper()
	cfg := m.Cfg
	H := cfg.HiddenSize
	qHeadDim := cfg.NumHeads * cfg.HeadDim
	oDim := cfg.OLoraRank * cfg.OGroups
	switch leaf {
	case "attn.wq_a.weight":
		return []int{cfg.QLoraRank, H}
	case "attn.wq_b.weight":
		return []int{qHeadDim, cfg.QLoraRank}
	case "attn.wkv.weight":
		return []int{v41KVLoraRankReduced(cfg), H}
	case "attn.wo_a.weight":
		return []int{cfg.OLoraRank, qHeadDim}
	case "attn.wo_b.weight":
		return []int{H, oDim}
	case "ffn.gate.weight":
		return []int{cfg.NumExperts, H}
	case "ffn.shared_experts.w1.weight", "ffn.shared_experts.w3.weight":
		return []int{cfg.MoEIntermediateSize, H}
	case "ffn.shared_experts.w2.weight":
		return []int{H, cfg.MoEIntermediateSize}
	}
	t.Fatalf("v41ProjectionShape: unknown leaf %s", leaf)
	return nil
}

// TestV41AttnResidentRead is the #13276 acceptance witness: a model whose
// attention + shared-expert projections live ONLY in the resident k-quant store
// runs the real reduced forward and emits finite logits. On the parent commit the
// m.tensor read panics "model: missing tensor model.layers.0.attn.wq_a.weight".
func TestV41AttnResidentRead(t *testing.T) {
	m := v41ReducedModel(t)

	// Move every quant-admitted per-layer projection to a resident Q8_0 store so
	// no projection is reachable through the f32 manifest.
	for _, leaf := range v41AttnLeaves {
		shape := v41ProjectionShape(t, m, leaf)
		raw := v41ResidentQ8Raw(shape[0], shape[1])
		v41MoveProjToResidentKQuant(m, 0, leaf, shape, raw, kindQ8_0)
	}
	for _, leaf := range v41AttnLeaves {
		if m.has(layerName(0, leaf)) {
			t.Fatalf("resident fixture still carries an f32 manifest entry for %s; the read is not exercised", leaf)
		}
	}

	// Admission is residency-complete and must pass.
	if err := m.v41ForwardAdmitted(); err != nil {
		t.Fatalf("resident-projection admission error = %v, want nil", err)
	}

	// The real forward must run instead of panicking in m.tensor.
	act, err := m.forwardV41([]int{1, 3}, nil)
	if err != nil {
		t.Fatalf("resident-projection forward error = %v, want nil", err)
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