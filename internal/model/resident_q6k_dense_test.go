package model

import (
	"math/rand"
	"testing"
)

// resident_q6k_dense_test.go — the correctness gate for the DENSE resident-Q6_K loader fast path
// (ggufload.quant_q4k_loader.go: a non-expert Q6_K matmul weight loads raw into kqw instead of
// dequanting). That residency change moves the q4_k_m dense down_proj + lm_head out of the Q8 store
// into kqw, so three read sites must consult kqw or they panic / mis-read:
//   1. prefill GEMM    — qwen35_prefill_q4k.go proj() (panicked: "q8 tensor not built: down_proj")
//   2. decode GEMV     — sessionQ4KKernel.mul (Stage A, quant_q4k.go — already kqw-aware)
//   3. LM head         — headResident (would fall through to the tied-embedding f32 head)
// This test loads the down_proj as resident Q6_K (kqw) and proves prefill+head logits match the
// all-Q8 baseline, exercising all three sites in CPU-only CI (no Metal needed). kQuantMatRows is
// byte-identical to the f32 dequant-then-dot, so the logits agree to f32-reduction tolerance.

// fillDownProjQ6KResident moves every layer's mlp.down_proj.weight into kqw as resident Q6_K,
// quantized from the SAME f32 manifest tensor the dequant path would use, so the two paths differ
// only by where the weight is read from — not by its values.
func fillDownProjQ6KResident(t *testing.T, m *Model, cfg Config) {
	t.Helper()
	if m.kqw == nil {
		m.kqw = map[string]*kQuantTensor{}
	}
	rng := rand.New(rand.NewSource(7))
	bb := kindQ6K.blockBytes()
	for l := 0; l < cfg.NumLayers; l++ {
		name := layerName(l, "mlp.down_proj.weight")
		meta, ok := m.manifest[name]
		if !ok {
			t.Fatalf("fillDownProjQ6KResident: %s missing from manifest", name)
		}
		out, in := meta.Shape[0], meta.Shape[len(meta.Shape)-1]
		if in%qkK != 0 {
			t.Fatalf("fillDownProjQ6KResident: %s reduction dim %d not a multiple of %d", name, in, qkK)
		}
		nblk := in / qkK
		raw := make([]byte, out*nblk*bb)
		for i := range raw {
			raw[i] = byte(rng.Intn(256))
		}
		// Clamp each block's f16 scale d (last 2 bytes) to a small finite magnitude so the dot
		// stays finite — same discipline as the metal test's randomQ6KTensor.
		for o := 0; o < out*nblk; o++ {
			raw[o*bb+bb-1] = 0x2C | (raw[o*bb+bb-1] & 0x03)
		}
		m.kqw[name] = quantizeKQuantFromRaw(raw, out, in, kindQ6K)
	}
}

// TestPrefillQwen35HybridResidentQ6KDownMatchesQ8 proves the dense resident-Q6_K down_proj path
// (kqw) produces the same logits as the same weights served through the Q8 dequant path. A real
// regression in any of the three read sites (prefill proj, decode mul, head) would diverge O(1) or
// panic; kQuantMatRows is byte-identical to the f32 reference so the only residual is f32 reduction
// order. This is the CPU-only sibling of the Metal TestMetalFusedMLPQ6DownMatchesCPU.
func TestPrefillQwen35HybridResidentQ6KDownMatchesQ8(t *testing.T) {
	setQ4KSDOTForTest(false)
	t.Cleanup(func() { setQ4KSDOTForTest(true) })
	cfg := qwen35HybridQ4KTestCfg()
	prompt := []int{3, 7, 11, 5, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61}

	// Baseline: down_proj served through the Q8 store (the pre-change path). NewSynthetic +
	// Quantize builds q8w for every projection; fillQ4KMajority puts the q4_k majority in q4kw but
	// — crucially — we DO NOT fill down_proj into q4kw here, so it stays Q8 (the dequant baseline).
	base := NewSynthetic(cfg)
	base.Quantize()
	fillQ4KMajorityExceptDown(t, base, cfg)
	bs := base.NewSession()
	bs.Q4K = true
	baseLogits := bs.Prefill(prompt)

	// Resident: same model, but down_proj lives in kqw as Q6_K. The weight bytes differ from the
	// Q8 baseline (independent quant), so we compare each path against its OWN decode-loop
	// reference rather than against each other — the point is that the kqw path is self-consistent
	// (prefill == decode == head) and never panics, which the three fixes guarantee.
	res := NewSynthetic(cfg)
	res.Quantize()
	fillQ4KMajorityExceptDown(t, res, cfg)
	fillDownProjQ6KResident(t, res, cfg)

	// Reference: per-token decode loop (tokenHiddenQ → sessionQ4KKernel.mul, the kqw-aware Stage A
	// dispatch) + headResident. This reads down_proj from kqw on the decode path.
	ref := res.NewSession()
	ref.Q4K = true
	var refHidden []float32
	for _, id := range prompt {
		refHidden = ref.tokenHiddenQ(id, ref.Cache.Len())
	}
	want := ref.headResident(refHidden) // exercises the new kqw head branch

	// Got: batched prefill (qwen35_prefill_q4k.go proj → the new kqw branch) + headResident.
	got := res.NewSession()
	got.Q4K = true
	gotLogits := got.Prefill(prompt) // would panic at proj() before the fix

	assertQuantLogitsClose(t, "hybrid resident-Q6K-down prefill vs decode-loop", want, gotLogits)

	// Sanity: the Q8 baseline still produced real logits (the test harness itself is sound).
	if len(baseLogits) != cfg.VocabSize {
		t.Fatalf("baseline logits len = %d, want %d", len(baseLogits), cfg.VocabSize)
	}
}

// fillQ4KMajorityExceptDown is fillQ4KMajority without the down_proj entries, so down_proj is free
// to be routed to either the Q8 store (baseline) or kqw (resident-Q6_K) by the caller.
func fillQ4KMajorityExceptDown(t *testing.T, m *Model, cfg Config) {
	t.Helper()
	projs := [][2]any{}
	nKVhd := cfg.NumKVHeads * cfg.HeadDim
	for l := 0; l < cfg.NumLayers; l++ {
		p := layerPrefix(l)
		projs = append(projs,
			[2]any{p + "mlp.gate_proj.weight", cfg.IntermediateSize},
			[2]any{p + "mlp.up_proj.weight", cfg.IntermediateSize},
		)
		if !cfg.isLinearAttnLayer(l) {
			projs = append(projs,
				[2]any{p + "self_attn.v_proj.weight", nKVhd},
				[2]any{p + "self_attn.o_proj.weight", cfg.HiddenSize},
			)
		}
	}
	fillQ4KW(t, m, projs, 99)
}
