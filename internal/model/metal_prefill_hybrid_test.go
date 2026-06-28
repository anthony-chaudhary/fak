//go:build darwin && cgo && fakmetal

package model

// metal_prefill_hybrid_test.go — the on-device correctness gate for the Metal hybrid
// (Qwen3.6 Gated-DeltaNet) prefill twin prefillBatchedMetalQwen35Hybrid (#71). Built ONLY under
// `-tags fakmetal`. The twin's entire CPU-side orchestration — both RMSNorms, the conv1d+SiLU
// mixer, the q/k L2-norm, the per-head delta-rule recurrent scan, the gated RMSNorm readout, the
// full-attention RoPE/GQA/output-gate, and every residual — is already proven host-independently
// against the CPU template by TestQwen35HybridViaMMMatchesCPUTemplate (no Mac, no cgo). This file
// adds the ONE residual that test cannot reach: the GPU f16 GEMM numerics the twin substitutes
// for the projection/MLP matmuls. It is the Mac-gated witness named as the last open step in
// experiments/qwen36/metal-hybrid-prefill-status-2026-06-28.md §3 (step 2).
//
// It holds the Metal hybrid prefill (s.Metal -> prefillBatchedMetalQwen35Hybrid) to the proven
// CPU Q8 hybrid prefill (s.Quant -> prefillQwen35HybridQ) on the SAME quantized weights: both
// read the identical Q8 store, so the only divergence is the projection backend — the CPU's
// qgemm8 vs the GPU's dequant-Q8->f16 MatMul — and the two must agree up to GPU f16
// float-accumulation order. That is the exact parity class TestMetalDecodeResidentMatchesCPU
// establishes for the resident decode forward (#67): an f16-dequant GPU GEMM held to the CPU Q8
// reference, logit cosine ~1.0 with the same argmax. A real wiring bug in the twin (wrong weight
// name, wrong per-layer-kind upload set, wrong GEMM stride) diverges O(1) per layer and trips the
// logit cosine / argmax — or the per-full-attention-layer KV cosine — below.

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/metalgemm"
)

// TestPrefillQwen35HybridMetalMatchesCPU prefills the same 16-token prompt through the CPU Q8
// hybrid path and the Metal hybrid twin from a fresh synthetic Qwen3.6-hybrid model, and asserts
// the device prefill produces the same final-token distribution (logit cosine + argmax) and
// advances the KV cache to the same per-layer state within the f16-GEMM drift band.
func TestPrefillQwen35HybridMetalMatchesCPU(t *testing.T) {
	if !metalgemm.Available() {
		t.Skip("no Metal device available")
	}
	defer metalgemm.Reset()

	cfg := qwen35HybridTestCfg()
	m := NewSynthetic(cfg)
	m.Quantize() // build the Q8 store the twin uploads (dequantQ8 -> f16) and the CPU path dots
	// 16 tokens: meets qwen35HybridQBatchMinPrompt so both Prefills take the batched hybrid route.
	prompt := []int{3, 7, 11, 5, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61}
	if !q8Qwen35HybridPrefillOK(cfg, len(prompt)) {
		t.Fatal("q8Qwen35HybridPrefillOK declined the synthetic hybrid cfg — neither path would take the hybrid route")
	}

	// Reference: the proven CPU Q8 hybrid prefill (s.Quant -> prefillQwen35HybridQ).
	ref := m.NewSession()
	ref.Quant = true
	want := ref.Prefill(prompt)

	// Device: the Metal hybrid twin (s.Metal -> prefillBatchedMetalQwen35Hybrid).
	got := m.NewSession()
	got.Metal = true
	gotLogits := got.Prefill(prompt)

	// Load-bearing gate: the full-prefill logits — a function of every projection on every layer,
	// the GDN recurrence, the full attention, and the head — match up to f16 accumulation order.
	cos, maxRel := cosineAndMaxRel(want, gotLogits)
	if argmaxF(want) != argmaxF(gotLogits) || cos < 0.999 {
		t.Errorf("metal hybrid prefill logits: cpu argmax=%d gpu argmax=%d cos=%.6f maxRel=%.4g (want same argmax, cos>=0.999)\n  cpu[:6]=%v\n  gpu[:6]=%v",
			argmaxF(want), argmaxF(gotLogits), cos, maxRel, head6(want), head6(gotLogits))
	} else {
		t.Logf("metal hybrid prefill logits: argmax=%d cos=%.6f maxRel=%.4g OK", argmaxF(gotLogits), cos, maxRel)
	}

	// The device prefill must advance the SAME cache shape decode/Evict/Clone consumes.
	if ref.Cache.Len() != got.Cache.Len() {
		t.Fatalf("metal hybrid prefill cache len = %d, want %d", got.Cache.Len(), ref.Cache.Len())
	}
	// Per-layer KV parity within the f16-GEMM band: the full-attention layers populate K/Kraw/V
	// (the linear-attention layers carry the recurrent/conv state, already covered transitively by
	// the logits above), so a self_attn projection-upload bug localizes here even when it partially
	// cancels in the pooled logits.
	for l := 0; l < cfg.NumLayers; l++ {
		if len(ref.Cache.K[l]) == 0 {
			continue // linear-attention layer: no K/Kraw/V store
		}
		if c := cosine(ref.Cache.K[l], got.Cache.K[l]); c < 0.999 {
			t.Errorf("metal hybrid prefill K layer %d cosine=%.6f (want >=0.999)", l, c)
		}
		if c := cosine(ref.Cache.Kraw[l], got.Cache.Kraw[l]); c < 0.999 {
			t.Errorf("metal hybrid prefill Kraw layer %d cosine=%.6f (want >=0.999)", l, c)
		}
		if c := cosine(ref.Cache.V[l], got.Cache.V[l]); c < 0.999 {
			t.Errorf("metal hybrid prefill V layer %d cosine=%.6f (want >=0.999)", l, c)
		}
	}
}
