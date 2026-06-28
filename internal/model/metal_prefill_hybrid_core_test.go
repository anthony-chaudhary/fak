package model

import "testing"

// TestQwen35HybridViaMMMatchesCPUTemplate is the host-independent correctness witness for the
// backend-agnostic hybrid prefill core (prefillQwen35HybridViaMM). The Metal twin
// (prefillBatchedMetalQwen35Hybrid, -tags fakmetal) is a thin wrapper that feeds this same core a
// GPU f16 GEMM, so its CPU-side logic — the conv1d+SiLU mixer, the q/k L2-norm, the delta-rule
// recurrent scan, the gated RMSNorm readout, the full-attention RoPE/GQA/output-gate, both
// RMSNorms and every residual — IS this file's, and is provable WITHOUT a Mac or `-tags fakmetal`:
// drive the core with a CPU mm that reproduces the proven prefillQwen35HybridQHidden path's
// per-projection qGemm8 and assert the whole prefill (logits + KV cache + linear-attn cache)
// matches that proven path.
//
// This catches the exact bug class the Metal lane is otherwise blind to off-device: a transcription
// error in the recurrence/attention/orchestration when the twin was hand-copied from the CPU
// template. Such an error diverges O(1) per layer and blows past the close-helper tolerances; the
// only residual under those tolerances is the documented grouped-vs-ungrouped Q8 GEMM float-order
// drift (qGemm8IntoMany in the template vs per-call qGemm8Into here), which is ~1e-6. What this
// does NOT witness — the GPU f16 GEMM numerics and on-device throughput — is the irreducibly
// Mac-gated residual that closes #71 (the `-tags fakmetal` parity run on the M3 Pro).
func TestQwen35HybridViaMMMatchesCPUTemplate(t *testing.T) {
	m := NewSynthetic(qwen35HybridTestCfg())
	m.Quantize()
	// 16 tokens meets qwen35HybridQBatchMinPrompt — the same prompt the batched-prefill gates use.
	prompt := []int{3, 7, 11, 5, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61}

	// Reference: the proven Q8 CPU hybrid prefill (prefillQwen35HybridQHidden), the template the
	// twin structurally copies.
	ref := m.NewSession()
	ref.Quant = true
	want := ref.headQ(ref.prefillQwen35HybridQHidden(prompt))

	// Under test: the backend-agnostic core fed a CPU mm. The mm reproduces the template's
	// per-projection math exactly — quantize the activation panel, then qGemm8Into against the
	// same m.q8 weight — so a faithful core is numerically identical to the template up to the
	// grouped-vs-ungrouped GEMM float-order drift the close-helpers tolerate.
	got := m.NewSession()
	got.Quant = true
	P := len(prompt)
	cpuMM := func(name string, X []float32, out int) []float32 {
		width := len(X) / P
		var panel q8Panel
		quantizeBatchPanelInto(&panel, X, P, width)
		Y := make([]float32, P*out)
		qGemm8Into(got.M.q8(name), &panel, Y)
		return Y
	}
	gotLogits := got.headQ(got.prefillQwen35HybridViaMM(prompt, cpuMM))

	assertQuantLogitsClose(t, "hybrid via-mm core vs CPU template logits", want, gotLogits)
	assertKVCacheQuantClose(t, "hybrid via-mm core vs CPU template", ref.Cache, got.Cache)
	assertLinearAttnCacheQuantClose(t, "hybrid via-mm core vs CPU template", ref.Cache.linear, got.Cache.linear)
}
