package model

import "testing"

// TestMetalQwen35HybridPrefillOK pins the routing gate that lifts requirePreNorm for the Metal
// hybrid prefill (#71). The gate is a pure Config predicate, so it is fully witnessable in the
// default build WITHOUT a Mac or `-tags fakmetal`: the irreducibly Mac-gated residual is the GPU
// twin's on-device parity/throughput, not this admission decision. It asserts the gate (a) admits
// exactly the same Qwen3.5/3.6 hybrid family the proven Q8 path admits (the delegation invariant —
// the Metal twin must not silently cover a different arch than the CPU template it mirrors), and
// (b) declines a non-hybrid model so the standard full-attention path keeps its existing
// prefillBatchedMetal/requirePreNorm route.
func TestMetalQwen35HybridPrefillOK(t *testing.T) {
	cfg := qwen35HybridQ4KTestCfg() // a real qwen35 hybrid topology the Q8 gate admits

	// Delegation invariant: identical verdict to q8Qwen35HybridPrefillOK across the prompt-length
	// boundary (the batch-min prompt that separates batched prefill from decode-shaped input).
	for _, P := range []int{0, 1, qwen35HybridQBatchMinPrompt - 1, qwen35HybridQBatchMinPrompt, 22, 64} {
		if got, want := metalQwen35HybridPrefillOK(cfg, P), q8Qwen35HybridPrefillOK(cfg, P); got != want {
			t.Errorf("metalQwen35HybridPrefillOK(hybrid, P=%d) = %v, want q8 gate %v", P, got, want)
		}
	}

	// Positive: the hybrid at the batch-min prompt is admitted to the Metal twin.
	if !metalQwen35HybridPrefillOK(cfg, qwen35HybridQBatchMinPrompt) {
		t.Fatalf("hybrid at P=%d should be admitted to the Metal hybrid twin", qwen35HybridQBatchMinPrompt)
	}

	// Negative on architecture: a non-hybrid (no linear_attention layer) model is declined, so it
	// keeps the existing prefillBatchedMetal/requirePreNorm path rather than the hybrid twin.
	std := cfg
	std.LayerTypes = nil // IsQwen35Hybrid() == false
	if metalQwen35HybridPrefillOK(std, 64) {
		t.Fatalf("non-hybrid model must NOT route to the Metal hybrid twin")
	}
}
