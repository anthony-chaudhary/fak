package model

// metal_prefill_hybrid_gate.go — the architecture gate that routes the Qwen3.6 hybrid
// (Gated-DeltaNet) family to the Metal hybrid-prefill twin instead of tripping
// requirePreNorm("Metal prefill"). It is UN-TAGGED (built in every configuration) so kv.go's
// dispatch can reference it whether or not the GPU twin is linked in; the twin it routes to is
// linked only under `-tags fakmetal` (metal_prefill_hybrid.go) with a default-build stub
// (metal_prefill_hybrid_stub.go).
//
// Before this gate, prefillBatchedMetal called requirePreNorm("Metal prefill"), which the
// hybrid passes (the hybrid IS PreNorm) but whose generic full-attention body could not run the
// linear_attention layers — so the hybrid fell back to the CPU and never touched the Metal
// prefill. Lifting that fence means routing the hybrid to a twin that keeps the GDN recurrence on
// the CPU and batches only the projection/MLP GEMMs on the GPU — the measured prefill lever
// (the projections are the wall; the GDN scan is ~0.5%; #65, #977). The shipped backend-agnostic
// core prefillQwen35HybridViaMM (metal_prefill_hybrid_core.go) is exactly that body; the Metal
// twin feeds it a GPU f16 GEMM (#71).

// metalQwen35HybridPrefillOK gates the Metal hybrid prefill. It is the SAME architecture gate the
// Q8 and resident-Q4_K hybrid paths use (q8Qwen35HybridPrefillOK) — the Metal path covers the
// identical Qwen3.5/3.6 hybrid family; only the projection backend differs.
func metalQwen35HybridPrefillOK(cfg Config, promptLen int) bool {
	return q8Qwen35HybridPrefillOK(cfg, promptLen)
}
