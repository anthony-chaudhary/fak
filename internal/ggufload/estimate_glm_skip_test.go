package ggufload

import "testing"

// TestTensorCPUOffloadExpertSkipsNextn pins that the memory-plan estimator classifies the GLM-5.2
// MTP ("nextn") head + vision tensors the SAME way the loader treats them — dropped, contributing
// nothing — rather than raising "no canonical mapping". Without this the --cpu-offload-experts
// pre-flight fit check (serveGGUFCPUOffloadMemoryPlan -> EstimateCPUOffloadExpertsMemoryPlan)
// would reject a real GLM-5.2 checkpoint the loader loads fine.
func TestTensorCPUOffloadExpertSkipsNextn(t *testing.T) {
	for _, n := range []string{
		"blk.78.nextn.eh_proj.weight",
		"blk.78.nextn.shared_head_norm.weight",
		"v.blk.0.attn_q.weight",
	} {
		host, err := tensorCPUOffloadExpert(n, "glm_moe_dsa")
		if err != nil {
			t.Errorf("tensorCPUOffloadExpert(%q) errored, want skip: %v", n, err)
		}
		if host {
			t.Errorf("tensorCPUOffloadExpert(%q) = host=true, a skipped tensor is not a host expert", n)
		}
	}
	// the split KV-b halves stay device (non-expert), and the batched experts stay host.
	if host, err := tensorCPUOffloadExpert("blk.0.attn_k_b.weight", "glm_moe_dsa"); err != nil || host {
		t.Errorf("attn_k_b -> host=%v err=%v, want device(false),nil", host, err)
	}
	if host, err := tensorCPUOffloadExpert("blk.0.ffn_gate_exps.weight", "glm_moe_dsa"); err != nil || !host {
		t.Errorf("ffn_gate_exps -> host=%v err=%v, want host(true),nil", host, err)
	}
}
