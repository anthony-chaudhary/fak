package model

import "testing"

// deepseek41_mla_route_test.go — regression witness for the DeepSeek-V4.1-Flash
// forward-route selection bug. The vcruz GGUF converter emits V4.1-Flash with the
// MLA low-rank attention layout (q_a/q_b/kv_a/o_proj_a/o_proj_b) and NO dense
// self_attn.q_proj leaf, and it sets ModelType == "deepseek41". usesMLAMoELayout()
// gates the MLA forward branch (hal.go tokenHALOutput), so if it does not include
// "deepseek41" the forward falls to the dense-MHA branch, requests the absent
// q_proj leaf, and panics `model: missing tensor` on hardware.
//
// The GGUF "deepseek41" identity is DISTINCT from the safetensors "deepseek_v41"
// identity that IsDeepSeekV41()/refuseDeepSeekV41Native() deliberately holds
// fail-closed. This test pins BOTH properties: the GGUF identity routes MLA and is
// NOT caught by the safetensors refusal.

func TestDeepSeek41UsesMLAMoELayout(t *testing.T) {
	ds41 := Config{ModelType: "deepseek41"}
	if !ds41.usesMLAMoELayout() {
		t.Fatalf("Config{ModelType: %q}.usesMLAMoELayout() = false, want true — the MLA forward would not run and the dense branch would request the absent self_attn.q_proj leaf", ds41.ModelType)
	}

	if got := (Config{ModelType: "llama"}).usesMLAMoELayout(); got {
		t.Errorf("Config{ModelType: %q}.usesMLAMoELayout() = true, want false (dense-MHA negative control)", "llama")
	}

	if !(Config{ModelType: "deepseek2"}).usesMLAMoELayout() {
		t.Errorf("Config{ModelType: %q}.usesMLAMoELayout() = false, want true (deepseek2 regression)", "deepseek2")
	}

	if err := refuseDeepSeekV41Native(ds41); err != nil {
		t.Fatalf("refuseDeepSeekV41Native(%q) = %v, want nil — the GGUF deepseek41 identity must be admitted to the MLA route, not held by the safetensors deepseek_v41 fail-closed gate", ds41.ModelType, err)
	}
}
