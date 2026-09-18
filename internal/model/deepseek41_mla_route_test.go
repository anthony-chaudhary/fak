package model

import "testing"

// deepseek41_mla_route_test.go — regression witness for the DeepSeek-V4.1-Flash
// forward-route selection. The vcruz GGUF converter emits V4.1-Flash as a NON-MLA
// attention (wkv -> head_dim, kv_norm, partial in-place RoPE) and sets
// ModelType == "deepseek41". The official reference
// (deepseek-ai/DeepSeek-V4.1-Flash, inference/model.py class Attention) proves
// this: self.wkv = Linear(dim, head_dim), self.kv_norm = RMSNorm(head_dim), rope
// on only the last qk_rope_head_dim (64). There is no kv_lora_rank, no kv_b_proj,
// and no separate k_pe, so the artifact's attn_kv.weight [5120,512] is correct and
// must reach the native non-MLA forwardV41 path (internal/model/v41_forward.go),
// NOT the GLM-DSA MLA forward (internal/model/glm_dsa_session.go).
//
// Before the fix IsDeepSeekV41() was FALSE for "deepseek41" (it matched only the
// safetensors "deepseek_v41"/"deepseek_v41_text" spellings) while
// usesMLAMoELayout() INCLUDED it, so kv.go fell through to tokenHiddenGLMDsa ->
// glmDsaAppendAttentionKV, which requested a 576-wide kv_a_proj_with_mqa from a
// [512,5120] tensor and panicked.
//
// The GGUF "deepseek41" identity is DISTINCT from the safetensors "deepseek_v41"
// identity that admitDeepSeekV41Published()/refuseDeepSeekV41Native() still hold
// fail-closed. This test pins BOTH: the GGUF identity routes native, and the
// safetensors identity keeps its published-config fail-closed gate.

func TestDeepSeek41IsDeepSeekV41(t *testing.T) {
	ds41 := Config{ModelType: "deepseek41"}
	if !ds41.IsDeepSeekV41() {
		t.Fatalf("Config{ModelType: %q}.IsDeepSeekV41() = false, want true — without it kv.go falls through to the GLM-DSA MLA forward and panics on the [512,5120] attn_kv tensor", ds41.ModelType)
	}
	if got := (Config{ModelType: "llama"}).IsDeepSeekV41(); got {
		t.Errorf("Config{ModelType: %q}.IsDeepSeekV41() = true, want false (negative control)", "llama")
	}
}

func TestDeepSeek41DoesNotUseMLAMoELayout(t *testing.T) {
	ds41 := Config{ModelType: "deepseek41"}
	if ds41.usesMLAMoELayout() {
		t.Fatalf("Config{ModelType: %q}.usesMLAMoELayout() = true, want false — V4.1 is a NON-MLA attention (wkv -> head_dim, kv_norm, partial rope); the MLA path would request a 576-wide kv_a_proj_with_mqa from the [512,5120] attn_kv tensor", ds41.ModelType)
	}
	if !(Config{ModelType: "deepseek2"}).usesMLAMoELayout() {
		t.Errorf("Config{ModelType: %q}.usesMLAMoELayout() = false, want true (deepseek2 regression)", "deepseek2")
	}
	if got := (Config{ModelType: "llama"}).usesMLAMoELayout(); got {
		t.Errorf("Config{ModelType: %q}.usesMLAMoELayout() = true, want false (dense-MHA negative control)", "llama")
	}
}

// TestDeepSeek41RefusalKeepsSafetensorsIdentityFailClosed pins the safetensors
// fail-closed property: the SAFETENSORS "deepseek_v41"/"deepseek_v41_text"
// identities (and any config carrying a parsed DeepSeekV41 envelope) are still
// refused by refuseDeepSeekV41Native, whose published-config gate
// (admitDeepSeekV41Published) is what admits an official checkpoint.
//
// The GGUF loader's canonical "deepseek41" identity is deliberately ADMITTED by
// this hook (returns nil). It has no published envelope, and its fail-closed gate
// is the native forward's own stage-aware admission ladder (v41ForwardAdmitted),
// not this safetensors hook. Refusing "deepseek41" here would refuse the exact
// staged vcruz GGUF at newModel / NewFromF32Tensors before the native non-MLA
// forwardV41 path could ever be reached.
func TestDeepSeek41RefusalKeepsSafetensorsIdentityFailClosed(t *testing.T) {
	// The safetensors identities with no parsed envelope must be refused.
	if err := refuseDeepSeekV41Native(Config{ModelType: "deepseek_v41"}); err == nil {
		t.Fatal("refuseDeepSeekV41Native(deepseek_v41) = nil, want the typed ErrV41NativeUnsupported (safetensors fail-closed gate)")
	}
	if err := refuseDeepSeekV41Native(Config{ModelType: "deepseek_v41_text"}); err == nil {
		t.Fatal("refuseDeepSeekV41Native(deepseek_v41_text) = nil, want the typed ErrV41NativeUnsupported")
	}
	// A non-V4.1 identity is never refused by this hook.
	if err := refuseDeepSeekV41Native(Config{ModelType: "llama"}); err != nil {
		t.Fatalf("refuseDeepSeekV41Native(llama) = %v, want nil", err)
	}
	// The GGUF identity is ADMITTED to the native path: its fail-closed gate is
	// v41ForwardAdmitted, not this safetensors hook.
	if err := refuseDeepSeekV41Native(Config{ModelType: "deepseek41"}); err != nil {
		t.Fatalf("refuseDeepSeekV41Native(deepseek41) = %v, want nil — the GGUF identity routes to the native non-MLA forwardV41 path, whose v41ForwardAdmitted ladder is the fail-closed gate; refusing it here would block the staged vcruz GGUF at the load seam", err)
	}
}

// TestDeepSeek41GGUFIdentityAdmittedToNativeLoad pins the load reachability the
// refusal-hook carve-out protects: a Config{ModelType:"deepseek41"} must not be
// refused by refuseDeepSeekV41Native, which is consulted at every production load
// seam — newModel (weights.go:374) and NewFromF32Tensors (weights.go:477, the GGUF
// load path's construction point), plus Load and ClassifyForwardPath. Before the
// carve-out the hook refused the GGUF identity here, so the staged vcruz Q2_K GGUF
// was rejected with ErrV41NativeUnsupported at load and never reached the native
// forward at all — a reachability regression versus the pre-change behavior.
//
// Limitation: this test asserts the refusal hook directly rather than driving a
// full newModel/NewFromF32Tensors construction (which needs a manifest-backed
// weight source). The hook is the sole V4.1-identity refusal at those seams, so a
// nil here is exactly the property those seams consult; the deeper forward-stage
// admission is covered separately by the v41ForwardAdmitted ladder tests.
func TestDeepSeek41GGUFIdentityAdmittedToNativeLoad(t *testing.T) {
	gguf := Config{ModelType: "deepseek41"}
	if err := refuseDeepSeekV41Native(gguf); err != nil {
		t.Fatalf("refuseDeepSeekV41Native(deepseek41) = %v, want nil — the GGUF identity must be admitted past the load seam (newModel/NewFromF32Tensors) to the native non-MLA forwardV41 path", err)
	}
	// The safetensors sibling must remain refused at the same seam, so the
	// carve-out is identity-precise and does not widen the admitted family.
	if err := refuseDeepSeekV41Native(Config{ModelType: "deepseek_v41"}); err == nil {
		t.Fatal("refuseDeepSeekV41Native(deepseek_v41) = nil at the load seam, want refused — the carve-out must not admit the safetensors identity")
	}
}
