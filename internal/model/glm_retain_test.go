package model

import "testing"

// glm_retain_test.go — the #3078/#3197 GLM-5.2 self-speculation substrate scaffold, model side.
// It is the inverse of TestGLMDropsMtpAndVisualTensorsAtLoad (glm_test.go): with RetainMTP set,
// the MTP ("mtp.") head is RETAINED by both load gates (skipLoadTensor / quantSourceTensorName)
// while the vision tower is still dropped. With RetainMTP clear (the default) the behavior is
// byte-identical to the historical drop, which the existing test already pins.

func TestGLMRetainsMtpHeadWhenRetainMTPSet(t *testing.T) {
	glmCfg := Config{ModelType: "glm_moe_dsa", Architectures: []string{"GlmMoeDsaForCausalLM"}}

	defer func() { RetainMTP = false }()
	RetainMTP = true

	// The MTP head survives both gates when retained.
	for _, name := range []string{"mtp.0.embed.weight", "mtp.head.weight"} {
		if skipLoadTensor(glmCfg, name) {
			t.Fatalf("RetainMTP=on: skipLoadTensor(%q)=true, want false (MTP retained)", name)
		}
		if got, keep := quantSourceTensorName(glmCfg, name); !keep || got != name {
			t.Fatalf("RetainMTP=on: quantSourceTensorName(%q)=(%q,%v), want kept unchanged", name, got, keep)
		}
	}

	// The vision tower is ALWAYS dropped, even under RetainMTP.
	for _, name := range []string{"model.visual.encoder.weight"} {
		if !skipLoadTensor(glmCfg, name) {
			t.Fatalf("RetainMTP=on: skipLoadTensor(%q)=false, want true (vision always dropped)", name)
		}
		if got, keep := quantSourceTensorName(glmCfg, name); keep || got != "" {
			t.Fatalf("RetainMTP=on: quantSourceTensorName(%q)=(%q,%v), want dropped", name, got, keep)
		}
	}

	// Forward tensors are still kept.
	for _, name := range []string{
		"model.embed_tokens.weight",
		"model.layers.0.self_attn.q_proj.weight",
	} {
		if skipLoadTensor(glmCfg, name) {
			t.Fatalf("RetainMTP=on: skipLoadTensor(%q)=true, want false (kept tensor)", name)
		}
	}

	// Default (flag OFF): the MTP head is dropped again — proving the retention is flag-gated,
	// so a non-scaffold load stays byte-identical.
	RetainMTP = false
	if !skipLoadTensor(glmCfg, "mtp.0.embed.weight") {
		t.Fatalf("RetainMTP=off: skipLoadTensor(mtp.*)=false, want true (drop restored)")
	}
	if got, keep := quantSourceTensorName(glmCfg, "mtp.0.embed.weight"); keep || got != "" {
		t.Fatalf("RetainMTP=off: quantSourceTensorName(mtp.*)=(%q,%v), want dropped", got, keep)
	}
}
