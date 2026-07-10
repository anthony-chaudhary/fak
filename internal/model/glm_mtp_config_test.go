package model

import (
	"encoding/json"
	"testing"
)

// glm_mtp_config_test.go pins the MTP self-speculation CONFIG SURFACE (#3078/#3197, the
// config half). The loaders already RETAIN the "mtp."/"nextn" head under RetainMTP
// (glm_retain_test.go), but retention alone never told the model the head's DEPTH.
// num_nextn_predict_layers is that surface: NumMTPLayers/HasMTPHead expose it, and
// SelfSpeculationSubstrateReady joins it to the retention flag. A dense checkpoint
// declares none, so every accessor is the zero/false no-op and the default decode is
// byte-identical.

// TestConfigParsesMTPHeadDepth: a GLM-5.2-shaped config.json declares a one-layer MTP
// head, and it flows through the custom UnmarshalJSON into NumNextNPredictLayers and the
// derived accessors.
func TestConfigParsesMTPHeadDepth(t *testing.T) {
	js := `{
		"model_type": "glm_moe_dsa",
		"architectures": ["GlmMoeDsaForCausalLM"],
		"hidden_size": 2048,
		"num_attention_heads": 16,
		"num_hidden_layers": 4,
		"vocab_size": 151552,
		"intermediate_size": 6144,
		"num_nextn_predict_layers": 1
	}`
	var cfg Config
	if err := json.Unmarshal([]byte(js), &cfg); err != nil {
		t.Fatalf("unmarshal GLM MTP config: %v", err)
	}
	if cfg.NumNextNPredictLayers != 1 {
		t.Fatalf("NumNextNPredictLayers = %d, want 1 (parsed from num_nextn_predict_layers)", cfg.NumNextNPredictLayers)
	}
	if got := cfg.NumMTPLayers(); got != 1 {
		t.Fatalf("NumMTPLayers() = %d, want 1", got)
	}
	if !cfg.HasMTPHead() {
		t.Fatalf("HasMTPHead() = false, want true (config declares an MTP head)")
	}
}

// TestConfigDenseCheckpointHasNoMTPHead: a dense Llama checkpoint declares no MTP head,
// so the field is absent and every accessor is the zero/false no-op. A negative/garbage
// value clamps to zero as well, so a self-speculation guard can never read a bogus depth.
func TestConfigDenseCheckpointHasNoMTPHead(t *testing.T) {
	js := `{
		"model_type": "llama",
		"architectures": ["LlamaForCausalLM"],
		"hidden_size": 4096,
		"num_attention_heads": 32,
		"num_hidden_layers": 4,
		"vocab_size": 32000,
		"intermediate_size": 11008
	}`
	var cfg Config
	if err := json.Unmarshal([]byte(js), &cfg); err != nil {
		t.Fatalf("unmarshal dense llama config: %v", err)
	}
	if got := cfg.NumMTPLayers(); got != 0 {
		t.Fatalf("NumMTPLayers() = %d, want 0 (no MTP head declared)", got)
	}
	if cfg.HasMTPHead() {
		t.Fatalf("HasMTPHead() = true, want false (dense checkpoint)")
	}
	cfg.NumNextNPredictLayers = -3
	if got := cfg.NumMTPLayers(); got != 0 {
		t.Fatalf("NumMTPLayers() with negative field = %d, want 0 (clamped)", got)
	}
	if cfg.HasMTPHead() {
		t.Fatalf("HasMTPHead() with negative field = true, want false (clamped)")
	}
}

// TestSelfSpeculationSubstrateReadyJoinsConfigAndRetention: the readiness predicate is
// the AND of the two independent substrate halves — a declared MTP head (config surface)
// and the RetainMTP retention flag (loader scaffold). Neither half alone is ready.
func TestSelfSpeculationSubstrateReadyJoinsConfigAndRetention(t *testing.T) {
	glm := Config{
		ModelType:             "glm_moe_dsa",
		Architectures:         []string{"GlmMoeDsaForCausalLM"},
		NumNextNPredictLayers: 1,
	}
	dense := Config{ModelType: "llama"}

	defer func() { RetainMTP = false }()

	// Default (RetainMTP off): a declared head is still not READY — nothing retained it.
	RetainMTP = false
	if glm.SelfSpeculationSubstrateReady() {
		t.Fatalf("RetainMTP=off: SelfSpeculationSubstrateReady()=true, want false (head declared but dropped)")
	}

	// Retain on + declared head: both halves agree -> ready.
	RetainMTP = true
	if !glm.SelfSpeculationSubstrateReady() {
		t.Fatalf("RetainMTP=on + MTP head: SelfSpeculationSubstrateReady()=false, want true")
	}

	// Retain on but no declared head: retaining nothing is not ready.
	if dense.SelfSpeculationSubstrateReady() {
		t.Fatalf("RetainMTP=on + no head: SelfSpeculationSubstrateReady()=true, want false (nothing to retain)")
	}
}
