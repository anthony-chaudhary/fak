package ggufload

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/model"
)

// gemma4_moe_config_test.go — witnesses that applyGemma4Config reads the shared MoE
// expert axes (issue #5494). Gemma 4 ships in BOTH shapes: a dense variant and a sparse
// Mixture-of-Experts variant (HF text_config enable_moe_block / num_experts /
// top_k_experts / moe_intermediate_size — google/gemma-4-26B-A4B-it declares 128 experts,
// top-8, moe_intermediate_size 704). Before the fix the gemma4 applier never touched the
// expert axes, so a genuine MoE checkpoint resolved with NumExperts == 0 and was silently
// configured as dense: nothing refused, the estimator just sized it without the expert FFN.
//
// A minimal in-memory File (metadata map, no tensor directory) is enough — the same idiom
// deepseek_v4_config_test.go uses — because applyGemma4Config reads only metadata: the
// expert scalars via intValueOrZero -> File.Uint64, and the per-layer geometry via
// File.IntArray / File.BoolArray. With no tensors the q/k-norm probe simply stays false.

// gemma4HeaderMetadata builds the minimum gemma4 metadata applyGemma4Config requires for
// nLayers layers: the per-layer sliding/global cadence, the per-layer kv-head counts, and
// the two key_length scalars. Callers layer the expert axes (or not) on top.
func gemma4HeaderMetadata(p string, nLayers int) map[string]Value {
	pattern := make([]Value, nLayers)
	kv := make([]Value, nLayers)
	for l := range pattern {
		// Last layer global, the rest sliding — the real gemma4 local/global cadence.
		pattern[l] = Value{Type: TypeBool, Value: l != nLayers-1}
		kv[l] = Value{Type: TypeUint32, Value: uint32(2)}
	}
	return map[string]Value{
		p + "attention.sliding_window_pattern": {Type: TypeArray, Value: pattern},
		p + "attention.head_count_kv":          {Type: TypeArray, Value: kv},
		p + "attention.key_length":             {Type: TypeUint32, Value: uint32(256)},
		p + "attention.key_length_swa":         {Type: TypeUint32, Value: uint32(128)},
	}
}

// TestApplyGemma4ConfigReadsMoEExpertAxes is the #5494 witness: a gemma4 header that
// declares the expert axes must resolve to a MoE Config — 128 experts, top-8 routing, and
// the expert FFN width — not to a dense one.
func TestApplyGemma4ConfigReadsMoEExpertAxes(t *testing.T) {
	const p = "gemma4."
	md := gemma4HeaderMetadata(p, 30)
	// The sparse shape: gemma-4-26B-A4B-it's 128 experts / top-8 / moe_intermediate_size 704.
	md[p+glmKeyExpertCount] = Value{Type: TypeUint32, Value: uint32(128)}
	md[p+glmKeyExpertUsedCount] = Value{Type: TypeUint32, Value: uint32(8)}
	md[p+glmKeyExpertFFNLength] = Value{Type: TypeUint32, Value: uint32(704)}

	f := &File{Metadata: md}
	cfg := model.Config{NumLayers: 30}
	if err := applyGemma4Config(f, p, &cfg); err != nil {
		t.Fatalf("applyGemma4Config: %v", err)
	}
	if cfg.NumExperts != 128 {
		t.Fatalf("NumExperts = %d, want 128 (a 128-expert gemma4 checkpoint loaded as dense)", cfg.NumExperts)
	}
	if cfg.NumExpertsPerTok != 8 {
		t.Fatalf("NumExpertsPerTok = %d, want 8 (top-k routing axis not read)", cfg.NumExpertsPerTok)
	}
	if cfg.MoEIntermediateSize != 704 {
		t.Fatalf("MoEIntermediateSize = %d, want 704 (expert FFN width not read)", cfg.MoEIntermediateSize)
	}
	if !cfg.IsMoE() {
		t.Fatal("cfg.IsMoE() = false for a 128-expert gemma4 header: the checkpoint is still configured dense")
	}
	// The fix must not disturb the per-layer geometry the applier already derived. That
	// geometry is a RELATION, not a size: gguf_config.go:385 sizes every per-layer slice by
	// cfg.NumLayers, and the loop below it spells layer l "sliding_attention" exactly when the
	// header's sliding_window_pattern[l] is true. So the expectation is read back off the same
	// metadata the applier read — one entry per DECLARED layer, each entry the type that
	// layer's pattern bit asks for — which holds at any layer count, not just this fixture's.
	pattern, ok := f.BoolArray(p + "attention.sliding_window_pattern")
	if !ok || len(pattern) < cfg.NumLayers {
		t.Fatalf("fixture is not self-describing: sliding_window_pattern present=%v len=%d, want >= NumLayers %d", ok, len(pattern), cfg.NumLayers)
	}
	if len(cfg.LayerTypes) != cfg.NumLayers {
		t.Fatalf("per-layer geometry regressed: len(LayerTypes) = %d, want one entry per declared layer (NumLayers = %d)", len(cfg.LayerTypes), cfg.NumLayers)
	}
	for l, got := range cfg.LayerTypes {
		wantType := "full_attention"
		if pattern[l] { // true == sliding / local, gguf_config.go:393
			wantType = "sliding_attention"
		}
		if got != wantType {
			t.Fatalf("LayerTypes[%d] = %q, want %q: the header's sliding_window_pattern[%d]=%v is not reaching the per-layer geometry", l, got, wantType, l, pattern[l])
		}
	}
}

// TestApplyGemma4ConfigDenseHeaderStaysDense pins the other half of the contract: a gemma4
// header with NO expert keys (the dense variant — enable_moe_block false) must still resolve
// to NumExperts == 0. applyMoEExpertCounts writes only present-and-positive values, so the
// dense path is untouched by the #5494 fix.
func TestApplyGemma4ConfigDenseHeaderStaysDense(t *testing.T) {
	const p = "gemma4."
	cfg := model.Config{NumLayers: 30}
	if err := applyGemma4Config(&File{Metadata: gemma4HeaderMetadata(p, 30)}, p, &cfg); err != nil {
		t.Fatalf("applyGemma4Config: %v", err)
	}
	if cfg.NumExperts != 0 || cfg.NumExpertsPerTok != 0 || cfg.MoEIntermediateSize != 0 {
		t.Fatalf("dense gemma4 header gained expert axes: experts=%d topk=%d ffn=%d",
			cfg.NumExperts, cfg.NumExpertsPerTok, cfg.MoEIntermediateSize)
	}
	if cfg.IsMoE() {
		t.Fatal("cfg.IsMoE() = true for a dense gemma4 header")
	}
}
