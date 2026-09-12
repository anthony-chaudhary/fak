package model

import (
	"encoding/json"
	"errors"
	"testing"
)

// TestV41VisionRetainedAsTypedMarker proves the official checkpoint's
// vision_config is DETECTED and retained as a fail-closed marker, and that a
// V4.1 vision request is refused with the typed error — never silently routed
// through the text-only forward.
func TestV41VisionRetainedAsTypedMarker(t *testing.T) {
	_, cfg := readDeepSeekV41Config(t)
	if !cfg.IsDeepSeekV41() {
		t.Fatal("official config not recognized as V4.1")
	}
	if !cfg.HasDeepSeekV41Vision() {
		t.Fatal("official vision_config was not retained as a vision marker")
	}
	if cfg.DeepSeekV41.Vision == nil {
		t.Fatal("DeepSeekV41.Vision is nil; vision_config metadata dropped")
	}
	if got := cfg.DeepSeekV41.Vision.ModelType; got != "deepseek_v41_vision" {
		t.Fatalf("vision model_type = %q, want deepseek_v41_vision", got)
	}
	if got := cfg.DeepSeekV41.Vision.NumLayers; got != 32 {
		t.Fatalf("vision num_hidden_layers = %d, want 32", got)
	}
	if got := cfg.DeepSeekV41.Vision.HiddenSize; got != 1024 {
		t.Fatalf("vision hidden_size = %d, want 1024", got)
	}
	if got := cfg.DeepSeekV41.Vision.MaxImageTokens; got != 1024 {
		t.Fatalf("vision max_image_tokens = %d, want 1024", got)
	}
	if err := cfg.RefuseDeepSeekV41Vision(); !errors.Is(err, ErrV41VisionUnsupported) {
		t.Fatalf("RefuseDeepSeekV41Vision error = %v, want ErrV41VisionUnsupported", err)
	}
}

// TestV41VisionImageTokenAloneIsAMarker proves a wrapper-declared image token
// with no retained vision_config metadata still fails closed.
func TestV41VisionImageTokenAloneIsAMarker(t *testing.T) {
	_, cfg := readDeepSeekV41Config(t)
	if cfg.ImageTokenID == 0 {
		t.Fatal("image_token_id was not parsed")
	}
	stripped := cfg
	stripped.DeepSeekV41 = &DeepSeekV41Config{WrapperModelType: "deepseek_v41", TextModelType: "deepseek_v41_text"}
	if !stripped.HasDeepSeekV41Vision() {
		t.Fatal("image_token_id alone did not raise the vision marker")
	}
	if err := stripped.RefuseDeepSeekV41Vision(); !errors.Is(err, ErrV41VisionUnsupported) {
		t.Fatalf("error = %v, want ErrV41VisionUnsupported", err)
	}
}

// TestV41VisionTextOnlyUnaffected proves the vision hook is inert when no
// vision is requested, and for every non-V4.1 family, so the text path is
// byte-for-byte unchanged.
func TestV41VisionTextOnlyUnaffected(t *testing.T) {
	textOnly := Config{ModelType: "deepseek_v41_text"}
	if textOnly.HasDeepSeekV41Vision() {
		t.Fatal("text-only V4.1 config reported a vision marker")
	}
	if err := textOnly.RefuseDeepSeekV41Vision(); err != nil {
		t.Fatalf("text-only V4.1 refused: %v", err)
	}
	for _, mt := range []string{"", "llama", "deepseek_v4", "qwen3"} {
		c := Config{ModelType: mt}
		if c.HasDeepSeekV41Vision() {
			t.Fatalf("family %q reported a V4.1 vision marker", mt)
		}
		if err := c.RefuseDeepSeekV41Vision(); err != nil {
			t.Fatalf("family %q refused: %v", mt, err)
		}
	}
}

// TestV41VisionRoundTripsThroughJSON proves the vision marker survives a
// Config JSON round trip (no silent capability gain or loss).
func TestV41VisionRoundTripsThroughJSON(t *testing.T) {
	_, cfg := readDeepSeekV41Config(t)
	encoded, err := json.Marshal(cfg)
	if err != nil {
		t.Fatal(err)
	}
	var back Config
	if err := json.Unmarshal(encoded, &back); err != nil {
		t.Fatalf("round trip: %v", err)
	}
	if !back.HasDeepSeekV41Vision() || back.DeepSeekV41.Vision == nil {
		t.Fatal("vision marker lost in round trip")
	}
	if err := back.RefuseDeepSeekV41Vision(); !errors.Is(err, ErrV41VisionUnsupported) {
		t.Fatalf("reparsed config error = %v, want ErrV41VisionUnsupported", err)
	}
}

// TestV41DSparkRetainedAsTypedMarker proves the official checkpoint's dspark_
// metadata is DETECTED and retained as a fail-closed marker, and that a V4.1
// DSpark request is refused with the typed error — speculation never silently
// runs through the plain autoregressive forward.
func TestV41DSparkRetainedAsTypedMarker(t *testing.T) {
	_, cfg := readDeepSeekV41Config(t)
	if !cfg.IsDeepSeekV41() {
		t.Fatal("official config not recognized as V4.1")
	}
	if !cfg.HasDeepSeekV41DSpark() {
		t.Fatal("official dspark_* metadata was not retained as a DSpark marker")
	}
	d := cfg.DeepSeekV41.DSpark
	if d == nil {
		t.Fatal("DeepSeekV41.DSpark is nil; dspark metadata dropped")
	}
	if d.BlockSize != 5 {
		t.Fatalf("dspark_block_size = %d, want 5", d.BlockSize)
	}
	if len(d.TargetLayerIDs) != 3 || d.TargetLayerIDs[0] != 37 || d.TargetLayerIDs[2] != 39 {
		t.Fatalf("dspark_target_layer_ids = %v, want [37 38 39]", d.TargetLayerIDs)
	}
	if d.MarkovRank != 256 {
		t.Fatalf("dspark_markov_rank = %d, want 256", d.MarkovRank)
	}
	if d.NoiseTokenID != 128799 {
		t.Fatalf("dspark_noise_token_id = %d, want 128799", d.NoiseTokenID)
	}
	if d.NumRoutedExperts != 128 || d.NumExpertsPerTok != 3 {
		t.Fatalf("dspark experts = (%d, %d), want (128, 3)", d.NumRoutedExperts, d.NumExpertsPerTok)
	}
	if err := cfg.RefuseDeepSeekV41DSpark(); !errors.Is(err, ErrV41DSparkUnsupported) {
		t.Fatalf("RefuseDeepSeekV41DSpark error = %v, want ErrV41DSparkUnsupported", err)
	}
}

// TestV41DSparkNextNHeadAloneIsAMarker closes the counterexample found in
// adversarial review: a V4.1 config that declares num_nextn_predict_layers (the
// MTP head count) but no dspark_* block must STILL fail closed rather than slip
// through as text-only.
func TestV41DSparkNextNHeadAloneIsAMarker(t *testing.T) {
	c := Config{ModelType: "deepseek_v41_text", DeepSeekV41: &DeepSeekV41Config{
		WrapperModelType: "deepseek_v41",
		TextModelType:    "deepseek_v41_text",
		DSpark:           &DeepSeekV41DSparkConfig{NextNPredictLayers: 3},
	}}
	if !c.HasDeepSeekV41DSpark() {
		t.Fatal("num_nextn_predict_layers alone did not raise the DSpark marker")
	}
	if err := c.RefuseDeepSeekV41DSpark(); !errors.Is(err, ErrV41DSparkUnsupported) {
		t.Fatalf("error = %v, want ErrV41DSparkUnsupported", err)
	}
}

// TestV41DSparkSubordinateFieldAloneIsAMarker closes the second half of the
// counterexample: a lone dspark_noise_token_id (or any subordinate dspark_
// field) must also fail closed.
func TestV41DSparkSubordinateFieldAloneIsAMarker(t *testing.T) {
	for name, d := range map[string]*DeepSeekV41DSparkConfig{
		"noise token":     {NoiseTokenID: 128799},
		"markov rank":     {MarkovRank: 256},
		"routed experts":  {NumRoutedExperts: 128},
		"experts per tok": {NumExpertsPerTok: 3},
	} {
		t.Run(name, func(t *testing.T) {
			c := Config{ModelType: "deepseek_v41_text", DeepSeekV41: &DeepSeekV41Config{DSpark: d}}
			if !c.HasDeepSeekV41DSpark() {
				t.Fatalf("subordinate %s alone did not raise the DSpark marker", name)
			}
			if err := c.RefuseDeepSeekV41DSpark(); !errors.Is(err, ErrV41DSparkUnsupported) {
				t.Fatalf("error = %v, want ErrV41DSparkUnsupported", err)
			}
		})
	}
}

// TestV41DSparkTextOnlyUnaffected proves the DSpark hook is inert when no
// DSpark is requested, and for every non-V4.1 family.
func TestV41DSparkTextOnlyUnaffected(t *testing.T) {
	textOnly := Config{ModelType: "deepseek_v41_text"}
	if textOnly.HasDeepSeekV41DSpark() {
		t.Fatal("text-only V4.1 config reported a DSpark marker")
	}
	if err := textOnly.RefuseDeepSeekV41DSpark(); err != nil {
		t.Fatalf("text-only V4.1 refused: %v", err)
	}
	for _, mt := range []string{"", "llama", "deepseek_v4", "qwen3"} {
		c := Config{ModelType: mt}
		if c.HasDeepSeekV41DSpark() {
			t.Fatalf("family %q reported a V4.1 DSpark marker", mt)
		}
		if err := c.RefuseDeepSeekV41DSpark(); err != nil {
			t.Fatalf("family %q refused: %v", mt, err)
		}
	}
}

// TestV41DSparkRoundTripsThroughJSON proves the DSpark marker survives a
// Config JSON round trip.
func TestV41DSparkRoundTripsThroughJSON(t *testing.T) {
	_, cfg := readDeepSeekV41Config(t)
	encoded, err := json.Marshal(cfg)
	if err != nil {
		t.Fatal(err)
	}
	var back Config
	if err := json.Unmarshal(encoded, &back); err != nil {
		t.Fatalf("round trip: %v", err)
	}
	if !back.HasDeepSeekV41DSpark() || back.DeepSeekV41.DSpark == nil {
		t.Fatal("DSpark marker lost in round trip")
	}
	if err := back.RefuseDeepSeekV41DSpark(); !errors.Is(err, ErrV41DSparkUnsupported) {
		t.Fatalf("reparsed config error = %v, want ErrV41DSparkUnsupported", err)
	}
}

// TestV41HooksClaimNoCapability pins the gold-plating boundary: the hooks carry
// no execution surface. The official config is admitted as a text-only shape,
// and the ONLY V4.1-requested paths are typed refusals.
func TestV41HooksClaimNoCapability(t *testing.T) {
	_, cfg := readDeepSeekV41Config(t)
	var vision, dspark bool
	for _, mt := range []string{"deepseek_v41", "deepseek_v41_text", ""} {
		c := cfg
		c.ModelType = mt
		if errors.Is(c.RefuseDeepSeekV41Vision(), ErrV41VisionUnsupported) {
			vision = true
		} else if c.RefuseDeepSeekV41Vision() != nil {
			t.Fatalf("model_type %q: unexpected vision error: %v", mt, c.RefuseDeepSeekV41Vision())
		}
		if errors.Is(c.RefuseDeepSeekV41DSpark(), ErrV41DSparkUnsupported) {
			dspark = true
		} else if c.RefuseDeepSeekV41DSpark() != nil {
			t.Fatalf("model_type %q: unexpected dspark error: %v", mt, c.RefuseDeepSeekV41DSpark())
		}
	}
	if !vision || !dspark {
		t.Fatalf("official config did not exercise both typed refusals: vision=%v dspark=%v", vision, dspark)
	}
}
