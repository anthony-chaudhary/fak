package model

import (
	"math"
	"testing"
)

// Partial rotary is not a Qwen-only axis. Any GGUF whose rope.dimension_count is smaller
// than its head_dim sets Config.PartialRotaryFactor (ggufload/gguf_config.go builds it as
// ropeDim/headDim), which is how GPT-NeoX (rotary_pct), StableLM (partial_rotary_factor)
// and Phi-2 all arrive. invFreqDenom used to hand back the FULL head_dim for every family
// except Qwen3.5/MiniMax, so those tables were built as 1/theta^(2j/head_dim) instead of
// HF's 1/theta^(2j/rotary_dim): every frequency came out too high and the rotation drifted
// further from HF the deeper into the context it ran. It went unnoticed because the two
// formulas agree exactly at j==0 -- the first frequency always matches.

func TestInvFreqDenomIsTheRotaryDimForEveryPartialRotaryFamily(t *testing.T) {
	// head_dim 16 at rotary_pct 0.25 -> rotary_dim 4, the published Pythia/GPT-NeoX geometry.
	cfg := Config{
		HiddenSize: 64, NumLayers: 2, NumHeads: 4, NumKVHeads: 4, HeadDim: 16,
		RopeTheta: 10000, PartialRotaryFactor: 0.25,
	}
	// Guard the premise: this must be reached through the FALL-THROUGH, otherwise the test
	// is only re-proving the Qwen3.5/MiniMax special case it is meant to generalize past.
	if cfg.IsQwen35Hybrid() || cfg.isMiniMax() {
		t.Fatal("fixture drifted into the Qwen3.5/MiniMax branch; it must exercise the fall-through")
	}
	if cfg.usesMLAMoELayout() {
		t.Fatal("fixture drifted into the MLA/MoE branch; it must exercise the fall-through")
	}
	if got, want := cfg.rotaryDim(), 4; got != want {
		t.Fatalf("rotaryDim() = %d, want %d", got, want)
	}
	if got, want := cfg.invFreqDenom(), cfg.rotaryDim(); got != want {
		t.Fatalf("invFreqDenom() = %d, want the rotary dim %d (denominating by head_dim %d makes every frequency too high)", got, want, cfg.HeadDim)
	}

	inv := invFreq(cfg, 0)
	if len(inv) != cfg.rotaryDim()/2 {
		t.Fatalf("inv_freq length %d, want rotary_dim/2 = %d", len(inv), cfg.rotaryDim()/2)
	}
	for j := range inv {
		want := 1.0 / math.Pow(cfg.RopeTheta, float64(2*j)/float64(cfg.rotaryDim()))
		if math.Abs(inv[j]-want) > 1e-12 {
			t.Fatalf("inv_freq[%d] = %.17g, want %.17g (HF: 1/theta^(2j/rotary_dim))", j, inv[j], want)
		}
	}
	// j==0 agrees under both denominators, so the numeric check above is only load-bearing
	// from j==1 on. Pin that entry against the head_dim table explicitly so a regression to
	// HeadDim cannot slip through on a fixture that happens to be too small to tell.
	if len(inv) < 2 {
		t.Fatalf("fixture too small to observe the divergence: %d frequencies", len(inv))
	}
	if headDimTable := 1.0 / math.Pow(cfg.RopeTheta, 2.0/float64(cfg.HeadDim)); math.Abs(inv[1]-headDimTable) < 1e-9 {
		t.Fatalf("inv_freq[1] = %.17g matches the head_dim-denominated table %.17g: invFreqDenom regressed to HeadDim", inv[1], headDimTable)
	}
}

func TestInvFreqDenomIsUnchangedForFullRotary(t *testing.T) {
	// The fix widens the denominator to rotaryDim(), which already collapses to HeadDim
	// whenever the factor is absent, non-positive, or >=1. Pin that it moves nothing for
	// the full-rotary families -- every Llama/Qwen2/Mistral table must stay bit-identical.
	for _, prf := range []float64{0, -1, 1, 1.5} {
		cfg := Config{
			HiddenSize: 64, NumLayers: 2, NumHeads: 4, NumKVHeads: 4, HeadDim: 16,
			RopeTheta: 10000, PartialRotaryFactor: prf,
		}
		if got := cfg.invFreqDenom(); got != cfg.HeadDim {
			t.Fatalf("partial_rotary_factor %v: invFreqDenom() = %d, want head_dim %d unchanged", prf, got, cfg.HeadDim)
		}
	}
}

func TestInvFreqDenomKeepsItsBranchOrder(t *testing.T) {
	// MLA keeps qk_rope_head_dim: cfg.HeadDim there comes from the larger MLA latent
	// attention.key_length, so the rotary width is the explicit qk_rope_head_dim, not a
	// fraction of head_dim.
	mla := Config{ModelType: "deepseek2", HeadDim: 24, QKRopeHeadDim: 8, RopeTheta: 10000}
	if !mla.usesMLAMoELayout() {
		t.Fatal("fixture is not an MLA/MoE layout; it cannot exercise the MLA branch")
	}
	if got := mla.invFreqDenom(); got != 8 {
		t.Fatalf("MLA invFreqDenom() = %d, want qk_rope_head_dim 8", got)
	}

	// And the ORDER: a config that is both a Qwen3.5 hybrid and an MLA layout must keep the
	// rotary_dim. The first branch reads redundant next to the fall-through but is not --
	// deleting it would silently hand this config the qk_rope_head_dim instead.
	both := Config{
		ModelType: "deepseek2", HeadDim: 24, QKRopeHeadDim: 8, RopeTheta: 10000,
		PartialRotaryFactor: 0.5, LayerTypes: []string{"linear_attention"},
	}
	if !both.IsQwen35Hybrid() || !both.usesMLAMoELayout() {
		t.Fatalf("fixture must satisfy BOTH branches: qwen35=%v mla=%v", both.IsQwen35Hybrid(), both.usesMLAMoELayout())
	}
	if got, want := both.invFreqDenom(), both.rotaryDim(); got != want {
		t.Fatalf("invFreqDenom() = %d, want the rotary dim %d: the Qwen3.5/MiniMax branch must win over the MLA branch", got, want)
	}
	if both.rotaryDim() == both.QKRopeHeadDim {
		t.Fatal("fixture cannot distinguish the two branches: rotary_dim equals qk_rope_head_dim")
	}
}
