package gateway

import (
	"testing"
)

// TestStructuralSpeculativeSelectorReachesDefaultFrontDoor is the config-witness
// for #1891: the STRUCTURAL cfg.SpeculativeMode selects the speculative route by
// construction (no ambient env var), and the FAK_SPECULATIVE / SPECULATIVE env
// vars remain the operator override that wins when set.
func TestStructuralSpeculativeSelectorReachesDefaultFrontDoor(t *testing.T) {
	// 1. Structural selector alone admits the eligible route (no env present).
	t.Run("structural mtp admits eligible Vulkan route without env", func(t *testing.T) {
		t.Setenv("FAK_SPECULATIVE", "")
		t.Setenv("SPECULATIVE", "")
		cfg := gatewayNGramEligibleConfig()
		cfg.SpeculativeMode = "mtp"
		// The n-gram predicate wants "ngram", so a structural "mtp" must NOT admit
		// the ngram route; the structural selector is consulted per-mode.
		if shouldEnableNGramSpeculative(cfg) {
			t.Fatal("structural mtp wrongly admitted the ngram route")
		}
		ngram := gatewayNGramEligibleConfig()
		ngram.SpeculativeMode = "ngram"
		if !shouldEnableNGramSpeculative(ngram) {
			t.Fatal("structural ngram did not admit the eligible Vulkan Qwen hybrid route without env")
		}
	})

	// 2. Env override wins over the structural selector.
	t.Run("env overrides structural selector", func(t *testing.T) {
		t.Setenv("FAK_SPECULATIVE", "ngram")
		t.Setenv("SPECULATIVE", "")
		cfg := gatewayNGramEligibleConfig()
		cfg.SpeculativeMode = "mtp"
		if !shouldEnableNGramSpeculative(cfg) {
			t.Fatal("FAK_SPECULATIVE=ngram did not override structural mtp")
		}

		t.Setenv("FAK_SPECULATIVE", "mtp")
		cfg = gatewayNGramEligibleConfig()
		cfg.SpeculativeMode = "ngram"
		if shouldEnableNGramSpeculative(cfg) {
			t.Fatal("FAK_SPECULATIVE=mtp did not override structural ngram")
		}
	})

	// 3. Empty structural selector + no env leaves the route ordinary (the
	// historical default is preserved for a config that never sets the field).
	t.Run("empty structural selector and no env stays ordinary", func(t *testing.T) {
		t.Setenv("FAK_SPECULATIVE", "")
		t.Setenv("SPECULATIVE", "")
		cfg := gatewayNGramEligibleConfig()
		if shouldEnableNGramSpeculative(cfg) {
			t.Fatal("empty selector admitted a route with no env override")
		}
	})

	// 4. The historical env OR semantics are preserved exactly: with conflicting
	// env values, the strict Metal/Vulkan admission guard still refuses, while the
	// plain n-gram OR still admits the route (this is the pre-existing contract
	// asserted by vulkan_mtp_test.go; the structural field must not change it).
	t.Run("conflicting env preserves the historical OR semantics", func(t *testing.T) {
		t.Setenv("FAK_SPECULATIVE", "mtp")
		t.Setenv("SPECULATIVE", "ngram")
		cfg := gatewayNGramEligibleConfig()
		cfg.SpeculativeMode = "ngram"
		if !shouldEnableNGramSpeculative(cfg) {
			t.Fatal("conflicting env changed the existing ngram OR semantics")
		}
	})
}
