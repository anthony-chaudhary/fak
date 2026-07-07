package cacheprice

import "testing"

// TestCanonicalMultipliers pins the published Anthropic prompt-cache multipliers. This is
// the ONE anchor for the 0.1 / 1.25 / 2.0 economic fact: gateway.CacheReadMultiplier and
// resume.CacheReadMultiplier are defined as these symbols and agent.defaultCacheReadMult is
// pinned equal to them (agent/cacheprice_pin_test.go), so this test is where a rate change is
// consciously acknowledged. If Anthropic republishes the rates, edit here and the compiler
// carries the new value to every consumer that reads it.
func TestCanonicalMultipliers(t *testing.T) {
	if ReadMultiplier != 0.1 || Write5mMultiplier != 1.25 || Write1hMultiplier != 2.0 {
		t.Fatalf("canonical cache multipliers moved: read=%v write5m=%v write1h=%v (want 0.1 / 1.25 / 2.0)",
			ReadMultiplier, Write5mMultiplier, Write1hMultiplier)
	}
}
