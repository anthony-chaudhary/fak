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

// TestShedTokenEquiv locks the proportional warm/cold blend that replaced the binary
// aggregate-warm flip (#2794/#2798). The warm portion — min(shed, warmWitness) — prices at
// ReadMultiplier (0.1x), the cold remainder at 1.0x; the function has no cliff, so the fak
// cache-value share stops swinging with a session's warm/cold mix.
func TestShedTokenEquiv(t *testing.T) {
	const eps = 1e-9
	cases := []struct {
		name       string
		shed, warm uint64
		want       float64
	}{
		{"fully cold: no witness keeps full input", 1000, 0, 1000},
		{"fully warm: witness covers the whole shed", 1000, 1000, 100},
		{"over-warm witness caps at shed", 1000, 5000, 100},
		{"cold-dominant sliver: only the witnessed token is cheap", 30_000, 1, 1*0.1 + 29_999},
		{"blended midpoint", 1000, 400, 400*0.1 + 600},
		{"nothing shed is worth nothing", 0, 5000, 0},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := ShedTokenEquiv(tc.shed, tc.warm); got < tc.want-eps || got > tc.want+eps {
				t.Fatalf("ShedTokenEquiv(%d, %d) = %v, want %v", tc.shed, tc.warm, got, tc.want)
			}
		})
	}

	// Continuity/monotonicity: as the warm witness grows, value only falls, and it never drops
	// below the fully-warm floor (shed*0.1) — the binary rule's cliff is gone.
	floor := float64(1000) * ReadMultiplier
	var prev float64 = 1e18
	for w := uint64(0); w <= 1200; w += 100 {
		v := ShedTokenEquiv(1000, w)
		if v > prev+eps {
			t.Fatalf("value rose (%v -> %v) as warm witness grew to %d: not monotone", prev, v, w)
		}
		if v < floor-eps {
			t.Fatalf("value %v dropped below the fully-warm floor %v at witness %d", v, floor, w)
		}
		prev = v
	}
}
