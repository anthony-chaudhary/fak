package cachemeta

import "testing"

// fak-test:runtime fast est=1ms lane=default
func TestPrefixStabilityBaselineOwnsContent(t *testing.T) {
	var tracker PrefixStabilityTracker
	turn := protectedSpan()
	tracker.Observe(turn)
	turn[0].Content[0] ^= 1
	first := tracker.Observe(turn)
	if first.State != PrefixMutated || first.FirstDivergentSegment != 0 {
		t.Fatalf("caller edit rewrote initial baseline: %+v", first)
	}
	// Every subsequent accepted observation must own its content too.
	turn[1].Content[0] ^= 1
	second := tracker.Observe(turn)
	if second.State != PrefixMutated || second.FirstDivergentSegment != 1 {
		t.Fatalf("caller edit rewrote updated baseline: %+v", second)
	}
	if score := tracker.Observe(turn); score.State != PrefixStable {
		t.Fatalf("unchanged content not stable: %+v", score)
	}
}

// fak-test:runtime fast est=1ms lane=default
func TestPrefixStabilityEmptyBaselineLifecycle(t *testing.T) {
	var tracker PrefixStabilityTracker
	check := func(turn []PromptSegment, state PrefixStabilityState, baseline, observed int) PrefixStabilityScore {
		t.Helper()
		score := tracker.Observe(turn)
		if score.State != state || score.BaselineTurn != baseline || score.ObservedTurn != observed {
			t.Fatalf("score=%+v want state=%s baseline=%d observed=%d", score, state, baseline, observed)
		}
		return score
	}
	check(nil, PrefixUnknown, 1, 1)
	check([]PromptSegment{}, PrefixStable, 1, 2)
	check(protectedSpan(), PrefixStable, 2, 3) // extending an empty prefix
	// Preserve Diverge's existing empty-next semantics: no remaining tokens
	// are rebilled. The observed empty turn must still become the baseline.
	check(nil, PrefixStable, 3, 4)
	check(nil, PrefixStable, 4, 5)
	tracker.Reset()
	check(nil, PrefixUnknown, 6, 6)
	check(nil, PrefixStable, 6, 7)
}
