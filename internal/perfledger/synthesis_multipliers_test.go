package perfledger

import (
	"errors"
	"math"
	"testing"
)

// fak-test:runtime fast est=50ms lane=default
func TestSynthesisMultipliers(t *testing.T) {
	trace := []SynthesisTurn{
		{TurnIndex: 0, PrefixTokens: 100, UniqueTokens: 20, OffsetNS: 1000, DurationNS: 200},
		{TurnIndex: 1, PrefixTokens: 250, UniqueTokens: 30, OffsetNS: 2000, DurationNS: 400},
	}

	scaled, err := ApplySynthesisMultipliers(trace, SynthesisMultiplierConfig{
		PrefixStretch: 2.0,
		Speedup:       4.0,
		MaxTokens:     400,
	})
	if err != nil {
		t.Fatalf("ApplySynthesisMultipliers: %v", err)
	}
	if scaled[0].PrefixTokens != 200 || scaled[1].PrefixTokens != 400 {
		t.Fatalf("prefix tokens = (%d, %d), want (200, 400)", scaled[0].PrefixTokens, scaled[1].PrefixTokens)
	}
	if scaled[0].OffsetNS != 1000 || scaled[1].OffsetNS != 1250 {
		t.Fatalf("offsets = (%v, %v), want (1000, 1250)", scaled[0].OffsetNS, scaled[1].OffsetNS)
	}
	if scaled[0].DurationNS != 50 || scaled[1].DurationNS != 100 {
		t.Fatalf("durations = (%v, %v), want (50, 100)", scaled[0].DurationNS, scaled[1].DurationNS)
	}

	for _, bad := range []SynthesisMultiplierConfig{
		{PrefixStretch: 0, Speedup: 1},
		{PrefixStretch: 1, Speedup: -2},
		{PrefixStretch: math.NaN(), Speedup: 1},
		{PrefixStretch: 1, Speedup: math.Inf(1)},
	} {
		if _, err := ApplySynthesisMultipliers(trace, bad); !errors.Is(err, errInvalidSynthesisMultiplier) {
			t.Fatalf("cfg %+v: expected errInvalidSynthesisMultiplier, got %v", bad, err)
		}
	}
}
