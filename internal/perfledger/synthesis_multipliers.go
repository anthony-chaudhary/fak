// Adapted from ai-dynamo/aiperf
// modified from the original): trace synthesis multipliers (prefix stretch,
// speedup) for synthetic multi-turn workload scaling.
package perfledger

import (
	"errors"
	"fmt"
	"math"
)

// SynthesisMultiplierConfig parameterizes trace-level prefix stretch and
// temporal speedup scaling.
type SynthesisMultiplierConfig struct {
	PrefixStretch float64 // Multiplier for shared-prefix token counts (> 0)
	Speedup       float64 // Divisor for inter-turn/duration latencies (> 0)
	MaxTokens     int     // Optional cap on stretched prefix tokens (<= 0 means uncapped)
}

// SynthesisTurn represents one turn in a synthetic multi-turn trace.
type SynthesisTurn struct {
	TurnIndex    int
	PrefixTokens int
	UniqueTokens int
	OffsetNS     float64
	DurationNS   float64
}

var errInvalidSynthesisMultiplier = errors.New("perfledger: invalid synthesis multiplier")

// ApplySynthesisMultipliers scales a trace's shared prefix lengths by PrefixStretch
// and compresses its timeline/durations by Speedup, returning a fresh slice.
func ApplySynthesisMultipliers(trace []SynthesisTurn, cfg SynthesisMultiplierConfig) ([]SynthesisTurn, error) {
	if math.IsNaN(cfg.PrefixStretch) || math.IsInf(cfg.PrefixStretch, 0) || cfg.PrefixStretch <= 0 {
		return nil, fmt.Errorf("%w: prefix stretch %v must be finite and > 0", errInvalidSynthesisMultiplier, cfg.PrefixStretch)
	}
	if math.IsNaN(cfg.Speedup) || math.IsInf(cfg.Speedup, 0) || cfg.Speedup <= 0 {
		return nil, fmt.Errorf("%w: speedup %v must be finite and > 0", errInvalidSynthesisMultiplier, cfg.Speedup)
	}

	out := make([]SynthesisTurn, len(trace))
	var anchorNS float64
	if len(trace) > 0 {
		anchorNS = trace[0].OffsetNS
	}
	for i, t := range trace {
		if t.PrefixTokens < 0 || t.UniqueTokens < 0 || t.OffsetNS < 0 || t.DurationNS < 0 {
			return nil, fmt.Errorf("%w: turn %d has negative metric", errInvalidSynthesisMultiplier, i)
		}
		stretched := int(math.Round(float64(t.PrefixTokens) * cfg.PrefixStretch))
		if cfg.MaxTokens > 0 && stretched > cfg.MaxTokens {
			stretched = cfg.MaxTokens
		}
		out[i] = SynthesisTurn{
			TurnIndex:    t.TurnIndex,
			PrefixTokens: stretched,
			UniqueTokens: t.UniqueTokens,
			OffsetNS:     anchorNS + (t.OffsetNS-anchorNS)/cfg.Speedup,
			DurationNS:   t.DurationNS / cfg.Speedup,
		}
	}
	return out, nil
}
