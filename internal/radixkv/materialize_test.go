package radixkv

import (
	"strings"
	"testing"
)

func sampleTargetFence() RACCompatibilityFence {
	return RACCompatibilityFence{
		ModelID:         "llama-3-8b",
		TokenizerID:     "llama-3",
		AttentionRegime: "causal",
		Dtype:           "fp16",
		QuantMode:       "none",
		RoPEScheme:      "default",
		MaxContextLen:   8192,
	}
}

// TestMaterialize_PrefixDirectAttach witnesses that prefix [0, P) attaches directly with zero shift.
func TestMaterialize_PrefixDirectAttach(t *testing.T) {
	targetFence := sampleTargetFence()

	// Plan with prefix [0, 40) and compute gap [40, 100)
	plan := &ReusePlan{
		PromptLen: 100,
		PrefixLen: 40,
		Spans: []PlanSpan{
			{
				Start:       0,
				End:         40,
				Action:      ActionDirectReuse,
				CandidateID: "", // prefix
				SourceStart: 0,
				SourceEnd:   40,
				Tier:        TierL1GPU,
			},
			{
				Start:  40,
				End:    100,
				Action: ActionCompute,
			},
		},
	}

	res, err := MaterializePlan(plan, targetFence, nil)
	if err != nil {
		t.Fatalf("MaterializePlan failed: %v", err)
	}

	if res.DirectAttachedTokens != 40 {
		t.Errorf("DirectAttachedTokens=%d, want 40", res.DirectAttachedTokens)
	}
	if res.RepairedTokens != 0 {
		t.Errorf("RepairedTokens=%d, want 0", res.RepairedTokens)
	}
	if res.RecomputedTokens != 60 {
		t.Errorf("RecomputedTokens=%d, want 60", res.RecomputedTokens)
	}
	if res.FailsafeRecompute {
		t.Errorf("FailsafeRecompute=true, want false")
	}

	// Action 0: prefix direct attach
	a0 := res.Actions[0]
	if a0.Mode != MatDirectAttach {
		t.Errorf("action 0 mode=%s, want %s", a0.Mode, MatDirectAttach)
	}
	if a0.ShiftTokens != 0 {
		t.Errorf("action 0 ShiftTokens=%d, want 0", a0.ShiftTokens)
	}
	if a0.DeltaPosition != 0 {
		t.Errorf("action 0 DeltaPosition=%d, want 0", a0.DeltaPosition)
	}
	if a0.RequiresReRoPE {
		t.Errorf("action 0 RequiresReRoPE=true, want false")
	}
	if a0.EstimatedTokens != 40 {
		t.Errorf("action 0 EstimatedTokens=%d, want 40", a0.EstimatedTokens)
	}

	// Action 1: compute gap
	a1 := res.Actions[1]
	if a1.Mode != MatFullRecompute {
		t.Errorf("action 1 mode=%s, want %s", a1.Mode, MatFullRecompute)
	}
	if a1.EstimatedTokens != 60 {
		t.Errorf("action 1 EstimatedTokens=%d, want 60", a1.EstimatedTokens)
	}
}

// TestMaterialize_MovedSpanSelectiveRepair witnesses that an addressed candidate with
// Start != SourceStart flags MatSelectiveRepair and RequiresReRoPE = true.
func TestMaterialize_MovedSpanSelectiveRepair(t *testing.T) {
	targetFence := sampleTargetFence()

	// Plan with displaced candidate spans:
	// cand-forward: [30, 70) with SourceStart=10 -> Shift = 30 - 10 = +20
	// cand-backward: [70, 90) with SourceStart=100 -> Shift = 70 - 100 = -30
	plan := &ReusePlan{
		PromptLen: 100,
		PrefixLen: 0,
		Spans: []PlanSpan{
			{
				Start:  0,
				End:    30,
				Action: ActionCompute,
			},
			{
				Start:       30,
				End:         70,
				Action:      ActionDirectReuse,
				CandidateID: "cand-forward",
				SourceStart: 10,
				SourceEnd:   50,
				Tier:        TierL2Host,
			},
			{
				Start:       70,
				End:         90,
				Action:      ActionSelectiveRepair,
				CandidateID: "cand-backward",
				SourceStart: 100,
				SourceEnd:   120,
				Tier:        TierL3Remote,
			},
			{
				Start:  90,
				End:    100,
				Action: ActionCompute,
			},
		},
	}

	candidateFences := map[string]RACCompatibilityFence{
		"cand-forward":  targetFence,
		"cand-backward": targetFence,
	}

	res, err := MaterializePlan(plan, targetFence, candidateFences)
	if err != nil {
		t.Fatalf("MaterializePlan failed: %v", err)
	}

	if res.FailsafeRecompute {
		t.Errorf("FailsafeRecompute=true, want false")
	}
	if res.DirectAttachedTokens != 0 {
		t.Errorf("DirectAttachedTokens=%d, want 0", res.DirectAttachedTokens)
	}
	if res.RepairedTokens != 60 {
		t.Errorf("RepairedTokens=%d, want 60", res.RepairedTokens)
	}
	if res.RecomputedTokens != 40 {
		t.Errorf("RecomputedTokens=%d, want 40", res.RecomputedTokens)
	}

	// Action 1: shifted forward (+20)
	a1 := res.Actions[1]
	if a1.Mode != MatSelectiveRepair {
		t.Errorf("action 1 mode=%s, want %s", a1.Mode, MatSelectiveRepair)
	}
	if a1.ShiftTokens != 20 {
		t.Errorf("action 1 ShiftTokens=%d, want 20", a1.ShiftTokens)
	}
	if a1.DeltaPosition != 20 {
		t.Errorf("action 1 DeltaPosition=%d, want 20", a1.DeltaPosition)
	}
	if !a1.RequiresReRoPE {
		t.Errorf("action 1 RequiresReRoPE=false, want true")
	}
	if a1.EstimatedTokens != 40 {
		t.Errorf("action 1 EstimatedTokens=%d, want 40", a1.EstimatedTokens)
	}

	// Action 2: shifted backward (-30)
	a2 := res.Actions[2]
	if a2.Mode != MatSelectiveRepair {
		t.Errorf("action 2 mode=%s, want %s", a2.Mode, MatSelectiveRepair)
	}
	if a2.ShiftTokens != -30 {
		t.Errorf("action 2 ShiftTokens=%d, want -30", a2.ShiftTokens)
	}
	if a2.DeltaPosition != -30 {
		t.Errorf("action 2 DeltaPosition=%d, want -30", a2.DeltaPosition)
	}
	if !a2.RequiresReRoPE {
		t.Errorf("action 2 RequiresReRoPE=false, want true")
	}
	if a2.EstimatedTokens != 20 {
		t.Errorf("action 2 EstimatedTokens=%d, want 20", a2.EstimatedTokens)
	}
}

// TestMaterialize_RegimeMismatchFailsClosed witnesses that a mismatched Dtype,
// QuantMode, or AttentionRegime fails closed to MatFullRecompute.
func TestMaterialize_RegimeMismatchFailsClosed(t *testing.T) {
	targetFence := sampleTargetFence()

	testCases := []struct {
		name       string
		candFence  RACCompatibilityFence
		wantReason string
	}{
		{
			name: "mismatched dtype",
			candFence: RACCompatibilityFence{
				ModelID:         targetFence.ModelID,
				TokenizerID:     targetFence.TokenizerID,
				AttentionRegime: targetFence.AttentionRegime,
				Dtype:           "bf16", // target is fp16
				QuantMode:       targetFence.QuantMode,
				RoPEScheme:      targetFence.RoPEScheme,
			},
			wantReason: "dtype mismatch",
		},
		{
			name: "mismatched quant mode",
			candFence: RACCompatibilityFence{
				ModelID:         targetFence.ModelID,
				TokenizerID:     targetFence.TokenizerID,
				AttentionRegime: targetFence.AttentionRegime,
				Dtype:           targetFence.Dtype,
				QuantMode:       "int4", // target is none
				RoPEScheme:      targetFence.RoPEScheme,
			},
			wantReason: "quant mode mismatch",
		},
		{
			name: "mismatched attention regime",
			candFence: RACCompatibilityFence{
				ModelID:         targetFence.ModelID,
				TokenizerID:     targetFence.TokenizerID,
				AttentionRegime: "softcap_50", // target is causal
				Dtype:           targetFence.Dtype,
				QuantMode:       targetFence.QuantMode,
				RoPEScheme:      targetFence.RoPEScheme,
			},
			wantReason: "attention regime mismatch",
		},
		{
			name: "mismatched model id",
			candFence: RACCompatibilityFence{
				ModelID:         "mistral-7b", // target is llama-3-8b
				TokenizerID:     targetFence.TokenizerID,
				AttentionRegime: targetFence.AttentionRegime,
				Dtype:           targetFence.Dtype,
				QuantMode:       targetFence.QuantMode,
				RoPEScheme:      targetFence.RoPEScheme,
			},
			wantReason: "model mismatch",
		},
		{
			name: "mismatched rope scheme",
			candFence: RACCompatibilityFence{
				ModelID:         targetFence.ModelID,
				TokenizerID:     targetFence.TokenizerID,
				AttentionRegime: targetFence.AttentionRegime,
				Dtype:           targetFence.Dtype,
				QuantMode:       targetFence.QuantMode,
				RoPEScheme:      "yarn_32k", // target is default
			},
			wantReason: "rope scheme mismatch",
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			plan := &ReusePlan{
				PromptLen: 50,
				PrefixLen: 0,
				Spans: []PlanSpan{
					{
						Start:       0,
						End:         50,
						Action:      ActionDirectReuse,
						CandidateID: "cand-test",
						SourceStart: 0,
						SourceEnd:   50,
					},
				},
			}

			candidateFences := map[string]RACCompatibilityFence{
				"cand-test": tc.candFence,
			}

			res, err := MaterializePlan(plan, targetFence, candidateFences)
			if err != nil {
				t.Fatalf("MaterializePlan failed: %v", err)
			}

			if !res.FailsafeRecompute {
				t.Errorf("FailsafeRecompute=false, want true on regime mismatch")
			}
			if res.DirectAttachedTokens != 0 {
				t.Errorf("DirectAttachedTokens=%d, want 0", res.DirectAttachedTokens)
			}
			if res.RepairedTokens != 0 {
				t.Errorf("RepairedTokens=%d, want 0", res.RepairedTokens)
			}
			if res.RecomputedTokens != 50 {
				t.Errorf("RecomputedTokens=%d, want 50", res.RecomputedTokens)
			}

			action := res.Actions[0]
			if action.Mode != MatFullRecompute {
				t.Errorf("action mode=%s, want %s", action.Mode, MatFullRecompute)
			}
			if action.ShiftTokens != 0 {
				t.Errorf("action ShiftTokens=%d, want 0", action.ShiftTokens)
			}
			if action.RequiresReRoPE {
				t.Errorf("action RequiresReRoPE=true, want false")
			}
			if !strings.Contains(action.Reason, tc.wantReason) {
				t.Errorf("action reason %q does not contain %q", action.Reason, tc.wantReason)
			}
		})
	}
}

// TestMaterialize_SWAWindowRecompute witnesses that when the attention regime is sliding
// window (SWA) and shift crosses a boundary, it falls back to MatFullRecompute.
func TestMaterialize_SWAWindowRecompute(t *testing.T) {
	swaFence := RACCompatibilityFence{
		ModelID:         "mistral-7b",
		TokenizerID:     "mistral-tok",
		AttentionRegime: "swa_window_4096",
		Dtype:           "fp16",
		QuantMode:       "none",
		RoPEScheme:      "default",
	}

	candidateFences := map[string]RACCompatibilityFence{
		"cand-swa-aligned": swaFence,
		"cand-swa-moved":   swaFence,
	}

	plan := &ReusePlan{
		PromptLen: 100,
		PrefixLen: 0,
		Spans: []PlanSpan{
			{
				Start:       0,
				End:         40,
				Action:      ActionDirectReuse,
				CandidateID: "cand-swa-aligned",
				SourceStart: 0,
				SourceEnd:   40, // aligned: Start == SourceStart
			},
			{
				Start:       40,
				End:         80,
				Action:      ActionDirectReuse,
				CandidateID: "cand-swa-moved",
				SourceStart: 10,
				SourceEnd:   50, // moved: Start (40) != SourceStart (10), crosses boundary
			},
			{
				Start:  80,
				End:    100,
				Action: ActionCompute,
			},
		},
	}

	res, err := MaterializePlan(plan, swaFence, candidateFences)
	if err != nil {
		t.Fatalf("MaterializePlan failed: %v", err)
	}

	// Action 0: Aligned SWA span attaches directly with zero shift
	a0 := res.Actions[0]
	if a0.Mode != MatDirectAttach {
		t.Errorf("action 0 (aligned) mode=%s, want %s", a0.Mode, MatDirectAttach)
	}
	if a0.ShiftTokens != 0 {
		t.Errorf("action 0 ShiftTokens=%d, want 0", a0.ShiftTokens)
	}
	if a0.RequiresReRoPE {
		t.Errorf("action 0 RequiresReRoPE=true, want false")
	}

	// Action 1: Moved SWA span must fall back to MatFullRecompute because SWA does not support shift
	a1 := res.Actions[1]
	if a1.Mode != MatFullRecompute {
		t.Errorf("action 1 (moved SWA) mode=%s, want %s", a1.Mode, MatFullRecompute)
	}
	if a1.ShiftTokens != 0 {
		t.Errorf("action 1 ShiftTokens=%d, want 0", a1.ShiftTokens)
	}
	if a1.RequiresReRoPE {
		t.Errorf("action 1 RequiresReRoPE=true, want false")
	}
	if !strings.Contains(a1.Reason, "regime does not support positional shift") {
		t.Errorf("action 1 reason %q does not mention shift incompatibility", a1.Reason)
	}

	if !res.FailsafeRecompute {
		t.Errorf("FailsafeRecompute=false, want true due to SWA shift fallback")
	}
	if res.DirectAttachedTokens != 40 {
		t.Errorf("DirectAttachedTokens=%d, want 40", res.DirectAttachedTokens)
	}
	if res.RepairedTokens != 0 {
		t.Errorf("RepairedTokens=%d, want 0", res.RepairedTokens)
	}
	if res.RecomputedTokens != 60 {
		t.Errorf("RecomputedTokens=%d, want 60 (40 moved SWA + 20 compute)", res.RecomputedTokens)
	}
}

// TestMaterialize_FullRecomputeGaps witnesses that compute gaps are marked MatFullRecompute.
func TestMaterialize_FullRecomputeGaps(t *testing.T) {
	targetFence := sampleTargetFence()

	candidateFences := map[string]RACCompatibilityFence{
		"cand-aligned": targetFence,
		"cand-moved":   targetFence,
	}

	plan := &ReusePlan{
		PromptLen: 100,
		PrefixLen: 10,
		Spans: []PlanSpan{
			{
				Start:       0,
				End:         10,
				Action:      ActionDirectReuse,
				CandidateID: "", // prefix
				SourceStart: 0,
				SourceEnd:   10,
			},
			{
				Start:  10,
				End:    30,
				Action: ActionCompute, // gap 1: [10, 30)
			},
			{
				Start:       30,
				End:         60,
				Action:      ActionDirectReuse,
				CandidateID: "cand-aligned",
				SourceStart: 30,
				SourceEnd:   60,
			},
			{
				Start:  60,
				End:    80,
				Action: ActionCompute, // gap 2: [60, 80)
			},
			{
				Start:       80,
				End:         100,
				Action:      ActionDirectReuse,
				CandidateID: "cand-moved",
				SourceStart: 60,
				SourceEnd:   80, // moved: Start=80, SourceStart=60 -> shift=+20
			},
		},
	}

	res, err := MaterializePlan(plan, targetFence, candidateFences)
	if err != nil {
		t.Fatalf("MaterializePlan failed: %v", err)
	}

	if res.FailsafeRecompute {
		t.Errorf("FailsafeRecompute=true, want false when all candidates succeed")
	}

	// Gap 1 (action 1)
	a1 := res.Actions[1]
	if a1.Mode != MatFullRecompute {
		t.Errorf("action 1 mode=%s, want %s", a1.Mode, MatFullRecompute)
	}
	if a1.EstimatedTokens != 20 {
		t.Errorf("action 1 EstimatedTokens=%d, want 20", a1.EstimatedTokens)
	}
	if a1.ShiftTokens != 0 {
		t.Errorf("action 1 ShiftTokens=%d, want 0", a1.ShiftTokens)
	}
	if a1.RequiresReRoPE {
		t.Errorf("action 1 RequiresReRoPE=true, want false")
	}

	// Gap 2 (action 3)
	a3 := res.Actions[3]
	if a3.Mode != MatFullRecompute {
		t.Errorf("action 3 mode=%s, want %s", a3.Mode, MatFullRecompute)
	}
	if a3.EstimatedTokens != 20 {
		t.Errorf("action 3 EstimatedTokens=%d, want 20", a3.EstimatedTokens)
	}
	if a3.ShiftTokens != 0 {
		t.Errorf("action 3 ShiftTokens=%d, want 0", a3.ShiftTokens)
	}
	if a3.RequiresReRoPE {
		t.Errorf("action 3 RequiresReRoPE=true, want false")
	}

	// Direct attached: prefix 10 + aligned 30 = 40
	if res.DirectAttachedTokens != 40 {
		t.Errorf("DirectAttachedTokens=%d, want 40", res.DirectAttachedTokens)
	}
	// Repaired: moved 20
	if res.RepairedTokens != 20 {
		t.Errorf("RepairedTokens=%d, want 20", res.RepairedTokens)
	}
	// Recomputed: gap1 (20) + gap2 (20) = 40
	if res.RecomputedTokens != 40 {
		t.Errorf("RecomputedTokens=%d, want 40", res.RecomputedTokens)
	}
	if res.DirectAttachedTokens+res.RepairedTokens+res.RecomputedTokens != 100 {
		t.Errorf("total token sum=%d, want 100", res.DirectAttachedTokens+res.RepairedTokens+res.RecomputedTokens)
	}
}

// TestMaterialize_ValidationErrors tests error handling for invalid arguments.
func TestMaterialize_ValidationErrors(t *testing.T) {
	targetFence := sampleTargetFence()

	// Nil plan
	if _, err := MaterializePlan(nil, targetFence, nil); err == nil {
		t.Errorf("expected error for nil plan, got nil")
	}

	// Missing ModelID in target fence
	emptyFence := targetFence
	emptyFence.ModelID = ""
	plan := &ReusePlan{PromptLen: 10, Spans: []PlanSpan{{Start: 0, End: 10, Action: ActionCompute}}}
	if _, err := MaterializePlan(plan, emptyFence, nil); err == nil {
		t.Errorf("expected error for empty ModelID in target fence, got nil")
	}

	// Negative MaxContextLen in target fence
	negFence := targetFence
	negFence.MaxContextLen = -1
	if _, err := MaterializePlan(plan, negFence, nil); err == nil {
		t.Errorf("expected error for negative MaxContextLen, got nil")
	}

	// Negative MaxContextLen in candidate fence
	badCandFence := targetFence
	badCandFence.MaxContextLen = -1
	candMap := map[string]RACCompatibilityFence{"c1": badCandFence}
	if _, err := MaterializePlan(plan, targetFence, candMap); err == nil {
		t.Errorf("expected error for negative MaxContextLen in candidate, got nil")
	}
}

// TestRACCompatibilityFence_MatchAndShift tests the fence matching and positional shift compatibility methods.
func TestRACCompatibilityFence_MatchAndShift(t *testing.T) {
	base := sampleTargetFence()

	// Exact match
	if ok, reason := base.Match(base); !ok || reason != "" {
		t.Errorf("identical fence should match, got ok=%v, reason=%q", ok, reason)
	}

	// Causal attention with default RoPE is shift compatible
	if !base.CompatibleWithShift(base) {
		t.Errorf("causal regime with default RoPE should be CompatibleWithShift")
	}

	// SWA is not shift compatible
	swaFence := base
	swaFence.AttentionRegime = "swa_window_4096"
	if swaFence.CompatibleWithShift(swaFence) {
		t.Errorf("SWA regime should not be CompatibleWithShift")
	}

	// Absolute RoPE is not shift compatible
	absFence := base
	absFence.RoPEScheme = "absolute"
	if absFence.CompatibleWithShift(absFence) {
		t.Errorf("absolute RoPE scheme should not be CompatibleWithShift")
	}

	// ALiBi is not shift compatible
	alibiFence := base
	alibiFence.RoPEScheme = "alibi"
	if alibiFence.CompatibleWithShift(alibiFence) {
		t.Errorf("alibi RoPE scheme should not be CompatibleWithShift")
	}

	// Unsupported attention regime fails match
	unsupFence := base
	unsupFence.AttentionRegime = "unsupported"
	if ok, _ := unsupFence.Match(base); ok {
		t.Errorf("unsupported regime should not match")
	}
	if unsupFence.CompatibleWithShift(unsupFence) {
		t.Errorf("unsupported regime should not be CompatibleWithShift")
	}
}
