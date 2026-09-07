package model

import (
	"testing"
	"time"
)

// TestQwen38MTP_Adaptive_ColdStartConvergence verifies that the governor starts at ColdStartDepth
// and smoothly initializes EMA metrics on the first step.
func TestQwen38MTP_Adaptive_ColdStartConvergence(t *testing.T) {
	cfg := DefaultQwen38AdaptiveConfig()
	cfg.ColdStartDepth = 2
	cfg.MaxDepth = 4

	gov, err := NewQwen38MTPAdaptiveDepthGovernor(cfg)
	if err != nil {
		t.Fatalf("failed to create governor: %v", err)
	}

	if gov.CurrentDepth() != 2 {
		t.Fatalf("initial depth = %d, want cold start depth 2", gov.CurrentDepth())
	}
	if gov.DowngradeReason() != Qwen38MTPEligible {
		t.Fatalf("initial downgrade reason = %q, want eligible", gov.DowngradeReason())
	}

	// First observation: moderate speedup within deadband [1.05, 1.20]
	step1 := Qwen38AdaptiveStepObservation{
		ProposedTokens: 2,
		AcceptedTokens: 1,
		StepLatency:    100 * time.Millisecond,
		TargetLatency:  60 * time.Millisecond,
		Speedup:        1.12, // within deadband
	}

	nextDepth, rec, err := gov.ObserveStep(step1)
	if err != nil {
		t.Fatalf("observe step error: %v", err)
	}

	if nextDepth != 2 {
		t.Fatalf("next depth = %d, want 2 (held in deadband)", nextDepth)
	}
	if rec.EMA_Speedup != 1.12 {
		t.Fatalf("first EMA speedup = %g, want 1.12", rec.EMA_Speedup)
	}
	if rec.EMA_AcceptanceRate != 0.5 {
		t.Fatalf("first EMA acceptance rate = %g, want 0.5", rec.EMA_AcceptanceRate)
	}
	if rec.NetNegative {
		t.Fatalf("rec.NetNegative is true for speedup 1.12")
	}
}

// TestQwen38MTP_Adaptive_HighAcceptanceRampUp asserts that sustained high acceptance and speedup
// ramps draft depth from cold start up to MaxDepth, honoring hysteresis confirmation windows.
func TestQwen38MTP_Adaptive_HighAcceptanceRampUp(t *testing.T) {
	cfg := DefaultQwen38AdaptiveConfig()
	cfg.ColdStartDepth = 2
	cfg.MaxDepth = 4
	cfg.HysteresisSteps = 3
	cfg.SpeedupThresholdHigh = 1.20

	gov, err := NewQwen38MTPAdaptiveDepthGovernor(cfg)
	if err != nil {
		t.Fatalf("failed to create governor: %v", err)
	}

	highStep := Qwen38AdaptiveStepObservation{
		ProposedTokens: 2,
		AcceptedTokens: 2,
		Speedup:        1.45,
	}

	// Steps 1 & 2: high speedup, but hysteresis streak < 3 -> should hold at depth 2
	for s := 1; s <= 2; s++ {
		d, rec, err := gov.ObserveStep(highStep)
		if err != nil {
			t.Fatalf("step %d error: %v", s, err)
		}
		if d != 2 {
			t.Fatalf("step %d depth = %d, want 2 (held by hysteresis)", s, d)
		}
		if rec.Action != "hold" {
			t.Fatalf("step %d action = %q, want hold", s, rec.Action)
		}
	}

	// Step 3: reaches hysteresis threshold 3 -> ramps up to depth 3
	d3, rec3, err := gov.ObserveStep(highStep)
	if err != nil {
		t.Fatalf("step 3 error: %v", err)
	}
	if d3 != 3 {
		t.Fatalf("step 3 depth = %d, want 3 (ramped up)", d3)
	}
	if rec3.Action != "ramp_up" {
		t.Fatalf("step 3 action = %q, want ramp_up", rec3.Action)
	}

	// High steps at depth 3: need 3 more high steps to reach depth 4
	highStep3 := Qwen38AdaptiveStepObservation{
		ProposedTokens: 3,
		AcceptedTokens: 3,
		Speedup:        1.60,
	}
	for s := 4; s <= 5; s++ {
		d, _, err := gov.ObserveStep(highStep3)
		if err != nil {
			t.Fatalf("step %d error: %v", s, err)
		}
		if d != 3 {
			t.Fatalf("step %d depth = %d, want 3", s, d)
		}
	}

	// Step 6: ramps up to MaxDepth = 4
	d6, rec6, err := gov.ObserveStep(highStep3)
	if err != nil {
		t.Fatalf("step 6 error: %v", err)
	}
	if d6 != 4 {
		t.Fatalf("step 6 depth = %d, want 4", d6)
	}
	if rec6.Action != "ramp_up" {
		t.Fatalf("step 6 action = %q, want ramp_up", rec6.Action)
	}

	// Additional steps at MaxDepth: must stay capped at 4
	highStep4 := Qwen38AdaptiveStepObservation{
		ProposedTokens: 4,
		AcceptedTokens: 4,
		Speedup:        1.70,
	}
	d7, rec7, err := gov.ObserveStep(highStep4)
	if err != nil {
		t.Fatalf("step 7 error: %v", err)
	}
	if d7 != 4 {
		t.Fatalf("step 7 depth = %d, want 4 (capped at max)", d7)
	}
	if rec7.Reason != "at_max_depth" {
		t.Fatalf("step 7 reason = %q, want at_max_depth", rec7.Reason)
	}
}

// TestQwen38MTP_Adaptive_TargetOnlyEscape_And_Probe asserts that when measured net speedup falls below 1.0,
// the governor escapes to depth 0 (disabling drafting), sets downgrade reason to net_latency_regressed,
// and probes depth 1 after ProbeInterval steps.
func TestQwen38MTP_Adaptive_TargetOnlyEscape_And_Probe(t *testing.T) {
	cfg := DefaultQwen38AdaptiveConfig()
	cfg.ColdStartDepth = 2
	cfg.ProbeInterval = 5
	cfg.TargetOnlyThreshold = 1.00

	gov, err := NewQwen38MTPAdaptiveDepthGovernor(cfg)
	if err != nil {
		t.Fatalf("failed to create governor: %v", err)
	}

	// Step 1: Good initial step to initialize
	_, _, err = gov.ObserveStep(Qwen38AdaptiveStepObservation{
		ProposedTokens: 2,
		AcceptedTokens: 1,
		Speedup:        1.10,
	})
	if err != nil {
		t.Fatalf("step 1 error: %v", err)
	}

	// Step 2: Severe regression (e.g. 0 accepted, heavy draft overhead) -> speedup 0.72 < 1.0
	regressedStep := Qwen38AdaptiveStepObservation{
		ProposedTokens: 2,
		AcceptedTokens: 0,
		StepLatency:    140 * time.Millisecond,
		TargetLatency:  50 * time.Millisecond,
		Speedup:        0.72,
	}

	d2, rec2, err := gov.ObserveStep(regressedStep)
	if err != nil {
		t.Fatalf("step 2 error: %v", err)
	}

	if d2 != 0 {
		t.Fatalf("step 2 depth = %d, want 0 (target-only escape)", d2)
	}
	if rec2.Action != "escape" {
		t.Fatalf("step 2 action = %q, want escape", rec2.Action)
	}
	if !rec2.NetNegative {
		t.Fatalf("rec2.NetNegative = false, want true for speedup 0.72")
	}
	if gov.DowngradeReason() != Qwen38MTPNetLatencyRegressed {
		t.Fatalf("downgrade reason = %q, want %q", gov.DowngradeReason(), Qwen38MTPNetLatencyRegressed)
	}

	// Target-only decode steps: steps 3..6 (4 steps in target-only mode)
	targetOnlyStep := Qwen38AdaptiveStepObservation{
		ProposedTokens: 0,
		AcceptedTokens: 0,
		Speedup:        1.0, // target-only baseline
	}

	for s := 3; s <= 6; s++ {
		d, rec, err := gov.ObserveStep(targetOnlyStep)
		if err != nil {
			t.Fatalf("target-only step %d error: %v", s, err)
		}
		if d != 0 {
			t.Fatalf("step %d depth = %d, want 0 (holding in target-only)", s, d)
		}
		if rec.Action != "hold" {
			t.Fatalf("step %d action = %q, want hold", s, rec.Action)
		}
	}

	// Step 7 (5th step in target-only): Probe interval (5) elapses -> probe depth 1
	d7, rec7, err := gov.ObserveStep(targetOnlyStep)
	if err != nil {
		t.Fatalf("step 7 error: %v", err)
	}
	if d7 != 1 {
		t.Fatalf("step 7 depth = %d, want 1 (probe depth 1)", d7)
	}
	if rec7.Action != "probe" {
		t.Fatalf("step 7 action = %q, want probe", rec7.Action)
	}

	// Probe test Case A: Probe step reports positive speedup (1.25) -> resume drafting!
	probeSuccessStep := Qwen38AdaptiveStepObservation{
		ProposedTokens: 1,
		AcceptedTokens: 1,
		Speedup:        1.25,
	}
	d8, rec8, err := gov.ObserveStep(probeSuccessStep)
	if err != nil {
		t.Fatalf("step 8 error: %v", err)
	}
	if d8 != 1 {
		t.Fatalf("step 8 depth = %d, want 1 (drafting resumed)", d8)
	}
	if rec8.Action != "resume" {
		t.Fatalf("step 8 action = %q, want resume", rec8.Action)
	}
	if gov.DowngradeReason() != Qwen38MTPEligible {
		t.Fatalf("after probe success, downgrade reason = %q, want eligible", gov.DowngradeReason())
	}
}

// TestQwen38MTP_Adaptive_HysteresisStability verifies that oscillating noise in observations
// does not trigger premature depth changes, guaranteeing stable depth execution.
func TestQwen38MTP_Adaptive_HysteresisStability(t *testing.T) {
	cfg := DefaultQwen38AdaptiveConfig()
	cfg.ColdStartDepth = 2
	cfg.MaxDepth = 4
	cfg.HysteresisSteps = 3
	cfg.SpeedupThresholdHigh = 1.25
	cfg.SpeedupThresholdLow = 1.05

	gov, err := NewQwen38MTPAdaptiveDepthGovernor(cfg)
	if err != nil {
		t.Fatalf("failed to create governor: %v", err)
	}

	// Alternating high and low steps: none reaches 3 consecutive steps
	alternating := []Qwen38AdaptiveStepObservation{
		{ProposedTokens: 2, AcceptedTokens: 2, Speedup: 1.35}, // high streak = 1
		{ProposedTokens: 2, AcceptedTokens: 1, Speedup: 1.02}, // low streak = 1, high streak reset
		{ProposedTokens: 2, AcceptedTokens: 2, Speedup: 1.30}, // high streak = 1, low streak reset
		{ProposedTokens: 2, AcceptedTokens: 1, Speedup: 1.15}, // deadband: streaks reset
		{ProposedTokens: 2, AcceptedTokens: 2, Speedup: 1.40}, // high streak = 1
		{ProposedTokens: 2, AcceptedTokens: 2, Speedup: 1.35}, // high streak = 2
		{ProposedTokens: 2, AcceptedTokens: 1, Speedup: 1.03}, // low streak = 1, high streak reset
	}

	for i, obs := range alternating {
		d, rec, err := gov.ObserveStep(obs)
		if err != nil {
			t.Fatalf("step %d error: %v", i+1, err)
		}
		if d != 2 {
			t.Fatalf("step %d depth = %d, want 2 (hysteresis prevented premature change)", i+1, d)
		}
		if rec.Action != "hold" {
			t.Fatalf("step %d action = %q, want hold", i+1, rec.Action)
		}
	}
}

// TestQwen38MTP_Adaptive_TraceLogging verifies that all step transitions are accurately captured,
// negative net performance is never hidden, and the emitted receipt passes schema validation.
func TestQwen38MTP_Adaptive_TraceLogging(t *testing.T) {
	cfg := DefaultQwen38AdaptiveConfig()
	cfg.ColdStartDepth = 2
	cfg.ProbeInterval = 3

	gov, err := NewQwen38MTPAdaptiveDepthGovernor(cfg)
	if err != nil {
		t.Fatalf("failed to create governor: %v", err)
	}

	steps := []Qwen38AdaptiveStepObservation{
		{ProposedTokens: 2, AcceptedTokens: 2, Speedup: 1.40},
		{ProposedTokens: 2, AcceptedTokens: 0, Speedup: 0.85}, // net negative!
		{ProposedTokens: 0, AcceptedTokens: 0, Speedup: 1.00},
		{ProposedTokens: 0, AcceptedTokens: 0, Speedup: 1.00},
		{ProposedTokens: 0, AcceptedTokens: 0, Speedup: 1.00}, // triggers probe
		{ProposedTokens: 1, AcceptedTokens: 1, Speedup: 1.30}, // probe success
	}

	for _, s := range steps {
		if _, _, err := gov.ObserveStep(s); err != nil {
			t.Fatalf("observe step error: %v", err)
		}
	}

	trace := gov.Trace()
	if len(trace) != len(steps) {
		t.Fatalf("trace length = %d, want %d", len(trace), len(steps))
	}

	// Verify that step 2 recorded net negative performance honestly
	rec2 := trace[1]
	if !rec2.NetNegative {
		t.Fatalf("trace[1].NetNegative = false, want true for speedup 0.85")
	}
	if rec2.Speedup != 0.85 {
		t.Fatalf("trace[1].Speedup = %g, want 0.85", rec2.Speedup)
	}
	if rec2.DepthAfter != 0 {
		t.Fatalf("trace[1].DepthAfter = %d, want 0", rec2.DepthAfter)
	}

	// Verify that Receipt summarizes the run accurately
	receipt := gov.Receipt()
	if err := receipt.Validate(); err != nil {
		t.Fatalf("receipt validation failed: %v", err)
	}
	if receipt.Engine != Qwen38EngineMTP {
		t.Fatalf("receipt engine = %q, want %q", receipt.Engine, Qwen38EngineMTP)
	}
	if receipt.NetNegativeSteps != 1 {
		t.Fatalf("receipt.NetNegativeSteps = %d, want 1", receipt.NetNegativeSteps)
	}
	if receipt.TargetOnlyEscapes != 1 {
		t.Fatalf("receipt.TargetOnlyEscapes = %d, want 1", receipt.TargetOnlyEscapes)
	}
	if receipt.ProbesExecuted != 1 {
		t.Fatalf("receipt.ProbesExecuted = %d, want 1", receipt.ProbesExecuted)
	}
	if receipt.TotalSteps != len(steps) {
		t.Fatalf("receipt.TotalSteps = %d, want %d", receipt.TotalSteps, len(steps))
	}
}
