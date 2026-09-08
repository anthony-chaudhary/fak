package model

import (
	"context"
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

// TestMetalMTPAdaptiveDepthChangesWithPressure is the regression witness for Issue #12332:
// It minimally connects Qwen38MTPAdaptiveDepthGovernor to real Metal coordinator rounds,
// feeds deterministic acceptance and cost observations through real StepRound executions,
// proves bounded K changes with hysteresis, proves target-only escape when drafting is net-negative,
// and confirms typed fak-native downgrade and envelope isolation without external fallback.
func TestMetalMTPAdaptiveDepthChangesWithPressure(t *testing.T) {
	ctx := context.Background()
	m := qwen38HybridMTPEnabledSyntheticModel(t)
	prompt := []int{0, 1, 2}

	// Reference autoregressive baseline at temperature zero for bit-exact ground truth
	refSes := m.NewSession()
	t.Cleanup(refSes.Close)
	const maxSeq = 64
	wantTokens := refSes.Generate(prompt, maxSeq)
	if len(wantTokens) < 32 {
		t.Fatalf("expected >= 32 reference tokens, got %d", len(wantTokens))
	}

	// Configure coordinator with dynamic adaptive governor:
	// K in [1..4], cold start K=2, hysteresis=3 steps, probe_interval=4 steps
	adCfg := DefaultQwen38AdaptiveConfig()
	adCfg.ColdStartDepth = 2
	adCfg.MaxDepth = 4
	adCfg.HysteresisSteps = 3
	adCfg.ProbeInterval = 4
	adCfg.SpeedupThresholdHigh = 1.20
	adCfg.SpeedupThresholdLow = 1.05
	adCfg.TargetOnlyThreshold = 1.00

	specSes := m.NewSession()
	t.Cleanup(specSes.Close)

	coord, err := specSes.NewMetalMTPCoordinator(MetalMTPConfig{
		DraftDepth:            4,
		Adaptive:              true,
		AdaptiveConfig:        &adCfg,
		EnforceGreedyTripwire: true,
	})
	if err != nil {
		t.Fatalf("NewMetalMTPCoordinator failed: %v", err)
	}
	t.Cleanup(func() { _ = coord.Close() })

	// 1. Verify cold start initialization and typed fak-native envelope
	if coord.CurrentDraftDepth() != 2 {
		t.Fatalf("initial draft depth = %d, want cold start depth 2", coord.CurrentDraftDepth())
	}
	if coord.DowngradeReason() != Qwen38MTPEligible {
		t.Fatalf("initial downgrade reason = %q, want eligible", coord.DowngradeReason())
	}
	if coord.Engine() != Qwen38EngineMTP {
		t.Fatalf("initial engine = %q, want fak-native MTP", coord.Engine())
	}

	// Helper for an agreeing drafter that matches reference autoregressive decode
	agreeingDrafter := NewMTPProposalGeneratorWithFn(func(ctx context.Context, committed []int, maxDraft int) ([]int, error) {
		numGen := len(committed) - len(prompt)
		if numGen < 0 || numGen+maxDraft > len(wantTokens) {
			return nil, nil
		}
		toks := make([]int, maxDraft)
		copy(toks, wantTokens[numGen:numGen+maxDraft])
		return toks, nil
	})
	coord.SetDrafter(agreeingDrafter)

	boundary := specSes.Prefill(prompt)
	committed := append([]int(nil), prompt...)

	// 2. High acceptance rounds: prove hysteresis delay before ramp-up (K=2 -> K=3)
	// Steps 1 & 2: high speedup, but streak < 3 -> draft depth must hold at 2
	for round := 1; round <= 2; round++ {
		acc, bonus, nextLogits, rErr := coord.StepRound(ctx, committed, boundary)
		if rErr != nil {
			t.Fatalf("round %d failed: %v", round, rErr)
		}
		if len(acc) != 2 {
			t.Fatalf("round %d accepted = %d, want 2", round, len(acc))
		}
		committed = append(committed, acc...)
		committed = append(committed, bonus)
		boundary = nextLogits

		if coord.CurrentDraftDepth() != 2 {
			t.Fatalf("round %d depth = %d, want 2 (held by hysteresis streak %d/3)", round, coord.CurrentDraftDepth(), round)
		}
	}

	// Step 3: reaches hysteresis streak of 3 -> ramps up to depth 3
	acc, bonus, nextLogits, rErr := coord.StepRound(ctx, committed, boundary)
	if rErr != nil {
		t.Fatalf("round 3 failed: %v", rErr)
	}
	if len(acc) != 2 {
		t.Fatalf("round 3 accepted = %d, want 2", len(acc))
	}
	committed = append(committed, acc...)
	committed = append(committed, bonus)
	boundary = nextLogits

	if coord.CurrentDraftDepth() != 3 {
		t.Fatalf("round 3 depth = %d, want 3 (ramped after 3 high streak steps)", coord.CurrentDraftDepth())
	}

	// 3. High acceptance rounds at depth 3: prove hysteresis to MaxDepth (K=3 -> K=4)
	for round := 4; round <= 5; round++ {
		acc, bonus, nextLogits, rErr := coord.StepRound(ctx, committed, boundary)
		if rErr != nil {
			t.Fatalf("round %d failed: %v", round, rErr)
		}
		if len(acc) != 3 {
			t.Fatalf("round %d accepted = %d, want 3", round, len(acc))
		}
		committed = append(committed, acc...)
		committed = append(committed, bonus)
		boundary = nextLogits

		if coord.CurrentDraftDepth() != 3 {
			t.Fatalf("round %d depth = %d, want 3 (held by hysteresis)", round, coord.CurrentDraftDepth())
		}
	}

	// Step 6: reaches hysteresis streak of 3 -> ramps up to MaxDepth 4
	acc, bonus, nextLogits, rErr = coord.StepRound(ctx, committed, boundary)
	if rErr != nil {
		t.Fatalf("round 6 failed: %v", rErr)
	}
	if len(acc) != 3 {
		t.Fatalf("round 6 accepted = %d, want 3", len(acc))
	}
	committed = append(committed, acc...)
	committed = append(committed, bonus)
	boundary = nextLogits

	if coord.CurrentDraftDepth() != 4 {
		t.Fatalf("round 6 depth = %d, want 4 (ramped to MaxDepth)", coord.CurrentDraftDepth())
	}

	// Step 7: prove bounded K (cannot exceed MaxDepth = 4)
	acc, bonus, nextLogits, rErr = coord.StepRound(ctx, committed, boundary)
	if rErr != nil {
		t.Fatalf("round 7 failed: %v", rErr)
	}
	if len(acc) != 4 {
		t.Fatalf("round 7 accepted = %d, want 4", len(acc))
	}
	committed = append(committed, acc...)
	committed = append(committed, bonus)
	boundary = nextLogits

	if coord.CurrentDraftDepth() != 4 {
		t.Fatalf("round 7 depth = %d, want 4 (bounded at MaxDepth)", coord.CurrentDraftDepth())
	}

	// 4. Introduce adversarial pressure / net-negative drafting:
	// Propose divergent tokens guaranteed to fail verification (0 accepted, all 4 rolled back)
	adversarialDrafter := NewMTPProposalGeneratorWithFn(func(ctx context.Context, committed []int, maxDraft int) ([]int, error) {
		toks := make([]int, maxDraft)
		for i := range toks {
			toks[i] = 99990 + i
		}
		return toks, nil
	})
	coord.SetDrafter(adversarialDrafter)

	acc, bonus, nextLogits, rErr = coord.StepRound(ctx, committed, boundary)
	if rErr != nil {
		t.Fatalf("pressure round 8 failed: %v", rErr)
	}
	if len(acc) != 0 {
		t.Fatalf("pressure round 8 accepted = %d, want 0 (complete rollback)", len(acc))
	}
	committed = append(committed, bonus)
	boundary = nextLogits

	// Prove immediate target-only escape when drafting is net-negative
	if coord.CurrentDraftDepth() != 0 {
		t.Fatalf("after net-negative drafting, depth = %d, want 0 (target-only escape)", coord.CurrentDraftDepth())
	}
	if coord.DowngradeReason() != Qwen38MTPNetLatencyRegressed {
		t.Fatalf("downgrade reason = %q, want %q", coord.DowngradeReason(), Qwen38MTPNetLatencyRegressed)
	}
	if coord.Engine() != Qwen38EngineTargetDecode {
		t.Fatalf("engine = %q, want %q", coord.Engine(), Qwen38EngineTargetDecode)
	}
	if !coord.InFallback() {
		t.Fatalf("InFallback = false, want true in target-only escape")
	}

	// 5. Target-only execution: verify serial steps and probe interval counting (ProbeInterval = 4)
	// Steps 1..3 in target-only mode: must hold depth 0
	for step := 1; step <= 3; step++ {
		acc, bonus, nextLogits, rErr = coord.StepRound(ctx, committed, boundary)
		if rErr != nil {
			t.Fatalf("target-only step %d failed: %v", step, rErr)
		}
		if len(acc) != 1 || bonus != -1 {
			t.Fatalf("target-only step %d produced acc=%v, bonus=%d, want 1 serial token", step, acc, bonus)
		}
		committed = append(committed, acc...)
		boundary = nextLogits

		if coord.CurrentDraftDepth() != 0 {
			t.Fatalf("target-only step %d depth = %d, want 0", step, coord.CurrentDraftDepth())
		}
	}

	// Step 4 in target-only mode: probe interval (4) elapses, triggering depth 1 probe
	acc, bonus, nextLogits, rErr = coord.StepRound(ctx, committed, boundary)
	if rErr != nil {
		t.Fatalf("target-only step 4 failed: %v", rErr)
	}
	committed = append(committed, acc...)
	boundary = nextLogits

	if coord.CurrentDraftDepth() != 1 {
		t.Fatalf("after probe interval elapsed, depth = %d, want 1 (probing)", coord.CurrentDraftDepth())
	}
	if !coord.AdaptiveGovernor().IsProbing() {
		t.Fatalf("expected governor to be probing")
	}

	// 6. Execute probe step with agreeing conditions: drafting resumes at depth 1
	coord.SetDrafter(agreeingDrafter)
	acc, bonus, nextLogits, rErr = coord.StepRound(ctx, committed, boundary)
	if rErr != nil {
		t.Fatalf("probe execution round failed: %v", rErr)
	}
	if len(acc) != 1 {
		t.Fatalf("probe round accepted = %d, want 1", len(acc))
	}
	committed = append(committed, acc...)
	committed = append(committed, bonus)
	boundary = nextLogits

	if coord.CurrentDraftDepth() != 1 {
		t.Fatalf("after probe success, depth = %d, want 1 (drafting resumed)", coord.CurrentDraftDepth())
	}
	if coord.DowngradeReason() != Qwen38MTPEligible {
		t.Fatalf("after probe success, downgrade reason = %q, want eligible", coord.DowngradeReason())
	}
	if coord.Engine() != Qwen38EngineMTP {
		t.Fatalf("after probe success, engine = %q, want fak-native MTP", coord.Engine())
	}
	if coord.InFallback() {
		t.Fatalf("after probe success, InFallback = true, want false")
	}

	// 7. Verify deterministic cost model pressure injection via SetStepCostFn
	// Even with 100% acceptance, high simulated step latency triggers net-negative speedup escape
	coord.SetStepCostFn(func(proposed, accepted int, base Qwen38AdaptiveStepObservation) Qwen38AdaptiveStepObservation {
		// Simulate severe GPU compute contention: 200ms step vs 50ms target
		base.StepLatency = 200 * time.Millisecond
		base.TargetLatency = 50 * time.Millisecond
		return base
	})

	acc, bonus, nextLogits, rErr = coord.StepRound(ctx, committed, boundary)
	if rErr != nil {
		t.Fatalf("latency pressure round failed: %v", rErr)
	}
	committed = append(committed, acc...)
	committed = append(committed, bonus)
	boundary = nextLogits

	// Net speedup for 1 proposed, 1 accepted at 200ms/50ms = (2 * 50ms) / 200ms = 0.50 < 1.00 -> escape
	if coord.CurrentDraftDepth() != 0 {
		t.Fatalf("latency pressure should trigger target-only escape, got depth %d", coord.CurrentDraftDepth())
	}
	if coord.DowngradeReason() != Qwen38MTPNetLatencyRegressed {
		t.Fatalf("latency pressure downgrade reason = %q, want regressed", coord.DowngradeReason())
	}

	// 8. Receipt validation: verify deterministic audit trail & schema compliance
	receipt := coord.AdaptiveGovernor().Receipt()
	if err := receipt.Validate(); err != nil {
		t.Fatalf("adaptive receipt validation failed: %v", err)
	}
	if receipt.Engine != Qwen38EngineMTP {
		t.Fatalf("receipt engine = %q, want %q", receipt.Engine, Qwen38EngineMTP)
	}
	if receipt.TargetOnlyEscapes < 2 {
		t.Fatalf("receipt target-only escapes = %d, want >= 2", receipt.TargetOnlyEscapes)
	}
	if receipt.ProbesExecuted < 1 {
		t.Fatalf("receipt probes executed = %d, want >= 1", receipt.ProbesExecuted)
	}
	if receipt.NetNegativeSteps < 2 {
		t.Fatalf("receipt net negative steps = %d, want >= 2", receipt.NetNegativeSteps)
	}

	// 9. Verify generated tokens match ground truth sequence bit-exactly
	genTokens := committed[len(prompt):]
	for i, tok := range genTokens {
		if tok != wantTokens[i] {
			t.Fatalf("token %d mismatch: got %d, want %d", i, tok, wantTokens[i])
		}
	}
}
