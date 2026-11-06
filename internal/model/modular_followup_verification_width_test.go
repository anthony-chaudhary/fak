package model

import (
	"testing"
	"time"
)

// These tests are clean-room study artifacts for the Modular OSS follow-up
// (candidate A: throughput-scored verification widths). They are written purely
// from the paraphrased behavioral contract under study and from this package's
// own governor source; no upstream implementation bytes were consulted or
// copied. The contract clauses exercised:
//
//  1. score each width by accepted-token yield divided by step time;
//  2. keep per-batch-size cost buckets;
//  3. periodically probe the widest width;
//  4. switch only past a margin.
//
// The live decision function under test is
// Qwen38MTPAdaptiveDepthGovernor.ObserveStep (qwen38_mtp_adaptive.go).

// TestVerificationWidthNoPeriodicWidestProbe is a disconfirming replay of
// clause 3. It builds a supported single-stream workload whose per-width
// yield surface makes width 4 strictly best while widths 2 and 3 sit inside
// the governor's deadband [1.05, 1.20):
//
//	width 1: speedup ~0.95 (net-negative)
//	width 2: speedup  1.10 (deadband-stable)
//	width 3: speedup  1.10 (deadband-stable)
//	width 4: speedup ~1.45 (best)
//
// A cold-started governor observes only what it executes. At width 2 every
// observation is deadband-stable, so the hysteresis machine holds forever and
// never samples widths 3 or 4. The replay asserts that after 60 steady-state
// steps the governor still sits at width 2 with zero probes executed, while an
// identical world reached through a transient favorable regime parks at width
// 4. Same world, same steady state, different settled width: the gap is the
// absent periodic widest-width probe, not a broken ramp mechanism.
func TestVerificationWidthNoPeriodicWidestProbe(t *testing.T) {
	const (
		targetLatNS = int64(11_000_000) // one serial target step, ns
		steadySteps = 60
		warmSteps   = 6 // favorable transient: ramps 2 -> 3 -> 4 (hysteresis 3 + 3)
	)
	// Steady-world step latency per width (ns), chosen so the measured-path
	// speedup (TargetLatency*(A+1)/StepLatency with A = depth-1) lands as above.
	stepLatNS := map[int]int64{
		1: 11_578_947, // 11e6*1/11_578_947 ~= 0.95 < 1.00
		2: 20_000_000, // 11e6*2/20e6    = 1.10 exactly
		3: 30_000_000, // 11e6*3/30e6    = 1.10 exactly
		4: 30_344_827, // 11e6*4/30_344_827 ~= 1.45
	}
	// Transient favorable regime: every width scores ~1.45.
	favorableLatNS := map[int]int64{
		2: 15_172_413, // 11e6*2/15_172_413 ~= 1.45
		3: 22_758_620, // 11e6*3/22_758_620 ~= 1.45
		4: 30_344_827, // 11e6*4/30_344_827 ~= 1.45
	}
	newGov := func() *Qwen38MTPAdaptiveDepthGovernor {
		gov, err := NewQwen38MTPAdaptiveDepthGovernor(DefaultQwen38AdaptiveConfig())
		if err != nil {
			t.Fatalf("NewQwen38MTPAdaptiveDepthGovernor: %v", err)
		}
		return gov
	}
	obsAt := func(depth int, latNS int64) Qwen38AdaptiveStepObservation {
		return Qwen38AdaptiveStepObservation{
			ProposedTokens: depth,
			AcceptedTokens: depth - 1,
			StepLatency:    time.Duration(latNS),
			TargetLatency:  time.Duration(targetLatNS),
		}
	}

	// Trajectory A: cold start straight into the steady world.
	govA := newGov()
	for i := 0; i < steadySteps; i++ {
		d := govA.CurrentDepth()
		if _, _, err := govA.ObserveStep(obsAt(d, stepLatNS[d])); err != nil {
			t.Fatalf("trajectory A step %d: %v", i+1, err)
		}
	}
	if got := govA.CurrentDepth(); got != 2 {
		t.Fatalf("trajectory A final depth = %d, want 2 (deadband lock at cold start)", got)
	}
	recsA := govA.Trace()
	if len(recsA) != steadySteps {
		t.Fatalf("trajectory A trace length = %d, want %d", len(recsA), steadySteps)
	}
	for i, r := range recsA {
		if r.DepthBefore != 2 || r.DepthAfter != 2 {
			t.Fatalf("trajectory A step %d visited depth %d->%d; widths 3 and 4 were never sampled", i+1, r.DepthBefore, r.DepthAfter)
		}
		if r.Action != "hold" {
			t.Fatalf("trajectory A step %d action = %q, want hold in deadband", i+1, r.Action)
		}
	}
	if rcpt := govA.Receipt(); rcpt.ProbesExecuted != 0 {
		t.Fatalf("trajectory A probes executed = %d, want 0 (no widest-width probing while active)", rcpt.ProbesExecuted)
	} else if err := rcpt.Validate(); err != nil {
		t.Fatalf("trajectory A receipt validation: %v", err)
	}

	// Trajectory B: six favorable steps first (the ramp machinery works), then
	// the identical steady world.
	govB := newGov()
	for i := 0; i < warmSteps; i++ {
		d := govB.CurrentDepth()
		if _, _, err := govB.ObserveStep(obsAt(d, favorableLatNS[d])); err != nil {
			t.Fatalf("trajectory B warm step %d: %v", i+1, err)
		}
	}
	if got := govB.CurrentDepth(); got != 4 {
		t.Fatalf("trajectory B post-warm depth = %d, want 4 (ramp to MaxDepth is reachable)", got)
	}
	for i := 0; i < steadySteps-warmSteps; i++ {
		d := govB.CurrentDepth()
		if _, _, err := govB.ObserveStep(obsAt(d, stepLatNS[d])); err != nil {
			t.Fatalf("trajectory B steady step %d: %v", i+1, err)
		}
	}
	if got := govB.CurrentDepth(); got != 4 {
		t.Fatalf("trajectory B final depth = %d, want 4 (steady at best width)", got)
	}
	ramps := 0
	for _, r := range govB.Trace() {
		if r.Action == "ramp_up" {
			ramps++
		}
	}
	if ramps != 2 {
		t.Fatalf("trajectory B ramp_up actions = %d, want 2 (2->3 and 3->4)", ramps)
	}
	if err := govB.Receipt().Validate(); err != nil {
		t.Fatalf("trajectory B receipt validation: %v", err)
	}

	// Disconfirmation: identical steady world, two settled widths. Trajectory A
	// yields ~1.10x per step at width 2 while width 4 would have yielded ~1.45x;
	// nothing in the active-state machine ever re-samples the widest width.
	t.Logf("steady world: width 2 settles at ~1.10x (locked), width 4 yields ~1.45x; cold-start governor never probed width 4")
}

// TestVerificationWidthScoreBlindToMeasuredLatencyWhenBaselineUnset is a
// disconfirming replay of clause 1 against the production wiring. No
// production path calls SetTargetLatency or SetStepCostFn on the coordinator,
// so obs.TargetLatency is zero and ComputeSpeedup falls back to the static
// analytic model (accepted+1)/(1+0.25*proposed), which ignores the measured
// StepLatency entirely. The replay shows two worlds with identical acceptance
// but a 10x difference in measured step cost produce bit-identical scores and
// identical depth trajectories, while arming the baseline flips the verdict to
// immediate target-only escape.
func TestVerificationWidthScoreBlindToMeasuredLatencyWhenBaselineUnset(t *testing.T) {
	newGov := func() *Qwen38MTPAdaptiveDepthGovernor {
		gov, err := NewQwen38MTPAdaptiveDepthGovernor(DefaultQwen38AdaptiveConfig())
		if err != nil {
			t.Fatalf("NewQwen38MTPAdaptiveDepthGovernor: %v", err)
		}
		return gov
	}
	// One round executed at the given depth: full-minus-one acceptance while
	// drafting, or zero draft proposals/acceptances after target-only escape.
	roundAt := func(depth int, targetLat, stepLat time.Duration) Qwen38AdaptiveStepObservation {
		return Qwen38AdaptiveStepObservation{
			ProposedTokens: depth,
			AcceptedTokens: max(0, depth-1),
			StepLatency:    stepLat,
			TargetLatency:  targetLat,
		}
	}
	runStream := func(gov *Qwen38MTPAdaptiveDepthGovernor, targetLat, stepLat time.Duration) []Qwen38AdaptiveTraceRecord {
		for i := 0; i < 6; i++ {
			d := gov.CurrentDepth()
			if _, _, err := gov.ObserveStep(roundAt(d, targetLat, stepLat)); err != nil {
				t.Fatalf("step %d: %v", i+1, err)
			}
		}
		return gov.Trace()
	}

	cheap := newGov()
	expensive := newGov()
	traceCheap := runStream(cheap, 0, 40*time.Millisecond)
	traceExpensive := runStream(expensive, 0, 400*time.Millisecond)

	if got := cheap.CurrentDepth(); got != 4 {
		t.Fatalf("cheap-world final depth = %d, want 4 (analytic score ramps to max)", got)
	}
	if got := expensive.CurrentDepth(); got != 4 {
		t.Fatalf("expensive-world final depth = %d, want 4: a 10x higher measured step cost is invisible without a target-latency baseline", got)
	}
	for i := range traceCheap {
		if traceCheap[i].Speedup != traceExpensive[i].Speedup {
			t.Fatalf("step %d speedup differs (%g vs %g) despite identical acceptance: baseline must be unset for both",
				i+1, traceCheap[i].Speedup, traceExpensive[i].Speedup)
		}
	}

	armed := newGov()
	traceArmed := runStream(armed, 10*time.Millisecond, 400*time.Millisecond)
	if got := armed.CurrentDepth(); got != 0 {
		t.Fatalf("armed expensive-world final depth = %d, want 0 (measured speedup 0.05 escapes to target-only)", got)
	}
	if traceArmed[0].Action != "escape" {
		t.Fatalf("armed expensive-world step 1 action = %q, want escape", traceArmed[0].Action)
	}
	for i, r := range traceArmed[1:] {
		if r.DepthBefore != 0 || r.ProposedTokens != 0 || r.AcceptedTokens != 0 {
			t.Fatalf("armed expensive-world step %d: want target-only with no draft tokens, got %+v", i+2, r)
		}
	}
	t.Logf("baseline unset: 40ms and 400ms step costs score identically (%.4f on the final round); baseline armed: 400ms step cost escapes to target-only", traceExpensive[5].Speedup)
}
