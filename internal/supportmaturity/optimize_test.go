package supportmaturity

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/rsiloop"
	"github.com/anthony-chaudhary/fak/internal/shipgate"
)

// oneStepHarness builds a real rsiloop.Harness with a SINGLE candidate measuring `metric`
// against `baseline` (higher-is-better), with the suite-green and truth-clean signals set.
// One candidate per run models one optimize step toward the next rung, so the run's terminal
// verdict (rsiloop.Result.Final) IS the run-level keep-bit. The metrics are scripted exactly
// as a real worktree probe would report them — shipgate, not the test, decides keep/revert.
func oneStepHarness(metric, baseline float64, green, clean bool) rsiloop.Harness {
	return rsiloop.Harness{
		MetricName:      "tok_per_s",
		LowerBetter:     false, // a higher throughput is the optimization gain
		BaselineRefName: "test-ref",
		BaselineMetric:  func() (float64, string, error) { return baseline, "deadbeef0000", nil },
		Candidates:      func() []rsiloop.Candidate { return []rsiloop.Candidate{{Label: "kernel-fastpath", Payload: 0}} },
		Measure: func(rsiloop.Candidate) (rsiloop.Measurement, error) {
			return rsiloop.Measurement{Metric: metric, SuiteGreen: green, TruthClean: clean}, nil
		},
	}
}

// runVerdict runs a real rsiloop over h and returns its terminal shipgate verdict — the
// non-forgeable keep-bit the loop derived, which PromoteOnRun consumes.
func runVerdict(t *testing.T, h rsiloop.Harness) shipgate.Decision {
	t.Helper()
	res, err := rsiloop.Run(h, nil, 3, 0)
	if err != nil {
		t.Fatalf("rsiloop.Run: %v", err)
	}
	return res.Final
}

// TestRsiloopRunAdvancesRungOnlyOnMeasuredGain is the #1253 witness: an rsiloop run advances
// a fixture cell's rung only on a non-author measured gain; a no-gain run leaves the rung
// unchanged. The rsiloop run is REAL (rsiloop.Run + shipgate.Evaluate decide the verdict);
// the cell's rung moves solely on that verdict, never on the test's say-so.
func TestRsiloopRunAdvancesRungOnlyOnMeasuredGain(t *testing.T) {
	cell := OptimizeCell{Name: "qwen3 x cuda", Current: M4Correct, Target: M6Parity}

	// A run that measures a strict throughput gain KEEPs -> the cell climbs exactly one rung.
	gain := runVerdict(t, oneStepHarness(220.0, 180.0, true, true))
	if gain != shipgate.KEEP {
		t.Fatalf("a strict measured gain must KEEP, got %s", gain)
	}
	kept := cell.PromoteOnRun(gain)
	if kept.Current != M5Optimized {
		t.Fatalf("kept gain must advance %s -> M5, got %s", M4Correct, kept.Current)
	}

	// A run that measures NO gain (candidate == baseline) REVERTs -> the rung is unchanged.
	noGain := runVerdict(t, oneStepHarness(180.0, 180.0, true, true))
	if noGain != shipgate.REVERT {
		t.Fatalf("a no-gain run must REVERT, got %s", noGain)
	}
	held := cell.PromoteOnRun(noGain)
	if held.Current != M4Correct {
		t.Fatalf("no-gain run must hold the rung at %s, got %s", M4Correct, held.Current)
	}

	// Anti-forgery: even a green+clean run that did not move the metric cannot promote —
	// the keep-bit, not correctness or cleanliness, is what climbs the ladder.
	stalled := runVerdict(t, oneStepHarness(180.0, 180.0, true, true))
	if cell.PromoteOnRun(stalled).Current != cell.Current {
		t.Fatalf("a green+clean but non-improving run must not promote")
	}
}

// TestLadderClimbsOneRungPerKeptRun walks M4 -> M5 -> M6 across successive kept runs and
// proves the climb stops at Target: a further kept gain at the horizon does not overshoot.
func TestLadderClimbsOneRungPerKeptRun(t *testing.T) {
	cell := OptimizeCell{Name: "gemma x metal", Current: M4Correct, Target: M6Parity}

	for _, want := range []Rung{M5Optimized, M6Parity} {
		v := runVerdict(t, oneStepHarness(2.0, 1.0, true, true)) // always a strict gain
		if v != shipgate.KEEP {
			t.Fatalf("expected KEEP climbing to %s, got %s", want, v)
		}
		cell = cell.PromoteOnRun(v)
		if cell.Current != want {
			t.Fatalf("climb landed on %s, want %s", cell.Current, want)
		}
	}

	// At Target there is no rung to climb: PromotionTarget refuses and even a kept run holds.
	if _, ok := cell.PromotionTarget(); ok {
		t.Fatalf("at Target %s PromotionTarget must report nothing to climb", cell.Target)
	}
	v := runVerdict(t, oneStepHarness(9.0, 1.0, true, true))
	if got := cell.PromoteOnRun(v).Current; got != M6Parity {
		t.Fatalf("a kept run at Target must not overshoot %s, got %s", M6Parity, got)
	}
}

// TestPromotionTarget pins the one-step-toward-Target rule directly, independent of any run.
func TestPromotionTarget(t *testing.T) {
	cases := []struct {
		current, target Rung
		wantNext        Rung
		wantOK          bool
	}{
		{M4Correct, M6Parity, M5Optimized, true},      // mid-climb: next is one rung up
		{M5Optimized, M6Parity, M6Parity, true},       // one below target: next IS target
		{M6Parity, M6Parity, M6Parity, false},         // at target: nothing to climb
		{M7BeyondSOTA, M6Parity, M7BeyondSOTA, false}, // above target: nothing to climb
	}
	for _, c := range cases {
		cell := OptimizeCell{Current: c.current, Target: c.target}
		next, ok := cell.PromotionTarget()
		if ok != c.wantOK || next != c.wantNext {
			t.Fatalf("PromotionTarget(cur=%s,tgt=%s) = (%s,%v), want (%s,%v)",
				c.current, c.target, next, ok, c.wantNext, c.wantOK)
		}
	}
}

// TestEscalateDoesNotPromote confirms only KEEP climbs: an ESCALATE verdict (the breaker
// handing a stalled loop to a human) is a no-gain outcome and must leave the rung put.
func TestEscalateDoesNotPromote(t *testing.T) {
	cell := OptimizeCell{Name: "phi x cpu", Current: M4Correct, Target: M5Optimized}
	if got := cell.PromoteOnRun(shipgate.ESCALATE).Current; got != M4Correct {
		t.Fatalf("ESCALATE must not promote, rung moved to %s", got)
	}
}
