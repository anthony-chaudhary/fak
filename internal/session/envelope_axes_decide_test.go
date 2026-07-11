package session

import (
	"testing"
	"time"
)

// envelope_axes_decide_test.go — issue #2762 (out-of-band operator control epic
// #2753): the per-axis draining witnesses for the three envelope axes the control
// route previously parsed and dropped. Each test states a low ceiling through the
// PARSED envelope (the exact value the control route applies), drives the session
// past it, and asserts the drain lands with the axis's closed reason and that the
// per-turn boundary gate (Decide) takes the stop — the "enforced boundary the
// loop drains at" done condition, per axis.

// TestDecideEnvelopeWallAxisDrainsAtCeiling: wall=45m applied live via
// SetWallClockLimit is enforced by the wall-clock boundary gate — within the
// envelope the run proceeds, past it the session drains with
// TIME_BUDGET_EXHAUSTED and the plain Decide gate agrees.
func TestDecideEnvelopeWallAxisDrainsAtCeiling(t *testing.T) {
	env, err := ParseBudgetEnvelope("wall=45m")
	if err != nil {
		t.Fatalf("ParseBudgetEnvelope: %v", err)
	}
	tbl := NewTable()
	t0 := time.Unix(1_700_000_000, 0)
	if _, ok := tbl.SetWallClockLimit("s", env.WallClockLimit(), t0); !ok {
		t.Fatal("SetWallClockLimit refused a live session")
	}

	if v := tbl.DecideTimeBudget("s", t0.Add(30*time.Minute)); !v.Proceed {
		t.Fatalf("within envelope = %+v, want proceed", v)
	}
	v := tbl.DecideTimeBudget("s", t0.Add(46*time.Minute))
	if v.Proceed || !v.Stop || v.Reason != ReasonTimeBudgetExhausted {
		t.Fatalf("past envelope = {Proceed:%v Stop:%v Reason:%q}, want stop with %s",
			v.Proceed, v.Stop, v.Reason, ReasonTimeBudgetExhausted)
	}
	if v := tbl.Decide("s"); v.Proceed || !v.Stop || v.Reason != ReasonTimeBudgetExhausted {
		t.Fatalf("Decide after wall drain = %+v, want stop with %s", v, ReasonTimeBudgetExhausted)
	}
}

// TestDecideEnvelopeWallAxisPreservesElapsedOnReset: an operator re-setting the
// wall limit mid-run keeps the lineage's already-elapsed time — the ceiling
// moves, the clock never rewinds.
func TestDecideEnvelopeWallAxisPreservesElapsedOnReset(t *testing.T) {
	tbl := NewTable()
	t0 := time.Unix(1_700_000_000, 0)
	tbl.StartTimeBudget("s", time.Hour, t0)

	// 40 minutes in, the operator cuts the ceiling to 30 minutes: already past it.
	if _, ok := tbl.SetWallClockLimit("s", 30*time.Minute, t0.Add(40*time.Minute)); !ok {
		t.Fatal("SetWallClockLimit refused a live session")
	}
	v := tbl.DecideTimeBudget("s", t0.Add(40*time.Minute))
	if v.Proceed || v.Reason != ReasonTimeBudgetExhausted {
		t.Fatalf("cut-below-elapsed = %+v, want immediate %s (elapsed preserved)", v, ReasonTimeBudgetExhausted)
	}
}

// TestDecideEnvelopeSpendAxisDrainsAtCeiling: spend=$0.05 applied live via
// SessionBudget is a real priced ceiling — the turn that crosses it drains the
// session immediately with BUDGET_SPEND_EXHAUSTED and Decide takes the stop.
func TestDecideEnvelopeSpendAxisDrainsAtCeiling(t *testing.T) {
	env, err := ParseBudgetEnvelope("spend=$0.05")
	if err != nil {
		t.Fatalf("ParseBudgetEnvelope: %v", err)
	}
	tbl := NewTable()
	if _, ok := tbl.SetBudget("s", env.SessionBudget()); !ok {
		t.Fatal("SetBudget refused a live session")
	}

	st := tbl.DebitUsage("s", Usage{OutputTokens: 10, CostMicroCents: 2 * MicroCentsPerCent})
	if st.Run != Running {
		t.Fatalf("under ceiling Run = %v, want Running", st.Run)
	}
	st = tbl.DebitUsage("s", Usage{OutputTokens: 10, CostMicroCents: 4 * MicroCentsPerCent})
	if st.Run != Draining || st.Reason != ReasonBudgetSpend {
		t.Fatalf("past ceiling = {Run:%v Reason:%q}, want Draining/%s", st.Run, st.Reason, ReasonBudgetSpend)
	}
	if v := tbl.Decide("s"); v.Proceed || !v.Stop || v.Reason != ReasonBudgetSpend {
		t.Fatalf("Decide after spend drain = %+v, want stop with %s", v, ReasonBudgetSpend)
	}
}

// TestDecideEnvelopeThroughputAxisDrainsBelowFloor: min_throughput=10 applied
// live via ThroughputBudget is an enforced sustained-rate floor — once the
// observed window passes the grace period with a rate under the floor, the
// session drains with THROUGHPUT_BELOW_FLOOR and Decide takes the stop.
func TestDecideEnvelopeThroughputAxisDrainsBelowFloor(t *testing.T) {
	env, err := ParseBudgetEnvelope("throughput=40/s,min_throughput=10/s")
	if err != nil {
		t.Fatalf("ParseBudgetEnvelope: %v", err)
	}
	tbl := NewTable()
	if _, ok := tbl.SetThroughputBudget("s", env.ThroughputBudget()); !ok {
		t.Fatal("SetThroughputBudget refused a live session")
	}

	// 50 tokens over 2s is 25 tok/s — above the floor, and still inside the grace
	// window anyway: no drain.
	st := tbl.DebitUsage("s", Usage{OutputTokens: 50, DurationNanos: int64(2 * time.Second)})
	if st.Run != Running {
		t.Fatalf("within grace Run = %v, want Running", st.Run)
	}
	// A crawling bulk turn: cumulative 110 tokens over 20s = 5.5 tok/s sustained,
	// past the grace window and under the 10 tok/s floor — drains, no continuation
	// (a fresh window does not make a slow session faster).
	st = tbl.DebitUsage("s", Usage{OutputTokens: 60, DurationNanos: int64(18 * time.Second)})
	if st.Run != Draining || st.Reason != ReasonThroughputFloor {
		t.Fatalf("below floor = {Run:%v Reason:%q}, want Draining/%s", st.Run, st.Reason, ReasonThroughputFloor)
	}
	if st.ContinuationID != "" {
		t.Fatalf("throughput drain minted continuation %q, want none", st.ContinuationID)
	}
	if v := tbl.Decide("s"); v.Proceed || !v.Stop || v.Reason != ReasonThroughputFloor {
		t.Fatalf("Decide after throughput drain = %+v, want stop with %s", v, ReasonThroughputFloor)
	}
}

// TestDecideEnvelopeThroughputAxisSoftExpectedNeverDrains: an expected-only
// envelope (throughput= with no min_throughput=) is the #1585 soft pace-shaping
// reference — it must never drain a session, however slow, and a session at a
// healthy sustained rate never trips a configured floor.
func TestDecideEnvelopeThroughputAxisSoftExpectedNeverDrains(t *testing.T) {
	env, err := ParseBudgetEnvelope("throughput=40/s")
	if err != nil {
		t.Fatalf("ParseBudgetEnvelope: %v", err)
	}
	tbl := NewTable()
	tbl.SetThroughputBudget("slow", env.ThroughputBudget())
	st := tbl.DebitUsage("slow", Usage{OutputTokens: 1, DurationNanos: int64(time.Minute)})
	if st.Run != Running || st.Reason != "" {
		t.Fatalf("expected-only axis = {Run:%v Reason:%q}, want Running (soft reference never drains)", st.Run, st.Reason)
	}

	floor, err := ParseBudgetEnvelope("min_throughput=10/s")
	if err != nil {
		t.Fatalf("ParseBudgetEnvelope: %v", err)
	}
	tbl.SetThroughputBudget("healthy", floor.ThroughputBudget())
	st = tbl.DebitUsage("healthy", Usage{OutputTokens: 600, DurationNanos: int64(20 * time.Second)})
	if st.Run != Running {
		t.Fatalf("healthy 30 tok/s under a 10 tok/s floor drained: %+v", st)
	}
	if v := tbl.Decide("healthy"); !v.Proceed {
		t.Fatalf("Decide for healthy session = %+v, want proceed", v)
	}
}
