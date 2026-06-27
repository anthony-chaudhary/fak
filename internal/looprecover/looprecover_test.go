package looprecover

import (
	"fmt"
	"testing"
)

const now = 1_000_000

// dispoOf returns the disposition the planner gave the run with id (or "").
func dispoOf(r Result, id string) Disposition {
	for _, x := range r.Runs {
		if x.RunID == id {
			return x.Disposition
		}
	}
	return ""
}

// started builds a started-but-unfinished run whose last event was ageSec ago.
func started(id string, ageSec int64) RunFact {
	return RunFact{RunID: id, LoopID: "L", Started: true, LastEventUnix: now - ageSec}
}

// TestWitnessedIsComplete: a witnessed run is done and never enters the worklist.
func TestWitnessedIsComplete(t *testing.T) {
	f := started("a", 99999)
	f.Witnessed = true
	r := Plan(Input{NowUnix: now, Runs: []RunFact{f}})
	if dispoOf(r, "a") != DispComplete || len(r.Recover) != 0 {
		t.Errorf("witnessed run = %q, recover %v, want complete/none", dispoOf(r, "a"), r.Recover)
	}
}

// TestOrphanedByStaleness: a started run silent past the stale window (worker liveness unknown)
// is orphaned and enters the worklist.
func TestOrphanedByStaleness(t *testing.T) {
	r := Plan(Input{NowUnix: now, StaleSeconds: 600, Runs: []RunFact{started("a", 700)}})
	if dispoOf(r, "a") != DispOrphaned {
		t.Fatalf("disposition = %q, want orphaned", dispoOf(r, "a"))
	}
	if len(r.Recover) != 1 || r.Recover[0] != "a" {
		t.Errorf("recover = %v, want [a]", r.Recover)
	}
}

// TestRecentStartedIsRunning: a started run with recent activity (within the stale window) is
// running, not orphaned.
func TestRecentStartedIsRunning(t *testing.T) {
	r := Plan(Input{NowUnix: now, StaleSeconds: 600, Runs: []RunFact{started("a", 60)}})
	if dispoOf(r, "a") != DispRunning {
		t.Errorf("disposition = %q, want running (recent)", dispoOf(r, "a"))
	}
}

// TestConfirmedLivenessOverridesStaleness: a confirmed-dead worker is orphaned at once however
// recent; a confirmed-live worker is never orphaned however ancient.
func TestConfirmedLivenessOverridesStaleness(t *testing.T) {
	dead := started("dead", 10) // very recent, but worker confirmed gone
	dead.WorkerKnown, dead.WorkerLive = true, false
	live := started("live", 999999) // ancient, but worker confirmed alive
	live.WorkerKnown, live.WorkerLive = true, true
	r := Plan(Input{NowUnix: now, StaleSeconds: 600, Runs: []RunFact{dead, live}})
	if dispoOf(r, "dead") != DispOrphaned {
		t.Errorf("confirmed-dead = %q, want orphaned even when recent", dispoOf(r, "dead"))
	}
	if dispoOf(r, "live") != DispRunning {
		t.Errorf("confirmed-live = %q, want running even when ancient", dispoOf(r, "live"))
	}
}

// TestEndedAndClaimedAreUnwitnessed: a run that ended or claimed done but was never witnessed is
// a re-verify candidate.
func TestEndedAndClaimedAreUnwitnessed(t *testing.T) {
	ended := started("e", 100)
	ended.Ended = true
	claimed := started("c", 100)
	claimed.Claimed = true
	r := Plan(Input{NowUnix: now, Runs: []RunFact{ended, claimed}})
	if dispoOf(r, "e") != DispUnwitnessed || dispoOf(r, "c") != DispUnwitnessed {
		t.Errorf("ended=%q claimed=%q, want both unwitnessed", dispoOf(r, "e"), dispoOf(r, "c"))
	}
	if r.UnwitnessedCount != 2 || len(r.Recover) != 2 {
		t.Errorf("unwitnessed=%d recover=%v, want 2/len2", r.UnwitnessedCount, r.Recover)
	}
}

// TestWitnessBeatsEnded: a run that both ended AND was witnessed is complete, not unwitnessed.
func TestWitnessBeatsEnded(t *testing.T) {
	f := started("a", 10)
	f.Ended, f.Witnessed = true, true
	if dispoOf(Plan(Input{NowUnix: now, Runs: []RunFact{f}}), "a") != DispComplete {
		t.Error("ended+witnessed should be complete")
	}
}

// TestFailedAndCanceledTerminal: failed and canceled runs are terminal failures, never recovery
// candidates (retry is the operator's call).
func TestFailedAndCanceledTerminal(t *testing.T) {
	f := started("f", 10)
	f.Failed = true
	c := started("c", 10)
	c.Canceled = true
	r := Plan(Input{NowUnix: now, Runs: []RunFact{f, c}})
	if dispoOf(r, "f") != DispFailed || dispoOf(r, "c") != DispFailed {
		t.Errorf("failed=%q canceled=%q, want both failed", dispoOf(r, "f"), dispoOf(r, "c"))
	}
	if len(r.Recover) != 0 {
		t.Errorf("recover = %v, want none (terminal failures are not auto-recovered)", r.Recover)
	}
}

// TestWorklistOrphanedFirstOldestFirst: the worklist orders orphaned before unwitnessed, and
// within a class the oldest (most stuck) first.
func TestWorklistOrphanedFirstOldestFirst(t *testing.T) {
	ended := started("unwit", 100)
	ended.Ended = true
	r := Plan(Input{NowUnix: now, StaleSeconds: 60, Runs: []RunFact{
		started("orphan-new", 200),  // orphaned, newer
		ended,                       // unwitnessed
		started("orphan-old", 5000), // orphaned, oldest
	}})
	want := []string{"orphan-old", "orphan-new", "unwit"}
	if fmt.Sprint(r.Recover) != fmt.Sprint(want) {
		t.Errorf("recover order = %v, want %v (orphaned oldest-first, then unwitnessed)", r.Recover, want)
	}
}

// TestNegativeStaleDisablesPresumption: a negative stale window disables the staleness
// presumption, so only a CONFIRMED-dead worker is orphaned.
func TestNegativeStaleDisablesPresumption(t *testing.T) {
	r := Plan(Input{NowUnix: now, StaleSeconds: -1, Runs: []RunFact{started("a", 999999)}})
	if dispoOf(r, "a") != DispRunning {
		t.Errorf("disposition = %q, want running (staleness disabled, worker unknown)", dispoOf(r, "a"))
	}
}

// TestDeterministicAndTotal: identical inputs give identical results; the empty input is defined.
func TestDeterministicAndTotal(t *testing.T) {
	in := Input{NowUnix: now, StaleSeconds: 600, Runs: []RunFact{
		started("a", 700), started("b", 100),
	}}
	if fmt.Sprint(Plan(in)) != fmt.Sprint(Plan(in)) {
		t.Error("Plan is not deterministic")
	}
	empty := Plan(Input{NowUnix: now})
	if len(empty.Runs) != 0 || len(empty.Recover) != 0 {
		t.Errorf("empty input = %+v, want empty", empty)
	}
}
