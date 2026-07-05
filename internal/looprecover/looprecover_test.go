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

// reasonOf returns the reason the planner gave the run with id (or "").
func reasonOf(r Result, id string) string {
	for _, x := range r.Runs {
		if x.RunID == id {
			return x.Reason
		}
	}
	return ""
}

// inRecover reports whether id is on the actionable recovery worklist.
func inRecover(r Result, id string) bool {
	for _, x := range r.Recover {
		if x == id {
			return true
		}
	}
	return false
}

// TestProbePidReuseDefense witnesses the pure primitive on its own: a live pid whose start-time
// identity matches is ALIVE; a gone pid is DEAD; a live pid whose start identity DIFFERS is a
// reused pid — a different process — so the recorded worker is DEAD, not fooled by the reuse. A
// zero pid or nil prober is UNKNOWN (staleness must decide).
func TestProbePidReuseDefense(t *testing.T) {
	aliveSame := Liveness(func(int) (string, bool) { return "boot-A", true })
	gone := Liveness(func(int) (string, bool) { return "", false })
	reused := Liveness(func(int) (string, bool) { return "boot-B", true }) // same pid, different boot

	if got := Probe(4242, "boot-A", aliveSame); got != ProbeAlive {
		t.Errorf("live pid, matching start = %q, want alive", got)
	}
	if got := Probe(4242, "boot-A", gone); got != ProbeDead {
		t.Errorf("gone pid = %q, want dead", got)
	}
	if got := Probe(4242, "boot-A", reused); got != ProbeDead {
		t.Errorf("reused pid (start differs) = %q, want dead (not fooled)", got)
	}
	if got := Probe(0, "boot-A", aliveSame); got != ProbeUnknown {
		t.Errorf("no pid = %q, want unknown", got)
	}
	if got := Probe(4242, "boot-A", nil); got != ProbeUnknown {
		t.Errorf("no prober = %q, want unknown", got)
	}
}

// TestRecoverProbeClassifiesLiveSilentVsDead is the issue witness end-to-end through the leaf:
// recover on a fixture with a live decoy pid classifies the silent-past-stale run ALIVE_SILENT
// and REFUSES re-dispatch; with the process gone it is DEAD (orphaned, on the worklist); and a
// pid-reuse fixture (same pid, different start time) is not fooled — it too is DEAD.
func TestRecoverProbeClassifiesLiveSilentVsDead(t *testing.T) {
	// A started run, silent well past the stale window, carrying a recorded worker identity.
	fixture := func(id string) RunFact {
		f := started(id, 5000) // silent past a 600s window
		f.WorkerPID, f.WorkerStart = 4242, "boot-A"
		return f
	}
	stale := int64(600)

	// Live decoy: the pid is held by the SAME process (matching start) — ALIVE_SILENT, refused.
	liveDecoy := Liveness(func(int) (string, bool) { return "boot-A", true })
	fLive := fixture("run").ApplyProbe(ProbeRun(fixture("run"), liveDecoy))
	rLive := Plan(Input{NowUnix: now, StaleSeconds: stale, Runs: []RunFact{fLive}})
	if dispoOf(rLive, "run") != DispRunning || reasonOf(rLive, "run") != ReasonAliveSilent {
		t.Errorf("live decoy = %q/%q, want running/alive_silent", dispoOf(rLive, "run"), reasonOf(rLive, "run"))
	}
	if inRecover(rLive, "run") {
		t.Error("a confirmed-alive worker must be REFUSED re-dispatch (not on the worklist)")
	}

	// Process gone: the pid is not held — DEAD, orphaned, on the worklist for re-dispatch.
	goneDecoy := Liveness(func(int) (string, bool) { return "", false })
	fGone := fixture("run").ApplyProbe(ProbeRun(fixture("run"), goneDecoy))
	rGone := Plan(Input{NowUnix: now, StaleSeconds: stale, Runs: []RunFact{fGone}})
	if dispoOf(rGone, "run") != DispOrphaned || reasonOf(rGone, "run") != ReasonWorkerDead {
		t.Errorf("gone decoy = %q/%q, want orphaned/worker_dead", dispoOf(rGone, "run"), reasonOf(rGone, "run"))
	}
	if !inRecover(rGone, "run") {
		t.Error("a confirmed-dead worker must be re-dispatched (on the worklist)")
	}

	// Pid reuse: the pid is held, but by a process with a DIFFERENT start — the worker is gone.
	reuseDecoy := Liveness(func(int) (string, bool) { return "boot-B", true })
	fReuse := fixture("run").ApplyProbe(ProbeRun(fixture("run"), reuseDecoy))
	rReuse := Plan(Input{NowUnix: now, StaleSeconds: stale, Runs: []RunFact{fReuse}})
	if dispoOf(rReuse, "run") != DispOrphaned || reasonOf(rReuse, "run") != ReasonWorkerDead {
		t.Errorf("pid-reuse decoy = %q/%q, want orphaned/worker_dead (not fooled)", dispoOf(rReuse, "run"), reasonOf(rReuse, "run"))
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
