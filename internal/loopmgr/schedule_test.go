package loopmgr

import (
	"errors"
	"path/filepath"
	"testing"
	"time"
)

// fold helper: append events to a fresh ledger and return the single loop's fold.
func foldLoop(t *testing.T, loopID string, evs ...Event) LoopSnapshot {
	t.Helper()
	path := filepath.Join(t.TempDir(), "loops.jsonl")
	for _, ev := range evs {
		ev.LoopID = loopID
		if _, err := Append(path, ev); err != nil {
			t.Fatalf("Append(%s): %v", ev.Kind, err)
		}
	}
	st, err := SnapshotFile(path, time.Unix(0, 1).UTC())
	if err != nil {
		t.Fatalf("SnapshotFile: %v", err)
	}
	for _, loop := range st.Loops {
		if loop.LoopID == loopID {
			return loop
		}
	}
	return LoopSnapshot{LoopID: loopID}
}

func TestScheduleValidateRequiresNamedMissedPolicy(t *testing.T) {
	// The load-bearing rule: an empty missed-run policy is rejected, never
	// silently defaulted.
	bad := Schedule{JobID: "j", IntervalSeconds: 60}
	if err := bad.Validate(); err == nil {
		t.Fatalf("Validate accepted an empty missed_run policy; it must be explicit")
	}
	for _, p := range []MissedRunPolicy{"sometimes", "", "CATCH-UP"} {
		s := Schedule{JobID: "j", IntervalSeconds: 60, MissedRun: p}
		if ValidMissedRunPolicy(p) {
			t.Fatalf("ValidMissedRunPolicy(%q) = true, want false", p)
		}
		if err := s.Validate(); err == nil {
			t.Fatalf("Validate accepted unnamed policy %q", p)
		}
	}
	for _, p := range []MissedRunPolicy{MissedCatchUp, MissedSkip} {
		s := Schedule{JobID: "j", IntervalSeconds: 60, MissedRun: p}
		if err := s.Validate(); err != nil {
			t.Fatalf("Validate(%q) = %v, want ok", p, err)
		}
	}
	if err := (Schedule{JobID: "", IntervalSeconds: 60, MissedRun: MissedSkip}).Validate(); err == nil || !errors.Is(err, ErrInvalidSchedule) {
		t.Fatalf("Validate accepted an empty job id or did not wrap ErrInvalidSchedule: %v", err)
	}
	if err := (Schedule{JobID: "j", IntervalSeconds: 0, MissedRun: MissedSkip}).Validate(); err == nil || !errors.Is(err, ErrInvalidSchedule) {
		t.Fatalf("Validate accepted interval <= 0 with no at_unix_nano or did not wrap ErrInvalidSchedule: %v", err)
	}
}

// TestScheduleOverlapLockNoDuplicateFireOnWake is the #763 acceptance: a schedule
// fired across a simulated sleep, while the prior run is still in flight (a start
// with no matching end), produces NO second fire — exactly one run survives.
func TestScheduleOverlapLockNoDuplicateFireOnWake(t *testing.T) {
	s := Schedule{JobID: "issue-dispatch/default", IntervalSeconds: 60, MissedRun: MissedCatchUp}

	// A run has started but not ended (the box slept mid-run).
	loop := foldLoop(t, s.JobID,
		Event{Kind: EventFire, Source: "schedule"},
		Event{Kind: EventAdmit, RunID: "run-1", Status: StatusAdmitted},
		Event{Kind: EventStart, RunID: "run-1"},
	)
	if !runInFlight(loop) {
		t.Fatalf("expected a run in flight (started=%d ended=%d)", loop.Started, loop.Ended)
	}

	// Wake long after the boundary — many windows elapsed. Overlap-lock must win.
	wake := time.Unix(0, loop.LastEventUnixNano).Add(10 * time.Minute)
	d := s.Next(loop, wake, loop.LastEventUnixNano)
	if d.Fire {
		t.Fatalf("overlap-lock failed: fired while a prior run was in flight (%+v)", d)
	}
	if d.Reason != ReasonOverlapLock {
		t.Fatalf("reason = %q, want %q", d.Reason, ReasonOverlapLock)
	}

	// Once the prior run ends, the same wake-up fires exactly once.
	loop2 := foldLoop(t, s.JobID,
		Event{Kind: EventFire, Source: "schedule"},
		Event{Kind: EventAdmit, RunID: "run-1", Status: StatusAdmitted},
		Event{Kind: EventStart, RunID: "run-1"},
		Event{Kind: EventEnd, RunID: "run-1", Status: StatusClaimedDone},
	)
	if runInFlight(loop2) {
		t.Fatalf("run still in flight after end (started=%d ended=%d)", loop2.Started, loop2.Ended)
	}
	wake2 := time.Unix(0, loop2.LastEventUnixNano).Add(10 * time.Minute)
	d2 := s.Next(loop2, wake2, loop2.LastEventUnixNano)
	if !d2.Fire {
		t.Fatalf("expected a fire after the prior run ended (%+v)", d2)
	}
}

// TestScheduleMissedRunPolicyHonored is the #763 acceptance for the missed
// window: with several windows elapsed unobserved, skip drops them (no run) and
// catch-up settles them with exactly one run.
func TestScheduleMissedRunPolicyHonored(t *testing.T) {
	base := time.Unix(1_000_000, 0).UTC()
	// A loop that last ended at base, then nothing for a long time.
	mk := func(policy MissedRunPolicy) (Schedule, LoopSnapshot) {
		s := Schedule{JobID: "missed/job", IntervalSeconds: 60, MissedRun: policy}
		loop := LoopSnapshot{
			LoopID:            s.JobID,
			Started:           1,
			Ended:             1, // no run in flight
			LastEventUnixNano: base.UnixNano(),
		}
		return s, loop
	}

	// 5 minutes later: 5 windows elapsed, several missed.
	now := base.Add(5 * time.Minute)

	sSkip, loopSkip := mk(MissedSkip)
	dSkip := sSkip.Next(loopSkip, now, base.UnixNano())
	if dSkip.Fire {
		t.Fatalf("skip policy fired on a missed window (%+v)", dSkip)
	}
	if dSkip.Reason != ReasonMissedSkipped {
		t.Fatalf("skip reason = %q, want %q", dSkip.Reason, ReasonMissedSkipped)
	}

	sCatch, loopCatch := mk(MissedCatchUp)
	dCatch := sCatch.Next(loopCatch, now, base.UnixNano())
	if !dCatch.Fire {
		t.Fatalf("catch-up policy did not fire to settle a missed window (%+v)", dCatch)
	}
	if dCatch.Reason != ReasonMissedCaughtUp {
		t.Fatalf("catch-up reason = %q, want %q", dCatch.Reason, ReasonMissedCaughtUp)
	}
}

// TestScheduleOnTimeFireAndNotDue covers the two non-missed paths: a single
// window due fires normally; before the boundary nothing fires.
func TestScheduleOnTimeFireAndNotDue(t *testing.T) {
	base := time.Unix(2_000_000, 0).UTC()
	s := Schedule{JobID: "ontime/job", IntervalSeconds: 60, MissedRun: MissedSkip}
	loop := LoopSnapshot{LoopID: s.JobID, Started: 1, Ended: 1, LastEventUnixNano: base.UnixNano()}

	// 30s in: before the first boundary.
	notDue := s.Next(loop, base.Add(30*time.Second), base.UnixNano())
	if notDue.Fire || notDue.Reason != ReasonScheduleNotDue {
		t.Fatalf("expected not-due at 30s, got %+v", notDue)
	}

	// 90s in: exactly one window owed.
	due := s.Next(loop, base.Add(90*time.Second), base.UnixNano())
	if !due.Fire || due.Reason != ReasonScheduleFire {
		t.Fatalf("expected on-time fire at 90s, got %+v", due)
	}
}

// TestJitterDeterministicPerJobID is the #763 jitter acceptance: the offset is
// stable for a given job id, distinct across job ids, and bounded by the window.
func TestJitterDeterministicPerJobID(t *testing.T) {
	a := Schedule{JobID: "job-a", IntervalSeconds: 300, MissedRun: MissedSkip, JitterSeconds: 60}
	b := Schedule{JobID: "job-b", IntervalSeconds: 300, MissedRun: MissedSkip, JitterSeconds: 60}

	// Stable across repeated calls (no clock, no map seed).
	if a.JitterOffsetNanos() != a.JitterOffsetNanos() {
		t.Fatalf("jitter offset not stable for job-a")
	}
	// Bounded by the window.
	window := int64(60) * int64(time.Second)
	if off := a.JitterOffsetNanos(); off < 0 || off >= window {
		t.Fatalf("jitter offset %d out of [0,%d)", off, window)
	}
	// Distinct across distinct job ids (the thundering-herd spread).
	if a.JitterOffsetNanos() == b.JitterOffsetNanos() {
		t.Fatalf("job-a and job-b share a jitter offset; herd not spread")
	}
	// Zero jitter window yields zero offset.
	z := Schedule{JobID: "job-a", IntervalSeconds: 300, MissedRun: MissedSkip, JitterSeconds: 0}
	if z.JitterOffsetNanos() != 0 {
		t.Fatalf("zero jitter window gave non-zero offset")
	}

	// Re-deriving in a fresh Schedule value gives the SAME offset — it is a pure
	// function of the id, not of any process state.
	a2 := Schedule{JobID: "job-a", IntervalSeconds: 999, MissedRun: MissedCatchUp, JitterSeconds: 60}
	if a.JitterOffsetNanos() != a2.JitterOffsetNanos() {
		t.Fatalf("jitter offset depends on more than the job id")
	}
}

func TestScheduleTimingModeMutualExclusion(t *testing.T) {
	// Both unset / zero -> ErrInvalidSchedule
	bothZero := Schedule{JobID: "bad-zero", MissedRun: MissedSkip}
	if err := bothZero.Validate(); err == nil || !errors.Is(err, ErrInvalidSchedule) {
		t.Fatalf("Validate accepted both timing modes zero; got err: %v", err)
	}

	// Both set (>0) -> ErrInvalidSchedule
	bothSet := Schedule{JobID: "bad-both", IntervalSeconds: 60, AtUnixNano: 1_000_000, MissedRun: MissedSkip}
	if err := bothSet.Validate(); err == nil || !errors.Is(err, ErrInvalidSchedule) {
		t.Fatalf("Validate accepted both timing modes set; got err: %v", err)
	}

	// Negative IntervalSeconds -> ErrInvalidSchedule
	negInterval := Schedule{JobID: "bad-neg-int", IntervalSeconds: -10, MissedRun: MissedSkip}
	if err := negInterval.Validate(); err == nil || !errors.Is(err, ErrInvalidSchedule) {
		t.Fatalf("Validate accepted negative IntervalSeconds; got err: %v", err)
	}

	// Negative AtUnixNano -> ErrInvalidSchedule
	negAt := Schedule{JobID: "bad-neg-at", AtUnixNano: -500, MissedRun: MissedSkip}
	if err := negAt.Validate(); err == nil || !errors.Is(err, ErrInvalidSchedule) {
		t.Fatalf("Validate accepted negative AtUnixNano; got err: %v", err)
	}

	// Valid recurring mode
	rec := Schedule{JobID: "ok-recurring", IntervalSeconds: 60, MissedRun: MissedSkip}
	if err := rec.Validate(); err != nil {
		t.Fatalf("Validate rejected valid recurring schedule: %v", err)
	}
	if !rec.IsRecurring() || rec.IsOneShot() {
		t.Fatalf("rec.IsRecurring() = %v, rec.IsOneShot() = %v, want true, false", rec.IsRecurring(), rec.IsOneShot())
	}

	// Valid one-shot mode
	oneShot := Schedule{JobID: "ok-oneshot", AtUnixNano: 1_700_000_000_000_000_000, MissedRun: MissedSkip}
	if err := oneShot.Validate(); err != nil {
		t.Fatalf("Validate rejected valid one-shot schedule: %v", err)
	}
	if !oneShot.IsOneShot() || oneShot.IsRecurring() {
		t.Fatalf("oneShot.IsOneShot() = %v, oneShot.IsRecurring() = %v, want true, false", oneShot.IsOneShot(), oneShot.IsRecurring())
	}
}

func TestScheduleOneShotDueFutureAndConsumed(t *testing.T) {
	targetTime := time.Date(2026, 9, 5, 12, 0, 0, 0, time.UTC)
	targetNanos := targetTime.UnixNano()

	s := Schedule{
		JobID:      "one-shot-job",
		AtUnixNano: targetNanos,
		MissedRun:  MissedSkip,
	}

	loop := LoopSnapshot{LoopID: s.JobID}

	// 1. Future (not yet due)
	before := targetTime.Add(-10 * time.Second)
	dBefore := s.Next(loop, before, 0)
	if dBefore.Fire {
		t.Fatalf("expected one-shot in future not to fire, got %+v", dBefore)
	}
	if dBefore.Reason != ReasonScheduleNotDue {
		t.Fatalf("reason = %q, want %q", dBefore.Reason, ReasonScheduleNotDue)
	}
	if dBefore.FireAtUnixNano != targetNanos {
		t.Fatalf("FireAtUnixNano = %d, want %d", dBefore.FireAtUnixNano, targetNanos)
	}

	// 2. Exactly due
	dDue := s.Next(loop, targetTime, 0)
	if !dDue.Fire {
		t.Fatalf("expected one-shot at target time to fire, got %+v", dDue)
	}
	if dDue.Reason != ReasonScheduleFire {
		t.Fatalf("reason = %q, want %q", dDue.Reason, ReasonScheduleFire)
	}
	if dDue.FireAtUnixNano != targetNanos {
		t.Fatalf("FireAtUnixNano = %d, want %d", dDue.FireAtUnixNano, targetNanos)
	}

	// 3. Past due (due in the past, unconsumed)
	after := targetTime.Add(10 * time.Minute)
	dAfter := s.Next(loop, after, 0)
	if !dAfter.Fire {
		t.Fatalf("expected unconsumed past-due one-shot to fire, got %+v", dAfter)
	}
	if dAfter.Reason != ReasonScheduleFire {
		t.Fatalf("reason = %q, want %q", dAfter.Reason, ReasonScheduleFire)
	}
	if dAfter.FireAtUnixNano != targetNanos {
		t.Fatalf("FireAtUnixNano = %d, want %d", dAfter.FireAtUnixNano, targetNanos)
	}

	// 4. Consumed via Schedule.Consumed field -> never re-fires
	sConsumed := s
	sConsumed.Consumed = true
	if !sConsumed.IsConsumed() {
		t.Fatalf("expected sConsumed.IsConsumed() == true")
	}
	dConsumed := sConsumed.Next(loop, after, 0)
	if dConsumed.Fire {
		t.Fatalf("expected consumed one-shot not to fire, got %+v", dConsumed)
	}
	if dConsumed.Reason != ReasonScheduleConsumed {
		t.Fatalf("reason = %q, want %q", dConsumed.Reason, ReasonScheduleConsumed)
	}

	// 5. Consumed via NextWithConsumed argument -> overrides unconsumed schedule
	dConsumedArg := s.NextWithConsumed(loop, after, 0, true)
	if dConsumedArg.Fire {
		t.Fatalf("expected NextWithConsumed(..., consumed=true) not to fire, got %+v", dConsumedArg)
	}
	if dConsumedArg.Reason != ReasonScheduleConsumed {
		t.Fatalf("reason = %q, want %q", dConsumedArg.Reason, ReasonScheduleConsumed)
	}

	// 6. Overlap lock wins even for due one-shot
	loopInFlight := LoopSnapshot{LoopID: s.JobID, Started: 1, Ended: 0}
	dOverlap := s.Next(loopInFlight, targetTime, 0)
	if dOverlap.Fire {
		t.Fatalf("expected overlap lock to block due one-shot, got %+v", dOverlap)
	}
	if dOverlap.Reason != ReasonOverlapLock {
		t.Fatalf("reason = %q, want %q", dOverlap.Reason, ReasonOverlapLock)
	}
}

func TestScheduleOneShotWithJitter(t *testing.T) {
	targetTime := time.Date(2026, 9, 5, 12, 0, 0, 0, time.UTC)
	targetNanos := targetTime.UnixNano()

	s := Schedule{
		JobID:         "jittered-oneshot",
		AtUnixNano:    targetNanos,
		MissedRun:     MissedSkip,
		JitterSeconds: 30,
	}

	jitter := s.JitterOffsetNanos()
	if jitter <= 0 || jitter >= 30*int64(time.Second) {
		t.Fatalf("expected deterministic jitter > 0 and < 30s, got %d", jitter)
	}

	expectedFireNanos := targetNanos + jitter
	loop := LoopSnapshot{LoopID: s.JobID}

	// At nominal target time (before jitter boundary): not due
	dNominal := s.Next(loop, targetTime, 0)
	if dNominal.Fire {
		t.Fatalf("fired before jitter offset: %+v", dNominal)
	}
	if dNominal.Reason != ReasonScheduleNotDue {
		t.Fatalf("reason = %q, want %q", dNominal.Reason, ReasonScheduleNotDue)
	}
	if dNominal.FireAtUnixNano != expectedFireNanos {
		t.Fatalf("FireAtUnixNano = %d, want %d", dNominal.FireAtUnixNano, expectedFireNanos)
	}

	// At jitter boundary: due
	atJitter := time.Unix(0, expectedFireNanos).UTC()
	dDue := s.Next(loop, atJitter, 0)
	if !dDue.Fire {
		t.Fatalf("expected fire at jittered boundary, got %+v", dDue)
	}
	if dDue.Reason != ReasonScheduleFire {
		t.Fatalf("reason = %q, want %q", dDue.Reason, ReasonScheduleFire)
	}
	if dDue.FireAtUnixNano != expectedFireNanos {
		t.Fatalf("FireAtUnixNano = %d, want %d", dDue.FireAtUnixNano, expectedFireNanos)
	}
}
