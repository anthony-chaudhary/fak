package loopmgr

import (
	"testing"
	"time"
)

// endEvent builds one `end` row with the reason/metrics a run declares its utility
// with, so a fixture can spell an outcome the same way `fak loop run` writes it.
func endEvent(loopID string, status RunStatus, reason string, tsUnixNano int64, metrics map[string]int64) Event {
	return Event{
		Schema:     SchemaEvent,
		LoopID:     loopID,
		Kind:       EventEnd,
		Status:     status,
		Reason:     reason,
		TSUnixNano: tsUnixNano,
		Metrics:    metrics,
	}
}

// TestFoldHealth_QuarantinedSchedulerLoopsAlert is the #6497 witness. It replays the
// EXACT ledger shape the 2026-08-11 maintenance audit found for the two quarantined
// task-scheduler loops — `scout-loop/task-scheduler` (two runs, both
// `end status=failed reason=EXIT_NONZERO exit_code=1`) and `logvault-capture` (two
// runs, both exit_code=2) — alongside a healthy peer, and asserts that the operator
// health fold now says out loud what the raw ledger proved and every projection over
// it used to swallow: four runs, four failures, zero successes.
//
// Before the utility partition this test could not be written: Ended counted every
// end regardless of outcome, so both broken loops reported Runs=2 with no failure
// signal anywhere on the row or in the roll-up, and a pane gating on Dark saw two
// loops ticking perfectly on cadence.
func TestFoldHealth_QuarantinedSchedulerLoopsAlert(t *testing.T) {
	now := time.Unix(1_786_500_000, 0).UTC()
	secAgo := func(s int64) int64 { return now.Add(-time.Duration(s) * time.Second).UnixNano() }

	// Both broken loops fire on a DAILY cadence and each ticked within the last hour,
	// so the liveness plane reads them LIVE. That is deliberate: the alarm this fold
	// must raise is orthogonal to darkness.
	const daily = int64(86_400)
	events := []Event{
		// scout-loop/task-scheduler: 2026-08-10 and 2026-08-11, both EXIT_NONZERO 1.
		tick("scout-loop/task-scheduler", EventFire, "", secAgo(90_000)),
		tick("scout-loop/task-scheduler", EventStart, StatusRunning, secAgo(89_999)),
		endEvent("scout-loop/task-scheduler", StatusFailed, "EXIT_NONZERO", secAgo(89_998), map[string]int64{"exit_code": 1}),
		tick("scout-loop/task-scheduler", EventFire, "", secAgo(3_600)),
		tick("scout-loop/task-scheduler", EventStart, StatusRunning, secAgo(3_599)),
		endEvent("scout-loop/task-scheduler", StatusFailed, "EXIT_NONZERO", secAgo(3_598), map[string]int64{"exit_code": 1}),

		// logvault-capture: same two days, both EXIT_NONZERO 2. Task Scheduler reported
		// LastTaskResult=0 for this one — the ledger is the authority that disagrees.
		tick("logvault-capture", EventFire, "", secAgo(90_000)),
		tick("logvault-capture", EventStart, StatusRunning, secAgo(89_999)),
		endEvent("logvault-capture", StatusFailed, "EXIT_NONZERO", secAgo(89_998), map[string]int64{"exit_code": 2}),
		tick("logvault-capture", EventFire, "", secAgo(3_600)),
		tick("logvault-capture", EventStart, StatusRunning, secAgo(3_599)),
		endEvent("logvault-capture", StatusFailed, "EXIT_NONZERO", secAgo(3_598), map[string]int64{"exit_code": 2}),

		// healthy: one run that produced two useful effects at a measured cost, then one
		// run that typed-declared there was nothing to do. Neither is a failure, and the
		// no-fuel tick must NOT be laundered into an effect.
		tick("healthy", EventStart, StatusRunning, secAgo(7_200)),
		endEvent("healthy", StatusClaimedDone, "", secAgo(7_100), map[string]int64{MetricUsefulEffects: 2, MetricCostMilliUSD: 430}),
		tick("healthy", EventStart, StatusRunning, secAgo(1_800)),
		endEvent("healthy", StatusClaimedDone, ReasonNoFuel, secAgo(1_700), map[string]int64{MetricCostMilliUSD: 12}),
	}

	reg := Registry{Jobs: map[string]Job{}}
	for _, id := range []string{"scout-loop/task-scheduler", "logvault-capture", "healthy"} {
		if err := reg.Put(Job{
			Schedule: Schedule{JobID: id, IntervalSeconds: daily, MissedRun: MissedSkip},
			State:    JobArmed,
		}, now); err != nil {
			t.Fatalf("registry.Put(%s): %v", id, err)
		}
	}

	rep := FoldHealth(Summarize(events, now), reg, now, HealthThresholds{})
	rows := map[string]HealthRow{}
	for _, r := range rep.Rows {
		rows[r.LoopID] = r
	}

	for _, id := range []string{"scout-loop/task-scheduler", "logvault-capture"} {
		row := rows[id]
		// The liveness plane is clean — which is exactly why the failure plane has to
		// exist. If this ever flips to dark the test is no longer proving the point.
		if row.State != HealthLive || row.Dark {
			t.Fatalf("%s state = %q dark=%v, want live/false (it ticks on cadence; the alarm must not depend on darkness)", id, row.State, row.Dark)
		}
		if row.Runs != 2 || row.Failed != 2 {
			t.Errorf("%s runs=%d failed=%d, want 2/2 (both recorded runs ended EXIT_NONZERO)", id, row.Runs, row.Failed)
		}
		if row.ConsecutiveFailures != 2 {
			t.Errorf("%s consecutiveFailures = %d, want 2", id, row.ConsecutiveFailures)
		}
		if !row.FailureAlert {
			t.Errorf("%s failureAlert = false, want true (the second failure is the first REPEATED failure)", id)
		}
		if !row.NeverSucceeded {
			t.Errorf("%s neverSucceeded = false, want true (zero successful recorded runs)", id)
		}
		if row.Effects != 0 || row.NoFuel != 0 || row.Unattributed != 0 {
			t.Errorf("%s effects=%d noFuel=%d unattributed=%d, want 0/0/0 (a failed run produces none of them)", id, row.Effects, row.NoFuel, row.Unattributed)
		}
	}

	h := rows["healthy"]
	if h.Failed != 0 || h.ConsecutiveFailures != 0 || h.FailureAlert || h.NeverSucceeded {
		t.Errorf("healthy failed=%d consecutive=%d alert=%v neverSucceeded=%v, want 0/0/false/false", h.Failed, h.ConsecutiveFailures, h.FailureAlert, h.NeverSucceeded)
	}
	if h.Effects != 1 || h.NoFuel != 1 || h.Unattributed != 0 {
		t.Errorf("healthy effects=%d noFuel=%d unattributed=%d, want 1/1/0 (one effect run, one typed no-fuel tick)", h.Effects, h.NoFuel, h.Unattributed)
	}
	if h.CostMilliUSD != 442 || h.CostedRuns != 2 {
		t.Errorf("healthy cost=%d over %d run(s), want 442 over 2", h.CostMilliUSD, h.CostedRuns)
	}
	// Ended partitions exactly: every ended run lands in exactly one bucket.
	if got := h.Failed + h.Effects + h.NoFuel + h.Unattributed; got != h.Runs {
		t.Errorf("healthy partition sums to %d, want Runs=%d", got, h.Runs)
	}

	ru := rep.Rollup
	if ru.Dark != 0 {
		t.Errorf("rollup.Dark = %d, want 0 — the fleet looks green on liveness alone", ru.Dark)
	}
	if ru.Runs != 6 {
		t.Errorf("rollup.Runs = %d, want 6 (2+2 failing runs plus the healthy loop's 2)", ru.Runs)
	}
	if ru.Failed != 4 || ru.FailureAlert != 2 || ru.NeverSucceeded != 2 {
		t.Errorf("rollup failed=%d failureAlert=%d neverSucceeded=%d, want 4/2/2 (four runs, four failures, two dead loops)", ru.Failed, ru.FailureAlert, ru.NeverSucceeded)
	}
	if ru.Effects != 1 || ru.NoFuel != 1 || ru.Unattributed != 0 {
		t.Errorf("rollup effects=%d noFuel=%d unattributed=%d, want 1/1/0", ru.Effects, ru.NoFuel, ru.Unattributed)
	}
	if ru.CostMilliUSD != 442 || ru.CostedRuns != 2 {
		t.Errorf("rollup cost=%d over %d run(s), want 442 over 2", ru.CostMilliUSD, ru.CostedRuns)
	}
}

// TestClassifyEnd_Partition pins the classification order and totality: a failing
// status beats a declared effect, a typed no-fuel reason beats an absent effect
// metric, a zero effect count is not an effect, and a bare exit-0 end is
// UNATTRIBUTED rather than being counted as a success.
func TestClassifyEnd_Partition(t *testing.T) {
	cases := []struct {
		name string
		ev   Event
		want EndOutcome
	}{
		{"failed", Event{Kind: EventEnd, Status: StatusFailed, Reason: "EXIT_NONZERO"}, OutcomeFailure},
		{"canceled", Event{Kind: EventEnd, Status: StatusCanceled}, OutcomeFailure},
		{"failed beats declared effect", Event{Kind: EventEnd, Status: StatusFailed, Metrics: map[string]int64{MetricUsefulEffects: 3}}, OutcomeFailure},
		{"failed beats no-fuel reason", Event{Kind: EventEnd, Status: StatusFailed, Reason: ReasonNoFuel}, OutcomeFailure},
		{"typed no-fuel", Event{Kind: EventEnd, Status: StatusClaimedDone, Reason: ReasonNoFuel}, OutcomeNoFuel},
		{"declared effect", Event{Kind: EventEnd, Status: StatusClaimedDone, Metrics: map[string]int64{MetricUsefulEffects: 1}}, OutcomeEffect},
		{"zero effects is not an effect", Event{Kind: EventEnd, Status: StatusClaimedDone, Metrics: map[string]int64{MetricUsefulEffects: 0}}, OutcomeUnattributed},
		{"bare exit 0", Event{Kind: EventEnd, Status: StatusClaimedDone}, OutcomeUnattributed},
		{"absent status falls back to claimed-done", Event{Kind: EventEnd}, OutcomeUnattributed},
	}
	for _, status := range []RunStatus{
		StatusAdmitted, StatusRefused, StatusRunning, StatusClaimedDone,
		StatusWitnessedDone, StatusWitnessRefused, StatusWitnessUnavailable,
	} {
		cases = append(cases, struct {
			name string
			ev   Event
			want EndOutcome
		}{"nonfailure status " + string(status), Event{Kind: EventEnd, Status: status}, OutcomeUnattributed})
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := ClassifyEnd(tc.ev); got != tc.want {
				t.Errorf("ClassifyEnd = %q, want %q", got, tc.want)
			}
		})
	}
}

// TestConsecutiveFailures_ResetsOnCompletion proves the alert lever is a STREAK, not
// a lifetime total: a loop that fails twice, recovers, then fails once is no longer
// alerting, while its lifetime Failed count keeps every failure. Without the reset an
// operator could never clear the alarm by fixing the loop.
func TestConsecutiveFailures_ResetsOnCompletion(t *testing.T) {
	now := time.Unix(2_000_000, 0).UTC()
	at := func(s int64) int64 { return now.Add(-time.Duration(s) * time.Second).UnixNano() }

	events := []Event{
		endEvent("flaky", StatusFailed, "EXIT_NONZERO", at(500), nil),
		endEvent("flaky", StatusFailed, "EXIT_NONZERO", at(400), nil),
	}
	snap := Summarize(events, now).Loops[0]
	if !snap.FailureAlert() || snap.ConsecutiveFailures != 2 {
		t.Fatalf("after two failures: consecutive=%d alert=%v, want 2/true", snap.ConsecutiveFailures, snap.FailureAlert())
	}
	if !snap.NeverSucceeded() {
		t.Fatalf("after two failures: neverSucceeded=false, want true")
	}

	// A completed run clears the streak and the never-succeeded verdict; the lifetime
	// failure count is untouched.
	events = append(events, endEvent("flaky", StatusClaimedDone, ReasonNoFuel, at(300), nil))
	snap = Summarize(events, now).Loops[0]
	if snap.ConsecutiveFailures != 0 || snap.FailureAlert() {
		t.Errorf("after recovery: consecutive=%d alert=%v, want 0/false", snap.ConsecutiveFailures, snap.FailureAlert())
	}
	if snap.NeverSucceeded() {
		t.Errorf("after recovery: neverSucceeded=true, want false (one run completed)")
	}
	if snap.Failed != 2 || snap.NoFuel != 1 || snap.Ended != 3 {
		t.Errorf("after recovery: failed=%d noFuel=%d ended=%d, want 2/1/3", snap.Failed, snap.NoFuel, snap.Ended)
	}

	// One more failure is a single data point, not a repeated failure: no alert yet.
	events = append(events, endEvent("flaky", StatusFailed, "EXIT_NONZERO", at(200), nil))
	snap = Summarize(events, now).Loops[0]
	if snap.ConsecutiveFailures != 1 || snap.FailureAlert() {
		t.Errorf("after one fresh failure: consecutive=%d alert=%v, want 1/false (alert is the FIRST REPEATED failure)", snap.ConsecutiveFailures, snap.FailureAlert())
	}
}

// TestSummarizeFrom_UtilityCountersFoldIncrementally proves the incremental path
// (used by the rotated-segment fold) carries the utility counters and the streak
// across a seed, so a rotation can never silently reset a loop's failure alarm.
func TestSummarizeFrom_UtilityCountersFoldIncrementally(t *testing.T) {
	now := time.Unix(3_000_000, 0).UTC()
	at := func(s int64) int64 { return now.Add(-time.Duration(s) * time.Second).UnixNano() }

	seed := Summarize([]Event{
		endEvent("rot", StatusFailed, "EXIT_NONZERO", at(900), map[string]int64{MetricCostMilliUSD: 100}),
	}, now).Loops

	got := SummarizeFrom(seed, []Event{
		endEvent("rot", StatusFailed, "EXIT_NONZERO", at(800), map[string]int64{MetricCostMilliUSD: 150}),
	}, now).Loops[0]

	if got.Failed != 2 || got.ConsecutiveFailures != 2 || !got.FailureAlert() {
		t.Errorf("folded failed=%d consecutive=%d alert=%v, want 2/2/true", got.Failed, got.ConsecutiveFailures, got.FailureAlert())
	}
	if got.CostMilliUSD != 250 || got.CostedRuns != 2 {
		t.Errorf("folded cost=%d over %d run(s), want 250 over 2", got.CostMilliUSD, got.CostedRuns)
	}
	// The seed must not be mutated by the incremental fold.
	if seed[0].Failed != 1 || seed[0].CostMilliUSD != 100 {
		t.Errorf("seed mutated: failed=%d cost=%d, want 1/100", seed[0].Failed, seed[0].CostMilliUSD)
	}
}
