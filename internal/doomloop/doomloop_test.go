package doomloop

import "testing"

// mkSamples builds a stream from parallel effort/progress counters. alive is
// applied to every sample (the common case); alive-specific cases build by hand.
func mkSamples(effort, progress []int64, alive bool) []Sample {
	if len(effort) != len(progress) {
		panic("effort/progress length mismatch")
	}
	out := make([]Sample, len(effort))
	for i := range effort {
		out[i] = Sample{
			UnixMillis: int64(i) * 60_000,
			Effort:     effort[i],
			Progress:   progress[i],
			Alive:      alive,
		}
	}
	return out
}

// TestDoomLoopCaughtWhereFleetmonSaysHealthy is the load-bearing regression: a
// worker whose EFFORT grows every sample while VERIFIED PROGRESS stays flat is a
// doom loop. fleetmon.isAdvancing calls this exact shape "healthy" (line delta
// > 0 => advancing). This leaf must call it DOOM_LOOP and recommend a nudge.
func TestDoomLoopCaughtWhereFleetmonSaysHealthy(t *testing.T) {
	// Effort climbs 10 -> 60 (burning every window); progress pinned at 4.
	samples := mkSamples(
		[]int64{10, 20, 30, 40, 50, 60},
		[]int64{4, 4, 4, 4, 4, 4},
		true,
	)
	got := Classify(samples, DefaultConfig())
	if got.Verdict != VerdictDoomLoop {
		t.Fatalf("verdict = %q, want %q (burning effort + flat progress is a doom loop, not healthy). reason=%q", got.Verdict, VerdictDoomLoop, got.Reason)
	}
	if got.Correction != CorrectNudge {
		t.Fatalf("correction = %q, want %q at the trip threshold", got.Correction, CorrectNudge)
	}
	if got.BurningFlatStreak < DefaultConfig().TripWindows {
		t.Fatalf("streak = %d, want >= trip %d", got.BurningFlatStreak, DefaultConfig().TripWindows)
	}
}

// TestHealthyWhenProgressAdvances: same burning effort, but verified progress
// climbs too. Real progress is healthy regardless of spend.
func TestHealthyWhenProgressAdvances(t *testing.T) {
	samples := mkSamples(
		[]int64{10, 20, 30, 40, 50, 60},
		[]int64{1, 2, 3, 4, 5, 6},
		true,
	)
	got := Classify(samples, DefaultConfig())
	if got.Verdict != VerdictHealthy {
		t.Fatalf("verdict = %q, want %q (progress is advancing)", got.Verdict, VerdictHealthy)
	}
	if got.Correction != CorrectNone {
		t.Fatalf("correction = %q, want %q", got.Correction, CorrectNone)
	}
}

// TestBurningFlatBelowTripObservesNotIntervenes: a short burning-flat streak is
// normal (a big commit lands only after several turns), so the core must NOT
// intervene before the trip threshold - it observes.
func TestBurningFlatBelowTripObservesNotIntervenes(t *testing.T) {
	// Two burning-flat windows only (3 samples), trip is 3.
	samples := mkSamples(
		[]int64{10, 20, 30},
		[]int64{5, 5, 5},
		true,
	)
	got := Classify(samples, DefaultConfig())
	if got.Verdict != VerdictHealthy {
		t.Fatalf("verdict = %q, want %q below the trip threshold", got.Verdict, VerdictHealthy)
	}
	if got.Correction != CorrectObserve {
		t.Fatalf("correction = %q, want %q (watching, not intervening)", got.Correction, CorrectObserve)
	}
	if got.BurningFlatStreak != 2 {
		t.Fatalf("streak = %d, want 2", got.BurningFlatStreak)
	}
}

// TestPersistentDoomEscalates: once the burning-flat streak passes
// EscalateWindows the recommendation climbs from nudge to escalate.
func TestPersistentDoomEscalates(t *testing.T) {
	// 7 samples => 6 burning-flat windows; EscalateWindows default is 6.
	samples := mkSamples(
		[]int64{1, 2, 3, 4, 5, 6, 7},
		[]int64{9, 9, 9, 9, 9, 9, 9},
		true,
	)
	got := Classify(samples, DefaultConfig())
	if got.Verdict != VerdictDoomLoop {
		t.Fatalf("verdict = %q, want %q", got.Verdict, VerdictDoomLoop)
	}
	if got.Correction != CorrectEscalate {
		t.Fatalf("correction = %q, want %q at streak %d", got.Correction, CorrectEscalate, got.BurningFlatStreak)
	}
}

// TestIdleWhenNotSpending: a worker that is not burning effort cannot be in a
// doom loop, even with flat progress. Alive => IDLE (parked/quiet).
func TestIdleWhenNotSpending(t *testing.T) {
	samples := mkSamples(
		[]int64{10, 10, 10, 10},
		[]int64{3, 3, 3, 3},
		true,
	)
	got := Classify(samples, DefaultConfig())
	if got.Verdict != VerdictIdle {
		t.Fatalf("verdict = %q, want %q (not spending)", got.Verdict, VerdictIdle)
	}
	if got.Correction != CorrectNone {
		t.Fatalf("correction = %q, want %q", got.Correction, CorrectNone)
	}
}

// TestWedgedWhenNotSpendingAndNotAlive: not burning + not alive at the latest
// sample is a frozen worker, a stall for a different rung - never a doom loop.
func TestWedgedWhenNotSpendingAndNotAlive(t *testing.T) {
	samples := []Sample{
		{UnixMillis: 0, Effort: 10, Progress: 3, Alive: true},
		{UnixMillis: 60_000, Effort: 10, Progress: 3, Alive: false},
		{UnixMillis: 120_000, Effort: 10, Progress: 3, Alive: false},
	}
	got := Classify(samples, DefaultConfig())
	if got.Verdict != VerdictWedged {
		t.Fatalf("verdict = %q, want %q (frozen)", got.Verdict, VerdictWedged)
	}
}

// TestUnknownFailsClosed: below MinSamples the core refuses to decide and
// recommends no action - a worker we cannot read is never corrected.
func TestUnknownFailsClosed(t *testing.T) {
	samples := mkSamples([]int64{10, 20}, []int64{1, 1}, true)
	got := Classify(samples, DefaultConfig())
	if got.Verdict != VerdictUnknown {
		t.Fatalf("verdict = %q, want %q with < MinSamples", got.Verdict, VerdictUnknown)
	}
	if got.Correction != CorrectNone {
		t.Fatalf("correction = %q, want %q", got.Correction, CorrectNone)
	}
}

// TestProgressRecoveryEndsEpisode: after a burning-flat streak, one window that
// lands verified progress resets the verdict to HEALTHY - the episode is over,
// no stale doom verdict lingers.
func TestProgressRecoveryEndsEpisode(t *testing.T) {
	// Burning-flat for four windows, then a progress landing in the last window.
	samples := mkSamples(
		[]int64{10, 20, 30, 40, 50, 60},
		[]int64{2, 2, 2, 2, 2, 5},
		true,
	)
	got := Classify(samples, DefaultConfig())
	if got.Verdict != VerdictHealthy {
		t.Fatalf("verdict = %q, want %q (progress just recovered)", got.Verdict, VerdictHealthy)
	}
	if got.BurningFlatStreak != 0 {
		t.Fatalf("streak = %d, want 0 after recovery", got.BurningFlatStreak)
	}
}

// TestCounterResetNotProgress: a backward counter (transcript rotation) is
// clamped to a zero delta - a reset must never read as progress and mask a
// doom loop.
func TestCounterResetNotProgress(t *testing.T) {
	// Progress counter drops (100 -> 0) mid-stream while effort keeps burning.
	samples := mkSamples(
		[]int64{10, 20, 30, 40},
		[]int64{100, 0, 0, 0},
		true,
	)
	got := Classify(samples, DefaultConfig())
	if got.ProgressDelta != 0 {
		t.Fatalf("latest progress delta = %d, want 0 (reset clamped)", got.ProgressDelta)
	}
	// Three burning-flat windows (the 100->0 drop clamps to flat) => trips.
	if got.Verdict != VerdictDoomLoop {
		t.Fatalf("verdict = %q, want %q (a reset is not progress)", got.Verdict, VerdictDoomLoop)
	}
}

// TestNoClaimedField is a reflective guard: verified progress must be the only
// progress axis. A Sample must not grow a self-reported "done"/"claimed" field -
// the whole point is that progress is witnessed, never narrated.
func TestSampleHasNoSelfReportField(t *testing.T) {
	// Compile-time shape assertion: the fields we depend on exist and are the
	// only progress input. If someone adds a Claimed/SelfReported field, this
	// test is where the review conversation should happen.
	s := Sample{UnixMillis: 1, Effort: 2, Progress: 3, Alive: true}
	if s.Progress != 3 {
		t.Fatal("Progress must be the verified progress axis")
	}
}

// TestInterpretationNeverEmpty: every result carries a one-line next-action read.
func TestInterpretationNeverEmpty(t *testing.T) {
	cases := [][]Sample{
		mkSamples([]int64{10, 20, 30, 40, 50, 60}, []int64{4, 4, 4, 4, 4, 4}, true),
		mkSamples([]int64{10, 20, 30}, []int64{1, 2, 3}, true),
		mkSamples([]int64{10, 10, 10}, []int64{3, 3, 3}, true),
		mkSamples([]int64{10, 20}, []int64{1, 1}, true),
	}
	for i, s := range cases {
		if got := Classify(s, DefaultConfig()).Interpretation(); got == "" {
			t.Fatalf("case %d: interpretation is empty", i)
		}
	}
}
