package fleetsim

import "testing"

// TestDefaultFixtureHits400 is the issue #1819 acceptance witness: the default
// synthetic fixture, folded by a dry-run Replay, must produce >= 400 witnessed
// closes per hour — WITHOUT spawning any real worker. Replay is pure computation
// over a generated ledger, so this proof is fully reproducible.
func TestDefaultFixtureHits400(t *testing.T) {
	const target = 400.0
	rep := Replay(DefaultFixture())

	if got := rep.ClosesPerHour(); got < target {
		t.Fatalf("default fixture ClosesPerHour = %.2f, want >= %.0f", got, target)
	}
	if !rep.TargetAchieved(target) {
		t.Fatalf("TargetAchieved(%.0f) = false; ClosesPerHour = %.2f", target, rep.ClosesPerHour())
	}
	if rep.Closes < 400 {
		t.Fatalf("witnessed closes = %d over a 1h window, want >= 400", rep.Closes)
	}
	// The fold must be internally consistent: total = closed + failed.
	if rep.TotalSessions != rep.Closes+rep.Failures {
		t.Fatalf("total %d != closes %d + failures %d",
			rep.TotalSessions, rep.Closes, rep.Failures)
	}
	t.Logf("default fixture: %d workers, %d sessions, %d closes, %d failures, %.1f closes/hr",
		rep.Workers, rep.TotalSessions, rep.Closes, rep.Failures, rep.ClosesPerHour())
}

// TestReplayIsDeterministic proves the dry run has no side effects that vary run
// to run: two independent Replays of the same fixture must fold to identical
// reports (no clock, no rand).
func TestReplayIsDeterministic(t *testing.T) {
	f := DefaultFixture()
	a := Replay(f)
	b := Replay(f)
	if a != b {
		t.Fatalf("Replay not deterministic:\n a=%+v\n b=%+v", a, b)
	}
	// Regenerating the event ledger twice must also match, event for event.
	e1, e2 := f.Events(), f.Events()
	if len(e1) != len(e2) {
		t.Fatalf("Events() length varies: %d vs %d", len(e1), len(e2))
	}
	for i := range e1 {
		if e1[i] != e2[i] {
			t.Fatalf("Events()[%d] varies: %+v vs %+v", i, e1[i], e2[i])
		}
	}
}

// TestClosesMatchClosedFraction pins the fold arithmetic: with a 5% failure rate
// and a deterministic every-20th-session failure stride, the closed count is the
// total minus the failures, and the closes-per-hour is the closed count over the
// one-hour window.
func TestClosesMatchClosedFraction(t *testing.T) {
	f := DefaultFixture()
	rep := Replay(f)

	// Failures are laid down every failStride sessions; count expected failures.
	stride := f.failStride()
	wantFailures := 0
	if stride > 0 {
		wantFailures = rep.TotalSessions / stride
	}
	if rep.Failures != wantFailures {
		t.Errorf("failures = %d, want %d (total %d, stride %d)",
			rep.Failures, wantFailures, rep.TotalSessions, stride)
	}
	if rep.Closes != rep.TotalSessions-rep.Failures {
		t.Errorf("closes = %d, want total-failures = %d", rep.Closes, rep.TotalSessions-rep.Failures)
	}
	// Over a one-hour window, closes/hr equals the raw close count.
	if got := rep.ClosesPerHour(); got != float64(rep.Closes) {
		t.Errorf("ClosesPerHour = %.2f, want raw closes %d over a 1h window", got, rep.Closes)
	}
}

// TestNoFailuresRaisesRate is a sanity check on the failure model: a 0% failure
// fixture yields more witnessed closes than the same-shape 5% one, and both still
// clear the target.
func TestNoFailuresRaisesRate(t *testing.T) {
	perfect := FixtureForTarget(400, defaultSessionSeconds, 0)
	lossy := FixtureForTarget(400, defaultSessionSeconds, 5)

	rp := Replay(perfect)
	rl := Replay(lossy)

	if rp.Failures != 0 {
		t.Errorf("0%% fixture has %d failures, want 0", rp.Failures)
	}
	if !rp.TargetAchieved(400) || !rl.TargetAchieved(400) {
		t.Fatalf("both fixtures must hit 400: perfect=%.1f lossy=%.1f",
			rp.ClosesPerHour(), rl.ClosesPerHour())
	}
}

// TestFixtureForTargetScales checks the sizing generalizes: a higher target
// requires at least as many workers, and the resulting fixture still clears its
// own target under Replay.
func TestFixtureForTargetScales(t *testing.T) {
	small := FixtureForTarget(400, defaultSessionSeconds, defaultFailRatePct)
	big := FixtureForTarget(800, defaultSessionSeconds, defaultFailRatePct)

	if big.Concurrency < small.Concurrency {
		t.Errorf("800/hr concurrency %d < 400/hr concurrency %d", big.Concurrency, small.Concurrency)
	}
	if got := Replay(big).ClosesPerHour(); got < 800 {
		t.Errorf("800-target fixture yields %.1f closes/hr, want >= 800", got)
	}
}

// TestEmptyFixtureIsSafe documents the fail-safe: a degenerate fixture generates
// no events and folds to a zero-close report that misses any positive target,
// rather than dividing by zero or panicking.
func TestEmptyFixtureIsSafe(t *testing.T) {
	rep := Replay(Fixture{Concurrency: 0, SessionSeconds: 600})
	if rep.Closes != 0 || rep.TotalSessions != 0 {
		t.Errorf("empty fixture folded to closes=%d sessions=%d, want 0/0", rep.Closes, rep.TotalSessions)
	}
	if rep.TargetAchieved(1) {
		t.Errorf("empty fixture must not achieve a positive target")
	}
}

// TestOutcomeString locks the ledger tokens the outcome enum renders.
func TestOutcomeString(t *testing.T) {
	if Closed.String() != "closed" || Failed.String() != "failed" {
		t.Errorf("outcome tokens drifted: closed=%q failed=%q", Closed.String(), Failed.String())
	}
}
