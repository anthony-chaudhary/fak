package frontierswe

import "testing"

// C14 honesty boundary (epic #1706, issue #1720). The compare_test.go coverage
// exercises TTSMetricOf only at correctness 1.0 (solved) and 0.5 (censored) — far
// from the threshold where the "reached vs censored" decision actually turns. This
// file pins the boundary itself: the exact correctness at which a graded trial earns
// a time-to-solution. That line is load-bearing — a trial one hair below 1.0 that is
// wrongly marked Reached would book a fabricated fast solution into the C12 compare;
// one wrongly marked censored would silently drop a real, near-perfect solve.

// correctnessSolved must accept a grade within DefaultScoreParityTol of 1.0 and
// reject one clearly below it. This pins both the direction of the comparison
// (>=, not >) and the magnitude of the tolerance: dropping the tolerance would
// censor 1.0-tol, and widening it would reach 1.0-3*tol. The tolerance is shared
// with the C11 parity gate on purpose (tts_metric.go's own comment), so the two
// gates agree on what "correct" means — this test is what keeps them agreeing.
func TestCorrectnessSolvedBoundary(t *testing.T) {
	cases := []struct {
		name        string
		correctness float64
		wantSolved  bool
	}{
		{"exactly 1.0", 1.0, true},
		{"one tolerance below 1.0 still solved", 1.0 - DefaultScoreParityTol, true},
		{"three tolerances below 1.0 is censored", 1.0 - 3*DefaultScoreParityTol, false},
		{"clearly partial is censored", 0.5, false},
		{"zero is censored", 0.0, false},
		{"above 1.0 is solved", 1.0 + DefaultScoreParityTol, true},
	}
	for _, c := range cases {
		if got := correctnessSolved(c.correctness); got != c.wantSolved {
			t.Errorf("%s: correctnessSolved(%.12g) = %v, want %v", c.name, c.correctness, got, c.wantSolved)
		}
	}
}

// A solved trial whose correctness sits just inside the parity band (1.0 - tol, not a
// clean 1.0) must still be timed at its full trajectory — Reached with the trace's
// wall-clock and turns, never zeroed. This guards the specific failure of dropping a
// near-perfect solve as if it were censored, which would lose a real TTS data point.
func TestTTSMetricReachedJustInsideBand(t *testing.T) {
	m := TTSMetricOf("git-to-zig", "edge", 1.0-DefaultScoreParityTol, traceOf(1234, 42), false)
	if !m.Reached {
		t.Fatalf("correctness within parity tolerance of 1.0 must be Reached, got censored")
	}
	if !floatsEqual(m.WallSecToCorrect, 1234) || m.TurnsToCorrect != 42 {
		t.Fatalf("solved trial timed at full trajectory, got wall=%v turns=%d", m.WallSecToCorrect, m.TurnsToCorrect)
	}
}

// A censored trial contributes no time-to-solution, but the row must still be
// inspectable and machine-joinable by C12: schema, task, trial id, and the real
// correctness are all carried through unchanged. Reached=false with zero wall/turns
// is the only honest shape — the zeros must never be read as a fast solve.
func TestTTSMetricCensoredCarriesIdentityNotTime(t *testing.T) {
	m := TTSMetricOf("swe-fix", "c9", 0.5, traceOf(999, 40), false)
	if m.Reached {
		t.Fatalf("correctness 0.5 must be censored, not Reached")
	}
	if m.WallSecToCorrect != 0 || m.TurnsToCorrect != 0 {
		t.Fatalf("censored trial contributes no time, got wall=%v turns=%d", m.WallSecToCorrect, m.TurnsToCorrect)
	}
	if m.Schema != TTSMetricSchema {
		t.Errorf("schema = %q, want %q", m.Schema, TTSMetricSchema)
	}
	if m.Task != "swe-fix" || m.TrialID != "c9" {
		t.Errorf("identity not carried: task=%q trial=%q", m.Task, m.TrialID)
	}
	if !floatsEqual(m.Correctness, 0.5) {
		t.Errorf("censored trial must carry its real correctness, got %v", m.Correctness)
	}
}
