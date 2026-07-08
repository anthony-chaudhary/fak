package gateway

import (
	"testing"
	"time"
)

// messages_volume_horizon_test.go — the volume-aware head-anchored horizon (headSessionPrior /
// volumeAwareHorizon / heldResidentPeak). The head-anchored burst gate's break-even is a TOKEN ratio
// but its horizon was a flat turn count; these witnesses pin that a demonstrably heavy-AND-deep trace
// keeps a repaying-turn floor (so a median heavy session is not starved by a short-session constant)
// while the token-light majority is byte-for-byte unchanged.

// TestVolumeAwareHorizon pins the pure held-volume horizon policy: a token-light trace keeps the base
// horizon verbatim; a heavy-but-shallow trace also keeps base (the max() floor is inert while served
// depth is low); only a heavy-AND-deep trace lengthens to served+headroom. It never SHORTENS the
// horizon at any depth — the change is purely additive for the heavy long tail.
func TestVolumeAwareHorizon(t *testing.T) {
	base := DefaultAssumedSessionTurns // 50
	thin := headHorizonHeavyResidentFloor - 1
	heavy := headHorizonHeavyResidentFloor        // exactly at the floor counts as heavy (>=)
	inflection := base - headHorizonHeavyHeadroom // served at/below which the floor is still inert

	cases := []struct {
		name     string
		served   int
		heldPeak int64
		want     int
	}{
		{"thin shallow keeps base", 5, thin, base},
		{"thin deep never extends a token-light session", 200, thin, base},
		{"heavy shallow keeps base (floor inert while served<base-headroom)", 5, heavy, base},
		{"heavy at the inflection keeps base", inflection, heavy, base},
		{"heavy just past the inflection extends by one", inflection + 1, heavy, inflection + 1 + headHorizonHeavyHeadroom},
		{"heavy deep extends to served+headroom", 40, heavy, 40 + headHorizonHeavyHeadroom},
		{"heavy very deep extends further", 200, heavy + 1_000_000, 200 + headHorizonHeavyHeadroom},
	}
	for _, tc := range cases {
		if got := volumeAwareHorizon(base, tc.served, tc.heldPeak); got != tc.want {
			t.Errorf("%s: volumeAwareHorizon(%d,%d,%d)=%d, want %d", tc.name, base, tc.served, tc.heldPeak, got, tc.want)
		}
	}
	// Never-shortens invariant: a heavy trace's horizon is >= base at every depth.
	for served := 0; served < 600; served++ {
		if got := volumeAwareHorizon(base, served, heavy); got < base {
			t.Fatalf("heavy horizon dropped below base at served=%d: got %d (base %d)", served, got, base)
		}
	}
}

// TestHeldResidentPeakTracksPeakNotLast proves the held-volume signal is the trace's PEAK resident
// window, not its last turn's: a big turn followed by a small turn keeps the demonstrated heaviness,
// so a single light turn cannot demote a session that has proven it holds a large context. Unknown
// traces and a nil metrics report 0 (the conservative base horizon).
func TestHeldResidentPeakTracksPeakNotLast(t *testing.T) {
	now := time.Now()
	m := newGatewayMetrics(now)
	trace := "peak-trace"
	m.observeHarnessCoherence(trace, now, "", false, "", false, false, 90000, 0, 2000) // resident ~92k
	m.observeHarnessCoherence(trace, now, "", false, "", false, false, 1000, 0, 500)   // resident ~1.5k — must NOT demote
	if got := m.heldResidentPeakTokens(trace); got < 90000 {
		t.Fatalf("heldResidentPeak must retain the peak (>=90000), got %d", got)
	}
	if got := m.heldResidentPeakTokens("never-seen"); got != 0 {
		t.Fatalf("unknown trace peak = %d, want 0", got)
	}
	var nilM *gatewayMetrics
	if got := nilM.heldResidentPeakTokens(trace); got != 0 {
		t.Fatalf("nil metrics peak = %d, want 0", got)
	}
}

// TestHeadSessionPriorVolumeAwareHorizon is the wiring witness: a heavy-and-deep trace's presumed
// horizon extends to served+headroom (so the burst gate keeps repaying room), a token-light trace at
// the SAME depth keeps the base horizon, a wired bounded turnsLeft still WINS over the whole prior,
// and the prior-disabled path is byte-for-byte the conservative (0,0) no-horizon behavior.
func TestHeadSessionPriorVolumeAwareHorizon(t *testing.T) {
	now := time.Now()
	newS := func(assume int) *Server { return &Server{assumeSessionTurns: assume, metrics: newGatewayMetrics(now)} }
	prime := func(s *Server, trace string, turns int, cacheRead int64) {
		for i := 0; i < turns; i++ {
			s.metrics.observeHarnessCoherence(trace, now, "", false, "", false, false, cacheRead, 0, 1000)
		}
	}

	// Heavy + deep: horizon extends to served+headroom; CurrentTurn is the real served depth+1.
	sHeavy := newS(DefaultAssumedSessionTurns)
	prime(sHeavy, "heavy", 40, 70000) // resident ~71k >= floor, 40 turns deep
	if total, cur := sHeavy.headSessionPrior(0, "heavy"); total != 40+headHorizonHeavyHeadroom || cur != 41 {
		t.Fatalf("heavy deep prior = (%d,%d), want (%d,41)", total, cur, 40+headHorizonHeavyHeadroom)
	}

	// Token-light trace at the same depth keeps the base horizon — inert for short/thin sessions.
	sThin := newS(DefaultAssumedSessionTurns)
	prime(sThin, "thin", 40, 3000) // resident ~4k, below the floor
	if total, _ := sThin.headSessionPrior(0, "thin"); total != DefaultAssumedSessionTurns {
		t.Fatalf("token-light TotalTurns = %d, want base %d", total, DefaultAssumedSessionTurns)
	}

	// A wired bounded horizon still wins over the prior (unchanged precedence).
	if total, cur := sHeavy.headSessionPrior(9, "heavy"); total != 10 || cur != 1 {
		t.Fatalf("wired turnsLeft=9 must win: got (%d,%d), want (10,1)", total, cur)
	}

	// Prior disabled (assumeSessionTurns<=0) → (0,0) even for a heavy deep trace.
	sOff := newS(0)
	prime(sOff, "heavy", 40, 70000)
	if total, cur := sOff.headSessionPrior(0, "heavy"); total != 0 || cur != 0 {
		t.Fatalf("prior disabled must return (0,0), got (%d,%d)", total, cur)
	}
}
