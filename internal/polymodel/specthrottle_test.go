package polymodel

import "testing"

// specthrottle_test.go — witnesses the adaptive draft-length throttle (#5257): high
// acceptance grows K toward the max, poor acceptance shrinks it toward the floor and then
// stops, a mid band holds steady (hysteresis, no flapping), the stop fires on persistently
// poor acceptance, and degenerate rounds fail closed to a full stop. All deterministic,
// no GPU, no wall clock — the throttle is fed plain (accepted, proposed) counts.

// feed folds a run of identical (accepted, proposed) rounds and returns the last verdict.
func feed(t *DraftLengthThrottle, accepted, proposed, rounds int) ThrottleVerdict {
	var v ThrottleVerdict
	for i := 0; i < rounds; i++ {
		v = t.Record(accepted, proposed)
	}
	return v
}

// TestDraftLengthThrottleGrow: sustained high acceptance grows K one step per round and
// caps at Max, never overshooting.
func TestDraftLengthThrottleGrow(t *testing.T) {
	th := NewDraftLengthThrottle(DraftLengthThrottleConfig{
		Window: 1, Low: 0.3, High: 0.7, Floor: 1, Max: 3, Step: 1, Start: 1,
	})
	// Every round accepts all proposed → rate 1.0 > High → grow.
	v := feed(th, 4, 4, 10)
	if v.Length != 3 {
		t.Fatalf("grow should cap at Max=3, got Length=%d (reason=%s rate=%.2f)", v.Length, v.Reason, v.Rate)
	}
	if !v.Drafting || v.Reason != "grow" {
		t.Fatalf("high acceptance should keep drafting via grow, got drafting=%v reason=%s", v.Drafting, v.Reason)
	}
	// Monotone non-overshoot: at no point does Length exceed Max.
	th2 := NewDraftLengthThrottle(DraftLengthThrottleConfig{Window: 1, High: 0.7, Max: 3, Start: 1})
	for i := 0; i < 10; i++ {
		if got := th2.Record(2, 2); got.Length > 3 {
			t.Fatalf("round %d overshot Max=3: Length=%d", i, got.Length)
		}
	}
}

// TestDraftLengthThrottleShrinkToFloor: sustained poor acceptance shrinks K toward the
// floor and HOLDS at the floor (never below, never 0) when the stop streak is out of reach.
func TestDraftLengthThrottleShrinkToFloor(t *testing.T) {
	th := NewDraftLengthThrottle(DraftLengthThrottleConfig{
		Window: 1, Low: 0.3, High: 0.7, Floor: 1, Max: 8, Step: 1, StopStreak: 1000, Start: 5,
	})
	// Zero acceptance → rate 0 < Low → shrink each round toward the floor.
	prev := 5
	for i := 0; i < 20; i++ {
		v := th.Record(0, 4)
		if v.Length > prev {
			t.Fatalf("round %d grew under poor acceptance: %d -> %d", i, prev, v.Length)
		}
		if v.Length < 1 {
			t.Fatalf("round %d fell below Floor=1: Length=%d", i, v.Length)
		}
		prev = v.Length
	}
	if th.Length() != 1 || !th.Drafting() {
		t.Fatalf("should converge to and hold at Floor=1 (still drafting), got Length=%d drafting=%v", th.Length(), th.Drafting())
	}
}

// TestDraftLengthThrottleStop: persistently poor acceptance trips a full stop once the
// below-Low streak reaches StopStreak (K=0, drafting off).
func TestDraftLengthThrottleStop(t *testing.T) {
	th := NewDraftLengthThrottle(DraftLengthThrottleConfig{
		Window: 2, Low: 0.3, High: 0.7, Floor: 1, Max: 8, Step: 1, StopStreak: 3, Start: 5,
	})
	// r1: warmup (window not full) → hold at Start.
	if v := th.Record(0, 4); v.Reason != "warmup" || v.Length != 5 {
		t.Fatalf("first round should warm up holding Start, got %+v", v)
	}
	// r2, r3: window full, poor → shrink (poor streak 1, 2). Not yet stopped.
	th.Record(0, 4)
	if v := th.Record(0, 4); v.Reason != "shrink" || !v.Drafting {
		t.Fatalf("pre-stop rounds should shrink and keep drafting, got %+v", v)
	}
	// r4: poor streak reaches StopStreak=3 → stop.
	v := th.Record(0, 4)
	if v.Reason != "stop" || v.Length != 0 || v.Drafting {
		t.Fatalf("persistently poor acceptance should stop (K=0, drafting off), got %+v", v)
	}
}

// TestDraftLengthThrottleHold: a mid-band acceptance rate holds K steady across many
// rounds — the low/high band is the hysteresis that prevents flapping on noise.
func TestDraftLengthThrottleHold(t *testing.T) {
	th := NewDraftLengthThrottle(DraftLengthThrottleConfig{
		Window: 1, Low: 0.3, High: 0.7, Floor: 1, Max: 8, Step: 1, Start: 3,
	})
	// rate 0.5 sits inside [Low, High] → hold at Start every round, no drift.
	for i := 0; i < 25; i++ {
		v := th.Record(1, 2)
		if v.Length != 3 || v.Reason != "hold" {
			t.Fatalf("mid band should hold Length=3, round %d got Length=%d reason=%s", i, v.Length, v.Reason)
		}
	}
}

// TestDraftLengthThrottleDegenerate: degenerate rounds (nothing proposed, negative counts,
// or more accepted than proposed) fail closed to a full stop and do not corrupt state.
func TestDraftLengthThrottleDegenerate(t *testing.T) {
	cases := []struct {
		name               string
		accepted, proposed int
	}{
		{"zero proposed", 0, 0},
		{"negative proposed", 1, -1},
		{"negative accepted", -1, 4},
		{"accepted exceeds proposed", 5, 4},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			th := NewDraftLengthThrottle(DraftLengthThrottleConfig{Window: 1, Max: 8, Start: 4})
			// Warm it up positive first, then hit it with the degenerate round.
			feed(th, 4, 4, 3)
			v := th.Record(tc.accepted, tc.proposed)
			if v.Reason != "degenerate" || v.Length != 0 || v.Drafting {
				t.Fatalf("%s should fail closed to stop, got %+v", tc.name, v)
			}
		})
	}
}

// TestDraftLengthThrottleWarmup: before the window fills, the throttle holds the Start
// length and reports a zero rate (not enough samples to adapt on).
func TestDraftLengthThrottleWarmup(t *testing.T) {
	th := NewDraftLengthThrottle(DraftLengthThrottleConfig{Window: 4, Start: 2, Max: 8})
	for i := 0; i < 3; i++ { // 3 < Window=4 → all warmup
		v := th.Record(4, 4)
		if v.Reason != "warmup" || v.Length != 2 || v.Rate != 0 {
			t.Fatalf("warmup round %d should hold Start=2 at rate 0, got %+v", i, v)
		}
	}
	// The 4th sample fills the window and now adapts (high rate → grow).
	if v := th.Record(4, 4); v.Reason != "grow" || v.Length != 3 {
		t.Fatalf("filling the window should adapt (grow), got %+v", v)
	}
}

// TestDraftLengthThrottleResume: after a stop, a high-acceptance probe round resumes
// drafting from the floor — the throttle grows speculation back when it starts paying again.
func TestDraftLengthThrottleResume(t *testing.T) {
	th := NewDraftLengthThrottle(DraftLengthThrottleConfig{
		Window: 1, Low: 0.3, High: 0.7, Floor: 1, Max: 4, Step: 1, StopStreak: 1, Start: 3,
	})
	// One poor round with StopStreak=1 → immediate stop.
	if v := th.Record(0, 4); v.Reason != "stop" || v.Drafting {
		t.Fatalf("poor round should stop, got %+v", v)
	}
	// A high-acceptance probe resumes drafting from the floor.
	v := th.Record(4, 4)
	if v.Reason != "grow" || !v.Drafting || v.Length != 1 {
		t.Fatalf("high probe after stop should resume at Floor=1, got %+v", v)
	}
	// Continued high acceptance keeps growing (toward Max).
	if v := th.Record(4, 4); v.Length != 2 {
		t.Fatalf("resumed drafting should keep growing, got Length=%d", v.Length)
	}
}
