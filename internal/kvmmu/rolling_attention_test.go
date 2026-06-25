package kvmmu_test

import (
	"math"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/kvmmu"
)

// rolling_attention_test.go — acceptance for issue #855 (rolling per-span attention
// accumulator: EMA + cumulative, one accumulator two lambda).
//
// These tests drive a KNOWN per-turn mass stream through AttributeRow + CloseTurn (the
// turn-fold), so the temporal reduction math is what is under test — not the softmax
// numerics (#852) nor the per-row attribution partition (#853, covered by attention_test).

// emaCumByID maps each segment id to its (EMA, Cumulative) rolling accumulators.
func rollingByID(c *kvmmu.Context) map[string][2]float64 {
	m := make(map[string][2]float64)
	for _, s := range c.Segments() {
		m[s.ID] = [2]float64{s.EMA, s.Cumulative}
	}
	return m
}

// TestCloseTurnLambdaOneIsCumulative: with lambda == 1 the EMA recurrence
// (EMA = 1*EMA + a) is exactly the undecayed cumulative sum — the identity #855 requires.
func TestCloseTurnLambdaOneIsCumulative(t *testing.T) {
	c := newAttnCtx(t)
	c.Append("A", "toolA", []int{1, 2}) // positions 0,1

	// A fixed per-turn mass stream on A: 0.3, 0.1, 0.5, 0.2 over four turns.
	stream := []float32{0.3, 0.1, 0.5, 0.2}
	var want float64
	for _, a := range stream {
		c.AttributeRow([]int{0}, []float32{a}) // all mass on position 0 (inside A)
		c.CloseTurn(1.0)
		want += float64(a)
	}

	got := rollingByID(c)["A"]
	if math.Abs(got[0]-got[1]) > 1e-9 {
		t.Errorf("lambda=1: EMA=%v != Cumulative=%v (identity broken)", got[0], got[1])
	}
	if math.Abs(got[1]-want) > 1e-6 {
		t.Errorf("Cumulative=%v, want %v (Σ stream)", got[1], want)
	}
}

// TestCloseTurnHotThenColdDiverges: a span hot early then idle ends with a HIGH cumulative
// (it mattered overall) but a LOW EMA (it is not hot now) — the two reductions diverge,
// which is the whole point of one-accumulator-two-lambda. A second span cold-then-hot is
// the mirror: low-ish cumulative but high EMA. Proven side by side.
func TestCloseTurnHotThenColdDiverges(t *testing.T) {
	c := newAttnCtx(t)
	c.Append("HOT", "toolH", []int{1, 2})  // positions 0,1  — hot early, then idle
	c.Append("COLD", "toolC", []int{3, 4}) // positions 2,3  — idle early, then hot

	const lambda = 0.5
	// 6 turns. HOT attends 1.0 for the first 3 turns then 0; COLD is the mirror.
	hotStream := []float32{1, 1, 1, 0, 0, 0}
	coldStream := []float32{0, 0, 0, 1, 1, 1}
	for i := range hotStream {
		c.AttributeRow([]int{0}, []float32{hotStream[i]})  // HOT (position 0)
		c.AttributeRow([]int{2}, []float32{coldStream[i]}) // COLD (position 2)
		c.CloseTurn(lambda)
	}

	got := rollingByID(c)
	hot, cold := got["HOT"], got["COLD"]

	// Cumulative is identical (both attended 3.0 total) — undecayed sees no difference.
	if math.Abs(hot[1]-3.0) > 1e-6 || math.Abs(cold[1]-3.0) > 1e-6 {
		t.Fatalf("cumulative HOT=%v COLD=%v, want both 3.0", hot[1], cold[1])
	}
	// EMA diverges sharply: HOT decayed to near-zero (last 3 turns it got nothing, halved
	// each turn), COLD is high (it just ran hot). This is the hot-now vs mattered-overall
	// split that a recency-only or cumulative-only signal cannot make.
	if hot[0] >= cold[0] {
		t.Errorf("EMA HOT=%v should be << COLD=%v (hot-then-cold has low EMA despite equal cumulative)", hot[0], cold[0])
	}
	// Quantitatively: HOT EMA = (((1*.5+1)*.5+1)*.5)*.5*.5 ... compute the exact value.
	// HOT: turns 1.0,1.0,1.0,0,0,0 with lambda 0.5:
	//   t1: .5*0 + 1 = 1
	//   t2: .5*1 + 1 = 1.5
	//   t3: .5*1.5 + 1 = 1.75
	//   t4: .5*1.75 + 0 = 0.875
	//   t5: .5*0.875 = 0.4375
	//   t6: .5*0.4375 = 0.21875
	if math.Abs(hot[0]-0.21875) > 1e-9 {
		t.Errorf("HOT EMA = %v, want 0.21875 (deterministic decay)", hot[0])
	}
	// COLD: 0,0,0,1,1,1 → t4:1, t5:1.5, t6:1.75
	if math.Abs(cold[0]-1.75) > 1e-9 {
		t.Errorf("COLD EMA = %v, want 1.75 (deterministic)", cold[0])
	}
	t.Logf("side by side: HOT {EMA %.5f, cum %.1f} vs COLD {EMA %.5f, cum %.1f} — equal cumulative, EMA diverges 8x", hot[0], hot[1], cold[0], cold[1])
}

// TestTrajectoryReconstructsHotTurns: the bounded trajectory ring records {a_s(t)} per
// turn in chronological order, so a post-hoc analyst can see exactly which turns a span
// was attended.
func TestTrajectoryReconstructsHotTurns(t *testing.T) {
	c := newAttnCtx(t)
	c.Append("A", "toolA", []int{1, 2}) // 0,1

	stream := []float32{0, 0.4, 0, 0.9, 0}
	for _, a := range stream {
		c.AttributeRow([]int{0}, []float32{a})
		c.CloseTurn(0.9)
	}

	traj := c.Trajectory("A")
	if len(traj) != len(stream) {
		t.Fatalf("trajectory len = %d, want %d", len(traj), len(stream))
	}
	for i, a := range stream {
		if math.Abs(traj[i]-float64(a)) > 1e-6 {
			t.Errorf("traj[%d] = %v, want %v", i, traj[i], a)
		}
	}
	// The hot turns are reconstructable: turns 1 and 3 (0-indexed) carried mass.
	if traj[1] == 0 || traj[3] == 0 {
		t.Errorf("expected nonzero mass on turns 1 and 3, got traj=%v", traj)
	}
}

// TestTrajectoryRingIsBounded: more than trajCap turns keeps only the most recent trajCap
// entries (the ring overwrites oldest), so per-span memory stays O(cap), and the retained
// window is chronological and correct.
func TestTrajectoryRingIsBounded(t *testing.T) {
	c := newAttnCtx(t)
	c.Append("A", "toolA", []int{1, 2}) // 0,1

	// Run trajCap+10 turns; turn t attends mass float64(t+1)/1000 so each is distinct.
	const extra = 10
	total := kvmmu.TrajCapForTest() + extra
	for t0 := 0; t0 < total; t0++ {
		c.AttributeRow([]int{0}, []float32{float32(t0+1) / 1000})
		c.CloseTurn(1.0)
	}
	traj := c.Trajectory("A")
	if len(traj) != kvmmu.TrajCapForTest() {
		t.Fatalf("trajectory len = %d, want trajCap %d (ring must be bounded)", len(traj), kvmmu.TrajCapForTest())
	}
	// The retained window is the LAST trajCap turns, oldest-first. The oldest retained
	// turn is index `extra` (0-based), so its mass is float64(extra+1)/1000.
	wantOldest := float64(extra+1) / 1000
	if math.Abs(traj[0]-wantOldest) > 1e-9 {
		t.Errorf("oldest retained traj[0] = %v, want %v (ring kept the most recent window)", traj[0], wantOldest)
	}
	wantNewest := float64(total) / 1000
	if math.Abs(traj[len(traj)-1]-wantNewest) > 1e-9 {
		t.Errorf("newest traj[-1] = %v, want %v", traj[len(traj)-1], wantNewest)
	}
}

// TestCloseTurnEvictResetsAllAccumulators: evicting a span clears its EMA, Cumulative, and
// trajectory (consistent with rung-2 zeroing Attended) while survivors keep theirs.
func TestCloseTurnEvictResetsAllAccumulators(t *testing.T) {
	c := newAttnCtx(t)
	c.Append("A", "toolA", []int{1, 2}) // 0,1
	c.Append("B", "toolB", []int{3, 4}) // 2,3

	// Two turns; both spans accumulate cumulative + EMA + trajectory.
	for i := 0; i < 2; i++ {
		c.AttributeRow([]int{0, 2}, []float32{0.5, 0.5})
		c.CloseTurn(0.5)
	}
	before := rollingByID(c)
	if before["A"][1] == 0 || before["B"][1] == 0 {
		t.Fatalf("setup: expected nonzero cumulative, got A=%v B=%v", before["A"], before["B"])
	}

	// Evict B. Its accumulators must all clear; A's must be untouched.
	if ev, ok := c.Quarantine("B"); !ok || ev == 0 {
		t.Fatalf("Quarantine(B) = (%d,%v)", ev, ok)
	}
	after := rollingByID(c)
	if after["B"][0] != 0 || after["B"][1] != 0 {
		t.Errorf("B after evict: EMA=%v cum=%v, want both 0", after["B"][0], after["B"][1])
	}
	if len(c.Trajectory("B")) != 0 {
		t.Errorf("B trajectory after evict = %v, want empty", c.Trajectory("B"))
	}
	if math.Abs(after["A"][0]-before["A"][0]) > 1e-12 || math.Abs(after["A"][1]-before["A"][1]) > 1e-12 {
		t.Errorf("A accumulators changed by evict of B: before=%v after=%v", before["A"], after["A"])
	}
}

// TestCloseTurnDeterministic: the same per-turn mass stream produces byte-identical
// accumulators across two independent Contexts — the fold has no hidden state.
func TestCloseTurnDeterministic(t *testing.T) {
	run := func() [2]float64 {
		c := newAttnCtx(t)
		c.Append("A", "toolA", []int{1, 2})
		for _, a := range []float32{0.2, 0.7, 0.1, 0.4} {
			c.AttributeRow([]int{0}, []float32{a})
			c.CloseTurn(0.8)
		}
		r := rollingByID(c)["A"]
		return r
	}
	a, b := run(), run()
	if a != b {
		t.Errorf("non-deterministic fold: %v != %v", a, b)
	}
}
