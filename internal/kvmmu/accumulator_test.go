package kvmmu_test

import (
	"math"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/kvmmu"
)

// accumulator_test.go — acceptance for issue #855 (rolling per-span attention accumulator).
//
// The accumulator's core (Observe) is a pure fold over a stream of per-turn per-span masses,
// so most tests drive it directly with a known stream — the temporal reduction math is what is
// under test, not the attribution partition (#853). The Context-integration tests drive it
// through the real Append / AttributeRow / Quarantine machinery to prove ObserveContext /
// ResetAttention / Forget compose with the live ledger.

// approx compares to 1e-6, the same tolerance the sibling attention tests use: masses that
// flow through the ledger originate as float32 attention weights, so float64 equality is too
// tight. The pure-float64 fold tests clear this bound with room to spare.
func approx(t *testing.T, name string, got, want float64) {
	t.Helper()
	if d := math.Abs(got - want); d > 1e-6 {
		t.Errorf("%s = %v, want %v (Δ=%v)", name, got, want, d)
	}
}

// TestHotThenColdDiverges is the headline acceptance: a span hot-then-cold keeps a HIGH
// cumulative (it really did draw that mass) but a LOW EMA (it is not hot now). A steadily-warm
// span with the SAME cumulative keeps a high EMA. The two reductions diverge exactly as the one-
// accumulator-two-λ design intends.
func TestHotThenColdDiverges(t *testing.T) {
	a := kvmmu.NewAttentionAccumulator(0.5, 0)

	// "hot": 1.0 for two turns, then cold (absent) for four.
	// "warm": 0.5 every turn for six turns — same total (3.0) as "hot" (2.0)... make them equal:
	// hot draws 1.5+1.5 = 3.0 over two hot turns; warm draws 0.5 each over six turns = 3.0.
	a.Observe(map[string]float64{"hot": 1.5, "warm": 0.5})
	a.Observe(map[string]float64{"hot": 1.5, "warm": 0.5})
	for i := 0; i < 4; i++ {
		a.Observe(map[string]float64{"warm": 0.5}) // hot is absent → decays
	}

	approx(t, "hot cumulative", a.Cumulative("hot"), 3.0)
	approx(t, "warm cumulative", a.Cumulative("warm"), 3.0)
	if a.Cumulative("hot") != a.Cumulative("warm") {
		t.Fatalf("setup: cumulatives should be equal, got hot=%v warm=%v", a.Cumulative("hot"), a.Cumulative("warm"))
	}

	// Same cumulative, but the cold span's EMA must be far below the warm span's — the divergence.
	if a.EMA("hot") >= a.EMA("warm") {
		t.Errorf("hot EMA %v should be < warm EMA %v (cold span is not hot now)", a.EMA("hot"), a.EMA("warm"))
	}

	// Exact EMA for "hot": after turn2 EMA = 0.5·1.5 + 1.5 = 2.25; then four pure decays by 0.5:
	// 2.25 → 1.125 → 0.5625 → 0.28125 → 0.140625.
	approx(t, "hot EMA", a.EMA("hot"), 2.25*math.Pow(0.5, 4))
}

// TestLambdaOneIsPlainSum: with λ=1 the EMA never decays, so EMA == Cumulative at every step —
// the post-hoc cumulative is the λ=1 special case of the same accumulator.
func TestLambdaOneIsPlainSum(t *testing.T) {
	a := kvmmu.NewAttentionAccumulator(1.0, 0)
	stream := []map[string]float64{
		{"x": 0.4, "y": 0.1},
		{"x": 0.0, "y": 0.7},
		{"x": 0.9},
	}
	for _, turn := range stream {
		a.Observe(turn)
	}
	for _, id := range []string{"x", "y"} {
		if a.EMA(id) != a.Cumulative(id) {
			t.Errorf("λ=1: EMA(%s)=%v != Cumulative(%s)=%v", id, a.EMA(id), id, a.Cumulative(id))
		}
	}
	approx(t, "x cumulative", a.Cumulative("x"), 1.3)
	approx(t, "y cumulative", a.Cumulative("y"), 0.8)
}

// TestLambdaZeroKeepsOnlyLastTurn: with λ=0 the EMA forgets all history each turn, collapsing to
// the most recent turn's mass — the opposite extreme of λ=1.
func TestLambdaZeroKeepsOnlyLastTurn(t *testing.T) {
	a := kvmmu.NewAttentionAccumulator(0, 0)
	a.Observe(map[string]float64{"s": 0.9})
	a.Observe(map[string]float64{"s": 0.2})
	a.Observe(map[string]float64{"s": 0.0}) // last turn cold → EMA 0
	approx(t, "λ=0 EMA == last turn", a.EMA("s"), 0.0)
	approx(t, "λ=0 cumulative still totals", a.Cumulative("s"), 1.1)

	b := kvmmu.NewAttentionAccumulator(0, 0)
	b.Observe(map[string]float64{"s": 0.9})
	b.Observe(map[string]float64{"s": 0.2})
	approx(t, "λ=0 EMA == last (0.2)", b.EMA("s"), 0.2)
}

// TestTrajectoryReconstructsTurns: the per-turn ring records which turns a span was hot, dense
// from its first observed turn (a cold turn is a real 0, not a gap).
func TestTrajectoryReconstructsTurns(t *testing.T) {
	a := kvmmu.NewAttentionAccumulator(0.5, 0)
	a.Observe(map[string]float64{"a": 0.3})           // turn 1
	a.Observe(map[string]float64{"a": 0.0, "b": 0.6}) // turn 2: a cold, b appears
	a.Observe(map[string]float64{"a": 0.7})           // turn 3: a hot, b cold (absent)

	want := []kvmmu.TurnMass{{Turn: 1, Mass: 0.3}, {Turn: 2, Mass: 0.0}, {Turn: 3, Mass: 0.7}}
	got := a.Trajectory("a")
	if len(got) != len(want) {
		t.Fatalf("a trajectory len = %d, want %d (%v)", len(got), len(want), got)
	}
	for i := range want {
		if got[i].Turn != want[i].Turn || math.Abs(got[i].Mass-want[i].Mass) > 1e-9 {
			t.Errorf("a traj[%d] = %+v, want %+v", i, got[i], want[i])
		}
	}

	// b first seen at turn 2, then absent at turn 3 → dense [{2,0.6},{3,0}].
	gb := a.Trajectory("b")
	if len(gb) != 2 || gb[0].Turn != 2 || gb[1].Turn != 3 || gb[1].Mass != 0 {
		t.Errorf("b trajectory = %+v, want [{2,0.6},{3,0}]", gb)
	}
}

// TestDeterministic: the same weight stream yields a byte-equal accumulator — the replay
// discipline every "witnessed" claim in the epic depends on.
func TestDeterministic(t *testing.T) {
	stream := []map[string]float64{
		{"p": 0.5, "q": 0.5},
		{"p": 0.2},
		{"q": 0.9, "r": 0.1},
		{"p": 0.0, "q": 0.0, "r": 0.4},
	}
	run := func() []kvmmu.SpanAttention {
		a := kvmmu.NewAttentionAccumulator(0.7, 0)
		for _, turn := range stream {
			a.Observe(turn)
		}
		return a.Snapshot()
	}
	s1, s2 := run(), run()
	if len(s1) != len(s2) {
		t.Fatalf("snapshot lengths differ: %d vs %d", len(s1), len(s2))
	}
	for i := range s1 {
		if s1[i].ID != s2[i].ID || s1[i].EMA != s2[i].EMA || s1[i].Cumulative != s2[i].Cumulative {
			t.Errorf("snapshot[%d] differs: %+v vs %+v", i, s1[i], s2[i])
		}
	}
}

// TestRingCapBounds: the trajectory ring is bounded — observing more than cap turns keeps only
// the most recent cap entries, so per-span memory is O(cap), not O(turns).
func TestRingCapBounds(t *testing.T) {
	const cap = 3
	a := kvmmu.NewAttentionAccumulator(1.0, cap)
	for i := 1; i <= 7; i++ {
		a.Observe(map[string]float64{"s": float64(i)})
	}
	tr := a.Trajectory("s")
	if len(tr) != cap {
		t.Fatalf("trajectory len = %d, want cap %d", len(tr), cap)
	}
	// Most recent three turns are 5,6,7.
	for i, want := range []int{5, 6, 7} {
		if tr[i].Turn != want || tr[i].Mass != float64(want) {
			t.Errorf("ring[%d] = %+v, want turn %d", i, tr[i], want)
		}
	}
	// Cumulative still totals ALL turns (1..7 = 28), even though the ring dropped early ones.
	approx(t, "cumulative survives ring drop", a.Cumulative("s"), 28)
}

// TestSnapshotSortedHotFirst: Snapshot orders spans hottest-EMA first so #856 can read the cold
// tail and #857 the hot head directly.
func TestSnapshotSortedHotFirst(t *testing.T) {
	a := kvmmu.NewAttentionAccumulator(1.0, 0)
	a.Observe(map[string]float64{"cold": 0.1, "hot": 0.9, "mid": 0.5})
	snap := a.Snapshot()
	if len(snap) != 3 || snap[0].ID != "hot" || snap[1].ID != "mid" || snap[2].ID != "cold" {
		t.Fatalf("snapshot order = %v, want [hot mid cold]", idsOf(snap))
	}
}

func idsOf(s []kvmmu.SpanAttention) []string {
	out := make([]string, len(s))
	for i := range s {
		out[i] = s[i].ID
	}
	return out
}

// TestForgetDropsState: Forget removes a span entirely (the real-time controller's reset-on-evict
// posture). Idempotent.
func TestForgetDropsState(t *testing.T) {
	a := kvmmu.NewAttentionAccumulator(0.5, 0)
	a.Observe(map[string]float64{"s": 1.0})
	if _, ok := a.Span("s"); !ok {
		t.Fatal("s should be known before Forget")
	}
	a.Forget("s")
	if _, ok := a.Span("s"); ok {
		t.Error("s should be gone after Forget")
	}
	if a.EMA("s") != 0 || a.Cumulative("s") != 0 {
		t.Error("forgotten span must read zero")
	}
	a.Forget("s") // idempotent, must not panic
}

// --- Context-integration tests: the accumulator over the real span ledger. ---

// TestObserveContextPerTurnWithReset proves the full turn-boundary loop: AttributeRow fills the
// per-turn Attended, ObserveContext folds it, ResetAttention clears it, and the NEXT turn's fold
// sees per-turn mass (not a running within-session sum).
func TestObserveContextPerTurnWithReset(t *testing.T) {
	c := newAttnCtx(t)
	c.Append("A", "toolA", []int{1, 2}) // positions 0,1
	c.Append("B", "toolB", []int{3, 4}) // positions 2,3

	a := kvmmu.NewAttentionAccumulator(1.0, 0) // λ=1 so EMA==cumulative, easy to reason about

	// Turn 1: all mass on A.
	c.AttributeRow([]int{0, 1, 2, 3}, []float32{0.5, 0.5, 0, 0})
	a.ObserveContext(c)
	c.ResetAttention()

	// ResetAttention must clear the per-turn scalar so turn 2 starts fresh.
	if m := attendedByID(c); m["A"] != 0 || m["B"] != 0 {
		t.Fatalf("after ResetAttention, Attended = %+v, want all 0", m)
	}

	// Turn 2: all mass on B.
	c.AttributeRow([]int{0, 1, 2, 3}, []float32{0, 0, 0.5, 0.5})
	a.ObserveContext(c)
	c.ResetAttention()

	// Each span drew 1.0 over its own single turn — per-turn, not summed into a 2.0 blob.
	approx(t, "A cumulative", a.Cumulative("A"), 1.0)
	approx(t, "B cumulative", a.Cumulative("B"), 1.0)
	// Trajectory shows the timing: A hot at turn 1, cold at turn 2; B the reverse.
	ta, tb := a.Trajectory("A"), a.Trajectory("B")
	if len(ta) != 2 || math.Abs(ta[0].Mass-1.0) > 1e-9 || ta[1].Mass != 0 {
		t.Errorf("A trajectory = %+v, want hot@1 cold@2", ta)
	}
	if len(tb) != 2 || tb[0].Mass != 0 || math.Abs(tb[1].Mass-1.0) > 1e-9 {
		t.Errorf("B trajectory = %+v, want cold@1 hot@2", tb)
	}
}

// TestObserveContextEvictForget: an evicted span's per-turn mass is already zeroed by the ledger,
// and the real-time controller's Forget drops its accumulator state too — reset-on-evict end to end.
func TestObserveContextEvictForget(t *testing.T) {
	c := newAttnCtx(t)
	c.Append("A", "toolA", []int{1, 2}) // 0,1
	c.Append("B", "toolB", []int{3, 4}) // 2,3

	a := kvmmu.NewAttentionAccumulator(0.5, 0)
	c.AttributeRow([]int{0, 1, 2, 3}, []float32{0.3, 0.3, 0.2, 0.2}) // A=0.6, B=0.4
	a.ObserveContext(c)
	c.ResetAttention()

	if _, ok := a.Span("A"); !ok {
		t.Fatal("A should be known after first fold")
	}

	// Evict A: the ledger zeroes A's mass; the controller forgets A from the accumulator.
	if ev, ok := c.Quarantine("A"); !ok || ev == 0 {
		t.Fatalf("Quarantine(A) = (%d,%v)", ev, ok)
	}
	a.Forget("A")

	// A's per-turn scalar is gone from the ledger, and the next fold sees only B.
	c.AttributeRow([]int{0, 1}, []float32{0.5, 0.5}) // B (renumbered to 0,1) = 1.0
	a.ObserveContext(c)

	if _, ok := a.Span("A"); ok {
		t.Error("A should be forgotten from the accumulator after evict+Forget")
	}
	// B kept its history and added the new turn: 0.4 then 1.0.
	approx(t, "B cumulative", a.Cumulative("B"), 1.4)
}
