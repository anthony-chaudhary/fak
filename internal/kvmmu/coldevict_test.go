package kvmmu_test

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/ctxmmu"
	"github.com/anthony-chaudhary/fak/internal/kvmmu"
	"github.com/anthony-chaudhary/fak/internal/model"
)

// coldevict_test.go — issue #856, rung 5 of the attention-witness epic (#851):
// attention-informed AND bit-exact eviction. The controller picks WHICH span to drop
// by coldness (lowest EMA, the recency-decayed attention mass from #855); the existing
// exact evictor (model.KVCache.Evict, re-RoPE + renumber) makes a WRITE-TIME / tail
// removal byte-identical to a run that never saw the span. These tests witness both
// halves: the SELECTION (coldest-first, pinned-excluded, attention-not-recency) and the
// EXACTNESS on the boundary where never-saw is true. They deliberately do NOT claim that
// retroactively evicting a middle span can un-contaminate later tokens that already
// attended to it; internal/model's eviction witness names that as the hard boundary.

// driveTurn attributes a synthetic attention row that puts `mass` on EVERY position the
// given segment owns, then is the per-turn input the caller folds with CloseTurn. It
// simulates the rung-1/2 observer feeding the rung-4 accumulator, so a span's EMA reflects
// how much it was attended — the real path the controller reads, not a hand-poked field.
func attributeSpanMass(c *kvmmu.Context, seg *kvmmu.Segment, mass float64) {
	if seg.Len == 0 {
		return
	}
	pos := make([]int, seg.Len)
	w := make([]float32, seg.Len)
	for i := 0; i < seg.Len; i++ {
		pos[i] = seg.From + i
		w[i] = float32(mass / float64(seg.Len))
	}
	c.AttributeRow(pos, w)
}

// TestEvictColdestTailEqualsNeverSaw is the load-bearing rung-5 witness: the controller
// drops the COLDEST (lowest-EMA) of two spans, and because that cold span is the tail at
// the eviction boundary, the post-eviction next-token distribution is BIT-IDENTICAL to a
// session that NEVER saw that span. This is the same honest boundary as write-time
// quarantine: nothing downstream has attended to the dropped span yet.
func TestEvictColdestTailEqualsNeverSaw(t *testing.T) {
	m := model.NewSynthetic(synthCfg())
	prefix := []int{1, 2, 3, 4, 5}
	cold := []int{10, 11, 12, 13} // will be attended little -> low EMA -> evicted
	warm := []int{20, 21, 22}     // attended heavily -> high EMA -> kept
	query := []int{30, 31}

	// Reference: a session that only ever prefilled prefix+warm+query (never saw cold).
	lNever := m.NewSession().Prefill(cat(prefix, warm, query))

	s := m.NewSession()
	c := kvmmu.NewWithGate(s, ctxmmu.New())
	c.Append("sys", "system", prefix)
	c.Pin("sys", true) // structural span: the caller pins the system prompt (policy gate)
	_, segWarm := c.Append("warm", "read_b", warm)
	_, segCold := c.Append("cold", "read_a", cold)

	// One turn of attention: warm gets 10x the mass of cold, then close the turn so the
	// EMA reflects it. lambda < 1 is the real-time controller knob.
	attributeSpanMass(c, segCold, 0.1)
	attributeSpanMass(c, segWarm, 1.0)
	c.CloseTurn(0.5)

	if segCold.EMA >= segWarm.EMA {
		t.Fatalf("test setup wrong: cold EMA %.4f should be < warm EMA %.4f", segCold.EMA, segWarm.EMA)
	}

	// Evict under pressure: free at least len(cold) positions. The coldest (cold) must go.
	ev := c.EvictColdest(len(cold))
	if len(ev) != 1 || ev[0].ID != "cold" {
		t.Fatalf("EvictColdest dropped %+v, want exactly the cold span", ev)
	}
	if c.CacheLen() != len(prefix)+len(warm) {
		t.Fatalf("after evicting cold, cache len = %d, want %d", c.CacheLen(), len(prefix)+len(warm))
	}

	// The witness: append the query and compare to never-saw-cold. Must be bit-identical.
	lGot, _ := c.Append("usr", "user", query)
	d := maxAbsDiff(lGot, lNever)
	t.Logf("max|Δ| coldest-evict-vs-never = %.3e (want 0); evicted id=%s EMA=%.4f", d, ev[0].ID, ev[0].EMA)
	if d != 0 {
		t.Fatalf("coldest-evicted distribution != never-saw-cold (max|Δ|=%.3e); attention-informed eviction is NOT bit-exact", d)
	}
}

// TestPinnedSurvivesAtZeroEMA proves the policy gate: a PINNED span is never evicted by
// the coldness controller even at EMA == 0 (the coldest possible). The system prompt /
// active goal stays resident under budget pressure while a warmer unpinned span is dropped
// instead — coldness proposes, the pin disposes.
func TestPinnedSurvivesAtZeroEMA(t *testing.T) {
	m := model.NewSynthetic(synthCfg())
	prefix := []int{1, 2, 3}
	pinnedSpan := []int{10, 11} // EMA stays 0 (never attended) but PINNED
	other := []int{20, 21, 22}  // attended -> warmer, unpinned

	s := m.NewSession()
	c := kvmmu.NewWithGate(s, ctxmmu.New())
	c.Append("sys", "system", prefix)
	c.Pin("sys", true)
	_, segPin := c.Append("pinme", "system_goal", pinnedSpan)
	_, segOther := c.Append("other", "read_b", other)

	// Give `other` some attention; leave `pinme` at EMA 0. Then pin `pinme`.
	attributeSpanMass(c, segOther, 1.0)
	c.CloseTurn(0.5)
	if segPin.EMA != 0 {
		t.Fatalf("pinned span should have EMA 0 (never attended), got %.4f", segPin.EMA)
	}
	if !c.Pin("pinme", true) {
		t.Fatal("Pin(pinme) did not find the live segment")
	}

	// Demand to free more positions than `other` alone provides — the controller must
	// still NOT touch the pinned span, even though it is the coldest.
	ev := c.EvictColdest(len(pinnedSpan) + len(other) + 5)
	for _, e := range ev {
		if e.ID == "pinme" {
			t.Fatalf("the PINNED span was evicted (EMA=0) — the policy gate failed: %+v", ev)
		}
	}
	// `other` (the only unpinned candidate) should have been dropped.
	if len(ev) != 1 || ev[0].ID != "other" {
		t.Fatalf("EvictColdest dropped %+v, want only the unpinned 'other'", ev)
	}
	// The pinned span is still resident.
	stillThere := false
	for _, sg := range c.Segments() {
		if sg.ID == "pinme" && !sg.Held {
			stillThere = true
		}
	}
	if !stillThere {
		t.Fatal("pinned span no longer live after EvictColdest")
	}
}

// TestColdBeforeWarm proves the ordering: given three unpinned spans of distinct EMA, the
// controller drops them in ascending-EMA (coldest-first) order, stopping once the target
// is met. With a target that frees exactly two of three spans, the two COLDEST go and the
// warmest stays.
func TestColdBeforeWarm(t *testing.T) {
	m := model.NewSynthetic(synthCfg())
	prefix := []int{1, 2}
	a := []int{10, 11} // coldest
	b := []int{20, 21} // middle
	d := []int{30, 31} // warmest

	s := m.NewSession()
	c := kvmmu.NewWithGate(s, ctxmmu.New())
	c.Append("sys", "system", prefix)
	_, segA := c.Append("a", "ta", a)
	_, segB := c.Append("b", "tb", b)
	_, segD := c.Append("d", "td", d)
	attributeSpanMass(c, segA, 0.1)
	attributeSpanMass(c, segB, 0.5)
	attributeSpanMass(c, segD, 2.0)
	c.CloseTurn(0.5)

	// Free at least len(a)+len(b) positions -> exactly the two coldest (a then b).
	ev := c.EvictColdest(len(a) + len(b))
	if len(ev) != 2 || ev[0].ID != "a" || ev[1].ID != "b" {
		t.Fatalf("eviction order = %+v, want [a, b] coldest-first", ev)
	}
	// The warmest (d) must still be resident.
	for _, sg := range c.Segments() {
		if sg.ID == "d" && sg.Held {
			t.Fatal("warmest span d was evicted before the colder a/b")
		}
	}
	_ = segD
}

// TestAttentionDisagreesWithLRU is the novelty case: attention-informed eviction differs
// from recency-only LRU. A span appended MOST RECENTLY (warmest by recency, so LRU would
// keep it last) but NEVER ATTENDED (coldest by EMA) is evicted BEFORE an OLD span that ran
// hot. A recency-only evictor would drop the old-but-hot span first; the attention
// controller drops the recent-but-unattended one. This is the quality lever the issue
// names — the selection ORDER provably differs from LRU on a case where attention and
// recency disagree.
func TestAttentionDisagreesWithLRU(t *testing.T) {
	m := model.NewSynthetic(synthCfg())
	prefix := []int{1, 2}
	oldHot := []int{10, 11}  // appended FIRST (oldest, lowest From) but attended heavily
	newCold := []int{20, 21} // appended LAST (newest, highest From) but never attended

	s := m.NewSession()
	c := kvmmu.NewWithGate(s, ctxmmu.New())
	c.Append("sys", "system", prefix)
	_, segOld := c.Append("oldhot", "t_old", oldHot)
	_, segNew := c.Append("newcold", "t_new", newCold)

	// oldHot is heavily attended; newCold gets nothing.
	attributeSpanMass(c, segOld, 2.0)
	c.CloseTurn(0.5)

	if segOld.From >= segNew.From {
		t.Fatalf("setup: oldHot.From %d should be < newCold.From %d (oldHot is older)", segOld.From, segNew.From)
	}
	if !(segNew.EMA < segOld.EMA) {
		t.Fatalf("setup: newCold EMA %.4f should be < oldHot EMA %.4f", segNew.EMA, segOld.EMA)
	}

	// Free one span's worth. Attention-informed picks newCold (coldest EMA), NOT the
	// oldest-by-position oldHot that an LRU/recency evictor would drop first.
	ev := c.EvictColdest(len(newCold))
	if len(ev) != 1 {
		t.Fatalf("expected 1 eviction, got %+v", ev)
	}
	if ev[0].ID != "newcold" {
		t.Fatalf("attention controller evicted %q; want 'newcold' (the recent-but-unattended span). "+
			"An LRU/recency evictor would have dropped 'oldhot' (oldest) — the selection must differ.", ev[0].ID)
	}
	// Confirm the LRU choice (oldest span) survived — the divergence is real, not vacuous.
	for _, sg := range c.Segments() {
		if sg.ID == "oldhot" && sg.Held {
			t.Fatal("oldHot (the LRU victim) was evicted — selection did not diverge from LRU")
		}
	}
	t.Logf("attention dropped newcold (EMA %.4f, newest); LRU would drop oldhot (EMA %.4f, oldest) — selection diverges",
		segNew.EMA, segOld.EMA)
}

// TestEvictUnderBudgetEnforcesResidency proves the budget-facing form: when residency
// exceeds the budget, EvictUnderBudget drops coldest spans until residency is back within
// budget; a budget at or above current residency is a no-op.
func TestEvictUnderBudgetEnforcesResidency(t *testing.T) {
	m := model.NewSynthetic(synthCfg())
	prefix := []int{1, 2}
	a := []int{10, 11, 12} // cold
	b := []int{20, 21}     // warm

	s := m.NewSession()
	c := kvmmu.NewWithGate(s, ctxmmu.New())
	c.Append("sys", "system", prefix)
	_, segA := c.Append("a", "ta", a)
	_, segB := c.Append("b", "tb", b)
	attributeSpanMass(c, segA, 0.1)
	attributeSpanMass(c, segB, 1.0)
	c.CloseTurn(0.5)

	total := len(prefix) + len(a) + len(b)
	if c.CacheLen() != total {
		t.Fatalf("cache len = %d, want %d", c.CacheLen(), total)
	}
	// No-op: budget >= current residency.
	if ev := c.EvictUnderBudget(total); ev != nil {
		t.Fatalf("EvictUnderBudget at full budget should be a no-op, dropped %+v", ev)
	}
	// Budget that requires dropping the cold span a (free len(a) positions).
	ev := c.EvictUnderBudget(total - len(a))
	if len(ev) != 1 || ev[0].ID != "a" {
		t.Fatalf("EvictUnderBudget dropped %+v, want the cold span 'a'", ev)
	}
	if c.CacheLen() > total-len(a) {
		t.Fatalf("after EvictUnderBudget, residency %d still over budget %d", c.CacheLen(), total-len(a))
	}
}
