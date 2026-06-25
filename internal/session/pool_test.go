package session

import (
	"sync"
	"testing"
)

// pool_test.go — the gate for the optional fleet-wide budget pool (#744): the shared
// ceiling enforces a single cap across N sessions, a reset DRAWS a fresh window out of it,
// and the Snapshot surface can REPORT against it. The headline invariant is conservation —
// the sum of every grant never exceeds the configured ceiling, even under concurrent draws.

// --- a bounded pool enforces a single ceiling across N draws -------------------

func TestPoolDrawEnforcesCeiling(t *testing.T) {
	p := NewPool(150_000)
	if got := p.Cap(); got != 150_000 {
		t.Fatalf("Cap = %d, want 150000", got)
	}
	// Three sessions each ask for 60k against a 150k pool: 60k+60k granted in full, the
	// third gets the remaining 30k as a partial grant (ok=false), and a fourth gets 0.
	if g, ok := p.Draw(60_000); g != 60_000 || !ok {
		t.Fatalf("draw#1 = (%d,%v), want (60000,true)", g, ok)
	}
	if g, ok := p.Draw(60_000); g != 60_000 || !ok {
		t.Fatalf("draw#2 = (%d,%v), want (60000,true)", g, ok)
	}
	if g, ok := p.Draw(60_000); g != 30_000 || ok {
		t.Fatalf("draw#3 = (%d,%v), want (30000,false) — partial grant, pool now dry", g, ok)
	}
	if r := p.Remaining(); r != 0 {
		t.Fatalf("Remaining after exhaustion = %d, want 0", r)
	}
	if g, ok := p.Draw(1); g != 0 || ok {
		t.Fatalf("draw#4 from a dry pool = (%d,%v), want (0,false)", g, ok)
	}
}

// --- an unbounded / nil pool never tightens -----------------------------------

func TestPoolUnboundedAndNilNeverTighten(t *testing.T) {
	for name, p := range map[string]*Pool{"zero-ceiling": NewPool(0), "negative": NewPool(-5), "nil": nil} {
		if g, ok := p.Draw(1_000_000); g != 1_000_000 || !ok {
			t.Fatalf("%s: Draw should grant in full, got (%d,%v)", name, g, ok)
		}
		if r := p.Remaining(); r != Unbounded {
			t.Fatalf("%s: Remaining = %d, want Unbounded(%d)", name, r, Unbounded)
		}
		if rep := p.Report(); rep.Bounded || rep.Remaining != Unbounded || rep.Drawn != 0 {
			t.Fatalf("%s: Report = %+v, want Bounded=false Remaining=Unbounded Drawn=0", name, rep)
		}
	}
}

// --- Return restores headroom but never above the ceiling ---------------------

func TestPoolReturnClampsToCeiling(t *testing.T) {
	p := NewPool(1000)
	p.Draw(800)
	p.Return(300) // would be 500, but ceiling is 1000 -> clamp not triggered, remaining 500
	if r := p.Remaining(); r != 500 {
		t.Fatalf("Remaining after draw(800)+return(300) = %d, want 500", r)
	}
	p.Return(9999) // over-return: a double-Return cannot inflate past the ceiling
	if r := p.Remaining(); r != 1000 {
		t.Fatalf("Remaining after over-Return = %d, want clamped 1000", r)
	}
	// Return is a no-op on an unbounded pool (nothing to restore against).
	un := NewPool(0)
	un.Return(123)
	if r := un.Remaining(); r != Unbounded {
		t.Fatalf("unbounded Remaining after Return = %d, want Unbounded", r)
	}
}

// --- Report renders the fleet-wide row a scheduler reads ----------------------

func TestPoolReportFields(t *testing.T) {
	p := NewPool(1000)
	p.Draw(250)
	rep := p.Report()
	want := PoolReport{Cap: 1000, Drawn: 250, Remaining: 750, Bounded: true}
	if rep != want {
		t.Fatalf("Report = %+v, want %+v", rep, want)
	}
}

// --- a reset DRAWS a fresh window out of the pool -----------------------------

func TestRecontinuePooledDrawsFromPool(t *testing.T) {
	tbl := NewTable()
	p := NewPool(10_000)

	// Drain the parent so the reset is the legitimate follow-on write.
	tbl.SetBudget("sess-a", Budget{TurnsLeft: Unbounded, TokensLeft: 0})

	// Reset wants a fresh 7k window: fully funded, child armed with TokensLeft=7000.
	child, ok := tbl.RecontinuePooled("sess-a", "sess-a-2", Budget{TurnsLeft: Unbounded, TokensLeft: 7_000}, p)
	if !ok {
		t.Fatalf("first pooled reset ok=false, want true (pool had headroom)")
	}
	if child.Budget.TokensLeft != 7_000 {
		t.Fatalf("child TokensLeft = %d, want 7000 (the granted draw)", child.Budget.TokensLeft)
	}
	if child.ParentTrace != "sess-a" || child.Generation != 1 || child.Reason != ReasonBudgetReset {
		t.Fatalf("child lineage = {parent:%q gen:%d reason:%q}, want {sess-a 1 %s}",
			child.ParentTrace, child.Generation, child.Reason, ReasonBudgetReset)
	}
	// The parent record is left exactly as the drain left it (budget exhaustion observable).
	if parent := tbl.Get("sess-a"); parent.Budget.TokensLeft != 0 {
		t.Fatalf("parent TokensLeft = %d, want 0 (drain left intact)", parent.Budget.TokensLeft)
	}

	// A second reset wants 7k but only 3k remains: clamped to 3k, ok=false (fleet ceiling hit).
	child2, ok := tbl.RecontinuePooled("sess-b", "sess-b-2", Budget{TurnsLeft: Unbounded, TokensLeft: 7_000}, p)
	if ok {
		t.Fatalf("second pooled reset ok=true, want false (pool only had 3000 left)")
	}
	if child2.Budget.TokensLeft != 3_000 {
		t.Fatalf("clamped child TokensLeft = %d, want 3000", child2.Budget.TokensLeft)
	}
	if r := p.Remaining(); r != 0 {
		t.Fatalf("pool Remaining after two resets = %d, want 0", r)
	}
}

// --- no pool / unbounded per-session budget is byte-identical to Recontinue ----

func TestRecontinuePooledNilPoolMatchesRecontinue(t *testing.T) {
	tblA, tblB := NewTable(), NewTable()
	want := Budget{TurnsLeft: 5, TokensLeft: 4096, ContextTokensLeft: 2048}

	pooled, ok := tblA.RecontinuePooled("p", "c", want, nil)
	plain := tblB.Recontinue("p", "c", want)
	if !ok {
		t.Fatalf("nil-pool reset ok=false, want true (no ceiling to refuse)")
	}
	if pooled.Budget != plain.Budget || pooled.Generation != plain.Generation || pooled.Reason != plain.Reason {
		t.Fatalf("nil-pool reset = %+v, want byte-identical to Recontinue %+v", pooled, plain)
	}

	// An UNBOUNDED per-session token budget opts out of the shared cap: it neither draws
	// nor reports against the pool, even when a bounded pool is supplied.
	p := NewPool(100)
	child, ok := tblA.RecontinuePooled("p2", "c2", Budget{TurnsLeft: Unbounded, TokensLeft: Unbounded}, p)
	if !ok || !child.Budget.tokensUnbounded() {
		t.Fatalf("unbounded-token reset = (%+v,%v), want unbounded TokensLeft and ok=true", child.Budget, ok)
	}
	if r := p.Remaining(); r != 100 {
		t.Fatalf("pool Remaining after an unbounded-token reset = %d, want 100 (undrawn)", r)
	}
}

// --- concurrent draws conserve the ceiling exactly ----------------------------

func TestPoolDrawConcurrentConservesCeiling(t *testing.T) {
	const ceiling = 100_000
	const per = 1_000
	p := NewPool(ceiling)

	var mu sync.Mutex
	total := 0
	var wg sync.WaitGroup
	// 200 racing draws of 1k each want 200k from a 100k pool: exactly 100 must succeed and
	// the granted sum must equal the ceiling — no over-draw, no lost tokens.
	for i := 0; i < 200; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			g, _ := p.Draw(per)
			mu.Lock()
			total += g
			mu.Unlock()
		}()
	}
	wg.Wait()
	if total != ceiling {
		t.Fatalf("sum of concurrent grants = %d, want exactly the ceiling %d (conservation)", total, ceiling)
	}
	if r := p.Remaining(); r != 0 {
		t.Fatalf("Remaining after concurrent exhaustion = %d, want 0", r)
	}
}
