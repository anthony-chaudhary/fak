package session

import (
	"fmt"
	"sync"
	"testing"
)

// --- RunState wire round-trip -------------------------------------------------

func TestRunStateStringParseRoundTrip(t *testing.T) {
	for _, rs := range []RunState{Running, Throttled, Paused, Draining, Stopped} {
		tok := rs.String()
		got, ok := ParseRunState(tok)
		if !ok || got != rs {
			t.Fatalf("round-trip %v -> %q -> (%v,%v), want (%v,true)", rs, tok, got, ok, rs)
		}
	}
	if s := RunState(99).String(); s != "unknown" {
		t.Fatalf("out-of-range String() = %q, want unknown", s)
	}
	if _, ok := ParseRunState("nonsense"); ok {
		t.Fatal("ParseRunState(nonsense) ok=true, want false (fail closed on an unknown token)")
	}
}

// --- defaults: an unseen session reads a live default, never a phantom stop ----

func TestGetUnseenReturnsDefaultRunning(t *testing.T) {
	tbl := NewTable()
	st := tbl.Get("never-seen")
	if st.TraceID != "never-seen" || st.Run != Running {
		t.Fatalf("unseen Get = %+v, want Running for never-seen", st)
	}
	if !st.Budget.turnsUnbounded() || !st.Budget.tokensUnbounded() {
		t.Fatalf("unseen budget = %+v, want unbounded both axes", st.Budget)
	}
	if tbl.Len() != 0 {
		t.Fatalf("Get must not create a record: Len=%d, want 0", tbl.Len())
	}
}

// --- Rev monotonicity: every write bumps Rev by exactly one -------------------

func TestWritesBumpRevByOne(t *testing.T) {
	tbl := NewTable()
	s1, ok := tbl.SetPriority("s", 5)
	if !ok || s1.Rev != 1 {
		t.Fatalf("first write Rev=%d ok=%v, want 1/true", s1.Rev, ok)
	}
	s2, _ := tbl.SetBudget("s", Budget{TurnsLeft: 3, TokensLeft: Unbounded})
	if s2.Rev != 2 {
		t.Fatalf("second write Rev=%d, want 2", s2.Rev)
	}
	s3, _ := tbl.SetPace("s", Pace{MaxTokensPerTurn: 100})
	if s3.Rev != 3 || s3.Priority != 5 || s3.Budget.TurnsLeft != 3 || s3.Pace.MaxTokensPerTurn != 100 {
		t.Fatalf("third write = %+v, want accumulated fields at Rev 3", s3)
	}
}

// TestTurnIntentIsAdvisoryProjection: the next-turn hint set (#807) records onto State,
// surfaces verbatim through Snapshot, bumps Rev like any write, and is rejected by a
// terminal session. The zero value reads as "no opinion" (IsZero) so an unset intent is
// never mistaken for a positive hint.
func TestTurnIntentIsAdvisoryProjection(t *testing.T) {
	if !(TurnIntent{}).IsZero() {
		t.Fatal("zero TurnIntent must report IsZero (the safe 'fall back to GPU-visible' default)")
	}
	hint := TurnIntent{EndsSoon: true, WillDiscard: true, SharesPrefixWith: "peer", ResultAlreadyKnown: true}
	if hint.IsZero() {
		t.Fatal("a populated TurnIntent must NOT report IsZero")
	}

	tbl := NewTable()
	st, ok := tbl.SetTurnIntent("s", hint)
	if !ok || st.Rev != 1 {
		t.Fatalf("SetTurnIntent ok=%v Rev=%d, want true/1", ok, st.Rev)
	}
	if st.Intent != hint {
		t.Fatalf("recorded intent = %+v, want %+v", st.Intent, hint)
	}

	// The scheduler's read carries it verbatim.
	snap := tbl.Snapshot()
	if len(snap) != 1 || snap[0].Intent != hint {
		t.Fatalf("Snapshot intent = %+v, want it to carry the hint verbatim", snap)
	}

	// A terminal session rejects the advisory write — no resurrection, no intent change.
	tbl2 := NewTable()
	if _, ok := tbl2.Transition("t", Stopped, "done"); !ok {
		t.Fatal("setup: could not stop session")
	}
	if _, ok := tbl2.SetTurnIntent("t", hint); ok {
		t.Fatal("a terminal session must reject SetTurnIntent")
	}
}

// --- the state machine: terminal sessions reject every change ----------------

func TestTransitionTerminalRejectsRevival(t *testing.T) {
	tbl := NewTable()
	if _, ok := tbl.Transition("s", Stopped, "done"); !ok {
		t.Fatal("running -> stopped should be allowed")
	}
	// Every mutation on a stopped session must be refused.
	if _, ok := tbl.Transition("s", Running, ""); ok {
		t.Fatal("stopped -> running ok=true, want false (cannot un-stop)")
	}
	if _, ok := tbl.SetBudget("s", Budget{TurnsLeft: 10}); ok {
		t.Fatal("SetBudget on stopped ok=true, want false")
	}
	if _, ok := tbl.SetPace("s", Pace{MaxTokensPerTurn: 1}); ok {
		t.Fatal("SetPace on stopped ok=true, want false")
	}
	if _, ok := tbl.SetPriority("s", 1); ok {
		t.Fatal("SetPriority on stopped ok=true, want false")
	}
}

func TestTransitionReasonSetAndCleared(t *testing.T) {
	tbl := NewTable()
	st, _ := tbl.Transition("s", Throttled, "gpu-contention")
	if st.Run != Throttled || st.Reason != "gpu-contention" {
		t.Fatalf("throttle = %+v, want Throttled/gpu-contention", st)
	}
	st, _ = tbl.Transition("s", Running, "")
	if st.Run != Running || st.Reason != "" {
		t.Fatalf("resume = %+v, want Running with cleared reason", st)
	}
}

// --- Decide: the per-turn gate ------------------------------------------------

func TestDecideNilTableIsPermissive(t *testing.T) {
	var tbl *Table // nil
	v := tbl.Decide("x")
	if !v.Proceed || v.MaxTokens != 0 || v.Stop {
		t.Fatalf("nil-table Decide = %+v, want permissive proceed (pre-table behavior)", v)
	}
}

func TestDecideUnboundedProceedsForever(t *testing.T) {
	tbl := NewTable()
	for i := 0; i < 1000; i++ {
		if v := tbl.Decide("s"); !v.Proceed {
			t.Fatalf("turn %d: unbounded session stopped: %+v", i, v)
		}
	}
}

func TestDecideTurnBudgetGivesExactlyNTurns(t *testing.T) {
	for _, n := range []int{1, 2, 5} {
		t.Run(fmt.Sprintf("budget=%d", n), func(t *testing.T) {
			tbl := NewTable()
			tbl.SetBudget("s", Budget{TurnsLeft: n, TokensLeft: Unbounded})
			proceeds := 0
			var last Verdict
			for i := 0; i < n+3; i++ {
				v := tbl.Decide("s")
				last = v
				if v.Proceed {
					proceeds++
				}
			}
			if proceeds != n {
				t.Fatalf("budget %d allowed %d turns, want exactly %d", n, proceeds, n)
			}
			if !last.Stop || last.Reason != ReasonBudgetTurns {
				t.Fatalf("after exhaustion last=%+v, want Stop with %s", last, ReasonBudgetTurns)
			}
			if last.State.Run != Stopped {
				t.Fatalf("exhausted session Run=%v, want Stopped", last.State.Run)
			}
		})
	}
}

func TestDecidePausedHoldsWithoutBurningBudget(t *testing.T) {
	tbl := NewTable()
	tbl.SetBudget("s", Budget{TurnsLeft: 3, TokensLeft: Unbounded})
	tbl.Transition("s", Paused, "")
	for i := 0; i < 10; i++ {
		v := tbl.Decide("s")
		if v.Proceed || v.Stop || v.Reason != ReasonPaused {
			t.Fatalf("paused Decide = %+v, want non-proceed non-stop PAUSED", v)
		}
	}
	// Resume and confirm the full budget survived the pause.
	tbl.Transition("s", Running, "")
	proceeds := 0
	for i := 0; i < 6; i++ {
		if tbl.Decide("s").Proceed {
			proceeds++
		}
	}
	if proceeds != 3 {
		t.Fatalf("after resume %d turns ran, want 3 (pause must not burn budget)", proceeds)
	}
}

func TestDecideDrainingTakenAtBoundaryThenStopped(t *testing.T) {
	tbl := NewTable()
	tbl.Transition("s", Draining, "operator-stop")
	v := tbl.Decide("s")
	if v.Proceed || !v.Stop {
		t.Fatalf("draining Decide = %+v, want non-proceed stop", v)
	}
	if v.State.Run != Stopped {
		t.Fatalf("after draining Decide Run=%v, want Stopped (taken at boundary)", v.State.Run)
	}
	// Idempotent: a second Decide on the now-Stopped session still stops, same reason class.
	v2 := tbl.Decide("s")
	if !v2.Stop || v2.State.Run != Stopped {
		t.Fatalf("second Decide on stopped = %+v, want stable Stopped", v2)
	}
}

func TestDecideTokenBudgetExhaustion(t *testing.T) {
	tbl := NewTable()
	tbl.SetBudget("s", Budget{TurnsLeft: Unbounded, TokensLeft: 100})
	// First turn proceeds (100 left, > 0).
	if v := tbl.Decide("s"); !v.Proceed {
		t.Fatalf("first turn with 100 tokens should proceed: %+v", v)
	}
	// Report the turn used 100 tokens -> 0 left.
	tbl.Debit("s", 100)
	// Next turn sees TokensLeft<=0 and drains.
	v := tbl.Decide("s")
	if v.Proceed || !v.Stop || v.Reason != ReasonBudgetTokens {
		t.Fatalf("after token exhaustion Decide = %+v, want stop %s", v, ReasonBudgetTokens)
	}
}

func TestContextBudgetMintsContinuationAndDrains(t *testing.T) {
	tbl := NewTable()
	tbl.SetBudget("s", Budget{TurnsLeft: Unbounded, TokensLeft: Unbounded, ContextTokensLeft: 150})

	st := tbl.DebitUsage("s", Usage{ContextTokens: 151})
	if st.Run != Draining || st.Reason != ReasonBudgetContext || st.ContinuationID == "" {
		t.Fatalf("context debit state = %+v, want draining %s with continuation id", st, ReasonBudgetContext)
	}
	if st.Budget.ContextTokensLeft != 0 {
		t.Fatalf("context budget left = %d, want 0 after exhaustion", st.Budget.ContextTokensLeft)
	}

	v := tbl.Decide("s")
	if v.Proceed || !v.Stop || v.Reason != ReasonBudgetContext || v.State.Run != Stopped {
		t.Fatalf("post-context-exhaustion Decide = %+v, want stopped %s", v, ReasonBudgetContext)
	}
	if v.State.ContinuationID != st.ContinuationID {
		t.Fatalf("continuation changed across drain boundary: %q -> %q", st.ContinuationID, v.State.ContinuationID)
	}
}

// TestRecontinueReArmsDrainedSessionOnFreshTrace proves the human-like reset: a
// budget-drained session is re-armed under its continuation id with a clean budget,
// the fresh session is live with the lineage recorded, and the drained PARENT record
// is left intact (the exhaustion event stays observable, never resurrected in place).
func TestRecontinueReArmsDrainedSessionOnFreshTrace(t *testing.T) {
	tbl := NewTable()
	tbl.SetBudget("s", Budget{TurnsLeft: Unbounded, TokensLeft: Unbounded, ContextTokensLeft: 150})
	drained := tbl.DebitUsage("s", Usage{ContextTokens: 151}) // -> Draining + continuation id
	if v := tbl.Decide("s"); v.State.Run != Stopped {         // take the drain to terminal
		t.Fatalf("expected stopped parent, got %+v", v)
	}
	child := drained.ContinuationID
	if child == "" {
		t.Fatalf("no continuation id minted to recontinue onto")
	}

	fresh := tbl.Recontinue("s", child, Budget{TurnsLeft: Unbounded, TokensLeft: Unbounded, ContextTokensLeft: 150})
	if fresh.TraceID != child || fresh.Run != Running {
		t.Fatalf("fresh session = %+v, want running under %q", fresh, child)
	}
	if fresh.ParentTrace != "s" || fresh.Generation != 1 || fresh.Reason != ReasonBudgetReset {
		t.Fatalf("lineage = parent=%q gen=%d reason=%q, want s/1/%s", fresh.ParentTrace, fresh.Generation, fresh.Reason, ReasonBudgetReset)
	}
	if fresh.Budget.ContextTokensLeft != 150 {
		t.Fatalf("fresh context budget = %d, want re-armed to 150", fresh.Budget.ContextTokensLeft)
	}
	// The fresh session advances; the drained parent stays Stopped (not revived).
	if v := tbl.Decide(child); !v.Proceed {
		t.Fatalf("re-armed session must proceed, got %+v", v)
	}
	if parent := tbl.Get("s"); parent.Run != Stopped {
		t.Fatalf("parent must remain stopped after recontinue, got %+v", parent)
	}
	// A SECOND reset chains the generation off the live child.
	gen2 := tbl.Recontinue(child, "win-2", Budget{TurnsLeft: Unbounded, TokensLeft: Unbounded, ContextTokensLeft: 150})
	if gen2.Generation != 2 || gen2.ParentTrace != child {
		t.Fatalf("second recontinue = gen=%d parent=%q, want 2/%q", gen2.Generation, gen2.ParentTrace, child)
	}
}

// TestSetBudgetStillRefusesTerminal is the no-regression guard: the existing control
// verb must NOT be the reset path — a stopped session still rejects a live SetBudget,
// so Recontinue (a new trace) is the only way to continue past a budget drain.
func TestSetBudgetStillRefusesTerminal(t *testing.T) {
	tbl := NewTable()
	tbl.Transition("s", Stopped, ReasonBudgetContext)
	if _, ok := tbl.SetBudget("s", Budget{TurnsLeft: 5}); ok {
		t.Fatalf("SetBudget on a terminal session must refuse (ok=false)")
	}
}

// TestRecontinueNilTableMintsDetachedChild proves the nil-receiver path is a sane
// no-op-permissive default (a loop with no table still gets a usable fresh child).
func TestRecontinueNilTableMintsDetachedChild(t *testing.T) {
	var tbl *Table
	fresh := tbl.Recontinue("parent", "child", Budget{TurnsLeft: 3})
	if fresh.TraceID != "child" || fresh.ParentTrace != "parent" || fresh.Reason != ReasonBudgetReset {
		t.Fatalf("nil-table recontinue = %+v, want detached child with lineage", fresh)
	}
}

func TestDebitIgnoresTerminalAndUnbounded(t *testing.T) {
	tbl := NewTable()
	// Unbounded token axis: Debit is a no-op on the budget.
	tbl.SetBudget("s", Budget{TurnsLeft: Unbounded, TokensLeft: Unbounded})
	st := tbl.Debit("s", 500)
	if !st.Budget.tokensUnbounded() {
		t.Fatalf("Debit on unbounded changed budget: %+v", st)
	}
	// Stopped session: Debit leaves it untouched.
	tbl.Transition("s2", Stopped, "x")
	before := tbl.Get("s2")
	after := tbl.Debit("s2", 10)
	if after.Rev != before.Rev {
		t.Fatalf("Debit mutated a stopped session: rev %d -> %d", before.Rev, after.Rev)
	}
}

// --- CompareAndSet: optimistic concurrency -----------------------------------

func TestCompareAndSet(t *testing.T) {
	tbl := NewTable()
	s0, _ := tbl.SetPriority("s", 1) // Rev 1
	// Stale expectation is rejected.
	if _, ok := tbl.CompareAndSet("s", 999, State{Priority: 7}); ok {
		t.Fatal("CAS with wrong Rev ok=true, want false")
	}
	// Correct expectation wins and the new Rev is assigned by the table.
	got, ok := tbl.CompareAndSet("s", s0.Rev, State{Priority: 7, Run: Running, Budget: Budget{TurnsLeft: Unbounded, TokensLeft: Unbounded}})
	if !ok || got.Priority != 7 || got.Rev != s0.Rev+1 || got.TraceID != "s" {
		t.Fatalf("CAS = %+v ok=%v, want Priority 7 at Rev %d for s", got, ok, s0.Rev+1)
	}
}

// --- LRU eviction: bounded, default-on-readd ----------------------------------

func TestLRUEvictionBoundsAndDefaultsOnReadd(t *testing.T) {
	tbl := NewTableWithLimit(2)
	tbl.SetPriority("a", 1)
	tbl.SetPriority("b", 2)
	tbl.SetPriority("c", 3) // evicts "a" (LRU)
	if tbl.Len() != 2 {
		t.Fatalf("Len=%d, want 2 (bounded)", tbl.Len())
	}
	// "a" was evicted: it reads the default again (a live default, not a phantom stop).
	a := tbl.Get("a")
	if a.Run != Running || a.Priority != 0 {
		t.Fatalf("evicted-then-read a = %+v, want default Running/0", a)
	}
}

func TestLRUTouchOnWriteKeepsHotEntry(t *testing.T) {
	tbl := NewTableWithLimit(2)
	tbl.SetPriority("a", 1)
	tbl.SetPriority("b", 2)
	tbl.SetPriority("a", 9) // touch "a" -> now "b" is LRU
	tbl.SetPriority("c", 3) // evicts "b", not "a"
	if tbl.Get("a").Priority != 9 {
		t.Fatal("hot entry a was evicted; touch-on-write failed")
	}
	if got := tbl.Get("b"); got.Priority != 0 {
		t.Fatalf("b should have been evicted (default), got Priority %d", got.Priority)
	}
}

// --- Snapshot: the scheduler's read ------------------------------------------

func TestSnapshotSortedForScheduler(t *testing.T) {
	tbl := NewTable()
	tbl.SetPriority("low", 10)
	tbl.SetPriority("hi", 1)
	tbl.SetPriority("mid", 5)
	snap := tbl.Snapshot()
	if len(snap) != 3 {
		t.Fatalf("snapshot len=%d, want 3", len(snap))
	}
	// Priority ascending: hi(1) < mid(5) < low(10) — the order a scheduler yields in.
	if snap[0].TraceID != "hi" || snap[1].TraceID != "mid" || snap[2].TraceID != "low" {
		t.Fatalf("snapshot order = [%s %s %s], want [hi mid low]", snap[0].TraceID, snap[1].TraceID, snap[2].TraceID)
	}
	// The snapshot is a copy: mutating it does not touch the table.
	snap[0].Priority = -1
	if tbl.Get("hi").Priority != 1 {
		t.Fatal("Snapshot returned a live reference, not a copy")
	}
}

func TestSnapshotTieBreakByRevThenTrace(t *testing.T) {
	tbl := NewTable()
	tbl.SetPriority("x", 0) // Rev 1
	tbl.SetPriority("y", 0) // Rev 1
	tbl.SetPriority("y", 0) // Rev 2 -> y is more recently changed
	snap := tbl.Snapshot()
	// Equal priority, y has higher Rev -> y first.
	if snap[0].TraceID != "y" {
		t.Fatalf("tie-break order = [%s %s], want y first (higher Rev)", snap[0].TraceID, snap[1].TraceID)
	}
}

// --- Reset --------------------------------------------------------------------

func TestResetClearsToDefault(t *testing.T) {
	tbl := NewTable()
	tbl.Transition("s", Stopped, "x")
	tbl.Reset("s")
	st := tbl.Get("s")
	if st.Run != Running || st.Rev != 0 {
		t.Fatalf("after reset = %+v, want fresh default (Running, Rev 0)", st)
	}
	if tbl.Len() != 0 {
		t.Fatalf("after reset Len=%d, want 0", tbl.Len())
	}
}

// --- concurrency: race-clean under -race -------------------------------------

func TestConcurrentDecideAndControlRaceClean(t *testing.T) {
	tbl := NewTable()
	const traces = 16
	var wg sync.WaitGroup
	for i := 0; i < traces; i++ {
		trace := fmt.Sprintf("t%d", i)
		tbl.SetBudget(trace, Budget{TurnsLeft: 50, TokensLeft: Unbounded})
		wg.Add(3)
		// Driver: the turn loop hammering Decide.
		go func() {
			defer wg.Done()
			for j := 0; j < 100; j++ {
				tbl.Decide(trace)
			}
		}()
		// Operator: live control writes.
		go func() {
			defer wg.Done()
			for j := 0; j < 100; j++ {
				tbl.SetPriority(trace, j)
				tbl.SetPace(trace, Pace{MaxTokensPerTurn: j})
			}
		}()
		// Scheduler: reading snapshots.
		go func() {
			defer wg.Done()
			for j := 0; j < 100; j++ {
				_ = tbl.Snapshot()
				_ = tbl.Get(trace)
			}
		}()
	}
	wg.Wait()
	if tbl.Len() != traces {
		t.Fatalf("after concurrent run Len=%d, want %d", tbl.Len(), traces)
	}
}
