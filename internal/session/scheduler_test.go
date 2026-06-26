package session

import (
	"sync"
	"testing"
	"time"
)

// slotRecorder collects SlotEvents thread-safely for assertions (the scheduler's
// callback may run on the table's observer goroutine).
type slotRecorder struct {
	mu     sync.Mutex
	events []SlotEvent
}

func (r *slotRecorder) observe(ev SlotEvent) {
	r.mu.Lock()
	r.events = append(r.events, ev)
	r.mu.Unlock()
}

func (r *slotRecorder) snapshot() []SlotEvent {
	r.mu.Lock()
	defer r.mu.Unlock()
	out := make([]SlotEvent, len(r.events))
	copy(out, r.events)
	return out
}

// attachedScheduler builds a table + a scheduler bound to it under the given policy,
// with no pass-through observers and the warning disabled (the scheduler only acts on
// exhaustion). Returns both so a test can drive the table and Pick the scheduler.
func attachedScheduler(t *testing.T, policy Policy) (*Table, *Scheduler) {
	t.Helper()
	tbl := NewTable()
	s := NewScheduler(policy)
	s.Attach(tbl, AttachOptions{})
	return tbl, s
}

// TestStrictPriorityPicksLowestEligible proves the trivially-correct default: with all
// sessions eligible, Pick returns the lowest Priority value (the head of the snapshot).
func TestStrictPriorityPicksLowestEligible(t *testing.T) {
	tbl, s := attachedScheduler(t, StrictPriority)
	tbl.SetPriority("s-b", 1)
	tbl.SetPriority("s-c", 2)
	tbl.SetPriority("s-a", 0) // lowest Priority value -> should win

	got, ok := s.Pick()
	if !ok || got.TraceID != "s-a" {
		t.Fatalf("Pick = (%q, %v), want (s-a, true)", got.TraceID, ok)
	}
	// Pick is read-only and reads the live snapshot, so it is idempotent here.
	if got2, ok2 := s.Pick(); !ok2 || got2.TraceID != "s-a" {
		t.Fatalf("second Pick = (%q, %v), want (s-a, true) — Pick must not mutate ordering", got2.TraceID, ok2)
	}
}

// TestStrictPrioritySkipsIneligible proves Pick skips paused / draining / stopped and
// budget-exhausted sessions even when they sort ahead, returning the first ELIGIBLE one.
func TestStrictPrioritySkipsIneligible(t *testing.T) {
	tbl, s := attachedScheduler(t, StrictPriority)

	tbl.SetPriority("paused", 0)
	tbl.Transition("paused", Paused, "") // ineligible by run-state, though it sorts first

	tbl.SetPriority("exhausted", 1)
	// Configured turns axis hit zero -> budget-exhausted -> ineligible, run-state still Running.
	tbl.SetBudget("exhausted", Budget{TurnsLeft: 0, TokensLeft: Unbounded})

	tbl.SetPriority("draining", 2)
	tbl.Transition("draining", Draining, "")

	tbl.SetPriority("stopped", 3)
	tbl.Transition("stopped", Stopped, "")

	tbl.SetPriority("winner", 4) // the only eligible session, sorts last

	got, ok := s.Pick()
	if !ok || got.TraceID != "winner" {
		t.Fatalf("Pick = (%q, %v), want (winner, true) — must skip all ineligible", got.TraceID, ok)
	}
}

// TestPriorityCutFlipsWinner proves the priority-queue "let an urgent one pass" move:
// A wins; an operator raises A's Priority value (lowers its rank), and B wins the NEXT
// Pick — purely because Pick re-reads the live snapshot, with no scheduler bookkeeping.
func TestPriorityCutFlipsWinner(t *testing.T) {
	tbl, s := attachedScheduler(t, StrictPriority)
	tbl.SetPriority("a", 0)
	tbl.SetPriority("b", 1)

	if got, ok := s.Pick(); !ok || got.TraceID != "a" {
		t.Fatalf("initial Pick = (%q, %v), want (a, true)", got.TraceID, ok)
	}

	// Operator demotes A (and could equally cut its budget) to let urgent B pass.
	tbl.SetPriority("a", 5)

	if got, ok := s.Pick(); !ok || got.TraceID != "b" {
		t.Fatalf("after demoting a, Pick = (%q, %v), want (b, true)", got.TraceID, ok)
	}
}

// TestWeightedFairTwoSessions proves the smooth weighted round-robin over a 3:1 weight
// split: priorities 0 and 2 map to weights (2-0)+1=3 and (2-2)+1=1, so a window of W=4
// picks is the exact smooth sequence a,a,b,a (counts a=3, b=1), and it repeats.
func TestWeightedFairTwoSessions(t *testing.T) {
	tbl, s := attachedScheduler(t, WeightedFair)
	tbl.SetPriority("a", 0) // weight (2-0)+1 = 3
	tbl.SetPriority("b", 2) // weight (2-2)+1 = 1

	// One window of W=4 picks: exact smooth sequence a,a,b,a (counts a=3, b=1).
	want := []string{"a", "a", "b", "a"}
	got := pickSeq(t, s, len(want))
	assertSeq(t, got, want)

	// A second window repeats identically — the credits returned to their start.
	got2 := pickSeq(t, s, len(want))
	assertSeq(t, got2, want)
}

// TestWeightedFairThreeSessions proves an exact 3:2:1 distribution: priorities 0,1,2 map
// to weights 3,2,1, and one window of W=6 picks yields the smooth sequence a,b,a,c,b,a.
func TestWeightedFairThreeSessions(t *testing.T) {
	tbl, s := attachedScheduler(t, WeightedFair)
	tbl.SetPriority("a", 0) // weight 3
	tbl.SetPriority("b", 1) // weight 2
	tbl.SetPriority("c", 2) // weight 1

	want := []string{"a", "b", "a", "c", "b", "a"}
	got := pickSeq(t, s, len(want))
	assertSeq(t, got, want)

	// Counts over the window are exactly proportional to the weights.
	counts := map[string]int{}
	for _, id := range got {
		counts[id]++
	}
	if counts["a"] != 3 || counts["b"] != 2 || counts["c"] != 1 {
		t.Fatalf("window counts = %v, want a:3 b:2 c:1", counts)
	}
}

// TestWeightedFairSkipsIneligible proves WeightedFair, like StrictPriority, only ever
// distributes picks among ELIGIBLE sessions: an exhausted/paused session is never picked.
func TestWeightedFairSkipsIneligible(t *testing.T) {
	tbl, s := attachedScheduler(t, WeightedFair)
	tbl.SetPriority("a", 0)
	tbl.SetPriority("b", 1)
	tbl.Transition("b", Paused, "") // remove b from contention

	for i, id := range pickSeq(t, s, 5) {
		if id != "a" {
			t.Fatalf("pick %d = %q, want a (b is paused, only a is eligible)", i, id)
		}
	}
}

// TestPickEmptyAndAllIneligible documents the (State, ok) signature: an empty table and
// an all-ineligible table both return ok=false with the zero State, under both policies.
func TestPickEmptyAndAllIneligible(t *testing.T) {
	for _, policy := range []Policy{StrictPriority, WeightedFair} {
		tbl, s := attachedScheduler(t, policy)

		if got, ok := s.Pick(); ok {
			t.Fatalf("[%s] empty Pick = (%+v, true), want ok=false", policy, got)
		}

		tbl.SetPriority("p", 0)
		tbl.Transition("p", Paused, "")
		tbl.SetPriority("x", 1)
		tbl.Transition("x", Stopped, "")

		if got, ok := s.Pick(); ok {
			t.Fatalf("[%s] all-ineligible Pick = (%+v, true), want ok=false", policy, got)
		}
	}
}

// TestSlotFreedOnBudgetExhaustion proves a drained context budget surfaces as one
// CauseBudgetExhausted slot event (observed through the WatchBudget seam, not re-derived).
func TestSlotFreedOnBudgetExhaustion(t *testing.T) {
	tbl, s := attachedScheduler(t, StrictPriority)
	rec := &slotRecorder{}
	s.OnSlotFreed(rec.observe)

	tbl.SetBudget("drain-me", Budget{TurnsLeft: Unbounded, TokensLeft: Unbounded, ContextTokensLeft: 100})
	st := tbl.DebitUsage("drain-me", Usage{ContextTokens: 100}) // drain it

	got := rec.snapshot()
	if len(got) != 1 || got[0].Cause != CauseBudgetExhausted || got[0].TraceID != "drain-me" {
		t.Fatalf("slot events = %+v, want one CauseBudgetExhausted for drain-me", got)
	}
	if got[0].Rev != st.Rev {
		t.Fatalf("slot event rev = %d, want the freeing write's rev %d", got[0].Rev, st.Rev)
	}
}

// TestSlotFreedOnTransitions proves pause / drain / stop each surface as one slot event
// with the right Cause, and that Throttled / Running flips are silent (no slot freed).
func TestSlotFreedOnTransitions(t *testing.T) {
	tbl, s := attachedScheduler(t, StrictPriority)
	rec := &slotRecorder{}
	s.OnSlotFreed(rec.observe)

	// Advancing flips must NOT free a slot.
	tbl.Transition("noisy", Throttled, "slow")
	tbl.Transition("noisy", Running, "")
	if got := rec.snapshot(); len(got) != 0 {
		t.Fatalf("Throttled/Running flips freed a slot: %+v, want none", got)
	}

	tbl.Transition("p", Paused, "")
	tbl.Transition("d", Draining, "")
	tbl.Transition("s", Stopped, "")

	got := rec.snapshot()
	if len(got) != 3 {
		t.Fatalf("slot events = %+v, want exactly 3 (paused/draining/stopped)", got)
	}
	want := map[string]SlotCause{"p": CausePaused, "d": CauseDraining, "s": CauseStopped}
	for _, ev := range got {
		if want[ev.TraceID] != ev.Cause {
			t.Fatalf("event %+v has wrong cause, want %v for %s", ev, want[ev.TraceID], ev.TraceID)
		}
	}
}

// TestAttachComposesPassThroughObservers proves Attach FANS OUT: a host's pass-through
// budget + transition observers still fire (the scheduler shares the single seam, it
// does not steal it), and the scheduler's own slot events fire too.
func TestAttachComposesPassThroughObservers(t *testing.T) {
	tbl := NewTable()
	s := NewScheduler(StrictPriority)

	budgetSeen := &recorder{}     // reused from observe_test.go (BudgetEvent recorder)
	transSeen := &transRecorder{} // reused from observe_test.go (TransitionEvent recorder)
	slotSeen := &slotRecorder{}
	s.OnSlotFreed(slotSeen.observe)

	s.Attach(tbl, AttachOptions{
		WarnFraction: 0.8,
		Budget:       budgetSeen.observe,
		Transitions:  transSeen.observe,
	})

	// Drive a budget warn + exhaustion and a transition.
	tbl.SetBudget("c", Budget{TurnsLeft: Unbounded, TokensLeft: Unbounded, ContextTokensLeft: 100})
	tbl.DebitUsage("c", Usage{ContextTokens: 90}) // crosses 80% -> warn (pass-through only)
	tbl.DebitUsage("c", Usage{ContextTokens: 10}) // drains -> exhausted (pass-through + slot)
	tbl.Transition("c2", Paused, "")              // transition (pass-through + slot)

	if bs := budgetSeen.snapshot(); len(bs) != 2 || bs[0].Kind != BudgetWarn || bs[1].Kind != BudgetExhausted {
		t.Fatalf("pass-through budget events = %+v, want [warn, exhausted]", bs)
	}
	if ts := transSeen.snapshot(); len(ts) != 1 || ts[0].To != Paused {
		t.Fatalf("pass-through transition events = %+v, want one To=Paused", ts)
	}
	// Scheduler sees only the slot-freeing subset: the exhaustion and the pause (not the warn).
	ss := slotSeen.snapshot()
	if len(ss) != 2 {
		t.Fatalf("slot events = %+v, want 2 (budget-exhausted + paused, warn excluded)", ss)
	}
	if ss[0].Cause != CauseBudgetExhausted || ss[0].TraceID != "c" {
		t.Fatalf("first slot event = %+v, want CauseBudgetExhausted for c", ss[0])
	}
	if ss[1].Cause != CausePaused || ss[1].TraceID != "c2" {
		t.Fatalf("second slot event = %+v, want CausePaused for c2", ss[1])
	}
}

func TestReserveKnownComingPromotesExactPrefixMatch(t *testing.T) {
	tbl, s := attachedScheduler(t, StrictPriority)
	now := time.Unix(100, 0)
	const prefix = "sha256:known-prefix"

	tbl.SetTurnIntent("tool-waiting", TurnIntent{ArrivingInMillis: 50, Prefix: prefix})
	reserved := s.ReserveKnownComing(now)
	if len(reserved) != 1 {
		t.Fatalf("reservations = %+v, want one", reserved)
	}
	r := reserved[0]
	if r.TraceID != "tool-waiting" || r.Prefix != prefix {
		t.Fatalf("reservation = %+v, want trace/prefix match", r)
	}
	if r.ArrivesAtUnixNano != now.Add(50*time.Millisecond).UnixNano() {
		t.Fatalf("arrives_at = %d, want %d", r.ArrivesAtUnixNano, now.Add(50*time.Millisecond).UnixNano())
	}
	if r.ExpiresAtUnixNano != now.Add(50*time.Millisecond).Add(DefaultReservationGrace).UnixNano() {
		t.Fatalf("expires_at = %d, want arrival+grace", r.ExpiresAtUnixNano)
	}

	promo, ok := s.PromoteReservation("tool-waiting", prefix, now.Add(20*time.Millisecond))
	if !ok {
		t.Fatalf("expected exact trace+prefix match to promote")
	}
	if promo.TraceID != "tool-waiting" || promo.Prefix != prefix || promo.PromotedAtUnixNano == 0 {
		t.Fatalf("promotion = %+v, want adopted reservation with promoted timestamp", promo)
	}
	if _, ok := s.PromoteReservation("tool-waiting", prefix, now.Add(21*time.Millisecond)); ok {
		t.Fatalf("reservation promoted twice; want one adoption only")
	}
}

func TestReserveKnownComingRequiresExactPrefixAndExpires(t *testing.T) {
	tbl, s := attachedScheduler(t, StrictPriority)
	now := time.Unix(200, 0)
	const prefix = "sha256:warm"

	tbl.SetTurnIntent("s", TurnIntent{ArrivingInMillis: 10, Prefix: prefix})
	if got := s.ReserveKnownComing(now); len(got) != 1 {
		t.Fatalf("reserve = %+v, want one", got)
	}
	if _, ok := s.PromoteReservation("s", "sha256:cold", now.Add(5*time.Millisecond)); ok {
		t.Fatalf("wrong prefix promoted; want cold path")
	}
	if got := s.Reservations(now.Add(5 * time.Millisecond)); len(got) != 1 {
		t.Fatalf("wrong-prefix promotion consumed reservation: %+v", got)
	}

	expiredAt := now.Add(10 * time.Millisecond).Add(DefaultReservationGrace)
	expired := s.ExpireReservations(expiredAt)
	if len(expired) != 1 || expired[0].TraceID != "s" || expired[0].Prefix != prefix {
		t.Fatalf("expired = %+v, want the stale warm reservation", expired)
	}
	if _, ok := s.PromoteReservation("s", prefix, expiredAt); ok {
		t.Fatalf("expired reservation promoted; want reclaimed cold path")
	}
}

func TestReserveKnownComingDropsStalePrefixOnIntentUpdate(t *testing.T) {
	tbl, s := attachedScheduler(t, StrictPriority)
	now := time.Unix(250, 0)

	tbl.SetTurnIntent("s", TurnIntent{ArrivingInMillis: 50, Prefix: "sha256:old"})
	if got := s.ReserveKnownComing(now); len(got) != 1 {
		t.Fatalf("initial reserve = %+v, want one", got)
	}

	tbl.SetTurnIntent("s", TurnIntent{ArrivingInMillis: 50, Prefix: "sha256:new"})
	got := s.ReserveKnownComing(now.Add(10 * time.Millisecond))
	if len(got) != 1 || got[0].Prefix != "sha256:new" {
		t.Fatalf("updated reserve = %+v, want only new prefix", got)
	}
	if _, ok := s.PromoteReservation("s", "sha256:old", now.Add(20*time.Millisecond)); ok {
		t.Fatalf("old prefix promoted after intent update; want stale reservation reclaimed")
	}
	if _, ok := s.PromoteReservation("s", "sha256:new", now.Add(20*time.Millisecond)); !ok {
		t.Fatalf("new prefix did not promote")
	}
}

func TestReservationsAreLowerClassThanRealPicks(t *testing.T) {
	tbl, s := attachedScheduler(t, StrictPriority)
	now := time.Unix(300, 0)

	tbl.SetPriority("future", 0)
	tbl.Transition("future", Paused, "")
	tbl.SetTurnIntent("future", TurnIntent{ArrivingInMillis: 100, Prefix: "sha256:future"})
	tbl.SetPriority("real", 10)

	if got := s.ReserveKnownComing(now); len(got) != 1 {
		t.Fatalf("reserve future = %+v, want one advisory hold", got)
	}
	picked, ok := s.Pick()
	if !ok || picked.TraceID != "real" {
		t.Fatalf("Pick with advisory reservation = (%q,%v), want real request", picked.TraceID, ok)
	}
}

// TestEligibleHelper exercises the exported eligibility rule directly across run-states
// and configured/unbounded budget axes.
func TestEligibleHelper(t *testing.T) {
	unbounded := Budget{TurnsLeft: Unbounded, TokensLeft: Unbounded}
	cases := []struct {
		name string
		st   State
		want bool
	}{
		{"running-unbounded", State{Run: Running, Budget: unbounded}, true},
		{"throttled-unbounded", State{Run: Throttled, Budget: unbounded}, true},
		{"paused", State{Run: Paused, Budget: unbounded}, false},
		{"draining", State{Run: Draining, Budget: unbounded}, false},
		{"stopped", State{Run: Stopped, Budget: unbounded}, false},
		{"turns-exhausted", State{Run: Running, Budget: Budget{TurnsLeft: 0, TokensLeft: Unbounded}}, false},
		{"tokens-exhausted", State{Run: Running, Budget: Budget{TurnsLeft: Unbounded, TokensLeft: 0}}, false},
		{"context-exhausted", State{Run: Running, Budget: Budget{TurnsLeft: Unbounded, TokensLeft: Unbounded, ContextTokensLeft: 0, ContextTokensCap: 100}}, false},
		{"context-unconfigured-zero", State{Run: Running, Budget: Budget{TurnsLeft: Unbounded, TokensLeft: Unbounded, ContextTokensLeft: 0, ContextTokensCap: 0}}, true},
		{"turns-remaining", State{Run: Running, Budget: Budget{TurnsLeft: 1, TokensLeft: Unbounded}}, true},
	}
	for _, c := range cases {
		if got := Eligible(c.st); got != c.want {
			t.Fatalf("Eligible(%s) = %v, want %v", c.name, got, c.want)
		}
	}
}

// TestNilAndUnattachedScheduler proves the sane no-op posture: a nil receiver never
// panics and Picks no winner, and a constructed-but-unattached scheduler Picks none.
func TestNilAndUnattachedScheduler(t *testing.T) {
	var nilSched *Scheduler
	if got, ok := nilSched.Pick(); ok {
		t.Fatalf("nil Pick = (%+v, true), want ok=false", got)
	}
	nilSched.OnSlotFreed(func(SlotEvent) {})     // must not panic
	nilSched.Attach(NewTable(), AttachOptions{}) // must not panic

	unattached := NewScheduler(WeightedFair)
	if got, ok := unattached.Pick(); ok {
		t.Fatalf("unattached Pick = (%+v, true), want ok=false", got)
	}
}

// pickSeq calls Pick n times and returns the winning TraceIDs, failing if any Pick has
// no winner (the sequence tests expect a full eligible set).
func pickSeq(t *testing.T, s *Scheduler, n int) []string {
	t.Helper()
	out := make([]string, 0, n)
	for i := 0; i < n; i++ {
		got, ok := s.Pick()
		if !ok {
			t.Fatalf("Pick %d returned no winner, want one", i)
		}
		out = append(out, got.TraceID)
	}
	return out
}

func assertSeq(t *testing.T, got, want []string) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("sequence length = %d, want %d (%v vs %v)", len(got), len(want), got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("pick sequence = %v, want %v (first diff at %d)", got, want, i)
		}
	}
}
