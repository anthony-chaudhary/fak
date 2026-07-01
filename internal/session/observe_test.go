package session

import (
	"sync"
	"testing"
)

// recorder collects budget events thread-safely for assertions.
type recorder struct {
	mu     sync.Mutex
	events []BudgetEvent
}

func (r *recorder) observe(ev BudgetEvent) {
	r.mu.Lock()
	r.events = append(r.events, ev)
	r.mu.Unlock()
}

func (r *recorder) snapshot() []BudgetEvent {
	r.mu.Lock()
	defer r.mu.Unlock()
	out := make([]BudgetEvent, len(r.events))
	copy(out, r.events)
	return out
}

// TestWatchBudgetWarnsOnceThenExhausts proves the #743 observer: a session seeded with a
// context budget fires exactly one BudgetWarn when consumption first crosses the warning
// share, then one BudgetExhausted (carrying a continuation id) when the budget drains.
func TestWatchBudgetWarnsOnceThenExhausts(t *testing.T) {
	const trace = "watch-1"
	tbl := NewTable()
	rec := &recorder{}
	tbl.WatchBudget(0.8, rec.observe) // warn at 80% consumed (20 remaining of 100)

	tbl.SetBudget(trace, Budget{TurnsLeft: Unbounded, TokensLeft: Unbounded, ContextTokensLeft: 100})

	// 70 consumed (30 left): below the 80% watermark — no event yet.
	tbl.DebitUsage(trace, Usage{ContextTokens: 70})
	if got := rec.snapshot(); len(got) != 0 {
		t.Fatalf("after 70/100 consumed, events = %+v, want none (still under 80%%)", got)
	}

	// 85 consumed (15 left): crosses the 80% watermark — exactly one warn.
	tbl.DebitUsage(trace, Usage{ContextTokens: 15})
	got := rec.snapshot()
	if len(got) != 1 || got[0].Kind != BudgetWarn {
		t.Fatalf("after crossing 80%%, events = %+v, want one BudgetWarn", got)
	}
	if got[0].TraceID != trace || got[0].ContextTokensCap != 100 || got[0].ContextTokensLeft != 15 {
		t.Fatalf("warn payload = %+v, want trace=%s cap=100 left=15", got[0], trace)
	}
	if got[0].ResidentContextTokens != 15 || got[0].ResidentContextCap != 100 {
		t.Fatalf("warn resident meter = %+v, want resident=15 cap=100", got[0])
	}
	if got[0].ResidentContextFraction < 0.14 || got[0].ResidentContextFraction > 0.16 {
		t.Fatalf("warn resident fraction = %v, want ~0.15", got[0].ResidentContextFraction)
	}
	if got[0].FractionConsumed < 0.84 || got[0].FractionConsumed > 0.86 {
		t.Fatalf("warn fraction = %v, want ~0.85", got[0].FractionConsumed)
	}

	// A further debit still above the watermark must NOT re-warn (warn fires once).
	tbl.DebitUsage(trace, Usage{ContextTokens: 5}) // 90 consumed, 10 left
	if got := rec.snapshot(); len(got) != 1 {
		t.Fatalf("warn must fire once; events = %+v", got)
	}

	// Drain it: one BudgetExhausted carrying the reset continuation id.
	st := tbl.DebitUsage(trace, Usage{ContextTokens: 50})
	got = rec.snapshot()
	if len(got) != 2 || got[1].Kind != BudgetExhausted {
		t.Fatalf("after exhaustion, events = %+v, want a trailing BudgetExhausted", got)
	}
	if got[1].ContinuationID == "" || got[1].ContinuationID != st.ContinuationID {
		t.Fatalf("exhausted event continuation id = %q, want the session's %q", got[1].ContinuationID, st.ContinuationID)
	}
	if got[1].Reason != ReasonBudgetContext || got[1].FractionConsumed != 1 {
		t.Fatalf("exhausted payload = %+v, want reason=%s fraction=1", got[1], ReasonBudgetContext)
	}
	if got[1].ResidentContextTokens != 50 || got[1].ResidentContextCap != 100 || got[1].ResidentContextFraction < 0.49 || got[1].ResidentContextFraction > 0.51 {
		t.Fatalf("exhausted resident meter = %+v, want resident=50 cap=100 fraction~0.50", got[1])
	}
}

// TestRelayResidentContextMeterReportsPerLegFraction proves the relay-facing meter:
// the budget observer receives the current resident context divided by the configured
// leg ceiling, not just the cumulative consumed share that drives rotation thresholds.
func TestRelayResidentContextMeterReportsPerLegFraction(t *testing.T) {
	const trace = "relay-resident-meter"
	tbl := NewTable()
	rec := &recorder{}
	tbl.WatchBudget(0.5, rec.observe) // warn at 50% consumed
	tbl.SetBudget(trace, Budget{TurnsLeft: Unbounded, TokensLeft: Unbounded, ContextTokensLeft: 1000})

	tbl.DebitUsage(trace, Usage{ContextTokens: 400}) // resident 40%, consumed 40%: no event yet
	if got := rec.snapshot(); len(got) != 0 {
		t.Fatalf("before warn threshold, events = %+v, want none", got)
	}

	tbl.DebitUsage(trace, Usage{ContextTokens: 125}) // resident 12.5%, consumed 52.5%: warning event
	got := rec.snapshot()
	if len(got) != 1 || got[0].Kind != BudgetWarn {
		t.Fatalf("relay meter events = %+v, want one BudgetWarn", got)
	}
	if got[0].ResidentContextTokens != 125 || got[0].ResidentContextCap != 1000 {
		t.Fatalf("relay resident meter = %+v, want resident=125 cap=1000", got[0])
	}
	if got[0].ResidentContextFraction < 0.124 || got[0].ResidentContextFraction > 0.126 {
		t.Fatalf("relay resident fraction = %v, want ~0.125", got[0].ResidentContextFraction)
	}
	if got[0].FractionConsumed < 0.524 || got[0].FractionConsumed > 0.526 {
		t.Fatalf("relay consumed fraction = %v, want ~0.525", got[0].FractionConsumed)
	}
}

// relayRecorder collects relay shadow events thread-safely for assertions.
type relayRecorder struct {
	mu     sync.Mutex
	events []RelayShadowEvent
}

func (r *relayRecorder) observe(ev RelayShadowEvent) {
	r.mu.Lock()
	r.events = append(r.events, ev)
	r.mu.Unlock()
}

func (r *relayRecorder) snapshot() []RelayShadowEvent {
	r.mu.Lock()
	defer r.mu.Unlock()
	out := make([]RelayShadowEvent, len(r.events))
	copy(out, r.events)
	return out
}

func TestRelayShadowWouldRotateAtSoftMarkOnce(t *testing.T) {
	const trace = "relay-shadow"
	tbl := NewTable()
	rec := &relayRecorder{}
	tbl.WatchRelayShadow(0.30, rec.observe)
	tbl.SetBudget(trace, Budget{TurnsLeft: Unbounded, TokensLeft: Unbounded, ContextTokensLeft: 5000})

	st := tbl.DebitUsage(trace, Usage{ContextTokens: 1000}) // resident 20%: below soft mark
	if st.Run != Running {
		t.Fatalf("below soft mark state = %+v, want Running", st)
	}
	if got := rec.snapshot(); len(got) != 0 {
		t.Fatalf("below soft mark events = %+v, want none", got)
	}

	st = tbl.DebitUsage(trace, Usage{ContextTokens: 1600}) // resident 32%: advisory arm
	if st.Run != Running || st.Reason != "" {
		t.Fatalf("soft mark must not transition state: %+v", st)
	}
	got := rec.snapshot()
	if len(got) != 1 {
		t.Fatalf("after crossing soft mark events = %+v, want exactly one", got)
	}
	if got[0].TraceID != trace || got[0].Reason != ReasonRelayArmed || got[0].Rev != st.Rev {
		t.Fatalf("relay shadow identity = %+v, want trace=%s reason=%s rev=%d", got[0], trace, ReasonRelayArmed, st.Rev)
	}
	if got[0].ResidentContextTokens != 1600 || got[0].ResidentContextCap != 5000 {
		t.Fatalf("relay shadow resident meter = %+v, want resident=1600 cap=5000", got[0])
	}
	if got[0].ResidentContextFraction < 0.319 || got[0].ResidentContextFraction > 0.321 {
		t.Fatalf("relay shadow resident fraction = %v, want ~0.32", got[0].ResidentContextFraction)
	}
	if got[0].SoftMark != 0.30 {
		t.Fatalf("relay shadow soft mark = %v, want 0.30", got[0].SoftMark)
	}

	st = tbl.DebitUsage(trace, Usage{ContextTokens: 1500}) // above soft mark again, still no duplicate
	if st.Run != Running {
		t.Fatalf("second above-soft debit state = %+v, want Running", st)
	}
	if got := rec.snapshot(); len(got) != 1 {
		t.Fatalf("relay shadow must fire once; events = %+v", got)
	}
}

func TestWatchBudgetExhaustionCarriesCacheAffinityDecision(t *testing.T) {
	const trace = "watch-affinity"
	tbl := NewTable()
	rec := &recorder{}
	tbl.WatchBudget(0, rec.observe)
	tbl.SetBudget(trace, Budget{TurnsLeft: Unbounded, TokensLeft: Unbounded, ContextTokensLeft: 10})

	st := tbl.DebitUsage(trace, Usage{ContextTokens: 11})
	got := rec.snapshot()
	if len(got) != 1 || got[0].Kind != BudgetExhausted {
		t.Fatalf("events = %+v, want one exhaustion", got)
	}
	if got[0].CacheAffinity != st.CacheAffinity {
		t.Fatalf("event cache affinity = %+v, want state decision %+v", got[0].CacheAffinity, st.CacheAffinity)
	}
	if got[0].CacheAffinity.Action != CacheAffinityPreserve || got[0].CacheAffinity.ToTraceID != st.ContinuationID {
		t.Fatalf("event cache affinity decision = %+v, want preserve to continuation", got[0].CacheAffinity)
	}
}

// TestWatchBudgetStraightToExhaustionSkipsWarn proves a single oversized debit that jumps
// past the watermark straight to zero fires only the exhaustion event, not a warning.
func TestWatchBudgetStraightToExhaustionSkipsWarn(t *testing.T) {
	const trace = "watch-2"
	tbl := NewTable()
	rec := &recorder{}
	tbl.WatchBudget(0.8, rec.observe)
	tbl.SetBudget(trace, Budget{TurnsLeft: Unbounded, TokensLeft: Unbounded, ContextTokensLeft: 100})

	tbl.DebitUsage(trace, Usage{ContextTokens: 200}) // one turn larger than the whole budget
	got := rec.snapshot()
	if len(got) != 1 || got[0].Kind != BudgetExhausted {
		t.Fatalf("oversized debit events = %+v, want a single BudgetExhausted (warn skipped)", got)
	}
	if got[0].FractionConsumed != 1 {
		t.Fatalf("over-debit fraction = %v, want clamped 1.0", got[0].FractionConsumed)
	}
}

// TestWatchBudgetDisabledWarnStillExhausts proves a warn fraction outside (0,1) disables
// the pre-exhaustion warning but leaves the exhaustion (reset) event firing.
func TestWatchBudgetDisabledWarnStillExhausts(t *testing.T) {
	const trace = "watch-3"
	tbl := NewTable()
	rec := &recorder{}
	tbl.WatchBudget(0, rec.observe) // warning disabled
	tbl.SetBudget(trace, Budget{TurnsLeft: Unbounded, TokensLeft: Unbounded, ContextTokensLeft: 10})

	tbl.DebitUsage(trace, Usage{ContextTokens: 9}) // 90% consumed — would warn if armed
	if got := rec.snapshot(); len(got) != 0 {
		t.Fatalf("warning disabled, events = %+v, want none before exhaustion", got)
	}
	tbl.DebitUsage(trace, Usage{ContextTokens: 5}) // drain
	got := rec.snapshot()
	if len(got) != 1 || got[0].Kind != BudgetExhausted {
		t.Fatalf("events = %+v, want a single BudgetExhausted", got)
	}
}

// TestDebitUsageNoObserverUnchanged proves the no-op default: with no observer wired the
// debit path neither fires nor changes the resulting state (cap is still stamped for a
// later WatchBudget, but no callback is invoked).
func TestDebitUsageNoObserverUnchanged(t *testing.T) {
	const trace = "watch-4"
	tbl := NewTable()
	tbl.SetBudget(trace, Budget{TurnsLeft: Unbounded, TokensLeft: Unbounded, ContextTokensLeft: 100})
	st := tbl.DebitUsage(trace, Usage{ContextTokens: 90})
	if st.Budget.ContextTokensLeft != 10 || st.Budget.ContextTokensCap != 100 {
		t.Fatalf("no-observer debit state = %+v, want left=10 cap=100", st.Budget)
	}
}

// transRecorder collects transition events thread-safely for assertions.
type transRecorder struct {
	mu     sync.Mutex
	events []TransitionEvent
}

func (r *transRecorder) observe(ev TransitionEvent) {
	r.mu.Lock()
	r.events = append(r.events, ev)
	r.mu.Unlock()
}

func (r *transRecorder) snapshot() []TransitionEvent {
	r.mu.Lock()
	defer r.mu.Unlock()
	out := make([]TransitionEvent, len(r.events))
	copy(out, r.events)
	return out
}

// TestWatchTransitionsFiresOncePerNotableBoundary proves the #761 SIGCHLD-equivalent: a
// flip into Paused / Draining / Stopped fires exactly one event carrying the closed reason
// token and a monotonic Rev; a Running/Throttled flip is silent; a no-op re-set into the
// same state fires nothing.
func TestWatchTransitionsFiresOncePerNotableBoundary(t *testing.T) {
	const trace = "trans-1"
	tbl := NewTable()
	rec := &transRecorder{}
	tbl.WatchTransitions(rec.observe)

	// Throttled then Running: neither is a "waiting/stopped" signal -> silent.
	tbl.Transition(trace, Throttled, "slow")
	tbl.Transition(trace, Running, "")
	if got := rec.snapshot(); len(got) != 0 {
		t.Fatalf("Throttled/Running flips must be silent; events = %+v", got)
	}

	// Paused (needs-input): one event. Reason is reason-free at the operator API but the
	// event carries the canonical PAUSED token.
	tbl.Transition(trace, Paused, "")
	got := rec.snapshot()
	if len(got) != 1 || got[0].To != Paused || got[0].Reason != ReasonPaused {
		t.Fatalf("after pause, events = %+v, want one To=Paused reason=%s", got, ReasonPaused)
	}
	if got[0].TraceID != trace || got[0].Rev == 0 {
		t.Fatalf("pause event = %+v, want trace=%s rev>0", got[0], trace)
	}
	pausedRev := got[0].Rev

	// Draining carrying an operator reason: one more, Rev advanced.
	tbl.Transition(trace, Draining, "operator-stop")
	got = rec.snapshot()
	if len(got) != 2 || got[1].To != Draining || got[1].Reason != "operator-stop" || got[1].Rev <= pausedRev {
		t.Fatalf("after drain, events = %+v, want a trailing To=Draining reason=operator-stop rev>%d", got, pausedRev)
	}

	// No-op re-set into the same Draining: notableTransition(from==to) is false -> silent.
	tbl.Transition(trace, Draining, "operator-stop")
	if got := rec.snapshot(); len(got) != 2 {
		t.Fatalf("re-set into the same state must be silent; events = %+v", got)
	}

	// Stopped: one event carrying the stop token (Stopped is terminal afterward).
	tbl.Transition(trace, Stopped, ReasonStopped)
	got = rec.snapshot()
	if len(got) != 3 || got[2].To != Stopped || got[2].Reason != ReasonStopped {
		t.Fatalf("after stop, events = %+v, want a trailing To=Stopped reason=%s", got, ReasonStopped)
	}
}

// TestWatchTransitionsViaCompareAndSet proves the operator --if-rev (CAS) path fires the
// observer too: without the CAS fire hook a pause applied with --if-rev would notify nothing.
func TestWatchTransitionsViaCompareAndSet(t *testing.T) {
	const trace = "trans-cas"
	tbl := NewTable()
	rec := &transRecorder{}
	tbl.WatchTransitions(rec.observe)

	seeded, _ := tbl.SetPriority(trace, 5)
	want := seeded
	want.Run = Paused
	if _, ok := tbl.CompareAndSet(trace, seeded.Rev, want); !ok {
		t.Fatalf("CompareAndSet at rev %d unexpectedly failed", seeded.Rev)
	}
	got := rec.snapshot()
	if len(got) != 1 || got[0].To != Paused || got[0].Reason != ReasonPaused {
		t.Fatalf("CAS pause events = %+v, want one To=Paused reason=%s", got, ReasonPaused)
	}
}

// TestWatchTransitionsNoObserverUnchanged proves the no-op default: with no observer wired
// Transition returns the same (State, ok) and never panics.
func TestWatchTransitionsNoObserverUnchanged(t *testing.T) {
	const trace = "trans-noop"
	tbl := NewTable()
	st, ok := tbl.Transition(trace, Paused, "")
	if !ok || st.Run != Paused {
		t.Fatalf("no-observer transition = (%+v, %v), want Paused/ok", st, ok)
	}
}
