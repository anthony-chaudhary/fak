package session

import (
	"encoding/json"
	"testing"
	"time"
)

// timebudget_test.go — pure-core tests for the wall-clock budget axis (issue #1584,
// epic #1570 "managed context"). Mirrors descriptor_test.go's fixedClock discipline: an
// explicit, hand-advanced now, never time.Now(), so accounting-across-a-simulated-
// restart is deterministic. Run with `go test ./internal/session -run TimeBudget -v`.

func tbClock() time.Time {
	return time.Date(2026, 6, 30, 9, 0, 0, 0, time.UTC)
}

// TestTimeBudgetElapsedSurvivesRestart proves the core persistence contract: a
// TimeBudget's accumulated ElapsedNanos is NOT an in-memory counter that resets to
// zero on a hidden restart. It is started, run for a while, paused (the shutdown
// write), round-tripped through JSON (simulating a process restart reading the value
// back from disk), and resumed — the post-restart Elapsed must equal pre-restart
// elapsed plus the new run's duration, never restarting from zero.
func TestTimeBudgetElapsedSurvivesRestart(t *testing.T) {
	t0 := tbClock()
	b := NewTimeBudget().WithLimit(time.Hour).Start(t0)

	// Run for 10 minutes, then "shut down": Pause folds the live duration in.
	t1 := t0.Add(10 * time.Minute)
	b = b.Pause(t1)
	if b.Running() {
		t.Fatalf("Pause left the clock running")
	}
	if got := b.Elapsed(t1); got != 10*time.Minute {
		t.Fatalf("elapsed after pause = %v, want 10m", got)
	}

	// Simulate an actual process restart: marshal to JSON and back, exactly what a
	// Descriptor round-trip through a file-backed store would do.
	raw, err := json.Marshal(b)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	var restored TimeBudget
	if err := json.Unmarshal(raw, &restored); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if restored.ElapsedNanos != b.ElapsedNanos || restored.LimitNanos != b.LimitNanos {
		t.Fatalf("JSON round-trip lost accounting: got %+v want %+v", restored, b)
	}
	if restored.Running() {
		t.Fatalf("restored budget must not be live-ticking (StartedAtUnixNano should be 0)")
	}

	// New process picks up at t2 (some time later — the "downtime" is NOT charged,
	// matching restoredPaused's documented conservative rule) and resumes the clock.
	t2 := t1.Add(5 * time.Minute)
	resumed := restored.Resume(t2)
	if !resumed.Running() {
		t.Fatalf("Resume did not re-arm the clock")
	}
	// Elapsed immediately after resume (before further time passes) is still just the
	// carried-forward total.
	if got := resumed.Elapsed(t2); got != 10*time.Minute {
		t.Fatalf("elapsed immediately after resume = %v, want 10m (carried forward)", got)
	}

	// Run 5 more minutes post-restart: total elapsed must be 15m, not 5m.
	t3 := t2.Add(5 * time.Minute)
	if got := resumed.Elapsed(t3); got != 15*time.Minute {
		t.Fatalf("elapsed after restart+5m = %v, want 15m (10m carried + 5m new)", got)
	}
	remaining, ok := resumed.Remaining(t3)
	if !ok {
		t.Fatalf("Remaining reported unbounded for a 1h-limit budget")
	}
	if remaining != 45*time.Minute {
		t.Fatalf("remaining = %v, want 45m (1h limit - 15m elapsed)", remaining)
	}
}

// TestTimeBudgetRestoredPausedDropsLiveInstantButKeepsElapsed proves the
// restoredPaused load-time contract directly: a descriptor snapshotted MID-TICK
// (StartedAtUnixNano set, as if the process died before an explicit Pause) restores
// with the clock stopped and the elapsed-so-far total intact — it neither charges the
// unknowable downtime against the budget nor keeps ticking from a dead process's
// timestamp.
func TestTimeBudgetRestoredPausedDropsLiveInstantButKeepsElapsed(t *testing.T) {
	t0 := tbClock()
	b := NewTimeBudget().WithLimit(time.Hour).Start(t0)
	midTick := b // StartedAtUnixNano still set: this is what got persisted before a crash

	restored := midTick.restoredPaused()
	if restored.Running() {
		t.Fatalf("restoredPaused left the clock running")
	}
	if restored.ElapsedNanos != 0 {
		t.Fatalf("restoredPaused should not fabricate elapsed time for a never-paused budget, got %d", restored.ElapsedNanos)
	}
	// Whatever now the new process picks, Elapsed reports only the carried total (0
	// here) until the caller explicitly Resumes.
	if got := restored.Elapsed(t0.Add(48 * time.Hour)); got != 0 {
		t.Fatalf("restoredPaused budget must not accrue time while not running, got %v", got)
	}
}

// TestTimeBudgetUnboundedNeverExceeded proves the permissive default: a TimeBudget
// with no configured limit (the zero value, or WithLimit(0)/WithLimit(TimeUnbounded))
// never reports Exceeded, and Remaining signals "not configured" via ok=false rather
// than a numeric zero (which would be indistinguishable from "no time left").
func TestTimeBudgetUnboundedNeverExceeded(t *testing.T) {
	t0 := tbClock()
	b := NewTimeBudget().Start(t0)
	future := t0.Add(365 * 24 * time.Hour)
	if b.Bounded() {
		t.Fatalf("zero-value TimeBudget reports Bounded=true")
	}
	if b.Exceeded(future) {
		t.Fatalf("unbounded budget reported Exceeded after a year")
	}
	if _, ok := b.Remaining(future); ok {
		t.Fatalf("Remaining on an unbounded budget should report ok=false")
	}
	v := b.Query(future)
	if v.Bounded || v.Exceeded {
		t.Fatalf("unbounded Query verdict = %+v, want Bounded=false Exceeded=false", v)
	}
}

// TestTimeBudgetExceededAtLimit proves the boundary condition: elapsed == limit counts
// as exceeded (not just strictly greater), matching Exceeded's >= comparison.
func TestTimeBudgetExceededAtLimit(t *testing.T) {
	t0 := tbClock()
	b := NewTimeBudget().WithLimit(time.Minute).Start(t0)
	atLimit := t0.Add(time.Minute)
	if !b.Exceeded(atLimit) {
		t.Fatalf("elapsed==limit must count as exceeded")
	}
	beforeLimit := t0.Add(time.Minute - time.Nanosecond)
	if b.Exceeded(beforeLimit) {
		t.Fatalf("one nanosecond before the limit must not be exceeded")
	}
}

// TestQueryTimeBudgetDoesNotMutate proves Table.QueryTimeBudget is a pure read: calling
// it any number of times never changes the session's stored TimeBudget or Rev, unlike
// DecideTimeBudget which may transition the session.
func TestQueryTimeBudgetDoesNotMutate(t *testing.T) {
	tbl := NewTable()
	t0 := tbClock()
	tbl.StartTimeBudget("s", 30*time.Minute, t0)
	before := tbl.Get("s")

	for i := 0; i < 5; i++ {
		v := tbl.QueryTimeBudget("s", t0.Add(time.Duration(i)*time.Minute))
		if !v.Bounded {
			t.Fatalf("query %d: expected bounded verdict", i)
		}
	}
	after := tbl.Get("s")
	if after.Rev != before.Rev {
		t.Fatalf("QueryTimeBudget mutated Rev: before=%d after=%d", before.Rev, after.Rev)
	}
	if after.Time != before.Time {
		t.Fatalf("QueryTimeBudget mutated the stored TimeBudget: before=%+v after=%+v", before.Time, after.Time)
	}

	// An unseen trace reports the permissive unbounded default rather than erroring.
	v := tbl.QueryTimeBudget("never-seen", t0)
	if v.Bounded {
		t.Fatalf("unseen trace should report an unbounded default verdict, got %+v", v)
	}
}

// TestDecideTimeBudgetStillFineProceeds proves the "still fine" branch: a bounded,
// not-yet-exceeded time budget leaves a running session untouched (Proceed=true, no
// state transition), so a caller not near its wall-clock limit pays no cost beyond the
// read.
func TestDecideTimeBudgetStillFineProceeds(t *testing.T) {
	tbl := NewTable()
	t0 := tbClock()
	tbl.StartTimeBudget("s", time.Hour, t0)

	v := tbl.DecideTimeBudget("s", t0.Add(10*time.Minute))
	if !v.Proceed || v.Stop {
		t.Fatalf("still-fine time budget verdict = %+v, want Proceed=true Stop=false", v)
	}
	if v.State.Run != Running {
		t.Fatalf("still-fine session must remain Running, got %v", v.State.Run)
	}
}

// TestDecideTimeBudgetStopsAtExhaustion proves the "stop" branch: a bounded time
// budget that has elapsed drives the session to a terminal state with the closed
// ReasonTimeBudgetExhausted token, and folds the final live duration into
// ElapsedNanos so the stopped record's accounting is exact — exactly like a
// token-budget exhaustion transitions through Draining to Stopped.
func TestDecideTimeBudgetStopsAtExhaustion(t *testing.T) {
	tbl := NewTable()
	t0 := tbClock()
	tbl.StartTimeBudget("s", 15*time.Minute, t0)

	exceededAt := t0.Add(20 * time.Minute)
	v := tbl.DecideTimeBudget("s", exceededAt)
	if v.Proceed || !v.Stop {
		t.Fatalf("exhausted time budget verdict = %+v, want Proceed=false Stop=true", v)
	}
	if v.Reason != ReasonTimeBudgetExhausted {
		t.Fatalf("stop reason = %q, want %q", v.Reason, ReasonTimeBudgetExhausted)
	}
	if v.State.Run != Stopped {
		t.Fatalf("time-exhausted session should reach Stopped, got %v", v.State.Run)
	}
	if v.State.Time.Running() {
		t.Fatalf("stopped session's clock must be paused, not left ticking")
	}
	if got := v.State.Time.ElapsedNanos; got < int64(20*time.Minute) {
		t.Fatalf("stopped session's elapsed total = %v, want >= 20m", time.Duration(got))
	}

	// A second Decide call after stopping is stable: terminal guard takes over, no
	// further mutation of the already-final Time accounting.
	final := tbl.Get("s")
	v2 := tbl.DecideTimeBudget("s", exceededAt.Add(time.Hour))
	if v2.State.Time.ElapsedNanos != final.Time.ElapsedNanos {
		t.Fatalf("re-deciding a stopped session must not keep accruing elapsed time: %d != %d", v2.State.Time.ElapsedNanos, final.Time.ElapsedNanos)
	}
}

// TestDecideTimeBudgetIgnoresPausedSession proves DecideTimeBudget defers to the
// existing Paused-run-state gate (mirroring Decide's own behavior) rather than
// draining a session that is merely operator-paused, even if its wall-clock envelope
// has technically elapsed.
func TestDecideTimeBudgetIgnoresPausedSession(t *testing.T) {
	tbl := NewTable()
	t0 := tbClock()
	tbl.StartTimeBudget("s", time.Minute, t0)
	tbl.Transition("s", Paused, "operator hold")

	v := tbl.DecideTimeBudget("s", t0.Add(time.Hour))
	if v.Proceed {
		t.Fatalf("paused session should not proceed")
	}
	if v.Stop {
		t.Fatalf("paused session should not be force-stopped by the time axis: %+v", v)
	}
	if v.Reason != ReasonPaused {
		t.Fatalf("paused verdict reason = %q, want %q", v.Reason, ReasonPaused)
	}
}

// TestTimeAndTokenBudgetsAreIndependentAxes is the direct test of the issue's "reports
// remaining time distinct from remaining tokens" requirement: a session can be
// simultaneously fine on tokens but exhausted on time, and the reverse, proving the
// two axes are tracked and decided independently.
func TestTimeBudgetAndTokenBudgetAreIndependentAxes(t *testing.T) {
	t0 := tbClock()

	// Case 1: plenty of tokens left, but the wall-clock envelope has elapsed.
	tbl := NewTable()
	tbl.SetBudget("s", Budget{TurnsLeft: Unbounded, TokensLeft: 1_000_000})
	tbl.StartTimeBudget("s", time.Minute, t0)

	tokenVerdict := tbl.Decide("s")
	if !tokenVerdict.Proceed {
		t.Fatalf("token axis should still be fine: %+v", tokenVerdict)
	}
	timeQuery := tbl.QueryTimeBudget("s", t0.Add(2*time.Minute))
	if !timeQuery.Exceeded {
		t.Fatalf("time axis should report exceeded independent of token state: %+v", timeQuery)
	}
	stillHasTokens := tbl.Get("s").Budget.TokensLeft
	if stillHasTokens != 1_000_000 {
		t.Fatalf("querying the time axis must not touch the token axis: tokens=%d", stillHasTokens)
	}

	// Case 2: the reverse — tokens exhausted, but the time envelope is nowhere near its
	// limit.
	tbl2 := NewTable()
	tbl2.SetBudget("s2", Budget{TurnsLeft: 1, TokensLeft: Unbounded})
	tbl2.StartTimeBudget("s2", time.Hour, t0)
	tbl2.Decide("s2")                // consumes the one turn
	turnVerdict := tbl2.Decide("s2") // now exhausted on turns
	if turnVerdict.Proceed || turnVerdict.Reason != ReasonBudgetTurns {
		t.Fatalf("token/turn axis should be exhausted: %+v", turnVerdict)
	}
	timeQuery2 := tbl2.QueryTimeBudget("s2", t0.Add(time.Minute))
	if timeQuery2.Exceeded {
		t.Fatalf("time axis should still be fine despite the turn axis draining: %+v", timeQuery2)
	}
	if timeQuery2.Remaining <= 0 {
		t.Fatalf("time axis should report positive remaining time: %+v", timeQuery2)
	}
}

// TestRecontinueAtCarriesTimeBudgetForward is the direct test of the issue's central
// "persists across resets" requirement applied to the existing hidden-restart
// mechanism: RecontinueAt must fold the parent's live wall-clock duration into its
// carried ElapsedNanos and re-arm the SAME accumulated total (plus the same limit) on
// the fresh child, rather than starting the child's clock at zero.
func TestRecontinueAtCarriesTimeBudgetForward(t *testing.T) {
	tbl := NewTable()
	t0 := tbClock()
	tbl.SetBudget("s", Budget{TurnsLeft: Unbounded, TokensLeft: Unbounded, ContextTokensLeft: 10})
	tbl.StartTimeBudget("s", time.Hour, t0)

	// Burn through the context budget to drain the parent (the existing reset trigger),
	// then let 12 minutes of wall-clock time pass before the reset actually happens.
	tbl.DebitUsage("s", Usage{ContextTokens: 11})
	tbl.Decide("s") // take the drain to terminal (Stopped)
	resetAt := t0.Add(12 * time.Minute)

	child := tbl.RecontinueAt("s", "s-child", Budget{TurnsLeft: Unbounded, TokensLeft: Unbounded, ContextTokensLeft: 10}, resetAt)

	if !child.Time.Bounded() {
		t.Fatalf("child lost the wall-clock limit across the reset: %+v", child.Time)
	}
	if got := child.Time.LimitNanos; got != int64(time.Hour) {
		t.Fatalf("child time limit = %v, want 1h carried forward", time.Duration(got))
	}
	if !child.Time.Running() {
		t.Fatalf("child's clock must be started fresh at the reset instant")
	}
	// Elapsed immediately at resetAt must equal the 12m the parent accrued — the reset
	// must not zero the wall-clock accounting the way a naive in-memory counter would.
	if got := child.Time.Elapsed(resetAt); got != 12*time.Minute {
		t.Fatalf("child elapsed at reset = %v, want 12m carried from parent", got)
	}

	// The parent's own persisted record is folded (paused) in place too, so its final
	// accounting is correct, without reviving it (still Stopped).
	parent := tbl.Get("s")
	if parent.Run != Stopped {
		t.Fatalf("parent must remain stopped after recontinue, got %v", parent.Run)
	}
	if parent.Time.Running() {
		t.Fatalf("parent's clock must be paused (folded), not left ticking, after recontinue")
	}
	if got := parent.Time.ElapsedNanos; got != int64(12*time.Minute) {
		t.Fatalf("parent elapsed after fold = %v, want 12m", time.Duration(got))
	}

	// Continue running the child a further 10 minutes: total lineage elapsed is 22m,
	// well short of the 1h limit, so it should still proceed on the time axis.
	laterQuery := tbl.QueryTimeBudget("s-child", resetAt.Add(10*time.Minute))
	if laterQuery.Exceeded {
		t.Fatalf("child should not be exceeded yet: %+v", laterQuery)
	}
	if got := laterQuery.Elapsed; got != 22*time.Minute {
		t.Fatalf("child elapsed after +10m = %v, want 22m (12m carried + 10m new)", got)
	}
}

// TestRecontinueAtNeverSeenParentDoesNotPhantomInsert proves the LRU-safety fix made
// during review: recontinuing off a parent trace the Table never actually registered
// (getLocked returns a synthesized default, not a stored record) must NOT insert a
// phantom entry into the table's internal state map that bypasses ordinary
// touch/trim bookkeeping. The table's Len() must only count the fresh child.
func TestRecontinueAtTimeBudgetNeverSeenParentDoesNotPhantomInsert(t *testing.T) {
	tbl := NewTable()
	t0 := tbClock()

	child := tbl.RecontinueAt("never-registered-parent", "child", Budget{TurnsLeft: 3}, t0)
	if child.TraceID != "child" || child.ParentTrace != "never-registered-parent" {
		t.Fatalf("child lineage wrong: %+v", child)
	}
	if got := tbl.Len(); got != 1 {
		t.Fatalf("Len() = %d, want 1 (only the child; the unseen parent must not phantom-insert)", got)
	}
}

// TestRecontinueAtUnconfiguredTimeBudgetIsNoBehaviorChange proves the additive-only
// guarantee: a parent with no wall-clock envelope configured at all (the ordinary case
// for every session that predates #1584, and any caller not opting into wall-clock
// budgets) carries forward a zero TimeBudget, so Recontinue's behavior for a caller not
// using this feature is unchanged.
func TestRecontinueAtUnconfiguredTimeBudgetIsNoBehaviorChange(t *testing.T) {
	tbl := NewTable()
	t0 := tbClock()
	tbl.SetBudget("s", Budget{TurnsLeft: 1, TokensLeft: Unbounded})
	tbl.Decide("s")
	tbl.Transition("s", Stopped, ReasonBudgetTurns)

	child := tbl.RecontinueAt("s", "child", Budget{TurnsLeft: 5}, t0)
	if child.Time.Bounded() || child.Time.Running() {
		t.Fatalf("child inherited a time budget from a parent that never configured one: %+v", child.Time)
	}
}

// TestPauseResumeTimeBudgetTableVerbs exercises the Table-level Pause/Resume pair
// directly (the verbs a real shutdown/restart sequence calls), confirming they compose
// exactly like the pure TimeBudget methods they wrap, and that PauseTimeBudget works
// even on a terminal (Stopped) session (its final accounting must still be foldable).
func TestPauseResumeTimeBudgetTableVerbs(t *testing.T) {
	tbl := NewTable()
	t0 := tbClock()
	tbl.StartTimeBudget("s", time.Hour, t0)

	pausedAt := t0.Add(7 * time.Minute)
	paused := tbl.PauseTimeBudget("s", pausedAt)
	if paused.Time.Running() {
		t.Fatalf("PauseTimeBudget left the clock running")
	}
	if got := paused.Time.ElapsedNanos; got != int64(7*time.Minute) {
		t.Fatalf("paused elapsed = %v, want 7m", time.Duration(got))
	}

	resumedAt := pausedAt.Add(3 * time.Minute) // "downtime" not charged
	resumed, ok := tbl.ResumeTimeBudget("s", resumedAt)
	if !ok {
		t.Fatalf("ResumeTimeBudget refused a live session")
	}
	if !resumed.Time.Running() {
		t.Fatalf("ResumeTimeBudget did not re-arm the clock")
	}
	if got := resumed.Time.Elapsed(resumedAt); got != 7*time.Minute {
		t.Fatalf("elapsed right after resume = %v, want 7m carried forward", got)
	}

	// Now stop the session and confirm PauseTimeBudget still works on a terminal
	// record (its final accounting must still be closeable).
	tbl.Transition("s", Stopped, "manual stop")
	finalPauseAt := resumedAt.Add(5 * time.Minute)
	final := tbl.PauseTimeBudget("s", finalPauseAt)
	if got := final.Time.ElapsedNanos; got != int64(12*time.Minute) {
		t.Fatalf("terminal session's folded elapsed = %v, want 12m (7m + 5m)", time.Duration(got))
	}

	// But ResumeTimeBudget on a terminal session must be refused (a stopped session's
	// clock should not resume ticking).
	if _, ok := tbl.ResumeTimeBudget("s", finalPauseAt.Add(time.Minute)); ok {
		t.Fatalf("ResumeTimeBudget must refuse a terminal session")
	}
}

// TestDescriptorRoundTripPreservesTimeBudget proves the full durable path end to end:
// descriptorFromState -> JSON (MemStore) -> RestoredState reattaches a session's
// wall-clock accounting exactly like Budget/Priority/Generation already round-trip,
// restoring the clock PAUSED even though it was persisted mid-tick.
func TestDescriptorRoundTripPreservesTimeBudget(t *testing.T) {
	store := NewMemStore()
	r := NewRegistry(store)
	t0 := fixedClock()

	live := NewTable()
	live.Restore("trace-t", State{TraceID: "trace-t", Run: Running})
	live.StartTimeBudget("trace-t", 20*time.Minute, t0)
	midTick := live.Get("trace-t")

	if _, err := r.Register("sess-t", "host-a", midTick, time.Hour, t0); err != nil {
		t.Fatalf("Register: %v", err)
	}

	// Simulate a restart: fresh registry view over the same store, fresh table.
	r2 := NewRegistry(store)
	d, ok, err := r2.Get("sess-t")
	if err != nil || !ok {
		t.Fatalf("Get after restart: ok=%v err=%v", ok, err)
	}
	if d.Time.LimitNanos != int64(20*time.Minute) {
		t.Fatalf("descriptor lost the wall-clock limit: %+v", d.Time)
	}

	restored := d.RestoredState()
	if restored.Time.Running() {
		t.Fatalf("restored session's clock must load paused, not ticking from a dead process's timestamp")
	}
	if !restored.Time.Bounded() {
		t.Fatalf("restored session lost its wall-clock envelope")
	}

	restartedTable := NewTable()
	restartedTable.Restore("trace-t", restored)
	// Before an explicit ResumeTimeBudget, elapsed stays at whatever was persisted
	// (here 0, since it was never explicitly paused before Register captured it) —
	// the caller must resume it explicitly to keep tracking real time.
	q := restartedTable.QueryTimeBudget("trace-t", t0.Add(time.Hour))
	if q.Exceeded {
		t.Fatalf("a freshly-restored, not-yet-resumed budget must not report exceeded from stale accounting: %+v", q)
	}
}
