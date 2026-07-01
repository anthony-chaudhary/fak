package session

import (
	"container/list"
	"sort"
	"sync"
	"time"
)

// DefaultTableLimit bounds the process-local per-session drive records. Gateways
// mint a non-empty TraceID per served session, so a long-running process must not
// retain every historical session forever — the same rationale as ifc's
// DefaultLedgerLimit, and the same value, so the two per-session tables age in
// lockstep.
const DefaultTableLimit = 8192

// ---------------------------------------------------------------------------
// Table — the per-session DRIVE state, keyed by TraceID. The structural twin of
// ifc.Ledger: a mark/lru/index triple kept in lockstep under one RWMutex. Reads
// (Get, Snapshot, Decide's read half) take RLock; writes (the control verbs and
// Decide's debit) take Lock. The empty key is the single-session default.
// ---------------------------------------------------------------------------

// Table holds the live drive state of every recent served session, keyed by
// TraceID, bounded LRU. It is the gateway-owned object the session routes read and
// write, and the data structure a scheduler reads via Snapshot. Construct with
// NewTable / NewTableWithLimit; the zero value is not usable.
type Table struct {
	mu    sync.RWMutex
	state map[string]State
	cap   int
	lru   *list.List
	index map[string]*list.Element

	// obs + warnFrac are the optional budget observer seam (#743): when wired via
	// WatchBudget, DebitUsage calls obs once the context budget crosses warnFrac
	// (consumed share) and again on exhaustion. Both default to the no-op (nil obs).
	obs      BudgetObserver
	warnFrac float64
	transObs TransitionObserver

	// relayObs + relaySoftMark are the Phase-0 relay shadow seam (#1866): when wired,
	// DebitUsage emits one advisory RELAY_ARMED would-fire signal when the resident-
	// context meter first crosses the configured soft mark. relayArmed remembers which
	// live traces already emitted it, so shadow mode is noisy exactly once per trace/leg.
	relayObs      RelayShadowObserver
	relaySoftMark float64
	relayArmed    map[string]bool

	// revObs is the optional every-revision sink (#630): when wired via
	// WatchRevisions, putLocked invokes it on every Rev bump, in Rev order, under
	// the lock — the source of the gateway's /v1/fak/session/changes drive-state
	// stream. nil (the default) is the byte-identical no-op.
	revObs RevisionObserver

	// spliceFn + resumeWaiters are the live-resume loop seam (#916): WaitResume parks a
	// one-shot channel per trace in resumeWaiters while a session is Paused, and the
	// transition write path closes those channels when the session leaves Paused (a resume
	// or a stop). spliceFn is the host-wired warm-KV reattach seam consulted on the
	// Paused->Running edge; nil (the default) makes every resume cold — the byte-identical
	// pre-resume path. Both are nil/empty on a fresh table.
	spliceFn      WarmKVSplicer
	resumeWaiters map[string][]chan struct{}
}

// NewTable returns a Table bounded by DefaultTableLimit sessions.
func NewTable() *Table { return NewTableWithLimit(DefaultTableLimit) }

// NewTableWithLimit builds a table with a bounded session record table. limit<=0
// uses DefaultTableLimit. The most recently touched sessions are retained.
func NewTableWithLimit(limit int) *Table {
	if limit <= 0 {
		limit = DefaultTableLimit
	}
	return &Table{
		state: map[string]State{},
		cap:   limit,
		lru:   list.New(),
		index: map[string]*list.Element{},
	}
}

func (t *Table) ensureLocked() {
	if t.cap <= 0 {
		t.cap = DefaultTableLimit
	}
	if t.state == nil {
		t.state = map[string]State{}
	}
	if t.lru == nil {
		t.lru = list.New()
	}
	if t.index == nil {
		t.index = map[string]*list.Element{}
	}
}

func (t *Table) touchLocked(trace string) {
	if el := t.index[trace]; el != nil {
		t.lru.MoveToFront(el)
		return
	}
	t.index[trace] = t.lru.PushFront(trace)
}

func (t *Table) trimLocked() {
	for len(t.state) > t.cap {
		el := t.lru.Back()
		if el == nil {
			return
		}
		trace := el.Value.(string)
		t.lru.Remove(el)
		delete(t.index, trace)
		delete(t.state, trace)
	}
}

// getLocked returns the current record for trace, or its default if unseen. It does
// NOT touch recency (the caller decides whether a read counts as a touch) and does
// NOT take the lock (the caller holds it).
func (t *Table) getLocked(trace string) State {
	if st, ok := t.state[trace]; ok {
		return st
	}
	return DefaultState(trace)
}

// putLocked writes a record and bumps its Rev, touching recency and trimming. It is
// the single mutation point every write verb funnels through, so Rev monotonicity
// and LRU bookkeeping are maintained in exactly one place.
func (t *Table) putLocked(st State) State {
	t.ensureLocked()
	prev := t.state[st.TraceID] // zero value if unseen; Rev 0 -> new Rev 1
	st.Rev = prev.Rev + 1
	t.state[st.TraceID] = st
	t.touchLocked(st.TraceID)
	t.trimLocked()
	// Stream this revision (#630). Fired under the lock, in Rev order, so a cursor
	// feed sees every drive change exactly once and monotonically; the sink is a
	// cheap in-process ring append that never re-enters the table (see
	// RevisionObserver). nil — the default — is a zero-overhead skip.
	if t.revObs != nil {
		t.revObs(st)
	}
	return st
}

// Get returns trace's current drive record (its default if unseen). It is a pure
// read — it does NOT touch recency, so polling a session's state from an operator
// UI never keeps a finished session resident or evicts an active one.
func (t *Table) Get(trace string) State {
	t.mu.RLock()
	defer t.mu.RUnlock()
	return t.getLocked(trace)
}

// Snapshot returns a copy of every retained session's drive record, sorted into the
// order a scheduler consumes: by Priority ascending (lower yields first), ties
// broken by Rev descending (the more recently changed session first), then TraceID
// for total determinism. This is the SCHEDULER's read — the table is its data
// structure; the scheduler reads this snapshot and picks who yields. The returned
// slice is a fresh copy, safe to sort/mutate by the caller. A read-only operation:
// it does not touch recency.
func (t *Table) Snapshot() []State {
	t.mu.RLock()
	out := make([]State, 0, len(t.state))
	for _, st := range t.state {
		out = append(out, st)
	}
	t.mu.RUnlock()
	sort.Slice(out, func(i, j int) bool {
		if out[i].Priority != out[j].Priority {
			return out[i].Priority < out[j].Priority
		}
		if out[i].Rev != out[j].Rev {
			return out[i].Rev > out[j].Rev
		}
		return out[i].TraceID < out[j].TraceID
	})
	return out
}

// Len reports the number of retained session records.
func (t *Table) Len() int {
	t.mu.RLock()
	defer t.mu.RUnlock()
	return len(t.state)
}

// Limit reports the configured maximum retained session records.
func (t *Table) Limit() int {
	t.mu.RLock()
	defer t.mu.RUnlock()
	if t.cap <= 0 {
		return DefaultTableLimit
	}
	return t.cap
}

// Reset clears a session's record (a fresh session / test isolation). The next
// touch reads the default — Running, unbounded budget. Mirrors ifc.Ledger.Reset.
func (t *Table) Reset(trace string) {
	t.mu.Lock()
	t.ensureLocked()
	delete(t.state, trace)
	if el := t.index[trace]; el != nil {
		t.lru.Remove(el)
		delete(t.index, trace)
	}
	t.mu.Unlock()
}

// ---------------------------------------------------------------------------
// Write verbs — the control surface. Each takes the lock, reads the current record,
// applies its one mutation, and writes through putLocked (which bumps Rev). Each
// returns the NEW state so a caller (the route handler) echoes back exactly what it
// set, Rev included.
// ---------------------------------------------------------------------------

// Transition requests a run-state change. The legal moves enforce the small state
// machine: a terminal (Stopped) session rejects every change (ok=false) — you start
// a new session, you do not un-stop one. Setting Throttled/Stopped records the
// reason; clearing back to Running clears it. Every other live->live move is
// allowed (the operator is trusted; the kernel only forbids resurrecting a
// terminal session).
func (t *Table) Transition(trace string, to RunState, reason string) (State, bool) {
	t.mu.Lock()
	cur := t.getLocked(trace)
	if cur.Run.terminal() {
		t.mu.Unlock()
		return cur, false // a stopped session is terminal; refuse to revive it
	}
	from := cur.Run
	cur.Run = to
	switch to {
	case Throttled, Paused, Draining, Stopped:
		cur.Reason = reason
	case Running:
		cur.Reason = ""
	}
	out := t.putLocked(cur)
	// A session LEAVING Paused (a resume back to live, or a paused->stop) wakes every
	// WaitResume parked on it (#916). Done under the lock so a waiter registered concurrently
	// is either signalled here or observes the new non-Paused state on its next read — never
	// orphaned. close() is cheap and non-blocking; the woken loop re-reads the live state.
	if from == Paused && to != Paused {
		t.signalResumeLocked(trace)
	}
	obs := t.transObs
	fire := obs != nil && notableTransition(from, to)
	t.mu.Unlock()
	if fire {
		obs(transitionEvent(out, from, to))
	}
	return out, true
}

// SetBudget re-sets a session's remaining allotment live — raise to extend/speed
// up, cut to slow down or to let an urgent session pass. A terminal session rejects
// the change. Pass Unbounded on an axis to clear its cap.
func (t *Table) SetBudget(trace string, b Budget) (State, bool) {
	return t.setLocked(trace, func(cur *State) { cur.Budget = b.withContextCap() })
}

// setLocked is the shared body of the live-setter verbs (SetBudget / SetPace /
// SetPriority / SetTurnIntent / SetGoal): under the table lock it rejects a terminal
// session (returning it with ok=false) and otherwise applies mutate to the current
// record and writes it back, bumping Rev. mutate receives a pointer to the working
// copy and must only adjust fields.
func (t *Table) setLocked(trace string, mutate func(cur *State)) (State, bool) {
	t.mu.Lock()
	defer t.mu.Unlock()
	cur := t.getLocked(trace)
	if cur.Run.terminal() {
		return cur, false
	}
	mutate(&cur)
	return t.putLocked(cur), true
}

// SetPace re-sets a session's per-turn throttle live. A terminal session rejects
// the change. A zero axis means "no opinion" (planner default).
func (t *Table) SetPace(trace string, p Pace) (State, bool) {
	return t.setLocked(trace, func(cur *State) { cur.Pace = p })
}

// SetPriority re-sets a session's scheduling rank live. A terminal session rejects
// the change. Lower yields first under contention; the table only records it — a
// scheduler reading Snapshot acts on it.
func (t *Table) SetPriority(trace string, priority int) (State, bool) {
	return t.setLocked(trace, func(cur *State) { cur.Priority = priority })
}

// SetTurnIntent records the ADVISORY next-turn hint set for a session (issue #807).
// A terminal session rejects the change. The table only RECORDS it — a scheduler
// reading Snapshot decides whether to act on it, and MUST degrade to the GPU-visible
// decision when the intent is zero or stale. The hint never gates correctness; it is a
// cost/latency lever (vCache posture). Setting it bumps Rev like any other write, so a
// /v1/fak/changes cursor sees the intent update.
func (t *Table) SetTurnIntent(trace string, intent TurnIntent) (State, bool) {
	return t.setLocked(trace, func(cur *State) { cur.Intent = intent })
}

// SetGoal records the session's active goal root (issue #849, the reachability-layer
// epic #844). A terminal session rejects the change. The table only RECORDS it — a
// scheduler reading Snapshot decides whether to rank by it, and behaves identically
// when the goal is zero. The goal never gates correctness; it is a retention/ranking
// root only. Setting it bumps Rev like any other write, so a /v1/fak/changes cursor
// sees the goal update and a concurrent reader observes a monotonic version.
func (t *Table) SetGoal(trace string, goal Goal) (State, bool) {
	return t.setLocked(trace, func(cur *State) { cur.Goal = goal })
}

// SetTimeBudget re-sets a session's wall-clock envelope live (issue #1584) — raise it
// to grant more real time, cut it to bound a runaway managed run. A terminal session
// rejects the change, matching SetBudget. Unlike SetBudget, this does NOT arm the
// clock: pass a TimeBudget built with WithLimit (Start/Running left false) to configure
// the envelope without starting it, or one already Started/Resumed to configure AND
// arm it in one write. StartTimeBudget is the common case (configure once, arm now).
func (t *Table) SetTimeBudget(trace string, b TimeBudget) (State, bool) {
	return t.setLocked(trace, func(cur *State) { cur.Time = b })
}

// StartTimeBudget configures trace's wall-clock envelope to limit and arms the clock at
// now in one write — the usual entry point for a managed run that wants "govern me to
// at most N wall-clock minutes starting now". limit<=0 configures an unbounded budget
// (TimeUnbounded) that is still started (Running() true), so Elapsed(now) still reports
// real elapsed time even with no cap — useful for observability-only wall-clock
// tracking. A terminal session rejects the change.
func (t *Table) StartTimeBudget(trace string, limit time.Duration, now time.Time) (State, bool) {
	return t.setLocked(trace, func(cur *State) {
		cur.Time = NewTimeBudget().WithLimit(limit).Start(now)
	})
}

// PauseTimeBudget folds trace's live wall-clock duration into its TimeBudget's
// ElapsedNanos and clears the running clock, as of now — the write a hidden restart (or
// a clean shutdown) makes BEFORE the process goes away, so the durable State (persisted
// via Descriptor/Registry exactly like Budget already is) carries the true elapsed
// total forward instead of losing the live run's duration. A terminal session still
// accepts this write (unlike most control verbs): a Stopped/Draining session's elapsed
// time must still be foldable so its final accounting is correct, mirroring how
// Restore may re-establish a terminal record faithfully. Pausing an already-paused (or
// time-unconfigured) session is a safe no-op.
func (t *Table) PauseTimeBudget(trace string, now time.Time) State {
	t.mu.Lock()
	defer t.mu.Unlock()
	cur := t.getLocked(trace)
	cur.Time = cur.Time.Pause(now)
	return t.putLocked(cur)
}

// ResumeTimeBudget re-arms trace's wall-clock clock at now after a hidden restart,
// preserving the ElapsedNanos a prior PauseTimeBudget (or a persisted Descriptor
// restore) carried forward — the read-side counterpart of PauseTimeBudget. A terminal
// session rejects the change (a stopped session's clock should not resume ticking).
func (t *Table) ResumeTimeBudget(trace string, now time.Time) (State, bool) {
	return t.setLocked(trace, func(cur *State) { cur.Time = cur.Time.Resume(now) })
}

// QueryTimeBudget answers "how much wall-clock budget does trace have left as of now"
// without mutating anything (Table.Decide's per-turn read is the mutating half; this is
// the pure query a `fak session status`/supervisor check calls as often as it likes).
// An unseen trace reports the default (unbounded, not running) TimeBudget's verdict.
func (t *Table) QueryTimeBudget(trace string, now time.Time) TimeQueryVerdict {
	t.mu.RLock()
	defer t.mu.RUnlock()
	return t.getLocked(trace).Time.Query(now)
}

// DecideTimeBudget is the wall-clock twin of Decide's budget check: given trace's
// current TimeBudget as of now, it reports whether the run should STOP (the envelope is
// exceeded), and if so drives the session to Draining/Stopped exactly like a token-axis
// exhaustion (ReasonTimeBudgetExhausted), folding the elapsed live duration into
// ElapsedNanos so the stopped record's accounting is final and correct. A session that
// is Paused/Draining/Stopped, or has no bounded time envelope, is left untouched and
// reports Proceed=true (Decide's own run-state gate already covers those cases; this
// verb only adds the wall-clock axis on top of a live session). Call this once per turn
// boundary alongside Decide — it is deliberately separate so a caller not using wall-
// clock budgets pays zero cost and sees zero behavior change.
func (t *Table) DecideTimeBudget(trace string, now time.Time) Verdict {
	t.mu.Lock()
	defer t.mu.Unlock()
	cur := t.getLocked(trace)
	if cur.Run.terminal() {
		return Verdict{Proceed: false, Stop: true, Reason: cur.stopReasonOr(ReasonStopped), State: cur}
	}
	if cur.Run == Paused {
		return Verdict{Proceed: false, Stop: false, Reason: ReasonPaused, State: cur}
	}
	if !cur.Time.Bounded() || !cur.Time.Exceeded(now) {
		return Verdict{Proceed: true, State: cur}
	}
	cur.Time = cur.Time.Pause(now) // fold the final live duration before stopping
	cur.Run = Draining
	cur.Reason = ReasonTimeBudgetExhausted
	final := t.finalizeDrainLocked(cur)
	return Verdict{Proceed: false, Stop: true, Reason: final.Reason, State: final}
}

// CompareAndSet applies want only if the session's current Rev equals expectRev —
// the optimistic-concurrency guard a stale operator UI is checked against, so a
// newer transition is never silently clobbered. want's TraceID and Rev are ignored
// (the key is trace; the new Rev is assigned by putLocked). ok=false means the Rev
// did not match (the caller re-reads and retries) OR the session is terminal.
func (t *Table) CompareAndSet(trace string, expectRev uint64, want State) (State, bool) {
	t.mu.Lock()
	cur := t.getLocked(trace)
	if cur.Run.terminal() || cur.Rev != expectRev {
		t.mu.Unlock()
		return cur, false
	}
	from := cur.Run
	want.TraceID = trace
	out := t.putLocked(want)
	// A CAS flip OFF Paused (the operator --if-rev resume path) wakes parked WaitResume
	// waiters, the same as a direct Transition off Paused (#916).
	if from == Paused && out.Run != Paused {
		t.signalResumeLocked(trace)
	}
	// A CAS-driven run-state flip (the operator --if-rev path) fires the transition observer
	// too — without this, a pause/stop applied with --if-rev would notify nothing. Staged
	// under the lock, fired after release (the same discipline as Transition).
	obs := t.transObs
	fire := obs != nil && notableTransition(from, out.Run)
	t.mu.Unlock()
	if fire {
		obs(transitionEvent(out, from, out.Run))
	}
	return out, true
}

// Recontinue re-arms a budget-drained session under a FRESH trace (its continuation
// id), carrying a clean budget — the "human-like reset" write. It is the one verb
// that may follow a terminal budget exhaustion: the parent session stays Stopped
// (its closed Reason preserved for audit), and a NEW live session is minted under
// child, linked back via ParentTrace with Generation incremented from the parent.
// This is deliberately NOT SetBudget (which refuses a terminal session — you do not
// un-stop one) and NOT Reset (which deletes the record): the parent's drained record
// is left intact so the budget-exhaustion event stays observable, and the fresh
// session is a new key, not a resurrection of the old one.
//
// The returned State is the fresh child (Running, fresh budget, ParentTrace=parent,
// Generation=parent.Generation+1, Reason=ReasonBudgetReset, Rev 1). The parent trace
// is left exactly as the budget drain left it, EXCEPT its TimeBudget's live clock is
// paused (folded into ElapsedNanos) — see RecontinueAt, which this delegates to with
// now defaulted to time.Now at the call boundary only (never inside a decision path).
// A nil receiver mints a detached default child (no table to record into) so a loop
// with no table behaves sanely.
func (t *Table) Recontinue(parent, child string, fresh Budget) State {
	return t.RecontinueAt(parent, child, fresh, time.Now())
}

// RecontinueWithTransaction is Recontinue with a caller-supplied reset audit row.
// Hosts that build a sessionreset.Seed pass the transaction sessionreset derived
// from that seed so the child carries seed digest / contributor / omitted-span
// evidence along with the table-owned lineage and budget re-arm fields.
func (t *Table) RecontinueWithTransaction(parent, child string, fresh Budget, tx ResetTransaction) State {
	return t.RecontinueAtWithTransaction(parent, child, fresh, time.Now(), tx)
}

// RecontinueAt is Recontinue with an explicit now, for deterministic testing of the
// wall-clock carry-forward (issue #1584): the fresh child's TimeBudget PRESERVES the
// parent lineage's total ElapsedNanos and LimitNanos (a hidden context reset must not
// zero the wall-clock envelope any more than it zeros Generation), pausing the parent's
// clock at now (folding its live duration in) and re-arming the SAME accumulated total
// on the child, started fresh at now. A parent with no time budget configured
// (Bounded()==false and never started) carries forward a zero TimeBudget, so a caller
// not using wall-clock budgets sees no behavior change.
func (t *Table) RecontinueAt(parent, child string, fresh Budget, now time.Time) State {
	return t.RecontinueAtWithTransaction(parent, child, fresh, now, ResetTransaction{})
}

// RecontinueAtWithTransaction is RecontinueAt with an explicit reset transaction.
// The table normalizes old/new trace and budget re-arm from its actual write, so a
// stale caller cannot attach a row that disagrees with the child it minted.
func (t *Table) RecontinueAtWithTransaction(parent, child string, fresh Budget, now time.Time, tx ResetTransaction) State {
	tx = normalizeResetTransaction(tx, parent, child, fresh)
	if t == nil {
		st := DefaultState(child)
		st.Budget = fresh.withContextCap()
		st.ParentTrace = parent
		st.Reason = ReasonBudgetReset
		st.CacheAffinity = cacheAffinityForContinuation(State{TraceID: parent}, child, ReasonBudgetReset)
		st.ResetTransaction = tx
		return st
	}
	t.mu.Lock()
	defer t.mu.Unlock()
	t.ensureLocked()
	parentSt := t.getLocked(parent)
	prevGen := parentSt.Generation
	pausedParentTime := parentSt.Time.Pause(now)
	if _, known := t.state[parent]; known {
		// Fold the parent's final live duration into its own persisted record so its
		// closed accounting is correct, WITHOUT bumping Rev — this is an internal
		// wall-clock accounting write, not an operator-visible transition (the parent's
		// Run/Reason/Budget from the earlier drain are left exactly as they were).
		parentSt.Time = pausedParentTime
		t.state[parent] = parentSt
	}
	carriedTime := pausedParentTime
	if carriedTime.Bounded() || carriedTime.ElapsedNanos > 0 {
		carriedTime = carriedTime.Start(now)
	}
	next := State{
		TraceID:          child,
		Run:              Running,
		Budget:           fresh.withContextCap(),
		ParentTrace:      parent,
		Generation:       prevGen + 1,
		Reason:           ReasonBudgetReset,
		Time:             carriedTime,
		CacheAffinity:    cacheAffinityForContinuation(parentSt, child, parentSt.Reason),
		ResetTransaction: tx,
	}
	return t.putLocked(next)
}
