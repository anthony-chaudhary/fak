package session

import (
	"container/list"
	"sort"
	"sync"
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
	t.mu.Lock()
	defer t.mu.Unlock()
	cur := t.getLocked(trace)
	if cur.Run.terminal() {
		return cur, false
	}
	cur.Budget = b.withContextCap()
	return t.putLocked(cur), true
}

// SetPace re-sets a session's per-turn throttle live. A terminal session rejects
// the change. A zero axis means "no opinion" (planner default).
func (t *Table) SetPace(trace string, p Pace) (State, bool) {
	t.mu.Lock()
	defer t.mu.Unlock()
	cur := t.getLocked(trace)
	if cur.Run.terminal() {
		return cur, false
	}
	cur.Pace = p
	return t.putLocked(cur), true
}

// SetPriority re-sets a session's scheduling rank live. A terminal session rejects
// the change. Lower yields first under contention; the table only records it — a
// scheduler reading Snapshot acts on it.
func (t *Table) SetPriority(trace string, priority int) (State, bool) {
	t.mu.Lock()
	defer t.mu.Unlock()
	cur := t.getLocked(trace)
	if cur.Run.terminal() {
		return cur, false
	}
	cur.Priority = priority
	return t.putLocked(cur), true
}

// SetTurnIntent records the ADVISORY next-turn hint set for a session (issue #807).
// A terminal session rejects the change. The table only RECORDS it — a scheduler
// reading Snapshot decides whether to act on it, and MUST degrade to the GPU-visible
// decision when the intent is zero or stale. The hint never gates correctness; it is a
// cost/latency lever (vCache posture). Setting it bumps Rev like any other write, so a
// /v1/fak/changes cursor sees the intent update.
func (t *Table) SetTurnIntent(trace string, intent TurnIntent) (State, bool) {
	t.mu.Lock()
	defer t.mu.Unlock()
	cur := t.getLocked(trace)
	if cur.Run.terminal() {
		return cur, false
	}
	cur.Intent = intent
	return t.putLocked(cur), true
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
// is left exactly as the budget drain left it. A nil receiver mints a detached
// default child (no table to record into) so a loop with no table behaves sanely.
func (t *Table) Recontinue(parent, child string, fresh Budget) State {
	if t == nil {
		st := DefaultState(child)
		st.Budget = fresh.withContextCap()
		st.ParentTrace = parent
		st.Reason = ReasonBudgetReset
		return st
	}
	t.mu.Lock()
	defer t.mu.Unlock()
	prevGen := t.getLocked(parent).Generation
	next := State{
		TraceID:     child,
		Run:         Running,
		Budget:      fresh.withContextCap(),
		ParentTrace: parent,
		Generation:  prevGen + 1,
		Reason:      ReasonBudgetReset,
	}
	return t.putLocked(next)
}
