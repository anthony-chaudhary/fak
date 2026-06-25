package session

// scheduler.go — the first CONSUMER of Table.Snapshot (#627). The table HOLDS the
// drive state and exposes Snapshot already sorted into scheduler-consumption order
// (Priority ascending, ties by Rev descending, then TraceID); it deliberately never
// PICKS a winner. This file is the separate type that does: a Scheduler READS the
// snapshot each boundary and names the one live session that should run next, and
// CONSUMES budget-exhaustion / pause / stop as "a slot just freed" scheduling events
// (delivered through the table's existing WatchBudget / WatchTransitions observer
// seams) instead of re-deriving them from a process scan.
//
// POSTURE. The Scheduler is policy; the Table is a value. The scheduler only READS
// Snapshot and SUBSCRIBES to events — it adds nothing to the table and holds no lock
// the table knows about, so the table stays a passive, policy-free data structure.
// Because Pick reads the LIVE snapshot every call, an operator who cuts one session's
// budget and raises its Priority (SetBudget + SetPriority) flips which session Pick
// returns on the very next boundary — the priority-queue "let an urgent one pass"
// move falls straight out of reading fresh state, no scheduler bookkeeping required.
//
// Foundation-only: stdlib (sync), off the request path, deterministic (no clock, no
// randomness) so every policy is unit-testable to an exact pick sequence.

import "sync"

// Policy selects how Pick breaks contention among the live, eligible sessions that
// share one gateway. The zero value is StrictPriority — the safe, trivially-correct
// default that simply honors the snapshot's existing sort.
type Policy uint8

const (
	// StrictPriority returns the FIRST eligible session in Snapshot order. Snapshot is
	// already sorted (Priority ascending, then Rev descending, then TraceID), so this
	// is deterministic and correct by construction: the lowest Priority value that is
	// eligible wins, and a budget cut / priority raise re-sorts the snapshot the next
	// time Pick reads it.
	StrictPriority Policy = iota
	// WeightedFair returns a deterministic weighted round-robin winner, giving a lower
	// Priority value a proportionally larger share of the picks while still letting
	// every eligible session make progress. See pickWeightedFairLocked for the exact
	// algorithm (smooth weighted round-robin — no clock, no randomness).
	WeightedFair
)

// String renders a Policy as its lowercase token; an out-of-range value renders
// "unknown" rather than panicking, matching the rest of the package's enums.
func (p Policy) String() string {
	switch p {
	case StrictPriority:
		return "strict-priority"
	case WeightedFair:
		return "weighted-fair"
	}
	return "unknown"
}

// SlotCause is the closed reason a scheduling slot freed — the "why" on a SlotEvent.
// It is a small total enum so a host reacts on a checkable token, never free text.
type SlotCause uint8

const (
	// CauseBudgetExhausted: a session drained a configured budget axis (observed via the
	// table's BudgetExhausted event on the WatchBudget seam).
	CauseBudgetExhausted SlotCause = iota
	// CausePaused: an operator paused the session (a hold, not an end) — its slot is
	// free while it waits.
	CausePaused
	// CauseDraining: a stop was requested; the session takes it at the next boundary.
	CauseDraining
	// CauseStopped: the session reached its terminal state.
	CauseStopped
)

// String renders a SlotCause as its lowercase token; an out-of-range value renders
// "unknown" rather than panicking.
func (c SlotCause) String() string {
	switch c {
	case CauseBudgetExhausted:
		return "budget-exhausted"
	case CausePaused:
		return "paused"
	case CauseDraining:
		return "draining"
	case CauseStopped:
		return "stopped"
	}
	return "unknown"
}

// SlotEvent is the immutable "a slot freed" signal the Scheduler emits to its host
// when a session leaves the eligible set (budget exhaustion, pause, drain, or stop).
// It is the scheduling-EVENT framing the package design names: the supervisor learns
// a slot opened the instant it happens, rather than re-deriving liveness from a
// process scan. Rev is the table revision at the freeing write, so a host can order
// or de-duplicate events against a /v1/fak/changes cursor.
type SlotEvent struct {
	TraceID string    `json:"trace_id"`
	Cause   SlotCause `json:"cause"`
	Rev     uint64    `json:"rev"`
}

// AttachOptions carries the optional pass-through observers and warn fraction Attach
// installs alongside the scheduler's own handlers. WatchBudget / WatchTransitions each
// hold exactly ONE observer, so the scheduler composes: it installs a fan-out handler
// that first calls the host's pass-through (if any) and then interprets the event for
// its own slot-freed accounting. A zero AttachOptions means the scheduler takes sole
// ownership of both seams (no pass-through, warning disabled).
type AttachOptions struct {
	// WarnFraction is forwarded verbatim to Table.WatchBudget, so a host that wants the
	// #743 pre-exhaustion warning still receives it through its pass-through observer.
	// The scheduler itself ignores BudgetWarn — a warning does not free a slot; only
	// BudgetExhausted does. A value outside (0,1) disables the warning (the table's
	// documented behavior), leaving only the exhaustion event firing.
	WarnFraction float64
	// Budget, if non-nil, is the host's pass-through budget observer (e.g. an operator
	// webhook). It is invoked for EVERY BudgetEvent before the scheduler interprets the
	// event. nil means the scheduler owns the budget seam alone.
	Budget BudgetObserver
	// Transitions, if non-nil, is the host's pass-through transition observer, invoked
	// for every TransitionEvent before the scheduler maps it to a slot-freed cause.
	Transitions TransitionObserver
}

// Scheduler reads a *Table via Snapshot and picks the live session that should run
// next under a chosen Policy, and consumes budget-exhaustion / pause / stop as
// slot-freed scheduling events. It is the policy layer that keeps the table policy-
// free. The zero value is not usable — construct with NewScheduler. A nil *Scheduler
// is a sane no-op (Pick returns no winner; OnSlotFreed / Attach do nothing), so a
// host with no scheduler wired behaves like the pre-scheduler path.
type Scheduler struct {
	mu     sync.Mutex
	policy Policy
	table  *Table
	// credits is the WeightedFair virtual-time state: a per-TraceID running credit,
	// keyed by trace, rebuilt each Pick to hold only the currently-eligible set (so it
	// is bounded and a session that leaves contention forgets its stale credit).
	credits map[string]int64
	onSlot  func(SlotEvent)
}

// NewScheduler builds an unattached scheduler under the given Policy. Bind it to a
// table with Attach before calling Pick; an unattached scheduler's Pick returns no
// winner.
func NewScheduler(policy Policy) *Scheduler {
	return &Scheduler{policy: policy, credits: map[string]int64{}}
}

// Attach binds the scheduler to a table and installs the internal slot-freed handlers
// on the table's existing WatchBudget / WatchTransitions seams, composing with any
// pass-through observers in opts. After Attach, Pick reads this table's live Snapshot
// and slot-freed events flow to the OnSlotFreed callback. Calling Attach again re-binds
// (the last Attach wins — it overwrites the table's single observer slots, by design).
// A nil receiver or nil table is a no-op.
//
// The composed handlers are safe against deadlock: the table fires observers AFTER it
// releases its own lock, and the scheduler's handler only takes the scheduler's lock
// (a distinct lock the table never holds) before invoking the host callback.
func (s *Scheduler) Attach(t *Table, opts AttachOptions) {
	if s == nil || t == nil {
		return
	}
	s.mu.Lock()
	s.table = t
	s.mu.Unlock()

	passBudget := opts.Budget
	t.WatchBudget(opts.WarnFraction, func(ev BudgetEvent) {
		if passBudget != nil {
			passBudget(ev)
		}
		// A warning is early progress, not a freed slot; only exhaustion frees one.
		if ev.Kind == BudgetExhausted {
			s.emitSlot(SlotEvent{TraceID: ev.TraceID, Cause: CauseBudgetExhausted, Rev: ev.Rev})
		}
	})

	passTrans := opts.Transitions
	t.WatchTransitions(func(ev TransitionEvent) {
		if passTrans != nil {
			passTrans(ev)
		}
		// WatchTransitions only fires on notable moves (into Paused/Draining/Stopped), so
		// Running/Throttled never reach here; slotCauseFor guards the mapping defensively.
		if cause, ok := slotCauseFor(ev.To); ok {
			s.emitSlot(SlotEvent{TraceID: ev.TraceID, Cause: cause, Rev: ev.Rev})
		}
	})
}

// OnSlotFreed registers the host callback invoked once per slot-freed event. Passing
// nil clears it. Safe to call any time, including before Attach; a nil receiver is a
// no-op. The callback runs on the table's observer goroutine (after the table lock is
// released), so it may block on slow work without stalling the table — the host owns
// fan-out and failure policy, exactly like the table's own observer seams.
func (s *Scheduler) OnSlotFreed(fn func(SlotEvent)) {
	if s == nil {
		return
	}
	s.mu.Lock()
	s.onSlot = fn
	s.mu.Unlock()
}

// emitSlot delivers one SlotEvent to the host callback (if any). It copies the callback
// reference under the lock and invokes it AFTER releasing the lock, so a slow host
// callback never holds the scheduler's mutex (and a re-entrant Pick from the callback
// cannot self-deadlock).
func (s *Scheduler) emitSlot(ev SlotEvent) {
	s.mu.Lock()
	cb := s.onSlot
	s.mu.Unlock()
	if cb != nil {
		cb(ev)
	}
}

// Pick names the live session that should run next, reading the attached table's LIVE
// Snapshot so an operator's budget/priority change is reflected on the very next call.
// ok is false when there is no eligible session — an empty table, an all-ineligible
// table (everyone paused/draining/stopped or budget-exhausted), or an unattached / nil
// scheduler. On ok==false the returned State is the zero value; a caller checks ok and
// idles the gateway rather than running a phantom session.
func (s *Scheduler) Pick() (State, bool) {
	if s == nil {
		return State{}, false
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.table == nil {
		return State{}, false
	}
	// Snapshot takes the table's own RLock (a distinct lock); it never calls back into
	// the scheduler, so holding s.mu across it cannot deadlock.
	snap := s.table.Snapshot()
	switch s.policy {
	case WeightedFair:
		return s.pickWeightedFairLocked(snap)
	default:
		return pickStrictPriority(snap)
	}
}

// pickStrictPriority returns the first eligible session in snapshot order. Snapshot is
// already sorted into consumption order, so "first eligible" IS the lowest-Priority
// eligible session, broken by Rev (most recently changed) then TraceID — deterministic
// and correct without any per-scheduler state.
func pickStrictPriority(snap []State) (State, bool) {
	for _, st := range snap {
		if Eligible(st) {
			return st, true
		}
	}
	return State{}, false
}

// pickWeightedFairLocked implements a deterministic SMOOTH weighted round-robin (the
// nginx algorithm) over the currently-eligible sessions. Caller holds s.mu.
//
// ALGORITHM. Each eligible session i is assigned an integer weight derived from its
// Priority relative to the current contention set: weight_i = (maxPriority - Priority_i)
// + 1, where maxPriority is the largest Priority value among the eligible sessions. So
// the highest Priority value (lowest share) gets weight 1, every step lower adds 1, and
// every weight is >= 1 — a lower Priority value yields a proportionally larger share.
// Let W = sum of the weights. On each Pick:
//
//  1. add every session's weight to its running credit c_i;
//  2. the winner is the session with the greatest credit (ties broken toward the
//     earlier-in-snapshot, i.e. higher-share, session — strict-greater keeps the first);
//  3. the winner repays the round by subtracting W from its credit.
//
// Over any window of W consecutive Picks (with a stable eligible set) each session i is
// chosen EXACTLY weight_i times, smoothly interleaved, and the credits return to their
// starting point — so the distribution is exact and the sequence is fully periodic,
// deterministic, and unit-testable to an exact pick order. No clock, no randomness.
//
// The credit map is rebuilt each call to hold only the currently-eligible traces, so it
// stays bounded and a session that leaves contention (paused, exhausted, evicted) drops
// its stale credit and rejoins fair; an empty contention set clears the state entirely.
func (s *Scheduler) pickWeightedFairLocked(snap []State) (State, bool) {
	// Eligible sessions in snapshot order — already Priority-ascending, so a larger-share
	// (lower Priority value) session sorts earlier and therefore wins credit ties.
	elig := make([]State, 0, len(snap))
	for _, st := range snap {
		if Eligible(st) {
			elig = append(elig, st)
		}
	}
	if len(elig) == 0 {
		s.credits = map[string]int64{} // nothing in contention: forget stale virtual time
		return State{}, false
	}

	maxPriority := elig[0].Priority
	for _, st := range elig {
		if st.Priority > maxPriority {
			maxPriority = st.Priority
		}
	}

	next := make(map[string]int64, len(elig))
	var total int64
	bestIdx := -1
	var bestCredit int64
	for i, st := range elig {
		w := int64(maxPriority-st.Priority) + 1
		total += w
		c := s.credits[st.TraceID] + w // carry prior credit (0 if newly eligible), then add weight
		next[st.TraceID] = c
		if bestIdx == -1 || c > bestCredit {
			bestIdx, bestCredit = i, c
		}
	}
	winner := elig[bestIdx]
	next[winner.TraceID] -= total
	s.credits = next // prune to the current contention set
	return winner, true
}

// Eligible reports whether a session may be PICKED to run next: its run-state advances
// (Running or Throttled — Paused/Draining/Stopped are held or ending, never eligible)
// AND it has not exhausted a configured budget axis. It is the helper both policies
// share, exported so a host can apply the same eligibility test (e.g. to render which
// sessions are in contention) without re-deriving the rule.
func Eligible(st State) bool {
	switch st.Run {
	case Running, Throttled:
		// advancing — fall through to the budget check
	default:
		return false
	}
	return !budgetExhausted(st.Budget)
}

// budgetExhausted reports whether any CONFIGURED budget axis has hit zero. An Unbounded
// (negative) turns/tokens axis never exhausts; a context axis is "configured" only when
// its cap is set (ContextTokensCap > 0 — the table stamps the cap from the remaining at
// set-time, and context 0-with-no-cap means "not configured", never exhausted). A
// configured axis at or below zero is exhausted.
func budgetExhausted(b Budget) bool {
	if !b.turnsUnbounded() && b.TurnsLeft <= 0 {
		return true
	}
	if !b.tokensUnbounded() && b.TokensLeft <= 0 {
		return true
	}
	if b.ContextTokensCap > 0 && b.ContextTokensLeft <= 0 {
		return true
	}
	return false
}

// slotCauseFor maps a notable run-state landing to its slot-freed cause. The bool is
// false for any non-notable state (Running/Throttled), so the transition handler never
// emits a slot event for a session that is merely advancing.
func slotCauseFor(to RunState) (SlotCause, bool) {
	switch to {
	case Paused:
		return CausePaused, true
	case Draining:
		return CauseDraining, true
	case Stopped:
		return CauseStopped, true
	}
	return 0, false
}
