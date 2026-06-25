package session

import "math"

// observe.go — the budget threshold/exhaustion observer seam (#743). A served session's
// context budget is observed today only AT exhaustion (DebitUsage drains it; the next
// boundary refuses or resets). This adds, on that same DebitUsage path, two earlier
// signals an external supervisor can act on:
//
//   - a PRE-EXHAUSTION warning, fired once when the budget first crosses a configured
//     consumed share (e.g. 80%), so a supervisor can extend the budget or wind the
//     session down BEFORE it drains, and
//   - an EXHAUSTION event carrying the continuation id, so an external monitor is notified
//     the instant the reset is triggered.
//
// Both ride one optional BudgetObserver the host wires (cmd/fak forwards it to an operator
// webhook). A nil observer is the byte-identical no-op default — the debit path is
// unchanged when nothing is watching. The package stays foundation-only: it delivers a
// typed value; the host owns the I/O (the webhook POST), exactly like the gateway never
// imports the engine.

// BudgetEventKind classifies a budget-lifecycle event a BudgetObserver is told about.
type BudgetEventKind uint8

const (
	// BudgetWarn fires once, when a session's context budget first crosses the
	// pre-exhaustion warning threshold (the configured consumed share, e.g. 80%) —
	// early enough that a supervisor can extend the budget or wind the session down
	// before it drains.
	BudgetWarn BudgetEventKind = iota
	// BudgetExhausted fires when the context budget hits zero — the reset trigger.
	// ContinuationID names the fresh window the session continues under.
	BudgetExhausted
)

// String renders the event kind as its lowercase wire token ("warn"/"exhausted"); an
// out-of-range value renders "unknown" rather than panicking.
func (k BudgetEventKind) String() string {
	switch k {
	case BudgetWarn:
		return "warn"
	case BudgetExhausted:
		return "exhausted"
	}
	return "unknown"
}

// BudgetEvent is the immutable snapshot a BudgetObserver receives. It is built under the
// table lock and delivered AFTER the lock is released, so an observer may do slow work
// (a webhook POST) without stalling the debit hot path or any other session.
type BudgetEvent struct {
	Kind              BudgetEventKind `json:"kind"`
	TraceID           string          `json:"trace_id"`
	ContinuationID    string          `json:"continuation_id,omitempty"` // set on Exhausted: the fresh-window handoff id
	Reason            string          `json:"reason,omitempty"`          // the closed budget reason token at this event
	Rev               uint64          `json:"rev"`
	ContextTokensLeft int             `json:"context_tokens_left"`
	ContextTokensCap  int             `json:"context_tokens_cap,omitempty"`
	FractionConsumed  float64         `json:"fraction_consumed"` // 0..1, the share of the context budget spent at this event
}

// BudgetObserver is the threshold-and-reset callback seam. The table invokes it from
// DebitUsage AFTER releasing its lock, so the callback is free to block (a webhook POST)
// without holding up other sessions. The host owns fan-out and failure policy — cmd/fak
// fires the webhook fire-and-forget, fail-open; the table only delivers the typed event.
type BudgetObserver func(BudgetEvent)

// TransitionEvent is the immutable snapshot a TransitionObserver receives when an
// operator run-state change lands. It is built under the table lock and delivered
// after release, mirroring BudgetEvent.
type TransitionEvent struct {
	TraceID        string   `json:"trace_id"`
	From           RunState `json:"from"`
	To             RunState `json:"to"`
	Reason         string   `json:"reason,omitempty"`
	ContinuationID string   `json:"continuation_id,omitempty"`
	Rev            uint64   `json:"rev"`
}

// TransitionObserver is the run-state boundary callback seam. The host owns
// fan-out and failure policy; the table only delivers typed transition values.
type TransitionObserver func(TransitionEvent)

// WatchBudget wires the pre-exhaustion warning + exhaustion observer. warnFraction is the
// consumed share (0..1) at which BudgetWarn fires — 0.8 warns at 80% of the context budget
// spent; a value <=0 or >=1 disables the warning (only BudgetExhausted then fires). obs==nil
// clears the seam (back to the no-op default). Safe to call on a live table; a nil receiver
// is a no-op.
func (t *Table) WatchBudget(warnFraction float64, obs BudgetObserver) {
	if t == nil {
		return
	}
	t.mu.Lock()
	t.obs = obs
	t.warnFrac = warnFraction
	t.mu.Unlock()
}

// WatchTransitions wires the run-state transition observer. obs==nil clears the
// seam. Safe to call on a live table; a nil receiver is a no-op.
func (t *Table) WatchTransitions(obs TransitionObserver) {
	if t == nil {
		return
	}
	t.mu.Lock()
	t.transObs = obs
	t.mu.Unlock()
}

// crossedWarnLocked reports whether THIS debit is the one that first pushes the context
// budget past the warning watermark: the prior remaining was above it and the new remaining
// is at or below it (but not yet exhausted — exhaustion is its own event). It fires exactly
// once per drain and only when an observer, a warn fraction in (0,1), and a known cap are
// all configured. Caller holds the lock.
func (t *Table) crossedWarnLocked(b Budget, prevLeft int) bool {
	if t.obs == nil || t.warnFrac <= 0 || t.warnFrac >= 1 || b.ContextTokensCap <= 0 {
		return false
	}
	w := warnWatermark(b.ContextTokensCap, t.warnFrac)
	return b.ContextTokensLeft > 0 && prevLeft > w && b.ContextTokensLeft <= w
}

// warnWatermark is the remaining-token level at which the consumed share first reaches
// frac: capacity*(1-frac), rounded (an 80% warn on a 100-token cap watermarks at 20
// remaining). Never negative.
func warnWatermark(capacity int, frac float64) int {
	w := int(math.Round(float64(capacity) * (1 - frac)))
	if w < 0 {
		w = 0
	}
	return w
}

// budgetEvent builds the observer payload from a freshly-written record. FractionConsumed
// is clamped to [0,1] so an over-debit (a single turn larger than the whole budget) still
// reports a sane 1.0 rather than a value past full.
func budgetEvent(st State, kind BudgetEventKind) BudgetEvent {
	capacity := st.Budget.ContextTokensCap
	var frac float64
	if capacity > 0 {
		frac = float64(capacity-st.Budget.ContextTokensLeft) / float64(capacity)
		if frac < 0 {
			frac = 0
		}
		if frac > 1 {
			frac = 1
		}
	}
	return BudgetEvent{
		Kind:              kind,
		TraceID:           st.TraceID,
		ContinuationID:    st.ContinuationID,
		Reason:            st.Reason,
		Rev:               st.Rev,
		ContextTokensLeft: st.Budget.ContextTokensLeft,
		ContextTokensCap:  capacity,
		FractionConsumed:  frac,
	}
}

func transitionEvent(st State, from, to RunState) TransitionEvent {
	reason := st.Reason
	if reason == "" {
		reason = canonicalReason(to)
	}
	return TransitionEvent{
		TraceID:        st.TraceID,
		From:           from,
		To:             to,
		Reason:         reason,
		ContinuationID: st.ContinuationID,
		Rev:            st.Rev,
	}
}

// notableTransition reports whether a move from->to is one a supervisor should be PUSHED
// about (#761): a flip INTO Paused (needs-input), Draining, or Stopped, and only when it
// actually changes the run-state (a no-op re-set must not notify). Running/Throttled are
// excluded — they are not "the agent is waiting / has stopped" signals, so the SIGCHLD-
// equivalent stays the terminal/paused boundary, fired exactly once per real transition.
func notableTransition(from, to RunState) bool {
	if from == to {
		return false
	}
	switch to {
	case Paused, Draining, Stopped:
		return true
	}
	return false
}

// canonicalReason maps a notable run-state to its closed stop-reason token — the default a
// transition push carries when the operator gave no free-text reason, so a push ALWAYS
// announces WHY with a recognized token (PAUSED / DRAINING / STOPPED) rather than a blank.
func canonicalReason(to RunState) string {
	switch to {
	case Paused:
		return ReasonPaused
	case Draining:
		return ReasonDrained
	case Stopped:
		return ReasonStopped
	}
	return ""
}
