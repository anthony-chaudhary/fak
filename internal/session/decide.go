package session

// decide.go — the per-turn gate. Decide is the ONE call the agent turn loop makes
// at each boundary; the whole point of this package is that the loop ASKS a value
// instead of reconstructing one. It folds the run-state machine and the budget
// debit into a single verdict: proceed (and under what per-turn cap) or stop (and
// why). Budget exhaustion here is the moment a session becomes Draining — the
// scheduling event a supervisor observes (a slot just freed) without a process scan.

// Verdict is what Decide returns to the turn loop. Proceed gates the loop: false
// ends the session this boundary. MaxTokens is the per-turn output cap to lower
// into the planner (0 = planner default). State is the (possibly just-debited)
// drive record. Stop is true exactly when the session has reached a terminal
// boundary this turn (Stopped, or Draining taken now); Reason names which closed
// cause, so the loop and a supervisor agree on why the slot freed.
type Verdict struct {
	Proceed   bool
	MaxTokens int
	MinGapMs  int
	State     State
	Stop      bool
	Reason    string
}

// QueryBudgetVerdict is the clarification/self-query budget gate. It is separate
// from Verdict because a query-budget miss should degrade the clarification path,
// not stop the main session.
type QueryBudgetVerdict struct {
	Proceed   bool
	Stop      bool
	Reason    string
	Remaining int
	State     State
}

// Stop reason tokens — the closed vocabulary Decide stamps, so "why did this turn
// not run" is a checkable field, never free text. They mirror the refusal-reason
// discipline the kernel uses elsewhere.
const (
	ReasonBudgetTurns          = "BUDGET_TURNS_EXHAUSTED"     // TurnsLeft hit zero
	ReasonBudgetTokens         = "BUDGET_TOKENS_EXHAUSTED"    // TokensLeft hit zero
	ReasonBudgetContext        = "BUDGET_CONTEXT_EXHAUSTED"   // ContextTokensLeft hit zero
	ReasonBudgetQueries        = "BUDGET_QUERIES_EXHAUSTED"   // ClarificationQueriesLeft hit zero
	ReasonBudgetToolCalls      = "BUDGET_TOOLCALLS_EXHAUSTED" // ToolCallsLeft hit zero — the runaway floor (#2887), debited per dispatched tool call, not per turn
	ReasonBudgetSpend          = "BUDGET_SPEND_EXHAUSTED"     // SpendMicroCentsLeft hit zero (priced dollar ceiling); never auto-reset — a spent cap is terminal, not a fresh-window continuation
	ReasonPaused               = "PAUSED"                     // operator hold; not terminal, the loop waits
	ReasonDrained              = "DRAINING"                   // operator stop, taken at this boundary
	ReasonTerminated           = "TERMINATED"                 // operator FORCEFUL stop (#2758): in-flight work cancelled at the next safe point, no new work — the deliberate counterpart of DRAINING (which lets the turn finish)
	ReasonStopped              = "STOPPED"                    // already terminal
	ReasonBudgetReset          = "BUDGET_RESET"               // budget-drained, then re-armed on a fresh window (Recontinue)
	ReasonProviderSessionClear = "PROVIDER_SESSION_CLEAR"     // provider /clear or /new ended one visible conversation and opened another
	ReasonResumeCancelled      = "RESUME_CANCELLED"           // a WaitResume parked on a Paused session ended because its context was cancelled (#916)
	ReasonInterrupted          = "INTERRUPTED"                // operator MID-FLIGHT interrupt (#5158), stamped by the loop (not Decide): the arm stops at its next clean turn boundary — softer than DRAINING (no drive-state transition) and never mid-tool, unlike TERMINATED
)

// Decide is the per-turn boundary gate. Given a session's TraceID it:
//
//  1. reads the current drive record (default Running/unbounded if unseen),
//  2. resolves terminal/paused/drain run-states to a non-proceed verdict WITHOUT
//     debiting (a held or stopped session must not burn budget),
//  3. for a live (Running/Throttled) session, debits one turn and checks the
//     remaining budget — a zero/negative remaining axis drives the session to
//     Draining (a real write, Rev bumped) so the exhaustion is observable, and
//     returns a non-proceed verdict carrying the exhaustion reason,
//  4. otherwise returns Proceed=true with the per-turn pace cap (MaxTokensPerTurn,
//     MinTurnGapMs) for the loop to apply.
//
// It is the table's only read-modify-write on the hot path; it takes the lock once.
// A nil receiver is a valid no-op-permissive gate (Proceed=true, no cap) so a loop
// with no table wired behaves byte-identically to the pre-table path — the caller
// does not need a nil check.
func (t *Table) Decide(trace string) Verdict {
	if t == nil {
		return Verdict{Proceed: true, State: DefaultState(trace)}
	}
	t.mu.Lock()
	defer t.mu.Unlock()
	cur := t.getLocked(trace)

	// (2) Non-advancing run-states: resolve without debiting. Shared with DebitToolCall
	// via nonAdvancingVerdictLocked so the Stopped/Paused/Draining/Terminating
	// finalization has ONE definition and the two gates cannot drift.
	if v, done := t.nonAdvancingVerdictLocked(cur); done {
		return v
	}

	// (3) Live session: debit one turn, then check the budget. The debit happens
	// BEFORE the check so a session with TurnsLeft==1 runs this turn and stops at the
	// next boundary — the allotment is "turns remaining", inclusive of the current.
	if !cur.Budget.turnsUnbounded() {
		cur.Budget.TurnsLeft--
		if cur.Budget.TurnsLeft < 0 {
			// Already exhausted on a prior turn: drive to Draining and stop now.
			cur.Run = Draining
			cur.Reason = ReasonBudgetTurns
			final := t.finalizeDrainLocked(cur)
			return Verdict{Proceed: false, Stop: true, Reason: final.Reason, State: final}
		}
	}
	if !cur.Budget.tokensUnbounded() && cur.Budget.TokensLeft <= 0 {
		cur.Run = Draining
		cur.Reason = ReasonBudgetTokens
		final := t.finalizeDrainLocked(cur)
		return Verdict{Proceed: false, Stop: true, Reason: final.Reason, State: final}
	}

	// (4) Proceed. Persist the turn debit and hand back the pace cap.
	out := t.putLocked(cur)
	return Verdict{
		Proceed:   true,
		MaxTokens: out.Pace.MaxTokensPerTurn,
		MinGapMs:  out.Pace.MinTurnGapMs,
		State:     out,
	}
}

// Debit decrements a session's remaining token budget by the reported usage of a
// just-completed turn, after the planner returns. Decide debits turns (it knows a
// turn is starting); only the loop knows the actual token usage, so it reports it
// here. A terminal/paused session is left unchanged. It returns the new state. A
// nil receiver is a no-op. Crossing zero is observed by the NEXT Decide (which sees
// TokensLeft<=0 and drains) — Debit itself does not transition, keeping the "stop
// is taken at a boundary" invariant.
func (t *Table) Debit(trace string, tokensUsed int) State {
	return t.DebitUsage(trace, Usage{OutputTokens: tokensUsed})
}

// DebitClarificationQuery spends one clarification/self-query ask from the
// session's query budget. Unconfigured budgets are permissive and do not create a
// session record. Exhaustion refuses only the clarification path: the main
// session remains live so the caller can continue with known context or ask the
// user through a different governed path.
func (t *Table) DebitClarificationQuery(trace string) QueryBudgetVerdict {
	if t == nil {
		return QueryBudgetVerdict{Proceed: true, Remaining: Unbounded, State: DefaultState(trace)}
	}
	t.mu.Lock()
	defer t.mu.Unlock()
	cur := t.getLocked(trace)

	switch cur.Run {
	case Stopped:
		return QueryBudgetVerdict{Proceed: false, Stop: true, Reason: cur.stopReasonOr(ReasonStopped), Remaining: cur.Budget.ClarificationQueriesLeft, State: cur}
	case Draining:
		return QueryBudgetVerdict{Proceed: false, Stop: true, Reason: cur.stopReasonOr(ReasonDrained), Remaining: cur.Budget.ClarificationQueriesLeft, State: cur}
	case Terminating:
		return QueryBudgetVerdict{Proceed: false, Stop: true, Reason: cur.stopReasonOr(ReasonTerminated), Remaining: cur.Budget.ClarificationQueriesLeft, State: cur}
	case Paused:
		return QueryBudgetVerdict{Proceed: false, Reason: ReasonPaused, Remaining: cur.Budget.ClarificationQueriesLeft, State: cur}
	}

	if !cur.Budget.clarificationQueriesBounded() {
		return QueryBudgetVerdict{Proceed: true, Remaining: Unbounded, State: cur}
	}
	if cur.Budget.ClarificationQueriesLeft <= 0 {
		return QueryBudgetVerdict{Proceed: false, Reason: ReasonBudgetQueries, Remaining: 0, State: cur}
	}
	cur.Budget.ClarificationQueriesLeft--
	out := t.putLocked(cur)
	return QueryBudgetVerdict{Proceed: true, Remaining: out.Budget.ClarificationQueriesLeft, State: out}
}

// DebitToolCall spends one DISPATCHED tool call from the session's runaway floor
// (#2887, the Hermes cron-hardening angle) and reports whether the loop may proceed
// with this call. It is the tool-call twin of the turn debit in Decide, but taken
// per tool call rather than per turn boundary, so a single turn that emits a long
// tool-call loop is cut at the budget too — the structural floor a scheduled agent's
// model cannot extend. An unbounded (unconfigured) axis proceeds without a record; a
// held/terminal session proceeds no new work; a live session with the ceiling already
// spent drives to Draining/Stopped with ReasonBudgetToolCalls, exactly like a turn or
// token exhaustion, so the stop is observable in the drive state and witnessed on exit.
// A nil receiver is a permissive no-op so a loop with no table behaves byte-identically
// to the pre-axis path.
func (t *Table) DebitToolCall(trace string) Verdict {
	if t == nil {
		return Verdict{Proceed: true, State: DefaultState(trace)}
	}
	t.mu.Lock()
	defer t.mu.Unlock()
	cur := t.getLocked(trace)

	// Non-advancing run-states: resolve without debiting (shared with Decide).
	if v, done := t.nonAdvancingVerdictLocked(cur); done {
		return v
	}

	if !cur.Budget.toolCallsBounded() {
		return Verdict{Proceed: true, State: cur}
	}
	if cur.Budget.ToolCallsLeft <= 0 {
		cur.Run = Draining
		cur.Reason = ReasonBudgetToolCalls
		final := t.finalizeDrainLocked(cur)
		return Verdict{Proceed: false, Stop: true, Reason: final.Reason, State: final}
	}
	cur.Budget.ToolCallsLeft--
	out := t.putLocked(cur)
	return Verdict{Proceed: true, State: out}
}

// nonAdvancingVerdictLocked resolves a non-advancing run-state to a non-proceed
// Verdict WITHOUT debiting any budget: Stopped/Paused report the held/terminal cause,
// and Draining/Terminating are finalized to Stopped idempotently (a real write, so a
// later call is stable). It returns (verdict, true) when cur.Run is non-advancing and
// the caller must return that verdict as-is; (zero, false) when the session is live
// (Running/Throttled) and the caller proceeds to its own debit path. Caller holds
// t.mu. This is the shared head of Decide and DebitToolCall — one definition of the
// run-state finalization so the two hot-path gates cannot drift.
func (t *Table) nonAdvancingVerdictLocked(cur State) (Verdict, bool) {
	switch cur.Run {
	case Stopped:
		return Verdict{Proceed: false, Stop: true, Reason: cur.stopReasonOr(ReasonStopped), State: cur}, true
	case Paused:
		// A hold, not an end: the loop must not burn a turn, but the session is not
		// terminal — a resume flips it back to Running. Stop stays false.
		return Verdict{Proceed: false, Stop: false, Reason: ReasonPaused, State: cur}, true
	case Draining:
		// The stop was requested earlier; take it at THIS boundary and finalize to
		// Stopped so a later call is idempotent and the record reflects the end.
		cur.Run = Stopped
		if cur.Reason == "" {
			cur.Reason = ReasonDrained
		}
		final := t.putLocked(cur)
		return Verdict{Proceed: false, Stop: true, Reason: final.Reason, State: final}, true
	case Terminating:
		// The forceful stop (#2758) was signalled mid-turn; the arm has usually
		// already aborted in-flight work on the terminate signal and calls back in
		// here to finalize. Same idempotent finalize as Draining, but with the
		// TERMINATED reason so the record never claims the drain's ran-to-completion
		// semantics.
		cur.Run = Stopped
		if cur.Reason == "" {
			cur.Reason = ReasonTerminated
		}
		final := t.putLocked(cur)
		return Verdict{Proceed: false, Stop: true, Reason: final.Reason, State: final}, true
	}
	return Verdict{}, false
}

// finalizeDrainLocked writes a budget-exhausted record straight to Stopped (the
// exhaustion is taken at this boundary, like an explicit Draining is). It is the
// shared tail of the two budget arms in Decide. Caller holds the lock.
func (t *Table) finalizeDrainLocked(cur State) State {
	cur.Run = Stopped
	return t.putLocked(cur)
}

// stopReasonOr returns the session's recorded Reason, or a fallback when none was
// stamped — so a Stopped session with an empty Reason still reports a closed token.
func (s State) stopReasonOr(fallback string) string {
	if s.Reason != "" {
		return s.Reason
	}
	return fallback
}
