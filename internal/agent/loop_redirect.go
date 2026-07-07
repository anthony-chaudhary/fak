package agent

// loop_redirect.go — the loop-side consumer of the out-of-band redirect / set-objective
// op (#2755), the objective-change twin of drainSteer (loop_session.go). Where
// drainSteer splices operator PROSE into the turn as a user MESSAGE, applyRedirect
// folds a structured redirect into the session's live OBJECTIVE and returns that
// objective's standing directive to carry into the turn as a first-class objective
// change — not another operator turn. This is the distinction #2755 exists to draw:
// redirect is APPLIED (consumed into objective state), steer is spliced (interpreted
// as prose).

import "github.com/anthony-chaudhary/fak/internal/sessionctl"

// applyRedirect drains any operator redirect enqueued for this run on the sessionctl
// mailbox at the turn boundary, folds it into the live per-session objective, and
// returns the current objective's model-facing directive to carry into THIS turn
// ("" = no objective / nothing to carry). Draining at the boundary — not at enqueue
// time — keeps an in-flight turn from being re-objectived beneath itself, the same
// clean-boundary discipline as drainSteer and the constraint mailbox.
//
// The channel is keyed by the run's trace id (the same id WithSessionTable /
// WithSessionGate wire), so a run with no trace (c.trace == "") has no mailbox and
// carries nothing — the historical loop is byte-for-byte unchanged. Once an objective
// is set, its directive is carried EVERY turn: the objective is a standing goal, not a
// one-shot splice, so the loop keeps pursuing it until a later redirect replaces it.
func (c runConfig) applyRedirect() string {
	if c.trace == "" {
		return ""
	}
	// Consume any queued redirect into the live objective. Refusals (e.g. a terminal
	// objective) are dropped here — the mailbox is drained either way; the operator
	// saw the closed refusal reason at the enqueue edge.
	sessionctl.ApplyPendingRedirect(c.trace)
	obj, ok := sessionctl.CurrentObjective(c.trace)
	if !ok {
		return ""
	}
	return obj.Directive()
}
