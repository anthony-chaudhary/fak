package agent

// loop_constraint.go — the loop-side consumer of the out-of-band add-constraint
// op (#2756), the floor-tightening twin of applyRedirect (loop_redirect.go).
// Where applyRedirect folds a structured redirect into the live OBJECTIVE,
// applyConstraints folds queued operator constraints into the live per-session
// constraint FLOOR at the turn boundary — never mid-turn, so an in-flight turn
// is not re-floored beneath itself — and returns the floor's standing directive
// to carry into the turn as a SYSTEM notice. constraintDenied is the per-call
// fold-in: the loop's tool dispatch consults the tightened floor BEFORE
// dispatching each call, so a tool forbidden mid-session is denied — never
// sent — with a typed receipt carrying the closed CONSTRAINT_* reason.

import "github.com/anthony-chaudhary/fak/internal/sessionctl"

// applyConstraints drains any operator add-constraint ops enqueued for this run
// on the sessionctl mailbox at the turn boundary, folds them into the live
// per-session floor, and returns the floor's model-facing standing directive
// ("" = unconstrained / no trace, so the historical loop is byte-for-byte
// unchanged). Refusals (e.g. a widen attempt racing a narrower lane) are
// dropped here — the mailbox is drained either way; the refusal is witnessed on
// the constraint Next records and at the enqueue edge.
func (c runConfig) applyConstraints() string {
	if c.trace == "" {
		return ""
	}
	sessionctl.ApplyPendingConstraints(c.trace)
	constraintFloor, ok := sessionctl.CurrentConstraintFloor(c.trace)
	if !ok {
		return ""
	}
	return constraintFloor.Directive()
}

// constraintDenied checks one tool call against the live tightened floor. When
// the floor denies it, the receipt content and trace event are filled with the
// closed CONSTRAINT_* reason and true is returned — the call must never be
// dispatched. A run with no trace or an unconstrained floor always allows, so
// the historical loop is unchanged. The target path is not extracted from raw
// args here (the loop cannot know every tool's argument shape); path-shaped
// deny rules and the lane bind at callers that know their target — tool-name
// tightening is what the loop enforces live.
func (c runConfig) constraintDenied(tool string, content *string, ev *traceEvent) bool {
	if c.trace == "" {
		return false
	}
	ref := sessionctl.ConstraintDenies(c.trace, tool, "")
	if ref == nil {
		return false
	}
	*content = ToolReceipt{
		Status:      ToolResultSkipped,
		Reason:      string(ref.Reason),
		Disposition: "TERMINAL",
		Fix:         "the operator forbade this tool mid-session; pursue the objective without it",
		Detail:      "denied by the out-of-band tightened session floor; never dispatched",
	}.JSON()
	ev.Verdict = "DENY"
	ev.Reason = string(ref.Reason)
	ev.By = "session-constraint"
	ev.Disposition = "TERMINAL"
	ev.Note = "DENIED by the out-of-band tightened session floor (#2756): " + ref.Error()
	return true
}
