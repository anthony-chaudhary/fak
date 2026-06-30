package loopdrive

import (
	"strings"

	"github.com/anthony-chaudhary/fak/internal/taskmgr"
)

// HandoffOutcome is the loop-drive completion-boundary verdict for the task
// handoff gate. It folds the three real cases the loop driver must distinguish:
// there was no agent task at this stop (fail-open), there was one and it carried
// a valid handoff (pass), or there was one and the handoff was missing/malformed
// (gated).
type HandoffOutcome string

const (
	// HandoffFailOpen means no agent task handoff was attached to this completion.
	// An ordinary non-agent stop must never be blocked, so the loop proceeds.
	HandoffFailOpen HandoffOutcome = "fail_open"
	// HandoffPass means a handoff was present and the fak.task-handoff.v1 gate
	// accepted it: a witnessed-done task with one/two next steps or an explicit
	// no_next_step_reason.
	HandoffPass HandoffOutcome = "pass"
	// HandoffGated means a handoff was present but the gate refused it. The
	// witnessed-done task did not push a next step or an explicit reason, so the
	// loop-drive completion is blocked until the agent supplies one.
	HandoffGated HandoffOutcome = "gated"
)

// HandoffGateReason names why HandoffGate reached its outcome, without leaking
// the raw taskmgr review shape to callers that only need a one-line summary.
const (
	// ReasonNoAgentTask is attached to a HandoffFailOpen outcome.
	ReasonNoAgentTask = "NO_AGENT_TASK"
	// ReasonHandoffOK is attached to a HandoffPass outcome.
	ReasonHandoffOK = "HANDOFF_OK"
	// ReasonHandoffRefused is attached to a HandoffGated outcome; the gate's own
	// closed reason codes are carried in HandoffGateResult.Reasons.
	ReasonHandoffRefused = "HANDOFF_REFUSED"
)

// HandoffGateResult is the loop-drive completion verdict plus the underlying
// fak.task-handoff.v1 review (zero-valued when no handoff was present).
type HandoffGateResult struct {
	Outcome HandoffOutcome
	Reason  string
	Summary string
	Reasons []string
	Review  taskmgr.HandoffReview
}

// OK reports whether the loop driver may complete: either there was no agent
// task to gate, or the handoff passed the gate.
func (r HandoffGateResult) OK() bool {
	return r.Outcome == HandoffFailOpen || r.Outcome == HandoffPass
}

// HandoffGate is the pure completion-boundary decision. It is intentionally
// spawn-free and side-effect-free: the command shell decides whether an agent
// task handoff is present (present=false for an ordinary non-agent stop) and
// supplies the parsed record.
//
//   - present == false: fail-open. A plain loop-drive completion with no agent
//     task must not be blocked.
//   - present == true: require, via the existing taskmgr.ReviewHandoff gate, a
//     witnessed-done task with one/two concrete next_steps or an explicit
//     no_next_step_reason. The verdict is the gate's, re-projected onto the
//     loop-drive outcome vocabulary.
func HandoffGate(present bool, h taskmgr.Handoff) HandoffGateResult {
	if !present {
		return HandoffGateResult{
			Outcome: HandoffFailOpen,
			Reason:  ReasonNoAgentTask,
			Summary: "no agent task handoff at this completion; fail-open",
		}
	}
	review := taskmgr.ReviewHandoff(h)
	if review.OK {
		return HandoffGateResult{
			Outcome: HandoffPass,
			Reason:  ReasonHandoffOK,
			Summary: handoffPassSummary(review),
			Review:  review,
		}
	}
	return HandoffGateResult{
		Outcome: HandoffGated,
		Reason:  ReasonHandoffRefused,
		Summary: "task handoff refused: " + strings.Join(review.Reasons, ","),
		Reasons: review.Reasons,
		Review:  review,
	}
}

func handoffPassSummary(review taskmgr.HandoffReview) string {
	if review.IssueCount == 0 {
		return "task handoff accepted: explicit no-next-step reason"
	}
	if review.IssueCount == 1 {
		return "task handoff accepted: 1 next step pushed"
	}
	return "task handoff accepted: next steps pushed"
}
