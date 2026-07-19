package agent

// loop_park.go — the loop-side consumer of the out-of-band operator approve/deny
// inbox (#2757), the gate-resolution twin of constraintDenied (loop_constraint.go).
// Where constraintDenied denies a floor-forbidden call BEFORE dispatch,
// parkEscalatedDeny intercepts a call the adjudication gate ALREADY refused with
// an ESCALATE disposition (the reversibility/witness family: preview + confirm
// token, satisfiable at HEAD only by the agent self-confirming) and PARKS it on
// the sessionctl pending-action queue for an EXTERNAL operator verdict:
//
//   - approve  -> the loop proceeds: it re-proposes the call through the NORMAL
//     syscall boundary — byte-identical plus the gate's own confirm-token echo,
//     or the operator's modified args freshly adjudicated — so the core-locked
//     adjudicator is untouched and still sees/journals every dispatch;
//   - deny     -> the loop aborts the call with a typed receipt carrying the
//     closed PARK_OPERATOR_DENIED reason; the call is never dispatched;
//   - timeout/abort -> explicit, never silent: a typed receipt with the closed
//     PARK_TIMEOUT / PARK_ABORTED reason (sessionctl witnessed the outcome).
//
// Parking is opt-in per session (sessionctl.EnableGateParking): a run with no
// trace, a non-ESCALATE outcome, or an unopened inbox falls straight through, so
// the historical loop is byte-for-byte unchanged.

import (
	"context"
	"encoding/json"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/kernel"
	"github.com/anthony-chaudhary/fak/internal/sessionctl"
)

// reversibilityConfirmArg mirrors adjudicator.ReversibilityConfirmArg — the
// argument key the reversibility gate's advertised "re-propose byte-identical +
// add _fak_confirm" recovery reads. Mirrored as a literal so the loop does not
// import the core-locked adjudicator package.
const reversibilityConfirmArg = "_fak_confirm"

// parkEscalatedDeny folds the operator inbox into the loop's dispatch site: when
// the just-adjudicated call came back an ESCALATE-gated DENY and the session's
// inbox is open, the loop parks on the gate until the operator verdict resolves
// it, then returns the honored outcome — the approved re-dispatch's real
// content/event, or the typed abort receipt. Every other call returns content/ev
// unchanged.
func (c runConfig) parkEscalatedDeny(ctx context.Context, k *kernel.Kernel, tool, rawArgs, engine, content string, ev traceEvent) (string, traceEvent) {
	if c.trace == "" || ev.Verdict != "DENY" || ev.Disposition != "ESCALATE" || !sessionctl.GateParkingEnabled(c.trace) {
		return content, ev
	}
	verdict, ref := sessionctl.ParkGatedAction(ctx, c.trace, sessionctl.GatedAction{
		Tool: tool, Args: rawArgs, Reason: ev.Reason, Preview: content,
	})
	if ref != nil {
		// Unresolved (timeout / cancelled park): explicit typed abort, never a
		// silent drop — the model reads the closed reason and adapts.
		ev.Reason = string(ref.Reason)
		ev.Disposition = "TERMINAL"
		ev.By = "session-park"
		ev.Note = "PARKED gated call aborted unresolved (#2757): " + ref.Error()
		return ToolReceipt{
			Status:      ToolResultError,
			Reason:      string(ref.Reason),
			Disposition: "TERMINAL",
			Fix:         "no operator verdict resolved this parked action; pursue the objective without it or re-propose for a fresh operator window",
			Detail:      "gated call parked on the operator inbox and aborted unresolved; never dispatched",
		}.JSON(), ev
	}
	if verdict.Kind == sessionctl.ParkDeny {
		ev.Reason = string(sessionctl.ParkOperatorDenied)
		ev.Disposition = "TERMINAL"
		ev.By = "session-park"
		ev.Note = "OPERATOR DENIED out of band (#2757): parked gated call aborted; never dispatched"
		return ToolReceipt{
			Status:      ToolResultError,
			Reason:      string(sessionctl.ParkOperatorDenied),
			Disposition: "TERMINAL",
			Fix:         "the operator denied this gated action out of band; pursue the objective without it",
			Detail:      "parked on the operator inbox and denied by the external verdict; never dispatched",
		}.JSON(), ev
	}
	// Approve: proceed through the NORMAL syscall boundary. Byte-identical
	// re-propose carries the gate's own confirm echo; operator-modified args are
	// dispatched as supplied for a FRESH adjudication (the old token binds only
	// the original bytes — modify never skips the gate). If the gate still
	// refuses, the fresh deny receipt returns unparked: one park per action.
	retryArgs := rawArgs
	if strings.TrimSpace(verdict.Args) != "" {
		retryArgs = verdict.Args
	} else if ev.ConfirmToken != "" {
		if confirmed, err := confirmArgs(rawArgs, ev.ConfirmToken); err == nil {
			retryArgs = confirmed
		}
	}
	redo := traceEvent{Turn: ev.Turn, Arm: ev.Arm, Tool: tool, RawArgs: retryArgs}
	content, ev = execViaKernel(ctx, k, tool, retryArgs, engine, redo)
	ev.Note = strings.TrimSuffix("OPERATOR APPROVED out of band (#2757): parked gated call re-proposed per the verdict; "+ev.Note, "; ")
	return content, ev
}

// confirmArgs re-renders the call args with the gate's confirm-token echo added
// under the reversibility confirm key — the gate's advertised recovery, authored
// by the loop on the operator's behalf.
func confirmArgs(rawArgs, token string) (string, error) {
	args := map[string]any{}
	if trimmed := strings.TrimSpace(rawArgs); trimmed != "" {
		if err := json.Unmarshal([]byte(trimmed), &args); err != nil {
			return "", err
		}
	}
	args[reversibilityConfirmArg] = token
	b, err := json.Marshal(args)
	if err != nil {
		return "", err
	}
	return string(b), nil
}
