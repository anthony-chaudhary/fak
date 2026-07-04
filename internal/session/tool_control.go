package session

import "github.com/anthony-chaudhary/fak/internal/abi"

// tool_control.go is the session-side contract for the tool/turn seam (#2630).
//
// A tool-call outcome answers "what happened to this proposed call?" A session
// control decision answers "what should the turn/session do next?" Those are
// separate control planes. The default invariant is deliberately boring:
// rejected, malformed, quarantined, or repaired tool calls are feedback values,
// and the session continues unless a declared session-control policy consumes
// those outcomes and emits a typed control decision.
//
// This file is intentionally owned by internal/session, not internal/policy: the
// live loop can act on these values without importing a higher-tier policy leaf.
// Policy may mirror the same tokens, but the session loop owns the state change.

// ToolCallOutcomeKind is the session-visible kind of one proposed tool call.
// It is not a turn/session decision; it is per-call feedback.
type ToolCallOutcomeKind uint8

const (
	// ToolCallOutcomeAllowed means the call survived adjudication and can make progress.
	ToolCallOutcomeAllowed ToolCallOutcomeKind = iota
	// ToolCallOutcomeRejected means the call was denied as a value with a closed reason.
	ToolCallOutcomeRejected
	// ToolCallOutcomeRepaired means the call was transformed into an admitted shape.
	ToolCallOutcomeRepaired
	// ToolCallOutcomeQuarantined means the call/result was held out of context.
	ToolCallOutcomeQuarantined
)

func (k ToolCallOutcomeKind) String() string {
	switch k {
	case ToolCallOutcomeAllowed:
		return "ALLOW"
	case ToolCallOutcomeRejected:
		return "REJECT"
	case ToolCallOutcomeRepaired:
		return "REPAIR"
	case ToolCallOutcomeQuarantined:
		return "QUARANTINE"
	default:
		return "TOOL_OUTCOME_UNKNOWN"
	}
}

// ToolCallDisposition is the actionable guidance attached to a refusal. It is
// feedback to the model/operator, not a session-control decision.
type ToolCallDisposition string

const (
	ToolDispositionNone      ToolCallDisposition = ""
	ToolDispositionRetryable ToolCallDisposition = "RETRYABLE"
	ToolDispositionWait      ToolCallDisposition = "WAIT"
	ToolDispositionEscalate  ToolCallDisposition = "ESCALATE"
	ToolDispositionTerminal  ToolCallDisposition = "TERMINAL"
)

// ToolCallOutcome is one call's adjudication result as consumed by the session
// loop. Reason is the closed per-tool refusal vocabulary; it must never be copied
// into the session-stop reason slot. If the surrounding turn made useful progress
// despite this call, set Progress=true so repeated-bad-call policies reset.
type ToolCallOutcome struct {
	Tool        string
	ToolCallID  string
	Kind        ToolCallOutcomeKind
	Reason      abi.ReasonCode
	Disposition ToolCallDisposition
	Progress    bool
}

// ReasonToken renders the per-tool refusal reason. An allowed/repaired call with
// no refusal returns "" so it cannot masquerade as a stop reason.
func (o ToolCallOutcome) ReasonToken() string {
	if o.Reason == abi.ReasonNone {
		return ""
	}
	return abi.ReasonName(o.Reason)
}

// BadForSessionControl reports whether this per-call outcome is eligible input
// to a declared repeated-bad-call policy. It does not make a control decision.
func (o ToolCallOutcome) BadForSessionControl() bool {
	if o.Progress || o.Reason == abi.ReasonNone {
		return false
	}
	return o.Kind == ToolCallOutcomeRejected || o.Kind == ToolCallOutcomeQuarantined
}

// DefaultControl is the load-bearing invariant: a tool-call outcome by itself
// keeps the turn/session going. Escalation requires a separate policy.
func (o ToolCallOutcome) DefaultControl() SessionControl {
	return SessionControl{Decision: SessionControlContinue}
}

// SessionControlDecision is the closed session-control vocabulary at this seam.
// It answers what the loop does, never why one tool call was refused.
type SessionControlDecision uint8

const (
	SessionControlContinue SessionControlDecision = iota
	SessionControlEndTurn
	SessionControlPause
	SessionControlStop
)

func (d SessionControlDecision) String() string {
	switch d {
	case SessionControlContinue:
		return "CONTINUE"
	case SessionControlEndTurn:
		return "END_TURN"
	case SessionControlPause:
		return "PAUSE_SESSION"
	case SessionControlStop:
		return "STOP_SESSION"
	default:
		return "SESSION_CONTROL_UNKNOWN"
	}
}

func (d SessionControlDecision) EndsTurn() bool {
	return d == SessionControlEndTurn || d == SessionControlPause || d == SessionControlStop
}

func (d SessionControlDecision) StopsSession() bool { return d == SessionControlStop }

// ReasonRepeatedToolRejection is the session-stop reason a declared policy uses
// when a run of bad tool-call outcomes reaches its stop threshold. It is
// deliberately not a tool refusal token such as MALFORMED.
const ReasonRepeatedToolRejection = "REPEATED_TOOL_REJECTION"

// SessionControl is the session loop's typed decision plus the evidence that
// produced it. ToolReason stays a per-call field; Reason names the control-plane
// reason only when Decision ends or stops the turn/session.
type SessionControl struct {
	Decision    SessionControlDecision
	Policy      string
	Reason      string
	Consecutive int
	Threshold   int
	Tool        string
	ToolCallID  string
	ToolReason  abi.ReasonCode
	Disposition ToolCallDisposition
}

func (c SessionControl) Continue() bool     { return c.Decision == SessionControlContinue }
func (c SessionControl) StopsSession() bool { return c.Decision.StopsSession() }

func (c SessionControl) ToolReasonToken() string {
	if c.ToolReason == abi.ReasonNone {
		return ""
	}
	return abi.ReasonName(c.ToolReason)
}

// SessionStopReason returns the declared session-control reason and true only
// for a real session stop. It never reads ToolReason.
func (c SessionControl) SessionStopReason() (string, bool) {
	if !c.StopsSession() {
		return "", false
	}
	return c.Reason, true
}

// RepeatedBadToolCallPolicy declares how a streak of bad tool outcomes escalates
// to session control. Zero thresholds mean "do not escalate"; that keeps a
// missing policy permissive for loop progress instead of recreating the accidental
// "four JSON errors stopped the session" failure mode.
type RepeatedBadToolCallPolicy struct {
	Name         string
	EndTurnAfter int
	StopAfter    int
	StopReason   string
}

func (p RepeatedBadToolCallPolicy) normalized() RepeatedBadToolCallPolicy {
	if p.Name == "" {
		p.Name = "repeated_bad_tool_call"
	}
	if p.StopReason == "" {
		p.StopReason = ReasonRepeatedToolRejection
	}
	if p.EndTurnAfter < 0 {
		p.EndTurnAfter = 0
	}
	if p.StopAfter < 0 {
		p.StopAfter = 0
	}
	if p.StopAfter > 0 && p.EndTurnAfter > 0 && p.StopAfter <= p.EndTurnAfter {
		p.StopAfter = p.EndTurnAfter + 1
	}
	return p
}

// RepeatedBadToolCallTracker is the per-session state for the declared policy.
// It counts only consecutive identical bad outcomes: same tool, same per-tool
// reason, same disposition, and no intervening progress.
type RepeatedBadToolCallTracker struct {
	Policy RepeatedBadToolCallPolicy

	count       int
	tool        string
	reason      abi.ReasonCode
	disposition ToolCallDisposition
}

// Reset clears the streak, for objective boundaries or manual operator retries.
func (t *RepeatedBadToolCallTracker) Reset() {
	t.count = 0
	t.tool = ""
	t.reason = abi.ReasonNone
	t.disposition = ToolDispositionNone
}

// Observe folds one per-call outcome into the declared policy and returns the
// session-control decision. Bad outcomes below threshold still return CONTINUE.
func (t *RepeatedBadToolCallTracker) Observe(o ToolCallOutcome) SessionControl {
	if !o.BadForSessionControl() {
		t.Reset()
		return o.DefaultControl()
	}
	if t.count > 0 && t.tool == o.Tool && t.reason == o.Reason && t.disposition == o.Disposition {
		t.count++
	} else {
		t.count = 1
		t.tool = o.Tool
		t.reason = o.Reason
		t.disposition = o.Disposition
	}

	p := t.Policy.normalized()
	ctrl := SessionControl{
		Decision:    SessionControlContinue,
		Policy:      p.Name,
		Consecutive: t.count,
		Tool:        o.Tool,
		ToolCallID:  o.ToolCallID,
		ToolReason:  o.Reason,
		Disposition: o.Disposition,
	}
	switch {
	case p.StopAfter > 0 && t.count >= p.StopAfter:
		ctrl.Decision = SessionControlStop
		ctrl.Reason = p.StopReason
		ctrl.Threshold = p.StopAfter
	case p.EndTurnAfter > 0 && t.count >= p.EndTurnAfter:
		ctrl.Decision = SessionControlEndTurn
		ctrl.Reason = p.StopReason
		ctrl.Threshold = p.EndTurnAfter
	}
	return ctrl
}
