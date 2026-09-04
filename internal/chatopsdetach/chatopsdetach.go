package chatopsdetach

import (
	"fmt"
	"time"
)

// Verb defines the mutating chatops action requested by an operator.
type Verb string

const (
	// VerbDispatch launches a background issue-resolution run.
	VerbDispatch Verb = "dispatch"
	// VerbResume resumes a paused or stalled execution loop.
	VerbResume Verb = "resume"
	// VerbBench triggers a background benchmark run.
	VerbBench Verb = "bench"
)

// Command encapsulates an inbound chatops request with its transport idempotency key.
type Command struct {
	Nonce   string
	Verb    Verb
	Target  string
	Channel string
	User    string
}

// Admission captures the front-door preflight decision for a requested command.
type Admission struct {
	Admitted bool
	RunID    string
	Lane     string
	Reason   string
}

// Record represents durable execution state tracked per command nonce to ensure idempotency.
type Record struct {
	Nonce string
	RunID string
	Lane  string
}

// Dispatched reports whether this nonce has already launched an execution run.
func (r Record) Dispatched() bool { return r.RunID != "" }

// Action denotes the lifecycle step to take in response to a command delivery.
type Action int

const (
	// Dispatch initiates a new background run when preflight admission succeeds.
	Dispatch Action = iota
	// ReAck re-emits a deterministic acknowledgment for an already-dispatched nonce.
	ReAck
	// Refuse rejects an unadmitted command without consuming execution capacity.
	Refuse
)

// String returns the diagnostic string representation of the action.
func (a Action) String() string {
	switch a {
	case Dispatch:
		return "DISPATCH"
	case ReAck:
		return "RE_ACK"
	case Refuse:
		return "REFUSE"
	default:
		return fmt.Sprintf("Action(%d)", int(a))
	}
}

// Decision contains the resolved action, outbox message, and durable record state.
type Decision struct {
	Action Action
	RunID  string
	Ack    string
	Reason string
	Record Record
}

// Decide resolves a command delivery against admission and prior records deterministically.
func Decide(cmd Command, adm Admission, prior Record) Decision {
	if prior.Dispatched() {
		return Decision{
			Action: ReAck,
			RunID:  prior.RunID,
			Ack:    ackText(cmd, prior.RunID, prior.Lane),
			Record: prior,
		}
	}
	if adm.Admitted {
		rec := Record{Nonce: cmd.Nonce, RunID: adm.RunID, Lane: adm.Lane}
		return Decision{
			Action: Dispatch,
			RunID:  adm.RunID,
			Ack:    ackText(cmd, adm.RunID, adm.Lane),
			Record: rec,
		}
	}
	return Decision{
		Action: Refuse,
		Reason: refuseText(cmd, adm.Reason),
		Record: Record{Nonce: cmd.Nonce},
	}
}

func ackText(cmd Command, runID, lane string) string {
	line := fmt.Sprintf(":hourglass_flowing_sand: %s %s — admitted as run `%s`", verbLabel(cmd.Verb), operand(cmd), runID)
	if lane != "" {
		line += " on lane " + lane
	}
	return line
}

func refuseText(cmd Command, reason string) string {
	if reason == "" {
		reason = "REFUSE_UNSPECIFIED"
	}
	return fmt.Sprintf(":no_entry: %s %s refused: %s — no run started, no seat taken", verbLabel(cmd.Verb), operand(cmd), reason)
}

func verbLabel(v Verb) string {
	if v == "" {
		return "command"
	}
	return string(v)
}

func operand(cmd Command) string {
	if cmd.Target == "" {
		return "(no target)"
	}
	return cmd.Target
}

// Stall tracks silence duration against allowed budget for detached run liveness.
type Stall struct {
	RunID     string
	Issue     string
	SilentFor time.Duration
	Budget    time.Duration
}

// PageMultiple defines the silence budget multiple that escalates a stall to operator alert.
const PageMultiple = 2

// SeverityStatus flags an advisory notice for non-urgent background issues.
const SeverityStatus = "status"

// SeverityOperator triggers immediate operator escalation when budgets are heavily exceeded.
const SeverityOperator = "operator"

// Escalation communicates whether and how urgently a stalled run should alert operators.
type Escalation struct {
	Escalate bool
	Severity string
	Text     string
}

// JudgeStall evaluates observed silence against budgeted duration to determine escalation level.
func JudgeStall(s Stall) Escalation {
	if s.Budget <= 0 || s.SilentFor <= s.Budget {
		return Escalation{Escalate: false}
	}
	sev := SeverityStatus
	if s.SilentFor >= time.Duration(PageMultiple)*s.Budget {
		sev = SeverityOperator
	}
	return Escalation{Escalate: true, Severity: sev, Text: stallText(s, sev)}
}

func stallText(s Stall, sev string) string {
	lead := ":warning:"
	if sev == SeverityOperator {
		lead = ":rotating_light: <!here>"
	}
	issue := ""
	if s.Issue != "" {
		issue = " (" + s.Issue + ")"
	}
	return fmt.Sprintf("%s dispatch run `%s`%s silent for %s (budget %s) — no witnessed progress; needs a look",
		lead, s.RunID, issue, s.SilentFor, s.Budget)
}
