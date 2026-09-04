package chatopsdetach

import (
	"fmt"
	"time"
)

// Verb is one chatops ACT verb — the mutating half of the control surface (start
// a detached run, resume a stalled loop), as opposed to a read-only query. The
// door (C4) maps a parsed message onto one of these; the kernel only labels it.
type Verb string

const (
	VerbDispatch Verb = "dispatch" // start a background issue-resolution run
	VerbResume   Verb = "resume"   // resume a stalled loop/run
	VerbBench    Verb = "bench"    // kick a background benchmark
)

// Command is one inbound act verb as the chatops door parsed it, carrying the
// idempotency Nonce the transport delivered it under (the Slack message ts, per
// slackwire's IdemEventType contract). The Nonce is the ONLY thing that makes a
// re-delivery recognizable: Slack has no server-side idempotency, so a dropped
// ack or a retried event can deliver the same command twice, and the kernel must
// collapse those to one run.
type Command struct {
	Nonce   string // the transport idempotency key (Slack message ts)
	Verb    Verb   // the act verb
	Target  string // the verb's operand: an issue ("#2265"), a run id, ...
	Channel string // where the ack/refusal is posted
	User    string // who issued it (for the audit trail; not used in routing)
}

// Admission is the guarded-dispatch front door's decision for a command, folded
// from internal/dispatchtick.EvaluatePreflight by the shell. The kernel NEVER
// makes this decision and NEVER mints a run: seats, lane leases, host capacity
// and the gate-pressure term are the preflight's job. The kernel only ROUTES an
// already-made verdict, which is exactly what keeps a refusal from queue-jumping
// the cap — there is no path here that dispatches without an Admitted verdict.
type Admission struct {
	Admitted bool   // SPAWN_OK from the preflight
	RunID    string // the minted run id, set iff Admitted
	Lane     string // the lane the run was admitted onto, for the ack line
	// Reason is a closed refusal token set iff !Admitted — one of the preflight's
	// vocabulary (REFUSE_AT_CAP, REFUSE_NO_SEAT, REFUSE_NO_ACCOUNT, REFUSE_HOST,
	// REFUSE_GATE/GATE_PRESSURE). The kernel treats it as opaque and threads it
	// verbatim so `dos man wedge <TOKEN> --explain` can still bind the token downstream.
	Reason string
}

// Record is the durable spool row the shell persists per command Nonce so a
// re-delivered command re-acks the SAME run instead of double-dispatching. It is
// written with a RunID ONLY once a command has actually dispatched — the fakrpc
// `done/<nonce> ⇒ skip` discipline (#930), keyed by the command nonce. A refusal
// records the nonce as SEEN but with no RunID: a command that never dispatched is
// re-admittable when capacity frees, so a later delivery re-runs admission rather
// than freezing the first "no" forever. The zero Record means the nonce is unseen.
type Record struct {
	Nonce string // the command nonce this row is keyed on
	RunID string // the dispatched run; "" ⇒ seen-but-not-dispatched (or unseen)
	Lane  string // the lane it dispatched onto, so a re-ack rebuilds an identical line
}

// Dispatched reports whether this nonce already launched a run — the one bit that
// makes a re-delivery a re-ack rather than a fresh dispatch.
func (r Record) Dispatched() bool { return r.RunID != "" }

// Action is what the shell must do for one command delivery.
type Action int

const (
	// Dispatch: admitted, first delivery. The shell launches the detached run and
	// posts the ack card. This is the ONLY action that starts a run.
	Dispatch Action = iota
	// ReAck: admitted, re-delivery of an already-dispatched nonce. The shell posts
	// the IDENTICAL ack again (the outbox nonce coalesces it on the wire) and
	// launches NOTHING. This is the idempotency guard.
	ReAck
	// Refuse: not admitted. The shell posts the structured refusal reason in the
	// command's thread and launches nothing — no seat is taken.
	Refuse
)

// String renders the action token for logs and test failures.
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

// Decision is what the shell must do for one delivery. Ack/Reason are the
// deterministic card text — byte-identical across deliveries of the same nonce,
// so a re-ack and its first ack coalesce on the durable outbox. Record is the
// spool row to persist; on ReAck/Refuse it equals the prior row's meaning, so a
// persist is idempotent.
type Decision struct {
	Action Action
	RunID  string // the run this delivery is bound to (Dispatch/ReAck)
	Ack    string // the ack line (Dispatch/ReAck) — deterministic per nonce
	Reason string // the structured refusal line (Refuse) — carries the closed token
	Record Record // the spool row the shell should persist
}

// Invariant: chatops detach decision logic is fail-closed and deterministic.
// Guard: prior dispatched records strictly take precedence over fresh admission to prevent double-dispatch.
//
// Decide routes one command delivery. It is a pure fold — (command, admission
// verdict, prior spool row) in, decision out — with no I/O, so a replay of the
// same inputs yields the same decision.
//
// Precedence, in order:
//
//  1. Already dispatched (prior.Dispatched()): ReAck the same run. This wins over
//     the fresh admission verdict — once a nonce owns a run, a re-delivery must
//     never start a second one, regardless of current capacity or a stale
//     re-admit. This is the double-delivery guard.
//  2. Admitted: Dispatch the minted run and record it.
//  3. Otherwise: Refuse with the structured reason, recording the nonce as seen
//     but not dispatched (re-admittable later).
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
		Record: Record{Nonce: cmd.Nonce}, // seen, not dispatched — re-admittable
	}
}

// ackText is the deterministic ack line: identical (command, run, lane) ⇒
// identical bytes, so a re-ack coalesces with its first ack on the durable
// outbox. It is a notification-grade plain line; the Block Kit card is the
// shell's job.
func ackText(cmd Command, runID, lane string) string {
	line := fmt.Sprintf(":hourglass_flowing_sand: %s %s — admitted as run `%s`", verbLabel(cmd.Verb), operand(cmd), runID)
	if lane != "" {
		line += " on lane " + lane
	}
	return line
}

// refuseText is the structured refusal line: it names the closed refusal token
// verbatim (so `dos man wedge <TOKEN> --explain` can bind it) and states plainly that nothing
// was started and no seat was taken — a refusal is a first-class, auditable
// answer, not a dropped command.
func refuseText(cmd Command, reason string) string {
	if reason == "" {
		reason = "REFUSE_UNSPECIFIED"
	}
	return fmt.Sprintf(":no_entry: %s %s refused: %s — no run started, no seat taken", verbLabel(cmd.Verb), operand(cmd), reason)
}

// verbLabel renders a verb for a human line, falling back to a neutral word for
// an empty verb so the ack/refusal never reads as a dangling fragment.
func verbLabel(v Verb) string {
	if v == "" {
		return "command"
	}
	return string(v)
}

// operand renders the command's target for a human line, falling back to a
// neutral phrase when the door supplied none.
func operand(cmd Command) string {
	if cmd.Target == "" {
		return "(no target)"
	}
	return cmd.Target
}

// Stall is the liveness state of a detached run, read OUT-OF-BAND from witnessed
// run state (the internal/dispatchtick witness sweep / the loop-event ledger) —
// never the worker's self-report. SilentFor is how long the run has produced no
// witnessed progress; Budget is the silence it is allowed before it counts as
// stalled.
type Stall struct {
	RunID     string        // the stalled run
	Issue     string        // the issue it was dispatched for (for the blockers line)
	SilentFor time.Duration // witnessed silence so far
	Budget    time.Duration // allowed silence before it is a stall
}

// PageMultiple is how many budgets of silence promote a stall from a background
// heads-up (blockerpost "status") to an operator page (blockerpost "operator").
const PageMultiple = 2

// Blockers severities this kernel routes to — kept byte-identical to the tokens
// internal/blockerpost recognizes so the escalation the shell posts is the one
// the blockers surface renders.
const (
	SeverityStatus   = "status"   // background note, no page
	SeverityOperator = "operator" // pages <!here> / the owner
)

// Escalation routes a stalled run to the blockers surface. Escalate is false for
// a run still within budget; Severity is a blockerpost severity; Text is the
// blockers line (already carrying the page prefix for an operator severity).
type Escalation struct {
	Escalate bool
	Severity string
	Text     string
}

// JudgeStall folds a run's liveness into a blockers-escalation decision. A run
// within its silence budget does not escalate. Past budget it escalates as a
// background "status" note; at or past PageMultiple budgets it escalates as an
// "operator" page. A non-positive Budget cannot stall (no budget declared ⇒ the
// judge abstains rather than paging on every run). Pure: state in, decision out.
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

// stallText is the blockers line for a stalled run: it states the witnessed
// silence and the budget it breached, and for an operator page it leads with the
// page prefix the blockers surface expands.
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
