// control_vocab.go — the CLOSED decision vocabulary at the tool/turn seam, and
// the barrier that keeps its two halves from being confused for each other.
//
// The managed-context runtime makes two structurally different decisions, and
// this file names both from closed sets so neither degrades into an ad hoc
// parser failure, a generic exit, or an accidental session stop (#2633):
//
//  1. A PER-TOOL rejection outcome — "why was THIS tool call refused?". Its
//     authority is the closed refusal vocabulary in internal/abi (abi.ReasonCode:
//     MALFORMED, POLICY_BLOCK, UNKNOWN_TOOL, ...). It is scoped to one call.
//
//  2. A TURN/SESSION control decision — "what does the loop do to the turn or
//     session as a whole?" (keep going, end the turn, pause, stop). That is the
//     ControlDecision vocabulary declared here.
//
// The failure this closes is a category error at the seam BETWEEN them. A run
// of malformed tool calls is a per-tool fact; the loop's *response* to that run
// ("stop the session") is a control decision. If the two share a namespace, a
// per-tool reason like "MALFORMED" can leak upward and be recorded as the
// reason the SESSION stopped — but the session was never asked to stop because
// of "MALFORMED"; it stopped because repeated rejection ESCALATED to a declared
// STOP_SESSION. This file makes the two vocabularies disjoint typed sets, routes
// the only sanctioned crossing through EscalateRepeatedRejection (which yields a
// declared ControlDecision, never the raw reason re-labelled), and gives Outcome
// two separate typed fields so serialization can never fold one into the other.
//
// This is the policy-facing contract other leaves consume; it deliberately does
// NOT import internal/session (the loop that acts on these decisions) — the
// vocabulary is defined here so both the tool floor and the session loop can
// name the same closed outcomes without a dependency cycle.

package policy

import (
	"encoding/json"
	"strconv"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// ControlDecision is the CLOSED vocabulary of turn/session control outcomes the
// managed loop may take. It is a DISTINCT named type from a per-tool rejection
// (abi.ReasonCode): it answers "what happens to the turn/session", never "why
// one call was refused". Values are ordered by escalation severity so a numeric
// max() over a turn's decisions yields the strongest action taken.
type ControlDecision uint8

const (
	// ControlContinue keeps the turn going: the rejection (if any) was handled
	// in-band and the model may retry. It is the non-escalating floor.
	ControlContinue ControlDecision = iota
	// ControlEndTurn ends THIS turn and hands control back to the model or
	// operator; the session stays live and resumable next turn.
	ControlEndTurn
	// ControlPauseSession parks the session under an operator-style hold. Not
	// terminal — the loop waits and may resume (mirrors the session Paused state).
	ControlPauseSession
	// ControlStopSession is a terminal stop of the session. This is the declared
	// outcome a repeated-rejection escalation reaches; it is NEVER a raw parser
	// error or a per-tool reason wearing a stop label.
	ControlStopSession
)

// controlNames is the closed control token space. These tokens are deliberately
// disjoint from every abi.ReasonCode name (the per-tool refusal vocabulary) so a
// control token can never resolve as a tool reason, nor a tool reason as a
// control token. TestControlAndToolVocabulariesAreDisjoint proves it.
var controlNames = map[ControlDecision]string{
	ControlContinue:     "CONTINUE",
	ControlEndTurn:      "END_TURN",
	ControlPauseSession: "PAUSE_SESSION",
	ControlStopSession:  "STOP_SESSION",
}

// String renders the declared control token, or "CONTROL(n)" for an
// out-of-range value (an unnamed decision is a bug, surfaced not swallowed).
func (d ControlDecision) String() string {
	if n, ok := controlNames[d]; ok {
		return n
	}
	return "CONTROL(" + strconv.Itoa(int(d)) + ")"
}

// IsValid reports whether d is one of the declared control decisions.
func (d ControlDecision) IsValid() bool {
	_, ok := controlNames[d]
	return ok
}

// EndsTurn reports whether the decision terminates the current turn (end-turn,
// pause, or stop all end the turn; continue does not).
func (d ControlDecision) EndsTurn() bool { return d >= ControlEndTurn }

// StopsSession reports whether the decision terminally stops the session. A
// pause is NOT a stop (it is resumable), so only ControlStopSession qualifies.
func (d ControlDecision) StopsSession() bool { return d == ControlStopSession }

// ParseControlDecision resolves a control token to its ControlDecision. It
// accepts ONLY the closed control vocabulary: feeding it any abi refusal-reason
// name (e.g. "MALFORMED") returns ok=false. This one-way gate is what stops a
// tool rejection from being parsed back in as a session-control decision.
func ParseControlDecision(token string) (ControlDecision, bool) {
	for d, n := range controlNames {
		if n == token {
			return d, true
		}
	}
	return ControlContinue, false
}

// ControlDecisions returns the closed control vocabulary in severity order —
// the value space a consumer may switch over or lint against.
func ControlDecisions() []ControlDecision {
	return []ControlDecision{ControlContinue, ControlEndTurn, ControlPauseSession, ControlStopSession}
}

// EscalationLadder declares, as policy, how a RUN of consecutive identical
// per-tool rejections escalates into a turn/session control decision. It is the
// only sanctioned bridge from the per-tool vocabulary to the control vocabulary:
// the thresholds are declared data, and the result is a typed ControlDecision,
// so "the model keeps emitting a bad call" becomes a DECLARED outcome rather
// than an ad hoc exit. Both thresholds count CONSECUTIVE identical rejections.
type EscalationLadder struct {
	// EndTurnAfter is the consecutive-rejection count at which the loop ends the
	// turn (giving the model a fresh turn to recover). <=0 uses the default.
	EndTurnAfter int
	// StopAfter is the consecutive-rejection count at which the loop stops the
	// session outright. <=0 uses the default. Must exceed EndTurnAfter after
	// defaults resolve, or Normalize lifts it so stop is never easier than
	// end-turn.
	StopAfter int
}

// Default escalation thresholds: one bad call is retryable in-band, a short run
// ends the turn, a longer run stops the session. Conservative on purpose — the
// point is that escalation is DECLARED and bounded, not that it fires early.
const (
	defaultEndTurnAfter = 3
	defaultStopAfter    = 6
)

// Normalize fills unset thresholds from the defaults and guarantees
// StopAfter > EndTurnAfter, so a stop can never be reached before an end-turn.
func (l EscalationLadder) Normalize() EscalationLadder {
	if l.EndTurnAfter <= 0 {
		l.EndTurnAfter = defaultEndTurnAfter
	}
	if l.StopAfter <= 0 {
		l.StopAfter = defaultStopAfter
	}
	if l.StopAfter <= l.EndTurnAfter {
		l.StopAfter = l.EndTurnAfter + 1
	}
	return l
}

// EscalateRepeatedRejection maps a run of `consecutive` identical per-tool
// rejections onto a DECLARED control decision. reason is the per-tool refusal
// that keeps recurring (e.g. abi.ReasonMalformed); it is used only to gate that
// there IS a rejection to escalate (ReasonNone => nothing to escalate =>
// ControlContinue). The returned value is a ControlDecision — never the reason
// string re-labelled as a stop cause. This is the seam #2633 names: repeated
// bad calls become a declared outcome, not a raw JSON/parser error.
func EscalateRepeatedRejection(reason abi.ReasonCode, consecutive int, ladder EscalationLadder) ControlDecision {
	if reason == abi.ReasonNone || consecutive <= 0 {
		return ControlContinue
	}
	l := ladder.Normalize()
	switch {
	case consecutive >= l.StopAfter:
		return ControlStopSession
	case consecutive >= l.EndTurnAfter:
		return ControlEndTurn
	default:
		return ControlContinue
	}
}

// Outcome is the policy-facing record of a decision at the tool/turn seam. It
// carries the per-tool rejection and the control decision as SEPARATE typed
// fields with SEPARATE serialized keys, so a marshal can never fold a tool
// rejection into the session-stop slot: "why the call was refused" (Reject) and
// "what the loop did about it" (Control) are distinct columns, always. Reject is
// abi.ReasonNone when the call was allowed; Control is ControlContinue when the
// turn simply proceeds.
type Outcome struct {
	Reject  abi.ReasonCode  // per-tool refusal; ReasonNone if the call was allowed
	Control ControlDecision // turn/session control action taken in response
}

// outcomeWire is the serialized shape of an Outcome. The two vocabularies land
// on two named keys — tool_reject (a refusal-reason name) and control (a control
// token) — which is the structural guarantee that one can never be read out of
// the other's slot.
type outcomeWire struct {
	ToolReject string `json:"tool_reject,omitempty"`
	Control    string `json:"control"`
}

// MarshalJSON emits the two vocabularies on two distinct keys. A ReasonNone
// reject is omitted (the call was allowed); the control token is always present.
func (o Outcome) MarshalJSON() ([]byte, error) {
	w := outcomeWire{Control: o.Control.String()}
	if o.Reject != abi.ReasonNone {
		w.ToolReject = abi.ReasonName(o.Reject)
	}
	return json.Marshal(w)
}

// SessionStopReason returns the declared control token to record as the reason
// the SESSION stopped, and true iff this outcome actually stops the session. It
// reads the Control field ONLY: the per-tool Reject reason is structurally
// excluded from the session-stop slot. This is the fold #2633 guards — an
// Outcome carrying a MALFORMED reject stops the session as "STOP_SESSION", never
// as "MALFORMED".
func (o Outcome) SessionStopReason() (token string, stopped bool) {
	if o.Control.StopsSession() {
		return o.Control.String(), true
	}
	return "", false
}
