package session

// ctlrefuse.go — the structured, closed-vocabulary refusal for the drive-state
// CONTROL ops (#2766, epic #2753). Every Table control verb (Transition,
// SetBudget, SetPace, SetPriority, CompareAndSet, ...) already REFUSES an
// illegal-for-state write — a terminal session rejects every change, a stale
// --if-rev loses its CAS race — but the refusal surfaces as a bare ok=false,
// which the route can only render as an unexplained 409. That is the gap #2766
// names: refusals exist but are not a uniform closed set across ops.
//
// This file gives that refusal a closed reason without touching any verb:
// ControlRefusalFor maps the (State, ok) pair every verb already returns onto
// exactly one of two closed tokens, so "cannot cancel a Stopped session" (and
// every other illegal-for-state control write) is a look-up-able reason, never
// free text or a silent boolean. The tokens are registered in dos.toml
// [reasons.*] (the repo's closed refusal vocabulary) and are deliberately
// DISJOINT from the stop-reason vocabulary (decide.go: PAUSED, DRAINING,
// BUDGET_*) and from the per-tool abi.ReasonCode set — a control-op refusal is
// "why your WRITE was rejected", never "why the session stopped" nor "why a
// tool call was refused" (the #2633 category discipline).
//
// The witness-of-applied half of #2766 lives loop-side: the table-driven test
// internal/agent/loop_control_witness_test.go proves each op's enqueued-vs-
// applied distinction and drives these refused paths per op. The adoption seam
// for serving these tokens over HTTP is the route owner's applySessionControl
// (cmd/fak), which binds at the vocabulary spine (#2754).

import "fmt"

// Control-op refusal tokens — the closed vocabulary for an ILLEGAL-FOR-STATE
// drive-state control write. Two tokens cover every refusal the Table's write
// verbs can produce today; ControlRefusalTokens enumerates them for
// completeness tests and vocabulary sync checks.
const (
	// ReasonControlSessionTerminal refuses any control write against a terminal
	// (Stopped) session: you cannot cancel, pause, resume, throttle, re-budget,
	// re-pace, or re-prioritize a stopped session — you start a new one
	// (Recontinue), you do not un-stop one.
	ReasonControlSessionTerminal = "CONTROL_SESSION_TERMINAL"
	// ReasonControlRevStale refuses an optimistic-concurrency (--if-rev) control
	// write whose expected revision no longer matches: a newer transition landed
	// between the caller's read and its write. Re-read and retry.
	ReasonControlRevStale = "CONTROL_REV_STALE"
)

// ControlRefusalTokens returns the closed control-refusal vocabulary — the
// value space a completeness test or a dos.toml sync check enumerates.
func ControlRefusalTokens() []string {
	return []string{ReasonControlSessionTerminal, ReasonControlRevStale}
}

// ControlRefusal is the structured refusal of one drive-state control op. It
// implements error so plumbing can thread it, but callers should switch on
// Reason (a closed token), never parse Detail.
type ControlRefusal struct {
	// Op names the refused control verb ("cancel", "pause", "budget", ...) —
	// observability only; the closed decision surface is Reason.
	Op string `json:"op"`
	// Reason is the closed refusal token (ControlRefusalTokens).
	Reason string `json:"reason"`
	// Detail is human-facing context (the live run-state, the stale rev).
	Detail string `json:"detail,omitempty"`
}

func (r *ControlRefusal) Error() string {
	if r.Detail == "" {
		return r.Reason
	}
	return r.Reason + ": " + r.Detail
}

// ControlRefusalFor maps the (State, ok) pair every Table control verb returns
// onto the structured refusal: nil for an applied write (ok=true), the closed
// terminal-session token when the refusing record is terminal, and the closed
// stale-revision token otherwise (the only other refusal the write verbs
// produce: a CompareAndSet that lost its race against a live session). The
// mapping is total over the verbs' actual refusal behavior, so a caller can
// wrap any existing verb without that verb changing shape.
func ControlRefusalFor(op string, st State, ok bool) *ControlRefusal {
	if ok {
		return nil
	}
	if st.Run.terminal() {
		return &ControlRefusal{
			Op:     op,
			Reason: ReasonControlSessionTerminal,
			Detail: fmt.Sprintf("session %s is %s (terminal); a stopped session cannot be %s — start a new session (Recontinue) instead", st.TraceID, st.Run, op),
		}
	}
	return &ControlRefusal{
		Op:     op,
		Reason: ReasonControlRevStale,
		Detail: fmt.Sprintf("session %s moved to rev %d since the caller's read; re-read and retry the %s", st.TraceID, st.Rev, op),
	}
}
