package agent

import "strings"

// loop_observe.go — the typed loop-progress observer (#5148, part of the harness-native
// owned-turn program #2388/#2387). It is the OUTBOUND half of loop ownership: a seam the
// owned loop emits typed lifecycle transitions through, which the native stream turns into
// structured SSE (internal/gateway/native_serve.go). On the proxy path the external
// harness owns the loop and its state is invisible to fak; here fak owns the loop, so it
// can publish exactly what it is doing — turn boundaries, per-call kernel verdicts, tool
// dispatch, result admission — as witnessed events, not narration.
//
// The observer is a pure additive RunOption in the family of WithSessionGate /
// WithSpeculator: unset, every emit is a no-op and the historical loop is byte-for-byte
// unchanged (resolveRunConfig's zero value carries a nil observer).

// ProgressEventKind is a typed loop-lifecycle event class. The kinds are a closed set,
// mirroring the native SSE event names a client gates on, so an observer can switch over
// them exhaustively.
type ProgressEventKind string

const (
	// ProgressTurnStarted marks the start of one model round-trip (before the planner is
	// called). Turn is the 1-based turn index, matching the per-call trace's Turn.
	ProgressTurnStarted ProgressEventKind = "turn_started"
	// ProgressToolStarted marks a tool call being dispatched through the kernel syscall
	// boundary, before its verdict is known. Carries Tool + CallID.
	ProgressToolStarted ProgressEventKind = "tool_started"
	// ProgressCallAdjudicated carries the kernel's verdict for a tool call (Verdict, and
	// Reason on a deny — the closed refusal token). This is the event a client gates on.
	ProgressCallAdjudicated ProgressEventKind = "call_adjudicated"
	// ProgressResultAdmitted marks a tool result entering the transcript, tagged with its
	// Taint disposition (clean | quarantined | tainted).
	ProgressResultAdmitted ProgressEventKind = "result_admitted"
	// ProgressTurnDone marks a turn completing normally (a final answer, or after this
	// turn's tool calls were all admitted). Abnormal stops (terminate, gate, error) carry
	// their reason on the terminal ArmMetrics witness instead.
	ProgressTurnDone ProgressEventKind = "turn_done"
)

// ProgressEvent is one typed loop-lifecycle transition. It carries only witnessed facts
// about a real transition, never narration. Fields are populated per Kind:
//   - turn_started / turn_done: Turn.
//   - tool_started: Turn, CallID, Tool.
//   - call_adjudicated: Turn, CallID, Tool, Verdict, Reason.
//   - result_admitted: Turn, CallID, Tool, Taint.
type ProgressEvent struct {
	Seq     uint64            `json:"seq"`
	Kind    ProgressEventKind `json:"kind"`
	Turn    int               `json:"turn"`
	CallID  string            `json:"call_id,omitempty"`
	Tool    string            `json:"tool,omitempty"`
	Verdict string            `json:"verdict,omitempty"` // ALLOW/DENY/TRANSFORM/QUARANTINE/... (call_adjudicated)
	Reason  string            `json:"reason,omitempty"`  // closed refusal token on a deny (call_adjudicated)
	Taint   string            `json:"taint,omitempty"`   // clean | quarantined | tainted (result_admitted)
}

// ProgressObserver receives typed loop-lifecycle events as the owned loop runs. It is
// called SYNCHRONOUSLY on the loop's own goroutine at each transition, so it must not
// block (a slow observer stalls the turn); a streaming caller does a bounded, non-blocking
// SSE write. A nil observer is the historical loop: no event is emitted.
type ProgressObserver func(ProgressEvent)

// WithProgressObserver wires a typed loop-progress observer into the owned loop. Unset,
// every emit is a no-op, so the loop is byte-for-byte the historical loop.
func WithProgressObserver(obs ProgressObserver) RunOption {
	return func(c *runConfig) { c.observer = obs }
}

// emitProgress delivers one typed lifecycle event to the wired observer. Nil-safe: with no
// observer the call is a no-op, so the historical loop pays nothing.
func (c *runConfig) emitProgress(ev ProgressEvent) {
	if c.observer == nil {
		return
	}
	c.progressSeq++
	ev.Seq = c.progressSeq
	c.observer(ev)
}

// admittedTaint classifies how a tool result entered the transcript, for the
// result_admitted event: a kernel-quarantined poisoned result, an injection-bearing
// result that still reached context (the baseline/naive disposition the loop measures),
// or a clean result. Derived from the same signals the loop already reads — the kernel's
// QUARANTINED annotation on the trace event and the injection substring the loop scans for
// InjectionInContext — so it introduces no new judgment, only surfaces the existing one.
func admittedTaint(ev traceEvent, content string) string {
	if strings.Contains(ev.Note, "QUARANTINED") {
		return "quarantined"
	}
	if strings.Contains(strings.ToLower(content), "ignore previous instructions") {
		return "tainted"
	}
	return "clean"
}
