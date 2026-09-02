package sessionctl

// park.go — the OUT-OF-BAND operator approve/deny INBOX of the operator control
// epic (#2757, epic #2753), the gate-resolution twin of the add-constraint op
// (constraint.go, #2756): where add-constraint narrows what the session MAY DO,
// the park op decides what a GATED action actually DOES — a call the
// reversibility/ESCALATE gate refused parks here as an addressable pending
// action, and an EXTERNAL operator resolves it with a typed approve/deny (with
// optional argument-modify) out of band. The loop parks on the gate until the
// verdict arrives or the park times out, then proceeds/aborts per the verdict.
//
// The property this op fixes: the shipped reversibility gate (internal/
// adjudicator/reversibility.go) is satisfied by the AGENT echoing its
// deterministic confirm token — a self-confirm. This inbox is the operator
// channel that gate lacks: the pending-action queue an operator lists, and the
// approve/deny ops that resolve one SPECIFIC parked action. The adjudicator
// itself is deliberately untouched (core-locked): an approve is honored by the
// loop re-proposing the call through the NORMAL syscall boundary (byte-identical
// + the gate's own confirm echo, or the operator's modified args freshly
// adjudicated), so the gate still sees and journals every dispatch.
//
// Like constraint.go this package owns ONLY the pending queue, the verdict
// payload + validation, the park/resolve rendezvous, and the Next audit
// witnesses. The loop-side consumer is internal/agent/loop_park.go: it parks an
// ESCALATE-gated deny here and honors the verdict at the SAME dispatch site.
// Timeout handling is explicit by design (#2757 confusion risk): an unresolved
// park is never silently dropped — it aborts with its closed reason, witnessed.
// The per-op control ENVELOPE binds at the vocabulary spine (#2754): the OpSpec
// row for this op (its PARK_* tokens, boundary, witness shape) registers
// together with its loop-side #2766 witness row, per the reservation in
// vocab.go — registering either half alone would let the two authoritative
// tables drift.

import (
	"context"
	"encoding/json"
	"fmt"
	"sort"
	"strings"
	"sync"
	"time"
)

// ParkVerdictKind is the closed set of operator resolutions for a parked action.
// Argument-modify is not a third kind: it is an approve carrying modified args.
type ParkVerdictKind string

const (
	// ParkApprove proceeds the parked action: the loop re-proposes the call
	// through the normal syscall boundary (byte-identical + the gate's confirm
	// echo, or the operator's modified args freshly adjudicated).
	ParkApprove ParkVerdictKind = "approve"
	// ParkDeny aborts the parked action: the loop returns a typed receipt with
	// the closed PARK_OPERATOR_DENIED reason and never dispatches the call.
	ParkDeny ParkVerdictKind = "deny"
)

// GatedAction is the parked call as the gate refused it — the payload an
// operator reads to decide. JSON tags are the wire shape the (spine-bound,
// #2754) route will carry.
type GatedAction struct {
	// Tool is the tool name of the gated call.
	Tool string `json:"tool"`
	// Args is the call's raw argument JSON, exactly as proposed.
	Args string `json:"args,omitempty"`
	// Reason is the closed refusal token the gate refused with.
	Reason string `json:"reason,omitempty"`
	// Preview is the gate's refusal preview/receipt — what the operator reads
	// to judge the action (never the tool's output; the call was not dispatched).
	Preview string `json:"preview,omitempty"`
}

// PendingAction is one parked gated action on the addressable queue, keyed by
// its ID for the approve/deny ops.
type PendingAction struct {
	// ID addresses this parked action for ResolveGatedAction.
	ID string `json:"id"`
	// Trace is the session the action is parked on.
	Trace string `json:"trace"`
	// Action is the gated call awaiting the operator verdict.
	Action GatedAction `json:"action"`
	// ParkedAt is when the loop parked (UTC).
	ParkedAt time.Time `json:"parked_at"`
}

// ParkVerdict is the typed operator resolution of one parked action.
type ParkVerdict struct {
	// Kind is the closed approve/deny resolution.
	Kind ParkVerdictKind `json:"kind"`
	// Args, when non-empty on an approve, is the operator's MODIFIED argument
	// JSON (a JSON object): the loop dispatches these args instead of the
	// original, freshly adjudicated. Illegal on a deny.
	Args string `json:"args,omitempty"`
	// Note is the operator's stated reason, carried for the audit trail only.
	Note string `json:"note,omitempty"`
}

// ParkRefuseReason is the closed refusal vocabulary for the park op — the
// PARK_* tokens reserved for the #2766 table. Two categories share the type:
// op refusals (a malformed park/resolve, an unknown action id) and parked-action
// outcomes (the explicit unresolved aborts, and the reason a denied action's
// receipt carries).
type ParkRefuseReason string

const (
	// ParkMalformed refuses a park or resolve whose shape is illegal — empty
	// trace/tool/id, an unknown verdict kind, argument-modify on a deny, or
	// modified args that are not a JSON object.
	ParkMalformed ParkRefuseReason = "PARK_MALFORMED"
	// ParkUnknownAction refuses an approve/deny addressing an action that is not
	// pending — never parked, already resolved, or already timed out.
	ParkUnknownAction ParkRefuseReason = "PARK_UNKNOWN_ACTION"
	// ParkTimeout aborts a parked action no operator resolved within the park
	// window. Explicit by design: an unresolved park is witnessed and aborted,
	// never silently dropped.
	ParkTimeout ParkRefuseReason = "PARK_TIMEOUT"
	// ParkAborted aborts a parked action whose wait ended before any verdict —
	// the park context was cancelled or the session was cleared at teardown.
	ParkAborted ParkRefuseReason = "PARK_ABORTED"
	// ParkOperatorDenied is the reason an operator-denied action's typed receipt
	// carries: the loop aborts the call, never dispatches it.
	ParkOperatorDenied ParkRefuseReason = "PARK_OPERATOR_DENIED"
)

// ParkRefusal is a structured, closed-reason refusal. It implements error so
// plumbing can thread it, but callers switch on Reason, never parse Detail.
type ParkRefusal struct {
	Reason ParkRefuseReason `json:"reason"`
	Detail string           `json:"detail,omitempty"`
}

func (r *ParkRefusal) Error() string { return refusalString(string(r.Reason), r.Detail) }

// parkRefuse builds a ParkRefusal in one line.
func parkRefuse(reason ParkRefuseReason, format string, args ...any) *ParkRefusal {
	return &ParkRefusal{Reason: reason, Detail: fmt.Sprintf(format, args...)}
}

// DefaultParkTimeout bounds a park no EnableGateParking configured. A parked
// loop must always have an explicit outcome; an unbounded wait is never one.
const DefaultParkTimeout = 10 * time.Minute

// parkedEntry is one parked action plus its resolution rendezvous. The verdict
// channel is buffered (one resolve per entry, delivered under the lock after
// the entry leaves the queue) and is closed only by ClearParked.
type parkedEntry struct {
	pending PendingAction
	seq     uint64
	verdict chan ParkVerdict
}

// park state is per-session, keyed by the run's trace id — the same keying as
// the constraint and redirect mailboxes. parkPending holds actions awaiting an
// operator verdict; parkNext holds the loop-side audit witnesses.
var (
	parkMu      sync.Mutex
	parkSeq     uint64
	parkWindow  = map[string]time.Duration{}
	parkPending = map[string]map[string]*parkedEntry{}
	parkNext    = map[string][]NextRecord{}
)

// EnableGateParking opens the operator inbox for trace: the loop parks
// ESCALATE-gated denies for this session instead of returning them straight to
// the model. timeout bounds each park (<= 0 selects DefaultParkTimeout).
// Parking is opt-in per session so a session with no operator listening keeps
// the historical gate behavior byte-for-byte.
func EnableGateParking(trace string, timeout time.Duration) *ParkRefusal {
	trace = strings.TrimSpace(trace)
	if trace == "" {
		return parkRefuse(ParkMalformed, "empty session trace")
	}
	if timeout <= 0 {
		timeout = DefaultParkTimeout
	}
	parkMu.Lock()
	defer parkMu.Unlock()
	parkWindow[trace] = timeout
	return nil
}

// GateParkingEnabled reports whether the operator inbox is open for trace — the
// loop-side predicate deciding park vs. the historical straight-through deny.
func GateParkingEnabled(trace string) bool {
	trace = strings.TrimSpace(trace)
	if trace == "" {
		return false
	}
	parkMu.Lock()
	defer parkMu.Unlock()
	_, ok := parkWindow[trace]
	return ok
}

// ParkGatedAction parks one gated action on the operator inbox and BLOCKS the
// calling loop until an operator verdict arrives, the park window elapses, or
// ctx is cancelled. It returns the verdict (approve or deny — the loop honors
// either), or the closed refusal for the explicit unresolved aborts
// (PARK_TIMEOUT / PARK_ABORTED). Every outcome is witnessed on the trace's park
// Next records at THIS consume point — the parked arm observing the resolution
// is the witness, never the queue write.
func ParkGatedAction(ctx context.Context, trace string, action GatedAction) (ParkVerdict, *ParkRefusal) {
	trace = strings.TrimSpace(trace)
	if trace == "" {
		return ParkVerdict{}, parkRefuse(ParkMalformed, "empty session trace")
	}
	if strings.TrimSpace(action.Tool) == "" {
		return ParkVerdict{}, parkRefuse(ParkMalformed, "a parked action needs a tool name")
	}
	if ctx == nil {
		ctx = context.Background()
	}
	parkMu.Lock()
	window := parkWindow[trace]
	if window <= 0 {
		window = DefaultParkTimeout
	}
	parkSeq++
	entry := &parkedEntry{
		pending: PendingAction{
			ID:       fmt.Sprintf("park-%d", parkSeq),
			Trace:    trace,
			Action:   action,
			ParkedAt: time.Now().UTC(),
		},
		seq:     parkSeq,
		verdict: make(chan ParkVerdict, 1),
	}
	byID := parkPending[trace]
	if byID == nil {
		byID = map[string]*parkedEntry{}
		parkPending[trace] = byID
	}
	byID[entry.pending.ID] = entry
	parkMu.Unlock()

	timer := time.NewTimer(window)
	defer timer.Stop()
	select {
	case v, ok := <-entry.verdict:
		if !ok {
			// ClearParked closed the rendezvous: session teardown. Not witnessed —
			// the trace's journal was cleared with the session; the loop-side
			// receipt is the record.
			return ParkVerdict{}, parkRefuse(ParkAborted, "session cleared while action %s (%s) was parked", entry.pending.ID, action.Tool)
		}
		return auditParkVerdict(trace, entry, v), nil
	case <-timer.C:
		return settleUnresolved(trace, entry, parkRefuse(ParkTimeout,
			"no operator verdict within %s for parked action %s (%s); aborted, never dispatched", window, entry.pending.ID, action.Tool))
	case <-ctx.Done():
		return settleUnresolved(trace, entry, parkRefuse(ParkAborted,
			"park wait cancelled for action %s (%s): %v", entry.pending.ID, action.Tool, ctx.Err()))
	}
}

// settleUnresolved finishes a park whose wait ended without a delivered verdict
// (timeout or context cancel). The deadline RACES the resolve: if an operator
// verdict landed between the trigger and the lock, the verdict wins — a verdict
// an operator sent is never converted into an abort.
func settleUnresolved(trace string, entry *parkedEntry, cause *ParkRefusal) (ParkVerdict, *ParkRefusal) {
	parkMu.Lock()
	byID := parkPending[trace]
	if _, still := byID[entry.pending.ID]; still {
		delete(byID, entry.pending.ID)
		if len(byID) == 0 {
			delete(parkPending, trace)
		}
		parkMu.Unlock()
		appendParkAudit(trace, MoveAnnotate,
			"unresolved: "+entry.pending.Action.Tool+" ("+entry.pending.ID+")",
			ApplyResult{Refusal: cause.Error()})
		return ParkVerdict{}, cause
	}
	parkMu.Unlock()
	// Already off the queue: a resolve (buffered verdict) or a teardown (closed
	// channel) beat the deadline.
	v, ok := <-entry.verdict
	if !ok {
		return ParkVerdict{}, parkRefuse(ParkAborted, "session cleared while action %s (%s) was parked", entry.pending.ID, entry.pending.Action.Tool)
	}
	return auditParkVerdict(trace, entry, v), nil
}

// auditParkVerdict lowers one consumed operator verdict onto the shared Next
// contract: an approve is the applied proceed; a deny is the closed
// PARK_OPERATOR_DENIED outcome the loop's receipt carries.
func auditParkVerdict(trace string, entry *parkedEntry, v ParkVerdict) ParkVerdict {
	payload := string(v.Kind) + ": " + entry.pending.Action.Tool + " (" + entry.pending.ID + ")"
	if v.Kind == ParkApprove && strings.TrimSpace(v.Args) != "" {
		payload += " (args modified)"
	}
	if note := strings.TrimSpace(v.Note); note != "" {
		payload += "; operator note: " + note
	}
	if v.Kind == ParkDeny {
		appendParkAudit(trace, MoveAnnotate, payload,
			ApplyResult{Refusal: parkRefuse(ParkOperatorDenied, "denied out of band by the operator").Error()})
		return v
	}
	appendParkAudit(trace, MoveContinue, payload, ApplyResult{Applied: true})
	return v
}

// appendParkAudit appends one park Next record for trace. Never called with parkMu
// held.
func appendParkAudit(trace string, kind MoveKind, payload string, result ApplyResult) {
	move := Move{
		Kind: kind, Render: RenderSystemDirective,
		Session: SessionAutonomous, Gate: "sessionctl-park",
		Source: "agent-gate-park", Payload: payload,
	}
	record, err := WitnessMove(move, result)
	if err != nil {
		return
	}
	parkMu.Lock()
	parkNext[trace] = append(parkNext[trace], record)
	parkMu.Unlock()
}

// PendingGatedActions returns the addressable pending queue for trace in park
// order — the read op an operator (CLI `fak session pending`, the spine-bound
// route) lists before resolving. The result is a snapshot copy.
func PendingGatedActions(trace string) []PendingAction {
	trace = strings.TrimSpace(trace)
	if trace == "" {
		return nil
	}
	parkMu.Lock()
	defer parkMu.Unlock()
	byID := parkPending[trace]
	if len(byID) == 0 {
		return nil
	}
	entries := make([]*parkedEntry, 0, len(byID))
	for _, e := range byID {
		entries = append(entries, e)
	}
	sort.Slice(entries, func(i, j int) bool { return entries[i].seq < entries[j].seq })
	out := make([]PendingAction, len(entries))
	for i, e := range entries {
		out[i] = e.pending
	}
	return out
}

// ResolveGatedAction is the typed operator approve/deny op: it resolves the
// pending action addressed by id with the verdict, waking the parked loop.
// A malformed verdict refuses at this edge with its closed reason; a verdict
// addressing an action that is not pending (never parked, already resolved,
// already timed out) refuses PARK_UNKNOWN_ACTION — a resolve is never
// double-delivered.
func ResolveGatedAction(trace, id string, v ParkVerdict) *ParkRefusal {
	trace = strings.TrimSpace(trace)
	id = strings.TrimSpace(id)
	if trace == "" || id == "" {
		return parkRefuse(ParkMalformed, "resolve needs a session trace and a pending action id")
	}
	switch v.Kind {
	case ParkApprove, ParkDeny:
	default:
		return parkRefuse(ParkMalformed, "unknown verdict kind %q (closed set: %s|%s)", v.Kind, ParkApprove, ParkDeny)
	}
	if args := strings.TrimSpace(v.Args); args != "" {
		if v.Kind == ParkDeny {
			return parkRefuse(ParkMalformed, "argument-modify rides only an approve; a deny aborts the action as parked")
		}
		var obj map[string]any
		if err := json.Unmarshal([]byte(args), &obj); err != nil {
			return parkRefuse(ParkMalformed, "modified args must be a JSON object: %v", err)
		}
	}
	parkMu.Lock()
	byID := parkPending[trace]
	entry, ok := byID[id]
	if !ok {
		parkMu.Unlock()
		return parkRefuse(ParkUnknownAction, "no parked action %q on this session (never parked, already resolved, or timed out)", id)
	}
	delete(byID, id)
	if len(byID) == 0 {
		delete(parkPending, trace)
	}
	// Buffered send under the lock: the entry just left the queue, so this is
	// the only verdict it can ever receive — the send never blocks.
	entry.verdict <- v
	parkMu.Unlock()
	return nil
}

// ReadParkNextRecords returns and clears the independently re-readable Next
// witnesses emitted as the trace's parked actions were resolved or aborted.
// Empty/no-op sessions produce no records.
func ReadParkNextRecords(trace string) []NextRecord {
	trace = strings.TrimSpace(trace)
	if trace == "" {
		return nil
	}
	parkMu.Lock()
	defer parkMu.Unlock()
	records := append([]NextRecord(nil), parkNext[trace]...)
	delete(parkNext, trace)
	return records
}

// ClearParked drops the inbox for trace — the session teardown hook so a
// finished run leaks no per-trace state. Any still-parked waiter is released
// with the closed PARK_ABORTED reason (its rendezvous is closed, never left
// blocking). Idempotent.
func ClearParked(trace string) {
	trace = strings.TrimSpace(trace)
	if trace == "" {
		return
	}
	parkMu.Lock()
	defer parkMu.Unlock()
	for _, e := range parkPending[trace] {
		close(e.verdict)
	}
	delete(parkPending, trace)
	delete(parkWindow, trace)
	delete(parkNext, trace)
}
