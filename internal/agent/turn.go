package agent

// turn.go — the suspend-and-resume TURN PRIMITIVE (#1318), the one net-new mechanism
// the native-harness program names as missing. The session gate (loop_session.go's
// gateTurn) TERMINATES the arm on a non-proceed verdict; this primitive instead lets the
// loop SUSPEND at a tool-call boundary — run a speculated, provably effect-free call
// ahead of the model and HOLD its provisional effect in a SEAM-4 BufferSink — then
// RESUME when the model's authoritative next call is known, PROMOTING the effect on a
// match or SQUASHING it on a miss, all WITHIN THE SAME turn index (the model-turn
// counter does not advance across the suspend). It is the missing suspension point the
// before-consumption write barrier (#1319) hangs on.
//
// It is built entirely on the already-shipped SEAM-4 driver (internal/abi/speculate.go,
// committed d859ec21 #812): Speculator.Predict issues the effect-free speculative call,
// BufferSink stages/promotes/retracts the provisional effect, and abi.Resolve drives the
// match->Commit / miss->Squash fork. This file is the TURN-GRANULAR wrapper + the live
// RunArm driver that gives Speculator.Predict its first non-test caller.

import (
	"context"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/kernel"
)

// Turn is one model turn that may carry a SUSPENDED speculation. Suspend stages a
// predicted call's provisional effect without advancing the turn index; Resume resolves
// it against the model's authoritative next call. The zero value is an unsuspended turn
// at index 0.
type Turn struct {
	index     int                   // the model-turn index this speculation rides within
	predicted *abi.ToolCall         // the speculated call held provisional (nil until Suspend)
	sink      *abi.BufferSink       // the store-buffer holding the provisional effect
	sinks     []abi.ProvisionalSink // the sink set abi.Resolve drives (just sink today)
	txn       abi.TxnID             // transaction scope (0 = auto, the BufferSink keys on epoch)
	suspended bool
}

// NewTurn builds a turn at the given model-turn index with a fresh provisional-effect
// BufferSink. The index is fixed for the turn's lifetime — Suspend/Resume never change
// it, which is the "stays within one turn index" invariant.
func NewTurn(index int) *Turn {
	sink := abi.NewBufferSink()
	return &Turn{index: index, sink: sink, sinks: []abi.ProvisionalSink{sink}}
}

// Suspend records a speculation at this turn's tool-call boundary: it stages the
// predicted call's provisional result in the BufferSink under the call's speculative
// epoch and marks the turn suspended. It does NOT advance the turn index — the whole
// point of suspend-vs-terminate. Returns the ProvisionalSink holding the effect, so a
// caller can witness the held (not-yet-committed) effect. A nil predicted call (or one
// carrying no epoch) is a no-op that leaves the turn unsuspended.
func (t *Turn) Suspend(predicted *abi.ToolCall, result abi.Ref) abi.ProvisionalSink {
	if predicted == nil || !predicted.Spec.Speculative {
		return t.sink
	}
	t.predicted = predicted
	t.sink.Stage(predicted.Spec.Epoch, result)
	t.suspended = true
	return t.sink
}

// Resume resolves a suspended speculation against the model's AUTHORITATIVE next call:
// a match Promotes the provisional effect (OutcomeCommitted), a miss Rolls it back
// (OutcomeSquashed) — the executable form of "squash actually undoes the effect". It
// returns OutcomeCommitted with no work when the turn was never suspended. After Resume
// the turn is no longer suspended; the BufferSink holds the committed effect on a match
// and nothing on a miss.
func (t *Turn) Resume(ctx context.Context, authoritative *abi.ToolCall) (abi.Outcome, error) {
	if !t.suspended {
		return abi.OutcomeCommitted, nil
	}
	t.suspended = false
	return abi.Resolve(ctx, t.sinks, t.txn, t.predicted, authoritative)
}

// Index is the model-turn index this turn rides within — unchanged across a
// suspend/resume, which is how the primitive proves a speculation stays inside one turn.
func (t *Turn) Index() int { return t.index }

// Suspended reports whether a speculation is currently held awaiting its authoritative
// resolution.
func (t *Turn) Suspended() bool { return t.suspended }

// Sink is the provisional-effect store-buffer, the forensic witness that a commit landed
// (Committed non-empty) or a squash left nothing (Committed empty, PendingEpochs 0).
func (t *Turn) Sink() *abi.BufferSink { return t.sink }

// ---------------------------------------------------------------------------
// specState — the live RunArm speculation driver (the non-test Predict caller).
// ---------------------------------------------------------------------------

// specState carries the in-flight speculation across RunArm's turns. It is created only
// when a speculator is wired (WithSpeculator); with no speculator RunArm never builds one
// and the loop is byte-for-byte the historical path. It holds the pending suspended Turn
// — a speculation issued after turn T, awaiting turn T+1's authoritative call.
type specState struct {
	spec    *abi.Speculator
	k       *kernel.Kernel
	pending *Turn
	// barred is set when the last resolved speculation SQUASHED: the immediately
	// following write-shaped call is a dependent write behind an unconfirmed speculative
	// read, so the before-consumption write barrier (#1319) blocks it from the engine.
	barred bool
}

// newSpecState builds the driver for a fak-arm run with a wired speculator. k is the
// loop's kernel (the speculated effect-free call dispatches through the same syscall
// boundary every real call does).
func newSpecState(spec *abi.Speculator, k *kernel.Kernel) *specState {
	return &specState{spec: spec, k: k}
}

// resolve drives a pending speculation to its outcome against the model's authoritative
// next call, recording the commit/squash on the arm metrics. It is the RESUME edge: a
// correct prediction promotes (the loop reused work the model would have spent a call
// on), a wrong one squashes (no trace left). A nil driver or no pending speculation is a
// no-op. The model-turn counter is untouched — resolution happens within the turn.
func (s *specState) resolve(ctx context.Context, authoritative *abi.ToolCall, m *ArmMetrics) {
	if s == nil || s.pending == nil {
		return
	}
	outcome, _ := s.pending.Resume(ctx, authoritative)
	switch outcome {
	case abi.OutcomeCommitted:
		m.SpecCommitted++
		s.barred = false // the speculative read was confirmed; a dependent write may commit
	case abi.OutcomeSquashed:
		m.SpecSquashed++
		s.barred = true // mispredict: the dependent follow-on write must NOT reach the engine
	}
	s.pending = nil
}

// barWrite reports whether a write-shaped call must be BARRED from dispatch — the
// before-consumption write barrier (#1319). A write that follows a SQUASHED speculation
// (a write behind an unconfirmed speculative read) is not committed: it never reaches the
// engine, so a mispredicted read can never leak a dependent durable effect. The barrier
// fires once (it clears after barring), and only for a write-shaped tool — a follow-on
// READ after a squash is harmless and proceeds. A nil driver or an unarmed barrier
// returns false (the call dispatches normally), so the historical loop is unchanged.
func (s *specState) barWrite(tool string, m *ArmMetrics) bool {
	if s == nil || !s.barred {
		return false
	}
	if !abi.IsWriteShaped(tool) {
		return false
	}
	s.barred = false
	m.WritesBarred++
	return true
}

// disarm clears any armed write barrier at a turn boundary so it never leaks into a later
// turn: the barrier gates only the writes in the SAME turn the squash resolved in. A
// follow-on read after a squash leaves the barrier armed (a read is harmless); disarm
// retires it once the turn's calls are processed. Nil-safe.
func (s *specState) disarm() {
	if s != nil {
		s.barred = false
	}
}

// speculate issues a speculation for the NEXT call after turn `turnIdx`'s tool calls ran:
// it asks the Speculator to Predict from the context signature + this turn's prior
// results, and — if a provably effect-free call is predicted — runs it through the kernel
// and SUSPENDS it into a fresh Turn (the provisional result staged in the BufferSink),
// to be resolved at the next turn boundary. This is the live, non-test caller of
// Speculator.Predict. A nil driver, a disabled speculator, or a no-prediction is a no-op
// (no pending turn), so the loop continues exactly as before.
func (s *specState) speculate(ctx context.Context, turnIdx int, sig string, prior []*abi.Result, m *ArmMetrics) {
	if s == nil || s.spec == nil {
		return
	}
	predicted := s.spec.Predict(sig, prior, 0)
	if predicted == nil {
		return // nothing to speculate (default-deny on effects, no matching pattern, …)
	}
	m.SpecIssued++
	// Before-consumption serve (#1319): a SPECULATIVE effect-free read is served from the
	// PREDICTION — its symbolically-derived result — WITHOUT engine dispatch. We do NOT
	// k.Syscall it (that would dispatch and bump EngineCalls); the derived args ARE the
	// provisional result, staged under the speculative epoch until the next authoritative
	// call promotes or squashes it. A live vDSO hit would serve it identically; the
	// prediction is the dispatch-free fallback. This is what makes the read provisional
	// and gives the write barrier something to gate on.
	m.SpecServed++
	turn := NewTurn(turnIdx)
	turn.Suspend(predicted, predicted.Args)
	s.pending = turn
}

// authoritativeCall builds the abi.ToolCall the model AUTHORITATIVELY emitted from a
// chat-layer tool call, so a suspended speculation can be matched against it (same tool
// + byte-identical args is a hit). The args ride inline — the speculation match compares
// inline bytes or content digests, and an inline ref needs no resolver.
func authoritativeCall(tc ToolCall) *abi.ToolCall {
	args := []byte(tc.Function.Arguments)
	return &abi.ToolCall{
		Tool: tc.Function.Name,
		Args: abi.Ref{Kind: abi.RefInline, Inline: args, Len: int64(len(args))},
	}
}
