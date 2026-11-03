// engine_lifecycle.go — the admit -> per-step -> stream -> reclaim engine seam.
//
// WHY: the wave-0 EngineDriver is one-shot — Complete buffers a whole turn, never
// consults ctx mid-decode, and exposes no token-timing point, so the gateway's
// "streaming" is synthesized from the finished turn (TTFT == whole-turn latency).
// vLLM-V1's EngineCore and SGLang's scheduler are instead built around this
// admit -> per-step-decode -> stream-output -> reclaim lifecycle, which this file adds.
//
// ADDITIVE, NOT BREAKING: EngineDriver is unchanged; the lifecycle is an OPTIONAL
// extension (LifecycleEngine) found by type assertion and advertised via Caps().
// AdmitOrShim drives any engine uniformly (a one-shot engine gets an honest shim).
//
// PUSH, NOT PULL: the engine drives the decode and surfaces tokens on a per-request
// channel; a pull Step() would force the native continuous-batching scheduler (which
// advances ALL lanes with ONE StepBatch call) to fake a step or serialize the batch.
// This file owns only the SEAM — the scheduler, paged-KV, and cancellation-loop wiring
// are sibling issues that consume this contract. Terminal metadata (finish-reason,
// usage, logprobs, n>1 sampling) is OUT of the base contract and rides additively on
// Result.Meta / Result.Ext (and keep engine.UpstreamToken additive-safe).

package abi

import "context"

// EngineLifecycleCap is the Capability an engine advertises from Caps() when it
// implements LifecycleEngine. It is a PER-ENGINE cap, deliberately kept out of the
// process-global Supported() set: kernel.Negotiate intersects caller caps with that
// global set and would wrongly report lifecycle for a one-shot SELECTED engine.
const EngineLifecycleCap Capability = "engine.lifecycle.v1"

// EngineToken is one streamed decode step: the generated token id plus optional
// incremental detokenized text ("" when the engine has no detokenizer), emitted per
// StepBatch/Step the engine performs for the request.
type EngineToken struct {
	ID   int
	Text string // optional incremental detok text ("" if none)
}

// EngineRequest is one in-flight decode admitted to a LifecycleEngine; the engine
// drives it and the consumer observes it token by token, reclaiming via Cancel.
//
// Tokens() streams this request's decode events and is CLOSED when the request is
// terminal (EOS / budget, cancelled, or faulted) — "channel closed" is the single
// done signal. Result() blocks until terminal, returning the assembled one-shot
// Result or the terminal error (context.Canceled when cancelled mid-decode); safe to
// call before/during/after draining Tokens(). Cancel() aborts at the next step
// boundary and reclaims the slot/KV (idempotent); cancelling ctx is equivalent.
//
// A consumer MUST drain Tokens() or Cancel(): doing neither blocks (leaks) the
// producer goroutine — and, for a native engine, holds its slot/KV — until exit.
type EngineRequest interface {
	Tokens() <-chan EngineToken
	Result() (*Result, error)
	Cancel()
}

// LifecycleEngine is the OPTIONAL admit -> per-step -> stream -> reclaim extension of
// EngineDriver. An implementer SHOULD advertise EngineLifecycleCap from Caps(); a
// non-implementer keeps running one-shot through Complete, reached via AdmitOrShim.
type LifecycleEngine interface {
	EngineDriver
	// Admit registers one decode request and returns a live handle whose Tokens()
	// streams the decode. ctx (or handle.Cancel) stops it at the next step boundary
	// and reclaims the slot/KV — the per-step control point Complete cannot offer.
	// Admit errors only if the request cannot be admitted; decode faults surface via Result().
	Admit(ctx context.Context, c *ToolCall) (EngineRequest, error)
}

// EngineSupportsLifecycle reports whether eng natively implements the lifecycle seam
// (in-process ground truth); cross-registry negotiation may use CapsHaveLifecycle.
func EngineSupportsLifecycle(eng EngineDriver) bool {
	_, ok := eng.(LifecycleEngine)
	return ok
}

// CapsHaveLifecycle reports whether an advertised capability set claims the lifecycle
// seam — the negotiation-time check with no type assertion.
func CapsHaveLifecycle(caps []Capability) bool {
	for _, c := range caps {
		if c == EngineLifecycleCap {
			return true
		}
	}
	return false
}

// AdmitOrShim drives ANY engine through the lifecycle seam. A LifecycleEngine returns
// its native admit handle; a one-shot engine is wrapped in a shim that runs Complete on
// a goroutine and degrades honestly — Tokens() emits nothing and closes when Complete
// returns, and Result() yields Complete's buffered output. The shim honors Cancel/ctx
// only as far as Complete does (a Complete ignoring ctx mid-decode will not stop early).
func AdmitOrShim(ctx context.Context, eng EngineDriver, c *ToolCall) (EngineRequest, error) {
	if le, ok := eng.(LifecycleEngine); ok {
		return le.Admit(ctx, c)
	}
	return newCompleteShim(ctx, eng, c), nil
}

// completeShimRequest adapts a one-shot Complete to the EngineRequest contract.
type completeShimRequest struct {
	tokens chan EngineToken
	done   chan struct{}
	cancel context.CancelFunc

	// written once before close(done); read only after <-done (close is the
	// happens-before edge, so no mutex is needed).
	res *Result
	err error
}

func newCompleteShim(ctx context.Context, eng EngineDriver, c *ToolCall) *completeShimRequest {
	cctx, cancel := context.WithCancel(ctx)
	r := &completeShimRequest{
		tokens: make(chan EngineToken),
		done:   make(chan struct{}),
		cancel: cancel,
	}
	go func() {
		res, err := eng.Complete(cctx, c)
		r.res, r.err = res, err
		close(r.tokens) // one-shot: no per-token stream to emit
		close(r.done)
	}()
	return r
}

// Tokens streams the one-shot shim's decode channel: it emits nothing and closes when Complete returns.
func (r *completeShimRequest) Tokens() <-chan EngineToken { return r.tokens }

// Result blocks until the wrapped Complete finishes, then yields its buffered result or terminal error.
func (r *completeShimRequest) Result() (*Result, error) {
	<-r.done
	return r.res, r.err
}

// Cancel cancels the derived context driving the wrapped Complete, stopping it wherever Complete honors ctx.
func (r *completeShimRequest) Cancel() { r.cancel() }

// Compile-time proof the shim satisfies the seam it bridges to.
var _ EngineRequest = (*completeShimRequest)(nil)
