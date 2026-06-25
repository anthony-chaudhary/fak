// engine_lifecycle.go — the admit -> per-step -> stream -> reclaim engine seam.
//
// WHY THIS EXISTS (the critique's #1 risk). The wave-0 EngineDriver contract is
// one-shot: Complete(ctx, *ToolCall) (*Result, error). It buffers a whole turn,
// never consults ctx inside the decode loop (so a client disconnect cannot stop
// or reclaim an in-flight generation), and gives no token-timing control point —
// which is why the gateway's "streaming" is synthesized from the finished turn
// (TTFT == whole-turn latency). vLLM-V1's EngineCore and SGLang's scheduler are
// instead built around an admit -> per-step-decode -> stream-output -> reclaim
// lifecycle, and their continuous batching, real streaming, and cancellation all
// hang off it. This file adds that lifecycle to fak's engine seam.
//
// ADDITIVE, NOT BREAKING. EngineDriver is unchanged: every engine that ships only
// Complete (the mock, the cassette, the gateway test echoes, localtools, the
// remote LiteLLM transport) still registers and runs exactly as before. The
// lifecycle is an OPTIONAL extension interface (LifecycleEngine) discovered by a
// type assertion, and advertised via Caps() (EngineLifecycleCap) so a consumer can
// negotiate streaming/cancel WITHOUT a type assertion (e.g. across the registry).
// A consumer drives ANY engine uniformly through AdmitOrShim: a lifecycle engine
// returns its native admit handle; a one-shot engine is wrapped in a shim that
// degrades honestly to "zero streamed tokens, then the buffered Result."
//
// DESIGNED AGAINST BOTH CONSUMER SHAPES (acceptance #4). The seam is a PUSH model:
// the engine drives the decode and surfaces tokens on a per-request channel. It is
// deliberately NOT a per-request pull Step(), because that would force the native
// continuous-batching scheduler — which advances ALL admitted lanes with ONE
// StepBatch call — to either fake a per-request step or serialize the batch. With
// the push shape both consumers fit the same interface:
//   - Track A (external adapter): one reader per request pumps the upstream
//     vLLM/SGLang async token stream into the request's channel; ctx abort
//     propagates to the upstream.
//   - Track B (native scheduler): ONE scheduler loop owns the BatchSession, runs
//     StepBatch once per iteration, and fans each lane's produced token into that
//     lane's request channel; cancelling a request frees its lane/KV.
// This file owns only the SEAM. The continuous-batching scheduler, paged-KV, and
// native cancellation-loop wiring are sibling issues that CONSUME this contract.
//
// TERMINAL METADATA IS DELIBERATELY OUT OF THE BASE CONTRACT. finish-reason,
// usage/token-accounting, per-token logprobs, and n>1 (parallel-sampling/beam)
// sequences are NOT in EngineToken/EngineRequest: this seam is greedy
// single-sequence base-item parity (the issue's non-goal). They ride ADDITIVELY on
// Result.Meta (open map) / Result.Ext (typed sidecar by reserved ExtKey range), and
// EngineToken/Result are structs that can grow new fields without breaking any
// consumer — so blessing the typed carriers is the sibling adapter issues' scope,
// not a future break of this interface. The one return shape an adapter author must
// keep additive-safe is the upstream token event: see engine.UpstreamToken (a
// struct, not a positional tuple, exactly so it can grow logprobs/finish-reason).

package abi

import "context"

// EngineLifecycleCap is the Capability an engine advertises from Caps() when it
// implements LifecycleEngine. It is a PER-ENGINE advertised cap: lifecycle support
// is a property of the bound engine, not of the process, so it is deliberately NOT
// passed to RegisterCapability / surfaced in the process-global Supported() set
// (kernel.Negotiate intersects caller caps with that global set and would wrongly
// report lifecycle for a process whose SELECTED engine is one-shot). The correct
// probes are CapsHaveLifecycle(eng.Caps()) at negotiation time and
// EngineSupportsLifecycle(eng) for the in-process ground truth.
const EngineLifecycleCap Capability = "engine.lifecycle.v1"

// EngineToken is one streamed decode step: the generated token id, plus optional
// incremental detokenized text (empty when the engine has no detokenizer — the
// in-kernel byte model does not, an external adapter may). One EngineToken is
// emitted per StepBatch/Step the engine performs for the request.
type EngineToken struct {
	ID   int    // the generated token id
	Text string // optional incremental detok text ("" if none)
}

// EngineRequest is one in-flight decode admitted to a LifecycleEngine. The decode
// is driven by the engine (a per-request reader for an external adapter; a shared
// StepBatch loop for the native scheduler); the consumer observes it token by
// token and reclaims it via Cancel.
//
// Contract:
//   - Tokens() streams the decode events for THIS request. The channel is CLOSED
//     when the request is terminal — EOS / token budget reached, cancelled, or
//     faulted. "Channel closed" is the single done signal.
//   - Result() blocks until the request is terminal, then returns the assembled
//     one-shot Result (the same value Complete would have produced) or the
//     terminal error (context.Canceled when cancelled mid-decode). It is safe to
//     call before, during, or after draining Tokens(), and any number of times.
//   - Cancel() aborts the request at the NEXT step boundary and signals the engine
//     to reclaim the request's slot/KV. It is idempotent and a no-op once the
//     request has already finished. Cancelling ctx (passed to Admit) is
//     equivalent — both are real per-step control points, unlike Complete's
//     buffered loop.
//
// A consumer MUST either drain Tokens() to completion or Cancel() the request:
// doing neither blocks (and so leaks) the request's producer goroutine — and, for
// a native engine, holds its slot/KV — until process exit. The AdmitOrShim
// one-shot path is exempt (its shim emits no per-token send and always closes).
type EngineRequest interface {
	Tokens() <-chan EngineToken
	Result() (*Result, error)
	Cancel()
}

// LifecycleEngine is the OPTIONAL admit -> per-step -> stream -> reclaim extension
// of EngineDriver. An engine that implements it SHOULD advertise EngineLifecycleCap
// from Caps(). An engine that does NOT implement it keeps running one-shot through
// Complete — the kernel and every other consumer reach it via AdmitOrShim.
type LifecycleEngine interface {
	EngineDriver
	// Admit registers one decode request and returns a live handle whose Tokens()
	// streams the decode. ctx (or handle.Cancel) stops the decode at the next step
	// boundary and reclaims the slot/KV — the real per-step control point the
	// one-shot Complete cannot offer. Admit returns an error only if the request
	// cannot be admitted at all (e.g. a closed/over-capacity scheduler); decode
	// faults surface later via Result().
	Admit(ctx context.Context, c *ToolCall) (EngineRequest, error)
}

// EngineSupportsLifecycle reports whether eng natively implements the lifecycle
// seam (in-process ground truth). For cross-registry negotiation a consumer may
// instead intersect Caps() with EngineLifecycleCap (CapsHaveLifecycle).
func EngineSupportsLifecycle(eng EngineDriver) bool {
	_, ok := eng.(LifecycleEngine)
	return ok
}

// CapsHaveLifecycle reports whether an advertised capability set claims the
// lifecycle seam. This is the negotiation-time check (no type assertion); a driver
// that advertises EngineLifecycleCap MUST implement LifecycleEngine.
func CapsHaveLifecycle(caps []Capability) bool {
	for _, c := range caps {
		if c == EngineLifecycleCap {
			return true
		}
	}
	return false
}

// AdmitOrShim drives ANY engine through the lifecycle seam. A LifecycleEngine
// returns its native admit handle (real streaming + per-step cancellation). A
// one-shot engine is wrapped in a shim that runs Complete on a goroutine and
// degrades honestly: its Tokens() channel emits NOTHING and closes when Complete
// returns, and Result() yields Complete's buffered output. The shim still honors
// Cancel/ctx as far as Complete itself does (a Complete that ignores ctx mid-decode
// will not stop early — which is exactly why such an engine does not advertise
// EngineLifecycleCap). This is the single bridge a consumer (gateway streaming,
// the kernel, the native scheduler's fallback) calls so a one-shot-only driver
// "still registers and runs" with no special-casing.
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

func (r *completeShimRequest) Tokens() <-chan EngineToken { return r.tokens }

func (r *completeShimRequest) Result() (*Result, error) {
	<-r.done
	return r.res, r.err
}

func (r *completeShimRequest) Cancel() { r.cancel() }

// Compile-time proof the shim satisfies the seam it bridges to.
var _ EngineRequest = (*completeShimRequest)(nil)
