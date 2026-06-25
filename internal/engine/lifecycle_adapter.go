package engine

// lifecycle_adapter.go — the EXTERNAL-adapter consumer of the lifecycle seam.
//
// This is the Track-A shape: an abi.LifecycleEngine that orchestrates an external
// serving engine (vLLM-V1 / SGLang / Dynamo) over its PUBLIC async token API,
// translating admit -> per-step -> stream -> abort onto that upstream. It is the
// counterpart of the in-kernel per-request engine (internal/modelengine) and the
// native-scheduler stub (modelengine.NativeScheduler): all three implement the SAME
// unchanged interface, which is the cross-shape review the seam exists to pass.
//
// The upstream is a pluggable seam (UpstreamFactory / UpstreamStream) so this
// compiles and is fully testable with NO live model or GPU — a test drives it with
// a scripted token source, exactly as the deterministic mock/cassette engines let
// the dispatch chain run offline. A real adapter implements UpstreamStream over the
// upstream's SSE/gRPC token stream; abort propagates through ctx. Per the issue's
// non-goal this does NOT fork the upstream — it governs it through its public API.

import (
	"context"
	"encoding/json"
	"strconv"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// UpstreamToken is one decoded step from an upstream stream. It is a STRUCT, not a
// positional return, EXACTLY so an adapter author can grow it with upstream-native
// fields (per-token logprobs/top-k, a finish-reason, usage) WITHOUT breaking every
// UpstreamStream implementation — a return-tuple could not grow additively. Done
// marks the turn finished and carries no token.
type UpstreamToken struct {
	ID   int    // the decoded token id
	Text string // optional incremental detok text ("" if none)
	Done bool   // the upstream finished the turn (no token in this event)
}

// UpstreamStream is the minimal async-token surface an external serving engine
// exposes for one request: pull the next decoded step, or learn the stream is
// finished, honoring ctx for mid-stream abort. A real adapter implements Next by
// reading the upstream's token stream; cancelling ctx aborts the upstream request.
type UpstreamStream interface {
	// Next returns the next decoded step. A non-nil err (including ctx.Err() on
	// abort) terminates the request; otherwise UpstreamToken.Done signals the
	// upstream finished the turn.
	Next(ctx context.Context) (UpstreamToken, error)
}

// UpstreamFactory opens an upstream stream for one admitted call. A real adapter
// submits the prompt to the upstream engine and returns a reader over its async
// token stream. An error here fails Admit (the request was never admitted).
type UpstreamFactory func(ctx context.Context, c *abi.ToolCall) (UpstreamStream, error)

// AdapterEngine is the external-adapter LifecycleEngine. Open is required; ID names
// the engine for the result payload + Caps.
type AdapterEngine struct {
	ID   string
	Open UpstreamFactory
}

// NewAdapterEngine builds an adapter over an upstream factory.
func NewAdapterEngine(id string, open UpstreamFactory) *AdapterEngine {
	return &AdapterEngine{ID: id, Open: open}
}

// Caps advertises the lifecycle seam (negotiable streaming/cancel) plus the
// adapter's id token.
func (a *AdapterEngine) Caps() []abi.Capability {
	return []abi.Capability{abi.Capability("engine.adapter." + a.ID), abi.EngineLifecycleCap}
}

// Admit opens the upstream stream and starts a reader goroutine that pumps its
// tokens onto the returned handle, honoring ctx (or handle.Cancel) so an abort
// stops mid-stream and propagates to the upstream via ctx.
func (a *AdapterEngine) Admit(ctx context.Context, c *abi.ToolCall) (abi.EngineRequest, error) {
	up, err := a.Open(ctx, c)
	if err != nil {
		return nil, err
	}
	cctx, cancel := context.WithCancel(ctx)
	r := &adapterRequest{
		tokens: make(chan abi.EngineToken),
		done:   make(chan struct{}),
		cancel: cancel,
		engine: a.ID,
		tool:   c.Tool,
		putCtx: ctx,
	}
	go r.pump(cctx, up)
	return r, nil
}

// Complete is the one-shot shim so the adapter also satisfies the bare
// EngineDriver: admit, drain the stream, return the assembled turn.
func (a *AdapterEngine) Complete(ctx context.Context, c *abi.ToolCall) (*abi.Result, error) {
	req, err := a.Admit(ctx, c)
	if err != nil {
		return nil, err
	}
	for range req.Tokens() {
	}
	res, err := req.Result()
	if err != nil {
		return nil, err
	}
	if res != nil && res.Call == nil {
		res.Call = c
	}
	return res, nil
}

// adapterRequest is one in-flight upstream-backed request.
type adapterRequest struct {
	tokens chan abi.EngineToken
	done   chan struct{}
	cancel context.CancelFunc

	engine string
	tool   string
	putCtx context.Context
	gen    []int

	// written once by pump before close(done); read only after <-done.
	res *abi.Result
	err error
}

func (r *adapterRequest) Tokens() <-chan abi.EngineToken { return r.tokens }

func (r *adapterRequest) Result() (*abi.Result, error) {
	<-r.done
	return r.res, r.err
}

func (r *adapterRequest) Cancel() { r.cancel() }

// pump reads the upstream stream and relays it token by token, checking ctx at the
// top of every loop and on every delivery so an abort lands mid-stream.
func (r *adapterRequest) pump(ctx context.Context, up UpstreamStream) {
	defer r.cancel()
	for {
		if err := ctx.Err(); err != nil {
			r.finish(nil, err)
			return
		}
		ut, err := up.Next(ctx)
		if err != nil {
			r.finish(nil, err)
			return
		}
		if ut.Done {
			r.finish(r.assemble(), nil)
			return
		}
		r.gen = append(r.gen, ut.ID)
		select {
		case r.tokens <- abi.EngineToken{ID: ut.ID, Text: ut.Text}:
		case <-ctx.Done():
			r.finish(nil, ctx.Err())
			return
		}
	}
}

// assemble builds the finished-turn result (the same shape as the in-kernel
// engine: tool + engine + generated token ids + token accounting).
func (r *adapterRequest) assemble() *abi.Result {
	body, _ := json.Marshal(struct {
		Tool   string `json:"tool"`
		Engine string `json:"engine"`
		Tokens []int  `json:"generated_tokens"`
	}{
		Tool:   r.tool,
		Engine: r.engine,
		Tokens: r.gen,
	})
	// input_tokens is intentionally omitted: an external upstream tokenizes the
	// prompt itself and does not report prompt length to the adapter, so unlike the
	// in-kernel engine this path cannot emit it — downstream Meta consumers must
	// treat input_tokens as optional. Typed usage / finish-reason carriers ride on
	// Result.Meta/Ext additively (sibling adapter-issue scope; see the abi seam doc).
	return &abi.Result{
		Payload: putBytes(r.putCtx, body),
		Status:  abi.StatusOK,
		Meta: map[string]string{
			"engine":        r.engine,
			"output_tokens": strconv.Itoa(len(r.gen)),
		},
	}
}

func (r *adapterRequest) finish(res *abi.Result, err error) {
	r.res, r.err = res, err
	close(r.tokens)
	close(r.done)
}

// AdapterEngine is a LifecycleEngine and each request satisfies EngineRequest —
// the same interface the in-kernel and native-scheduler consumers implement.
var (
	_ abi.LifecycleEngine = (*AdapterEngine)(nil)
	_ abi.EngineRequest   = (*adapterRequest)(nil)
)
