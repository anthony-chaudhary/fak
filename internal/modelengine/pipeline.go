package modelengine

// pipeline.go — EngineDriver/LifecycleEngine reachability for the network PP worker.
//
// internal/model owns the bit-exact pipeline-parallel substrate: TCPTransport moves
// hidden-state frames and ServeBand runs a resident layer band on the peer. PipelineEngine
// is the thin modelengine bridge for that substrate. It is deliberately conservative:
// one admitted request re-forwards the growing sequence through RunPipelineAcrossWorkers
// once per generated token, then streams that token on the abi.LifecycleEngine channel.
// That is slower than an incremental PP scheduler, but it puts the real ServeBand loop
// behind the same EngineDriver admit/stream/reclaim surface as the registered in-kernel
// engine without forking the proven worker code.

import (
	"context"
	"fmt"
	"sync"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/model"
)

// PipelineEngineID is the default id for a PP head engine when a host registers one.
const PipelineEngineID = "inkernel-pipeline"

// PipelineEngine drives the FIRST pipeline stage in-process and reaches the remaining
// stages through downstream, typically a TCPTransport whose peer is model.ServeBand.
// It satisfies abi.LifecycleEngine, so hosts can register/select it through the normal
// EngineDriver path instead of a test-only harness.
type PipelineEngine struct {
	first      model.PipelineStage
	downstream model.StageTransport
	tok        NLTokenizer
	maxTokens  int

	// TCPTransport is a request/reply stream over one net.Conn; serialize Send users so
	// concurrent admitted requests cannot interleave length-prefixed frames.
	mu sync.Mutex
}

// NewPipelineEngine builds a lifecycle-capable PP head engine. first must be the
// First stage. If first is not also Last, downstream must reach the next ServeBand worker.
func NewPipelineEngine(first model.PipelineStage, downstream model.StageTransport) *PipelineEngine {
	return &PipelineEngine{first: first, downstream: downstream, maxTokens: genTokens}
}

// SetTokenizer matches Engine.SetTokenizer for hosts that preload a real tokenizer.
func (e *PipelineEngine) SetTokenizer(t NLTokenizer) {
	if e == nil || t == nil {
		return
	}
	e.tok = t
}

// Caps advertises both the PP engine identity and the lifecycle seam it implements.
func (e *PipelineEngine) Caps() []abi.Capability {
	return []abi.Capability{"engine.inkernel.pipeline", abi.EngineLifecycleCap}
}

// Complete is the one-shot EngineDriver shim over Admit, matching Engine and NativeScheduler.
func (e *PipelineEngine) Complete(ctx context.Context, c *abi.ToolCall) (*abi.Result, error) {
	req, err := e.Admit(ctx, c)
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

// Admit starts a streamed generation over the network PP pipeline.
func (e *PipelineEngine) Admit(ctx context.Context, c *abi.ToolCall) (abi.EngineRequest, error) {
	if e == nil {
		return nil, fmt.Errorf("modelengine: nil PipelineEngine")
	}
	if e.first.Model == nil {
		return nil, fmt.Errorf("modelengine: pipeline first stage has nil model")
	}
	if !e.first.Spec.First {
		return nil, fmt.Errorf("modelengine: pipeline first stage band [%d,%d) is not marked First", e.first.Spec.Lo, e.first.Spec.Hi)
	}
	if !e.first.Spec.Last && e.downstream == nil {
		return nil, fmt.Errorf("modelengine: pipeline multi-stage engine needs downstream transport")
	}

	cctx, cancel := context.WithCancel(ctx)
	r := &pipelineRequest{
		ctx:    cctx,
		cancel: cancel,
		tokens: make(chan abi.EngineToken, 1),
		done:   make(chan struct{}),
	}
	args := refBytes(ctx, c.Args)
	prompt := buildPipelinePrompt(e.tok, c.Tool, args, e.first.Model.Cfg.VocabSize)
	go e.run(r, ctx, c.Tool, prompt)
	return r, nil
}

func (e *PipelineEngine) run(r *pipelineRequest, putCtx context.Context, tool string, prompt []int) {
	defer close(r.done)
	defer close(r.tokens)

	ids := append([]int(nil), prompt...)
	gen := make([]int, 0, e.decodeBudget())
	for len(gen) < e.decodeBudget() {
		if err := r.ctx.Err(); err != nil {
			r.err = err
			return
		}
		e.mu.Lock()
		logits, err := model.RunPipelineAcrossWorkers(ids, e.first, e.downstream)
		e.mu.Unlock()
		if err != nil {
			r.err = err
			return
		}
		if len(logits) == 0 || len(logits[len(logits)-1]) == 0 {
			r.err = fmt.Errorf("modelengine: pipeline produced empty logits")
			return
		}
		next := argmax(logits[len(logits)-1])
		select {
		case r.tokens <- abi.EngineToken{ID: next}:
		case <-r.ctx.Done():
			r.err = r.ctx.Err()
			return
		}
		gen = append(gen, next)
		if e.first.Model.Cfg.IsEOS(next) {
			break
		}
		ids = append(ids, next)
	}
	r.res = assemblePipelineResult(putCtx, tool, len(prompt), gen, e.tok)
}

func (e *PipelineEngine) decodeBudget() int {
	if e.maxTokens > 0 {
		return e.maxTokens
	}
	return genTokens
}

func buildPipelinePrompt(tok NLTokenizer, tool string, args []byte, vocab int) []int {
	if tok != nil {
		if ids, err := tok.Encode(tool + " " + string(args)); err == nil && len(ids) > 0 {
			if len(ids) > maxPromptTokens {
				ids = ids[:maxPromptTokens]
			}
			if idsWithinVocab(ids, vocab) {
				return ids
			}
		}
	}
	return tokenize(tool, args, vocab)
}

func assemblePipelineResult(ctx context.Context, tool string, promptLen int, gen []int, tok NLTokenizer) *abi.Result {
	return assembleSyscallResult(ctx, tool, PipelineEngineID, "smollm2-inkernel-pipeline", promptLen, gen, tok)
}

type pipelineRequest struct {
	ctx    context.Context
	cancel context.CancelFunc

	tokens chan abi.EngineToken
	done   chan struct{}

	res *abi.Result
	err error
}

func (r *pipelineRequest) Tokens() <-chan abi.EngineToken { return r.tokens }

func (r *pipelineRequest) Result() (*abi.Result, error) {
	<-r.done
	return r.res, r.err
}

func (r *pipelineRequest) Cancel() { r.cancel() }

var (
	_ abi.LifecycleEngine = (*PipelineEngine)(nil)
	_ abi.EngineRequest   = (*pipelineRequest)(nil)
)
