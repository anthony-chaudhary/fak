package engine

// continuous_batcher_wire.go — the production caller that makes the
// continuous-batching capability reachable from the ordinary dispatch path.
//
// ContinuousBatcher (continuous_batcher.go) is the ported mini-sglang scheduler:
// budgeted chunked prefill (M1) and decode-resident-between-steps with stable UID
// ordering (M2). This file wires that scheduler onto the abi.LifecycleEngine seam
// as the "batcher" engine so a gateway/harness consumer can reach it via
// abi.Engine(EngineIDBatcher) / AdmitOrShim instead of only through in-package
// tests.
//
// Track-B shape (engine_lifecycle.go): ONE scheduler goroutine owns the batcher,
// runs StepPhase once per iteration, and fans each lane's produced tokens into
// that lane's EngineRequest channel. The engine drives the decode; the consumer
// streams, and cancelling a request (or its ctx) frees the lane.

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strconv"
	"sync"
	"time"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/enginestep"
	"github.com/anthony-chaudhary/fak/internal/refutil"
)

// EngineIDBatcher is the registered engine id for the continuous-batching driver.
const EngineIDBatcher = "batcher"

const (
	defaultBatchingStepInterval = time.Millisecond
	defaultBatchingMaxSteps     = 4096
)

// BatchingEngineConfig configures the batching lifecycle driver.
type BatchingEngineConfig struct {
	// Batcher is the underlying ContinuousBatcher configuration.
	Batcher ContinuousBatcherConfig
	// StepInterval is the pacing delay between scheduler iterations (<= 0 -> 1ms).
	StepInterval time.Duration
	// MaxSteps bounds consecutive no-progress scheduler steps before the loop parks
	// until a new request wakes it; 0 or negative disables the bound. It does not
	// limit total work — progress resets the counter.
	MaxSteps int

	// recorder is the per-engine step recorder. Nil means enginestep.Default.
	// Unexported so production callers cannot accidentally redirect the process
	// recorder; in-package tests inject a private recorder instead of mutating
	// enginestep.Default.
	recorder *enginestep.Recorder
}

// DefaultBatchingEngineConfig returns the default batching driver configuration:
// the calibrated batcher config with budgeted chunked prefill enabled.
func DefaultBatchingEngineConfig() BatchingEngineConfig {
	batcher := DefaultContinuousBatcherConfig()
	batcher.PrefillBudget = DefaultPrefillBudgetTokens
	return BatchingEngineConfig{
		Batcher:      batcher,
		StepInterval: defaultBatchingStepInterval,
		MaxSteps:     defaultBatchingMaxSteps,
	}
}

// BatchingEngine is the abi.LifecycleEngine that owns a ContinuousBatcher and a
// single step-loop goroutine, translating admit/reclaim across the lifecycle seam.
type BatchingEngine struct {
	cfg BatchingEngineConfig
	cb  *ContinuousBatcher

	mu       sync.Mutex
	requests map[string]*batcherRequest

	ctx       context.Context
	cancel    context.CancelFunc
	done      chan struct{}
	wake      chan struct{}
	closeOnce sync.Once

	// recorder receives one observation per scheduler step and per admission.
	// It defaults to enginestep.Default in NewBatchingEngine; a zero-value
	// BatchingEngine leaves it nil and every Observe* call is a no-op (the
	// enginestep methods all guard a nil receiver).
	recorder *enginestep.Recorder
}

// NewBatchingEngine constructs the batching driver and starts its step loop.
func NewBatchingEngine(cfg ...BatchingEngineConfig) (*BatchingEngine, error) {
	c := DefaultBatchingEngineConfig()
	if len(cfg) > 0 {
		c = cfg[0]
		if c.StepInterval <= 0 {
			c.StepInterval = defaultBatchingStepInterval
		}
		if c.MaxSteps <= 0 {
			c.MaxSteps = defaultBatchingMaxSteps
		}
	}

	cb, err := NewContinuousBatcher(c.Batcher)
	if err != nil {
		return nil, err
	}

	recorder := c.recorder
	if recorder == nil {
		recorder = enginestep.Default
	}

	ctx, cancel := context.WithCancel(context.Background())
	e := &BatchingEngine{
		cfg:      c,
		cb:       cb,
		requests: make(map[string]*batcherRequest),
		ctx:      ctx,
		cancel:   cancel,
		done:     make(chan struct{}),
		wake:     make(chan struct{}, 1),
		recorder: recorder,
	}
	go e.run()
	return e, nil
}

// Caps advertises the lifecycle seam plus the batcher's own id token.
func (e *BatchingEngine) Caps() []abi.Capability {
	return []abi.Capability{abi.EngineLifecycleCap, abi.Capability("engine.batcher")}
}

// batcherAdmitArgs is the JSON shape accepted on c.Args.
type batcherAdmitArgs struct {
	SessionID      string `json:"session_id"`
	PromptTokens   []int  `json:"prompt_tokens"`
	TargetTokens   int    `json:"target_tokens"`
	ChunkedPrefill bool   `json:"chunked_prefill"`
	Depth          int    `json:"depth"`
}

// Admit parses the call args into a SubagentRequest, submits it to the batcher,
// registers the live handle, and returns it. The decode is driven by the loop.
func (e *BatchingEngine) Admit(ctx context.Context, c *abi.ToolCall) (abi.EngineRequest, error) {
	if c == nil {
		return nil, ErrNilRequest
	}
	// Refuse early if the engine is already closed: the step loop will never run
	// again, so a request registered now would have no producer and Result() would
	// block forever while its slot/KV leaks.
	if e.ctx.Err() != nil {
		return nil, ErrBatcherClosed
	}

	var args batcherAdmitArgs
	if raw := refutil.Bytes(ctx, c.Args); len(raw) > 0 {
		if err := json.Unmarshal(raw, &args); err != nil {
			return nil, fmt.Errorf("batcher: invalid args: %w", err)
		}
	}

	req := &SubagentRequest{
		SessionID:      args.SessionID,
		PromptTokens:   args.PromptTokens,
		TargetTokens:   args.TargetTokens,
		ChunkedPrefill: args.ChunkedPrefill,
		ExecutionDepth: args.Depth,
	}
	sessionID, err := e.cb.Submit(req)
	if err != nil {
		return nil, err
	}

	rctx, rcancel := context.WithCancel(ctx)
	r := &batcherRequest{
		sessionID:  sessionID,
		tool:       c.Tool,
		engine:     EngineIDBatcher,
		tokens:     make(chan abi.EngineToken, req.TargetTokens+16),
		done:       make(chan struct{}),
		ctx:        rctx,
		cancel:     rcancel,
		putCtx:     ctx,
		enqueuedAt: time.Now(),
	}

	e.mu.Lock()
	e.requests[sessionID] = r
	// Close the Admit/Close race: the early ctx check above is not atomic with
	// registration. If Close (and the loop's final finishAll) interleaved, refuse
	// the handle now, unregister it, and free the batcher slot so Result() never
	// waits on a producer that will not run again. Delete under e.mu, then cancel
	// outside it so we never hold e.mu across cb.mu.
	if e.ctx.Err() != nil {
		delete(e.requests, sessionID)
		e.mu.Unlock()
		_ = e.cb.Cancel(sessionID)
		rcancel()
		return nil, ErrBatcherClosed
	}
	// Instrumentation only: publish the waiting-queue depth and start the
	// end-to-end request timer. A nil recorder is a no-op in both methods.
	e.recorder.SetQueueDepth(e.cb.WaitingQueueLength())
	r.recorderDone = e.recorder.RequestStart()
	e.mu.Unlock()

	e.signal()
	return r, nil
}

// Complete is the one-shot shim so the engine also satisfies the bare EngineDriver:
// admit the call, drain its token stream, and return the assembled turn.
func (e *BatchingEngine) Complete(ctx context.Context, c *abi.ToolCall) (*abi.Result, error) {
	return completeViaAdmit(ctx, e, c)
}

// Close stops the step loop and closes the underlying batcher.
func (e *BatchingEngine) Close() error {
	e.closeOnce.Do(func() { e.cancel() })
	<-e.done
	return e.cb.Close()
}

// run is the single scheduler loop: it steps the batcher, drains produced tokens
// to each request, and finishes requests whose slot completed. It parks when there
// is no outstanding work and wakes on the next Admit.
func (e *BatchingEngine) run() {
	defer close(e.done)
	idle := 0
	for {
		if e.ctx.Err() != nil {
			e.finishAll(e.ctx.Err())
			return
		}
		if e.outstanding() == 0 {
			select {
			case <-e.wake:
			case <-e.ctx.Done():
				e.finishAll(e.ctx.Err())
				return
			}
			continue
		}

		stepStart := time.Now()
		res, err := e.cb.StepPhase(e.ctx)
		if err != nil {
			if e.ctx.Err() != nil {
				e.finishAll(e.ctx.Err())
				return
			}
			if errors.Is(err, ErrBatcherClosed) {
				e.finishAll(err)
				return
			}
		}
		e.observeStep(res, time.Since(stepStart))
		e.pumpAll()

		if res != nil && (res.PrefillTokens > 0 || res.DecodeTokens > 0 ||
			len(res.PromotedSessionIDs) > 0 || len(res.RetiredSessionIDs) > 0) {
			idle = 0
		} else {
			idle++
		}
		if e.cfg.MaxSteps > 0 && idle >= e.cfg.MaxSteps {
			// No progress for MaxSteps steps while work is outstanding: park the
			// loop rather than spin. A new Admit resets the bound.
			select {
			case <-e.wake:
				idle = 0
			case <-e.ctx.Done():
				e.finishAll(e.ctx.Err())
				return
			}
			continue
		}

		if e.cfg.StepInterval > 0 {
			select {
			case <-e.ctx.Done():
				e.finishAll(e.ctx.Err())
				return
			case <-time.After(e.cfg.StepInterval):
			}
		}
	}
}

// observeStep feeds one StepPhase result into the engine recorder. It is pure
// instrumentation: no control flow, locking, ordering, or return value depends on
// it, and every recorder method is a no-op on a nil recorder. Arm selection uses
// res.Phase with the token counts as disambiguator, so a step that both prefills
// and decodes is observed under both arms rather than dropped.
func (e *BatchingEngine) observeStep(res *BatchStepResult, stepDur time.Duration) {
	if res == nil {
		return
	}
	if res.PrefillTokens > 0 || res.Phase == PhasePrefill {
		e.recorder.ObservePrefillChunk(res.PrefillTokens, stepDur)
		e.recorder.ObservePhase(enginestep.PhasePrefill, stepDur)
	}
	if res.DecodeTokens > 0 || res.Phase == PhaseDecode {
		e.recorder.ObserveDecodeStep(enginestep.PathBatched, res.ActiveSlots, stepDur)
	}
	e.recorder.ObserveCohort(res.ActiveSlots)
	if res.PrefixReuseTokens > 0 {
		e.recorder.ObservePrefixMatched(res.PrefixReuseTokens)
	}
	// Admission wait is enqueue -> the step that promotes the request out of the
	// waiting queue. A directly-admitted request never appears here.
	for _, id := range res.PromotedSessionIDs {
		e.mu.Lock()
		r := e.requests[id]
		var waited time.Duration
		if r != nil && !r.enqueuedAt.IsZero() {
			waited = time.Since(r.enqueuedAt)
			r.enqueuedAt = time.Time{}
		}
		e.mu.Unlock()
		if r != nil && waited > 0 {
			e.recorder.ObservePhase(enginestep.PhaseAdmissionWait, waited)
		}
	}
}

// pumpAll drains every in-flight request's slot into its request channel.
func (e *BatchingEngine) pumpAll() {
	e.mu.Lock()
	reqs := make([]*batcherRequest, 0, len(e.requests))
	for _, r := range e.requests {
		reqs = append(reqs, r)
	}
	e.mu.Unlock()

	for _, r := range reqs {
		e.pumpRequest(r)
	}
}

// pumpRequest forwards one request's buffered slot tokens, finishing or cancelling
// it when the slot terminal or the request ctx is done.
func (e *BatchingEngine) pumpRequest(r *batcherRequest) {
	if r.finished() {
		return
	}
	select {
	case <-r.ctx.Done():
		e.cancelRequest(r)
		return
	default:
	}

	slot, ok := e.cb.GetSlot(r.sessionID)
	if !ok {
		return
	}

	for {
		select {
		case tok, open := <-slot.Tokens():
			if !open {
				e.finishRequest(r, r.assemble())
				return
			}
			r.gen = append(r.gen, tok)
			select {
			case r.tokens <- abi.EngineToken{ID: tok}:
			case <-r.ctx.Done():
				e.cancelRequest(r)
				return
			case <-e.ctx.Done():
				return
			}
		default:
			select {
			case <-slot.Done():
				e.finishRequest(r, r.assemble())
			case <-r.ctx.Done():
				e.cancelRequest(r)
			default:
			}
			return
		}
	}
}

func (e *BatchingEngine) finishRequest(r *batcherRequest, res *abi.Result) {
	r.finish(res, nil)
	e.removeRequest(r)
}

func (e *BatchingEngine) cancelRequest(r *batcherRequest) {
	_ = e.cb.Cancel(r.sessionID)
	r.finish(nil, context.Canceled)
	e.removeRequest(r)
}

func (e *BatchingEngine) finishAll(err error) {
	e.mu.Lock()
	reqs := make([]*batcherRequest, 0, len(e.requests))
	for _, r := range e.requests {
		reqs = append(reqs, r)
	}
	e.mu.Unlock()

	for _, r := range reqs {
		if err == nil {
			r.finish(r.assemble(), nil)
		} else {
			r.finish(nil, err)
		}
		e.removeRequest(r)
	}
}

func (e *BatchingEngine) removeRequest(r *batcherRequest) {
	e.mu.Lock()
	delete(e.requests, r.sessionID)
	e.mu.Unlock()
}

func (e *BatchingEngine) outstanding() int {
	e.mu.Lock()
	defer e.mu.Unlock()
	return len(e.requests)
}

func (e *BatchingEngine) signal() {
	select {
	case e.wake <- struct{}{}:
	default:
	}
}

// batcherRequest is one in-flight batcher-backed request.
type batcherRequest struct {
	sessionID string
	tool      string
	engine    string

	tokens chan abi.EngineToken
	done   chan struct{}
	ctx    context.Context
	cancel context.CancelFunc
	putCtx context.Context

	// gen is appended only by the single scheduler goroutine; read only after done.
	gen []int

	// enqueuedAt is the Admit wall time; cleared once the promoting step records
	// admission wait. recorderDone closes the end-to-end PhaseRequest timer.
	enqueuedAt   time.Time
	recorderDone func()

	requestFinish
}

func (r *batcherRequest) Tokens() <-chan abi.EngineToken { return r.tokens }

func (r *batcherRequest) Result() (*abi.Result, error) {
	<-r.done
	return r.res, r.err
}

func (r *batcherRequest) Cancel() { r.cancel() }

func (r *batcherRequest) finished() bool {
	select {
	case <-r.done:
		return true
	default:
		return false
	}
}

func (r *batcherRequest) finish(res *abi.Result, err error) {
	if r.recorderDone != nil {
		r.recorderDone()
		r.recorderDone = nil
	}
	r.requestFinish.complete(r.tokens, r.done, res, err)
}

// assemble builds the finished-turn result (tool + engine + generated token ids,
// with the same meta shape the adapter engines emit).
func (r *batcherRequest) assemble() *abi.Result {
	body, _ := json.Marshal(struct {
		Tool   string `json:"tool"`
		Engine string `json:"engine"`
		Tokens []int  `json:"generated_tokens"`
	}{
		Tool:   r.tool,
		Engine: r.engine,
		Tokens: r.gen,
	})
	return &abi.Result{
		Payload: putBytes(r.putCtx, body),
		Status:  abi.StatusOK,
		Meta: map[string]string{
			"engine":        r.engine,
			"output_tokens": strconv.Itoa(len(r.gen)),
		},
	}
}

// mustDefaultBatchingEngine builds the package default; a construction failure
// is a programming error (the defaults are fixed and valid).
func mustDefaultBatchingEngine() *BatchingEngine {
	e, err := NewBatchingEngine()
	if err != nil {
		panic(fmt.Sprintf("engine: default batching engine: %v", err))
	}
	return e
}

// DefaultBatchingEngine is registered under EngineIDBatcher. It is inert until a
// call is admitted: its loop parks on an empty request set.
var DefaultBatchingEngine = mustDefaultBatchingEngine()

func init() {
	abi.RegisterEngine(EngineIDBatcher, DefaultBatchingEngine)
}

var (
	_ abi.LifecycleEngine = (*BatchingEngine)(nil)
	_ abi.EngineRequest   = (*batcherRequest)(nil)
)
