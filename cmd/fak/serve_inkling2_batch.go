package main

import (
	"context"
	"fmt"
	"os"
	"sync"
	"sync/atomic"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/ggufload"
)

// Admission collects a cohort while arrivals keep coming within gap of each other,
// bounded by maxWait from the first arrival. A fixed short window closes before
// concurrent HTTP clients finish parsing, tokenizing and templating, splitting one
// burst into several narrow cohorts.
const (
	inkling2BatchGapDefault     = 4 * time.Millisecond
	inkling2BatchMaxWaitDefault = 25 * time.Millisecond
	inkling2BatchDurationMax    = 100 * time.Millisecond
)

// inkling2ServeBatchEnabled gates the cohort engine; unset keeps serialized serving.
func inkling2ServeBatchEnabled(lookup func(string) (string, bool)) bool {
	v, _ := lookup("FAK_INKLING2_SERVE_BATCH")
	return v == "1"
}

func inkling2BatchDurationEnv(lookup func(string) (string, bool), name string, def time.Duration) (time.Duration, error) {
	v, ok := lookup(name)
	if !ok || v == "" {
		return def, nil
	}
	d, err := time.ParseDuration(v)
	if err != nil || d <= 0 || d > inkling2BatchDurationMax {
		return 0, fmt.Errorf("%s=%q must be a duration in (0, %s]", name, v, inkling2BatchDurationMax)
	}
	return d, nil
}

// inkling2Clock is injected so admission timing is testable without real sleeps.
type inkling2Clock interface {
	Now() time.Time
	// After arms a one-shot timer; stop releases it early.
	After(d time.Duration) (c <-chan time.Time, stop func())
}

type inkling2RealClock struct{}

func (inkling2RealClock) Now() time.Time { return time.Now() }
func (inkling2RealClock) After(d time.Duration) (<-chan time.Time, func()) {
	t := time.NewTimer(d)
	return t.C, func() { t.Stop() }
}

type inkling2Admission struct {
	gap, maxWait time.Duration
	max          int
	clock        inkling2Clock
}

func inkling2AdmissionFromEnv(lookup func(string) (string, bool)) (inkling2Admission, error) {
	gap, err := inkling2BatchDurationEnv(lookup, "FAK_INKLING2_BATCH_GAP", inkling2BatchGapDefault)
	if err != nil {
		return inkling2Admission{}, err
	}
	wait, err := inkling2BatchDurationEnv(lookup, "FAK_INKLING2_BATCH_WAIT", inkling2BatchMaxWaitDefault)
	if err != nil {
		return inkling2Admission{}, err
	}
	return inkling2Admission{gap: gap, maxWait: wait, max: ggufload.Inkling2LoopMaxBatch, clock: inkling2RealClock{}}, nil
}

// collect grows a cohort from first until the queue goes quiet for gap, the
// cohort reaches max, maxWait elapses since collection began, or the queue closes.
// The deadline starts at collection, not arrival: requests that queued behind the
// previous cohort are already waiting and must still be gathered. A row whose
// caller has already gone is answered here and never takes a slot, so it is never
// prefilled; a nil result means no live row was collected.
func (a inkling2Admission) collect(first *inkling2QueuedRequest, queue <-chan *inkling2QueuedRequest) []*inkling2QueuedRequest {
	if !inkling2RowLive(first) {
		return nil
	}
	group := []*inkling2QueuedRequest{first}
	deadline := a.clock.Now().Add(a.maxWait)
	for len(group) < a.max {
		select {
		case r, ok := <-queue:
			if !ok {
				return group
			}
			if inkling2RowLive(r) {
				group = append(group, r)
			}
			continue
		default:
		}
		wait := min(a.gap, deadline.Sub(a.clock.Now()))
		if wait <= 0 {
			break
		}
		fire, stop := a.clock.After(wait)
		select {
		case r, ok := <-queue:
			stop()
			if !ok {
				return group
			}
			if inkling2RowLive(r) {
				group = append(group, r)
			}
		case <-fire:
			return group
		}
	}
	return group
}

// inkling2RowLive reports whether the row's caller still waits. A row whose caller
// has gone is completed with the context error; done is buffered and the send is
// non-blocking, so no reader is ever needed.
func inkling2RowLive(r *inkling2QueuedRequest) bool {
	if r.ctx == nil || r.ctx.Err() == nil {
		return true
	}
	select {
	case r.done <- inkling2QueuedResult{err: r.ctx.Err()}:
	default:
	}
	return false
}

type inkling2QueuedRequest struct {
	entered time.Time
	ctx     context.Context
	prompt  []int
	maxNew  int
	stops   map[int]bool
	sink    *inkling2RowSink
	done    chan inkling2QueuedResult
}

// inkling2RowSink is the worker's only path to a caller's emit. Generate closes it
// before returning early, and close waits out an emit in flight, so the worker can
// never call emit after its caller has returned.
type inkling2RowSink struct {
	mu     sync.Mutex
	closed bool
	emit   func(int) bool
}

func (s *inkling2RowSink) send(id int) bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	return !s.closed && s.emit(id)
}

func (s *inkling2RowSink) close() {
	s.mu.Lock()
	s.closed = true
	s.mu.Unlock()
}

type inkling2QueuedResult struct {
	gen agent.Inkling2LoopGeneration
	err error
}

type inkling2ScalarGenerate func(context.Context, []int, int, map[int]bool, func(int) bool) (agent.Inkling2LoopGeneration, error)
type inkling2BatchGenerate func([]ggufload.Inkling2LoopBatchRequest) ([]ggufload.Inkling2LoopBatchResult, error)

// inkling2LoopBatchEngine owns the only GPU executor: a lone admission runs the
// scalar engine, a wider cohort shares one row-batched greedy forward.
type inkling2LoopBatchEngine struct {
	admission      inkling2Admission
	contextTokens  int
	queue          chan *inkling2QueuedRequest
	peak           atomic.Int32
	scalarGroups   atomic.Uint64
	batchGroups    atomic.Uint64
	scalarGenerate inkling2ScalarGenerate
	batchGenerate  inkling2BatchGenerate
	// releaseArenas frees the cached cohort arenas, keeping the weight binding, so
	// a lone request's scalar session does not have to fit beside the widest
	// cohort's KV. arenasCached is touched only by the run worker.
	releaseArenas func()
	arenasCached  bool
	trace         bool
}

var _ agent.Inkling2LoopGenerator = (*inkling2LoopBatchEngine)(nil)

// newInkling2LoopBatchEngine keeps one cohort state for the engine's life: the run
// worker is its only user.
func newInkling2LoopBatchEngine(scalar inkling2ScalarGenerate, model *ggufload.Inkling2LoopModel, backend compute.Backend, adm inkling2Admission, lookup func(string) (string, bool)) *inkling2LoopBatchEngine {
	cohorts := ggufload.NewInkling2LoopCohorts(model, backend)
	e := newInkling2LoopBatchEngineWith(scalar, cohorts.Generate, adm, ggufload.Inkling2LoopResidentContext)
	e.releaseArenas = cohorts.ReleaseArenas
	if v, _ := lookup("FAK_INKLING2_TRACE"); v == "1" {
		e.trace = true
	}
	go e.run()
	return e
}

func newInkling2LoopBatchEngineWith(scalar inkling2ScalarGenerate, batch inkling2BatchGenerate, adm inkling2Admission, contextTokens int) *inkling2LoopBatchEngine {
	return &inkling2LoopBatchEngine{
		admission: adm, contextTokens: contextTokens,
		queue:          make(chan *inkling2QueuedRequest, adm.max),
		scalarGenerate: scalar, batchGenerate: batch,
	}
}

// SerializesGPUAccess lets the planner admit concurrent turns: the single run
// worker is the serialization boundary.
func (*inkling2LoopBatchEngine) SerializesGPUAccess() bool { return true }

func (e *inkling2LoopBatchEngine) Generate(ctx context.Context, prompt []int, maxNew int, stops map[int]bool, emit func(int) bool) (agent.Inkling2LoopGeneration, error) {
	if ctx == nil || len(prompt) == 0 || maxNew <= 0 || emit == nil || len(prompt)+maxNew-1 > e.contextTokens {
		return agent.Inkling2LoopGeneration{}, fmt.Errorf("inkling2 batch queue: invalid request")
	}
	if err := ctx.Err(); err != nil {
		return agent.Inkling2LoopGeneration{}, err
	}
	r := &inkling2QueuedRequest{entered: e.admission.clock.Now(), ctx: ctx, prompt: append([]int(nil), prompt...), maxNew: maxNew, stops: stops,
		sink: &inkling2RowSink{emit: emit}, done: make(chan inkling2QueuedResult, 1)}
	select {
	case e.queue <- r:
	case <-ctx.Done():
		return agent.Inkling2LoopGeneration{}, ctx.Err()
	}
	select {
	case result := <-r.done:
		return result.gen, result.err
	case <-ctx.Done():
		// the worker retires the row at its next step; closing the sink lets the
		// caller go now without the worker ever reaching emit again
		r.sink.close()
		select {
		case result := <-r.done:
			return result.gen, result.err
		default:
			return agent.Inkling2LoopGeneration{}, ctx.Err()
		}
	}
}

func (e *inkling2LoopBatchEngine) run() {
	for first := range e.queue {
		if group := e.admission.collect(first, e.queue); len(group) > 0 {
			e.execute(group)
		}
	}
}

func (e *inkling2LoopBatchEngine) execute(group []*inkling2QueuedRequest) {
	// The worker goroutine sits outside net/http's per-handler recovery, so a
	// panic here would kill the server; fail the cohort instead. done has a
	// one-slot buffer, so the non-blocking send skips rows already answered.
	defer func() {
		if p := recover(); p != nil {
			err := fmt.Errorf("inkling2 batch worker panic: %v", p)
			for _, r := range group {
				select {
				case r.done <- inkling2QueuedResult{err: err}:
				default:
				}
			}
		}
	}()
	e.observePeak(len(group))
	if e.trace {
		now := e.admission.clock.Now()
		waits := make([]int64, len(group))
		for i, r := range group {
			waits[i] = now.Sub(r.entered).Nanoseconds()
		}
		fmt.Fprintf(os.Stderr, "inkling2.admission width=%d queue_wait_ns=%v\n", len(group), waits)
	}
	if len(group) == 1 {
		e.scalarGroups.Add(1)
		if e.arenasCached && e.releaseArenas != nil {
			e.releaseArenas()
		}
		e.arenasCached = false
		r := group[0]
		gen, err := e.scalarGenerate(r.ctx, r.prompt, r.maxNew, r.stops, r.sink.send)
		gen.Width = 1
		r.done <- inkling2QueuedResult{gen: gen, err: err}
		return
	}
	e.batchGroups.Add(1)
	e.arenasCached = true
	e.runBatch(group)
}

func (e *inkling2LoopBatchEngine) observePeak(width int) {
	for old := e.peak.Load(); int32(width) > old && !e.peak.CompareAndSwap(old, int32(width)); old = e.peak.Load() {
	}
}

func inkling2BatchRowResult(br ggufload.Inkling2LoopBatchResult, width int) inkling2QueuedResult {
	return inkling2QueuedResult{err: br.Err, gen: agent.Inkling2LoopGeneration{
		Generated: br.Generated, Stopped: br.Stopped, Width: width,
		PrefillSeconds: br.Stats.Prefill.Seconds(), DecodeSeconds: br.Stats.Decode.Seconds(),
	}}
}

// runBatch answers each row the moment it ends, so a row that stops or is cancelled
// early does not wait for the cohort's longest row.
func (e *inkling2LoopBatchEngine) runBatch(group []*inkling2QueuedRequest) {
	answered := make([]bool, len(group))
	reqs := make([]ggufload.Inkling2LoopBatchRequest, len(group))
	for i, r := range group {
		i, r := i, r
		reqs[i] = ggufload.Inkling2LoopBatchRequest{
			Context: r.ctx, Prompt: r.prompt, MaxTokens: r.maxNew,
			Stop: func(token int) bool { return r.stops[token] },
			Emit: r.sink.send,
			Done: func(br ggufload.Inkling2LoopBatchResult) {
				answered[i] = true
				r.done <- inkling2BatchRowResult(br, len(group))
			},
		}
	}
	results, fatal := e.batchGenerate(reqs)
	for i, r := range group {
		if answered[i] {
			continue
		}
		if fatal != nil || i >= len(results) {
			if fatal == nil {
				fatal = fmt.Errorf("inkling2 batch: %d results for %d rows", len(results), len(group))
			}
			r.done <- inkling2QueuedResult{err: fatal}
			continue
		}
		r.done <- inkling2BatchRowResult(results[i], len(group))
	}
}
