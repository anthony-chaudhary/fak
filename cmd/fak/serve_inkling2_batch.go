package main

import (
	"context"
	"errors"
	"fmt"
	"os"
	"slices"
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

// inkling2BatchAdmitting runs a cohort that takes queued requests into the slots of
// rows that finish: admit is polled at each boundary and returns the next request that
// fits the cohort, if one is waiting.
type inkling2BatchAdmitting func([]ggufload.Inkling2LoopBatchRequest, func(int) (ggufload.Inkling2LoopBatchRequest, bool)) ([]ggufload.Inkling2LoopBatchResult, error)

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
	// batchAdmitting, when set, runs cohorts with admission instead of batchGenerate.
	batchAdmitting inkling2BatchAdmitting
	// pending is the first queued request a running cohort could not fit: it is the
	// next cohort's first row. Only the run worker touches it.
	pending *inkling2QueuedRequest
	// admittedRows counts requests that joined a running cohort.
	admittedRows atomic.Uint64
	// split partitions a cohort by each row's len(prompt)+maxNew into sub-cohorts that
	// fit device memory; nil runs every collected group as one cohort.
	split func(positions []int) ([][]int, error)
	// splitGroups counts cohorts split for memory, up front or after a refusal.
	splitGroups atomic.Uint64
	// releaseArenas frees the cached cohort arenas, keeping the weight binding, so
	// a lone request's scalar session does not have to fit beside the widest
	// cohort's KV. arenasCached is touched only by the run worker.
	releaseArenas func()
	arenasCached  bool
	trace         bool
	// loneCohort serves a lone request as a one-row cohort rather than on the
	// serialized engine, so it reads and feeds the cohorts' cross-request prefix store
	loneCohort bool
	// prefixStats reads the prefix store's counters for the trace; nil when it is off
	prefixStats func() ggufload.Inkling2PrefixStoreStats
}

var _ agent.Inkling2LoopGenerator = (*inkling2LoopBatchEngine)(nil)

// newInkling2LoopBatchEngine keeps one cohort state for the engine's life: the run
// worker is its only user.
func newInkling2LoopBatchEngine(scalar inkling2ScalarGenerate, model *ggufload.Inkling2LoopModel, backend compute.Backend, adm inkling2Admission, lookup func(string) (string, bool)) *inkling2LoopBatchEngine {
	cohorts := ggufload.NewInkling2LoopCohorts(model, backend)
	e := newInkling2LoopBatchEngineWith(scalar, cohorts.Generate, adm, ggufload.Inkling2LoopResidentContext)
	e.releaseArenas = cohorts.ReleaseArenas
	e.batchAdmitting = cohorts.GenerateAdmitting
	e.split = cohorts.Split
	if on, _ := cohorts.UsePrefixStore(lookup); on {
		e.loneCohort = true
		e.prefixStats = cohorts.PrefixStats
	}
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

// run serves cohorts in arrival order. A request a running cohort could not fit is
// the next cohort's first row, ahead of anything still queued.
func (e *inkling2LoopBatchEngine) run() {
	for {
		first := e.pending
		e.pending = nil
		if first == nil {
			r, ok := <-e.queue
			if !ok {
				return
			}
			first = r
		}
		if group := e.admission.collect(first, e.queue); len(group) > 0 {
			e.execute(group)
		}
	}
}

func (e *inkling2LoopBatchEngine) execute(group []*inkling2QueuedRequest) {
	// The worker goroutine sits outside net/http's per-handler recovery, so a
	// panic here would kill the server; fail the cohort instead. done has a
	// one-slot buffer, so the non-blocking send skips rows already answered.
	// owned grows as a running cohort admits rows, so a panic answers those too.
	owned := group
	defer func() {
		if p := recover(); p != nil {
			err := fmt.Errorf("inkling2 batch worker panic: %v", p)
			for _, r := range owned {
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
	if len(group) == 1 && !e.loneCohort {
		e.runScalar(group[0])
		return
	}
	for _, g := range e.splitGroup(group) {
		e.runGroup(g, &owned)
	}
}

// runScalar serves one row on the serialized engine, beside the weight binding only.
func (e *inkling2LoopBatchEngine) runScalar(r *inkling2QueuedRequest) {
	e.scalarGroups.Add(1)
	if e.arenasCached && e.releaseArenas != nil {
		e.releaseArenas()
	}
	e.arenasCached = false
	gen, err := e.scalarGenerate(r.ctx, r.prompt, r.maxNew, r.stops, r.sink.send)
	gen.Width = 1
	r.done <- inkling2QueuedResult{gen: gen, err: err}
}

// splitGroup partitions a collected cohort into the sub-cohorts split plans, in plan
// order. A failed plan keeps the cohort whole, so the batch reports the failure.
func (e *inkling2LoopBatchEngine) splitGroup(group []*inkling2QueuedRequest) [][]*inkling2QueuedRequest {
	if e.split == nil {
		return [][]*inkling2QueuedRequest{group}
	}
	positions := make([]int, len(group))
	for i, r := range group {
		positions[i] = len(r.prompt) + r.maxNew
	}
	plan, err := e.split(positions)
	if err != nil || len(plan) <= 1 {
		return [][]*inkling2QueuedRequest{group}
	}
	e.splitGroups.Add(1)
	groups := make([][]*inkling2QueuedRequest, len(plan))
	widths := make([]int, len(plan))
	for g, rows := range plan {
		for _, i := range rows {
			groups[g] = append(groups[g], group[i])
		}
		widths[g] = len(rows)
	}
	if e.trace {
		fmt.Fprintf(os.Stderr, "inkling2.split width=%d groups=%v\n", len(group), widths)
	}
	return groups
}

// runGroup serves a sub-cohort: one row on the serialized engine, more as a batch. A
// batch refused for device memory before any row ran is halved and each half served
// the same way, so memory pressure narrows a cohort down to the serialized engine
// instead of failing its rows.
func (e *inkling2LoopBatchEngine) runGroup(group []*inkling2QueuedRequest, owned *[]*inkling2QueuedRequest) {
	if len(group) == 1 && !e.loneCohort {
		e.runScalar(group[0])
		return
	}
	e.batchGroups.Add(1)
	e.arenasCached = true
	ran := e.runBatch(group, owned)
	if e.trace && e.prefixStats != nil {
		fmt.Fprintf(os.Stderr, "inkling2.prefix width=%d store=%+v\n", len(group), e.prefixStats())
	}
	if ran {
		return
	}
	if len(group) == 1 {
		// a one-row cohort refused device memory still fits the serialized engine
		e.runScalar(group[0])
		return
	}
	e.splitGroups.Add(1)
	if e.trace {
		fmt.Fprintf(os.Stderr, "inkling2.split refused width=%d halves=%d+%d\n", len(group), len(group)/2, len(group)-len(group)/2)
	}
	half := len(group) / 2
	e.runGroup(group[:half], owned)
	e.runGroup(group[half:], owned)
}

func (e *inkling2LoopBatchEngine) observePeak(width int) {
	for old := e.peak.Load(); int32(width) > old && !e.peak.CompareAndSwap(old, int32(width)); old = e.peak.Load() {
	}
}

func inkling2BatchRowResult(br ggufload.Inkling2LoopBatchResult, width int) inkling2QueuedResult {
	return inkling2QueuedResult{err: br.Err, gen: agent.Inkling2LoopGeneration{
		Generated: br.Generated, Stopped: br.Stopped, Width: width, CachedPromptTokens: br.Restored,
		PrefillSeconds: br.Stats.Prefill.Seconds(), DecodeSeconds: br.Stats.Decode.Seconds(),
	}}
}

// inkling2Cohort is the set of rows one batch run owns: rows are its collected and
// admitted rows, answered marks those already completed, and owned is the worker's
// list of every row it must answer if the run panics.
type inkling2Cohort struct {
	width    int
	rows     []*inkling2QueuedRequest
	owned    *[]*inkling2QueuedRequest
	answered []bool
}

// request wraps row i of the cohort for the batch: its Done answers the caller the
// moment the row ends, while the cohort may still be stepping.
func (c *inkling2Cohort) request(i int, r *inkling2QueuedRequest) ggufload.Inkling2LoopBatchRequest {
	return ggufload.Inkling2LoopBatchRequest{
		Context: r.ctx, Prompt: r.prompt, MaxTokens: r.maxNew,
		Stop: func(token int) bool { return r.stops[token] },
		Emit: r.sink.send,
		Done: func(br ggufload.Inkling2LoopBatchResult) {
			c.answered[i] = true
			r.done <- inkling2BatchRowResult(br, c.width)
		},
	}
}

// admit records r as a cohort row and wraps it.
func (c *inkling2Cohort) admit(r *inkling2QueuedRequest) ggufload.Inkling2LoopBatchRequest {
	*c.owned = append(*c.owned, r)
	c.rows = append(c.rows, r)
	c.answered = append(c.answered, false)
	return c.request(len(c.answered)-1, r)
}

// admitQueued is the admission callback of a running cohort. It takes the next live
// queued request if it fits maxPositions. One that does not fit is held as pending and
// nothing behind it is admitted, so a large request is not starved by small ones.
func (e *inkling2LoopBatchEngine) admitQueued(c *inkling2Cohort, maxPositions int) (ggufload.Inkling2LoopBatchRequest, bool) {
	for e.pending == nil {
		var r *inkling2QueuedRequest
		var ok bool
		select {
		case r, ok = <-e.queue:
		default:
			return ggufload.Inkling2LoopBatchRequest{}, false
		}
		if !ok {
			return ggufload.Inkling2LoopBatchRequest{}, false
		}
		if !inkling2RowLive(r) {
			continue
		}
		if len(r.prompt)+r.maxNew > maxPositions {
			e.pending = r
			return ggufload.Inkling2LoopBatchRequest{}, false
		}
		e.admittedRows.Add(1)
		return c.admit(r), true
	}
	return ggufload.Inkling2LoopBatchRequest{}, false
}

// runBatch answers each row the moment it ends, so a row that stops or is cancelled
// early does not wait for the cohort's longest row. Rows admitted while the cohort runs
// are answered the same way; any row the batch leaves unanswered gets the batch's error.
// It reports false, answering nothing, when the batch was refused device memory before
// any row ran: the caller serves those rows on narrower cohorts.
func (e *inkling2LoopBatchEngine) runBatch(group []*inkling2QueuedRequest, owned *[]*inkling2QueuedRequest) bool {
	c := &inkling2Cohort{width: len(group), rows: append([]*inkling2QueuedRequest(nil), group...), owned: owned, answered: make([]bool, len(group))}
	reqs := make([]ggufload.Inkling2LoopBatchRequest, len(group))
	for i, r := range group {
		reqs[i] = c.request(i, r)
	}
	var results []ggufload.Inkling2LoopBatchResult
	var fatal error
	if e.batchAdmitting != nil {
		results, fatal = e.batchAdmitting(reqs, func(maxPositions int) (ggufload.Inkling2LoopBatchRequest, bool) {
			return e.admitQueued(c, maxPositions)
		})
	} else {
		results, fatal = e.batchGenerate(reqs)
	}
	var refused *ggufload.Inkling2BatchMemoryError
	if errors.As(fatal, &refused) && len(c.rows) == len(group) && !slices.Contains(c.answered, true) {
		return false
	}
	for i, r := range c.rows {
		if c.answered[i] {
			continue
		}
		if fatal == nil && i < len(results) {
			r.done <- inkling2BatchRowResult(results[i], c.width)
			continue
		}
		err := fatal
		if err == nil {
			err = fmt.Errorf("inkling2 batch: %d results for %d rows", len(results), len(group))
		}
		r.done <- inkling2QueuedResult{err: err}
	}
	return true
}
