package agent

import (
	"context"
	"reflect"
	"runtime"
	"sync"
	"sync/atomic"
	"time"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/enginestep"
	"github.com/anthony-chaudhary/fak/internal/model"
)

const inKernelDecodeCohortMax = 8

type inKernelCoalesceKind uint8

const (
	inKernelCoalesceQwen inKernelCoalesceKind = iota
	inKernelCoalesceV41
)

var qwenSharedReceiptProbe struct {
	sync.Mutex
	fn func(*model.BatchSession, []int, []bool) ([][]float32, int, int64, bool)
}

func installQwenSharedReceiptProbeForTest(fn func(*model.BatchSession, []int, []bool) ([][]float32, int, int64, bool)) func() {
	qwenSharedReceiptProbe.Lock()
	old := qwenSharedReceiptProbe.fn
	qwenSharedReceiptProbe.fn = fn
	qwenSharedReceiptProbe.Unlock()
	return func() {
		qwenSharedReceiptProbe.Lock()
		qwenSharedReceiptProbe.fn = old
		qwenSharedReceiptProbe.Unlock()
	}
}

func runQwenSharedReceiptProbe(bs *model.BatchSession, ids []int, active []bool) ([][]float32, int, int64, bool) {
	qwenSharedReceiptProbe.Lock()
	fn := qwenSharedReceiptProbe.fn
	qwenSharedReceiptProbe.Unlock()
	if fn == nil {
		return nil, 0, 0, false
	}
	return fn(bs, ids, active)
}

// BatchDecodeEnabled reports whether this planner is wired for the continuous-batch
// decode path (the coalescer gate coalescesQwenDecode builds on). It is a readback
// seam for tests and serve-reachability receipts: operators can confirm the turnkey
// fan-out (#1590) is routed through the batched forward rather than serialized on
// devMu. It does not mutate the planner.
func (p *InKernelPlanner) BatchDecodeEnabled() bool {
	return p != nil && p.batchDecode
}

// AdmitCoalescedDecodeForTest relaxes the DEVICE half of coalescesQwenDecode
// (metal/q4k) so a cross-package test can drive the REAL coalescer over a synthetic
// hybrid model on the CPU dense forward, with no physical device. It enables the
// batchDecode wiring and returns a restore func that puts every mutated gate back.
//
// It is deliberately a test seam: production turns coalescing on structurally via
// InKernelPlannerConfig.BatchDecode (cmd/fak/up.go) and the device gates stay tied to
// the real Metal/Q4_K load, so a synthetic model can never be mistaken for a device
// forward in a served process.
func (p *InKernelPlanner) AdmitCoalescedDecodeForTest() func() {
	if p == nil {
		return func() {}
	}
	prevBatch, prevMetal, prevQ4K := p.batchDecode, p.metal, p.q4k
	p.batchDecode = true
	p.metal = true
	p.q4k = true
	return func() {
		p.batchDecode = prevBatch
		p.metal = prevMetal
		p.q4k = prevQ4K
	}
}

// CoalesceReadyLenForTest returns the number of requests currently queued for the
// next coalesced cohort. A cross-package test polls this to release the cohort leader
// once all N fan-out requests have arrived, making coalescing deterministic instead of
// racy. It is a read-only test seam.
func (p *InKernelPlanner) CoalesceReadyLenForTest() int {
	if p == nil {
		return 0
	}
	p.coalesceMu.Lock()
	defer p.coalesceMu.Unlock()
	return len(p.coalesceReady)
}

// SetCoalesceReadyHookForTest installs a barrier the cohort LEADER calls after it
// becomes the drainer but before it drains. A test blocks here until all N expected
// requests have arrived, so a same-prefix fan-out coalesces deterministically instead
// of depending on the leader's runtime.Gosched racing N request goroutines. It
// returns a restore func that clears the hook.
func (p *InKernelPlanner) SetCoalesceReadyHookForTest(hook func()) func() {
	if p == nil {
		return func() {}
	}
	p.coalesceMu.Lock()
	prev := p.coalesceReadyHook
	p.coalesceReadyHook = hook
	p.coalesceMu.Unlock()
	return func() {
		p.coalesceMu.Lock()
		p.coalesceReadyHook = prev
		p.coalesceMu.Unlock()
	}
}

type InKernelBatchReceipt struct {
	CohortID      uint64 `json:"cohort_id"`
	CohortSize    int    `json:"cohort_size"`
	SharedPanels  int    `json:"shared_panels"`
	SharedRows    int    `json:"shared_rows,omitempty"`
	SharedMACs    int64  `json:"shared_macs"`
	SessionCloses uint32 `json:"session_closes"`
}
type inKernelCoalesceResult struct {
	result inKernelGenerateResult
	err    error
}

type inKernelCoalesceRequest struct {
	kind         inKernelCoalesceKind
	ctx          context.Context
	run          func(context.Context) (inKernelGenerateResult, error)
	prepared     chan *decodeLane
	proceed      chan error
	result       chan inKernelCoalesceResult
	done         chan struct{}
	drain        chan struct{}
	decodePass   atomic.Uint32
	receipt      InKernelBatchReceipt
	closes       atomic.Uint32
	receiptReady chan struct{}
	receiptOnce  sync.Once
	terminalErr  error
	panicValue   any
	enqueued     time.Time // coalescer arrival, for the cohort_wait phase
}

type inKernelCoalesceContextKey struct{}

// coalescesQwenDecode admits the Metal host route or a backend with the exact
// independent Qwen batch contract. Device serialization alone does not qualify
// a backend for cohort execution: unsupported backends retain the ordinary
// whole-turn devMu route, including startup coherence probes (fak#14118).
func (p *InKernelPlanner) coalescesQwenDecode() bool {
	if p == nil || !p.batchDecode || !p.q4k || p.m == nil || !p.m.Cfg.IsQwen35Hybrid() {
		return false
	}
	if p.backend == nil {
		return p.metal
	}
	batch, ok := p.backend.(compute.Qwen35IndependentDecodeBatchBackend)
	if !ok || (reflect.ValueOf(batch).Kind() == reflect.Pointer && reflect.ValueOf(batch).IsNil()) {
		return false
	}
	return batch.Qwen35IndependentDecodeBatchPath() == compute.Qwen35IndependentDecodeBatchPath
}

func (p *InKernelPlanner) runCoalescedGenerate(ctx context.Context, run func(context.Context) (inKernelGenerateResult, error)) (inKernelGenerateResult, error) {
	return p.runCoalescedGenerateKind(ctx, run, inKernelCoalesceQwen)
}

func (p *InKernelPlanner) runCoalescedGenerateKind(ctx context.Context, run func(context.Context) (inKernelGenerateResult, error), kind inKernelCoalesceKind) (inKernelGenerateResult, error) {
	req := &inKernelCoalesceRequest{
		kind:         kind,
		ctx:          ctx,
		run:          run,
		prepared:     make(chan *decodeLane, 1),
		proceed:      make(chan error, 1),
		result:       make(chan inKernelCoalesceResult, 1),
		done:         make(chan struct{}),
		drain:        make(chan struct{}, 1),
		receiptReady: make(chan struct{}),
		enqueued:     time.Now(),
	}
	p.coalesceMu.Lock()
	p.coalesceReady = append(p.coalesceReady, req)
	enginestep.Default.SetQueueDepth(len(p.coalesceReady))
	leader := !p.coalesceRunning
	if leader {
		p.coalesceRunning = true
	}
	hook := p.coalesceReadyHook
	p.coalesceMu.Unlock()

	if leader {
		var hookPanic any
		func() {
			defer func() { hookPanic = recover() }()
			if hook != nil {
				hook()
			} else {
				runtime.Gosched()
			}
		}()
		if hookPanic != nil {
			p.withdrawCoalescedRequest(req, true)
			abortErr := errInKernelQwenCohortAborted
			if kind == inKernelCoalesceV41 {
				abortErr = errInKernelV41CohortAborted
			}
			req.result <- inKernelCoalesceResult{err: abortErr}
			close(req.done)
			req.terminalErr = abortErr
			req.panicValue = hookPanic
			req.publishReceipt()
			out := <-req.result
			return finishCoalescedRequest(req, out)
		}
		go p.drainCoalescedGenerates()
	} else {
		select {
		case <-req.drain:
			if ctx.Err() != nil && p.withdrawCoalescedRequest(req, true) {
				return inKernelGenerateResult{}, ctx.Err()
			}
			go p.drainCoalescedGenerates()
		case out := <-req.result:
			return finishCoalescedRequest(req, out)
		case <-ctx.Done():
			if p.withdrawCoalescedRequest(req, false) {
				return inKernelGenerateResult{}, ctx.Err()
			}
			// Already selected: the run closure may own a session or shared
			// forward state. Wait for its result AND cleanup receipt below.
		}
	}
	out := <-req.result
	return finishCoalescedRequest(req, out)
}

func finishCoalescedRequest(req *inKernelCoalesceRequest, out inKernelCoalesceResult) (inKernelGenerateResult, error) {
	<-req.receiptReady
	out.result.batchReceipt = req.receipt
	if req.terminalErr != nil {
		out.err = req.terminalErr
	}
	if req.panicValue != nil {
		panic(req.panicValue)
	}
	return out.result, out.err
}

func (req *inKernelCoalesceRequest) publishReceipt() {
	req.receiptOnce.Do(func() { close(req.receiptReady) })
}

// withdrawCoalescedRequest removes only a still-queued follower. Selection and
// withdrawal share coalesceMu, so success proves its run closure cannot start.
// ownsBaton is true only when this caller just received req.drain. Otherwise a
// simultaneous cancellation/handoff may have left the baton buffered there.
func (p *InKernelPlanner) withdrawCoalescedRequest(req *inKernelCoalesceRequest, ownsBaton bool) bool {
	p.coalesceMu.Lock()
	defer p.coalesceMu.Unlock()
	for i, pending := range p.coalesceReady {
		if pending != req {
			continue
		}
		copy(p.coalesceReady[i:], p.coalesceReady[i+1:])
		last := len(p.coalesceReady) - 1
		p.coalesceReady[last] = nil
		p.coalesceReady = p.coalesceReady[:last]
		enginestep.Default.SetQueueDepth(len(p.coalesceReady))
		if !ownsBaton {
			select {
			case <-req.drain:
				ownsBaton = true
			default:
			}
		}
		if ownsBaton {
			if len(p.coalesceReady) == 0 {
				p.coalesceRunning = false
			} else {
				p.coalesceReady[0].drain <- struct{}{}
			}
		}
		// Without a baton, the active/initial drainer still owns handoff,
		// even when withdrawal has temporarily made its pending queue empty.
		return true
	}
	return false
}

func (p *InKernelPlanner) drainCoalescedGenerates() {
	p.coalesceMu.Lock()
	n := len(p.coalesceReady)
	if n == 0 {
		p.coalesceRunning = false
		p.coalesceMu.Unlock()
		return
	}
	kind := p.coalesceReady[0].kind
	limit := inKernelDecodeCohortMax
	if kind == inKernelCoalesceV41 {
		limit = 4
	}
	if n > limit {
		n = limit
	}
	for i := 1; i < n; i++ {
		if p.coalesceReady[i].kind != kind {
			n = i // Never skip a different kind or reorder its FIFO position.
			break
		}
	}
	cohort := append([]*inKernelCoalesceRequest(nil), p.coalesceReady[:n]...)
	p.coalesceReady = p.coalesceReady[n:]
	enginestep.Default.SetQueueDepth(len(p.coalesceReady))
	p.coalesceMu.Unlock()
	for _, req := range cohort {
		enginestep.Default.ObservePhase(enginestep.PhaseCohortWait, time.Since(req.enqueued))
	}
	if kind == inKernelCoalesceV41 {
		fatal := p.runV41DecodeCohortCore(cohort, inKernelDecodeV41Lanes)
		p.finishCoalescedCohort()
		if fatal != nil && fatal.request != nil {
			fatal.request.publishReceipt()
		}
		return
	}
	final := p.runDecodeCohort(cohort)
	p.finishCoalescedCohort()
	for _, req := range final {
		req.publishReceipt()
	}
}

func (p *InKernelPlanner) finishCoalescedCohort() {
	p.coalesceMu.Lock()
	if len(p.coalesceReady) == 0 {
		p.coalesceRunning = false
	} else {
		// Keep ownership live across the handoff so arrivals cannot elect a second
		// drainer. Only the first unprocessed request receives the one-slot baton;
		// it drains the next cohort in its own Complete stack.
		p.coalesceReady[0].drain <- struct{}{}
	}
	p.coalesceMu.Unlock()
}

func (p *InKernelPlanner) dequeueV41CoalescedRequests(limit int) []*inKernelCoalesceRequest {
	return p.dequeueCoalescedRequests(inKernelCoalesceV41, limit)
}

func (p *InKernelPlanner) dequeueCoalescedRequests(kind inKernelCoalesceKind, limit int) []*inKernelCoalesceRequest {
	if limit <= 0 {
		return nil
	}
	p.coalesceMu.Lock()
	count := len(p.coalesceReady)
	if count > limit {
		count = limit
	}
	for index := 0; index < count; index++ {
		if p.coalesceReady[index].kind != kind {
			count = index
			break
		}
	}
	if count == 0 {
		p.coalesceMu.Unlock()
		return nil
	}
	requests := append([]*inKernelCoalesceRequest(nil), p.coalesceReady[:count]...)
	for index := 0; index < count; index++ {
		p.coalesceReady[index] = nil
	}
	p.coalesceReady = p.coalesceReady[count:]
	enginestep.Default.SetQueueDepth(len(p.coalesceReady))
	p.coalesceMu.Unlock()
	for _, request := range requests {
		enginestep.Default.ObservePhase(enginestep.PhaseCohortWait, time.Since(request.enqueued))
	}
	return requests
}

func coalescedDecode(ctx context.Context, lane *decodeLane) (bool, error) {
	req, ok := ctx.Value(inKernelCoalesceContextKey{}).(*inKernelCoalesceRequest)
	if ok {
		defer req.closes.Add(1)
	}
	if !ok {
		return false, nil
	}
	if req.decodePass.Add(1) > 1 {
		// The first coordinated pass may already have advanced this lane's KV/GDN
		// state before a later operation failed. Replaying Session.Step would apply
		// the accepted token twice, so cohort failures are terminal and fail closed.
		return true, lane.err
	}
	req.prepared <- lane
	return true, <-req.proceed
}
