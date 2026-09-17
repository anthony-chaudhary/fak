package agent

import (
	"context"
	"runtime"
	"sync"
	"sync/atomic"

	"github.com/anthony-chaudhary/fak/internal/model"
)

const inKernelDecodeCohortMax = 8

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
	SharedMACs    int64  `json:"shared_macs"`
	SessionCloses uint32 `json:"session_closes"`
}
type inKernelCoalesceResult struct {
	result inKernelGenerateResult
	err    error
}

type inKernelCoalesceRequest struct {
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
}

type inKernelCoalesceContextKey struct{}

func (p *InKernelPlanner) coalescesQwenDecode() bool {
	return p != nil && p.batchDecode && p.q4k && p.metal && p.m != nil && p.m.Cfg.IsQwen35Hybrid()
}

func (p *InKernelPlanner) runCoalescedGenerate(ctx context.Context, run func(context.Context) (inKernelGenerateResult, error)) (inKernelGenerateResult, error) {
	req := &inKernelCoalesceRequest{
		ctx:          ctx,
		run:          run,
		prepared:     make(chan *decodeLane, 1),
		proceed:      make(chan error, 1),
		result:       make(chan inKernelCoalesceResult, 1),
		done:         make(chan struct{}),
		drain:        make(chan struct{}, 1),
		receiptReady: make(chan struct{}),
	}
	p.coalesceMu.Lock()
	p.coalesceReady = append(p.coalesceReady, req)
	leader := !p.coalesceRunning
	if leader {
		p.coalesceRunning = true
	}
	hook := p.coalesceReadyHook
	p.coalesceMu.Unlock()

	if leader {
		if hook != nil {
			hook()
		} else {
			runtime.Gosched()
		}
		p.drainCoalescedGenerates()
	} else {
		select {
		case <-req.drain:
			p.drainCoalescedGenerates()
		case out := <-req.result:
			<-req.receiptReady
			out.result.batchReceipt = req.receipt
			return out.result, out.err
		}
	}
	out := <-req.result
	<-req.receiptReady
	out.result.batchReceipt = req.receipt
	return out.result, out.err
}

func (p *InKernelPlanner) drainCoalescedGenerates() {
	p.coalesceMu.Lock()
	n := len(p.coalesceReady)
	if n == 0 {
		p.coalesceRunning = false
		p.coalesceMu.Unlock()
		return
	}
	if n > inKernelDecodeCohortMax {
		n = inKernelDecodeCohortMax
	}
	cohort := append([]*inKernelCoalesceRequest(nil), p.coalesceReady[:n]...)
	p.coalesceReady = p.coalesceReady[n:]
	p.coalesceMu.Unlock()
	p.runDecodeCohort(cohort)

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

func (p *InKernelPlanner) runDecodeCohort(cohort []*inKernelCoalesceRequest) {
	p.devMu.Lock()
	defer p.devMu.Unlock()

	lanes := make([]*decodeLane, 0, len(cohort))
	prepared := make([]*inKernelCoalesceRequest, 0, len(cohort))
	for _, req := range cohort {
		go func(r *inKernelCoalesceRequest) {
			runCtx := context.WithValue(r.ctx, inKernelCoalesceContextKey{}, r)
			res, err := r.run(runCtx)
			close(r.done)
			select {
			case r.prepared <- nil:
			default:
			}
			r.result <- inKernelCoalesceResult{result: res, err: err}
		}(req)
		if lane := <-req.prepared; lane != nil {
			lane.ctx = req.ctx
			lanes = append(lanes, lane)
			prepared = append(prepared, req)
		}
	}

	var decodeErr error
	var panels int
	var macs int64
	func() {
		defer func() {
			if r := recover(); r != nil {
				if err, ok := recoverDevicePanic(r); ok {
					decodeErr = err
					return
				}
				panic(r)
			}
		}()
		if p.coalesceBatchHook != nil {
			p.coalesceBatchHook(len(lanes))
		}
		panels, macs = inKernelDecodeLanesBatched(context.Background(), lanes, p.m, p.quant)
		if p.coalesceSharedHook != nil {
			p.coalesceSharedHook(panels, macs)
		}
	}()
	for _, req := range prepared {
		req.proceed <- decodeErr
	}
	for _, req := range cohort {
		<-req.done
	}
	cohortID := p.coalesceCohortID.Add(1)
	receipt := InKernelBatchReceipt{CohortID: cohortID, CohortSize: len(lanes), SharedPanels: panels, SharedMACs: macs}
	for _, req := range prepared {
		receipt.SessionCloses += req.closes.Load()
	}
	for _, req := range cohort {
		req.receipt = receipt
		close(req.receiptReady)
	}
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
