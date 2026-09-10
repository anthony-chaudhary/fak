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
	}
	out := <-req.result
	<-req.receiptReady
	out.result.batchReceipt = req.receipt
	return out.result, out.err
}

func (p *InKernelPlanner) drainCoalescedGenerates() {
	for {
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
	}
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
