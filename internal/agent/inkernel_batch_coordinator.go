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
	CohortID     uint64 `json:"cohort_id"`
	CohortSize   int    `json:"cohort_size"`
	SharedPanels int    `json:"shared_panels"`
	SharedMACs   int64  `json:"shared_macs"`
}
type inKernelCoalesceResult struct {
	result inKernelGenerateResult
	err    error
}

type inKernelCoalesceRequest struct {
	ctx        context.Context
	run        func(context.Context) (inKernelGenerateResult, error)
	prepared   chan *decodeLane
	proceed    chan error
	result     chan inKernelCoalesceResult
	done       chan struct{}
	decodePass atomic.Uint32
	receipt    InKernelBatchReceipt
}

type inKernelCoalesceContextKey struct{}

func (p *InKernelPlanner) coalescesQwenDecode() bool {
	return p != nil && p.batchDecode && p.q4k && p.metal && p.m != nil && p.m.Cfg.IsQwen35Hybrid()
}

func (p *InKernelPlanner) runCoalescedGenerate(ctx context.Context, run func(context.Context) (inKernelGenerateResult, error)) (inKernelGenerateResult, error) {
	req := &inKernelCoalesceRequest{
		ctx:      ctx,
		run:      run,
		prepared: make(chan *decodeLane, 1),
		proceed:  make(chan error, 1),
		result:   make(chan inKernelCoalesceResult, 1),
		done:     make(chan struct{}),
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
	cohortID := p.coalesceCohortID.Add(1)
	receipt := InKernelBatchReceipt{CohortID: cohortID, CohortSize: len(lanes), SharedPanels: panels, SharedMACs: macs}
	for _, req := range prepared {
		req.receipt = receipt
		req.proceed <- decodeErr
	}
	for _, req := range cohort {
		<-req.done
	}
}

func coalescedDecode(ctx context.Context, lane *decodeLane) (bool, error) {
	req, ok := ctx.Value(inKernelCoalesceContextKey{}).(*inKernelCoalesceRequest)
	if !ok {
		return false, nil
	}
	if req.decodePass.Add(1) > 1 {
		// OOM retry stays under the coordinator's device lock, but retries this
		// request alone after the failed cohort forward has returned.
		_, _ = inKernelDecodeLanesBatched(ctx, []*decodeLane{lane}, lane.s.M, lane.s.Quant)
		return true, lane.err
	}
	req.prepared <- lane
	return true, <-req.proceed
}
