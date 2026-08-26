package agent

import (
	"context"
	"runtime"
	"sync/atomic"
)

const inKernelDecodeCohortMax = 8

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
}

type inKernelCoalesceContextKey struct{}

func (p *InKernelPlanner) coalescesQwenDecode() bool {
	return p != nil && p.batchDecode && p.q4k && p.m != nil && p.m.Cfg.IsQwen35Hybrid()
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
		inKernelDecodeLanesBatched(context.Background(), lanes, p.m, p.quant)
	}()
	for _, req := range prepared {
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
		inKernelDecodeLanesBatched(ctx, []*decodeLane{lane}, lane.s.M, lane.s.Quant)
		return true, lane.err
	}
	req.prepared <- lane
	return true, <-req.proceed
}
