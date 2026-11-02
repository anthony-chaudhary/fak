package agent

import (
	"context"
	"sync"
)

// inKernelDeviceGate owns the single forward slot shared by ordinary requests
// and decode cohorts. Its zero value is ready for use. Like a mutex, it must not
// be copied after first use. A canceled waiter never launches a helper goroutine
// that could acquire the slot later or release somebody else's forward.
type inKernelDeviceGate struct {
	once sync.Once
	slot chan struct{}
}

func (g *inKernelDeviceGate) init() {
	g.once.Do(func() { g.slot = make(chan struct{}, 1) })
}

// Lock acquires an unconditional device owner.
func (g *inKernelDeviceGate) Lock() {
	g.init()
	g.slot <- struct{}{}
}

func (g *inKernelDeviceGate) LockContext(ctx context.Context) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	g.init()
	select {
	case <-ctx.Done():
		return ctx.Err()
	case g.slot <- struct{}{}:
		// Cancellation and a free slot can become ready together. Return only
		// our newly acquired slot before reporting cancellation in that race.
		if err := ctx.Err(); err != nil {
			g.Unlock()
			return err
		}
		return nil
	}
}

// LockCohort retires canceled selected requests before their run closures start,
// while preserving the ordered live subset. A nonempty return owns the gate;
// an empty return owns nothing. Started requests retain the coordinator's usual
// session-cleanup boundary. The drainer supplies a bounded selection.
func (g *inKernelDeviceGate) LockCohort(cohort []*inKernelCoalesceRequest) []*inKernelCoalesceRequest {
	pending := append([]*inKernelCoalesceRequest(nil), cohort...)
	prune := func() {
		kept := pending[:0]
		for _, req := range pending {
			if err := req.ctx.Err(); err != nil {
				// No run closure owns this request yet. Complete its cancellation
				// without claiming a device operation or a session-close receipt.
				req.result <- inKernelCoalesceResult{err: err}
				close(req.done)
				req.publishReceipt()
				continue
			}
			kept = append(kept, req)
		}
		clear(pending[len(kept):])
		pending = kept
	}
	for {
		prune()
		if len(pending) == 0 {
			return nil
		}
		acquired := func() bool {
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			stops := make([]func() bool, 0, len(pending))
			defer func() {
				for _, stop := range stops {
					stop()
				}
			}()
			for _, req := range pending {
				if req.ctx.Done() != nil {
					// Every callback captures this attempt's cancel function only.
					// Even a callback racing stop cannot affect a later attempt,
					// publish a request, or release somebody else's device owner.
					stops = append(stops, context.AfterFunc(req.ctx, cancel))
				}
			}
			return g.LockContext(ctx) == nil
		}()
		if !acquired {
			continue
		}
		// A context may be canceled before its callback runs. Recheck after
		// grant; only successful acquisition permits releasing this slot.
		prune()
		if len(pending) == 0 {
			g.Unlock()
		}
		return pending
	}
}

func (g *inKernelDeviceGate) Unlock() {
	g.init()
	select {
	case <-g.slot:
	default:
		panic("agent: unlock of unlocked device gate")
	}
}
