package agent

import (
	"context"
	"sync"
	"sync/atomic"
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

// LockCohort abandons a not-yet-started cohort only when every participant is
// canceled. After acquisition, the coordinator retains its usual cleanup boundary.
// The drainer supplies a nonempty selection with each request's original context.
func (g *inKernelDeviceGate) LockCohort(cohort []*inKernelCoalesceRequest) error {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	var remaining atomic.Int32
	remaining.Store(int32(len(cohort)))
	stops := make([]func() bool, 0, len(cohort))
	defer func() {
		for _, stop := range stops {
			stop()
		}
	}()
	for _, req := range cohort {
		stops = append(stops, context.AfterFunc(req.ctx, func() {
			if remaining.Add(-1) == 0 {
				cancel()
			}
		}))
	}
	if err := g.LockContext(ctx); err != nil {
		return err
	}
	// AfterFunc callbacks may still be scheduled when the slot becomes free.
	// Check the participant contexts before admitting any run closure.
	for _, req := range cohort {
		if req.ctx.Err() == nil {
			return nil
		}
	}
	g.Unlock()
	return context.Canceled
}

func (g *inKernelDeviceGate) Unlock() {
	g.init()
	select {
	case <-g.slot:
	default:
		panic("agent: unlock of unlocked device gate")
	}
}
