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

// Lock retains the cohort owner's existing non-cancelable ownership boundary.
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

func (g *inKernelDeviceGate) Unlock() {
	g.init()
	select {
	case <-g.slot:
	default:
		panic("agent: unlock of unlocked device gate")
	}
}
