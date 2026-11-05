package routerfront

import (
	"context"
	"time"
)

const (
	OutcomePromoted          = "promoted"
	OutcomeAbortedResumed    = "aborted_resumed"
	OutcomeAbortedHeldFailed = "aborted_held_failed"
)

type SwapPlan struct {
	New string
	// Generation, when set, is the token the successor must echo in
	// GenerationHeader on both probes before it is promoted.
	Generation                                string
	HealthPath, ReadyPath                     string
	HealthTimeout, ReadyTimeout, DrainTimeout time.Duration
	StopOld                                   func(ctx context.Context) error
	OldAlive                                  func() bool
}

type SwapReceipt struct {
	Old, New       string
	Outcome        string
	HoldMS         int64
	InFlightAtHold int
	HeldRequests   int
	DrainedClean   bool
	Err            string
}

func (p *SwapPlan) defaults() {
	if p.HealthPath == "" {
		p.HealthPath = "/healthz"
	}
	if p.ReadyPath == "" {
		p.ReadyPath = "/readyz"
	}
	if p.HealthTimeout <= 0 {
		p.HealthTimeout = 10 * time.Second
	}
	if p.ReadyTimeout <= 0 {
		p.ReadyTimeout = 30 * time.Second
	}
	if p.DrainTimeout <= 0 {
		p.DrainTimeout = 30 * time.Second
	}
}

// Swap moves traffic from the active generation to p.New. The successor must
// pass its health probe before any hold, so a dead successor never stalls
// clients; readiness is probed only after StopOld because the successor cannot
// become ready while the old generation still owns the shared lock.
func (f *Front) Swap(ctx context.Context, p SwapPlan) (SwapReceipt, error) {
	f.swapMu.Lock()
	defer f.swapMu.Unlock()
	p.defaults()
	old := f.Active()
	rec := SwapReceipt{Old: old, New: p.New}
	fail := func(outcome string, err error) (SwapReceipt, error) {
		rec.Outcome = outcome
		rec.Err = err.Error()
		return rec, err
	}

	hctx, cancel := context.WithTimeout(ctx, p.HealthTimeout)
	err := f.ProbeReadyGeneration(hctx, p.New, p.HealthPath, p.Generation)
	cancel()
	if err != nil {
		return fail(OutcomeAbortedResumed, err)
	}

	start := time.Now()
	f.Hold()
	rec.InFlightAtHold = f.InFlight(old)
	if old != "" {
		dctx, cancel := context.WithTimeout(ctx, p.DrainTimeout)
		rec.DrainedClean = f.WaitIdle(dctx, old) == nil
		cancel()
	} else {
		rec.DrainedClean = true
	}

	var stopErr error
	if p.StopOld != nil {
		stopErr = p.StopOld(ctx)
		if stopErr != nil {
			f.opts.Logf("routerfront: stop old generation %s: %v", old, stopErr)
		}
	}

	rctx, cancel := context.WithTimeout(ctx, p.ReadyTimeout)
	err = f.ProbeReadyGeneration(rctx, p.New, p.ReadyPath, p.Generation)
	cancel()
	if err != nil {
		rec.HoldMS = time.Since(start).Milliseconds()
		if old != "" && p.OldAlive != nil && p.OldAlive() {
			rec.HeldRequests = f.resume()
			return fail(OutcomeAbortedResumed, err)
		}
		f.mu.Lock()
		f.failed = true
		rec.HeldRequests = f.heldTotal
		f.mu.Unlock()
		return fail(OutcomeAbortedHeldFailed, err)
	}

	n, err := f.promote(p.New, p.Generation)
	rec.HoldMS = time.Since(start).Milliseconds()
	if err != nil {
		return fail(OutcomeAbortedHeldFailed, err)
	}
	rec.HeldRequests = n
	rec.Outcome = OutcomePromoted
	if stopErr != nil {
		rec.Err = stopErr.Error()
	}
	return rec, nil
}
