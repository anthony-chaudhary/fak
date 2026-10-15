package model

import (
	"context"
	"fmt"
	"sync"
	"time"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// V41BatchEligibility retains model weights for one request, including ordinary
// fallback. Eligibility is advisory; construction and StepV41Batch re-admit
// their actual state. It is neither a backend owner nor hardware qualification.
type V41BatchEligibility struct {
	model    *Model
	eligible bool
	once     sync.Once
}

func (e *V41BatchEligibility) Eligible() bool { return e != nil && e.eligible }

// Finish must be deferred directly at a named-error request boundary. Releasing
// the final pin can retire device weights, so all Sessions, queue receipts and
// planner/backend locks must have finished first. A primary error or panic wins
// over retirement failure; an otherwise unhandled cleanup panic still escapes.
func (e *V41BatchEligibility) Finish(err *error) {
	primary := recover()
	finishModelCleanup(primary, err, func() {
		if e != nil {
			e.once.Do(e.model.releaseWeightSession)
		}
	})
}

func finishModelCleanup(primary any, err *error, cleanup func()) {
	var secondary any
	func() {
		defer func() { secondary = recover() }()
		cleanup()
	}()
	if primary != nil {
		panic(primary)
	}
	if secondary != nil && (err == nil || *err == nil) {
		panic(secondary)
	}
}

// AcquireV41BatchEligibility pins first, copies metadata under q8Mu, then
// probes under the provider's shared recording-domain owner. Neither lifetime
// nor metadata locks span provider callbacks. A busy owner waits with ctx;
// cancellation never turns into unsynchronized fallback.
func (m *Model) AcquireV41BatchEligibility(ctx context.Context, be compute.Backend, maxRows int) (eligibility *V41BatchEligibility, err error) {
	if !m.holdModelWeights() {
		return nil, &ModelWeightsClosedError{}
	}
	e := &V41BatchEligibility{model: m}
	defer func() {
		if eligibility == nil {
			// No eligibility was transferred. Preserve the acquisition unwind
			// or returned error while suppressing only rollback cleanup panics.
			defer func() { _ = recover() }()
			e.once.Do(m.releaseWeightSession)
		}
	}()
	if err = ctx.Err(); err != nil {
		return nil, err
	}
	if !m.Cfg.IsDeepSeekV41() || be == nil || maxRows < 2 || maxRows > v41SharedHeadMaxLanes {
		return e, nil
	}
	layout, valid := m.v41HeadLayoutSnapshot("", 0, 0, true)
	provider, hasOwner := be.(compute.ForwardExecutionOwnerBackend)
	if !hasOwner {
		return e, nil
	}
	owner := provider.ForwardExecutionOwner()
	if owner == nil {
		return nil, fmt.Errorf("model: backend advertises a nil execution owner")
	}
	if !valid || (layout.dtype != compute.F32 && layout.dtype != compute.Q2_K && layout.dtype != compute.Q6_K) {
		return e, nil
	}
	capability, ok := be.(v41SharedHeadBackend)
	if !ok {
		return e, nil
	}
	for !owner.TryLock() {
		timer := time.NewTimer(time.Millisecond)
		select {
		case <-ctx.Done():
			timer.Stop()
			return nil, ctx.Err()
		case <-timer.C:
		}
	}
	// Registered after failure cleanup: the owner always unwinds before the
	// final request pin can retire model/device residency.
	defer owner.Unlock()
	if err = ctx.Err(); err != nil {
		return nil, err
	}
	if !be.Caps().DeviceMemory {
		return e, nil
	}
	if _, ok := v41HeadLayoutSupported(be, layout); ok {
		e.eligible = capability.V41SharedHeadAvailable(layout.dtype, layout.out, layout.in, maxRows)
	}
	return e, nil
}
