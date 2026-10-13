package agent

import (
	"context"
	"errors"
	"fmt"
	"time"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/enginestep"
	"github.com/anthony-chaudhary/fak/internal/model"
)

var errInKernelV41CohortAborted = errors.New("agent: V4.1 cohort aborted; no request replay permitted")

// runV41CoalescedGenerate is a dormant enqueue boundary, not a Complete route.
// These structural checks are advisory: they do not query or promise support
// for the actual head dtype/layout. StepV41Batch alone admits prepared Sessions
// and acquires the backend owner. A future Complete admission must additionally
// query exact model eligibility without treating that query as runtime success.
func (p *InKernelPlanner) runV41CoalescedGenerate(ctx context.Context, run func(context.Context) (inKernelGenerateResult, error)) (inKernelGenerateResult, error) {
	decline := func() (inKernelGenerateResult, error) {
		return inKernelGenerateResult{}, fmt.Errorf("%w: planner lacks V4.1 cohort prerequisites", model.ErrV41BatchUnavailable)
	}
	if p == nil || !p.batchDecode || p.m == nil || !p.m.Cfg.IsDeepSeekV41() || p.backend == nil {
		return decline()
	}
	owner, ok := p.backend.(compute.ForwardExecutionOwnerBackend)
	if !ok || owner.ForwardExecutionOwner() == nil {
		return decline() // An implemented nil owner also fails closed.
	}
	if _, ok := p.backend.(interface {
		V41SharedHeadAvailable(compute.Dtype, int, int, int) bool
	}); !ok || !p.backend.Caps().DeviceMemory {
		return decline()
	}
	return p.runCoalescedGenerateKind(ctx, run, inKernelCoalesceV41)
}

type inKernelV41RequestState struct {
	req   *inKernelCoalesceRequest
	lane  *decodeLane
	panic any // Written by the request goroutine, read only after req.done.
}

// runV41DecodeCohort owns the planner gate, never the backend owner. Preparation,
// StepV41Batch, public Step fallback and request-owned Session.Close each use
// their own acquiring model boundaries. The decode argument is the real frozen
// driver in the queue dispatcher; private tests may substitute its operations.
func (p *InKernelPlanner) runV41DecodeCohort(cohort []*inKernelCoalesceRequest, decode func(context.Context, []*decodeLane, *model.Model) (inKernelV41DecodeReceipt, error)) {
	deviceWait := time.Now()
	p.devMu.Lock()
	defer p.devMu.Unlock()
	enginestep.Default.ObservePhase(enginestep.PhaseDeviceWait, time.Since(deviceWait))

	states := make([]inKernelV41RequestState, len(cohort))
	prepared := make([]*inKernelV41RequestState, 0, len(cohort))
	lanes := make([]*decodeLane, 0, len(cohort))
	var primary any
	started := 0
	for i, req := range cohort {
		state := &states[i]
		state.req = req
		go func(state *inKernelV41RequestState) {
			r := state.req
			var out inKernelCoalesceResult
			defer func() {
				if cause := recover(); cause != nil {
					// Close has unwound, but this boundary cannot distinguish a
					// cleanup panic from unrelated post-decode work. Never hide
					// an unknown panic merely because the lane already failed.
					if err, known := recoverDevicePanic(cause); known {
						out.err = err
					} else {
						state.panic, out.err = cause, errInKernelV41CohortAborted
					}
					if state.lane != nil && state.lane.err != nil {
						out.err = state.lane.err // Preserve its result, not suppress panic.
					}
				}
				r.result <- out
				close(r.done)
				select {
				case r.prepared <- nil:
				default:
				}
			}()
			// Preserve the original context and the same decodePass fence used
			// by admitDeviceOOMRetry. Never create a replacement request.
			runCtx := context.WithValue(r.ctx, inKernelCoalesceContextKey{}, r)
			out.result, out.err = r.run(runCtx)
		}(state)
		started++
		if lane := <-req.prepared; lane != nil {
			state.lane = lane
			lane.ctx = req.ctx
			prepared, lanes = append(prepared, state), append(lanes, lane)
		} else {
			<-req.done
			if state.panic != nil {
				primary = state.panic
				break // A fatal preparation cannot authorize more preparation.
			}
		}
	}
	for _, req := range cohort[started:] {
		req.result <- inKernelCoalesceResult{err: errInKernelV41CohortAborted}
		close(req.done)
	}

	var receipt inKernelV41DecodeReceipt
	var terminal error
	if primary == nil {
		func() {
			defer func() {
				if cause := recover(); cause != nil {
					if err, known := recoverDevicePanic(cause); known {
						terminal = err
					} else {
						primary = cause
					}
				}
			}()
			enginestep.Default.ObserveCohort(len(lanes))
			if p.coalesceBatchHook != nil {
				p.coalesceBatchHook(len(lanes))
			}
			receipt, terminal = decode(context.Background(), lanes, p.m)
			if p.coalesceSharedHook != nil {
				p.coalesceSharedHook(receipt.sharedPanels, 0) // No fabricated MAC estimate.
			}
		}()
	}
	for _, state := range prepared {
		lane := state.lane
		// The driver's terminal is diagnostic. Only an unfinished lane needs
		// an abort error; completed healthy peers retain their own success.
		if lane.err == nil && !lane.done && lane.gen < lane.maxNew {
			lane.err = terminal
			if lane.err == nil {
				lane.err = errInKernelV41CohortAborted
			}
			lane.done = true
		}
		state.req.proceed <- lane.err
	}
	for i, req := range cohort {
		<-req.done // Includes every run closure's deferred session teardown.
		if primary == nil && states[i].panic != nil {
			primary = states[i].panic
		}
	}
	result := InKernelBatchReceipt{
		CohortID: p.coalesceCohortID.Add(1), CohortSize: len(lanes),
		SharedPanels: receipt.sharedPanels, SharedRows: receipt.sharedRows,
	}
	for _, state := range prepared {
		result.SessionCloses += state.req.closes.Load()
	}
	for _, req := range cohort {
		req.receipt = result
		close(req.receiptReady)
	}
	if primary != nil {
		panic(primary) // Original identity, after all joins and receipt publication.
	}
}
