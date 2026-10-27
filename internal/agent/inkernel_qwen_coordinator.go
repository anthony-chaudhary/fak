package agent

import (
	"context"
	"errors"
	"time"

	"github.com/anthony-chaudhary/fak/internal/enginestep"
)

var errInKernelQwenCohortAborted = errors.New("agent: Qwen cohort aborted; no request replay permitted")

type inKernelQwenRequestState struct {
	req   *inKernelCoalesceRequest
	lane  *decodeLane
	panic any
}

// runDecodeCohort retains the device gate across native forwards and late
// preparation. A completed iteration is the only admission boundary: new requests
// run their ordinary scoped lookup and suffix prefill while existing sessions are
// paused, never concurrently with their shared device forward. Each request still
// owns its own session and its Complete-level memory reservations.
//
// The lifecycle follows the V4.1 coordinator: retire and join cleanup before slot
// reuse, publish early results independently, and preserve FIFO across model kinds.
// Final receipts are returned to the drainer for publication after gate release
// and baton handoff, so an idle completion cannot leave ownership behind it.
func (p *InKernelPlanner) runDecodeCohort(cohort []*inKernelCoalesceRequest) (final []*inKernelCoalesceRequest) {
	deviceWait := time.Now()
	p.devMu.Lock()
	defer p.devMu.Unlock()
	enginestep.Default.ObservePhase(enginestep.PhaseDeviceWait, time.Since(deviceWait))

	cohortID := p.coalesceCohortID.Add(1)
	active := make([]*inKernelQwenRequestState, 0, inKernelDecodeCohortMax)
	var joinedCloses uint32
	var panels, peak int
	var macs int64
	var terminal error
	var primary any
	var cause *inKernelCoalesceRequest
	setPrimary := func(value any, request *inKernelCoalesceRequest) {
		if primary != nil {
			return
		}
		primary, cause, terminal = value, request, errInKernelQwenCohortAborted
		if cause == nil && len(active) > 0 {
			cause = active[0].req
		}
		if cause == nil && len(final) > 0 {
			cause = final[0]
		}
	}
	publish := func(state *inKernelQwenRequestState) {
		request := state.req
		request.receipt.CohortID = cohortID
		request.receipt.SessionCloses = joinedCloses
		if len(active) == 0 || primary != nil {
			final = append(final, request)
		} else {
			request.publishReceipt()
		}
	}
	prepare := func(requests []*inKernelCoalesceRequest) {
		for _, request := range requests {
			state := &inKernelQwenRequestState{req: request}
			if primary != nil {
				request.result <- inKernelCoalesceResult{err: errInKernelQwenCohortAborted}
				close(request.done)
				publish(state)
				continue
			}
			go func() {
				var out inKernelCoalesceResult
				defer func() {
					if value := recover(); value != nil {
						if err, known := recoverDevicePanic(value); known {
							out.err = err
						} else {
							state.panic, out.err = value, errInKernelQwenCohortAborted
						}
					}
					request.result <- out
					close(request.done)
					select {
					case request.prepared <- nil:
					default:
					}
				}()
				runCtx := context.WithValue(request.ctx, inKernelCoalesceContextKey{}, request)
				out.result, out.err = request.run(runCtx)
			}()
			if lane := <-request.prepared; lane != nil {
				state.lane, lane.ctx = lane, request.ctx
				active = append(active, state)
			} else {
				<-request.done
				joinedCloses += request.closes.Load()
				if state.panic != nil {
					setPrimary(state.panic, request)
				}
				publish(state)
			}
		}
	}
	retire := func(all bool) bool {
		finished := make([]*inKernelQwenRequestState, 0, len(active))
		kept := active[:0]
		for _, state := range active {
			lane := state.lane
			if all || lane.done || lane.gen >= lane.maxNew {
				if all && lane.err == nil && !lane.done && lane.gen < lane.maxNew {
					lane.err, lane.done = terminal, true
				}
				finished = append(finished, state)
			} else {
				kept = append(kept, state)
			}
		}
		clear(active[len(kept):])
		active = kept
		for _, state := range finished {
			state.req.proceed <- state.lane.err
		}
		for _, state := range finished {
			<-state.req.done // Includes continuation admission and Session.Close.
			joinedCloses += state.req.closes.Load()
			if state.panic != nil {
				setPrimary(state.panic, state.req)
			}
		}
		for _, state := range finished {
			if primary != nil && state.req != cause {
				state.req.terminalErr = errInKernelQwenCohortAborted
			}
			publish(state)
		}
		return len(finished) != 0
	}

	func() {
		defer func() {
			if value := recover(); value != nil {
				if err, known := recoverDevicePanic(value); known {
					terminal = err
				} else {
					setPrimary(value, nil)
				}
			}
		}()
		prepare(cohort)
		if p.coalesceBatchHook != nil {
			p.coalesceBatchHook(len(active))
		}
		var iteration *inKernelDecodeIteration
		for len(active) > 0 && primary == nil {
			if iteration == nil {
				lanes := make([]*decodeLane, len(active))
				for index, state := range active {
					lanes[index] = state.lane
					state.req.receipt.CohortSize = max(state.req.receipt.CohortSize, len(active))
				}
				peak = max(peak, len(active))
				iteration = newInKernelDecodeIteration(lanes, p.m, p.quant)
			}
			beforePanels, beforeMACs := iteration.sharedPanels, iteration.sharedMACs
			iteration.step(context.Background())
			stepPanels := iteration.sharedPanels - beforePanels
			stepMACs := iteration.sharedMACs - beforeMACs
			panels, macs = panels+stepPanels, macs+stepMACs
			for index, state := range active {
				// The final sampled token is not forwarded. Its retiring lane
				// must not receive another request's shared-work credit.
				if iteration.active[index] {
					state.req.receipt.SharedPanels += stepPanels
					state.req.receipt.SharedMACs += stepMACs
				}
			}
			changed := retire(false)
			if primary != nil || len(active) == 0 {
				break // An empty episode yields to the ordinary next-cohort handoff.
			}
			// One bounded queue selection per completed step prevents repeated
			// failed preparations from starving the still-running decode lanes.
			requests := p.dequeueCoalescedRequests(inKernelCoalesceQwen, inKernelDecodeCohortMax-len(active))
			if len(requests) > 0 {
				prepare(requests)
				changed = true
			}
			if changed {
				iteration = nil
			}
		}
		if p.coalesceSharedHook != nil {
			p.coalesceSharedHook(panels, macs)
		}
	}()
	enginestep.Default.ObserveCohort(peak)
	if len(active) > 0 {
		if terminal == nil {
			terminal = errInKernelQwenCohortAborted
		}
		retire(true)
	}
	for _, request := range final {
		if primary != nil {
			if request == cause {
				request.panicValue = primary
			} else {
				request.terminalErr = errInKernelQwenCohortAborted
			}
		}
	}
	return final
}
