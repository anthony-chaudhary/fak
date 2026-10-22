package agent

import (
	"context"
	"errors"
	"fmt"
	"log"
	"time"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/enginestep"
	"github.com/anthony-chaudhary/fak/internal/model"
)

var errInKernelV41CohortAborted = errors.New("agent: V4.1 cohort aborted; no request replay permitted")

// runV41CoalescedGenerate validates the production V4.1 coordinator boundary.
// StepV41Batch still owns exact runtime admission and backend execution ownership.
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
	req           *inKernelCoalesceRequest
	lane          *decodeLane
	panic         any
	basePanels    int
	baseRows      int
	endPanels     int
	endRows       int
	endCloses     uint32
	peakActive    int
	participated  bool
	ended         bool
	receiptStored bool
}

type inKernelV41Fatal struct {
	value   any
	request *inKernelCoalesceRequest
}

func (p *InKernelPlanner) runV41DecodeCohort(cohort []*inKernelCoalesceRequest, decode func(context.Context, []*decodeLane, *model.Model) (inKernelV41DecodeReceipt, error)) {
	fatal := p.runV41DecodeCohortCore(cohort, decode)
	if fatal != nil {
		if fatal.request != nil {
			fatal.request.publishReceipt()
		}
		panic(fatal.value)
	}
}

func (p *InKernelPlanner) runV41DecodeCohortCore(cohort []*inKernelCoalesceRequest, decode func(context.Context, []*decodeLane, *model.Model) (inKernelV41DecodeReceipt, error)) *inKernelV41Fatal {
	deviceWait := time.Now()
	p.devMu.Lock()
	gateHeld := true
	defer func() {
		if gateHeld {
			p.devMu.Unlock()
		}
	}()
	enginestep.Default.ObservePhase(enginestep.PhaseDeviceWait, time.Since(deviceWait))

	cohortID := p.coalesceCohortID.Add(1)
	active := make([]*inKernelV41RequestState, 0, 4)
	byLane := make(map[*decodeLane]*inKernelV41RequestState, 4)
	sharedPanels := 0
	sharedRows := 0
	episodePeak := 0
	joinedCloses := uint32(0)
	var primary any
	var cause *inKernelV41RequestState
	setPrimary := func(value any, state *inKernelV41RequestState) {
		if value == nil || primary != nil {
			return
		}
		primary = value
		cause = state
		if cause == nil && len(active) > 0 {
			cause = active[0]
		}
	}
	receiptFor := func(state *inKernelV41RequestState) InKernelBatchReceipt {
		receipt := InKernelBatchReceipt{CohortID: cohortID, SessionCloses: state.endCloses}
		if state.participated {
			endPanels, endRows := sharedPanels, sharedRows
			if state.ended {
				endPanels, endRows = state.endPanels, state.endRows
			}
			receipt.CohortSize = state.peakActive
			receipt.SharedPanels = max(0, endPanels-state.basePanels)
			receipt.SharedRows = max(0, endRows-state.baseRows)
		}
		return receipt
	}
	publish := func(state *inKernelV41RequestState) {
		if state.receiptStored {
			return
		}
		state.req.receipt = receiptFor(state)
		state.receiptStored = true
		if primary != nil && state == cause {
			state.req.panicValue = primary
			return
		}
		state.req.publishReceipt()
	}
	removeActive := func(states []*inKernelV41RequestState) {
		retiring := make(map[*inKernelV41RequestState]bool, len(states))
		for _, state := range states {
			retiring[state] = true
			delete(byLane, state.lane)
		}
		kept := active[:0]
		for _, state := range active {
			if !retiring[state] {
				kept = append(kept, state)
			}
		}
		keptLength := len(kept)
		for index := keptLength; index < len(active); index++ {
			active[index] = nil
		}
		active = active[:keptLength]
	}
	prepare := func(requests []*inKernelCoalesceRequest) []*decodeLane {
		selected := make([]*inKernelV41RequestState, len(requests))
		for index, request := range requests {
			state := &inKernelV41RequestState{req: request, basePanels: sharedPanels, baseRows: sharedRows}
			selected[index] = state
		}
		admitted := make([]*decodeLane, 0, len(requests))
		started := 0
		for _, state := range selected {
			go func(state *inKernelV41RequestState) {
				request := state.req
				var out inKernelCoalesceResult
				defer func() {
					if cause := recover(); cause != nil {
						if err, known := recoverDevicePanic(cause); known {
							out.err = err
						} else {
							state.panic, out.err = cause, errInKernelV41CohortAborted
						}
						if state.lane != nil && state.lane.err != nil {
							out.err = state.lane.err
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
			}(state)
			started++
			if lane := <-state.req.prepared; lane != nil {
				state.lane = lane
				lane.ctx = state.req.ctx
				active = append(active, state)
				byLane[lane] = state
				admitted = append(admitted, lane)
			} else {
				<-state.req.done
				state.endCloses = joinedCloses
				if state.panic != nil {
					setPrimary(state.panic, state)
					publish(state)
					break
				}
				publish(state)
			}
		}
		if primary != nil {
			for _, state := range selected[started:] {
				state.req.result <- inKernelCoalesceResult{err: errInKernelV41CohortAborted}
				close(state.req.done)
				state.req.terminalErr = errInKernelV41CohortAborted
				state.endCloses = joinedCloses
				publish(state)
			}
		}
		return admitted
	}
	lanes := prepare(cohort)
	membership := func(lanes []*decodeLane) {
		if len(lanes) > episodePeak {
			episodePeak = len(lanes)
		}
		for _, lane := range lanes {
			if state := byLane[lane]; state != nil {
				state.participated = true
				if len(lanes) > state.peakActive {
					state.peakActive = len(lanes)
				}
			}
		}
	}
	membership(lanes)
	finishGroup := func(states []*inKernelV41RequestState) {
		for _, state := range states {
			state.req.proceed <- state.lane.err
		}
		for _, state := range states {
			<-state.req.done
			if state.panic != nil {
				setPrimary(state.panic, state)
			}
		}
		for _, state := range states {
			joinedCloses += state.req.closes.Load()
		}
		for _, state := range states {
			state.endCloses = joinedCloses
			if primary != nil && state != cause {
				state.req.terminalErr = errInKernelV41CohortAborted
			}
			publish(state)
		}
		removeActive(states)
	}
	retire := func(lanes []*decodeLane) error {
		states := make([]*inKernelV41RequestState, 0, len(lanes))
		for _, lane := range lanes {
			if state := byLane[lane]; state != nil {
				state.endPanels, state.endRows, state.ended = sharedPanels, sharedRows, true
				states = append(states, state)
			}
		}
		if len(states) > 0 {
			finishGroup(states)
		}
		if primary != nil {
			for _, state := range active {
				if state != cause {
					state.req.terminalErr = errInKernelV41CohortAborted
				}
			}
			return errInKernelV41CohortAborted
		}
		return nil
	}
	lifecycle := &inKernelV41Lifecycle{
		progress: func(panels, rows int) {
			sharedPanels, sharedRows = panels, rows
		},
		membership: membership,
		retire:     retire,
	}
	lifecycle.admit = func(freeSlots int) ([]*decodeLane, error) {
		freeSlots = min(freeSlots, 4-len(active))
		admitted := make([]*decodeLane, 0, freeSlots)
		attempts := 0
		for len(admitted) < freeSlots && attempts < 4 && primary == nil {
			limit := min(freeSlots-len(admitted), 4-attempts)
			requests := p.dequeueV41CoalescedRequests(limit)
			if len(requests) == 0 {
				break
			}
			attempts += len(requests)
			admitted = append(admitted, prepare(requests)...)
		}
		if primary != nil {
			return admitted, errInKernelV41CohortAborted
		}
		return admitted, nil
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
						setPrimary(cause, nil)
					}
				}
			}()
			if p.coalesceBatchHook != nil {
				p.coalesceBatchHook(len(lanes))
			}
			decodeCtx := context.WithValue(context.Background(), inKernelV41LifecycleContextKey{}, lifecycle)
			receipt, terminal = decode(decodeCtx, lanes, p.m)
		}()
	}
	if receipt.sharedPanels > sharedPanels {
		sharedPanels = receipt.sharedPanels
	}
	if receipt.sharedRows > sharedRows {
		sharedRows = receipt.sharedRows
	}
	func() {
		defer func() {
			if recovered := recover(); recovered != nil {
				if len(active) == 0 {
					log.Printf("agent: V4.1 cohort observer panic after all requests retired: %v", recovered)
					return
				}
				setPrimary(recovered, nil)
			}
		}()
		if p.coalesceSharedHook != nil {
			p.coalesceSharedHook(sharedPanels, 0)
		}
		enginestep.Default.ObserveCohort(episodePeak)
	}()
	if primary != nil {
		terminal = errInKernelV41CohortAborted
	}
	remaining := append([]*inKernelV41RequestState(nil), active...)
	for _, state := range remaining {
		lane := state.lane
		if !state.ended {
			state.endPanels, state.endRows, state.ended = sharedPanels, sharedRows, true
		}
		if lane.err == nil && !lane.done && lane.gen < lane.maxNew {
			lane.err = terminal
			if lane.err == nil {
				lane.err = errInKernelV41CohortAborted
			}
			lane.done = true
		}
		if primary != nil && state != cause {
			lane.err = errInKernelV41CohortAborted
			lane.done = true
			state.req.terminalErr = errInKernelV41CohortAborted
		}
	}
	if len(remaining) > 0 {
		finishGroup(remaining)
	}
	p.devMu.Unlock()
	gateHeld = false
	if primary != nil {
		if cause == nil {
			return &inKernelV41Fatal{value: primary}
		}
		cause.req.panicValue = primary
		return &inKernelV41Fatal{value: primary, request: cause.req}
	}
	return nil
}
