package agent

import (
	"context"
	"errors"
	"math"
	"os"
	"strconv"
	"time"

	"github.com/anthony-chaudhary/fak/internal/enginestep"
)

var errInKernelQwenCohortAborted = errors.New("agent: Qwen cohort aborted; no request replay permitted")

// inKernelDecodeShare adapts TensorFold's wall-time split between prompt steps
// and live decode rounds to the in-kernel coalescer's prepare/decode visits.
// A prompt step taking ns while decoders are live owes them ns*(1-p)/p of
// decode time (p is the prompt's share of wall time, floored at
// inKernelDecodeShareMinPrompt); decode rounds pay the debt down. A decode
// share outside (0,1) keeps p=1: nothing is ever owed and whole-prompt arrival
// ordering is preserved. An optional row budget caps the rows a visit may fill
// beside live decoders; 0 means no cap. In this seam a row is one prepared
// prompt-fill pass alongside the running lanes.
//
// Source: ashhart/TensorFold zig/src/core/lane_host_fill.zig @9e8c368b0b4c2c7892a0c9f0f302d7c164016209
// plus docs/api.md:236 (Apache-2.0; idea adapted into idiomatic Go, no code
// copied). See THIRD_PARTY_NOTICES.md.
const inKernelDecodeShareMinPrompt = 0.01 // below it a step's debt is minutes of decoding, and is no share

// inKernelDecodeShareDebtCapNs caps the owed-debt accumulator at 2s so a long
// prompt fill cannot silence decode rounds for minutes: decode rounds resume
// at least every cap window, bounding how long live lanes can be starved by
// one prepare visit.
const inKernelDecodeShareDebtCapNs = int64(2_000_000_000)

// FAK_INKERNEL_DECODE_SHARE gives live decode rounds a wall-time share in
// (0,1); an explicit value outside (0,1) keeps whole-prompt arrival ordering,
// and an unset value defaults to 0.5.
const inKernelDecodeShareEnv = "FAK_INKERNEL_DECODE_SHARE"

// FAK_INKERNEL_PROMPT_ROW_BUDGET caps the prompt rows a visit fills beside
// live decoders; unset or non-positive means no cap.
const inKernelPromptRowBudgetEnv = "FAK_INKERNEL_PROMPT_ROW_BUDGET"

type inKernelDecodeShare struct {
	promptShare float64 // p: the prompt's share of wall time beside decoders; 1 owes nothing
	debtNs      int64   // decode time owed to live decoders before the next prompt step
	rowBudget   int     // prompt rows a visit may fill beside live decoders; 0: none
}

// newInKernelDecodeShare maps a decode-share fraction onto the prompt's
// complement, floored at inKernelDecodeShareMinPrompt. A fraction outside
// (0,1) keeps the default p=1, so decode-share 0 preserves whole-prompt
// arrival ordering.
func newInKernelDecodeShare(decodeShare float64, rowBudget int) inKernelDecodeShare {
	share := inKernelDecodeShare{promptShare: 1, rowBudget: rowBudget}
	if decodeShare > 0 && decodeShare < 1 {
		share.promptShare = max(1-decodeShare, inKernelDecodeShareMinPrompt)
	}
	return share
}

// inKernelDecodeShareFromEnv reads the operator knobs once per cohort episode;
// an unset share defaults to 0.5 and every parse failure falls back to the
// historical no-debt ordering.
func inKernelDecodeShareFromEnv() inKernelDecodeShare {
	fraction := 0.0
	if raw := os.Getenv(inKernelDecodeShareEnv); raw != "" {
		if value, err := strconv.ParseFloat(raw, 64); err == nil {
			fraction = value
		}
	} else {
		// Default ON (0.5): live decode rounds get half the wall beside prompt
		// steps. The q2kxl rank-1 sweep measured the unset default starving
		// decode lanes of mid-episode prompt fills (cohort_wait 251s average,
		// decode window 370s holding ~23s of steps, aggregate decode 25.3 ->
		// 0.9 tok/s from C=1 to C=16). An explicit 0 preserves the historical
		// whole-prompt arrival ordering.
		fraction = 0.5
	}
	budget := 0
	if raw := os.Getenv(inKernelPromptRowBudgetEnv); raw != "" {
		if value, err := strconv.Atoi(raw); err == nil {
			budget = value
		}
	}
	return newInKernelDecodeShare(fraction, budget)
}

// stepped charges a prompt step of ns taken while decoders were live. The debt
// saturates instead of overflowing.
func (s *inKernelDecodeShare) stepped(ns int64, decoding bool) {
	if s.promptShare >= 1 || !decoding || ns <= 0 {
		return
	}
	owed := float64(ns) * (1 - s.promptShare) / s.promptShare
	if owed > math.MaxInt64/2 {
		owed = math.MaxInt64 / 2
	}
	charge := int64(owed)
	if s.debtNs > inKernelDecodeShareDebtCapNs-charge {
		s.debtNs = inKernelDecodeShareDebtCapNs
		return
	}
	s.debtNs += charge
}

// rounded pays a decode round of ns back against the owed debt, floored at zero.
func (s *inKernelDecodeShare) rounded(ns int64) {
	s.debtNs = max(0, s.debtNs-ns)
}

// retired discards debt when its last beneficiary leaves. An incoming prompt
// that is still preparing must not give its future decoder another lane's debt.
func (s *inKernelDecodeShare) retired(decoding bool) {
	if !decoding {
		s.debtNs = 0
	}
}

// due reports whether a prompt step may run now: none owed, or nothing decoding.
func (s *inKernelDecodeShare) due(decoding bool) bool {
	return s.debtNs <= 0 || !decoding
}

// visitLimit caps a visit's planned prompt rows by the row budget; no budget
// leaves the visit uncapped.
func (s inKernelDecodeShare) visitLimit(planned int) int {
	if s.rowBudget <= 0 || s.rowBudget >= planned {
		return planned
	}
	return s.rowBudget
}

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
	cohort = p.devMu.LockCohort(cohort)
	if len(cohort) == 0 {
		enginestep.Default.ObservePhase(enginestep.PhaseDeviceWait, time.Since(deviceWait))
		return nil
	}
	completed := false
	defer func() {
		if !completed {
			// Goexit does not return to drainCoalescedGenerates. Its normal
			// handoff must still happen, after the device gate is released.
			p.finishCoalescedCohort()
			for _, request := range final {
				request.publishReceipt()
			}
		}
	}()
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
	var preparing *inKernelQwenRequestState
	var boundary *inKernelPrefillBoundary
	var boundaryPending bool
	var unstarted []*inKernelCoalesceRequest
	var iteration *inKernelDecodeIteration
	var stepRound func() bool
	share := inKernelDecodeShareFromEnv()
	health := newBatchStepHealth(p)
	healthCalibrated := health.calibrated()
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
		// A distinct paused preparation can keep this episode alive after
		// its last old decoder retires. Publish that joined decoder promptly.
		ending := len(active) == 0 && (preparing == nil || preparing == state)
		if ending || primary != nil || terminal != nil {
			final = append(final, request)
		} else {
			request.publishReceipt()
		}
	}
	prepare := func(requests []*inKernelCoalesceRequest, interleave bool) {
		for index, request := range requests {
			state := &inKernelQwenRequestState{req: request}
			unstarted = requests[index+1:]
			if primary != nil || terminal != nil {
				request.result <- inKernelCoalesceResult{err: errInKernelQwenCohortAborted}
				close(request.done)
				publish(state)
				continue
			}
			preparing = state
			boundary = nil
			if interleave {
				boundary = &inKernelPrefillBoundary{ready: make(chan struct{}), resume: make(chan error)}
			}
			requestBoundary := boundary
			segmentStart, segmentDecoding := time.Now(), len(active) > 0
			go func() {
				var out inKernelCoalesceResult
				returned := false
				defer func() {
					if value := recover(); value != nil {
						if err, known := recoverDevicePanic(value); known {
							out.err = err
						} else {
							state.panic, out.err = value, errInKernelQwenCohortAborted
						}
					}
					if !returned && out.err == nil {
						// Goexit is not a successful empty generation.
						out.err = errInKernelQwenCohortAborted
					}
					request.result <- out
					close(request.done)
					select {
					case request.prepared <- nil:
					default:
					}
				}()
				runCtx := context.WithValue(request.ctx, inKernelCoalesceContextKey{}, request)
				if requestBoundary != nil {
					runCtx = context.WithValue(runCtx, inKernelPrefillBoundaryKey{}, requestBoundary)
				}
				out.result, out.err = request.run(runCtx)
				returned = true
			}()
			var lane *decodeLane
			if requestBoundary == nil {
				lane = <-request.prepared
			} else {
			waiting:
				for {
					select {
					case lane = <-request.prepared:
						share.stepped(time.Since(segmentStart).Nanoseconds(), segmentDecoding)
						break waiting
					case <-requestBoundary.ready:
						boundaryPending = true
						// Debit before the round: rounded deliberately discards
						// surplus credit, so charging at prepare's end loses it.
						share.stepped(time.Since(segmentStart).Nanoseconds(), segmentDecoding)
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
							if len(active) > 0 && primary == nil && terminal == nil {
								stepRound()
							}
						}()
						var resumeErr error
						if primary != nil || terminal != nil {
							// A causal device OOM belongs to its decoder. Passing
							// it to an unadmitted preparation could restart prefill
							// through generateReusedWithOOMRetry during our join.
							resumeErr = errInKernelQwenCohortAborted
						}
						segmentStart, segmentDecoding = time.Now(), len(active) > 0
						requestBoundary.resume <- resumeErr
						boundaryPending = false
					}
				}
			}
			if lane != nil {
				state.lane, lane.ctx = lane, request.ctx
				active = append(active, state)
				iteration = nil
			} else {
				<-request.done
				joinedCloses += request.closes.Load()
				if state.panic != nil {
					setPrimary(state.panic, request)
				}
				publish(state)
			}
			preparing, boundary = nil, nil
		}
		unstarted = nil
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
	defer func() {
		if completed {
			return
		}
		if terminal == nil {
			terminal = errInKernelQwenCohortAborted
		}
		if preparing != nil {
			if boundaryPending {
				boundary.resume <- errInKernelQwenCohortAborted
				boundaryPending = false
			}
			<-preparing.req.done
			joinedCloses += preparing.req.closes.Load()
			if preparing.panic != nil {
				setPrimary(preparing.panic, preparing.req)
			}
			publish(preparing)
		}
		for _, request := range unstarted {
			request.result <- inKernelCoalesceResult{err: errInKernelQwenCohortAborted}
			close(request.done)
			publish(&inKernelQwenRequestState{req: request})
		}
		retire(true)
		for _, request := range final {
			if primary != nil && request == cause {
				request.panicValue = primary
			} else {
				request.terminalErr = errInKernelQwenCohortAborted
			}
		}
	}()
	stepRound = func() bool {
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
		stepStart := time.Now()
		switch {
		case !health.enabled():
			// Policy disabled (margin 0): the historical all-batched step loop,
			// no calibration step, probes or degradation.
			iteration.step(context.Background())
		case !healthCalibrated:
			// Seed the per-token floor from one real serial step.
			iteration.stepSerial(context.Background())
			health.observeSerial(time.Since(stepStart), iteration.forwarded)
			healthCalibrated = true
		case health.useSerial():
			iteration.stepSerial(context.Background())
			health.observeSerial(time.Since(stepStart), iteration.forwarded)
		case health.probeDue():
			// Periodic serial probe: refresh the floor so a backlog-inflated
			// seed cannot permanently excuse a collapsing batched path.
			iteration.stepSerial(context.Background())
			health.observeSerial(time.Since(stepStart), iteration.forwarded)
			health.sinceProbe = 0
		default:
			iteration.step(context.Background())
			if iteration.serialSteps == 0 {
				// Only a truly batched forward is guarded: a scalar fallback
				// already ran serially, and its timing must neither pollute the
				// floor nor trip the violation budget - the same invariant HEAD
				// uses to withhold the PathBatched observation.
				health.observeBatched(time.Since(stepStart), iteration.forwarded)
			}
		}
		share.rounded(time.Since(stepStart).Nanoseconds())
		stepPanels := iteration.sharedPanels - beforePanels
		stepMACs := iteration.sharedMACs - beforeMACs
		panels, macs = panels+stepPanels, macs+stepMACs
		for index, state := range active {
			if iteration.active[index] {
				state.req.receipt.SharedPanels += stepPanels
				state.req.receipt.SharedMACs += stepMACs
			}
		}
		changed := retire(false)
		if changed {
			iteration = nil
			share.retired(len(active) > 0)
		}
		return changed
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
		prepare(cohort, false)
		if p.coalesceBatchHook != nil {
			p.coalesceBatchHook(len(active))
		}
		for len(active) > 0 && primary == nil && terminal == nil {
			changed := stepRound()
			if primary != nil || terminal != nil || len(active) == 0 {
				break // An empty episode yields to the ordinary next-cohort handoff.
			}
			// Owed decode time outranks new prompt steps this visit; the
			// still-running decode lanes keep stepping until the debt clears.
			if !share.due(len(active) > 0) {
				continue
			}
			// One bounded queue selection per completed step prevents repeated
			// failed preparations from starving the still-running decode lanes.
			limit := share.visitLimit(inKernelDecodeCohortMax - len(active))
			requests := p.dequeueCoalescedRequests(inKernelCoalesceQwen, limit)
			if len(requests) > 0 {
				prepStart := time.Now()
				interleave := share.promptShare < 1
				prepare(requests, interleave)
				if !interleave {
					share.stepped(time.Since(prepStart).Nanoseconds(), len(active) > 0)
				}
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
	completed = true
	return final
}
