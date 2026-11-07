package agent

import (
	"fmt"
	"os"
	"strconv"
	"time"
)

// batchStepHealthEnv tunes the batched->serial degradation margin, in
// thousandths (1250 = a batched step may cost up to 1.25x the per-token floor
// before it counts as a violation). 0 disables the policy; unset keeps the
// default margin.
const batchStepHealthEnv = "FAK_INKERNEL_BATCH_HEALTH_MARGIN"

// batchStepProbeEvery is the batched-step cadence at which the cohort runs one
// serial probe step to refresh the per-token floor. Without probes, a floor
// seeded by a backlog-inflated first step would never be corrected (the
// inflated budget would let a collapsing batched path run forever).
const batchStepProbeEvery = 8

// batchHealthDiagEnabled gates the typed policy-transition lines
// (fak_batch_health ...) used by the q2kxl concurrency-collapse investigation.
// Off by default; FAK_BATCH_DIAG=1 turns it on. Pure observation: no behavior change.
var batchHealthDiagEnabled = os.Getenv("FAK_BATCH_DIAG") == "1"

// batchStepHealth guards the cohort decode loop against a batched forward that
// underperforms the per-token decode-step floor. Measured on rtp-test-h200
// (q2kxl rank-1): a fresh serve batches well at C<=4 (aggregate 35-41 tok/s)
// while C=8 inflates the batched step ~5x and an aged pool collapses every
// cell (aggregate decode 25.3 -> 0.9 tok/s from C=1 to C=16). The policy keeps
// a planner-level per-token floor - a ratcheting MIN over timed serial steps
// AND healthy batched steps, both folded as wall/lanes - and degrades the
// episode to serial steps after two consecutive batched steps whose per-token
// cost exceeds floor x margin. Serial probes alone would ratchet the floor
// HIGH: in a mixed prefill+decode stream every probe queues behind the shared
// CUDA stream's backlog (measured 2.84s wall for 8 lanes), so healthy low-lane
// batched steps co-feed the floor and the MIN rejects the polluted samples.
// Degradation is correctness-neutral: the serial driver is bit-for-bit
// identical per lane (the same contract StepBatchActive pins), so only the
// wall time changes - a collapsing batched path becomes the proven serial
// floor instead of a collapse.
type batchStepHealth struct {
	p          *InKernelPlanner
	marginNum  int
	violations int
	degraded   bool
	sinceProbe int
}

func newBatchStepHealth(p *InKernelPlanner) *batchStepHealth {
	h := &batchStepHealth{p: p, marginNum: 1250}
	if raw := os.Getenv(batchStepHealthEnv); raw != "" {
		if v, err := strconv.Atoi(raw); err == nil {
			switch {
			case v == 0:
				h.marginNum = 0 // explicit disable
			case v >= 100:
				h.marginNum = v
			}
		}
	}
	return h
}

// enabled reports whether the policy is armed at all: a non-zero margin
// against a live planner. A disabled policy keeps the historical all-batched
// step loop exactly - no calibration step, no probes, no degradation.
func (h *batchStepHealth) enabled() bool {
	return h != nil && h.marginNum > 0 && h.p != nil
}

// calibrated reports whether the planner already carries a per-token floor
// from an earlier episode or probe.
func (h *batchStepHealth) calibrated() bool {
	return h != nil && h.marginNum > 0 && h.p != nil && h.p.batchSerialPerLaneNS.Load() > 0
}

// floor returns the planner-level per-token decode-step floor, or 0 when unknown.
func (h *batchStepHealth) floor() time.Duration {
	if h == nil || h.p == nil {
		return 0
	}
	if ns := h.p.batchSerialPerLaneNS.Load(); ns > 0 {
		return time.Duration(ns)
	}
	return 0
}

// foldFloor folds one per-token sample into the planner floor as a ratcheting
// MIN: the floor only moves down, so a slow regime (a queued prefill backlog
// absorbed by one step) cannot inflate the budget that guards the batched path.
// It reports the value the sample replaced, so callers can witness a seed
// (old==0) versus a lowering (old>sample).
func (h *batchStepHealth) foldFloor(per int64) (old int64, changed bool) {
	for {
		cur := h.p.batchSerialPerLaneNS.Load()
		if cur > 0 && cur <= per {
			return cur, false
		}
		if h.p.batchSerialPerLaneNS.CompareAndSwap(cur, per) {
			return cur, true
		}
	}
}

// observeSerial folds one timed serial step of lanes lanes into the floor as a
// ratcheting MIN per-token sample (wall/lanes). During a degraded episode the
// serial driver keeps feeding the floor silently; only a non-degraded serial
// step is a probe and gets the floor_update witness line.
func (h *batchStepHealth) observeSerial(d time.Duration, lanes int) {
	if h == nil || h.marginNum == 0 || h.p == nil || d <= 0 || lanes <= 0 {
		return
	}
	per := (d / time.Duration(lanes)).Nanoseconds()
	if per <= 0 {
		return
	}
	old, changed := h.foldFloor(per)
	if !batchHealthDiagEnabled || !changed {
		return
	}
	if old == 0 {
		fmt.Fprintf(os.Stderr, "fak_batch_health floor_seed_ns=%d source=serial lanes=%d\n", per, lanes)
	} else if !h.degraded {
		fmt.Fprintf(os.Stderr, "fak_batch_health floor_update_ns=%d old_ns=%d source=probe\n", per, old)
	}
}

// useSerial reports whether this episode has degraded to serial steps.
func (h *batchStepHealth) useSerial() bool { return h != nil && h.degraded }

// observeBatched records one batched step of d for lanes lanes. It flips the
// episode to serial on the second consecutive per-token violation of the floor
// budget and asks for a serial probe step on the probe cadence (reported via
// probeDue). A non-violating step resets the violation streak and feeds the
// floor: a healthy batched step is evidence the path can run at that per-token
// cost, and the ratcheting MIN rejects the backlog-inflated samples that
// pollute serial probes in a mixed prefill+decode stream.
func (h *batchStepHealth) observeBatched(d time.Duration, lanes int) {
	if h == nil || h.marginNum == 0 || h.degraded || lanes <= 0 {
		return
	}
	h.sinceProbe++
	floor := h.floor()
	per := d / time.Duration(lanes)
	if floor <= 0 {
		// No floor yet: a batched step cannot violate, and its per-token cost
		// seeds the floor like any healthy sample.
		if per.Nanoseconds() <= 0 {
			return
		}
		old, changed := h.foldFloor(per.Nanoseconds())
		if batchHealthDiagEnabled && changed && old == 0 {
			fmt.Fprintf(os.Stderr, "fak_batch_health floor_seed_ns=%d source=batched lanes=%d\n", per.Nanoseconds(), lanes)
		}
		return
	}
	if per > floor*time.Duration(h.marginNum)/1000 {
		h.violations++
		if h.violations >= 2 {
			h.degraded = true
			if batchHealthDiagEnabled {
				fmt.Fprintf(os.Stderr, "fak_batch_health degraded floor_ns=%d step_ns=%d lanes=%d violations=2\n",
					int64(floor), int64(d), lanes)
			}
		}
		return
	}
	h.violations = 0
	if per.Nanoseconds() <= 0 {
		return
	}
	h.foldFloor(per.Nanoseconds())
}

// probeDue reports whether the next step should be a serial probe (floor
// refresh) instead of a batched step.
func (h *batchStepHealth) probeDue() bool {
	return h != nil && h.marginNum > 0 && !h.degraded && h.calibrated() &&
		h.sinceProbe >= batchStepProbeEvery
}
