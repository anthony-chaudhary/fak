package dispatchtick

// preflight_churn.go -- the host_churn admission cap term: fold a MEASURED whole-host
// process-spawn burst DOWN into the spawn preflight so that when the box is already in a
// spawn storm -- typically several INDEPENDENT dispatchers (FleetIssueDispatchCodex,
// FleetIssueDispatchGlm, `accounts launch`) each co-launching a wave in the same window --
// a new dispatcher backs off instead of piling onto a saturated scheduler.
//
// WHY A SEPARATE TERM. The existing per-loop cadence floor (loopmgr governor,
// MinIntervalSeconds) keys on ONE loop's LoopID + LastEventUnixNano: it stops a single
// flapping/doubled scheduler from storming ITSELF. It cannot see a SECOND, DIFFERENT loop
// -- three dispatchers each pass their own cadence floor and still co-arrive. The host_churn
// term is the missing CROSS-dispatcher gate: it reads a whole-host signal (processes born in
// the last sample interval) that every dispatcher observes identically, so any one of them
// backing off relieves the shared blast radius. This is the "reaper that watches CHURN, not
// usage" the stall diagnosis calls for, moved to admission time -- refuse the spawn that
// would deepen the storm rather than reap after the freeze.
//
// The signal: internal/stallscan already classifies a whole-machine spawn storm from cheap
// point-in-time counters (Sample.SpawnBurst -> CauseSpawnStorm) -- processes that appeared
// since the previous sample, the exact counter whose crossing coincided with the live
// desktop freeze in the reference capture. The impure shell samples it ONCE per tick
// (cheaply, or reuses a recent stallscan self-monitor reading) and passes the count in; this
// fold stays pure (state in, decision out) and never itself spawns a process during
// admission, so it cannot add to the churn it measures.
//
// Like the gate and rate terms this is a COMPOSABLE fold over the PreflightResult: it can
// only LOWER the effective cap, abstains below the threshold, and holds a cold-start floor so
// a burst throttles GROWTH, not liveness -- a fleet that lost every worker into a storm is
// still allowed ONE probe to learn whether the burst subsided, rather than freezing at a zero
// cap forever.

import (
	"fmt"
)

// PreflightRefuseChurn is the verdict a spawn preflight returns when a host spawn-storm is
// the sole binding admission term. It is not safe-to-spawn, so the sweep stops on it exactly
// as it does on REFUSE_AT_CAP / REFUSE_NO_SEAT / REFUSE_GATE / REFUSE_RATE_LIMIT.
const PreflightRefuseChurn = "REFUSE_HOST_CHURN"

// HostChurnBackoff is the closed-vocabulary refusal token PreflightRefuseChurn carries in its
// reason. It MUST stay byte-identical to the dos.toml [reasons.HOST_CHURN_BACKOFF] declaration
// so the token this fold emits is the one `dos check-reason` verifies and the loop routes on.
const HostChurnBackoff = "HOST_CHURN_BACKOFF"

// DefaultChurnBurstThreshold is the WINDOW-UNKNOWN burst floor: the fewest processes born on
// the host within one sample interval that arm the backoff, compared as a bare count. It
// mirrors stallscan's own SpawnBurstStall default (8), the calibrated crossing at which a spawn
// storm coincided with a desktop freeze on the reference box, measured there as a NET process
// delta. Below it, ordinary process turnover is noise and the term abstains. The impure shell
// overlays FAK_CHURN_BURST_THRESHOLD.
//
// Prefer the rate path (DefaultChurnBurstRate) whenever the caller knows its window: see the
// warning there for why a bare count cannot be calibrated.
const DefaultChurnBurstThreshold = 8

// DefaultChurnBurstRate is the burst floor in the only unit that can be calibrated across
// callers: gross process births per SECOND. It tracks stallscan's SpawnBurstRateStall (150/sec)
// and MUST move with it — the two gates read the same signal, so a split calibration would let
// admission refuse a host the classifier calls calm, or vice versa.
//
// WHY A RATE AND NOT A COUNT. A count of 8 means "storm" only relative to some unstated
// interval. Live capture on the reference box (2026-08-05, 101 one-second ticks under ordinary
// fleet load) measured a median of 22 gross births/sec, p95 63, max 83 — every one of which
// clears a count threshold of 8. Feeding a gross birth count into the count path would refuse
// dispatch on ~95% of ticks of a healthy box: a gate that always refuses is not safer than one
// that never fires, it is the same defect pointed the other way. 150/sec sits ~1.8x above that
// measured max. As with stallscan, this bounds FALSE POSITIVES against a measured negative
// class; sensitivity is unproven until a freeze is captured with a window attached.
const DefaultChurnBurstRate = 150.0

// DefaultChurnMinWorkers is the cold-start floor: the fewest workers the churn term admits even
// under a live host storm. Backoff throttles GROWTH, never liveness -- a floor of 0 would let a
// storm that killed every worker freeze the fleet at a zero cap with no probe left to witness
// recovery. Holding a minimal presence keeps one probe live. The shell overlays
// FAK_CHURN_MIN_WORKERS.
const DefaultChurnMinWorkers = 1

// ChurnCheck carries the MEASURED whole-host spawn burst the backoff term folds. Recent is the
// count of processes that appeared on the host within one stallscan sample interval -- computed
// by the impure shell from a stallscan Sample, never a worker self-report. The zero value means
// "no burst signal" (Recent 0 < any positive threshold) and never lowers the cap, so a caller
// that wires nothing keeps the existing behavior byte-for-byte.
type ChurnCheck struct {
	// Recent is the host-wide count of processes born within the last sample interval
	// (stallscan Sample.SpawnBurst). It is a WHOLE-HOST count, not scoped to this dispatcher,
	// which is precisely what makes the term see co-launching PEER dispatchers.
	Recent int
	// WindowSeconds is the wall-clock span Recent was counted over (stallscan
	// Sample.SpawnWindowSeconds). > 0 selects the RATE comparison against RateThreshold; 0
	// means the window is unknown and the legacy bare-count comparison applies. A count with
	// no window is not a rate, and the two are not interchangeable: see DefaultChurnBurstRate.
	WindowSeconds float64
	// Threshold is the window-unknown burst floor that arms the backoff; <= 0 means
	// DefaultChurnBurstThreshold.
	Threshold int
	// RateThreshold is the births/sec floor used when WindowSeconds > 0; <= 0 means
	// DefaultChurnBurstRate.
	RateThreshold float64
	// MinWorkers is the cold-start floor the backoff holds to even under a storm; <= 0 means
	// DefaultChurnMinWorkers. The shell sets it from FAK_CHURN_MIN_WORKERS.
	MinWorkers int
}

// rate converts Recent to births/sec, reporting ok only when the window is known. Pure.
func (c ChurnCheck) rate() (float64, bool) {
	if c.Recent <= 0 || c.WindowSeconds <= 0 {
		return 0, false
	}
	return float64(c.Recent) / c.WindowSeconds, true
}

// rateThreshold resolves the births/sec floor, defaulting to DefaultChurnBurstRate.
func (c ChurnCheck) rateThreshold() float64 {
	if c.RateThreshold <= 0 {
		return DefaultChurnBurstRate
	}
	return c.RateThreshold
}

// threshold resolves the burst floor, defaulting a zero/negative Threshold to the built-in
// DefaultChurnBurstThreshold so the zero-value check stays hermetic.
func (c ChurnCheck) threshold() int {
	if c.Threshold <= 0 {
		return DefaultChurnBurstThreshold
	}
	return c.Threshold
}

// floor resolves the cold-start allowance, defaulting a zero/negative MinWorkers to
// DefaultChurnMinWorkers so the zero-value keeps the liveness-preserving default.
func (c ChurnCheck) floor() int {
	if c.MinWorkers <= 0 {
		return DefaultChurnMinWorkers
	}
	return c.MinWorkers
}

// pressured reports whether the measured host burst clears the arming threshold. Pure: state
// in, decision out; a sub-threshold burst abstains (no-op). Compares a RATE when the caller
// carried its window, and a bare count otherwise — never the two against each other.
func (c ChurnCheck) pressured() bool {
	if r, ok := c.rate(); ok {
		return r >= c.rateThreshold()
	}
	return c.Recent >= c.threshold()
}

// ApplyChurnBackpressure folds a whole-host spawn burst into an already-evaluated preflight as
// the host_churn cap term. A live storm holds the fleet at max(live, floor) -- admit no NEW
// concurrent worker onto a saturated scheduler beyond a minimal cold-start probe -- which can
// only LOWER the effective cap.
//
// A WARM fleet (live at/above the floor) freezes at its live count and refuses with
// PreflightRefuseChurn so the sweep stops on it and the loop backs off this tick. A COLD fleet
// (live below the floor, e.g. a storm just killed every worker) is lowered to the floor and kept
// SPAWN_OK, so one probe still runs to witness whether the storm cleared rather than deadlocking
// at a zero cap.
//
// The fold is a no-op when the preflight ALREADY refused for a higher-precedence reason (host /
// seat / at-cap / account / gate / rate): the fleet is then already not growing, so the storm is
// not the sole binding term and the existing verdict stands. It is also a no-op below the arming
// threshold and when the floor meets/exceeds the existing cap (the term never manufactures
// capacity).
func ApplyChurnBackpressure(res PreflightResult, c ChurnCheck) PreflightResult {
	// Bottom-up backpressure on a SAFE-to-spawn preflight only.
	if !res.OK {
		return res
	}
	if !c.pressured() || res.Live >= res.Cap {
		return res
	}
	// Hold at max(live, floor). res.Live < res.Cap holds here (a SPAWN_OK preflight has
	// headroom), so a floor at or above the cap means the term cannot bind -- it never RAISES
	// the cap, so leave the preflight untouched.
	hold := res.Live
	if floor := c.floor(); hold < floor {
		hold = floor
	}
	if hold >= res.Cap {
		return res
	}
	res.Cap = hold
	res.Headroom = hold - res.Live
	res.CapTerms.EffectiveCap = hold
	res.CapTerms.Limiting = "host_churn"
	if res.Headroom > 0 {
		// Cold fleet: the term lowered the cap to the floor (throttling growth) but left a
		// minimal cold-start probe, so the verdict stays SPAWN_OK.
		return res
	}
	// Warm fleet (live at/above the floor): no headroom above the hold -- freeze and refuse
	// with the closed HOST_CHURN_BACKOFF token so the sweep terminates on it.
	res.OK = false
	res.Verdict = PreflightRefuseChurn
	res.Reason = hostChurnBackoffReason(c, res.Live)
	return res
}

// hostChurnBackoffReason names the closed HOST_CHURN_BACKOFF refusal token and cites the
// measured burst (processes born in one interval) so a reader -- and `dos check-reason` -- can
// bind both the refusal class and its evidence.
func hostChurnBackoffReason(c ChurnCheck, live int) string {
	const tail = " -- a whole-host spawn storm, typically several dispatchers co-launching waves at once; holding the fleet at %d live worker(s). Admit no new concurrent load onto a saturated scheduler until the burst subsides. This is a CROSS-dispatcher gate the per-loop cadence floor cannot see (each loop passes its own floor yet they still co-arrive)."
	if r, ok := c.rate(); ok {
		// Cite the rate AND the raw count/window it came from, so a reader can audit the
		// conversion instead of taking the derived number on trust.
		return fmt.Sprintf("%s: %.0f process(es)/sec spawned on this host (%d in %.2fs, >= %.0f/sec)"+tail,
			HostChurnBackoff, r, c.Recent, c.WindowSeconds, c.rateThreshold(), live)
	}
	return fmt.Sprintf("%s: %d process(es) spawned on this host within one sample interval (>= %d)"+tail,
		HostChurnBackoff, c.Recent, c.threshold(), live)
}
