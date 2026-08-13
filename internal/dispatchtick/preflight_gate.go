package dispatchtick

// preflight_gate.go -- the fifth (backpressure) admission cap term: fold MEASURED
// gate-latency health UP into the spawn preflight so a slow kernel earns spawn
// reluctance (#2221, gap G3 / risk R4 of the dynamic-range epic #2218).
//
// The four terms EvaluatePreflight already folds (max-workers, dos target, host cap,
// seat cap) all flow admission DOWN the band ladder; none consults gate health, so a
// fleet keeps admitting workers onto a kernel whose guard-hook p99 has breached its
// 250ms budget (internal/turntaxmeter/hooklat.go) -- the fast inner loop saturates
// silently while the slow outer loop commands more load. This is the missing UP edge:
// a breached hook-latency rollup (or a standing overhead-budget breach) LOWERS the
// effective cap, never raises it, and when it is the SOLE binding term for a WARM
// fleet the tick refuses with the closed GATE_PRESSURE token so the sweep terminates
// on it like any other refusal.
//
// The one liveness carve-out (the cold-start floor, DefaultGateMinWorkers): pressure
// throttles GROWTH, not existence. A COLD fleet is held to a minimal floor rather than
// frozen at a zero cap, because a slow kernel that never admitted its first worker
// could never clear the backlog whose windowed p99 gates it -- the signal would never
// recover on its own and the fleet would deadlock. See ApplyGateBackpressure.
//
// It is deliberately a COMPOSABLE fold over the PreflightResult rather than another
// branch inside EvaluatePreflight: the signal is the measured rollup the impure shell
// computed from the observed hook-observation stream (never a worker self-report), and
// keeping the fold pure (state in, decision out) preserves the dos_arbitrate-style
// replay the honest-fence asks for. Abstains (no-op) on a thin sample exactly as
// hooklat does (turntaxmeter.MinHookAlarmSamples, applied by JudgeHookLatency).

import (
	"fmt"

	"github.com/anthony-chaudhary/fak/internal/turntaxmeter"
)

// PreflightRefuseGate is the verdict a spawn preflight returns when degraded gate
// health is the sole binding admission term. It is not safe-to-spawn, so the sweep
// stops on it exactly as it does on REFUSE_AT_CAP / REFUSE_NO_SEAT.
const PreflightRefuseGate = "REFUSE_GATE"

// GatePressure is the closed-vocabulary refusal token PreflightRefuseGate carries in
// its reason. It MUST stay byte-identical to the dos.toml [reasons.GATE_PRESSURE]
// declaration so the token this fold emits is the one `dos man wedge <TOKEN> --explain` verifies and
// the loop routes to a replan.
const GatePressure = "GATE_PRESSURE"

// DefaultGateMinWorkers is the cold-start floor: the fewest workers the gate admits
// even under sustained pressure. Backpressure must throttle GROWTH, never liveness --
// a floor of 0 is the deadlock this constant exists to forbid: a cold fleet (0 live)
// pressured to a 0 cap can never start the first worker, so the slow kernel never
// clears its own backlog and the windowed p99 that gates it never recovers. Holding a
// minimal presence keeps the fleet live while still refusing to GROW onto a slow
// kernel. The impure shell overlays FAK_GATE_MIN_WORKERS on top of this.
const DefaultGateMinWorkers = 1

// GateCheck carries the MEASURED gate-health state the backpressure term folds. The
// signal is the guard-hook latency rollup total (the p99 tail the budget judges) and a
// standing overhead-budget breach flag -- both computed by the impure shell from the
// observed hook-observation stream (internal/turntaxmeter), never a worker
// self-report. The zero value means "no gate signal" and never lowers the cap, so a
// caller that wires nothing keeps the existing four-term behavior and replay of the
// same state yields the same decision.
type GateCheck struct {
	// Hook is the all-verbs hook-latency rollup total. Its Count carries the sample
	// floor: JudgeHookLatency ABSTAINS below turntaxmeter.MinHookAlarmSamples, so a
	// thin sample is a no-op here exactly as `fak hooklat` already treats it.
	Hook turntaxmeter.HookLatencyStats
	// HookBudgetMS is the declared p99 budget the tail is judged against; <= 0 means
	// no budget and can never breach. The shell passes DefaultHookP99BudgetMS.
	HookBudgetMS float64
	// OverheadBreach marks a standing OVERHEAD_BUDGET_EXCEEDED breach observed on a
	// lifecycle rung (internal/turntaxmeter.CheckSpan). It is already an accumulated
	// verdict, so unlike the hook tail it carries no separate sample floor.
	OverheadBreach bool
	// MinWorkers is the cold-start floor the backpressure fold holds to even under
	// pressure: the fewest workers admitted so throttling stays a brake on GROWTH
	// rather than a deadlock at zero. Zero (the default) means DefaultGateMinWorkers;
	// the impure shell sets it from FAK_GATE_MIN_WORKERS.
	MinWorkers int
}

// floor resolves the cold-start allowance the pressured fold holds to. A zero or
// negative MinWorkers means the built-in DefaultGateMinWorkers, so the zero-value
// GateCheck keeps the correct liveness-preserving default and stays hermetic.
func (g GateCheck) floor() int {
	if g.MinWorkers <= 0 {
		return DefaultGateMinWorkers
	}
	return g.MinWorkers
}

// gatePressure reads the measured gate-health state back to a spawn-reluctance
// decision. A standing overhead-budget breach names OVERHEAD_BUDGET_EXCEEDED; a
// hook-latency tail over budget (with enough samples) names GATE_LATENCY_REGRESSION; a
// thin or within-budget tail abstains. Pure: state in, (pressured, cause-token) out.
func gatePressure(g GateCheck) (bool, string) {
	if g.OverheadBreach {
		return true, turntaxmeter.OverheadBudgetExceeded
	}
	if v := turntaxmeter.JudgeHookLatency(g.Hook, g.HookBudgetMS); !v.OK {
		return true, v.Reason
	}
	return false, ""
}

// ApplyGateBackpressure folds gate health into an already-evaluated preflight as the
// fifth cap term. Degraded gate health holds the fleet at max(live, floor) -- admit no
// NEW worker onto a slow kernel BEYOND a minimal cold-start presence -- which can only
// LOWER the effective cap, never raise it.
//
// Backpressure throttles GROWTH, not liveness. A WARM fleet (live at or above the
// floor) freezes at its live count and refuses with PreflightRefuseGate, exactly as
// before: the sweep stops on it like any other refusal. A COLD fleet (live below the
// floor) is instead lowered to the floor and kept SPAWN_OK, so a slow kernel still
// gets its minimal presence rather than deadlocking at a zero cap forever -- the first
// worker must be admitted for the kernel to clear the very backlog whose windowed p99
// gates it, or the gate could never recover on its own.
//
// The fold is a no-op when the preflight ALREADY refused for a higher-precedence
// reason (host / seat / at-cap / account): the fleet is then already not growing, so
// gate pressure is not the sole binding term and the existing verdict stands. It is
// also a no-op on a healthy gate, a thin sample, or when the floor meets/exceeds the
// existing cap (the gate cannot manufacture capacity).
func ApplyGateBackpressure(res PreflightResult, g GateCheck) PreflightResult {
	// The gate is bottom-up backpressure on a SAFE-to-spawn preflight only.
	if !res.OK {
		return res
	}
	pressured, cause := gatePressure(g)
	if !pressured || res.Live >= res.Cap {
		return res
	}
	// Hold at max(live, floor). res.Live < res.Cap holds here (a SPAWN_OK preflight
	// has headroom), so a floor at or above the cap means the gate cannot bind -- it
	// never RAISES the cap, so leave the preflight untouched.
	hold := res.Live
	if floor := g.floor(); hold < floor {
		hold = floor
	}
	if hold >= res.Cap {
		return res
	}
	res.Cap = hold
	res.Headroom = hold - res.Live
	res.CapTerms.EffectiveCap = hold
	res.CapTerms.Limiting = "gate"
	if res.Headroom > 0 {
		// Cold fleet: the gate lowered the cap to the floor (throttling growth) but
		// left a minimal cold-start allowance, so the verdict stays SPAWN_OK.
		return res
	}
	// Warm fleet (live at/above the floor): no headroom above the hold -- freeze and
	// refuse with the closed GATE_PRESSURE token so the sweep terminates on it.
	res.OK = false
	res.Verdict = PreflightRefuseGate
	res.Reason = gatePressureReason(cause, g, res.Live)
	return res
}

// gatePressureReason names the closed GATE_PRESSURE refusal token and cites the
// measured cause (GATE_LATENCY_REGRESSION or OVERHEAD_BUDGET_EXCEEDED) so a reader --
// and `dos man wedge <TOKEN> --explain` -- can bind both the refusal class and its evidence.
func gatePressureReason(cause string, g GateCheck, live int) string {
	if cause == turntaxmeter.OverheadBudgetExceeded {
		return fmt.Sprintf("%s: a lifecycle rung breached its overhead budget (%s); holding the fleet at %d live worker(s) - admit no new load onto a slow kernel until the regression clears",
			GatePressure, cause, live)
	}
	return fmt.Sprintf("%s: measured hook p99 %.1fms over the %.0fms budget (%s, n=%d); holding the fleet at %d live worker(s) - admit no new load onto a slow kernel until the tail recovers",
		GatePressure, g.Hook.P99MS, g.HookBudgetMS, cause, g.Hook.Count, live)
}
