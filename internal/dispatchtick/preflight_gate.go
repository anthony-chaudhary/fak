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
// effective cap, never raises it, and when it is the SOLE binding term the tick
// refuses with the closed GATE_PRESSURE token so the sweep terminates on it like any
// other refusal.
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
// declaration so the token this fold emits is the one `dos check-reason` verifies and
// the loop routes to a replan.
const GatePressure = "GATE_PRESSURE"

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
// fifth cap term. Degraded gate health freezes the fleet at its current live count --
// admit no NEW worker onto a slow kernel -- which can only LOWER the effective cap.
//
// The fold is a no-op when the preflight ALREADY refused for a higher-precedence
// reason (host / seat / at-cap / account): the fleet is then already not growing, so
// gate pressure is not the sole binding term and the existing verdict stands. It is
// also a no-op on a healthy gate or a thin sample. Only a would-be SPAWN_OK whose
// headroom the gate removes flips to PreflightRefuseGate with a GATE_PRESSURE-token
// reason that cites the measured cause.
func ApplyGateBackpressure(res PreflightResult, g GateCheck) PreflightResult {
	// The gate is bottom-up backpressure on a SAFE-to-spawn preflight only.
	if !res.OK {
		return res
	}
	pressured, cause := gatePressure(g)
	if !pressured || res.Live >= res.Cap {
		return res
	}
	// Freeze at the live count -- admit no NEW worker onto a slow kernel. res.Live <
	// res.Cap holds here (a SPAWN_OK preflight has headroom), so this only LOWERS the
	// effective cap.
	res.Cap = res.Live
	res.Headroom = 0
	res.CapTerms.EffectiveCap = res.Cap
	res.CapTerms.Limiting = "gate"
	res.OK = false
	res.Verdict = PreflightRefuseGate
	res.Reason = gatePressureReason(cause, g, res.Live)
	return res
}

// gatePressureReason names the closed GATE_PRESSURE refusal token and cites the
// measured cause (GATE_LATENCY_REGRESSION or OVERHEAD_BUDGET_EXCEEDED) so a reader --
// and `dos check-reason` -- can bind both the refusal class and its evidence.
func gatePressureReason(cause string, g GateCheck, live int) string {
	if cause == turntaxmeter.OverheadBudgetExceeded {
		return fmt.Sprintf("%s: a lifecycle rung breached its overhead budget (%s); holding the fleet at %d live worker(s) - admit no new load onto a slow kernel until the regression clears",
			GatePressure, cause, live)
	}
	return fmt.Sprintf("%s: measured hook p99 %.1fms over the %.0fms budget (%s, n=%d); holding the fleet at %d live worker(s) - admit no new load onto a slow kernel until the tail recovers",
		GatePressure, g.Hook.P99MS, g.HookBudgetMS, cause, g.Hook.Count, live)
}
