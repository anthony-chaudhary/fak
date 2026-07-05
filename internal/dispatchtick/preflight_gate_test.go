package dispatchtick

import (
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/turntaxmeter"
)

// sameAdmission reports whether the fold left the admission decision untouched --
// PreflightResult embeds a map (HostCapacityInfo.Components) so it is not directly
// comparable; the fold only ever writes verdict/cap/headroom/reason/limiting.
func sameAdmission(a, b PreflightResult) bool {
	return a.OK == b.OK && a.Verdict == b.Verdict && a.Reason == b.Reason &&
		a.Cap == b.Cap && a.Headroom == b.Headroom &&
		a.CapTerms.EffectiveCap == b.CapTerms.EffectiveCap && a.CapTerms.Limiting == b.CapTerms.Limiting
}

// gateBaseline returns a SPAWN_OK preflight with real headroom (cap 5, live 2) so a
// gate fold that freezes at the live count is a visible cap reduction (5 -> 2).
func gateBaseline(t *testing.T) PreflightResult {
	t.Helper()
	in := preflightInput()
	in.MaxWorkers = 5
	in.Kernel = KernelCheck{Alive: IntPtr(2), Target: IntPtr(9), Verdict: "FILLING"}
	res := EvaluatePreflight(in)
	if res.Verdict != PreflightOKVerdict || res.Cap != 5 || res.Live != 2 || res.Headroom != 3 {
		t.Fatalf("baseline = %s cap/live/headroom=%d/%d/%d, want SPAWN_OK 5/2/3", res.Verdict, res.Cap, res.Live, res.Headroom)
	}
	return res
}

func TestApplyGateBackpressureBreachLowersCapAndRefuses(t *testing.T) {
	// An injected breached hook-latency rollup (p99 400ms over the 250ms budget, well
	// past MinHookAlarmSamples) is the fifth cap term: it freezes the fleet at its live
	// count and refuses with GATE_PRESSURE, citing the measured GATE_LATENCY_REGRESSION.
	got := ApplyGateBackpressure(gateBaseline(t), GateCheck{
		Hook:         turntaxmeter.HookLatencyStats{Count: 12, P99MS: 400},
		HookBudgetMS: turntaxmeter.DefaultHookP99BudgetMS,
	})
	if got.OK {
		t.Fatalf("gate breach must refuse (the sweep stops on !ok); got ok=true verdict=%s", got.Verdict)
	}
	if got.Verdict != PreflightRefuseGate {
		t.Fatalf("verdict = %s, want %s", got.Verdict, PreflightRefuseGate)
	}
	if got.Cap != 2 || got.Headroom != 0 {
		t.Fatalf("cap/headroom = %d/%d, want the fleet frozen at live: 2/0", got.Cap, got.Headroom)
	}
	if got.CapTerms.EffectiveCap != 2 || got.CapTerms.Limiting != "gate" {
		t.Fatalf("cap_terms effective/limiting = %d/%q, want 2/gate", got.CapTerms.EffectiveCap, got.CapTerms.Limiting)
	}
	if !strings.Contains(got.Reason, GatePressure) || !strings.Contains(got.Reason, turntaxmeter.GateLatencyRegression) {
		t.Fatalf("reason must name the %s refusal token and cite the measured %s cause; got %q",
			GatePressure, turntaxmeter.GateLatencyRegression, got.Reason)
	}
	// The refusal token is a real closed-vocabulary member (declared in dos.toml),
	// surfaced verbatim in the Map() the shell serializes.
	if v := got.Map()["verdict"]; v != PreflightRefuseGate {
		t.Fatalf("map verdict = %v, want %s", v, PreflightRefuseGate)
	}
}

// coldGateBaseline returns a SPAWN_OK preflight for a COLD fleet (0 live, cap 5) so a
// pressured fold's cold-start floor is a visible allowance rather than a deadlock.
func coldGateBaseline(t *testing.T) PreflightResult {
	t.Helper()
	in := preflightInput()
	in.MaxWorkers = 5
	in.Kernel = KernelCheck{Alive: IntPtr(0), Target: IntPtr(9), Verdict: "FILLING"}
	res := EvaluatePreflight(in)
	if res.Verdict != PreflightOKVerdict || res.Cap != 5 || res.Live != 0 || res.Headroom != 5 {
		t.Fatalf("cold baseline = %s cap/live/headroom=%d/%d/%d, want SPAWN_OK 5/0/5", res.Verdict, res.Cap, res.Live, res.Headroom)
	}
	return res
}

func TestApplyGateBackpressureColdStartFloorKeepsMinimalSpawn(t *testing.T) {
	// A cold fleet under genuine pressure must NOT deadlock at a zero cap: the floor
	// holds it to a minimal cold-start presence and keeps the verdict SPAWN_OK, so the
	// first worker can start and begin clearing the very backlog the gate reacts to.
	got := ApplyGateBackpressure(coldGateBaseline(t), GateCheck{
		Hook:         turntaxmeter.HookLatencyStats{Count: 40, P99MS: 900},
		HookBudgetMS: turntaxmeter.DefaultHookP99BudgetMS,
	})
	if !got.OK || got.Verdict != PreflightOKVerdict {
		t.Fatalf("cold fleet under pressure must stay SPAWN_OK (throttle growth, not liveness); got ok=%v verdict=%s", got.OK, got.Verdict)
	}
	if got.Cap != DefaultGateMinWorkers || got.Headroom != DefaultGateMinWorkers {
		t.Fatalf("cap/headroom = %d/%d, want the cold-start floor %d/%d", got.Cap, got.Headroom, DefaultGateMinWorkers, DefaultGateMinWorkers)
	}
	if got.CapTerms.EffectiveCap != DefaultGateMinWorkers || got.CapTerms.Limiting != "gate" {
		t.Fatalf("cap_terms effective/limiting = %d/%q, want %d/gate", got.CapTerms.EffectiveCap, got.CapTerms.Limiting, DefaultGateMinWorkers)
	}
}

func TestApplyGateBackpressureCustomFloorAllowsMoreColdStart(t *testing.T) {
	// A raised floor (FAK_GATE_MIN_WORKERS) lets an operator keep more workers alive
	// under pressure; the gate still only LOWERS the cap (5 -> 3), never raises it.
	got := ApplyGateBackpressure(coldGateBaseline(t), GateCheck{
		Hook:         turntaxmeter.HookLatencyStats{Count: 40, P99MS: 900},
		HookBudgetMS: turntaxmeter.DefaultHookP99BudgetMS,
		MinWorkers:   3,
	})
	if !got.OK || got.Cap != 3 || got.Headroom != 3 {
		t.Fatalf("custom floor 3: got ok=%v cap/headroom=%d/%d, want SPAWN_OK 3/3", got.OK, got.Cap, got.Headroom)
	}
}

func TestApplyGateBackpressureFloorAtOrAboveCapIsNoOp(t *testing.T) {
	// The gate cannot manufacture capacity: a floor at or above the existing cap
	// leaves the SPAWN_OK preflight untouched rather than relabeling it.
	base := coldGateBaseline(t)
	got := ApplyGateBackpressure(base, GateCheck{
		Hook:         turntaxmeter.HookLatencyStats{Count: 40, P99MS: 900},
		HookBudgetMS: turntaxmeter.DefaultHookP99BudgetMS,
		MinWorkers:   10,
	})
	if !sameAdmission(got, base) {
		t.Fatalf("floor above cap must be a no-op; got %+v, want %+v", got, base)
	}
}

func TestApplyGateBackpressureWarmFleetStillFreezesAndRefuses(t *testing.T) {
	// The warm-fleet contract is unchanged: a fleet already at/above the floor freezes
	// at its live count and refuses, so the sweep terminates on gate pressure.
	got := ApplyGateBackpressure(gateBaseline(t), GateCheck{
		Hook:         turntaxmeter.HookLatencyStats{Count: 40, P99MS: 900},
		HookBudgetMS: turntaxmeter.DefaultHookP99BudgetMS,
	})
	if got.OK || got.Verdict != PreflightRefuseGate || got.Cap != 2 || got.Headroom != 0 {
		t.Fatalf("warm fleet must freeze at live and refuse; got ok=%v verdict=%s cap/headroom=%d/%d, want REFUSE_GATE 2/0", got.OK, got.Verdict, got.Cap, got.Headroom)
	}
}

func TestApplyGateBackpressureOverheadBreachRefuses(t *testing.T) {
	got := ApplyGateBackpressure(gateBaseline(t), GateCheck{OverheadBreach: true})
	if got.OK || got.Verdict != PreflightRefuseGate {
		t.Fatalf("verdict = %s ok=%v, want REFUSE_GATE", got.Verdict, got.OK)
	}
	if !strings.Contains(got.Reason, GatePressure) || !strings.Contains(got.Reason, turntaxmeter.OverheadBudgetExceeded) {
		t.Fatalf("reason must name %s and cite %s; got %q", GatePressure, turntaxmeter.OverheadBudgetExceeded, got.Reason)
	}
}

func TestApplyGateBackpressureAbstainsOnThinSample(t *testing.T) {
	// A p99 over budget on a handful of rows is a spike, not a tail: JudgeHookLatency
	// reports Thin below MinHookAlarmSamples and the fold must not touch the verdict.
	base := gateBaseline(t)
	got := ApplyGateBackpressure(base, GateCheck{
		Hook:         turntaxmeter.HookLatencyStats{Count: turntaxmeter.MinHookAlarmSamples - 1, P99MS: 9000},
		HookBudgetMS: turntaxmeter.DefaultHookP99BudgetMS,
	})
	if got.Verdict != PreflightOKVerdict || got.Cap != base.Cap {
		t.Fatalf("thin sample must abstain: got %s cap=%d, want SPAWN_OK cap=%d", got.Verdict, got.Cap, base.Cap)
	}
}

func TestApplyGateBackpressureWithinBudgetIsNoOp(t *testing.T) {
	base := gateBaseline(t)
	got := ApplyGateBackpressure(base, GateCheck{
		Hook:         turntaxmeter.HookLatencyStats{Count: 40, P99MS: 100},
		HookBudgetMS: turntaxmeter.DefaultHookP99BudgetMS,
	})
	if !sameAdmission(got, base) {
		t.Fatalf("a within-budget rollup must be a no-op; got %+v, want %+v", got, base)
	}
}

func TestApplyGateBackpressureNoBudgetNeverBreaches(t *testing.T) {
	base := gateBaseline(t)
	got := ApplyGateBackpressure(base, GateCheck{
		Hook:         turntaxmeter.HookLatencyStats{Count: 40, P99MS: 9000},
		HookBudgetMS: 0, // report-only: no declared budget can never breach
	})
	if !sameAdmission(got, base) {
		t.Fatalf("no-budget rollup must be a no-op; got %+v, want %+v", got, base)
	}
}

func TestApplyGateBackpressureDoesNotOverridePriorRefusal(t *testing.T) {
	// Gate pressure lowers the cap only when it is the SOLE binding term. A preflight
	// that already refused at cap keeps its higher-precedence verdict untouched.
	in := preflightInput()
	in.MaxWorkers = 2
	in.Kernel = KernelCheck{Alive: IntPtr(5), Target: IntPtr(9), Verdict: "OVER_TARGET"}
	atCap := EvaluatePreflight(in)
	if atCap.Verdict != PreflightRefuseAtCap {
		t.Fatalf("precondition: verdict = %s, want REFUSE_AT_CAP", atCap.Verdict)
	}
	got := ApplyGateBackpressure(atCap, GateCheck{
		Hook:         turntaxmeter.HookLatencyStats{Count: 12, P99MS: 400},
		HookBudgetMS: turntaxmeter.DefaultHookP99BudgetMS,
	})
	if !sameAdmission(got, atCap) {
		t.Fatalf("gate must not override a prior refusal; got %+v, want %+v", got, atCap)
	}
}
