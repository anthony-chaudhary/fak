package dispatchtick

import (
	"strings"
	"testing"
)

// churnBaseline returns a SPAWN_OK preflight with real headroom (cap 5, live 2) so a
// host-storm fold that freezes at the live count is a visible cap reduction (5 -> 2).
func churnBaseline(t *testing.T) PreflightResult {
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

// coldChurnBaseline returns a SPAWN_OK preflight for a COLD fleet (0 live, cap 5) so a
// storm fold's cold-start floor is a visible probe allowance rather than a deadlock.
func coldChurnBaseline(t *testing.T) PreflightResult {
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

func TestApplyChurnBackpressureBurstLowersCapAndRefuses(t *testing.T) {
	// A measured whole-host spawn burst (10 >= the default 8 threshold) is the host_churn
	// cap term: it freezes the warm fleet at its live count and refuses with
	// HOST_CHURN_BACKOFF so the sweep stops backing off this tick.
	got := ApplyChurnBackpressure(churnBaseline(t), ChurnCheck{Recent: 10})
	if got.OK {
		t.Fatalf("a storm must refuse (the sweep stops on !ok); got ok=true verdict=%s", got.Verdict)
	}
	if got.Verdict != PreflightRefuseChurn {
		t.Fatalf("verdict = %s, want %s", got.Verdict, PreflightRefuseChurn)
	}
	if got.Cap != 2 || got.Headroom != 0 {
		t.Fatalf("cap/headroom = %d/%d, want the fleet frozen at live: 2/0", got.Cap, got.Headroom)
	}
	if got.CapTerms.EffectiveCap != 2 || got.CapTerms.Limiting != "host_churn" {
		t.Fatalf("cap_terms effective/limiting = %d/%q, want 2/host_churn", got.CapTerms.EffectiveCap, got.CapTerms.Limiting)
	}
	if !strings.Contains(got.Reason, HostChurnBackoff) {
		t.Fatalf("reason must name the %s refusal token; got %q", HostChurnBackoff, got.Reason)
	}
	// The cross-dispatcher rationale is load-bearing enough to state in the reason.
	if !strings.Contains(got.Reason, "cadence floor") {
		t.Fatalf("reason must explain it is the cross-dispatcher gate the per-loop cadence floor cannot see; got %q", got.Reason)
	}
	if v := got.Map()["verdict"]; v != PreflightRefuseChurn {
		t.Fatalf("map verdict = %v, want %s", v, PreflightRefuseChurn)
	}
}

func TestApplyChurnBackpressureColdStartFloorKeepsMinimalSpawn(t *testing.T) {
	// A cold fleet that just lost every worker into a storm must NOT deadlock at a zero cap:
	// the floor holds it to one cold-start probe and keeps the verdict SPAWN_OK, so the fleet
	// can witness whether the storm subsided.
	got := ApplyChurnBackpressure(coldChurnBaseline(t), ChurnCheck{Recent: 12})
	if !got.OK || got.Verdict != PreflightOKVerdict {
		t.Fatalf("cold fleet under a storm must stay SPAWN_OK (throttle growth, not liveness); got ok=%v verdict=%s", got.OK, got.Verdict)
	}
	if got.Cap != DefaultChurnMinWorkers || got.Headroom != DefaultChurnMinWorkers {
		t.Fatalf("cap/headroom = %d/%d, want the cold-start floor %d/%d", got.Cap, got.Headroom, DefaultChurnMinWorkers, DefaultChurnMinWorkers)
	}
	if got.CapTerms.EffectiveCap != DefaultChurnMinWorkers || got.CapTerms.Limiting != "host_churn" {
		t.Fatalf("cap_terms effective/limiting = %d/%q, want %d/host_churn", got.CapTerms.EffectiveCap, got.CapTerms.Limiting, DefaultChurnMinWorkers)
	}
}

func TestApplyChurnBackpressureCustomFloorAllowsMoreColdStart(t *testing.T) {
	// A raised floor keeps more probes alive under a storm; the term still only LOWERS the
	// cap (5 -> 3), never raises it.
	got := ApplyChurnBackpressure(coldChurnBaseline(t), ChurnCheck{Recent: 12, MinWorkers: 3})
	if !got.OK || got.Cap != 3 || got.Headroom != 3 {
		t.Fatalf("custom floor 3: got ok=%v cap/headroom=%d/%d, want SPAWN_OK 3/3", got.OK, got.Cap, got.Headroom)
	}
}

func TestApplyChurnBackpressureFloorAtOrAboveCapIsNoOp(t *testing.T) {
	// The term cannot manufacture capacity: a floor at or above the existing cap leaves the
	// SPAWN_OK preflight untouched rather than relabeling it.
	base := coldChurnBaseline(t)
	got := ApplyChurnBackpressure(base, ChurnCheck{Recent: 12, MinWorkers: 10})
	if !sameAdmission(got, base) {
		t.Fatalf("floor above cap must be a no-op; got %+v, want %+v", got, base)
	}
}

func TestApplyChurnBackpressureBelowThresholdAbstains(t *testing.T) {
	// Ordinary process turnover below the arming threshold is noise, not a storm: the fold
	// must not touch the verdict.
	base := churnBaseline(t)
	got := ApplyChurnBackpressure(base, ChurnCheck{Recent: DefaultChurnBurstThreshold - 1})
	if got.Verdict != PreflightOKVerdict || got.Cap != base.Cap {
		t.Fatalf("below-threshold must abstain: got %s cap=%d, want SPAWN_OK cap=%d", got.Verdict, got.Cap, base.Cap)
	}
}

func TestApplyChurnBackpressureCustomThresholdArms(t *testing.T) {
	// A lowered threshold lets an operator arm the backoff on a smaller burst; four spawns at
	// Threshold=4 now bind and freeze the warm fleet.
	got := ApplyChurnBackpressure(churnBaseline(t), ChurnCheck{Recent: 4, Threshold: 4})
	if got.OK || got.Verdict != PreflightRefuseChurn || got.Cap != 2 {
		t.Fatalf("custom threshold 4: got ok=%v verdict=%s cap=%d, want REFUSE_HOST_CHURN cap=2", got.OK, got.Verdict, got.Cap)
	}
}

func TestApplyChurnBackpressureZeroValueIsNoOp(t *testing.T) {
	// The zero-value check (nothing wired) never lowers the cap, so a caller that wires
	// nothing keeps the existing behavior byte-for-byte.
	base := churnBaseline(t)
	got := ApplyChurnBackpressure(base, ChurnCheck{})
	if !sameAdmission(got, base) {
		t.Fatalf("zero-value check must be a no-op; got %+v, want %+v", got, base)
	}
}

func TestApplyChurnBackpressureDoesNotOverridePriorRefusal(t *testing.T) {
	// The churn term lowers the cap only when it is the SOLE binding term. A preflight that
	// already refused at cap keeps its higher-precedence verdict untouched.
	in := preflightInput()
	in.MaxWorkers = 2
	in.Kernel = KernelCheck{Alive: IntPtr(5), Target: IntPtr(9), Verdict: "OVER_TARGET"}
	atCap := EvaluatePreflight(in)
	if atCap.Verdict != PreflightRefuseAtCap {
		t.Fatalf("precondition: verdict = %s, want REFUSE_AT_CAP", atCap.Verdict)
	}
	got := ApplyChurnBackpressure(atCap, ChurnCheck{Recent: 20})
	if !sameAdmission(got, atCap) {
		t.Fatalf("churn term must not override a prior refusal; got %+v, want %+v", got, atCap)
	}
}
