package dispatchtick

import (
	"strings"
	"testing"
	"time"
)

// rateBaseline returns a SPAWN_OK preflight with real headroom (cap 5, live 2) so a
// rate-burst fold that freezes at the live count is a visible cap reduction (5 -> 2).
func rateBaseline(t *testing.T) PreflightResult {
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

// coldRateBaseline returns a SPAWN_OK preflight for a COLD backend (0 live, cap 5) so a
// burst fold's cold-start floor is a visible probe allowance rather than a deadlock.
func coldRateBaseline(t *testing.T) PreflightResult {
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

func TestApplyRateLimitBackpressureBurstLowersCapAndRefuses(t *testing.T) {
	// A measured burst of rate_limit worker exits (5 >= the default 3 threshold) is the
	// rate_budget cap term: it freezes the backend at its live count and refuses with
	// RATE_LIMIT_BACKOFF so the sweep routes to a different provider.
	got := ApplyRateLimitBackpressure(rateBaseline(t), RateLimitCheck{Recent: 5, Window: 15 * time.Minute})
	if got.OK {
		t.Fatalf("a burst must refuse (the sweep stops on !ok); got ok=true verdict=%s", got.Verdict)
	}
	if got.Verdict != PreflightRefuseRateLimit {
		t.Fatalf("verdict = %s, want %s", got.Verdict, PreflightRefuseRateLimit)
	}
	if got.Cap != 2 || got.Headroom != 0 {
		t.Fatalf("cap/headroom = %d/%d, want the backend frozen at live: 2/0", got.Cap, got.Headroom)
	}
	if got.CapTerms.EffectiveCap != 2 || got.CapTerms.Limiting != "rate" {
		t.Fatalf("cap_terms effective/limiting = %d/%q, want 2/rate", got.CapTerms.EffectiveCap, got.CapTerms.Limiting)
	}
	if !strings.Contains(got.Reason, RateLimitBackoff) {
		t.Fatalf("reason must name the %s refusal token; got %q", RateLimitBackoff, got.Reason)
	}
	// The disambiguation is load-bearing enough to state in the reason: fake 429s are excluded.
	if !strings.Contains(got.Reason, "excluded") {
		t.Fatalf("reason must state that weekly/model/login walls are excluded; got %q", got.Reason)
	}
	if v := got.Map()["verdict"]; v != PreflightRefuseRateLimit {
		t.Fatalf("map verdict = %v, want %s", v, PreflightRefuseRateLimit)
	}
}

func TestApplyRateLimitBackpressureColdStartFloorKeepsMinimalSpawn(t *testing.T) {
	// A cold backend that just lost every worker to a burst must NOT deadlock at a zero
	// cap: the floor holds it to one cold-start probe and keeps the verdict SPAWN_OK, so
	// the fleet can witness whether the overload cleared.
	got := ApplyRateLimitBackpressure(coldRateBaseline(t), RateLimitCheck{Recent: 8, Window: 15 * time.Minute})
	if !got.OK || got.Verdict != PreflightOKVerdict {
		t.Fatalf("cold backend under a burst must stay SPAWN_OK (throttle growth, not liveness); got ok=%v verdict=%s", got.OK, got.Verdict)
	}
	if got.Cap != DefaultRateLimitMinWorkers || got.Headroom != DefaultRateLimitMinWorkers {
		t.Fatalf("cap/headroom = %d/%d, want the cold-start floor %d/%d", got.Cap, got.Headroom, DefaultRateLimitMinWorkers, DefaultRateLimitMinWorkers)
	}
	if got.CapTerms.EffectiveCap != DefaultRateLimitMinWorkers || got.CapTerms.Limiting != "rate" {
		t.Fatalf("cap_terms effective/limiting = %d/%q, want %d/rate", got.CapTerms.EffectiveCap, got.CapTerms.Limiting, DefaultRateLimitMinWorkers)
	}
}

func TestApplyRateLimitBackpressureCustomFloorAllowsMoreColdStart(t *testing.T) {
	// A raised floor keeps more probes alive under a burst; the term still only LOWERS
	// the cap (5 -> 3), never raises it.
	got := ApplyRateLimitBackpressure(coldRateBaseline(t), RateLimitCheck{Recent: 8, Window: 15 * time.Minute, MinWorkers: 3})
	if !got.OK || got.Cap != 3 || got.Headroom != 3 {
		t.Fatalf("custom floor 3: got ok=%v cap/headroom=%d/%d, want SPAWN_OK 3/3", got.OK, got.Cap, got.Headroom)
	}
}

func TestApplyRateLimitBackpressureFloorAtOrAboveCapIsNoOp(t *testing.T) {
	// The term cannot manufacture capacity: a floor at or above the existing cap leaves
	// the SPAWN_OK preflight untouched rather than relabeling it.
	base := coldRateBaseline(t)
	got := ApplyRateLimitBackpressure(base, RateLimitCheck{Recent: 8, Window: 15 * time.Minute, MinWorkers: 10})
	if !sameAdmission(got, base) {
		t.Fatalf("floor above cap must be a no-op; got %+v, want %+v", got, base)
	}
}

func TestApplyRateLimitBackpressureBelowThresholdAbstains(t *testing.T) {
	// A single stray rate_limit exit is noise, not a burst: below the arming threshold
	// the fold must not touch the verdict.
	base := rateBaseline(t)
	got := ApplyRateLimitBackpressure(base, RateLimitCheck{Recent: DefaultRateLimitMin429 - 1, Window: 15 * time.Minute})
	if got.Verdict != PreflightOKVerdict || got.Cap != base.Cap {
		t.Fatalf("below-threshold must abstain: got %s cap=%d, want SPAWN_OK cap=%d", got.Verdict, got.Cap, base.Cap)
	}
}

func TestApplyRateLimitBackpressureCustomThresholdArms(t *testing.T) {
	// A lowered threshold lets an operator arm the backoff on a smaller cluster; two
	// exits at Threshold=2 now bind and freeze the warm fleet.
	got := ApplyRateLimitBackpressure(rateBaseline(t), RateLimitCheck{Recent: 2, Threshold: 2, Window: 15 * time.Minute})
	if got.OK || got.Verdict != PreflightRefuseRateLimit || got.Cap != 2 {
		t.Fatalf("custom threshold 2: got ok=%v verdict=%s cap=%d, want REFUSE_RATE_LIMIT cap=2", got.OK, got.Verdict, got.Cap)
	}
}

func TestApplyRateLimitBackpressureZeroValueIsNoOp(t *testing.T) {
	// The zero-value check (nothing wired) never lowers the cap.
	base := rateBaseline(t)
	got := ApplyRateLimitBackpressure(base, RateLimitCheck{})
	if !sameAdmission(got, base) {
		t.Fatalf("zero-value check must be a no-op; got %+v, want %+v", got, base)
	}
}

func TestApplyRateLimitBackpressureDoesNotOverridePriorRefusal(t *testing.T) {
	// The rate term lowers the cap only when it is the SOLE binding term. A preflight that
	// already refused at cap keeps its higher-precedence verdict untouched.
	in := preflightInput()
	in.MaxWorkers = 2
	in.Kernel = KernelCheck{Alive: IntPtr(5), Target: IntPtr(9), Verdict: "OVER_TARGET"}
	atCap := EvaluatePreflight(in)
	if atCap.Verdict != PreflightRefuseAtCap {
		t.Fatalf("precondition: verdict = %s, want REFUSE_AT_CAP", atCap.Verdict)
	}
	got := ApplyRateLimitBackpressure(atCap, RateLimitCheck{Recent: 9, Window: 15 * time.Minute})
	if !sameAdmission(got, atCap) {
		t.Fatalf("rate term must not override a prior refusal; got %+v, want %+v", got, atCap)
	}
}
