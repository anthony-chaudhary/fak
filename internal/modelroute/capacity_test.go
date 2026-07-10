package modelroute

import "testing"

// The capacity kernel's load-bearing promise is conservative-degrade: an absent
// capacity signal must NEVER reroute and must leave today's route byte-identical.
// These tests pin that, plus each ceiling's trip condition and the sourced constants.

func TestAssessCapacity_ConservativeDegradeOnNoSignal(t *testing.T) {
	// Empty demand + empty ceiling => no actionable signal => CAPACITY_UNKNOWN, never reroute.
	v := AssessCapacity(CapacityDemand{}, CapacityCeiling{})
	if v.Reason != CapacityUnknown {
		t.Fatalf("no-signal reason = %q, want %q", v.Reason, CapacityUnknown)
	}
	if v.Reroute() {
		t.Fatal("no-signal verdict must not reroute")
	}
}

func TestAssessCapacity_ParamCeilingConfiguredButNoModelSizeDegrades(t *testing.T) {
	// A ceiling with no measured model size is not actionable on the param axis.
	v := AssessCapacity(CapacityDemand{}, CapacityCeiling{FaithfulParamsB: LocalFaithfulCeilingBillion})
	if v.Reason != CapacityUnknown {
		t.Fatalf("reason = %q, want %q (no model size to compare)", v.Reason, CapacityUnknown)
	}
}

func TestAssessCapacity_OverParamCeilingReroutes(t *testing.T) {
	v := AssessCapacity(
		CapacityDemand{ModelParamsB: 70},
		CapacityCeiling{FaithfulParamsB: LocalFaithfulCeilingBillion},
	)
	if !v.Reroute() {
		t.Fatalf("70B over a 7B ceiling must reroute, got %q", v.Reason)
	}
	if !v.OverParam || v.OverWindow {
		t.Fatalf("OverParam=%v OverWindow=%v, want param-only", v.OverParam, v.OverWindow)
	}
}

func TestAssessCapacity_AtParamCeilingFits(t *testing.T) {
	// "<= 7B faithful": a model exactly at the ceiling is served locally (strict >).
	v := AssessCapacity(
		CapacityDemand{ModelParamsB: 7.0},
		CapacityCeiling{FaithfulParamsB: LocalFaithfulCeilingBillion},
	)
	if v.Reason != CapacityOK {
		t.Fatalf("7B at a 7B ceiling should fit, got %q", v.Reason)
	}
}

func TestAssessCapacity_WindowOversubscribeReroutes(t *testing.T) {
	// window 1000, default 15% headroom => usable 850; demand 900 > 850 => reroute.
	v := AssessCapacity(
		CapacityDemand{ContextWindow: 1000, PromptTokens: 800, ExpectedOutputTokens: 100},
		CapacityCeiling{},
	)
	if !v.Reroute() {
		t.Fatalf("900 tok over an 850 usable window must reroute, got %q", v.Reason)
	}
	if v.OverParam || !v.OverWindow {
		t.Fatalf("OverParam=%v OverWindow=%v, want window-only", v.OverParam, v.OverWindow)
	}
}

func TestAssessCapacity_WindowFitsUnderHeadroom(t *testing.T) {
	// demand 800 <= usable 850 => fits.
	v := AssessCapacity(
		CapacityDemand{ContextWindow: 1000, PromptTokens: 800},
		CapacityCeiling{},
	)
	if v.Reason != CapacityOK {
		t.Fatalf("800 tok under an 850 usable window should fit, got %q (%s)", v.Reason, v.Detail)
	}
}

func TestAssessCapacity_ZeroWindowIsUnboundedNotZeroCapacity(t *testing.T) {
	// ContextWindow == 0 means unknown/unbounded, NOT a zero-token window; the window
	// axis is inert and a huge prompt does not reroute on that axis alone.
	v := AssessCapacity(
		CapacityDemand{ContextWindow: 0, PromptTokens: 1_000_000},
		CapacityCeiling{},
	)
	if v.Reason != CapacityUnknown {
		t.Fatalf("zero-window demand must degrade, got %q", v.Reason)
	}
}

func TestAssessCapacity_CustomHeadroom(t *testing.T) {
	// HeadroomFraction 0.5 => usable 500; demand 600 > 500 => reroute.
	v := AssessCapacity(
		CapacityDemand{ContextWindow: 1000, PromptTokens: 600},
		CapacityCeiling{HeadroomFraction: 0.5},
	)
	if !v.Reroute() || !v.OverWindow {
		t.Fatalf("600 tok over a 500 usable window (50%% headroom) must reroute, got %q", v.Reason)
	}
}

func TestAssessCapacity_OutOfRangeHeadroomFallsBackToDefault(t *testing.T) {
	// An out-of-range headroom (>=1) falls back to the 15% default rather than
	// zeroing usable capacity and rerouting everything.
	v := AssessCapacity(
		CapacityDemand{ContextWindow: 1000, PromptTokens: 800},
		CapacityCeiling{HeadroomFraction: 5},
	)
	if v.Reason != CapacityOK {
		t.Fatalf("out-of-range headroom should fall back to default (usable 850), got %q", v.Reason)
	}
}

func TestAssessCapacity_BothCeilingsTrip(t *testing.T) {
	v := AssessCapacity(
		CapacityDemand{ModelParamsB: 70, ContextWindow: 1000, PromptTokens: 900},
		CapacityCeiling{FaithfulParamsB: LocalFaithfulCeilingBillion},
	)
	if !v.Reroute() {
		t.Fatalf("over both ceilings must reroute, got %q", v.Reason)
	}
	if !v.OverParam || !v.OverWindow {
		t.Fatalf("OverParam=%v OverWindow=%v, want both", v.OverParam, v.OverWindow)
	}
}

func TestAssessCapacity_ParamSignalOnlyFits(t *testing.T) {
	// Param axis configured and fits, no window signal => decided OK (not Unknown).
	v := AssessCapacity(
		CapacityDemand{ModelParamsB: 6.9},
		CapacityCeiling{FaithfulParamsB: LocalFaithfulCeilingBillion},
	)
	if v.Reason != CapacityOK {
		t.Fatalf("6.9B under a 7B ceiling should fit, got %q", v.Reason)
	}
}

func TestCapacityReasonClosedSet(t *testing.T) {
	for _, r := range []CapacityReason{CapacityOK, CapacityReroute, CapacityUnknown, SeatExhausted} {
		if !r.Valid() {
			t.Errorf("%q should be a known CapacityReason", r)
		}
	}
	if CapacityReason("CAPACITY_BOGUS").Valid() {
		t.Error("an unknown token must not validate as a CapacityReason")
	}
}

func TestSourcedCapacityConstants(t *testing.T) {
	// Pin the documented constants so a silent edit is caught (docs cite 7B and 15%).
	if LocalFaithfulCeilingBillion != 7.0 {
		t.Errorf("LocalFaithfulCeilingBillion = %v, want 7.0 (hardware-limits-and-capacity.md:72)", LocalFaithfulCeilingBillion)
	}
	if DefaultDeviceHeadroom != 0.15 {
		t.Errorf("DefaultDeviceHeadroom = %v, want 0.15 (hardware-limits-and-capacity.md:303)", DefaultDeviceHeadroom)
	}
}

// --- Seat-pool exhaustion trigger (#3577) -------------------------------------------

func TestAssessCapacity_SeatExhaustedReroutes(t *testing.T) {
	// SeatPoolExhausted is the sole signal (no local ceiling, no window demand):
	// the provider seat pool is blocked fleet-wide => reroute to fleet compute with
	// the distinct SEAT_EXHAUSTED cause, not a model-shape CAPACITY_REROUTE.
	v := AssessCapacity(CapacityDemand{SeatPoolExhausted: true}, CapacityCeiling{})
	if v.Reason != SeatExhausted {
		t.Fatalf("seat-exhausted reason = %q, want %q", v.Reason, SeatExhausted)
	}
	if !v.Reroute() {
		t.Fatal("seat-exhausted verdict must reroute")
	}
	if v.OverParam || v.OverWindow {
		t.Fatalf("seat exhaustion is not a model-shape overflow: OverParam=%v OverWindow=%v, want both false", v.OverParam, v.OverWindow)
	}
}

func TestAssessCapacity_SeatNotExhaustedIsInertNoSignal(t *testing.T) {
	// The absent-signal invariant on the seat axis: SeatPoolExhausted=false with no
	// other signal must degrade to CAPACITY_UNKNOWN, never reroute. A false is not
	// evidence of a healthy pool — it is simply no evidence.
	v := AssessCapacity(CapacityDemand{SeatPoolExhausted: false}, CapacityCeiling{})
	if v.Reason != CapacityUnknown {
		t.Fatalf("seat-not-exhausted, no other signal => %q, want %q", v.Reason, CapacityUnknown)
	}
	if v.Reroute() {
		t.Fatal("an absent seat signal must not reroute")
	}
}

func TestAssessCapacity_SeatExhaustedIndependentOfModelFit(t *testing.T) {
	// A model that FITS every local ceiling still reroutes when the seat pool is
	// blocked: the seat trigger is orthogonal to the model-shape ceilings.
	v := AssessCapacity(
		CapacityDemand{ModelParamsB: 3, ContextWindow: 8000, PromptTokens: 100, SeatPoolExhausted: true},
		CapacityCeiling{FaithfulParamsB: LocalFaithfulCeilingBillion},
	)
	if v.Reason != SeatExhausted {
		t.Fatalf("a fitting model with an exhausted seat pool => %q, want %q", v.Reason, SeatExhausted)
	}
	if !v.Reroute() {
		t.Fatal("seat exhaustion must reroute even when the model fits locally")
	}
}

func TestAssessCapacity_ModelShapeOverflowPrecedesSeat(t *testing.T) {
	// When a local ceiling ALSO trips, the model-shape overflow is reported as the
	// cause (CAPACITY_REROUTE with its OverParam flag) so the reroute destination can
	// honor the size need; either way the task reroutes. Pins the intended precedence.
	v := AssessCapacity(
		CapacityDemand{ModelParamsB: 70, SeatPoolExhausted: true},
		CapacityCeiling{FaithfulParamsB: LocalFaithfulCeilingBillion},
	)
	if v.Reason != CapacityReroute || !v.OverParam {
		t.Fatalf("param overflow + seat exhaustion => reason %q OverParam=%v, want CAPACITY_REROUTE + OverParam", v.Reason, v.OverParam)
	}
	if !v.Reroute() {
		t.Fatal("param overflow with seat exhaustion must still reroute")
	}
}

// --- RouteWithCapacity end-to-end over a manifest -----------------------------------

// capacityManifest is a minimal manifest whose TOP rule fires only on the reroute
// label, with a distinct default — so a reroute is observable as a rule-name change.
func capacityManifest() Manifest {
	return Manifest{
		Version: "test",
		Default: Plan{Members: []Member{{Model: "local-small"}}, Reason: "default local"},
		Rules: []Rule{{
			Name:  "capacity-reroute-to-fleet",
			Match: Match{Labels: map[string]string{CapacityRerouteLabel: CapacityRerouteValue}},
			Plan:  Plan{Members: []Member{{Model: "fleet-large"}}, Reason: "local ceiling exceeded -> fleet GPU"},
		}},
	}
}

func TestRouteWithCapacity_RerouteStampsLabelAndPicksFleetRule(t *testing.T) {
	m := capacityManifest()
	s := Subject{Aspect: AspectRequest, PromptTokens: 900}
	dec, v := m.RouteWithCapacity(s,
		CapacityDemand{ModelParamsB: 70, ContextWindow: 1000, PromptTokens: 900},
		CapacityCeiling{FaithfulParamsB: LocalFaithfulCeilingBillion},
	)
	if !v.Reroute() {
		t.Fatalf("verdict should reroute, got %q", v.Reason)
	}
	if !dec.Matched || dec.RuleName != "capacity-reroute-to-fleet" {
		t.Fatalf("reroute should fire the fleet rule, got matched=%v rule=%q", dec.Matched, dec.RuleName)
	}
	if got := dec.Plan.Primary(); got != "fleet-large" {
		t.Fatalf("reroute should select fleet-large, got %q", got)
	}
}

func TestRouteWithCapacity_NoSignalRoutesByteIdentical(t *testing.T) {
	m := capacityManifest()
	s := Subject{Aspect: AspectRequest, PromptTokens: 100, Labels: map[string]string{"tenant": "acme"}}

	// With no capacity signal, RouteWithCapacity must equal a plain Route(s).
	want := m.Route(s)
	got, v := m.RouteWithCapacity(s, CapacityDemand{}, CapacityCeiling{})

	if v.Reason != CapacityUnknown {
		t.Fatalf("no-signal verdict = %q, want %q", v.Reason, CapacityUnknown)
	}
	if got.Matched != want.Matched || got.RuleName != want.RuleName || got.Plan.Primary() != want.Plan.Primary() {
		t.Fatalf("no-signal route diverged from Route(s): got %+v want %+v", got, want)
	}
	// The subject echoed in the decision must be unchanged (no capacity label leaked in).
	if _, ok := got.Subject.Labels[CapacityRerouteLabel]; ok {
		t.Fatal("no-signal route must not stamp the capacity label")
	}
}

func TestRouteWithCapacity_FitsRoutesLocalNotFleet(t *testing.T) {
	m := capacityManifest()
	s := Subject{Aspect: AspectRequest}
	dec, v := m.RouteWithCapacity(s,
		CapacityDemand{ModelParamsB: 3, ContextWindow: 8000, PromptTokens: 100},
		CapacityCeiling{FaithfulParamsB: LocalFaithfulCeilingBillion},
	)
	if v.Reason != CapacityOK {
		t.Fatalf("a small model that fits should be CAPACITY_OK, got %q", v.Reason)
	}
	if dec.Matched || dec.Plan.Primary() != "local-small" {
		t.Fatalf("a fitting task should fall to the local default, got matched=%v primary=%q", dec.Matched, dec.Plan.Primary())
	}
}

func TestRouteWithCapacity_DoesNotMutateCallerLabels(t *testing.T) {
	m := capacityManifest()
	shared := map[string]string{"tenant": "acme"}
	s := Subject{Aspect: AspectRequest, PromptTokens: 900, Labels: shared}

	_, v := m.RouteWithCapacity(s,
		CapacityDemand{ModelParamsB: 70},
		CapacityCeiling{FaithfulParamsB: LocalFaithfulCeilingBillion},
	)
	if !v.Reroute() {
		t.Fatalf("precondition: expected a reroute, got %q", v.Reason)
	}
	if _, ok := shared[CapacityRerouteLabel]; ok {
		t.Fatal("reroute must not mutate the caller's shared Labels map")
	}
	if len(shared) != 1 {
		t.Fatalf("caller's Labels map was mutated: %v", shared)
	}
}

func TestRouteWithCapacity_SeatExhaustedStampsLabelAndPicksFleetRule(t *testing.T) {
	// The seat-exhaustion trigger reuses the SAME reroute label, so #3520's match
	// rule fires unchanged — a seat-blocked task spills to the fleet-GPU lane just
	// like a model-shape overflow does.
	m := capacityManifest()
	s := Subject{Aspect: AspectRequest, PromptTokens: 100}
	dec, v := m.RouteWithCapacity(s,
		CapacityDemand{SeatPoolExhausted: true},
		CapacityCeiling{},
	)
	if v.Reason != SeatExhausted || !v.Reroute() {
		t.Fatalf("verdict should be a SEAT_EXHAUSTED reroute, got %q reroute=%v", v.Reason, v.Reroute())
	}
	if !dec.Matched || dec.RuleName != "capacity-reroute-to-fleet" {
		t.Fatalf("seat exhaustion should fire the fleet rule, got matched=%v rule=%q", dec.Matched, dec.RuleName)
	}
	if got := dec.Plan.Primary(); got != "fleet-large" {
		t.Fatalf("seat-exhausted reroute should select fleet-large, got %q", got)
	}
}
