package modelroute

import "testing"

// TestUndeclaredWorkStaysOffTheCheapRungs is the honesty claim of the whole placement
// story: fak does not move your traffic onto a laptop because a tool name looked
// harmless. Absent a declaration the subject carries no class, PolicyFor puts it at the
// strictest floor, and the placement lands where it landed before — a vendor.
func TestUndeclaredWorkStaysOffTheCheapRungs(t *testing.T) {
	c := ClassOf(Subject{Aspect: AspectToolCall, Tool: "grep"})
	if c.Declared {
		t.Fatalf("a bare tool call must not be treated as declared: %+v", c)
	}
	if c.Class != "" {
		t.Fatalf("undeclared subject carries class %q, want the empty class", c.Class)
	}
	if !hasReason(c.Reasons, ReasonClassUndeclared) {
		t.Fatalf("reasons = %v, want %q", c.Reasons, ReasonClassUndeclared)
	}
	// And the floor that empty class produces is the strictest one.
	pol := PolicyFor(c.Class)
	if pol.RequiredTier != TierT0 {
		t.Fatalf("undeclared class floor = %v, want T0", pol.RequiredTier)
	}
	// End to end: it reaches the vendor even though cheaper measured rungs exist.
	p, _, err := threeZoneRoster().PlaceSubject(Subject{Aspect: AspectToolCall, Tool: "grep"}, measured())
	if err != nil {
		t.Fatal(err)
	}
	if p.Zone != ZoneVendor {
		t.Fatalf("undeclared tool call placed in zone %q, want %q", p.Zone, ZoneVendor)
	}
}

// TestDeclaringTheClassIsWhatMovesWorkDown is the other half: the ONLY thing that buys
// a cheap rung is an operator saying what the work is. Same subject, same roster, one
// label — and it lands on the engineer's own machine.
func TestDeclaringTheClassIsWhatMovesWorkDown(t *testing.T) {
	s := Subject{Aspect: AspectToolCall, Tool: "grep", Labels: map[string]string{ClassLabel: "routine"}}
	p, c, err := threeZoneRoster().PlaceSubject(s, measured())
	if err != nil {
		t.Fatal(err)
	}
	if !c.Declared || c.Class != ClassRoutine {
		t.Fatalf("classification = %+v, want a declared routine class", c)
	}
	if p.Zone != ZoneDevice || !p.SelfHosted() {
		t.Fatalf("declared routine work placed in zone %q, want %q", p.Zone, ZoneDevice)
	}
}

// TestATypoIsNotADeclaration pins that a misspelled class reads as "nobody said", not
// as permission. This is the failure mode a config-driven placer is most likely to hit
// in production, and the expensive-but-correct answer is the safe one.
func TestATypoIsNotADeclaration(t *testing.T) {
	for _, bad := range []string{"routin", "ROUTINE-ish", "cheap", "", "t2"} {
		c := ClassOf(Subject{Labels: map[string]string{ClassLabel: bad}})
		if c.Declared {
			t.Fatalf("label %q was treated as a declaration: %+v", bad, c)
		}
		if !hasReason(c.Reasons, ReasonClassLabelUnrecognized) {
			t.Fatalf("label %q reasons = %v, want %q — the operator must see their config did nothing",
				bad, c.Reasons, ReasonClassLabelUnrecognized)
		}
	}
	// Case and surrounding whitespace are tolerated, because those are not typos.
	for _, ok := range []string{"ROUTINE", " routine ", "Routine"} {
		c := ClassOf(Subject{Labels: map[string]string{ClassLabel: ok}})
		if !c.Declared || c.Class != ClassRoutine {
			t.Fatalf("label %q = %+v, want a declared routine class", ok, c)
		}
	}
}

// TestEveryDeclaredClassRoundTrips keeps the parser total over the closed vocabulary, so
// adding a WorkClass without teaching this file about it is a test failure rather than a
// silent demotion to "undeclared".
func TestEveryDeclaredClassRoundTrips(t *testing.T) {
	for _, cls := range []WorkClass{ClassUltraHard, ClassNormalImpl, ClassRoutine, ClassSecurityRelease} {
		c := ClassOf(Subject{Labels: map[string]string{ClassLabel: string(cls)}})
		if !c.Declared || c.Class != cls {
			t.Fatalf("class %q did not round-trip: %+v", cls, c)
		}
	}
}

// TestScoutAspectIsItsOwnDeclaration pins the single inference this file allows, and
// pins WHY it is allowed: AspectScout is defined as the cheap classify-first probe, so
// the aspect names the work's nature. No other aspect gets this treatment.
func TestScoutAspectIsItsOwnDeclaration(t *testing.T) {
	c := ClassOf(Subject{Aspect: AspectScout})
	if !c.Declared || c.Class != ClassRoutine {
		t.Fatalf("scout aspect = %+v, want a routine class", c)
	}
	if !hasReason(c.Reasons, ReasonClassFromAspect) {
		t.Fatalf("reasons = %v, want %q", c.Reasons, ReasonClassFromAspect)
	}
	for _, a := range []Aspect{AspectRequest, AspectToolCall, AspectQuery, AspectState, AspectStep} {
		if got := ClassOf(Subject{Aspect: a}); got.Declared {
			t.Fatalf("aspect %q was treated as a declaration: %+v — only scout names its own nature", a, got)
		}
	}
}

// TestComplexityRatchetsOnlyUpward pins the one-way ratchet. A high-complexity subject
// raises a routine floor, and no complexity value can ever lower one — so a subject that
// understates its complexity gains nothing, which is the property that makes accepting
// complexity as an input safe at all.
func TestComplexityRatchetsOnlyUpward(t *testing.T) {
	raised := ClassOf(Subject{
		Complexity: ComplexityHigh,
		Labels:     map[string]string{ClassLabel: "routine"},
	})
	if raised.Class != ClassNormalImpl {
		t.Fatalf("high-complexity routine work classified %q, want %q", raised.Class, ClassNormalImpl)
	}
	if !hasReason(raised.Reasons, ReasonClassRaisedByComplexity) {
		t.Fatalf("reasons = %v, want %q", raised.Reasons, ReasonClassRaisedByComplexity)
	}
	// No complexity value may make ANY declared class less demanding.
	for _, cls := range []WorkClass{ClassUltraHard, ClassNormalImpl, ClassRoutine, ClassSecurityRelease} {
		base := PolicyFor(cls).RequiredTier
		for _, cx := range []Complexity{ComplexityAny, ComplexityLow, ComplexityMedium, ComplexityHigh} {
			got := ClassOf(Subject{Complexity: cx, Labels: map[string]string{ClassLabel: string(cls)}})
			floor := PolicyFor(got.Class).RequiredTier
			if !floor.MeetsRequirement(base) {
				t.Fatalf("class %q at complexity %q relaxed the floor from %v to %v",
					cls, cx, base, floor)
			}
		}
	}
}

// TestSecurityWorkStaysOffTheDeviceThroughTheWholeComposition re-asserts the safety
// property across the composed call, not just Place() in isolation — a joint is exactly
// where a floor gets dropped by accident.
func TestSecurityWorkStaysOffTheDeviceThroughTheWholeComposition(t *testing.T) {
	s := Subject{
		Aspect: AspectToolCall,
		Tool:   "delete_file",
		Labels: map[string]string{ClassLabel: string(ClassSecurityRelease)},
	}
	p, c, err := threeZoneRoster().PlaceSubject(s, measured())
	if err != nil {
		t.Fatal(err)
	}
	if c.Class != ClassSecurityRelease {
		t.Fatalf("classification = %+v, want the security class", c)
	}
	if p.Zone == ZoneDevice {
		t.Fatalf("security work placed on the device rung (%q)", p.Model)
	}
	if !p.Escalated {
		t.Fatal("the device rung existed and lost; that must be recorded")
	}
}

// TestPlaceSubjectPropagatesTheFailure pins that composing does not swallow the
// fail-loud behaviour: an unresolvable candidate is still an error at the joint.
func TestPlaceSubjectPropagatesTheFailure(t *testing.T) {
	r := threeZoneRoster()
	r.Default = ""
	_, _, err := r.PlaceSubject(
		Subject{Labels: map[string]string{ClassLabel: "routine"}},
		[]Candidate{{Model: "nope", Capability: TierT2, Measured: true}},
	)
	if err == nil {
		t.Fatal("PlaceSubject must surface an unresolvable candidate, not swallow it")
	}
}
