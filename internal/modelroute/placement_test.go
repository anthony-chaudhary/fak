package modelroute

import (
	"strings"
	"testing"
)

// threeZoneRoster is the deployment epic #5416 is about, in miniature: an
// engineer's own machine, two servers the company operates, and a vendor.
func threeZoneRoster() Roster {
	return Roster{
		Accounts: []Account{
			{ID: "laptop", Kind: KindLocal, BaseURL: "http://127.0.0.1:11434/v1"},
			{ID: "corp-glm", Kind: KindFleet, BaseURL: "http://glm.infer.corp.internal:8000/v1"},
			{ID: "corp-kimi", Kind: KindFleet, BaseURL: "http://kimi.infer.corp.internal:8000/v1"},
			{ID: "frontier", Kind: KindAnthropic, CredEnv: "ANTHROPIC_API_KEY"},
		},
		Bindings: []Binding{
			{Model: "tiny", Account: "laptop", UpstreamModel: "qwen3.6-4b"},
			{Model: "corp-mid", Account: "corp-glm", UpstreamModel: "glm-5.2"},
			{Model: "corp-agentic", Account: "corp-kimi", UpstreamModel: "kimi-k3"},
			{Model: "frontier", Account: "frontier", UpstreamModel: "claude-opus-5"},
		},
		Default: "laptop",
	}
}

// measured is the ladder as an operator who HAS capability evidence would declare
// it: a 4B local model for routine work, a company GLM/Kimi for ordinary
// implementation, a frontier vendor model for ultra-hard work.
func measured() []Candidate {
	return []Candidate{
		{Model: "tiny", Capability: TierT2, Measured: true},
		{Model: "corp-mid", Capability: TierT1, Measured: true},
		{Model: "corp-agentic", Capability: TierT1, Measured: true},
		{Model: "frontier", Capability: TierT0, Measured: true},
	}
}

// TestPlaceTakesTheCheapestRungThatCanServe is the thesis of epic #5416 as an
// executable claim: routine work stays on the engineer's own machine, ordinary
// implementation lands on company hardware, and only genuinely ultra-hard work
// reaches a third-party lab. If this inverts, the whole design is inverted.
func TestPlaceTakesTheCheapestRungThatCanServe(t *testing.T) {
	r := threeZoneRoster()
	cases := []struct {
		class      WorkClass
		wantZone   PlacementZone
		wantModel  string
		selfHosted bool
	}{
		{ClassRoutine, ZoneDevice, "tiny", true},
		{ClassNormalImpl, ZoneFleet, "corp-mid", true},
		{ClassUltraHard, ZoneVendor, "frontier", false},
	}
	for _, c := range cases {
		p, err := r.Place(c.class, measured())
		if err != nil {
			t.Fatalf("class %q: %v", c.class, err)
		}
		if p.Zone != c.wantZone {
			t.Fatalf("class %q placed in zone %q, want %q (ladder: %s)",
				c.class, p.Zone, c.wantZone, describeLadder(p.Ladder))
		}
		if p.Model != c.wantModel {
			t.Fatalf("class %q placed on model %q, want %q", c.class, p.Model, c.wantModel)
		}
		if p.SelfHosted() != c.selfHosted {
			t.Fatalf("class %q SelfHosted = %v, want %v", c.class, p.SelfHosted(), c.selfHosted)
		}
		if !p.Choice.Admitted {
			t.Fatalf("class %q produced a placement that was not admitted: %+v", c.class, p.Choice)
		}
		// The route the kernel will see must carry the zone back out.
		if got := ZoneOfRoute(p.Target.EngineRoute()); got != c.wantZone {
			t.Fatalf("class %q route %q parses to zone %q, want %q",
				c.class, p.Target.EngineRoute(), got, c.wantZone)
		}
	}
}

// TestSecurityWorkNeverFallsToTheCheapRung is the safety claim. The
// security-release-destructive class carries a floor that never drops to routine,
// so a cheap local model must not take it EVEN THOUGH it is the cheapest rung, is
// bound, and is measured. This is the case where "prefer the cheapest rung" and
// "do not do something dangerous badly" pull against each other, and the floor wins.
func TestSecurityWorkNeverFallsToTheCheapRung(t *testing.T) {
	r := threeZoneRoster()
	p, err := r.Place(ClassSecurityRelease, measured())
	if err != nil {
		t.Fatalf("security work must still be placeable: %v", err)
	}
	if p.Zone == ZoneDevice {
		t.Fatalf("security/release/destructive work landed on the device rung (%q) — the class floor was ignored", p.Model)
	}
	if p.Model != "corp-mid" {
		t.Fatalf("security work placed on %q, want the fleet rung that meets the T1 floor", p.Model)
	}
	if !p.Escalated {
		t.Fatal("placement must record that a cheaper rung existed and lost")
	}
	// And the device rung must say WHY it lost, in closed vocabulary.
	var deviceReasons []string
	for _, v := range p.Ladder {
		if v.Zone == ZoneDevice {
			deviceReasons = v.Reasons
		}
	}
	if !hasReason(deviceReasons, ReasonZoneUnderTier) {
		t.Fatalf("device rung reasons = %v, want %q", deviceReasons, ReasonZoneUnderTier)
	}
}

// TestInvertedTierComparisonWouldBeCaught guards the specific trap this file's
// doc comment calls out. WorkTier numbers are INVERTED — T0 < T1 < T2 numerically,
// but T0 is the MOST demanding. A placer that used a raw `<` instead of
// MeetsRequirement would admit the least capable model for the hardest work.
//
// Here every self-hosted rung is genuinely too weak for ultra-hard work, so a
// correct placer escalates to the vendor. A placer with the comparison backwards
// would happily place T2 "tiny" on T0 work.
func TestInvertedTierComparisonWouldBeCaught(t *testing.T) {
	r := threeZoneRoster()
	p, err := r.Place(ClassUltraHard, measured())
	if err != nil {
		t.Fatalf("ultra-hard work must be placeable on the vendor rung: %v", err)
	}
	if p.Model != "frontier" || p.Zone != ZoneVendor {
		t.Fatalf("ultra-hard work placed on %q in zone %q — a T2/T1 model cannot serve a T0 floor",
			p.Model, p.Zone)
	}
	// Re-assert the admission arithmetic through the canonical predicate rather
	// than a raw comparison, so this test cannot itself encode the inversion it is
	// guarding against.
	if !p.Choice.Capability.MeetsRequirement(p.Choice.RequiredTier) {
		t.Fatalf("placed capability %v does not meet required %v", p.Choice.Capability, p.Choice.RequiredTier)
	}
	// The trap, made explicit: the natural-looking raw compare
	// `capability >= required` reads TRUE for the weakest model against the
	// strictest floor (T2=2 >= T0=0), while the canonical predicate reads FALSE.
	// The two genuinely disagree here, so this corpus really would catch a placer
	// that swapped one for the other.
	weakest, strictestFloor := TierT2, TierT0
	if weakest.MeetsRequirement(strictestFloor) {
		t.Fatal("MeetsRequirement is inverted: T2 must not satisfy a T0 floor")
	}
	if !(weakest >= strictestFloor) {
		t.Fatal("tier numbering is no longer inverted; this guard needs re-deriving")
	}
}

// TestUnmeasuredCapabilityCannotDescendTheLadder pins the rule that keeps the
// placer honest while capability measurement is still a stub. An operator who has
// bound a local and a fleet model but measured NEITHER gets the vendor rung — the
// status quo — rather than a free "saving" justified by a number nobody computed.
func TestUnmeasuredCapabilityCannotDescendTheLadder(t *testing.T) {
	r := threeZoneRoster()
	asserted := []Candidate{
		{Model: "tiny", Capability: TierT2},     // Measured: false
		{Model: "corp-mid", Capability: TierT1}, // Measured: false
		{Model: "frontier", Capability: TierT0}, // Measured: false
	}
	p, err := r.Place(ClassRoutine, asserted)
	if err != nil {
		t.Fatalf("an all-unmeasured ladder must still place (on the top rung): %v", err)
	}
	if p.Zone != ZoneVendor {
		t.Fatalf("unmeasured capability placed on zone %q — an asserted number must not win a cheap rung", p.Zone)
	}
	if p.SelfHosted() {
		t.Fatal("an unmeasured placement must not be counted as a self-hosted saving")
	}
	for _, v := range p.Ladder {
		if v.Zone == ZoneDevice && !hasReason(v.Reasons, ReasonZoneUnmeasured) {
			t.Fatalf("device rung reasons = %v, want %q", v.Reasons, ReasonZoneUnmeasured)
		}
	}
	// The moment the operator measures the cheap rung, the SAME work moves down.
	withEvidence := []Candidate{
		{Model: "tiny", Capability: TierT2, Measured: true},
		{Model: "frontier", Capability: TierT0},
	}
	moved, err := r.Place(ClassRoutine, withEvidence)
	if err != nil {
		t.Fatalf("measured ladder: %v", err)
	}
	if moved.Zone != ZoneDevice {
		t.Fatalf("measured routine work placed in zone %q, want %q", moved.Zone, ZoneDevice)
	}
	if !moved.SelfHosted() {
		t.Fatal("a measured device placement is a self-hosted saving")
	}
}

// TestUnknownWorkClassStaysConservative pins that an unrecognized class does not
// get inferred into a cheap tier. An unknown class carries the highest floor, so it
// must reach the frontier rung even though cheaper measured rungs are available.
func TestUnknownWorkClassStaysConservative(t *testing.T) {
	r := threeZoneRoster()
	p, err := r.Place(WorkClass("something-nobody-declared"), measured())
	if err != nil {
		t.Fatalf("unknown class must still place: %v", err)
	}
	if p.Zone != ZoneVendor {
		t.Fatalf("unknown work class placed in zone %q — an undeclared class must stay conservative", p.Zone)
	}
	if !p.Escalated {
		t.Fatal("unknown class must record that cheaper rungs lost")
	}
}

// TestPlacementIsDeterministic pins that the same inputs give the same answer, and
// that the full ladder is always reported in rank order. A placement surface that
// reordered between runs would make an operator's "why did this change?" question
// unanswerable.
func TestPlacementIsDeterministic(t *testing.T) {
	r := threeZoneRoster()
	first, err := r.Place(ClassNormalImpl, measured())
	if err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 8; i++ {
		again, err := r.Place(ClassNormalImpl, measured())
		if err != nil {
			t.Fatal(err)
		}
		if again.Model != first.Model || again.Zone != first.Zone || again.Escalated != first.Escalated {
			t.Fatalf("placement is not deterministic: %+v then %+v", first, again)
		}
		if describeLadder(again.Ladder) != describeLadder(first.Ladder) {
			t.Fatalf("ladder differs between runs:\n %s\n %s", describeLadder(first.Ladder), describeLadder(again.Ladder))
		}
	}
	// Every zone appears exactly once, in ladder order.
	if len(first.Ladder) != len(Zones()) {
		t.Fatalf("ladder has %d rungs, want %d: %s", len(first.Ladder), len(Zones()), describeLadder(first.Ladder))
	}
	for i := 1; i < len(first.Ladder); i++ {
		if first.Ladder[i-1].Zone.Rank() >= first.Ladder[i].Zone.Rank() {
			t.Fatalf("ladder is not in rank order: %s", describeLadder(first.Ladder))
		}
	}
}

// TestPlaceFailsLoudOnMisconfiguration pins that placement never degrades into a
// silent fallback. A candidate the roster cannot resolve is a config bug the
// operator must see — quietly continuing to bill a vendor is the worst outcome.
func TestPlaceFailsLoudOnMisconfiguration(t *testing.T) {
	r := threeZoneRoster()
	r.Default = "" // no silent catch-all

	if _, err := r.Place(ClassRoutine, nil); err == nil {
		t.Fatal("placing with no candidates must fail loud")
	}
	_, err := r.Place(ClassRoutine, []Candidate{{Model: "does-not-exist", Capability: TierT2, Measured: true}})
	if err == nil {
		t.Fatal("an unresolvable candidate must fail loud, not be skipped")
	}
	if !strings.Contains(err.Error(), "does-not-exist") {
		t.Fatalf("error must name the bad candidate, got %v", err)
	}
	if _, err := r.Place(ClassRoutine, []Candidate{{Capability: TierT2, Measured: true}}); err == nil {
		t.Fatal("an empty model id must fail loud")
	}
}

// TestNoZoneCanServeIsAnError pins the ladder-exhausted case: if nothing on any
// rung meets the floor, placement fails with the closed reason rather than
// inventing a placement.
func TestNoZoneCanServeIsAnError(t *testing.T) {
	r := threeZoneRoster()
	tooWeak := []Candidate{
		{Model: "tiny", Capability: TierT2, Measured: true},
		{Model: "corp-mid", Capability: TierT2, Measured: true},
	}
	_, err := r.Place(ClassUltraHard, tooWeak)
	if err == nil {
		t.Fatal("ultra-hard work with only routine-capable models must fail, not place")
	}
	if !strings.Contains(err.Error(), ReasonNoZoneCanServe) {
		t.Fatalf("error must carry %q, got %v", ReasonNoZoneCanServe, err)
	}
	if !strings.Contains(err.Error(), ReasonZoneUnderTier) {
		t.Fatalf("error must explain each rung with a closed reason, got %v", err)
	}
}

// TestEscalationIsNotClaimedWhenTheCheapestRungWins guards the bookkeeping bug this
// walk is prone to: a rung that rejects its FIRST candidate and admits its SECOND
// has not escalated past anything, and must not be reported as though it had.
func TestEscalationIsNotClaimedWhenTheCheapestRungWins(t *testing.T) {
	r := threeZoneRoster()
	// Two fleet candidates; the first is too weak for T1, the second serves.
	cands := []Candidate{
		{Model: "corp-mid", Capability: TierT2, Measured: true},     // under-tier for normal-impl
		{Model: "corp-agentic", Capability: TierT1, Measured: true}, // serves
	}
	p, err := r.Place(ClassNormalImpl, cands)
	if err != nil {
		t.Fatal(err)
	}
	if p.Zone != ZoneFleet || p.Model != "corp-agentic" {
		t.Fatalf("placed %q in %q, want corp-agentic on the fleet rung", p.Model, p.Zone)
	}
	if p.Escalated {
		t.Fatal("placement on the cheapest rung that had ANY candidate must not report an escalation")
	}
}

func hasReason(reasons []string, want string) bool {
	for _, r := range reasons {
		if r == want {
			return true
		}
	}
	return false
}
