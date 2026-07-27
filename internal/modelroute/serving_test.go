package modelroute

import (
	"reflect"
	"strings"
	"testing"
)

// servingReport builds a valid snapshot over the threeZoneRoster fixture.
func servingReport(covers []PlacementZone, models map[string]ServingObservation) ServingReport {
	return ServingReport{Schema: ServingReportSchema, Covers: covers, Models: models}
}

// up/down/degraded/unknown are the four states with no timestamp, for the tests
// that declare no freshness bound.
func up() ServingObservation       { return ServingObservation{State: ServingUp} }
func down() ServingObservation     { return ServingObservation{State: ServingDown} }
func degraded() ServingObservation { return ServingObservation{State: ServingDegraded} }
func unsure() ServingObservation   { return ServingObservation{State: ServingUnknown} }

// placeOrFail runs the ladder with a snapshot and fails the test on a refusal.
func placeOrFail(t *testing.T, r Roster, class WorkClass, cands []Candidate, s ServingReport) Placement {
	t.Helper()
	p, err := r.PlaceWithServing(class, cands, s)
	if err != nil {
		t.Fatalf("class %q: %v", class, err)
	}
	return p
}

// rungReasons returns the reason tokens recorded for one rung of a walk.
func rungReasons(p Placement, z PlacementZone) []string {
	for _, v := range p.Ladder {
		if v.Zone == z {
			return v.Reasons
		}
	}
	return nil
}

// TestPlaceWithoutAServingReportIsTheOldPlacementExactly is the property that
// makes this addition safe to land on a fleet that has no prober: with no
// snapshot, every placement is byte-identical to what Place decided before
// liveness existed. Place delegates with a zero report, so this is checking the
// delegation is honest — that an empty snapshot really does say nothing, rather
// than saying "unknown" and quietly gating every rung.
func TestPlaceWithoutAServingReportIsTheOldPlacementExactly(t *testing.T) {
	r := threeZoneRoster()
	asserted := []Candidate{
		{Model: "tiny", Capability: TierT2},
		{Model: "corp-mid", Capability: TierT1},
		{Model: "frontier", Capability: TierT0},
	}
	classes := []WorkClass{ClassRoutine, ClassNormalImpl, ClassUltraHard, ClassSecurityRelease, WorkClass("nobody-declared-this")}
	pools := map[string][]Candidate{"measured": measured(), "asserted": asserted, "empty": nil}
	for _, class := range classes {
		for name, pool := range pools {
			want, wantErr := r.Place(class, pool)
			got, gotErr := r.PlaceWithServing(class, pool, ServingReport{})
			if (wantErr == nil) != (gotErr == nil) {
				t.Fatalf("class %q pool %s: Place err=%v but PlaceWithServing err=%v", class, name, wantErr, gotErr)
			}
			if wantErr != nil && wantErr.Error() != gotErr.Error() {
				t.Fatalf("class %q pool %s: errors differ:\n  %v\n  %v", class, name, wantErr, gotErr)
			}
			if !reflect.DeepEqual(want, got) {
				t.Fatalf("class %q pool %s: an empty serving report changed the placement:\n  want %+v\n  got  %+v", class, name, want, got)
			}
		}
	}
}

// TestServingFailsOverWithinTheRungBeforeLeavingIt is the headline of track H.
// One of two company GPU hosts is down. The right answer is the OTHER company
// host — not a vendor. A zone-keyed liveness signal would have taken the whole
// fleet rung out and sent this to a third-party lab to route around a machine
// whose neighbor was idle; keying per bound model is what keeps the money on the
// company's own hardware.
func TestServingFailsOverWithinTheRungBeforeLeavingIt(t *testing.T) {
	r := threeZoneRoster()
	s := servingReport([]PlacementZone{ZoneFleet}, map[string]ServingObservation{
		"corp-mid":     down(),
		"corp-agentic": up(),
	})
	p := placeOrFail(t, r, ClassNormalImpl, measured(), s)
	if p.Zone != ZoneFleet {
		t.Fatalf("one dead fleet host moved the work to zone %q, want %q (ladder: %s)", p.Zone, ZoneFleet, describeLadder(p.Ladder))
	}
	if p.Model != "corp-agentic" {
		t.Fatalf("placed on %q, want the surviving company host %q", p.Model, "corp-agentic")
	}
	if !p.SelfHosted() {
		t.Fatalf("a single dead host cost the org its self-hosted placement")
	}
	if !p.FailedOver {
		t.Fatalf("FailedOver = false, but a candidate was passed over on a liveness observation")
	}
	// A failover WITHIN a rung must not move Escalated. That bit answers "did a
	// cheaper rung lose this work?", and the answer here is whatever it was before
	// the host died — the device rung still lost on tier, exactly as it always did.
	// Pinning it against the no-outage baseline rather than a literal keeps the two
	// bits independent instead of quietly re-deriving one from the other.
	baseline, err := r.Place(ClassNormalImpl, measured())
	if err != nil {
		t.Fatalf("baseline placement: %v", err)
	}
	if p.Escalated != baseline.Escalated {
		t.Fatalf("Escalated moved from %v to %v because one host in the rung died; that bit is about capability, not uptime",
			baseline.Escalated, p.Escalated)
	}
	if baseline.FailedOver {
		t.Fatalf("the no-outage baseline already reports FailedOver; the bit is not measuring the outage")
	}
	if !hasReason(rungReasons(p, ZoneFleet), ReasonZoneServingDown) {
		t.Fatalf("fleet reasons %v do not record %q", rungReasons(p, ZoneFleet), ReasonZoneServingDown)
	}
}

// TestServingFailsOverUpTheLadderWhenTheWholeRungIsDown: when there is no
// surviving company host, the work does reach a vendor — and the artifact says
// it went there because the rung was DOWN, not because the rung was incapable.
func TestServingFailsOverUpTheLadderWhenTheWholeRungIsDown(t *testing.T) {
	r := threeZoneRoster()
	s := servingReport([]PlacementZone{ZoneFleet}, map[string]ServingObservation{
		"corp-mid":     down(),
		"corp-agentic": down(),
	})
	p := placeOrFail(t, r, ClassNormalImpl, measured(), s)
	if p.Zone != ZoneVendor {
		t.Fatalf("whole fleet rung down placed in zone %q, want %q", p.Zone, ZoneVendor)
	}
	if !p.FailedOver || !p.Escalated {
		t.Fatalf("FailedOver=%v Escalated=%v, want both true", p.FailedOver, p.Escalated)
	}
	fleet := rungReasons(p, ZoneFleet)
	downs := 0
	for _, rsn := range fleet {
		if rsn == ReasonZoneServingDown {
			downs++
		}
	}
	if downs != 2 {
		t.Fatalf("fleet rung recorded %d %q tokens for 2 dead hosts: %v", downs, ReasonZoneServingDown, fleet)
	}
}

// TestServingFailedOverIsFalseWhenTheRungWasMerelyIncapable keeps the two bits
// from collapsing into one. Ultra-hard work reaches a vendor because nothing
// cheaper clears the floor; nothing was down, and claiming a failover would send
// an operator hunting for a dead host that does not exist.
func TestServingFailedOverIsFalseWhenTheRungWasMerelyIncapable(t *testing.T) {
	r := threeZoneRoster()
	s := servingReport([]PlacementZone{ZoneDevice, ZoneFleet, ZoneVendor}, map[string]ServingObservation{
		"tiny": up(), "corp-mid": up(), "corp-agentic": up(), "frontier": up(),
	})
	p := placeOrFail(t, r, ClassUltraHard, measured(), s)
	if p.Zone != ZoneVendor {
		t.Fatalf("ultra-hard work placed in zone %q, want %q", p.Zone, ZoneVendor)
	}
	if !p.Escalated {
		t.Fatalf("Escalated = false, but cheaper rungs were bound and lost")
	}
	if p.FailedOver {
		t.Fatalf("FailedOver = true with every host reported up — a capability escalation was mislabeled as an outage")
	}
}

// TestServingSilenceOnlyCountsWhereTheSnapshotClaimsCoverage is rule 1. A report
// that says it watches the fleet and then names no fleet model has told the
// placer something: its prober is not reporting, and the rung is not eligible. A
// report that never claimed the rung has told it nothing at all.
func TestServingSilenceOnlyCountsWhereTheSnapshotClaimsCoverage(t *testing.T) {
	r := threeZoneRoster()

	claimed := servingReport([]PlacementZone{ZoneFleet}, map[string]ServingObservation{"tiny": up()})
	p := placeOrFail(t, r, ClassNormalImpl, measured(), claimed)
	if p.Zone != ZoneVendor {
		t.Fatalf("a covered-but-unreported fleet rung still took the work (zone %q); a crashed prober must not read as healthy", p.Zone)
	}
	if !hasReason(rungReasons(p, ZoneFleet), ReasonZoneServingUnknown) {
		t.Fatalf("fleet reasons %v do not record %q", rungReasons(p, ZoneFleet), ReasonZoneServingUnknown)
	}
	if !p.FailedOver {
		t.Fatalf("FailedOver = false, but the fleet rung was passed over on a liveness verdict")
	}

	unclaimed := servingReport(nil, map[string]ServingObservation{"tiny": up()})
	q := placeOrFail(t, r, ClassNormalImpl, measured(), unclaimed)
	if q.Zone != ZoneFleet || q.Model != "corp-mid" {
		t.Fatalf("an unclaimed rung was gated anyway: placed %q in %q, want corp-mid in fleet", q.Model, q.Zone)
	}
	if q.FailedOver {
		t.Fatalf("FailedOver = true with no rung claimed and nothing reported down")
	}
}

// TestServingHonorsAnObservationOutsideDeclaredCoverage is rule 2, and it is the
// one that closes a fail-open. An operator whose prober reports corp-mid DOWN but
// who forgot to list the fleet rung in Covers must still not have work sent to
// corp-mid: coverage governs what SILENCE means, and nothing else.
func TestServingHonorsAnObservationOutsideDeclaredCoverage(t *testing.T) {
	r := threeZoneRoster()
	s := servingReport([]PlacementZone{ZoneDevice}, map[string]ServingObservation{
		"tiny":     up(),
		"corp-mid": down(), // reported, but the fleet rung was never declared covered
	})
	p := placeOrFail(t, r, ClassNormalImpl, measured(), s)
	if p.Model == "corp-mid" {
		t.Fatalf("work was placed on a host reported DOWN because its rung was missing from Covers — a config typo disabled the gate")
	}
	if p.Model != "corp-agentic" || p.Zone != ZoneFleet {
		t.Fatalf("placed %q in %q, want corp-agentic in fleet (the unreported sibling is not gated)", p.Model, p.Zone)
	}
	if !hasReason(rungReasons(p, ZoneFleet), ReasonZoneServingDown) {
		t.Fatalf("fleet reasons %v do not record %q", rungReasons(p, ZoneFleet), ReasonZoneServingDown)
	}
}

// TestServingDegradedTakesTheWorkAndSaysSo. A loaded host is still a host. If
// strain shed work to a vendor, the fleet would be abandoned at exactly the hour
// it carries the most tokens — the inverse of what this epic is for. The state is
// recorded on the rung that took the work, and it is NOT a failover.
func TestServingDegradedTakesTheWorkAndSaysSo(t *testing.T) {
	r := threeZoneRoster()
	s := servingReport([]PlacementZone{ZoneFleet}, map[string]ServingObservation{
		"corp-mid": degraded(), "corp-agentic": up(),
	})
	p := placeOrFail(t, r, ClassNormalImpl, measured(), s)
	if p.Model != "corp-mid" || p.Zone != ZoneFleet {
		t.Fatalf("a degraded host shed its work: placed %q in %q, want corp-mid in fleet", p.Model, p.Zone)
	}
	if p.FailedOver {
		t.Fatalf("FailedOver = true, but nothing was passed over — degraded is recorded, not acted on")
	}
	if !hasReason(rungReasons(p, ZoneFleet), ReasonZoneServingDegraded) {
		t.Fatalf("fleet reasons %v do not record %q — strain that leaves no trace cannot be acted on later",
			rungReasons(p, ZoneFleet), ReasonZoneServingDegraded)
	}
}

// TestServingUnknownStateIsPassedOverEvenWhenObserved: a probe that ran and could
// not tell is an admission of ignorance, not a pass. It is honored outside
// declared coverage exactly like any other observation.
func TestServingUnknownStateIsPassedOverEvenWhenObserved(t *testing.T) {
	r := threeZoneRoster()
	s := servingReport(nil, map[string]ServingObservation{"corp-mid": unsure(), "corp-agentic": up()})
	p := placeOrFail(t, r, ClassNormalImpl, measured(), s)
	if p.Model != "corp-agentic" {
		t.Fatalf("placed %q, want corp-agentic — an explicit UNKNOWN was treated as serving", p.Model)
	}
	if !hasReason(rungReasons(p, ZoneFleet), ReasonZoneServingUnknown) {
		t.Fatalf("fleet reasons %v do not record %q", rungReasons(p, ZoneFleet), ReasonZoneServingUnknown)
	}
}

// TestServingFreshnessIsFailClosedInEveryDirection is rule 3 as a grid. Under a
// declared bound, the ONLY thing that keeps a candidate eligible is an
// observation that can be shown fresh. The negative-age row is the one that
// matters most: a producer whose clock runs ahead would otherwise pin a rung open
// forever, and there is no bug report for "the gate silently stopped working".
func TestServingFreshnessIsFailClosedInEveryDirection(t *testing.T) {
	cases := []struct {
		name      string
		maxAge    int64
		asOf      int64
		observed  int64
		wantFresh bool
	}{
		{"no bound declared honors any age", 0, 1_000_000, 1, true},
		{"no bound declared honors a missing stamp", 0, 1_000_000, 0, true},
		{"well inside the bound", 60, 1_000_000, 999_990, true},
		{"exactly at the bound", 60, 1_000_000, 999_940, true},
		{"one second past the bound", 60, 1_000_000, 999_939, false},
		{"observation carries no stamp", 60, 1_000_000, 0, false},
		{"snapshot carries no stamp to measure against", 60, 0, 999_990, false},
		{"neither carries a stamp", 60, 0, 0, false},
		{"stamped after the snapshot containing it", 60, 1_000_000, 1_000_030, false},
		{"stamped far in the future", 60, 1_000_000, 9_000_000, false},
	}
	for _, c := range cases {
		s := ServingReport{MaxAgeSeconds: c.maxAge, AsOfUnix: c.asOf}
		if got := s.fresh(ServingObservation{State: ServingUp, ObservedUnix: c.observed}); got != c.wantFresh {
			t.Fatalf("%s: fresh = %v, want %v (max_age=%d as_of=%d observed=%d)",
				c.name, got, c.wantFresh, c.maxAge, c.asOf, c.observed)
		}
	}
}

// TestServingStaleUpIsNotAPassOnTheLadder binds the freshness rule to a real
// placement: an "up" from before the bound does not keep the rung, and the ladder
// says stale rather than down, so the operator debugs the prober and not the host.
func TestServingStaleUpIsNotAPassOnTheLadder(t *testing.T) {
	r := threeZoneRoster()
	s := servingReport([]PlacementZone{ZoneFleet}, map[string]ServingObservation{
		"corp-mid":     {State: ServingUp, ObservedUnix: 1_000},   // hours old
		"corp-agentic": {State: ServingUp, ObservedUnix: 999_990}, // recent
	})
	s.AsOfUnix, s.MaxAgeSeconds = 1_000_000, 60
	p := placeOrFail(t, r, ClassNormalImpl, measured(), s)
	if p.Model != "corp-agentic" {
		t.Fatalf("placed %q, want corp-agentic — a stale UP kept a rung eligible", p.Model)
	}
	reasons := rungReasons(p, ZoneFleet)
	if !hasReason(reasons, ReasonZoneServingStale) {
		t.Fatalf("fleet reasons %v do not record %q", reasons, ReasonZoneServingStale)
	}
	if hasReason(reasons, ReasonZoneServingDown) {
		t.Fatalf("a stale observation was reported as %q; that sends the operator to reboot a healthy host", ReasonZoneServingDown)
	}
	if !p.FailedOver {
		t.Fatalf("FailedOver = false, but a candidate was passed over on an unshowable freshness claim")
	}
}

// TestServingRunsAfterTheTierGateSoTheStableReasonWins. A candidate that is both
// below the work's floor AND on a dead box reports UNDER-TIER. The tier fact
// holds tomorrow; the liveness fact may not. If the transient reason won, the
// same misconfiguration would diagnose differently depending on when the operator
// looked, and rebooting the host would not change the placement one bit.
func TestServingRunsAfterTheTierGateSoTheStableReasonWins(t *testing.T) {
	r := threeZoneRoster()
	s := servingReport([]PlacementZone{ZoneDevice, ZoneFleet}, map[string]ServingObservation{
		"tiny": down(), "corp-mid": down(), "corp-agentic": down(),
	})
	// ClassUltraHard's floor refuses tiny on capability alone.
	p := placeOrFail(t, r, ClassUltraHard, measured(), s)
	device := rungReasons(p, ZoneDevice)
	if !hasReason(device, ReasonZoneUnderTier) {
		t.Fatalf("device reasons %v do not record %q", device, ReasonZoneUnderTier)
	}
	if hasReason(device, ReasonZoneServingDown) {
		t.Fatalf("device reasons %v recorded a transient outage over the permanent tier refusal", device)
	}
}

// TestServingCanOnlyRemoveCandidatesNeverAdmitThem. Reporting a model UP is not
// evidence about what it can do. If a liveness snapshot could rescue a candidate
// the tier floor refuses, a prober would have become a capability grader.
func TestServingCanOnlyRemoveCandidatesNeverAdmitThem(t *testing.T) {
	r := threeZoneRoster()
	everythingUp := servingReport([]PlacementZone{ZoneDevice, ZoneFleet, ZoneVendor}, map[string]ServingObservation{
		"tiny": up(), "corp-mid": up(), "corp-agentic": up(), "frontier": up(),
	})
	for _, class := range []WorkClass{ClassRoutine, ClassNormalImpl, ClassUltraHard, ClassSecurityRelease} {
		bare, bareErr := r.Place(class, measured())
		lively, livelyErr := r.PlaceWithServing(class, measured(), everythingUp)
		if (bareErr == nil) != (livelyErr == nil) {
			t.Fatalf("class %q: reporting every host UP changed whether placement succeeded", class)
		}
		if bareErr != nil {
			continue
		}
		if bare.Zone != lively.Zone || bare.Model != lively.Model {
			t.Fatalf("class %q: an all-up snapshot moved the placement from %q/%q to %q/%q",
				class, bare.Zone, bare.Model, lively.Zone, lively.Model)
		}
	}
}

// TestServingNeverPlacesWorkOnADownCandidate sweeps every class against every
// single-host outage in the fixture. The gate is only worth having if this holds
// without exception, so it is asserted as a property rather than a case.
func TestServingNeverPlacesWorkOnADownCandidate(t *testing.T) {
	r := threeZoneRoster()
	hosts := []string{"tiny", "corp-mid", "corp-agentic", "frontier"}
	classes := []WorkClass{ClassRoutine, ClassNormalImpl, ClassUltraHard, ClassSecurityRelease}
	for _, dead := range hosts {
		s := servingReport([]PlacementZone{ZoneDevice, ZoneFleet, ZoneVendor}, map[string]ServingObservation{})
		for _, h := range hosts {
			s.Models[h] = up()
		}
		s.Models[dead] = down()
		for _, class := range classes {
			p, err := r.PlaceWithServing(class, measured(), s)
			if err != nil {
				continue // the ladder ran out; a loud refusal is a correct outcome
			}
			if p.Model == dead {
				t.Fatalf("class %q was placed on %q, which the snapshot reports DOWN", class, dead)
			}
		}
	}
}

// TestServingWholeLadderDownIsALoudRefusal. When nothing is answering anywhere,
// the placer refuses and names the reason on every rung. Silently picking a dead
// vendor because it was the last rung would turn an outage into a mystery.
func TestServingWholeLadderDownIsALoudRefusal(t *testing.T) {
	r := threeZoneRoster()
	s := servingReport([]PlacementZone{ZoneDevice, ZoneFleet, ZoneVendor}, map[string]ServingObservation{
		"tiny": down(), "corp-mid": down(), "corp-agentic": down(), "frontier": down(),
	})
	_, err := r.PlaceWithServing(ClassRoutine, measured(), s)
	if err == nil {
		t.Fatalf("a ladder with nothing answering still produced a placement")
	}
	if !strings.Contains(err.Error(), ReasonNoZoneCanServe) || !strings.Contains(err.Error(), ReasonZoneServingDown) {
		t.Fatalf("refusal does not name both the outcome and the cause: %v", err)
	}
}

// TestServingLadderTokensStayInTheClosedVocabulary. A status surface renders these
// verbatim; a token invented here would be free text with extra steps.
func TestServingLadderTokensStayInTheClosedVocabulary(t *testing.T) {
	known := map[string]bool{
		ReasonPlacedInZone: true, ReasonZoneNoCandidate: true, ReasonZoneUnderTier: true,
		ReasonZoneUnmeasured: true, ReasonEscalatedPast: true, ReasonNoZoneCanServe: true,
		ReasonTopRungUnmeasured: true, ReasonZoneNotReached: true,
		ReasonZoneServingDown: true, ReasonZoneServingUnknown: true,
		ReasonZoneServingStale: true, ReasonZoneServingDegraded: true,
	}
	r := threeZoneRoster()
	reports := []ServingReport{
		servingReport([]PlacementZone{ZoneFleet}, map[string]ServingObservation{"corp-mid": down()}),
		servingReport([]PlacementZone{ZoneDevice, ZoneFleet}, map[string]ServingObservation{"tiny": degraded()}),
		servingReport([]PlacementZone{ZoneFleet}, map[string]ServingObservation{"corp-mid": unsure(), "corp-agentic": up()}),
		{Schema: ServingReportSchema, AsOfUnix: 1_000_000, MaxAgeSeconds: 60,
			Covers: []PlacementZone{ZoneFleet},
			Models: map[string]ServingObservation{"corp-mid": {State: ServingUp, ObservedUnix: 1}, "corp-agentic": {State: ServingUp, ObservedUnix: 999_999}}},
	}
	for i, s := range reports {
		for _, class := range []WorkClass{ClassRoutine, ClassNormalImpl, ClassUltraHard} {
			p, err := r.PlaceWithServing(class, measured(), s)
			if err != nil {
				continue
			}
			for _, v := range p.Ladder {
				for _, rsn := range v.Reasons {
					if !known[rsn] {
						t.Fatalf("report %d class %q rung %q emitted unknown token %q", i, class, v.Zone, rsn)
					}
				}
			}
		}
	}
}

// TestServingReportValidateRefusesWhatWouldSilentlyDisableTheGate. Every case here
// is one where ignoring the malformed field would leave the operator believing a
// rung was gated when it was not — which is strictly worse than no gate at all,
// because it is believed.
func TestServingReportValidateRefusesWhatWouldSilentlyDisableTheGate(t *testing.T) {
	valid := []struct {
		name string
		rep  ServingReport
	}{
		{"the zero report Place delegates with", ServingReport{}},
		{"a full report", servingReport([]PlacementZone{ZoneFleet}, map[string]ServingObservation{"corp-mid": up()})},
		{"coverage with no observations", servingReport([]PlacementZone{ZoneFleet}, nil)},
	}
	for _, c := range valid {
		if err := c.rep.Validate(); err != nil {
			t.Fatalf("%s: rejected a valid report: %v", c.name, err)
		}
	}

	invalid := []struct {
		name string
		rep  ServingReport
	}{
		{"a zone nobody defined, which would gate nothing", servingReport([]PlacementZone{"datacenter"}, map[string]ServingObservation{"corp-mid": up()})},
		{"a state nobody defined", servingReport([]PlacementZone{ZoneFleet}, map[string]ServingObservation{"corp-mid": {State: "flaky"}})},
		{"an empty state, which is not 'up'", servingReport([]PlacementZone{ZoneFleet}, map[string]ServingObservation{"corp-mid": {}})},
		{"an observation under an empty model id", servingReport([]PlacementZone{ZoneFleet}, map[string]ServingObservation{"": up()})},
		{"a negative freshness bound", ServingReport{Schema: ServingReportSchema, MaxAgeSeconds: -1, Models: map[string]ServingObservation{"corp-mid": up()}}},
		{"a schema from some other producer", ServingReport{Schema: "fak.modelroute.serving.v2", Models: map[string]ServingObservation{"corp-mid": up()}}},
		{"observations with no schema at all", ServingReport{Models: map[string]ServingObservation{"corp-mid": up()}}},
		{"coverage with no schema at all", ServingReport{Covers: []PlacementZone{ZoneFleet}}},
	}
	for _, c := range invalid {
		if err := c.rep.Validate(); err == nil {
			t.Fatalf("%s: accepted", c.name)
		}
	}
}

// TestPlaceWithServingRefusesAnInvalidSnapshotRatherThanIgnoringIt holds the walk
// to the rule Place already keeps for an unresolvable candidate: a typo in a
// placement config surfaces as a misconfiguration, never as traffic quietly
// continuing to a host the operator believed they had gated.
func TestPlaceWithServingRefusesAnInvalidSnapshotRatherThanIgnoringIt(t *testing.T) {
	r := threeZoneRoster()
	bad := servingReport([]PlacementZone{"datacentre"}, map[string]ServingObservation{"corp-mid": down()})
	p, err := r.PlaceWithServing(ClassNormalImpl, measured(), bad)
	if err == nil {
		t.Fatalf("an invalid snapshot was ignored and placement proceeded to %q", p.Model)
	}
	if !strings.Contains(err.Error(), "datacentre") {
		t.Fatalf("refusal does not name the offending value: %v", err)
	}
	if p.Model != "" || p.Zone != "" {
		t.Fatalf("a refused placement still returned a decision: %+v", p)
	}
}

// TestServingIsDeterministic. The report is map-backed, and placement must not
// inherit map iteration order — an artifact that reads differently on replay
// cannot be used to explain a bill.
func TestServingIsDeterministic(t *testing.T) {
	r := threeZoneRoster()
	s := servingReport([]PlacementZone{ZoneDevice, ZoneFleet, ZoneVendor}, map[string]ServingObservation{
		"tiny": down(), "corp-mid": down(), "corp-agentic": degraded(), "frontier": up(),
	})
	first := placeOrFail(t, r, ClassNormalImpl, measured(), s)
	for i := 0; i < 50; i++ {
		again := placeOrFail(t, r, ClassNormalImpl, measured(), s)
		if !reflect.DeepEqual(first, again) {
			t.Fatalf("iteration %d differs:\n  %+v\n  %+v", i, first, again)
		}
	}
}

// TestServingReportCarriesNoCredential holds the "credential-free" claim in the
// doc comment to something checkable. A liveness snapshot gets pasted into
// issues and dashboards; the day someone adds a base_url or a token to it, that
// stops being safe, and a comment will not notice.
func TestServingReportCarriesNoCredential(t *testing.T) {
	banned := []string{"cred", "token", "secret", "key", "password", "auth", "url", "header"}
	for _, typ := range []reflect.Type{reflect.TypeOf(ServingReport{}), reflect.TypeOf(ServingObservation{})} {
		for i := 0; i < typ.NumField(); i++ {
			name := strings.ToLower(typ.Field(i).Name)
			for _, b := range banned {
				if strings.Contains(name, b) {
					t.Fatalf("%s.%s looks like a credential or endpoint field; a liveness snapshot must stay safe to publish",
						typ.Name(), typ.Field(i).Name)
				}
			}
		}
	}
}
