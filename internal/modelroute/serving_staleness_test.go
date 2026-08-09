package modelroute

import "testing"

// STALENESS IS MEASURED AGAINST A CLOCK THE REPORT DOES NOT OWN (issue #5636).
//
// serving.go already has a freshness rule, and it is entirely INTERNAL: fresh()
// compares an observation's stamp against the report's OWN AsOfUnix. That answers
// "was this observation recent when the snapshot was assembled?" and nothing else.
// A snapshot is therefore self-certifying — a producer that died last Tuesday left
// behind a file whose as-of and observation stamps are ten seconds apart forever, so
// every internal freshness check passes forever, and the ladder keeps placing work on
// a rung nobody has looked at in a week.
//
// That is the gap these tests pin. The missing question is "is the SNAPSHOT recent?",
// which cannot be answered from inside the document; it needs a clock the reader
// supplies. Placement must stay pure, so the clock never reaches Place — the rule is
// a transform applied to the report BEFORE it is placed against, and it is the
// consumer (cmd/fak, which is already allowed to read a clock and a filesystem) that
// supplies now.

// staleFixture is a snapshot that is INTERNALLY perfect and externally ancient: both
// company hosts observed up, ten seconds before the snapshot was assembled, under a
// declared sixty-second bound. Every rule in serving.go's header passes on it. It was
// assembled a week before anyone read it.
const (
	staleAsOf   int64 = 1_700_000_000
	staleWeekOn       = staleAsOf + 7*24*60*60
)

func staleFixture() ServingReport {
	return ServingReport{
		Schema:        ServingReportSchema,
		AsOfUnix:      staleAsOf,
		MaxAgeSeconds: 60,
		Covers:        []PlacementZone{ZoneFleet},
		Models: map[string]ServingObservation{
			"corp-mid":     {State: ServingUp, ObservedUnix: staleAsOf - 10},
			"corp-agentic": {State: ServingUp, ObservedUnix: staleAsOf - 10},
		},
	}
}

// TestServingReportStaleness is the witness issue #5636 names.
//
// It asserts the premise first — that a week-old snapshot is honored today exactly as
// if it had been taken a moment ago — because the value of the rule is only legible
// against the behaviour it replaces. Then it asserts the rule: read at a clock a week
// past its as-of stamp, the same document must stop speaking.
func TestServingReportStaleness(t *testing.T) {
	r := threeZoneRoster()

	// THE PREMISE. Placed as written, the ancient snapshot still lands ordinary
	// implementation work on the company fleet. Nothing here is wrong per serving.go's
	// own rules, which is exactly the problem: the report certifies its own freshness.
	raw := placeOrFail(t, r, ClassNormalImpl, measured(), staleFixture())
	if raw.Zone != ZoneFleet {
		t.Fatalf("premise broken: a self-consistent snapshot should still place on the fleet rung today, got %q", raw.Zone)
	}

	// THE RULE. The same bytes, read a week later, must degrade to UNKNOWN. Unknown is
	// the specific answer required: the fleet is not observed down, it is unobserved,
	// and the ladder's own vocabulary distinguishes those.
	degraded := staleFixture().DegradeStale(staleWeekOn)
	for _, model := range []string{"corp-mid", "corp-agentic"} {
		obs, ok := degraded.Models[model]
		if !ok {
			t.Fatalf("%s: a stale observation was DELETED rather than degraded; outside declared coverage silence gates nothing, so dropping the entry fails OPEN — the exact failure the rule exists to prevent", model)
		}
		if obs.State != ServingUnknown {
			t.Errorf("%s: stale observation degraded to %q, want %q", model, obs.State, ServingUnknown)
		}
		if obs.State == ServingDown {
			t.Errorf("%s: a snapshot nobody refreshed is not an OUTAGE; degrading to down would report a fleet failure that was never observed", model)
		}
	}

	// The ladder must now walk PAST the fleet rung rather than onto it, and must say
	// why in the closed vocabulary.
	placed := placeOrFail(t, r, ClassNormalImpl, measured(), degraded)
	if placed.Zone == ZoneFleet {
		t.Fatalf("a week-stale snapshot still placed work on the fleet rung: a ladder verdict must not read as measured when it was asserted")
	}
	reasons := rungReasons(placed, ZoneFleet)
	if !hasReason(reasons, ReasonZoneServingUnknown) {
		t.Errorf("fleet rung reasons %v carry no %q token; the escalation has to be explainable or the bill is a mystery", reasons, ReasonZoneServingUnknown)
	}
}

// TestDegradeStaleLeavesAFreshReportByteIdentical is the no-op half. An operator whose
// producer is alive must see exactly the placement they see today; a rule that fires on
// a healthy snapshot would be a regression dressed as a safety feature.
func TestDegradeStaleLeavesAFreshReportByteIdentical(t *testing.T) {
	r := threeZoneRoster()
	// Read one second after assembly, well inside the declared bound.
	fresh := staleFixture().DegradeStale(staleAsOf + 1)
	want := placeOrFail(t, r, ClassNormalImpl, measured(), staleFixture())
	got := placeOrFail(t, r, ClassNormalImpl, measured(), fresh)
	if want.Zone != got.Zone {
		t.Fatalf("a fresh snapshot changed placement: want zone %q, got %q", want.Zone, got.Zone)
	}
	for model, obs := range fresh.Models {
		if obs.State != ServingUp {
			t.Errorf("%s: a fresh observation was degraded to %q", model, obs.State)
		}
	}
	if fresh.StaleAsOf(staleAsOf + 1) {
		t.Error("a snapshot one second old reads as stale under a sixty-second bound")
	}
}

// TestStaleAsOfRefusesToGuessWithoutABound pins the three ways the rule declines to
// fire. Each one is a case where firing would either invent a bound the operator never
// declared or duplicate a refusal serving.go already makes — and in no case does
// declining leave a stale positive standing.
func TestStaleAsOfRefusesToGuessWithoutABound(t *testing.T) {
	base := staleFixture()

	// No bound declared. The honest reading of declining to state a TTL is that
	// observations are honored at any age; that is already what fresh() does, and the
	// wall clock must not invent a bound nobody asked for.
	noBound := base
	noBound.MaxAgeSeconds = 0
	if noBound.StaleAsOf(staleWeekOn) {
		t.Error("a report declaring no freshness bound was called stale; the rule invented a TTL the operator never declared")
	}

	// No as-of stamp. There is nothing to measure an age against — but this is NOT a
	// fail-open, because serving.go's rule 3 already fails every observation in such a
	// report closed as unshowably fresh. The wall-clock rule has nothing to add.
	noStamp := base
	noStamp.AsOfUnix = 0
	if noStamp.StaleAsOf(staleWeekOn) {
		t.Error("an unstamped report was degraded by the wall-clock rule; rule 3 already skips every observation in it as stale, and degrading on top changes the token without changing the verdict")
	}
	for model, obs := range noStamp.DegradeStale(staleWeekOn).Models {
		if obs.State != ServingUp {
			t.Errorf("%s: DegradeStale rewrote an unstamped report's state to %q, changing the reason token an operator reads from %q to %q for no change in outcome",
				model, obs.State, ReasonZoneServingStale, ReasonZoneServingUnknown)
		}
	}

	// No clock. A caller with no clock cannot check the claim; it must not be granted
	// one either way, and the internal rule still governs.
	if base.StaleAsOf(0) {
		t.Error("the rule fired with no clock supplied")
	}
}

// TestAReportStampedInTheFutureIsStale mirrors the negative-age guard fresh() already
// holds. A producer whose clock runs ahead would otherwise pin a rung open forever,
// and a broken producer must never be able to do that.
func TestAReportStampedInTheFutureIsStale(t *testing.T) {
	base := staleFixture()
	if !base.StaleAsOf(staleAsOf - 1) {
		t.Fatal("a snapshot stamped in the reader's future was accepted; a producer with a fast clock can pin the fleet rung open forever")
	}
	for model, obs := range base.DegradeStale(staleAsOf - 1).Models {
		if obs.State != ServingUnknown {
			t.Errorf("%s: a future-stamped observation stayed %q", model, obs.State)
		}
	}
}

// TestDegradeStaleDoesNotMutateTheReceiver matters because the consumer summarises the
// snapshot it loaded and places against the degraded one. If the transform wrote
// through, the summary would report states that were never observed.
func TestDegradeStaleDoesNotMutateTheReceiver(t *testing.T) {
	original := staleFixture()
	_ = original.DegradeStale(staleWeekOn)
	for model, obs := range original.Models {
		if obs.State != ServingUp {
			t.Errorf("%s: DegradeStale mutated its receiver (now %q); the caller's own copy must survive so the summary can still report what was observed", model, obs.State)
		}
	}
}

// TestADegradedStaleReportStillValidates guards the shape: the transform must produce a
// report the rest of the package will accept, or the consumer refuses its own output.
func TestADegradedStaleReportStillValidates(t *testing.T) {
	if err := staleFixture().DegradeStale(staleWeekOn).Validate(); err != nil {
		t.Fatalf("DegradeStale produced a report that does not validate: %v", err)
	}
}
