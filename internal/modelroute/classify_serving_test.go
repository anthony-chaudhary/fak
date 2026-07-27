package modelroute

import (
	"reflect"
	"testing"
)

// Witnesses for PlaceSubjectWithServing — the subject-shaped entry point's half of the
// serving pair (epic #5416, tracks E and H).
//
// The class-shaped call already has TestPlaceWithoutAServingReportIsTheOldPlacementExactly
// asserting that an empty snapshot changes nothing. That property does NOT transfer for
// free: a composed call could satisfy it underneath while consulting the report itself, or
// while quietly dropping it. Both halves are asserted here, because a report that reaches
// the ladder for one entry point and not the other is precisely how an operator ends up
// gating a rung on one surface and not on the one their fleet actually calls.

// subjectFor builds the subject a declared work class arrives on.
func subjectFor(class WorkClass) Subject {
	return Subject{Labels: map[string]string{"work_class": string(class)}}
}

// TestASubjectPlacedWithNoSnapshotIsThePlainSubjectPlacementExactly is the delegation
// being honest: PlaceSubject is PlaceSubjectWithServing with an empty report, so an
// empty report must be indistinguishable from not having one. An equality over every
// class and pool is the only shape of this claim an edit cannot satisfy while
// special-casing a rung.
func TestASubjectPlacedWithNoSnapshotIsThePlainSubjectPlacementExactly(t *testing.T) {
	r := threeZoneRoster()
	classes := []WorkClass{ClassRoutine, ClassNormalImpl, ClassUltraHard, ClassSecurityRelease, WorkClass("nobody-declared-this")}
	pools := map[string][]Candidate{"measured": measured(), "empty": nil}
	for _, class := range classes {
		for name, pool := range pools {
			wantP, wantC, wantErr := r.PlaceSubject(subjectFor(class), pool)
			gotP, gotC, gotErr := r.PlaceSubjectWithServing(subjectFor(class), pool, ServingReport{})
			if (wantErr == nil) != (gotErr == nil) {
				t.Fatalf("class %q pool %s: PlaceSubject err=%v but PlaceSubjectWithServing err=%v", class, name, wantErr, gotErr)
			}
			if !reflect.DeepEqual(wantP, gotP) || !reflect.DeepEqual(wantC, gotC) {
				t.Fatalf("class %q pool %s: an empty snapshot changed the answer:\n  want %+v / %+v\n  got  %+v / %+v",
					class, name, wantP, wantC, gotP, gotC)
			}
		}
	}
}

// And the other half: a snapshot handed to the subject-shaped call must actually reach the
// ladder. A composed entry point that accepts a report and forgets to pass it down would
// pass the equality above perfectly — it would look like a working gate on every surface
// that calls it, while every dead host kept taking work.
func TestASnapshotHandedToTheSubjectCallReallyGatesTheRung(t *testing.T) {
	r := threeZoneRoster()
	// Routine work with a measured 4B on the laptop: the device rung is where this
	// belongs, and does land, with no snapshot in play.
	base, _, err := r.PlaceSubject(subjectFor(ClassRoutine), measured())
	if err != nil {
		t.Fatal(err)
	}
	if base.Zone != ZoneDevice {
		t.Fatalf("the fixture no longer places routine work on the device rung (got %q); the gate below would prove nothing", base.Zone)
	}
	// Now the laptop's server is not answering.
	rep := servingReport([]PlacementZone{ZoneDevice}, map[string]ServingObservation{"tiny": down()})
	p, cls, err := r.PlaceSubjectWithServing(subjectFor(ClassRoutine), measured(), rep)
	if err != nil {
		t.Fatal(err)
	}
	if p.Zone == ZoneDevice {
		t.Fatalf("a DOWN observation did not keep work off the rung through the subject call: %+v", p)
	}
	if !p.FailedOver {
		t.Errorf("the placement moved for a liveness reason and does not report it as a failover: %+v", p)
	}
	// The classification still rides back untouched — the snapshot answers where work
	// can run, never what the work IS.
	if cls.Class != ClassRoutine || !cls.Declared {
		t.Errorf("the snapshot disturbed the classification: %+v", cls)
	}
	if !hasReason(rungReasons(p, ZoneDevice), ReasonZoneServingDown) {
		t.Errorf("the device rung does not record why it was passed over: %v", rungReasons(p, ZoneDevice))
	}
}
