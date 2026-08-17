package workdelivery

import (
	"testing"
	"time"
)

func TestAdaptersChangeOnlyTheirOwnedAxis(t *testing.T) {
	now := time.Date(2026, 8, 17, 18, 0, 0, 0, time.UTC)
	base := WorkUnit{Schema: Schema, ID: "issue-7105", Artifacts: []Artifact{{Path: "internal/workdelivery/adapters.go", Kind: "source"}}, Axes: InitialAxes()}

	recorded, err := RecordingObservation(base, "abc123", "agent", now)
	if err != nil {
		t.Fatal(err)
	}
	afterRecord, err := Apply(base, *recorded.Receipt)
	if err != nil {
		t.Fatal(err)
	}
	if afterRecord.Axes.Authoring != AuthoringRecorded || afterRecord.Axes.Admission != AdmissionUndeclared || afterRecord.Axes.Verification != VerificationUnverified || afterRecord.Axes.Integration != IntegrationUnintegrated || afterRecord.Axes.Release != ReleaseNotReady {
		t.Fatalf("recording collapsed independent state: %+v", afterRecord.Axes)
	}

	admitted := afterRecord
	admitted.Axes.Admission = AdmissionAdmitted
	verified, err := VerificationObservation(admitted, true, "affected-tests", "test://green", "ci", now)
	if err != nil {
		t.Fatal(err)
	}
	afterVerify, err := Apply(admitted, *verified.Receipt)
	if err != nil {
		t.Fatal(err)
	}
	if afterVerify.Axes.Verification != VerificationPassed || afterVerify.Axes.Integration != IntegrationUnintegrated || afterVerify.Axes.Release != ReleaseNotReady {
		t.Fatalf("verification collapsed downstream state: %+v", afterVerify.Axes)
	}

	integrated, err := IntegrationObservation(afterVerify, true, "origin/main@abc123", "sync", now)
	if err != nil {
		t.Fatal(err)
	}
	afterPush, err := Apply(afterVerify, *integrated.Receipt)
	if err != nil {
		t.Fatal(err)
	}
	if afterPush.Axes.Integration != IntegrationIntegrated || afterPush.Axes.Release != ReleaseNotReady {
		t.Fatalf("integration inferred release readiness: %+v", afterPush.Axes)
	}
	if _, err := RequireReleaseReady(afterPush, nil); err == nil {
		t.Fatal("pushed green unit was inferred release-ready")
	}
}

func TestBlockedObservationRequiresCanonicalExactIdentity(t *testing.T) {
	obs, err := BlockedObservation(AdapterFleet, "issue-7105/adapters", "full-tests", "unknown-irreducible", []Evidence{{Kind: "log", Reference: "ci://run/7"}}, "bisect independent checks")
	if err != nil {
		t.Fatal(err)
	}
	if obs.UnitID != "issue-7105/adapters" || obs.Stage != "full-tests" || obs.Bottleneck != "unknown-irreducible" {
		t.Fatalf("lost exact blocker identity: %+v", obs)
	}
	if _, err := BlockedObservation(AdapterFleet, "", "full-tests", "unknown-irreducible", nil, "split"); err == nil {
		t.Fatal("empty aggregate unit should fail closed")
	}
	if _, err := BlockedObservation(AdapterDispatch, "unit", "ci-red", "unknown-irreducible", nil, "split"); err == nil {
		t.Fatal("non-canonical stage should fail closed")
	}
}

func TestFailedVerificationDoesNotMutateIntegrationOrRelease(t *testing.T) {
	unit := WorkUnit{Schema: Schema, ID: "unit-red", Axes: InitialAxes()}
	unit.Axes.Authoring = AuthoringRecorded
	unit.Axes.Admission = AdmissionAdmitted
	obs, err := VerificationObservation(unit, false, "full-tests", "ci://run/red", "ci", time.Now())
	if err != nil {
		t.Fatal(err)
	}
	updated, err := Apply(unit, *obs.Receipt)
	if err != nil {
		t.Fatal(err)
	}
	if updated.Axes.Verification != VerificationFailed || updated.Axes.Integration != IntegrationUnintegrated || updated.Axes.Release != ReleaseNotReady {
		t.Fatalf("failed verification mutated unrelated axes: %+v", updated.Axes)
	}
	if obs.Stage != "build" || obs.Bottleneck != "verification-failure" {
		t.Fatalf("non-canonical failure: %+v", obs)
	}
}

func TestReleaseAdmissionRequiresMatchingWitnessedReceipt(t *testing.T) {
	unit := WorkUnit{Schema: Schema, ID: "release-unit", Axes: Axes{AuthoringRecorded, AdmissionAdmitted, VerificationPassed, IntegrationIntegrated, ReleaseNotReady}}
	obs, err := ReleaseReadinessObservation(unit, "shipgate://run/9", "release-gate", time.Unix(9, 0))
	if err != nil {
		t.Fatal(err)
	}
	ready, err := Apply(unit, *obs.Receipt)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := RequireReleaseReady(ready, nil); err == nil {
		t.Fatal("ready state without its receipt should fail closed")
	}
	if _, err := RequireReleaseReady(ready, obs.Receipt); err != nil {
		t.Fatalf("explicit witnessed readiness refused: %v", err)
	}
	unwitnessed := *obs.Receipt
	unwitnessed.Evidence = []Evidence{{Kind: "release-readiness", Reference: "shipgate://run/9"}}
	if _, err := RequireReleaseReady(ready, &unwitnessed); err == nil {
		t.Fatal("unwitnessed readiness evidence should fail closed")
	}
}
