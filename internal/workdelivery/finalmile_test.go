package workdelivery

import (
	"strings"
	"testing"
	"time"
)

func releaseReadyUnit() WorkUnit {
	return WorkUnit{Schema: Schema, ID: "issue-8781", Revision: "cmd/fak@r123+gabc", Axes: Axes{
		Authoring: AuthoringRecorded, Admission: AdmissionAdmitted, Verification: VerificationPassed,
		Integration: IntegrationIntegrated, Release: ReleaseReady,
		Activation: ActivationInactive, Acceptance: AcceptanceUnaccepted,
	}}
}

func activationReceipt(unit WorkUnit) Receipt {
	return Receipt{Schema: Schema, UnitID: unit.ID, Gate: "runtime.activation", ObservedAt: time.Unix(10, 0).UTC(),
		Transition:      Transition{Axis: AxisActivation, From: string(ActivationInactive), To: string(ActivationActivated)},
		RuntimeIdentity: &RuntimeIdentity{Revision: unit.Revision, BuildDigest: "sha256:build", ConfigDigest: "sha256:config"},
		Evidence:        []Evidence{{Kind: "runtime-version", Reference: "fak version output", Witnessed: true}},
	}
}

func TestReleaseReadyDoesNotImplyActivationOrAcceptance(t *testing.T) {
	unit := releaseReadyUnit()
	if unit.Axes.Activation != ActivationInactive || unit.Axes.Acceptance != AcceptanceUnaccepted {
		t.Fatalf("release implied final-mile state: %#v", unit.Axes)
	}
}

func TestActivationAndAcceptanceAdvanceIndependently(t *testing.T) {
	unit := releaseReadyUnit()
	activated, err := Apply(unit, activationReceipt(unit))
	if err != nil {
		t.Fatal(err)
	}
	if activated.Axes.Activation != ActivationActivated || activated.Axes.Acceptance != AcceptanceUnaccepted {
		t.Fatalf("activation changed wrong axes: %#v", activated.Axes)
	}
	receipt := Receipt{Schema: Schema, UnitID: unit.ID, Gate: "operator.journey", ObservedAt: time.Unix(11, 0).UTC(),
		Transition:      Transition{Axis: AxisAcceptance, From: string(AcceptanceUnaccepted), To: string(AcceptanceAccepted)},
		RuntimeIdentity: activated.Activated, Journey: "post-hook result appears in operator TUI",
		Evidence: []Evidence{{Kind: "captured-render", Reference: "docs/_witnesses/operator.txt", Witnessed: true}},
	}
	accepted, err := Apply(activated, receipt)
	if err != nil {
		t.Fatal(err)
	}
	if accepted.Axes.Acceptance != AcceptanceAccepted || accepted.Accepted.Journey != receipt.Journey {
		t.Fatalf("acceptance missing: %#v", accepted)
	}
}

func TestFinalMileReceiptsFailClosedOnIdentityMismatch(t *testing.T) {
	unit := releaseReadyUnit()
	bad := activationReceipt(unit)
	bad.RuntimeIdentity.Revision = "old-build"
	if _, err := Apply(unit, bad); err == nil || !strings.Contains(err.Error(), "does not match") {
		t.Fatalf("activation mismatch err = %v", err)
	}
	activated, err := Apply(unit, activationReceipt(unit))
	if err != nil {
		t.Fatal(err)
	}
	identity := *activated.Activated
	identity.ConfigDigest = "sha256:stale-config"
	badAcceptance := Receipt{Schema: Schema, UnitID: unit.ID, Gate: "operator.journey", ObservedAt: time.Unix(12, 0).UTC(),
		Transition: Transition{Axis: AxisAcceptance, From: string(AcceptanceUnaccepted), To: string(AcceptanceAccepted)}, RuntimeIdentity: &identity,
		Journey: "TUI click changes view", Evidence: []Evidence{{Kind: "render", Reference: "capture", Witnessed: true}}}
	if _, err := Apply(activated, badAcceptance); err == nil || !strings.Contains(err.Error(), "does not match") {
		t.Fatalf("acceptance mismatch err = %v", err)
	}
}
