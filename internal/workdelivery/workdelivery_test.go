package workdelivery

import (
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
	"time"
)

func TestRecordingDoesNotImplyOtherAxes(t *testing.T) {
	unit := WorkUnit{Schema: Schema, ID: "issue-7101", Axes: InitialAxes()}
	receipt := Receipt{
		Schema: Schema, UnitID: unit.ID, Gate: "git.commit", ObservedAt: time.Unix(1, 0).UTC(),
		Transition: Transition{Axis: AxisAuthoring, From: string(AuthoringDraft), To: string(AuthoringRecorded)},
		Evidence:   []Evidence{{Kind: "commit", Reference: "abc123", Witnessed: true}},
	}
	got, err := Apply(unit, receipt)
	if err != nil {
		t.Fatal(err)
	}
	want := InitialAxes()
	want.Authoring = AuthoringRecorded
	if !reflect.DeepEqual(got.Axes, want) {
		t.Fatalf("axes = %#v, want %#v", got.Axes, want)
	}
}

func TestAxesRemainIndependent(t *testing.T) {
	states := []Axes{
		{Authoring: AuthoringRecorded, Admission: AdmissionExcluded, Verification: VerificationUnverified, Integration: IntegrationUnintegrated, Release: ReleaseNotReady},
		{Authoring: AuthoringRecorded, Admission: AdmissionAdmitted, Verification: VerificationPassed, Integration: IntegrationIntegrated, Release: ReleaseNotReady},
	}
	for _, axes := range states {
		unit := WorkUnit{Schema: Schema, ID: "independent", Axes: axes}
		if err := unit.Validate(); err != nil {
			t.Fatalf("%#v: %v", axes, err)
		}
	}
}

func TestBlockedReceiptDoesNotAdvanceAxis(t *testing.T) {
	unit := WorkUnit{Schema: Schema, ID: "blocked", Axes: InitialAxes()}
	receipt := Receipt{Schema: Schema, UnitID: unit.ID, Gate: "compile-set", ObservedAt: time.Unix(2, 0).UTC(),
		Transition: Transition{Axis: AxisAdmission, From: string(AdmissionUndeclared), To: string(AdmissionAdmitted)},
		Blocker:    &Blocker{Code: "MISSING_DECLARATION", Gate: "compile-set", MissingDiscriminator: "manifest", NextAction: "declare unit artifacts"},
	}
	got, err := Apply(unit, receipt)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(got, unit) {
		t.Fatalf("blocked receipt changed unit: %#v", got)
	}
}

func TestValidationFailsClosed(t *testing.T) {
	tests := []WorkUnit{
		{Schema: "", ID: "x", Axes: InitialAxes()},
		{Schema: Schema, ID: "", Axes: InitialAxes()},
		{Schema: Schema, ID: "x", Axes: Axes{Authoring: "committed", Admission: AdmissionUndeclared, Verification: VerificationUnverified, Integration: IntegrationUnintegrated, Release: ReleaseNotReady}},
	}
	for _, unit := range tests {
		if err := unit.Validate(); err == nil {
			t.Fatalf("Validate(%#v) succeeded", unit)
		}
	}
}

func TestJSONGoldens(t *testing.T) {
	for _, name := range []string{"recorded-unadmitted", "integrated-not-ready"} {
		path := filepath.Join("testdata", name+".json")
		data, err := os.ReadFile(path)
		if err != nil {
			t.Fatal(err)
		}
		var unit WorkUnit
		if err := json.Unmarshal(data, &unit); err != nil {
			t.Fatal(err)
		}
		if err := unit.Validate(); err != nil {
			t.Fatal(err)
		}
		got, err := json.MarshalIndent(unit, "", "  ")
		if err != nil {
			t.Fatal(err)
		}
		if strings.TrimSpace(string(got)) != strings.TrimSpace(string(data)) {
			t.Fatalf("%s does not round-trip\ngot:\n%s", path, got)
		}
	}
}
