package workdelivery

import (
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"testing"
	"time"
)

type deliveryE2EWitness struct {
	Schema     string               `json:"schema"`
	Unit       WorkUnit             `json:"unit"`
	CompileSet CompileSet           `json:"compile_set"`
	Receipts   []AdapterObservation `json:"receipts"`
}

func TestCapturedDeliverySpine(t *testing.T) {
	fixture := readDeliveryE2EFixture[deliveryE2EWitness](t, "happy-path.json")
	if fixture.Schema != "fak.work-delivery-e2e/v1" {
		t.Fatalf("schema=%q", fixture.Schema)
	}
	now := time.Date(2026, 8, 17, 18, 30, 0, 0, time.UTC)
	first := produceCapturedDeliverySpine(t, now)
	second := produceCapturedDeliverySpine(t, now)
	if !reflect.DeepEqual(first, second) {
		firstJSON, _ := json.MarshalIndent(first, "", "  ")
		secondJSON, _ := json.MarshalIndent(second, "", "  ")
		t.Fatalf("two complete explicit-time producer witnesses differ:\nfirst=%s\nsecond=%s", firstJSON, secondJSON)
	}
	for _, path := range first.CompileSet.Admitted {
		if path == "fixture/recorded_broken.go" {
			t.Fatal("committed excluded source entered compile set")
		}
	}
	if !reflect.DeepEqual(first, fixture) {
		gotJSON, _ := json.MarshalIndent(first, "", "  ")
		t.Fatalf("captured spine drifted:\n%s", gotJSON)
	}
}

func produceCapturedDeliverySpine(t *testing.T, now time.Time) deliveryE2EWitness {
	t.Helper()
	const unitID = "issue-7106"
	set, err := DeriveCompileSetAt([]WorkUnit{
		compileUnit("recorded-excluded", AdmissionExcluded, "fixture/recorded_broken.go"),
		compileUnit(unitID, AdmissionAdmitted, "fixture/active.go"),
	}, now)
	if err != nil {
		t.Fatal(err)
	}
	unit := WorkUnit{Schema: Schema, ID: unitID, Artifacts: []Artifact{{Path: "fixture/active.go", Kind: "go-source"}}, Axes: InitialAxes()}
	recorded, err := RecordingObservation(unit, "abc123", "fak commit", now)
	if err != nil {
		t.Fatal(err)
	}
	unit, err = Apply(unit, *recorded.Receipt)
	if err != nil {
		t.Fatal(err)
	}
	unit.Axes.Admission = AdmissionAdmitted // witnessed by CompileSet.Receipts, independently of commit.
	verified, err := VerificationObservation(unit, true, "affected-tests", "test://green", "ci", now)
	if err != nil {
		t.Fatal(err)
	}
	unit, err = Apply(unit, *verified.Receipt)
	if err != nil {
		t.Fatal(err)
	}
	integrated, err := IntegrationObservation(unit, true, "origin/main@abc123", "sync", now)
	if err != nil {
		t.Fatal(err)
	}
	unit, err = Apply(unit, *integrated.Receipt)
	if err != nil {
		t.Fatal(err)
	}
	readyObs, err := ReleaseReadinessObservation(unit, "shipgate://run/9", "release", now)
	if err != nil {
		t.Fatal(err)
	}
	unit, err = Apply(unit, *readyObs.Receipt)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := RequireReleaseReady(unit, readyObs.Receipt); err != nil {
		t.Fatal(err)
	}
	got := []AdapterObservation{recorded, verified, integrated, readyObs}
	if len(set.Excluded) != 1 || set.Excluded[0] != "fixture/recorded_broken.go" {
		t.Fatalf("recorded-but-unadmitted witness missing from excluded set: %+v", set.Excluded)
	}
	for _, artifact := range set.Admitted {
		if artifact == "fixture/recorded_broken.go" {
			t.Fatalf("recorded-but-unadmitted artifact entered compile set: %+v", set.Admitted)
		}
	}
	return deliveryE2EWitness{Schema: "fak.work-delivery-e2e/v1", Unit: unit, CompileSet: set, Receipts: got}
}

func TestCapturedAggregateFailureConvergesToLeaf(t *testing.T) {
	observations := readDeliveryE2EFixture[[]FailureObservation](t, "failure-path.json")
	wantKinds := []DiagnosisKind{DiagnosisSplit, DiagnosisSplit, DiagnosisLeaf}
	for i, observation := range observations {
		diagnosis, err := Diagnose(observation)
		if err != nil {
			t.Fatal(err)
		}
		if diagnosis.Kind != wantKinds[i] {
			t.Fatalf("step %d kind=%s", i, diagnosis.Kind)
		}
		if i < len(observations)-1 {
			found := false
			for _, child := range diagnosis.Children {
				if child.ID == observations[i+1].Scope.ID {
					found = true
				}
			}
			if !found {
				t.Fatalf("step %d does not contain next failing child %q: %+v", i, observations[i+1].Scope.ID, diagnosis.Children)
			}
		} else if diagnosis.Unit == nil || diagnosis.Unit.ID != "unit-bad-test" || diagnosis.Gate != "full-tests" || len(diagnosis.Evidence) == 0 {
			t.Fatalf("final diagnosis not exact: %+v", diagnosis)
		}
	}
}

func readDeliveryE2EFixture[T any](t *testing.T, name string) T {
	t.Helper()
	var value T
	data, err := os.ReadFile(filepath.Join("testdata", "e2e", name))
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(data, &value); err != nil {
		t.Fatal(err)
	}
	return value
}
