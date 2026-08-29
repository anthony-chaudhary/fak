package workdelivery

import (
	"errors"
	"reflect"
	"testing"
	"time"
)

func compileUnit(id string, admission AdmissionState, paths ...string) WorkUnit {
	artifacts := make([]Artifact, 0, len(paths))
	for _, path := range paths {
		artifacts = append(artifacts, Artifact{Path: path, Kind: "go-source"})
	}
	axes := InitialAxes()
	axes.Authoring = AuthoringRecorded
	axes.Admission = admission
	return WorkUnit{Schema: Schema, ID: id, Artifacts: artifacts, Axes: axes}
}

func TestDeriveCompileSetSeparatesRecordedFromAdmitted(t *testing.T) {
	set, err := DeriveCompileSet([]WorkUnit{
		compileUnit("recorded", AdmissionExcluded, "fixture/recorded.go"),
		compileUnit("active", AdmissionAdmitted, "fixture/active.go"),
	})
	if err != nil {
		t.Fatal(err)
	}
	if got := set.Excluded; len(got) != 1 || got[0] != "fixture/recorded.go" {
		t.Fatalf("excluded = %v", got)
	}
	if got := set.Admitted; len(got) != 1 || got[0] != "fixture/active.go" {
		t.Fatalf("admitted = %v", got)
	}
	if len(set.Receipts) != 2 || set.Receipts[0].Gate != "go.compile-set" {
		t.Fatalf("receipts = %#v", set.Receipts)
	}
}

func TestDeriveCompileSetFailsClosedOnUndeclared(t *testing.T) {
	_, err := DeriveCompileSet([]WorkUnit{compileUnit("unknown", AdmissionUndeclared, "fixture/unknown.go")})
	var admissionErr *CompileAdmissionError
	if !errors.As(err, &admissionErr) || admissionErr.Code != "MISSING_DECLARATION" {
		t.Fatalf("error = %#v", err)
	}
}

func TestDeriveCompileSetRejectsConflictingArtifact(t *testing.T) {
	_, err := DeriveCompileSet([]WorkUnit{
		compileUnit("one", AdmissionExcluded, "fixture/shared.go"),
		compileUnit("two", AdmissionAdmitted, "fixture/shared.go"),
	})
	var admissionErr *CompileAdmissionError
	if !errors.As(err, &admissionErr) || admissionErr.Code != "CONFLICTING_DECLARATION" {
		t.Fatalf("error = %#v", err)
	}
}

func TestDeriveCompileSetAtIsDeterministicAndUsesOneObservationTime(t *testing.T) {
	observedAt := time.Date(2026, 8, 17, 18, 30, 0, 0, time.FixedZone("fixture", -7*60*60))
	units := []WorkUnit{
		compileUnit("recorded", AdmissionExcluded, "fixture/recorded.go"),
		compileUnit("active", AdmissionAdmitted, "fixture/active.go"),
	}
	first, err := DeriveCompileSetAt(units, observedAt)
	if err != nil {
		t.Fatal(err)
	}
	second, err := DeriveCompileSetAt(units, observedAt)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(first, second) {
		t.Fatalf("same explicit-time producer input drifted:\nfirst=%+v\nsecond=%+v", first, second)
	}
	wantTime := observedAt.UTC()
	for _, receipt := range first.Receipts {
		if !receipt.ObservedAt.Equal(wantTime) || receipt.ObservedAt.Location() != time.UTC {
			t.Fatalf("receipt %q observed_at=%s, want UTC %s", receipt.UnitID, receipt.ObservedAt, wantTime)
		}
	}
}

func TestDeriveCompileSetAtRejectsMissingObservationTime(t *testing.T) {
	_, err := DeriveCompileSetAt([]WorkUnit{compileUnit("active", AdmissionAdmitted, "fixture/active.go")}, time.Time{})
	var admissionErr *CompileAdmissionError
	if !errors.As(err, &admissionErr) || admissionErr.Code != "MISSING_OBSERVED_AT" {
		t.Fatalf("error = %#v", err)
	}
}
