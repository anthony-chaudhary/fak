package workdelivery

import (
	"encoding/json"
	"reflect"
	"strings"
	"testing"
)

func TestDiagnoseKnownLeaf(t *testing.T) {
	got, err := Diagnose(FailureObservation{
		Scope:        FailureScope{ID: "ci", Units: []DiagnosticUnit{{ID: "gateway"}, {ID: "engine"}}},
		FailedUnitID: "engine", Class: FailureCompile, Gate: "go-build",
		Evidence:     []Evidence{{Kind: "stderr", Reference: "build.log", Witnessed: true}},
		CheckCommand: "fak validate --unit {unit}",
	})
	if err != nil {
		t.Fatal(err)
	}
	if got.Kind != DiagnosisLeaf || got.Unit == nil || got.Unit.ID != "engine" {
		t.Fatalf("got %#v", got)
	}
	if got.NextAction != "fak validate --unit engine" {
		t.Fatalf("next = %q", got.NextAction)
	}
	if got.Gate != "go-build" || len(got.Evidence) != 1 {
		t.Fatalf("diagnostic identity lost: %#v", got)
	}
}

func TestDiagnoseRecursivelyConverges(t *testing.T) {
	observation := FailureObservation{Scope: FailureScope{ID: "ci", Units: []DiagnosticUnit{
		{ID: "agent/a", Tree: "agent", IndependentlyCheckable: true},
		{ID: "agent/b", Tree: "agent", IndependentlyCheckable: true},
		{ID: "gateway/a", Tree: "gateway", IndependentlyCheckable: true},
		{ID: "gateway/b", Tree: "gateway", IndependentlyCheckable: true},
	}}, Class: FailureUnknown, Gate: "ci", CheckCommand: "check {unit}"}
	first, err := Diagnose(observation)
	if err != nil {
		t.Fatal(err)
	}
	if first.Kind != DiagnosisSplit || len(first.Children) != 2 {
		t.Fatalf("first = %#v", first)
	}
	if ids(first.Children[0]) != "agent/a,agent/b" || ids(first.Children[1]) != "gateway/a,gateway/b" {
		t.Fatalf("nondeterministic children: %#v", first.Children)
	}

	second, err := Diagnose(FailureObservation{Scope: first.Children[1], Class: FailureUnknown, Gate: "ci", CheckCommand: "check {unit}"})
	if err != nil {
		t.Fatal(err)
	}
	if second.Kind != DiagnosisSplit || len(second.Children) != 2 {
		t.Fatalf("second = %#v", second)
	}
	third, err := Diagnose(FailureObservation{Scope: second.Children[1], Class: FailureUnknown, Gate: "ci", CheckCommand: "check {unit}"})
	if err != nil {
		t.Fatal(err)
	}
	if third.Kind != DiagnosisLeaf || third.Unit == nil || third.Unit.ID != "gateway/b" {
		t.Fatalf("third = %#v", third)
	}
}

func TestDiagnoseCycleIsTypedIrreducible(t *testing.T) {
	got, err := Diagnose(FailureObservation{Scope: FailureScope{ID: "cycle", Units: []DiagnosticUnit{
		{ID: "a", Tree: "a", Dependencies: []string{"b"}}, {ID: "b", Tree: "b", Dependencies: []string{"a"}},
	}}, Class: FailureCompile, Gate: "compile"})
	if err != nil {
		t.Fatal(err)
	}
	if got.Kind != DiagnosisIrreducible || got.Blocker == nil || got.Blocker.Code != "CYCLIC_DEPENDENCY" {
		t.Fatalf("got %#v", got)
	}
	if got.Blocker.Detail != "a -> b -> a" {
		t.Fatalf("cycle = %q", got.Blocker.Detail)
	}
}

func TestDiagnoseNoBoundaryIsTypedIrreducible(t *testing.T) {
	got, err := Diagnose(FailureObservation{Scope: FailureScope{ID: "external", Units: []DiagnosticUnit{{ID: "a"}, {ID: "b"}}}, Class: FailureExternal, Gate: "provider"})
	if err != nil {
		t.Fatal(err)
	}
	if got.Kind != DiagnosisIrreducible || got.Blocker == nil || got.Blocker.Code != "NO_DISCRIMINATOR" {
		t.Fatalf("got %#v", got)
	}
	if got.Blocker.MissingDiscriminator == "" || got.NextAction == "" {
		t.Fatalf("missing recovery contract: %#v", got)
	}
}

func TestDiagnosisJSONGolden(t *testing.T) {
	got, err := Diagnose(FailureObservation{Scope: FailureScope{ID: "ci", Units: []DiagnosticUnit{
		{ID: "agent", Tree: "internal/agent", IndependentlyCheckable: true}, {ID: "gateway", Tree: "internal/gateway", IndependentlyCheckable: true},
	}}, Class: FailureUnknown, Gate: "make-ci"})
	if err != nil {
		t.Fatal(err)
	}
	data, err := json.MarshalIndent(got, "", "  ")
	if err != nil {
		t.Fatal(err)
	}
	const want = `{
  "schema": "fak.work-delivery-diagnosis/v1",
  "kind": "split",
  "scope_id": "ci",
  "class": "unknown",
  "gate": "make-ci",
  "children": [
    {
      "id": "ci/1",
      "units": [
        {
          "id": "agent",
          "tree": "internal/agent",
          "independently_checkable": true
        }
      ]
    },
    {
      "id": "ci/2",
      "units": [
        {
          "id": "gateway",
          "tree": "internal/gateway",
          "independently_checkable": true
        }
      ]
    }
  ],
  "next_action": "run the gate for each child scope, then diagnose the failing child"
}`
	if strings.TrimSpace(string(data)) != strings.TrimSpace(want) {
		t.Fatalf("golden mismatch\n%s", data)
	}
}

func ids(scope FailureScope) string {
	values := make([]string, len(scope.Units))
	for i, unit := range scope.Units {
		values[i] = unit.ID
	}
	return strings.Join(values, ",")
}

func TestPartitionStableAcrossInputOrder(t *testing.T) {
	a := FailureObservation{Scope: FailureScope{ID: "s", Units: []DiagnosticUnit{{ID: "z", Tree: "b"}, {ID: "a", Tree: "a"}}}, Class: FailureTest, Gate: "test"}
	b := a
	b.Scope.Units = []DiagnosticUnit{a.Scope.Units[1], a.Scope.Units[0]}
	x, _ := Diagnose(a)
	y, _ := Diagnose(b)
	if !reflect.DeepEqual(x, y) {
		t.Fatalf("partition depends on input order:\n%#v\n%#v", x, y)
	}
}
