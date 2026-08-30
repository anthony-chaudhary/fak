package trajectoryassurance

import (
	"encoding/json"
	"reflect"
	"strings"
	"testing"
)

func boolp(value bool) *bool    { return &value }
func int64p(value int64) *int64 { return &value }

func evidence(reason string) Evidence {
	return Evidence{Source: "typed_metric", Provenance: "run-7", Authority: "local_verifier", Freshness: "same_run", Reason: reason}
}

func healthyInput() Input {
	return Input{
		DeterministicFloor:  []DeterministicCheck{{Name: "tests", Passed: boolp(true), Evidence: evidence("tests passed")}},
		ObjectiveProgress:   Observation{State: Pass, Evidence: evidence("objective advanced")},
		Efficiency:          EfficiencyInput{Outcome: boolp(true), ConstraintsSatisfied: boolp(true), ParentUnits: int64p(10), ChildUnits: []int64{3, 2}, AccountingComplete: true, Evidence: evidence("efficient")},
		DelegationIntegrity: Observation{State: Pass, Evidence: evidence("delegation verified")},
		SemanticReview:      Observation{State: Pass, Evidence: evidence("review passed")},
	}
}

func TestAssessIssueCases(t *testing.T) {
	tests := []struct {
		name               string
		mutate             func(*Input)
		want               State
		missing            []string
		conflicts          int
		wantRecommendation string
	}{
		{name: "all layers pass", want: Pass, wantRecommendation: RecommendationContinue},
		{name: "deterministic failure dominates semantic pass", mutate: func(in *Input) { in.DeterministicFloor[0].Passed = boolp(false) }, want: Fail, conflicts: 1, wantRecommendation: RecommendationOperatorReview},
		{name: "missing evidence is unknown", mutate: func(in *Input) { in.ObjectiveProgress.Evidence.Source = "" }, want: Unknown, missing: []string{"objective_progress"}, wantRecommendation: RecommendationDeepenAudit},
		{name: "semantic warning is warning", mutate: func(in *Input) { in.SemanticReview.State = Warn; in.SemanticReview.Evidence.Reason = "review concern" }, want: Warn, wantRecommendation: RecommendationDeepenAudit},
		{name: "efficiency requires outcome and constraints", mutate: func(in *Input) { in.Efficiency.Outcome = nil; in.Efficiency.ConstraintsSatisfied = nil }, want: Unknown, missing: []string{"efficiency_with_quality"}, wantRecommendation: RecommendationDeepenAudit},
		{name: "efficiency requires total parent child accounting", mutate: func(in *Input) { in.Efficiency.AccountingComplete = false }, want: Unknown, missing: []string{"efficiency_with_quality"}, wantRecommendation: RecommendationDeepenAudit},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			in := healthyInput()
			if test.mutate != nil {
				test.mutate(&in)
			}
			got := Assess(in)
			if got.State != test.want {
				t.Fatalf("state = %s, want %s", got.State, test.want)
			}
			if !got.Shadow {
				t.Fatal("shadow = false")
			}
			if got.SchemaVersion != SchemaVersion {
				t.Fatalf("schema = %q", got.SchemaVersion)
			}
			for _, layer := range got.Layers {
				if strings.TrimSpace(layer.ReasonToken) == "" {
					t.Fatalf("layer %s has empty reason token", layer.Name)
				}
			}
			if !reflect.DeepEqual(got.MissingEvidence, test.missing) {
				t.Fatalf("missing = %#v, want %#v", got.MissingEvidence, test.missing)
			}
			if len(got.Conflicts) != test.conflicts {
				t.Fatalf("conflicts = %d, want %d", len(got.Conflicts), test.conflicts)
			}
			if got.Recommendation != test.wantRecommendation {
				t.Fatalf("recommendation = %q, want %q", got.Recommendation, test.wantRecommendation)
			}
			if strings.TrimSpace(got.Recommendation) == "" || strings.Contains(got.Recommendation, "\n") {
				t.Fatalf("recommendation = %q", got.Recommendation)
			}
		})
	}
}

func TestDeterministicOrderingAndPrivacySafeSerialization(t *testing.T) {
	in := healthyInput()
	in.ObjectiveID = "objective-7"
	in.TrajectoryID = "trajectory-9"
	in.ObservationWindow = "turns:1-12"
	in.DeterministicFloor = []DeterministicCheck{
		{Name: "zeta", Passed: boolp(true), Evidence: evidence("z")},
		{Name: "alpha", Passed: boolp(true), Evidence: evidence("a")},
	}
	first, err := Marshal(Assess(in))
	if err != nil {
		t.Fatal(err)
	}
	second, err := Marshal(Assess(in))
	if err != nil {
		t.Fatal(err)
	}
	if string(first) != string(second) {
		t.Fatalf("serialization differs:\n%s\n%s", first, second)
	}
	for _, forbidden := range []string{"prompt", "tool_payload", "tool_input", "raw_payload", "callback", "stop", "kill", "mutate"} {
		if strings.Contains(string(first), `"`+forbidden+`"`) {
			t.Fatalf("privacy/authority field %q serialized: %s", forbidden, first)
		}
	}
	var decoded map[string]any
	if err := json.Unmarshal(first, &decoded); err != nil {
		t.Fatal(err)
	}
	if len(decoded) != 10 { //boundarylint:ignore CHANGE_DETECTOR_TEST the serialized assurance report schema contains exactly ten top-level fields
		t.Fatalf("top-level field count = %d: %v", len(decoded), decoded)
	}
}

func TestAssessHasNoActionCapability(t *testing.T) {
	for _, typ := range []reflect.Type{reflect.TypeOf(Input{}), reflect.TypeOf(Receipt{})} {
		for i := 0; i < typ.NumField(); i++ {
			field := typ.Field(i)
			if field.Type.Kind() == reflect.Func {
				t.Fatalf("%s exposes callback %s", typ, field.Name)
			}
			name := strings.ToLower(field.Name)
			if strings.Contains(name, "stop") || strings.Contains(name, "kill") || strings.Contains(name, "mutate") {
				t.Fatalf("%s exposes action field %s", typ, field.Name)
			}
		}
	}
	in := healthyInput()
	before, err := json.Marshal(in)
	if err != nil {
		t.Fatal(err)
	}
	_ = Assess(in)
	after, err := json.Marshal(in)
	if err != nil {
		t.Fatal(err)
	}
	if string(before) != string(after) {
		t.Fatalf("Assess mutated input:\nbefore %s\nafter  %s", before, after)
	}
}

func TestMarshalRejectsNonShadowReceipt(t *testing.T) {
	receipt := Assess(healthyInput())
	receipt.Shadow = false
	if _, err := Marshal(receipt); err == nil {
		t.Fatal("Marshal accepted shadow=false")
	}
}
