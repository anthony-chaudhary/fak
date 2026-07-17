package bench

import (
	"encoding/json"
	"os"
	"testing"
)

func TestAdherence(t *testing.T) {
	data, err := os.ReadFile("testdata/negation_adherence_trace.json")
	if err != nil {
		t.Fatal(err)
	}
	var turns []AdherenceTurn
	if err := json.Unmarshal(data, &turns); err != nil {
		t.Fatal(err)
	}
	report, err := RunAdherence(turns)
	if err != nil {
		t.Fatal(err)
	}
	if report.Before.WorkloadHash != report.After.WorkloadHash {
		t.Fatal("arms replayed different workloads")
	}
	if !report.Before.Observed || !report.After.Observed || report.Before.Host == "" {
		t.Fatalf("missing OBSERVED host provenance: %+v", report)
	}
	if report.Before.OnlyAdherence != .5 || report.After.OnlyAdherence != 1 || report.Delta.OnlyAdherence != .5 {
		t.Fatalf("only scores: %+v", report)
	}
	if report.Before.NeverAdherence != .5 || report.After.NeverAdherence != 1 || report.Delta.NeverAdherence != .5 {
		t.Fatalf("never scores: %+v", report)
	}
	artifact, _ := json.MarshalIndent(report, "", "  ")
	t.Log(string(artifact))
}

func TestAdherenceConstraintScorers(t *testing.T) {
	cases := []struct {
		name     string
		c        Constraint
		calls    []string
		violated bool
	}{
		{"only follows", Constraint{Kind: ConstraintOnly, Tools: []string{"search"}}, []string{"search"}, false},
		{"only violation", Constraint{Kind: ConstraintOnly, Tools: []string{"search"}}, []string{"refund"}, true},
		{"never follows", Constraint{Kind: ConstraintNever, Tool: "refund"}, []string{"search"}, false},
		{"never violation", Constraint{Kind: ConstraintNever, Tool: "refund"}, []string{"refund"}, true},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, err := ConstraintViolated(tc.c, tc.calls)
			if err != nil || got != tc.violated {
				t.Fatalf("got (%v,%v), want %v", got, err, tc.violated)
			}
		})
	}
}
