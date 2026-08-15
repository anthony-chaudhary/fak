package harnesscrossover

import "testing"

func TestEvaluatePublishesNegativeWithoutMetricChange(t *testing.T) {
	runs := []Run{
		{TaskID: "c", Explanation: Seconds{Provenance: "modeled"}, Provenance: "modeled"},
		{TaskID: "l", Explanation: Seconds{Provenance: "modeled"}, Provenance: "modeled"},
		{TaskID: "i", Explanation: Seconds{Provenance: "modeled"}, Provenance: "modeled"},
	}
	s := Study{
		Schema: Schema, ID: "negative",
		Tasks:   []Task{{ID: "c", Domain: "coding"}, {ID: "l", Domain: "legal"}, {ID: "i", Domain: "integrated"}},
		Weights: Weights{SwitchActionSeconds: 10},
		Alternatives: []Alternative{
			{ID: "native", Kind: "native-profile", Documentation: []Source{{URL: "https://example.test"}}, Setup: Seconds{Value: 1, Provenance: "modeled"}, Maintenance: Seconds{Value: 1, Provenance: "modeled"}, Runs: runs},
			{ID: "fak", Kind: "contextual-harness", Documentation: []Source{{URL: "https://example.test"}}, Setup: Seconds{Value: 20, Provenance: "modeled"}, Maintenance: Seconds{Value: 5, Provenance: "modeled"}, Runs: runs},
		},
	}
	r := Evaluate(s)
	if r.Winner != "native" || r.Verdict != "native-profile wins under declared costs" {
		t.Fatalf("%+v", r)
	}
}

func TestParseRequiresAllDomainsAndProvenance(t *testing.T) {
	_, err := Parse([]byte(`{"schema":"fak.harness-crossover-study/v1alpha1","id":"x","tasks":[{"id":"a","domain":"coding"},{"id":"b","domain":"legal"}],"alternatives":[{},{}]}`))
	if err == nil {
		t.Fatal("invalid study accepted")
	}
}

func TestCrossoverCondition(t *testing.T) {
	runsNative := []Run{
		{TaskID: "c", SwitchActions: 0, Explanation: Seconds{Provenance: "modeled"}, Provenance: "modeled"},
		{TaskID: "l", SwitchActions: 1, Explanation: Seconds{Provenance: "modeled"}, Provenance: "modeled"},
		{TaskID: "i", SwitchActions: 1, Explanation: Seconds{Provenance: "modeled"}, Provenance: "modeled"},
	}
	runsContext := []Run{
		{TaskID: "c", Explanation: Seconds{Provenance: "modeled"}, Provenance: "modeled"},
		{TaskID: "l", Explanation: Seconds{Provenance: "modeled"}, Provenance: "modeled"},
		{TaskID: "i", Explanation: Seconds{Provenance: "modeled"}, Provenance: "modeled"},
	}
	s := Study{Schema: Schema, ID: "cross", Tasks: []Task{{ID: "c", Domain: "coding"}, {ID: "l", Domain: "legal"}, {ID: "i", Domain: "integrated"}}, Weights: Weights{SwitchActionSeconds: 10}, Alternatives: []Alternative{
		{ID: "native", Kind: "native-profile", Documentation: []Source{{URL: "https://example.test"}}, Setup: Seconds{Provenance: "modeled"}, Maintenance: Seconds{Provenance: "modeled"}, Runs: runsNative},
		{ID: "fak", Kind: "contextual-harness", Documentation: []Source{{URL: "https://example.test"}}, Setup: Seconds{Value: 100, Provenance: "modeled"}, Maintenance: Seconds{Provenance: "modeled"}, Runs: runsContext},
	}}
	r := Evaluate(s)
	if r.Crossover == nil || r.Crossover.BreakEvenSwitches == nil || *r.Crossover.BreakEvenSwitches != 10 {
		t.Fatalf("unexpected crossover: %+v", r.Crossover)
	}
}
