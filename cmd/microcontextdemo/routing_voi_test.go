package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

func TestRoutingVOIHasCrossoverAndBoundedRegret(t *testing.T) {
	p := filepath.Join(t.TempDir(), "routing.json")
	if err := runRoutingVOI(p, 6105, 24, 200); err != nil {
		t.Fatal(err)
	}
	if err := verifyRoutingVOI(p); err != nil {
		t.Fatal(err)
	}
	b, _ := os.ReadFile(p)
	var r routingVOIReport
	if err := json.Unmarshal(b, &r); err != nil {
		t.Fatal(err)
	}
	for _, m := range r.Mixtures {
		if m.AdaptiveRegret > 45 {
			t.Fatalf("%s regret too large: %f", m.Name, m.AdaptiveRegret)
		}
	}
}

func TestVerifyRoutingVOIRejectsNoCrossover(t *testing.T) {
	r := routingVOIReport{Schema: routingVOISchema, Trials: 2, RecordsTrial: 20, Limits: []string{"a", "b", "c"}}
	for i := 0; i < 5; i++ {
		m := routingMixtureRun{Name: "x", Winner: "adaptive"}
		for _, p := range []string{"always-model", "always-filter-tool", "adaptive", "oracle"} {
			m.Policies = append(m.Policies, routingPolicyStat{Policy: p, Quality: 1, MeanLatencyMS: 1, MeanCostUnits: 1})
		}
		r.Mixtures = append(r.Mixtures, m)
	}
	b, _ := json.Marshal(r)
	p := filepath.Join(t.TempDir(), "bad.json")
	_ = os.WriteFile(p, b, 0644)
	if verifyRoutingVOI(p) == nil {
		t.Fatal("accepted experiment without crossover")
	}
}
