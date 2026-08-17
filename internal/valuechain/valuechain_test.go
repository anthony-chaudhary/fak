package valuechain

import "testing"

func f(v float64) *float64 { return &v }
func TestAuditSupportSharedContextEconomics(t *testing.T) {
	m := Manifest{Schema: Schema, Name: "support", Stages: []Stage{{ID: "gpu", Kind: "hardware"}, {ID: "serve", Kind: "serving", DependsOn: []string{"gpu"}}, {ID: "shared_context", Kind: "context", DependsOn: []string{"serve"}}, {ID: "ticket", Kind: "outcome", DependsOn: []string{"shared_context"}}}, Arms: []Arm{{ID: "baseline", Default: true}, {ID: "shared"}}, Outcomes: []Outcome{{ID: "ticket_resolved", Unit: "ticket"}}}
	in := Input{Schema: Schema, Observations: []Observation{
		{ID: "b1", TraceID: "tb", SessionID: "sb", PairID: "p1", StageID: "ticket", Arm: "baseline", Turns: 10, CostUSD: f(2), Outcomes: map[string]float64{"ticket_resolved": 1}, Provenance: "provider"},
		{ID: "setup", TraceID: "ts", StageID: "shared_context", Arm: "shared", CostUSD: f(.2), CostKey: "shared-setup", Provenance: "meter"},
		{ID: "s1", TraceID: "ts", SessionID: "s1", PairID: "p1", StageID: "ticket", Arm: "shared", Turns: 5, CostUSD: f(.5), Outcomes: map[string]float64{"ticket_resolved": 1}, Provenance: "provider"},
		{ID: "s2", TraceID: "ts", SessionID: "s2", StageID: "ticket", Arm: "shared", Turns: 5, CostUSD: f(.5), Outcomes: map[string]float64{"ticket_resolved": 1}, Provenance: "provider"},
	}}
	rep, err := Audit(m, in)
	if err != nil {
		t.Fatal(err)
	}
	shared := armByID(rep.Arms, "shared")
	if shared == nil || shared.CostUSD == nil || *shared.CostUSD != 1.2 {
		t.Fatalf("shared cost = %#v", shared)
	}
	if shared.Sessions != 2 || shared.CostPerOutcome["ticket_resolved"] != .6 {
		t.Fatalf("shared report = %#v", shared)
	}
	if rep.Comparison == nil || rep.Comparison.Design != "paired" {
		t.Fatalf("comparison = %#v", rep.Comparison)
	}
}
func TestAuditRejectsCycleAndAmbiguousSharedCost(t *testing.T) {
	m := Manifest{Schema: Schema, Name: "x", Stages: []Stage{{ID: "a", Kind: "x", DependsOn: []string{"b"}}, {ID: "b", Kind: "x", DependsOn: []string{"a"}}}, Arms: []Arm{{ID: "x"}}}
	if _, err := Audit(m, Input{Schema: Schema}); err == nil {
		t.Fatal("wanted cycle error")
	}
	m.Stages = []Stage{{ID: "a", Kind: "x"}}
	m.Arms = []Arm{{ID: "x"}, {ID: "y"}}
	in := Input{Schema: Schema, Observations: []Observation{{ID: "1", TraceID: "1", StageID: "a", Arm: "x", CostUSD: f(1), CostKey: "same", Provenance: "p"}, {ID: "2", TraceID: "2", StageID: "a", Arm: "y", CostUSD: f(1), CostKey: "same", Provenance: "p"}}}
	if _, err := Audit(m, in); err == nil {
		t.Fatal("wanted ambiguous cost error")
	}
}
func TestUnknownCostStaysAbsent(t *testing.T) {
	m := Manifest{Schema: Schema, Name: "x", Stages: []Stage{{ID: "a", Kind: "benchmark"}}, Arms: []Arm{{ID: "latest"}}, Outcomes: []Outcome{{ID: "passed", Unit: "case"}}}
	rep, err := Audit(m, Input{Schema: Schema, Observations: []Observation{{ID: "bench", TraceID: "packet", StageID: "a", Arm: "latest", Turns: 1, Outcomes: map[string]float64{"passed": 1}, Provenance: "agenticbench-result-packet"}}})
	if err != nil {
		t.Fatal(err)
	}
	if rep.Arms[0].CostUSD != nil || rep.Arms[0].CostPerTurn != nil {
		t.Fatalf("unknown cost became known: %#v", rep.Arms[0])
	}
}
