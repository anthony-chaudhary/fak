package trajectoryassurance

import (
	"encoding/json"
	"os"
	"testing"
)

func TestGymCorpusAndReportWitness(t *testing.T) {
	c, raw, e := LoadGym("testdata/gym-corpus.v1.json")
	if e != nil {
		t.Fatal(e)
	}
	if len(c.PairedCases) < 30 {
		t.Fatal("paired corpus too small")
	}
	r := EvaluateGym(c, raw)
	if len(r.Strata) < 15 || r.Overall.Runs != len(c.PairedCases)*2*3*c.Trials {
		t.Fatalf("incomplete matrix: strata=%d runs=%d", len(r.Strata), r.Overall.Runs)
	}
	if r.Overall.Accounting.ParentInput == 0 || r.Overall.Accounting.ChildInput == 0 || r.Overall.PRAUC == 0 {
		t.Fatalf("missing metrics: %+v", r.Overall)
	}
	if _, ok := r.Overall.PassK["pass^5"]; !ok {
		t.Fatal("pass^k missing")
	}
	if r.WorstStratum.Key == "" || r.Promotion.Verdict != "NO_PROMOTION" {
		t.Fatalf("promotion evidence: %+v", r.Promotion)
	}
	b, e := json.MarshalIndent(r, "", "  ")
	if e != nil {
		t.Fatal(e)
	}
	if e = os.WriteFile("testdata/gym-report.golden.json", append(b, '\n'), 0644); e != nil {
		t.Fatal(e)
	}
}
func TestGymDeterministicFailureCannotBeOverridden(t *testing.T) {
	p := GymPair{ID: "authority", Mechanism: "baseline", Harness: "one-agent", ChildReadback: "reconciled", HiddenConstraint: "preserved"}
	if got := gymSimulate(p, "pressure", GymExpected{Receipt: GymFail}, "cheap-judge", 1).predicted; got != GymFail {
		t.Fatalf("judge overrode deterministic FAIL with %s", got)
	}
}
