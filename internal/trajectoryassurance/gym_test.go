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

func TestGymCorpusV2Promote(t *testing.T) {
	c, raw, err := LoadGym("testdata/gym-corpus.v2.json")
	if err != nil {
		t.Fatal(err)
	}
	if len(c.PairedCases) != 32 {
		t.Fatalf("expected 32 paired cases, got %d", len(c.PairedCases))
	}
	if c.Provenance != "anonymized production empirical traces" {
		t.Fatalf("unexpected provenance: %s", c.Provenance)
	}
	r := EvaluateGym(c, raw)
	if r.Promotion.Verdict != "PROMOTE" {
		t.Fatalf("expected verdict PROMOTE, got %s (reasons: %v)", r.Promotion.Verdict, r.Promotion.Reasons)
	}
	if len(r.Promotion.Reasons) != 0 {
		t.Fatalf("expected 0 reasons, got %v", r.Promotion.Reasons)
	}
}
