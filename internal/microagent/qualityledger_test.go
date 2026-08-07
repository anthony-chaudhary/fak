package microagent

import (
	"encoding/json"
	"errors"
	"os"
	"testing"
)

func TestQualityLedgerIngestsSynthetic10k(t *testing.T) {
	w := SourceRun{Schema: "fak-microcontext-spine/1", Verdict: "PASS", LogicalShards: 10000, PhysicalWorkers: 64, Completed: 10000, TurnCount: 10000, ElapsedMS: 1000, Mode: "synthetic"}
	b, _ := json.Marshal(w)
	l, err := IngestSourceRun(b, "synthetic-10k", "base", OutcomeCheckFunc(func(string) error { return nil }), 16)
	if err != nil {
		t.Fatal(err)
	}
	if l.Verification.Passed != 10000 || len(l.SampleIDs) != 16 || l.ClaimFamilies.UsefulWork.PerWallSecond != 10000 {
		t.Fatalf("ledger=%+v", l)
	}
}
func TestQualityLedgerIngestsModelWitness(t *testing.T) {
	b, err := os.ReadFile("../../experiments/microcontext/s6-groq-api-only-4-pass-2026-08-06.json")
	if err != nil {
		t.Fatal(err)
	}
	l, err := IngestSourceRun(b, "model-live", "base", OutcomeCheckFunc(func(id string) error {
		if id == "ctx-00000003" {
			return errors.New("fixture reject")
		}
		return nil
	}), 2)
	if err != nil {
		t.Fatal(err)
	}
	if l.Verification.Passed != 3 || l.Verification.Failed != 1 || l.ClaimFamilies.Inference.UsageResponses != 4 {
		t.Fatalf("ledger=%+v", l)
	}
}
func TestQualityLedgerCannotOmitFailures(t *testing.T) {
	l := QualityLedger{Schema: QualityLedgerSchema, RunID: "r", BaseID: "b", Submitted: 2, Retired: 2, Verification: VerificationSummary{Checked: 2, Passed: 2}}
	if err := VerifyQualityLedger(l); err == nil {
		t.Fatal("expected denominator refusal")
	}
}

func TestQualityLedgerExposesOutcomeCounters(t *testing.T) {
	w := SourceRun{Schema: "fak-microcontext-spine/1", LogicalShards: 3, PhysicalWorkers: 1, Completed: 2, Failed: 1, TurnCount: 2, ElapsedMS: 1000}
	b, _ := json.Marshal(w)
	l, err := IngestSourceRun(b, "run", "base", OutcomeCheckFunc(func(string) error { return nil }), 0)
	if err != nil {
		t.Fatal(err)
	}
	if l.Outcomes["success"] != 2 || l.Outcomes["error"] != 1 || l.Outcomes["refusal"] != 0 {
		t.Fatalf("outcomes=%v", l.Outcomes)
	}
	l.Outcomes["error"] = 0
	if err := VerifyQualityLedger(l); err == nil {
		t.Fatal("expected outcome reconciliation refusal")
	}
}
