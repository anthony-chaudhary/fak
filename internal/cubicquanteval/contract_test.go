package cubicquanteval

import (
	"encoding/json"
	"os"
	"testing"
)

func fixture(t *testing.T) []byte {
	t.Helper()
	b, err := os.ReadFile("testdata/evaluation-v1.json")
	if err != nil {
		t.Fatal(err)
	}
	return b
}

func TestPinnedFixtureProducesCompleteObservedLedger(t *testing.T) {
	got := Evaluate(Request{Scope: ScopeReconstruction, FixtureJSON: fixture(t)})
	if got.Outcome != Supported || got.Reason != ReasonEvaluated {
		t.Fatalf("%s %s: %s", got.Outcome, got.Reason, got.Detail)
	}
	if got.Evidence != "observed" || got.InputBasis != "modeled-synthetic" {
		t.Fatalf("evidence labels = %q/%q", got.Evidence, got.InputBasis)
	}
	if got.Artifact.ID != "arxiv:2608.06763v1" || got.Recipe.Seed != 42 || got.Runtime.Delegate != "cpu-go-stdlib" || got.Model.WeightsSHA256 != "none" {
		t.Fatalf("provenance not pinned: %+v", got)
	}
	if len(got.Rows) != 24 {
		t.Fatalf("rows=%d want 24", len(got.Rows))
	}
	seen := map[string]bool{}
	for _, row := range got.Rows {
		seen[row.Distribution] = true
		if row.Bits < 1 || row.Bits > 8 || row.Groups != 3 {
			t.Fatalf("bad row: %+v", row)
		}
		if row.CubicRMSE < 0 || row.TunedUniformRMSE < 0 || row.TunedNonUniformRMSE < 0 {
			t.Fatalf("negative RMSE: %+v", row)
		}
		if row.Decision != "integrate" && row.Decision != "abstain" {
			t.Fatalf("decision=%q", row.Decision)
		}
	}
	for _, d := range []string{"uniform", "gaussian", "laplace"} {
		if !seen[d] {
			t.Fatalf("missing %s", d)
		}
	}
}

func TestTypedBoundariesNeverFallBack(t *testing.T) {
	b := fixture(t)
	var f map[string]any
	if err := json.Unmarshal(b, &f); err != nil {
		t.Fatal(err)
	}
	f["schema_version"] = "fak.cubicquanteval.fixture/v2"
	unknown, _ := json.Marshal(f)
	cases := []struct {
		name    string
		req     Request
		outcome Outcome
		reason  ReasonCode
	}{
		{"unknown-version", Request{Scope: ScopeReconstruction, FixtureJSON: unknown}, Unsupported, ReasonUnknownSchema},
		{"unknown-scope", Request{Scope: "training", FixtureJSON: b}, Unsupported, ReasonCombinationRejected},
		{"model-quality", Request{Scope: ScopeModelQuality, FixtureJSON: b}, Delegate, ReasonQualityReroute},
		{"hardware", Request{Scope: ScopeHardwarePerformance, FixtureJSON: b}, Delegate, ReasonAcceleratorReroute},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := Evaluate(tc.req)
			if got.Outcome != tc.outcome || got.Reason != tc.reason {
				t.Fatalf("got %s/%s: %s", got.Outcome, got.Reason, got.Detail)
			}
		})
	}
}

func TestTamperedProvenanceIsUnsupported(t *testing.T) {
	var f map[string]any
	if err := json.Unmarshal(fixture(t), &f); err != nil {
		t.Fatal(err)
	}
	f["artifact"].(map[string]any)["sha256"] = "invented"
	b, _ := json.Marshal(f)
	got := Evaluate(Request{Scope: ScopeReconstruction, FixtureJSON: b})
	if got.Outcome != Unsupported || got.Reason != ReasonProvenanceMismatch {
		t.Fatalf("got %s/%s", got.Outcome, got.Reason)
	}
}

func TestDeterministicIndependentFixtureRead(t *testing.T) {
	b := fixture(t)
	a := Evaluate(Request{Scope: ScopeReconstruction, FixtureJSON: b})
	c := Evaluate(Request{Scope: ScopeReconstruction, FixtureJSON: b})
	aj, _ := json.Marshal(a.Rows)
	cj, _ := json.Marshal(c.Rows)
	if string(aj) != string(cj) {
		t.Fatal("ledger is not deterministic")
	}
}
