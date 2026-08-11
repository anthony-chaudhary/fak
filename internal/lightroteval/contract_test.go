package lightroteval

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"testing"
)

type fixtureFile struct {
	Request Request `json:"request"`
	Result  Result  `json:"result"`
}

func fixture(samples [][]float64) Request {
	b, _ := json.Marshal(samples)
	s := sha256.Sum256(b)
	return Request{ContractVersion: ContractVersion, Bits: 4, Evidence: EvidenceModeled, Samples: samples, Provenance: Provenance{Artifact: ArtifactProvenance{ID: "lightrot-bounded-matrix", Version: "v1", SHA256: hex.EncodeToString(s[:]), Source: "internal/lightroteval/testdata"}, Model: ModelProvenance{ID: "synthetic-hidden-state", Revision: "fixture-v1", SHA256: "1111111111111111111111111111111111111111111111111111111111111111", License: "CC0-1.0"}, Recipe: RecipeProvenance{ID: "lightrot", Version: RecipeVersion, PaperID: PaperID, PaperSHA256: PaperSHA256, Seed: 6250, BlockSize: 4}, Runtime: RuntimeProvenance{ID: "fak/lightroteval", Version: RuntimeVersion, Backend: "cpu-reference-f64"}, Hardware: RuntimeEnvelope()}}
}
func TestWitnessFixtures(t *testing.T) {
	paths, e := filepath.Glob("testdata/*.json")
	if e != nil || len(paths) < 3 {
		t.Fatalf("need >=3 fixtures: %v %d", e, len(paths))
	}
	for _, p := range paths {
		t.Run(filepath.Base(p), func(t *testing.T) {
			raw, e := os.ReadFile(p)
			if e != nil {
				t.Fatal(e)
			}
			var f fixtureFile
			if e = json.Unmarshal(raw, &f); e != nil {
				t.Fatal(e)
			}
			got := Evaluate(f.Request)
			if !reflect.DeepEqual(got, f.Result) {
				g, _ := json.MarshalIndent(got, "", "  ")
				t.Fatalf("witness mismatch\ngot %s\nwant %s", g, raw)
			}
		})
	}
}
func TestSupportedCarriesFullEvaluation(t *testing.T) {
	r := fixture([][]float64{{8, .2, -.1, .1}, {7.5, .1, -.2, .2}, {-7.8, -.2, .1, -.1}, {.2, .1, -.2, 7}})
	got := Evaluate(r)
	if got.Outcome != OutcomeSupported || got.Reason != ReasonEvaluated {
		t.Fatalf("%+v", got)
	}
	if !reflect.DeepEqual(CandidateIDs(got), []string{"lightrot", "no_rotation", "tuned_rotation"}) {
		t.Fatalf("missing bounded baselines: %+v", got.Candidates)
	}
	for _, c := range got.Candidates {
		if c.Metrics.ReconstructionAccuracy == 0 || c.Cost.RuntimeScalarOps == 0 || c.Cost.WallEvidence != EvidenceModeled {
			t.Fatalf("incomplete candidate: %+v", c)
		}
	}
	if got.Evidence != EvidenceModeled || got.ClaimCheck.Verdict != "not-yet" {
		t.Fatal("modeled fixture promoted to observed claim")
	}
}
func TestTypedUnsupportedAndDelegate(t *testing.T) {
	base := fixture([][]float64{{1, 2, 3, 4}, {4, 3, 2, 1}})
	tests := []struct {
		name   string
		mut    func(*Request)
		out    Outcome
		reason ReasonCode
	}{{"contract", func(r *Request) { r.ContractVersion = "lightroteval/v99" }, OutcomeUnsupported, ReasonUnknownContract}, {"recipe", func(r *Request) { r.Provenance.Recipe.Version = "unknown" }, OutcomeUnsupported, ReasonUnknownRecipe}, {"runtime", func(r *Request) { r.Provenance.Runtime.Backend = "cuda" }, OutcomeDelegate, ReasonUnknownRuntime}, {"bits", func(r *Request) { r.Bits = 1 }, OutcomeUnsupported, ReasonInvalidInput}, {"shape", func(r *Request) { r.Samples[0] = r.Samples[0][:3] }, OutcomeUnsupported, ReasonInvalidInput}, {"digest", func(r *Request) { r.Provenance.Artifact.SHA256 = "" }, OutcomeUnsupported, ReasonInvalidProvenance}, {"observed", func(r *Request) { r.Evidence = EvidenceObserved }, OutcomeDelegate, ReasonUnknownRuntime}}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			r := base
			r.Samples = clone(base.Samples)
			tc.mut(&r)
			got := Evaluate(r)
			if got.Outcome != tc.out || got.Reason != tc.reason {
				t.Fatalf("got %s/%s", got.Outcome, got.Reason)
			}
		})
	}
}
func clone(x [][]float64) [][]float64 {
	o := make([][]float64, len(x))
	for i := range x {
		o[i] = append([]float64(nil), x[i]...)
	}
	return o
}
