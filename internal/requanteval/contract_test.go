package requanteval

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"testing"
)

type witnessFixture struct {
	Request Request `json:"request"`
	Result  Result  `json:"result"`
}

func artifactDigest(target []float64, h [][]float64) string {
	b, _ := json.Marshal(struct {
		Target  []float64   `json:"target"`
		Hessian [][]float64 `json:"hessian"`
	}{target, h})
	sum := sha256.Sum256(b)
	return hex.EncodeToString(sum[:])
}
func fixture(seed uint64, target []float64, h [][]float64) Request {
	return Request{ContractVersion: ContractVersion, RecipeVersion: RecipeVersion, RuntimeVersion: RuntimeVersion, Seed: seed, Grid: []float64{-1, 0, 1}, InitialCodes: []int{0, 1}, Target: target, Hessian: h, MaxSweeps: 8, Provenance: Provenance{ArtifactID: "synthetic-coupled-quadratic", ArtifactVersion: "v1", ArtifactSHA256: artifactDigest(target, h), Initializer: "round-to-nearest", InitializerVersion: "fixture-v1", Source: "internal/requanteval/testdata"}, Quality: QualityProbe{ID: "synthetic-linear-probe", Version: "v1", Inputs: [][]float64{{1, 0}, {0, 1}, {1, 1}}, Expected: []float64{-.8, -.4, -1.2}}}
}
func TestEvaluateWitnessFixtures(t *testing.T) {
	paths, err := filepath.Glob("testdata/*.json")
	if err != nil || len(paths) < 3 {
		t.Fatalf("need >=3 independently readable fixtures: %v, %d", err, len(paths))
	}
	for _, path := range paths {
		t.Run(filepath.Base(path), func(t *testing.T) {
			raw, err := os.ReadFile(path)
			if err != nil {
				t.Fatal(err)
			}
			var f witnessFixture
			if err := json.Unmarshal(raw, &f); err != nil {
				t.Fatal(err)
			}
			got := Evaluate(f.Request)
			if !reflect.DeepEqual(got, f.Result) {
				g, _ := json.MarshalIndent(got, "", "  ")
				t.Fatalf("witness mismatch\n got: %s\nwant: %s", g, raw)
			}
		})
	}
}
func TestTypedUnsupportedOutcomes(t *testing.T) {
	tests := []struct {
		name    string
		mutate  func(*Request)
		outcome Outcome
		reason  ReasonCode
	}{{"contract", func(r *Request) { r.ContractVersion = "requanteval/v99" }, OutcomeUnsupported, ReasonUnknownContract}, {"recipe", func(r *Request) { r.RecipeVersion = "unknown" }, OutcomeUnsupported, ReasonUnknownRecipe}, {"grid", func(r *Request) { r.Grid = []float64{0, 0} }, OutcomeUnsupported, ReasonInvalidGrid}, {"shape", func(r *Request) { r.Target = nil }, OutcomeUnsupported, ReasonInvalidShape}, {"provenance", func(r *Request) { r.Provenance.ArtifactSHA256 = "" }, OutcomeUnsupported, ReasonInvalidProvenance}}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			r := fixture(1, []float64{-.8, -.4}, [][]float64{{1, -.8}, {-.8, 1}})
			tc.mutate(&r)
			got := Evaluate(r)
			if got.Outcome != tc.outcome || got.Reason != tc.reason {
				t.Fatalf("got %s/%s", got.Outcome, got.Reason)
			}
		})
	}
}
func TestSameInitializationAndFixedGrid(t *testing.T) {
	req := fixture(6253, []float64{-.8, -.4}, [][]float64{{1, -.8}, {-.8, 1}})
	got := Evaluate(req)
	if got.Outcome != OutcomeSupported || got.Metrics.FinalMSE >= got.Metrics.InitialMSE {
		t.Fatalf("expected measured fixture improvement: %+v", got)
	}
	if !reflect.DeepEqual(got.InitialCodes, req.InitialCodes) {
		t.Fatal("initialization was not preserved in witness")
	}
	for _, c := range got.RefinedCodes {
		if c < 0 || c >= len(req.Grid) {
			t.Fatal("refinement left fixed grid")
		}
	}
	if got.Quality == nil || got.Quality.FinalPredictionMSE <= got.Quality.InitialPredictionMSE {
		t.Fatal("witness must preserve the modeled reconstruction/quality tradeoff")
	}
	if got.Evidence != EvidenceModeled || got.ClaimCheck.Verdict != "not-yet" {
		t.Fatal("fixture must not be promoted to observed model/hardware evidence")
	}
}
