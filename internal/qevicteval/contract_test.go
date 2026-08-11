package qevicteval

import (
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

func TestNamedResearchWitness(t *testing.T) {
	paths, err := filepath.Glob("testdata/*.json")
	if err != nil || len(paths) < 3 {
		t.Fatalf("need >=3 fixtures: %v (%d)", err, len(paths))
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
			if got := Evaluate(f.Request); !reflect.DeepEqual(got, f.Result) {
				g, _ := json.MarshalIndent(got, "", "  ")
				t.Fatalf("witness mismatch\ngot %s\nfixture %s", g, raw)
			}
		})
	}
}

func TestTypedUnsupportedAndDelegate(t *testing.T) {
	raw, _ := os.ReadFile("testdata/recovery-observed.json")
	var f witnessFixture
	if err := json.Unmarshal(raw, &f); err != nil {
		t.Fatal(err)
	}
	tests := []struct {
		name    string
		mutate  func(*Request)
		outcome Outcome
		reason  ReasonCode
	}{
		{"contract", func(r *Request) { r.ContractVersion = "qevicteval/v99" }, OutcomeUnsupported, ReasonUnknownContract},
		{"recipe", func(r *Request) { r.Provenance.RecipeRevision = "v2" }, OutcomeUnsupported, ReasonUnknownRecipe},
		{"artifact", func(r *Request) { r.Provenance.ArtifactSHA256 = "sha256:invented" }, OutcomeUnsupported, ReasonInvalidArtifact},
		{"trace", func(r *Request) {
			r.Trace[0].QEvictTier = "mystery"
			r.Provenance.ArtifactSHA256 = TraceDigest(r.Trace)
		}, OutcomeUnsupported, ReasonInvalidTrace},
		{"runtime", func(r *Request) { r.Provenance.RuntimeID = "external/runtime" }, OutcomeDelegate, ReasonUnknownRuntime},
		{"observation", func(r *Request) { r.Runtime = nil }, OutcomeDelegate, ReasonRuntimeEvidenceRequired},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			r := f.Request
			r.Trace = append([]WindowEvent(nil), f.Request.Trace...)
			tc.mutate(&r)
			got := Evaluate(r)
			if got.Outcome != tc.outcome || got.Reason != tc.reason || got.Decision != DecisionAbstain {
				t.Fatalf("got %+v", got)
			}
		})
	}
}

func TestModeledAndObservedRemainDistinct(t *testing.T) {
	raw, _ := os.ReadFile("testdata/recovery-observed.json")
	var f witnessFixture
	_ = json.Unmarshal(raw, &f)
	got := Evaluate(f.Request)
	if got.Evidence.CapacityAndDrift != EvidenceModeled || got.Evidence.Latency != EvidenceObserved {
		t.Fatalf("evidence conflated: %+v", got.Evidence)
	}
	if got.Metrics.RecoveryEvents != 1 || got.Metrics.RecoveryReadBytes != 25 || got.Metrics.LatencyOverheadNS != 1.047 {
		t.Fatalf("named witness lost: %+v", got.Metrics)
	}
}

var benchSink float64

func BenchmarkOrdinaryEviction(b *testing.B) {
	trace := []WindowEvent{{Step: 0, WindowID: "recent", FullBytes: 100, QuantizedBytes: 25, FutureAttentionMass: .02}, {Step: 1, WindowID: "drifting-history", FullBytes: 100, QuantizedBytes: 25, FutureAttentionMass: .40, OrdinaryEvicted: true}, {Step: 2, WindowID: "cold-history", FullBytes: 100, QuantizedBytes: 25, FutureAttentionMass: .01, OrdinaryEvicted: true}}
	b.ResetTimer()
	for n := 0; n < b.N; n++ {
		var missed float64
		for _, e := range trace {
			if e.OrdinaryEvicted {
				missed += e.FutureAttentionMass
			}
		}
		benchSink = missed
	}
}
func BenchmarkQEvictReplay(b *testing.B) {
	trace := []WindowEvent{{Step: 0, WindowID: "recent", FullBytes: 100, QuantizedBytes: 25, FutureAttentionMass: .02, QEvictTier: "full"}, {Step: 1, WindowID: "drifting-history", FullBytes: 100, QuantizedBytes: 25, FutureAttentionMass: .40, QEvictTier: "recoverable", Reactivated: true}, {Step: 2, WindowID: "cold-history", FullBytes: 100, QuantizedBytes: 25, FutureAttentionMass: .01, QEvictTier: "deleted"}}
	b.ResetTimer()
	for n := 0; n < b.N; n++ {
		var missed float64
		for _, e := range trace {
			if e.QEvictTier == "deleted" {
				missed += e.FutureAttentionMass
			}
			if e.Reactivated {
				benchSink += float64(e.QuantizedBytes)
			}
		}
		benchSink = missed
	}
}
func BenchmarkRecoveryRead(b *testing.B) {
	src := make([]byte, 25)
	dst := make([]byte, 100)
	b.ResetTimer()
	for n := 0; n < b.N; n++ {
		copy(dst, src)
		benchSink = float64(dst[0])
	}
}
