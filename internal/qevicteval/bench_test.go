package qevicteval

import (
	"testing"
)

var benchEvalSink Result

func TestQEvictEvalBenchmark(t *testing.T) {
	testing.Benchmark(BenchmarkQEvictEval)
}

func BenchmarkQEvictEval(b *testing.B) {
	trace := []WindowEvent{
		{
			Step:                0,
			WindowID:            "recent",
			FullBytes:           100,
			QuantizedBytes:      25,
			FutureAttentionMass: 0.02,
			OrdinaryEvicted:     false,
			QEvictTier:          "full",
			Reactivated:         false,
		},
		{
			Step:                1,
			WindowID:            "drifting-history",
			FullBytes:           100,
			QuantizedBytes:      25,
			FutureAttentionMass: 0.40,
			OrdinaryEvicted:     true,
			QEvictTier:          "recoverable",
			Reactivated:         true,
		},
		{
			Step:                2,
			WindowID:            "cold-history",
			FullBytes:           100,
			QuantizedBytes:      25,
			FutureAttentionMass: 0.01,
			OrdinaryEvicted:     true,
			QEvictTier:          "deleted",
			Reactivated:         false,
		},
	}

	req := Request{
		ContractVersion: ContractVersion,
		Provenance: Provenance{
			ArtifactID:      "qevict-benchmark-trace",
			ArtifactVersion: "bench-v1",
			ArtifactSHA256:  TraceDigest(trace),
			RecipeID:        RecipeID,
			RecipeRevision:  "v1",
			RecipeSource:    "https://arxiv.org/abs/2608.05326v1",
			RuntimeID:       RuntimeID,
			RuntimeVersion:  "go1.26",
		},
		Trace: trace,
		Runtime: &RuntimeObservation{
			Evidence:              EvidenceObserved,
			Platform:              "linux/amd64",
			Device:                "cpu",
			Command:               "go test -bench .",
			CapturedAt:            "2026-09-04T00:00:00Z",
			RuntimeArtifactSHA256: "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
			OrdinaryLatencyNS:     3.0,
			QEvictLatencyNS:       4.0,
			RecoveryLatencyNS:     0.5,
		},
	}

	b.ReportAllocs()
	b.ResetTimer()
	for n := 0; n < b.N; n++ {
		res := Evaluate(req)
		benchEvalSink = res
	}
}
