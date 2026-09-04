package quantcompat

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/quantmeta"
)

// BenchmarkQuantCompat benchmarks operating envelope adjudication across supported,
// delegated, and conversion-required request configurations in a loop.
func BenchmarkQuantCompat(b *testing.B) {
	q4 := descriptor(quantmeta.Format("gguf"), "groupwise")
	reqs := []Request{
		{
			Artifact: q4,
			Runtime: Runtime{
				ID:       "llama.cpp",
				Formats:  []quantmeta.Format{quantmeta.Format("gguf")},
				Methods:  []string{"groupwise"},
				Hardware: []string{"cpu"},
			},
			Hardware: "cpu",
		},
		{
			Artifact: q4,
			Runtime: Runtime{
				ID:              "gateway",
				Hardware:        []string{"cpu"},
				ExternalFormats: []quantmeta.Format{quantmeta.Format("gguf")},
			},
			Hardware: "cpu",
		},
		{
			Artifact: q4,
			Runtime: Runtime{
				ID:                 "cuda-engine",
				Hardware:           []string{"cuda-sm90"},
				ConvertibleFormats: []quantmeta.Format{quantmeta.Format("gguf")},
			},
			Hardware: "cuda-sm90",
		},
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		req := reqs[i%len(reqs)]
		res := Adjudicate(req)
		if res.Status == StatusRejected {
			b.Fatalf("adjudication unexpectedly rejected: %+v", res)
		}
	}
}

// TestBenchmarkOperatingEnvelope exercises operating envelope adjudication in a test loop
// to ensure benchmark requests evaluate correctly and verify test file maturity.
func TestBenchmarkOperatingEnvelope(t *testing.T) {
	q4 := descriptor(quantmeta.Format("gguf"), "groupwise")
	req := Request{
		Artifact: q4,
		Runtime: Runtime{
			ID:       "llama.cpp",
			Formats:  []quantmeta.Format{quantmeta.Format("gguf")},
			Methods:  []string{"groupwise"},
			Hardware: []string{"cpu"},
		},
		Hardware: "cpu",
	}

	res := Adjudicate(req)
	if res.Status != StatusDirect || res.Reason != ReasonCompatible {
		t.Fatalf("unexpected adjudication result: %+v", res)
	}
}
