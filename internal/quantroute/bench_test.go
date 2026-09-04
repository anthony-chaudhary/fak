package quantroute

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/quantcompat"
	"github.com/anthony-chaudhary/fak/internal/quantmeta"
)

func benchArtifact() quantcompat.Request {
	return quantcompat.Request{
		Artifact: quantmeta.Descriptor{
			Schema:     quantmeta.SchemaVersion,
			Artifact:   &quantmeta.ArtifactSpec{ContainerID: "gguf"},
			Weight:     &quantmeta.Weight{Format: quantmeta.Format("int4"), Bits: 4},
			Provenance: quantmeta.ProvenanceSpec{MethodID: "groupwise"},
		},
	}
}

func benchCandidates() []Candidate {
	return []Candidate{
		{
			Provider: "preferred",
			Runtime: quantcompat.Runtime{
				ID:       "preferred",
				Formats:  []quantmeta.Format{quantmeta.Format("safetensors")},
				Methods:  []string{"groupwise"},
				Hardware: []string{"cpu"},
			},
			Hardware: "cpu",
		},
		{
			Provider: "fallback",
			Runtime: quantcompat.Runtime{
				ID:       "fallback",
				Formats:  []quantmeta.Format{quantmeta.Format("gguf")},
				Methods:  []string{"groupwise"},
				Hardware: []string{"cpu"},
			},
			Hardware: "cpu",
		},
	}
}

func TestBenchmarkQuantRoute(t *testing.T) {
	req := benchArtifact()
	cands := benchCandidates()
	res := Select(req, cands)
	if res.Code != CodeSelected || res.Candidate == nil || res.Candidate.Provider != "fallback" {
		t.Fatalf("expected fallback selection, got %+v", res)
	}
}

func BenchmarkQuantRoute(b *testing.B) {
	req := benchArtifact()
	cands := benchCandidates()

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		res := Select(req, cands)
		if res.Code != CodeSelected {
			b.Fatalf("unexpected route outcome: %v", res.Code)
		}
	}
}
