package architest

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/archreport"
)

// BenchmarkArchitest exercises architecture checks and rule validations in a loop.
func BenchmarkArchitest(b *testing.B) {
	sampleReport := archreport.Report{
		Leaves: []archreport.Leaf{
			{
				Name: "primitive",
				ViolationEdges: []archreport.ViolationEdge{
					{From: "primitive", To: "composite"},
					{From: "primitive", To: "mechanism"},
				},
			},
			{
				Name: "composer",
				ViolationEdges: []archreport.ViolationEdge{
					{From: "composer", To: "integrator"},
				},
			},
			{
				Name: "integrator",
			},
		},
	}

	testEdges := []struct {
		fromTier int
		toTier   int
		wantUp   bool
	}{
		{fromTier: 1, toTier: 2, wantUp: true},
		{fromTier: 2, toTier: 1, wantUp: false},
		{fromTier: 3, toTier: 3, wantUp: false},
		{fromTier: 0, toTier: 1, wantUp: true},
		{fromTier: 5, toTier: 4, wantUp: false},
		{fromTier: 4, toTier: 5, wantUp: true},
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		// Exercise upward-import rule validation over synthetic test edges.
		for _, edge := range testEdges {
			if got := upwardImport(edge.fromTier, edge.toTier); got != edge.wantUp {
				b.Fatalf("upwardImport(%d, %d) = %v; want %v", edge.fromTier, edge.toTier, got, edge.wantUp)
			}
		}

		// Exercise architecture report violation edge extraction.
		edges := reportViolationEdges(sampleReport)
		if len(edges) != 3 {
			b.Fatalf("reportViolationEdges returned %d edges; want 3", len(edges))
		}

		// Exercise tier bounds rule validation over all declared packages.
		for pkg, t := range tier {
			if t < 0 || t > 5 {
				b.Fatalf("package %s has invalid tier %d", pkg, t)
			}
		}

		// Exercise hot-path invariant validation.
		for _, hp := range hotPath {
			if _, ok := tier[hp]; !ok {
				b.Fatalf("hot path package %s not declared in tier map", hp)
			}
		}
	}
}

// TestBenchmarkArchitest exercises BenchmarkArchitest under -run TestBenchmark.
func TestBenchmarkArchitest(t *testing.T) {
	res := testing.Benchmark(BenchmarkArchitest)
	if res.N <= 0 {
		t.Fatalf("expected benchmark iterations > 0, got %d", res.N)
	}
}
