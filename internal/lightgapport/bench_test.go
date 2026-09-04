package lightgapport

import (
	"path/filepath"
	"runtime"
	"testing"
)

func benchRepoRoot() string {
	_, f, _, _ := runtime.Caller(0)
	return filepath.Clean(filepath.Join(filepath.Dir(f), "..", ".."))
}

// BenchmarkLightGapPort benchmarks swap verification and witness checking in a loop.
func BenchmarkLightGapPort(b *testing.B) {
	root := benchRepoRoot()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		r, err := Load(root)
		if err != nil {
			b.Fatalf("Load failed: %v", err)
		}
		if len(r.Swaps) == 0 {
			b.Fatal("no swaps found in contract")
		}
	}
}

func TestBenchmarkLightGapPortHarness(t *testing.T) {
	root := repoRoot(t)
	r, err := Load(root)
	if err != nil {
		t.Fatalf("Load returned unexpected error: %v", err)
	}
	if len(r.Swaps) != 5 {
		t.Fatalf("expected 5 swaps, got %d", len(r.Swaps))
	}
}
