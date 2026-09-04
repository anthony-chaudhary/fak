package memorycotravel

import (
	"os"
	"path/filepath"
	"testing"
)

// BenchmarkMemoryCoTravel exercises the memory cotravel planning logic in a loop.
func BenchmarkMemoryCoTravel(b *testing.B) {
	tmp := b.TempDir()
	srcCfg := filepath.Join(tmp, "src")
	dstCfg := filepath.Join(tmp, "dst")
	slug := "bench-project"
	srcMem := filepath.Join(srcCfg, "projects", slug, "memory")
	dstMem := filepath.Join(dstCfg, "projects", slug, "memory")

	if err := os.MkdirAll(srcMem, 0o755); err != nil {
		b.Fatal(err)
	}
	if err := os.MkdirAll(dstMem, 0o755); err != nil {
		b.Fatal(err)
	}

	files := []string{"context.md", "scratch.md", "history.md"}
	for _, f := range files {
		if err := os.WriteFile(filepath.Join(srcMem, f), []byte("content: "+f), 0o644); err != nil {
			b.Fatal(err)
		}
	}
	if err := os.WriteFile(filepath.Join(dstMem, "scratch.md"), []byte("existing"), 0o644); err != nil {
		b.Fatal(err)
	}

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		plan := PlanOneDir(srcMem, dstMem, Additive)
		if len(plan) != len(files) {
			b.Fatalf("expected %d plan items, got %d", len(files), len(plan))
		}
	}
}

// TestBenchmarkMemoryCoTravel ensures the benchmark logic executes cleanly under unit test verification.
func TestBenchmarkMemoryCoTravel(t *testing.T) {
	res := testing.Benchmark(BenchmarkMemoryCoTravel)
	if res.N <= 0 {
		t.Fatalf("benchmark did not execute iterations: %+v", res)
	}
}
