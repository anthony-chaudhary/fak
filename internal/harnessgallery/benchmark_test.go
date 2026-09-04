package harnessgallery

import (
	"path/filepath"
	"testing"
)

var (
	benchBlueprintsSink []Blueprint
	benchBlueprintSink  Blueprint
	benchResultSink     InitResult
)

// BenchmarkHarnessGallery measures end-to-end catalog extraction, ID lookup,
// and structural constraint validation across all starter blueprints.
func BenchmarkHarnessGallery(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		list := Builtins()
		for _, item := range list {
			found, ok := Find(item.ID)
			if !ok {
				b.Fatalf("blueprint %q not found", item.ID)
			}
			benchBlueprintSink = found
		}
		if err := Validate(list); err != nil {
			b.Fatalf("validation failed: %v", err)
		}
		benchBlueprintsSink = list
	}
}

// BenchmarkBuiltins measures catalog isolation, slicing, and sorting overhead.
func BenchmarkBuiltins(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		list := Builtins()
		benchBlueprintsSink = list
	}
}

// BenchmarkValidate measures structural, delivery-path, and capability conflict validation.
func BenchmarkValidate(b *testing.B) {
	list := Builtins()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := Validate(list); err != nil {
			b.Fatalf("validate failed: %v", err)
		}
	}
}

// BenchmarkInit measures blueprint manifest generation, markdown rendering, and file scaffolding.
func BenchmarkInit(b *testing.B) {
	dir := b.TempDir()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		target := filepath.Join(dir, "pack")
		res, err := Init("readonly-support", target)
		if err != nil {
			b.Fatalf("init failed: %v", err)
		}
		benchResultSink = res
	}
}

// TestBenchmarkHarnessGallerySanity verifies that BenchmarkHarnessGallery executes cleanly.
func TestBenchmarkHarnessGallerySanity(t *testing.T) {
	res := testing.Benchmark(BenchmarkHarnessGallery)
	if res.N <= 0 {
		t.Fatalf("expected benchmark iterations > 0, got %d", res.N)
	}
}
