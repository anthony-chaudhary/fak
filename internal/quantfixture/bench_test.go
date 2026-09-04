package quantfixture

import (
	"path/filepath"
	"testing"
)

func BenchmarkQuantFixture(b *testing.B) {
	dir := filepath.Join("testdata")
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		manifest, err := LoadAndVerify(dir)
		if err != nil {
			b.Fatalf("LoadAndVerify failed: %v", err)
		}
		if len(manifest.Fixtures) == 0 {
			b.Fatal("unexpected empty fixtures")
		}
	}
}

func TestBenchmarkSmoke(t *testing.T) {
	manifest, err := LoadAndVerify("testdata")
	if err != nil {
		t.Fatalf("LoadAndVerify failed: %v", err)
	}
	if len(manifest.Fixtures) == 0 {
		t.Fatal("expected fixtures in testdata")
	}
}
