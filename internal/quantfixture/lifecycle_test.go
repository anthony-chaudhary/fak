package quantfixture

import (
	"testing"
)

// Invariant: Quantization fixture manifests must verify cryptographic digests of committed fixtures.
// Guard: LoadAndVerify returns an error when files are tampered with or missing from manifest.

func TestQuantFixtureLifecycle(t *testing.T) {
	t.Parallel()

	manifest, err := LoadAndVerify("testdata")
	if err != nil {
		t.Fatalf("LoadAndVerify failed: %v", err)
	}
	if len(manifest.Fixtures) == 0 {
		t.Fatal("expected non-empty fixtures in manifest")
	}
}
