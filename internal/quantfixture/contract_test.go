package quantfixture

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestManifestVerifiesRedistributableFixtures(t *testing.T) {
	manifest, err := LoadAndVerify("testdata")
	if err != nil {
		t.Fatal(err)
	}
	if len(manifest.Fixtures) != 3 {
		t.Fatalf("fixtures = %d, want 3", len(manifest.Fixtures))
	}
	kinds := map[string]bool{}
	for _, fixture := range manifest.Fixtures {
		kinds[fixture.Kind] = true
	}
	for _, kind := range []string{"gguf", "safetensors", "bin"} {
		if !kinds[kind] {
			t.Errorf("missing %s fixture", kind)
		}
	}
}

func TestVerifierRejectsTampering(t *testing.T) {
	dir := t.TempDir()
	entries, err := os.ReadDir("testdata")
	if err != nil {
		t.Fatal(err)
	}
	for _, entry := range entries {
		data, err := os.ReadFile(filepath.Join("testdata", entry.Name()))
		if err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(dir, entry.Name()), data, 0o600); err != nil {
			t.Fatal(err)
		}
	}
	if err := os.WriteFile(filepath.Join(dir, "tiny.gguf.fixture"), []byte("tampered"), 0o600); err != nil {
		t.Fatal(err)
	}
	_, err = LoadAndVerify(dir)
	if err == nil || !strings.Contains(err.Error(), "hash mismatch") {
		t.Fatalf("error = %v, want hash mismatch", err)
	}
}

func TestUnknownRecipeFailsClosed(t *testing.T) {
	if _, err := Generate("future-recipe"); err == nil {
		t.Fatal("unknown recipe unexpectedly accepted")
	}
}
