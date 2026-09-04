package ociartifact

import (
	"testing"
)

// Invariant: OCI artifact collection building must preserve content digests and produce valid manifest layouts.
// Guard: Build rejects configs missing schema, name, or valid media type mappings.

func TestOCIArtifactLifecycle(t *testing.T) {
	t.Parallel()

	payloads := map[string][]byte{"test.txt": []byte("hello world")}
	cfg := Config{
		Schema:  "fak.oci.collection/v1",
		Name:    "lifecycle-test",
		Version: "1.0.0",
		Objects: []Object{
			{Name: "test", Kind: "text", MediaType: "text/plain", Path: "test.txt"},
		},
	}

	art, err := Build(cfg, payloads, nil)
	if err != nil {
		t.Fatalf("Build failed: %v", err)
	}
	if art.Parsed.SchemaVersion != 2 {
		t.Fatalf("expected Manifest SchemaVersion 2, got %d", art.Parsed.SchemaVersion)
	}
}
