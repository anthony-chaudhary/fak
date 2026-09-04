package ociartifact

import (
	"net/http/httptest"
	"testing"
)

// Invariant: Performance benchmarks measure real registry resolution operations with zero mock bypasses.
// Precondition: Test registry must be initialized and seeded with an artifact manifest before timing starts.
// Postcondition: b.N resolution iterations complete successfully without triggering unexpected errors.

func benchmarkArtifact(b *testing.B) Artifact {
	b.Helper()
	payloads := map[string][]byte{
		"skills/review.json": []byte(`{"name":"review"}`),
	}
	cfg := Config{
		Schema:  ProfileVersion,
		Name:    "bench",
		Version: "1.0.0",
		Objects: []Object{
			{Name: "review", Kind: "skill", MediaType: SkillMediaType, Path: "skills/review.json"},
		},
	}
	a, err := Build(cfg, payloads, nil)
	if err != nil {
		b.Fatalf("Build failed: %v", err)
	}
	return a
}

// BenchmarkOCIArtifactResolve measures the throughput and allocations of Client.Resolve
// against an in-memory OCI test registry.
func BenchmarkOCIArtifactResolve(b *testing.B) {
	a := benchmarkArtifact(b)
	reg := newTestRegistry()
	srv := httptest.NewServer(reg)
	defer srv.Close()

	c := Client{Base: srv.URL, Repository: "acme/fak", HTTP: srv.Client()}
	if err := c.Push("mixed:mutable", a); err != nil {
		b.Fatalf("Push failed: %v", err)
	}

	b.ReportAllocs()
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		desc, err := c.Resolve("mixed:mutable")
		if err != nil {
			b.Fatalf("Resolve failed: %v", err)
		}
		if desc.Digest != a.Manifest.Digest {
			b.Fatalf("digest mismatch: got %s, want %s", desc.Digest, a.Manifest.Digest)
		}
	}
}
