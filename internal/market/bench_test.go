package market

import (
	"encoding/json"
	"testing"
)

func benchFixture() Descriptor {
	artifact := []byte("compiled fixture\n")
	return ComputeAdapter.Descriptor(Descriptor{
		ID:             "example.org/echo",
		Module:         "internal/compute@r3+gabcdef0",
		Compatibility:  Compatibility{Min: 1, Max: 1},
		Artifact:       "fixture.bin",
		ArtifactSHA256: SHA256(artifact),
		Trust:          TrustCompiled,
		OnError:        ErrorClosed,
		Capabilities:   []string{"gpu.execute"},
	})
}

// BenchmarkMarket exercises core market parsing and catalog serialization operations in a loop.
func BenchmarkMarket(b *testing.B) {
	d := benchFixture()
	raw, err := json.Marshal(Catalog{Schema: Schema, Extensions: []Descriptor{d}})
	if err != nil {
		b.Fatalf("setup marshal failed: %v", err)
	}
	versions := map[string]int{"fak-compute": 1}

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		cat, err := Parse(raw, versions)
		if err != nil {
			b.Fatalf("parse failed: %v", err)
		}
		if len(cat.Extensions) == 0 {
			b.Fatal("unexpected empty catalog")
		}
		if _, err := Marshal(cat); err != nil {
			b.Fatalf("marshal failed: %v", err)
		}
	}
}

func TestBenchmarkMarket(t *testing.T) {
	d := benchFixture()
	raw, err := json.Marshal(Catalog{Schema: Schema, Extensions: []Descriptor{d}})
	if err != nil {
		t.Fatalf("setup marshal failed: %v", err)
	}
	versions := map[string]int{"fak-compute": 1}
	cat, err := Parse(raw, versions)
	if err != nil {
		t.Fatalf("parse failed: %v", err)
	}
	if len(cat.Extensions) != 1 {
		t.Fatalf("expected 1 extension, got %d", len(cat.Extensions))
	}
}
