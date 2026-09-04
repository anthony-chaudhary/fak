package harnessresolve

import (
	"context"
	"encoding/json"
	"testing"
)

// BenchmarkHarnessResolve benchmarks the end-to-end resolution of component dependencies,
// asset layer composition, and deterministic lock synthesis.
func BenchmarkHarnessResolve(b *testing.B) {
	manifest := validManifest()
	layers := []string{"company", "legal"}
	env := Environment{OS: "linux", Arch: "amd64", Contract: "v1"}
	ctx := context.Background()

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		res, err := Resolve(ctx, manifest, layers, env)
		if err != nil {
			b.Fatalf("resolve failed: %v", err)
		}
		if res.Lock.ID == "" {
			b.Fatal("empty lock id")
		}
	}
}

// BenchmarkVerifyLock benchmarks cryptographic verification and digest matching of resolved product locks.
func BenchmarkVerifyLock(b *testing.B) {
	manifest := validManifest()
	res, err := Resolve(context.Background(), manifest, []string{"company", "legal"}, Environment{OS: "linux", Arch: "amd64", Contract: "v1"})
	if err != nil {
		b.Fatalf("resolve setup failed: %v", err)
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := VerifyLock(res.Lock); err != nil {
			b.Fatalf("verify lock failed: %v", err)
		}
	}
}

// BenchmarkParseManifest benchmarks JSON deserialization and schema validation of product manifests.
func BenchmarkParseManifest(b *testing.B) {
	raw, err := json.Marshal(validManifest())
	if err != nil {
		b.Fatalf("marshal setup failed: %v", err)
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		m, err := Parse(raw)
		if err != nil {
			b.Fatalf("parse failed: %v", err)
		}
		if len(m.Components) == 0 {
			b.Fatal("empty components")
		}
	}
}
