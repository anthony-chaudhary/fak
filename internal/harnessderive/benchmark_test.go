package harnessderive

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/harnesscompose"
	"github.com/anthony-chaudhary/fak/internal/harnessresolve"
)

// BenchmarkHarnessDerive benchmarks end-to-end lock derivation across multiple typed deltas
// including instruction replacement and policy deny narrowing.
func BenchmarkHarnessDerive(b *testing.B) {
	base := fixture(b, false, false)
	base.Assets = append(base.Assets, harnesscompose.EffectiveAsset{
		Kind:   "policy",
		ID:     "tools",
		Grants: []string{"search", "shell", "read"},
		Source: "company:support",
	})
	if err := harnessresolve.ReidentifyLock(&base); err != nil {
		b.Fatal(err)
	}

	req := Request{
		Layer: "benchmark",
		Deltas: []Delta{
			{Capability: "instruction:style", Operation: "replace", Value: "detailed"},
			{Capability: "policy:tools", Operation: "deny", Denies: []string{"shell"}},
		},
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		res, err := Derive(base, req)
		if err != nil {
			b.Fatalf("derive failed: %v", err)
		}
		if res.Lock.ID == "" {
			b.Fatal("empty derived lock id")
		}
	}
}

// BenchmarkHarnessDeriveSingleDelta benchmarks lock derivation with a single instruction replacement.
func BenchmarkHarnessDeriveSingleDelta(b *testing.B) {
	base := fixture(b, false, false)
	req := Request{
		Layer: "benchmark",
		Deltas: []Delta{
			{Capability: "instruction:style", Operation: "replace", Value: "detailed"},
		},
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		res, err := Derive(base, req)
		if err != nil {
			b.Fatalf("derive failed: %v", err)
		}
		if res.Lock.ID == "" {
			b.Fatal("empty derived lock id")
		}
	}
}

// BenchmarkVerifyReceipt benchmarks cryptographic validation and digest verification of derivation receipts.
func BenchmarkVerifyReceipt(b *testing.B) {
	base := fixture(b, false, false)
	res, err := Derive(base, Request{
		Layer: "benchmark",
		Deltas: []Delta{
			{Capability: "instruction:style", Operation: "replace", Value: "detailed"},
		},
	})
	if err != nil {
		b.Fatalf("derive setup failed: %v", err)
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := VerifyReceipt(res.Receipt); err != nil {
			b.Fatalf("verify receipt failed: %v", err)
		}
	}
}
