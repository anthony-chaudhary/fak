package bitnetruntime

import (
	"context"
	"testing"
)

func BenchmarkParseReport(b *testing.B) {
	raw := []byte("version: 2026.08\nbitnet.cpp: 1.0.0\nkernels: i2_s,tl1,tl2\n")
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		rt, reason := ParseReport(raw)
		if reason != "" || rt.Version != "1.0.0" {
			b.Fatalf("unexpected parse failure: reason=%v, rt=%+v", reason, rt)
		}
	}
}

func BenchmarkAdmit(b *testing.B) {
	rt := Runtime{
		Name:    RuntimeName,
		Version: "1.0.0",
		Build:   "2026.08",
		Kernels: []Kernel{KernelI2S, KernelTL2},
	}
	host := Host{
		OS:       "linux",
		Arch:     "amd64",
		Features: []string{"avx2", "fma"},
	}
	model := Model{
		ID:                  "test-model",
		Alphabet:            AlphabetTernary,
		Kernel:              KernelTL2,
		BitsPerWeightStored: 1.667,
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		res := Admit(rt, host, model)
		if res.Outcome != OutcomeDelegate {
			b.Fatalf("unexpected outcome: %v", res.Outcome)
		}
	}
}

func BenchmarkDiscoverAndAdmit(b *testing.B) {
	raw := []byte("version: 2026.08\nbitnet.cpp: 1.0.0\nkernels: i2_s,tl1\n")
	prober := func(ctx context.Context) ([]byte, error) {
		return raw, nil
	}
	host := Host{
		OS:       "darwin",
		Arch:     "arm64",
		Features: []string{"neon"},
	}
	model := Model{
		ID:                  "test-model-arm",
		Alphabet:            AlphabetTernary,
		Kernel:              KernelTL1,
		BitsPerWeightStored: 2.0,
	}
	ctx := context.Background()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		res := DiscoverAndAdmit(ctx, prober, host, model)
		if res.Outcome != OutcomeDelegate {
			b.Fatalf("unexpected outcome: %v", res.Outcome)
		}
	}
}
