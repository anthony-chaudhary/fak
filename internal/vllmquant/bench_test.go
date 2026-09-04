package vllmquant

import (
	"testing"
)

// BenchmarkVLLMQuant exercises quantization kernel selection in a loop to measure
// allocation overhead and adjudication latency across kernel candidate decisions.
func BenchmarkVLLMQuant(b *testing.B) {
	groupSize := 128
	sym := false
	req := Request{
		Schema: SchemaVersion,
		Artifact: Artifact{
			QuantMethod: MethodAWQ,
			WeightBits:  4,
			GroupSize:   &groupSize,
			Symmetric:   &sym,
		},
		Server: Server{
			Version:                 "0.8.5",
			Kernels:                 []Kernel{KernelAWQ, KernelAWQMarlin},
			ComputeCapability:       80,
			Dtype:                   "float16",
			KernelOrderIsPreference: true,
		},
	}

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		sel := Adjudicate(req)
		if sel.Outcome != OutcomeSupported {
			b.Fatalf("unexpected outcome: %v", sel.Outcome)
		}
	}
}

func TestBenchmarkVLLMQuantSmoke(t *testing.T) {
	res := testing.Benchmark(BenchmarkVLLMQuant)
	if res.N == 0 {
		t.Fatal("benchmark did not run")
	}
}
