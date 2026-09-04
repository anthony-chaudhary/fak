package wipfence

import (
	"testing"
)

// BenchmarkWIPFence measures performance of repeated fence, inspect, and unfence operations.
func BenchmarkWIPFence(b *testing.B) {
	const body = "package main\n\nfunc x() {}\n"
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		fenced, changed, err := Fence(body, "bench_test")
		if err != nil || !changed {
			b.Fatalf("Fence: changed=%v, err=%v", changed, err)
		}
		tag, ok := IsFenced(fenced)
		if !ok || tag != "wip_bench_test" {
			b.Fatalf("IsFenced: ok=%v, tag=%q", ok, tag)
		}
		unfenced, changed, err := Unfence(fenced)
		if err != nil || !changed || unfenced != body {
			b.Fatalf("Unfence: changed=%v, err=%v", changed, err)
		}
	}
}
