package deepseekv4kv

import "testing"

// BenchmarkDeepSeekV4KV exercises servable prefix calculation across representative
// token sequence lengths and sub-cache combinations in a loop.
func BenchmarkDeepSeekV4KV(b *testing.B) {
	kinds := []Kind{KindCSA, KindHCA, KindSWA, KindTail}
	seqs := []int{128, 512, Ctx128K, Ctx512K, Ctx1M}

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		seq := seqs[i%len(seqs)]
		sink := ServablePrefixUnits(seq, kinds)
		if sink <= 0 {
			b.Fatalf("expected positive servable prefix units, got %d", sink)
		}
	}
}
