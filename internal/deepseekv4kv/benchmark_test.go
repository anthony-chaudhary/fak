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

// BenchmarkAmplify measures the amplification calculation across policies and context lengths.
func BenchmarkAmplify(b *testing.B) {
	policies := []Policy{FullSWACache, PeriodicCheckpoint, ZeroSWACache}
	seqs := ReportContexts

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		p := policies[i%len(policies)]
		s := seqs[i%len(seqs)]
		amp := Amplify(s, p, 4096)
		if amp.StorageAmp <= 0 {
			b.Fatalf("unexpected storage amp: %f", amp.StorageAmp)
		}
	}
}

// BenchmarkSubCacheUnits measures normalized storage accounting across cache kinds.
func BenchmarkSubCacheUnits(b *testing.B) {
	kinds := []Kind{KindCSA, KindHCA, KindSWA, KindTail}
	seqs := ReportContexts

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		k := kinds[i%len(kinds)]
		s := seqs[i%len(seqs)]
		units := SubCacheUnits(k, s)
		if units <= 0 {
			b.Fatalf("unexpected units: %f", units)
		}
	}
}

// BenchmarkReport measures full amplification report generation across all policies.
func BenchmarkReport(b *testing.B) {
	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		rows := Report(4096)
		if len(rows) == 0 {
			b.Fatal("unexpected empty report")
		}
	}
}

// BenchmarkValidateBlockAccounting measures the fail-closed self-check verification.
func BenchmarkValidateBlockAccounting(b *testing.B) {
	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		if err := ValidateBlockAccounting(); err != nil {
			b.Fatalf("validation failed: %v", err)
		}
	}
}
