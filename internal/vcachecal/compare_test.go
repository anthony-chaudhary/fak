package vcachecal

import "testing"

func TestCompareLocalKeepsAllocationAlternativesExplicit(t *testing.T) {
	got := CompareLocal()
	want := []struct{ name, kind string }{{"fak native concentration-weighted allocation", "native"}, {"equal-share cache allocation", "baseline"}, {"request-volume proportional allocation", "baseline"}, {"fak + LMCache", "integration"}, {"fak + Mooncake", "integration"}, {"vLLM cache-aware routing", "external"}, {"SGLang HiCache and cache-aware scheduling", "external"}}
	if len(got.Arms) != len(want) {
		t.Fatalf("arms=%d", len(got.Arms))
	}
	for i, a := range got.Arms {
		if a.Name != want[i].name || a.Kind != want[i].kind {
			t.Fatalf("arm[%d]=%+v", i, a)
		}
		if i < 3 {
			if !a.Available || a.Buckets != 3 || a.AllocatedBytes != 1200 {
				t.Fatalf("local=%+v", a)
			}
			continue
		}
		if a.Available || a.Correct || a.Latency != 0 || a.Buckets != 0 || a.AllocatedBytes != 0 || a.CapturedValue != 0 || a.CostUSD != 0 {
			t.Fatalf("unwitnessed=%+v", a)
		}
	}
	if !got.Arms[0].Correct || got.Arms[1].Correct || got.Arms[2].Correct {
		t.Fatalf("oracle=%+v", got.Arms)
	}
}
func BenchmarkAllocateByConcentration(b *testing.B) {
	buckets, total, topK := allocationFixture()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = AllocateByConcentration(buckets, total, topK)
	}
}
