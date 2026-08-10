package vdso

import "testing"

func TestCompareLocalKeepsExternalCachesExplicit(t *testing.T) {
	r := CompareLocal(10)
	if r.Schema != ComparisonSchema || r.Complete {
		t.Fatalf("schema/complete=%q/%v", r.Schema, r.Complete)
	}
	if len(r.Arms) != 4 {
		t.Fatalf("arms=%d", len(r.Arms))
	}
	for _, a := range r.Arms[:2] {
		if !a.Available || a.Correctness != 1 {
			t.Fatalf("local=%+v", a)
		}
	}
	if r.Arms[0].UpstreamCalls != 0 || r.Arms[1].UpstreamCalls != 10 {
		t.Fatalf("upstream counts=%d/%d", r.Arms[0].UpstreamCalls, r.Arms[1].UpstreamCalls)
	}
	for _, a := range r.Arms[2:] {
		if a.Available || a.UnavailableReason == "" {
			t.Fatalf("external=%+v", a)
		}
	}
}
func BenchmarkToolResultCacheComparison(b *testing.B) {
	v := New(16)
	v.RegisterStatic("system_version", []byte(`{"version":"1.2.3"}`))
	c := comparisonCall()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = v.Lookup(b.Context(), &c)
	}
}
