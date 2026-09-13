package model

import "testing"

// TestShadowMissResolvesViaShadowWithinGuard is the issue's primary witness: a ring miss whose expert
// has a resident low-precision shadow, measured within the NMSE bound, resolves via the shadow and
// saves the exact-load latency.
func TestShadowMissResolvesViaShadowWithinGuard(t *testing.T) {
	c := NewShadowCache(1024)
	res := c.AdmitShadowExpert(ShadowExpert{Layer: 0, Expert: 3, Bytes: 128, NMSE: 0.11, NMSEMeasured: true})
	if !res.Admitted {
		t.Fatalf("admit refused: %s", res.Reason)
	}

	p := ExpertMissPolicy{Enabled: true, Cache: c, ShadowComputeCostBytes: 8}
	d := p.ResolveExpertMiss(ExpertMissRequest{Layer: 0, Expert: 3, ExactLoadBytes: 4096, ExactLoadLatencyNanos: 9000})
	if d.Resolution != MissResolveShadow {
		t.Fatalf("got %s, want shadow (%s)", d.Resolution, d.Reason)
	}
	if d.NMSE != 0.11 || d.LatencySavedNanos != 9000 {
		t.Fatalf("got %+v, want NMSE=0.11 latencySaved=9000", d)
	}
	if d.Reason == "" {
		t.Fatalf("shadow decision must carry a reason")
	}

	st := c.Stats()
	if st.Hits != 1 || st.Lookups != 1 {
		t.Fatalf("stats %+v, want hits=1 lookups=1", st)
	}
}

// TestShadowGuardRejectsBeyondNMSEBound pins the fail-closed quality guard: an over-bound copy is
// never admitted, and a miss for it resolves exactly rather than degrading silently.
func TestShadowGuardRejectsBeyondNMSEBound(t *testing.T) {
	c := NewShadowCache(1024)
	res := c.AdmitShadowExpert(ShadowExpert{Layer: 1, Expert: 7, Bytes: 64, NMSE: ShadowNMSECeiling + 0.01, NMSEMeasured: true})
	if res.Admitted {
		t.Fatalf("over-bound shadow was admitted: %+v", res)
	}
	if c.Stats().Count != 0 {
		t.Fatalf("over-bound shadow stored anyway: %+v", c.Stats())
	}

	p := ExpertMissPolicy{Enabled: true, Cache: c, ShadowComputeCostBytes: 1}
	d := p.ResolveExpertMiss(ExpertMissRequest{Layer: 1, Expert: 7, ExactLoadBytes: 4096, ExactLoadLatencyNanos: 9000})
	if d.Resolution != MissResolveExact {
		t.Fatalf("got %s, want exact (fail closed)", d.Resolution)
	}
}

// TestShadowUnmeasuredIsRefused pins that an UNMEASURED copy (NMSEMeasured=false) can never be
// admitted nor selected: "we did not measure" and "we measured and it was bad" are both fail-closed.
func TestShadowUnmeasuredIsRefused(t *testing.T) {
	c := NewShadowCache(1024)
	if res := c.AdmitShadowExpert(ShadowExpert{Layer: 2, Expert: 1, Bytes: 64, NMSE: 0.0, NMSEMeasured: false}); res.Admitted {
		t.Fatalf("unmeasured shadow admitted: %+v", res)
	}

	// Defense-in-depth: even a shadow forced into the map out of band must not be selected.
	c.shadows[shadowKey{2, 1}] = ShadowExpert{Layer: 2, Expert: 1, Bytes: 64, NMSE: 0.0, NMSEMeasured: false}
	p := ExpertMissPolicy{Enabled: true, Cache: c, ShadowComputeCostBytes: 1}
	d := p.ResolveExpertMiss(ExpertMissRequest{Layer: 2, Expert: 1, ExactLoadBytes: 4096})
	if d.Resolution != MissResolveExact {
		t.Fatalf("got %s, want exact for unmeasured shadow", d.Resolution)
	}
	if c.Stats().GuardRejections != 1 {
		t.Fatalf("stats %+v, want guard_rejections=1", c.Stats())
	}
}

// TestShadowDisabledIsByteIdenticalExactPath pins default-off: a zero policy resolves every miss
// exactly and never touches the cache.
func TestShadowDisabledIsByteIdenticalExactPath(t *testing.T) {
	c := NewShadowCache(1024)
	c.AdmitShadowExpert(ShadowExpert{Layer: 0, Expert: 0, Bytes: 8, NMSE: 0.01, NMSEMeasured: true})

	var p ExpertMissPolicy // zero value: disabled
	d := p.ResolveExpertMiss(ExpertMissRequest{Layer: 0, Expert: 0, ExactLoadBytes: 4096, ExactLoadLatencyNanos: 5000})
	if d.Resolution != MissResolveExact {
		t.Fatalf("disabled policy resolved %s, want exact", d.Resolution)
	}
	if st := c.Stats(); st.Lookups != 0 {
		t.Fatalf("disabled policy touched the cache: %+v", st)
	}
}

// TestShadowNotCheaperResolvesExact pins the stated latency/quality rule: a shadow whose compute is
// not strictly cheaper than the exact load is declined even when its quality is acceptable.
func TestShadowNotCheaperResolvesExact(t *testing.T) {
	c := NewShadowCache(1024)
	c.AdmitShadowExpert(ShadowExpert{Layer: 0, Expert: 1, Bytes: 32, NMSE: 0.05, NMSEMeasured: true})

	p := ExpertMissPolicy{Enabled: true, Cache: c, ShadowComputeCostBytes: 4096}
	d := p.ResolveExpertMiss(ExpertMissRequest{Layer: 0, Expert: 1, ExactLoadBytes: 4096, ExactLoadLatencyNanos: 9000})
	if d.Resolution != MissResolveExact {
		t.Fatalf("got %s, want exact when shadow is not cheaper", d.Resolution)
	}
}

// TestShadowCacheStaysBounded pins the byte bound and the deterministic LRU eviction: peaks never
// exceed the budget and the coldest shadow is the one dropped.
func TestShadowCacheStaysBounded(t *testing.T) {
	c := NewShadowCache(100)
	if res := c.AdmitShadowExpert(ShadowExpert{Layer: 0, Expert: 0, Bytes: 60, NMSE: 0.02, NMSEMeasured: true}); !res.Admitted {
		t.Fatalf("admit a: %s", res.Reason)
	}
	// Touch expert 0 so expert 1 becomes the colder candidate on the next admit.
	c.ShadowLookup(0, 0)
	res := c.AdmitShadowExpert(ShadowExpert{Layer: 0, Expert: 1, Bytes: 60, NMSE: 0.03, NMSEMeasured: true})
	if !res.Admitted {
		t.Fatalf("admit b: %s", res.Reason)
	}
	if len(res.Evicted) != 1 || res.Evicted[0].Expert != 0 {
		t.Fatalf("evicted %+v, want expert 0 (coldest)", res.Evicted)
	}
	st := c.Stats()
	if st.Bytes > st.BudgetBytes || st.PeakBytes > st.BudgetBytes {
		t.Fatalf("cache exceeded budget: %+v", st)
	}
	if _, ok := c.ShadowLookup(0, 1); !ok {
		t.Fatalf("expert 1 should be resident after admit")
	}
}

// TestShadowOversizeNeverAdmitted pins that a copy larger than the whole budget is refused rather
// than evicting everything to hold something that can never be resident.
func TestShadowOversizeNeverAdmitted(t *testing.T) {
	c := NewShadowCache(50)
	res := c.AdmitShadowExpert(ShadowExpert{Layer: 0, Expert: 0, Bytes: 64, NMSE: 0.02, NMSEMeasured: true})
	if res.Admitted {
		t.Fatalf("oversize shadow admitted: %+v", res)
	}
	if c.Stats().Count != 0 {
		t.Fatalf("oversize shadow stored: %+v", c.Stats())
	}
}

// TestShadowDefaultOffEmptyCacheIsExact pins that an enabled policy with an empty cache still
// resolves every miss exactly — the missing-shadow branch is fail-closed too.
func TestShadowDefaultOffEmptyCacheIsExact(t *testing.T) {
	p := ExpertMissPolicy{Enabled: true, Cache: NewShadowCache(1024), ShadowComputeCostBytes: 1}
	d := p.ResolveExpertMiss(ExpertMissRequest{Layer: 9, Expert: 4, ExactLoadBytes: 4096, ExactLoadLatencyNanos: 9000})
	if d.Resolution != MissResolveExact {
		t.Fatalf("got %s, want exact for a missing shadow", d.Resolution)
	}
}
