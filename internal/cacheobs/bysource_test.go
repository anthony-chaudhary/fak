package cacheobs

import (
	"sync"
	"testing"
)

// TestObserveBySourceParticles is the parts==total invariant the issue names (#3896): a
// known provenance decomposition books into three buckets whose sum is exactly the total,
// and ReusedTokens is the non-recomputed remainder — so the source axis reconciles with the
// depth axis's notion of "reused" without either being able to drift.
func TestObserveBySourcePartsEqualTotal(t *testing.T) {
	o := New()
	o.ObserveBySource(SourceLocalCompute, 300)
	o.ObserveBySource(SourceLocalHit, 500)
	o.ObserveBySource(SourceExternalTransfer, 200)

	s := o.SourceSnapshot()
	if s.LocalComputeTokens != 300 || s.LocalHitTokens != 500 || s.ExternalTransferTokens != 200 {
		t.Fatalf("buckets = (compute=%d hit=%d external=%d), want (300 500 200)",
			s.LocalComputeTokens, s.LocalHitTokens, s.ExternalTransferTokens)
	}
	if got := s.LocalComputeTokens + s.LocalHitTokens + s.ExternalTransferTokens; got != s.TotalTokens {
		t.Fatalf("parts (%d) != total (%d): the parts==total invariant is broken", got, s.TotalTokens)
	}
	if s.TotalTokens != 1000 {
		t.Fatalf("total = %d, want 1000", s.TotalTokens)
	}
	if s.ReusedTokens != 700 {
		t.Fatalf("reused = %d, want 700 (hit + external)", s.ReusedTokens)
	}
	// The disaggregation dividend: external transfer is 200/1000 of served value.
	if got := s.ExternalTransferRatio; got < 0.2-1e-9 || got > 0.2+1e-9 {
		t.Fatalf("external transfer ratio = %v, want 0.2", got)
	}
	if got := s.LocalHitRatio; got < 0.5-1e-9 || got > 0.5+1e-9 {
		t.Fatalf("local hit ratio = %v, want 0.5", got)
	}
}

// TestObserveBySourceAccumulates confirms repeated books into the same bucket sum, and that
// non-positive counts and out-of-range sources are ignored (the closed vocabulary), so a
// summed snapshot can only under-count, never mis-attribute.
func TestObserveBySourceAccumulates(t *testing.T) {
	o := New()
	o.ObserveBySource(SourceExternalTransfer, 100)
	o.ObserveBySource(SourceExternalTransfer, 250)
	o.ObserveBySource(SourceLocalHit, 0)     // ignored: no value to attribute
	o.ObserveBySource(SourceLocalHit, -5)    // ignored: negative
	o.ObserveBySource(ReuseSource(99), 1000) // ignored: outside the closed vocabulary

	s := o.SourceSnapshot()
	if s.ExternalTransferTokens != 350 {
		t.Fatalf("external transfer = %d, want 350", s.ExternalTransferTokens)
	}
	if s.LocalHitTokens != 0 || s.LocalComputeTokens != 0 {
		t.Fatalf("expected only external booked, got (compute=%d hit=%d)", s.LocalComputeTokens, s.LocalHitTokens)
	}
	if s.TotalTokens != 350 {
		t.Fatalf("total = %d, want 350 (the out-of-range 1000 must not have landed anywhere)", s.TotalTokens)
	}
}

// TestSourceSnapshotEmpty pins the idle-process contract: no observations means the zero
// snapshot with no phantom ratios, and a nil observer is safe.
func TestSourceSnapshotEmpty(t *testing.T) {
	if s := New().SourceSnapshot(); s.TotalTokens != 0 || s.ExternalTransferRatio != 0 || s.LocalHitRatio != 0 {
		t.Fatalf("idle snapshot reported a phantom split: %+v", s)
	}
	var nilObs *Observer
	if s := nilObs.SourceSnapshot(); s != (SourceStats{}) {
		t.Fatalf("nil observer must return the zero SourceStats, got %+v", s)
	}
	nilObs.ObserveBySource(SourceLocalHit, 100) // must not panic
}

// TestReuseSourceString locks the label spelling against vLLM's by_source values so a fak
// exposition and a vLLM one read on the same axis.
func TestReuseSourceString(t *testing.T) {
	for src, want := range map[ReuseSource]string{
		SourceLocalCompute:     "local_compute",
		SourceLocalHit:         "local_cache_hit",
		SourceExternalTransfer: "external_kv_transfer",
		ReuseSource(42):        "unknown",
	} {
		if got := src.String(); got != want {
			t.Fatalf("ReuseSource(%d).String() = %q, want %q", int(src), got, want)
		}
	}
}

// TestObserveBySourceConcurrent exercises the lock under -race: many goroutines booking into
// all three buckets must leave an exact, reconciling total (parts==total holds under
// contention, no lost updates).
func TestObserveBySourceConcurrent(t *testing.T) {
	o := New()
	const goroutines, perG = 8, 1000
	var wg sync.WaitGroup
	for g := 0; g < goroutines; g++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := 0; i < perG; i++ {
				o.ObserveBySource(SourceLocalCompute, 1)
				o.ObserveBySource(SourceLocalHit, 2)
				o.ObserveBySource(SourceExternalTransfer, 3)
			}
		}()
	}
	wg.Wait()

	s := o.SourceSnapshot()
	wantCompute := uint64(goroutines * perG * 1)
	wantHit := uint64(goroutines * perG * 2)
	wantExternal := uint64(goroutines * perG * 3)
	if s.LocalComputeTokens != wantCompute || s.LocalHitTokens != wantHit || s.ExternalTransferTokens != wantExternal {
		t.Fatalf("lost updates under contention: got (compute=%d hit=%d external=%d), want (%d %d %d)",
			s.LocalComputeTokens, s.LocalHitTokens, s.ExternalTransferTokens, wantCompute, wantHit, wantExternal)
	}
	if s.LocalComputeTokens+s.LocalHitTokens+s.ExternalTransferTokens != s.TotalTokens {
		t.Fatalf("parts != total under contention: %+v", s)
	}
}
