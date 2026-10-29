package agent

import (
	"context"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/model"
)

// [SW-VERIFIED] A long generation grows the lease one private page per page boundary
// and releases to zero.
func TestInKernelPagedPrefixLeaseGrowsOverGeneratedTokens(t *testing.T) {
	cfg := tinyCfg()
	l, err := newInKernelPagedPrefixLedger(cfg, 64, 16)
	if err != nil {
		t.Fatal(err)
	}
	ids := synthIDs(cfg.VocabSize, 40, 1) // 3 prompt pages, 8 slots free in the last
	lease := l.acquire("", ids)
	if lease == nil {
		t.Fatal("no lease")
	}
	base := l.pool.Pool().PhysicalBlocksAllocated()
	steps := []struct {
		gen    int
		active int64
	}{{1, 0}, {8, 0}, {9, 1}, {24, 1}, {25, 2}, {40, 2}}
	for _, st := range steps {
		lease.grow(st.gen)
		if tel := l.telemetry(); tel.ActiveDecodePages != st.active {
			t.Fatalf("after %d generated tokens active decode pages = %d, want %d", st.gen, tel.ActiveDecodePages, st.active)
		}
	}
	if live := l.pool.Pool().PhysicalBlocksAllocated(); live != base+2 {
		t.Fatalf("live pages = %d, want %d (prompt chain + 2 decode)", live, base+2)
	}
	lease.release()
	lease.release() // idempotent
	tel := l.telemetry()
	if tel.ActiveDecodePages != 0 || tel.ActiveLeases != 0 || tel.DecodePages != 2 {
		t.Fatalf("after release telemetry = %+v, want 0 active decode pages, 0 active leases, 2 cumulative decode pages", tel)
	}
	if live := l.pool.Pool().PhysicalBlocksAllocated(); live != base {
		t.Fatalf("live pages after release = %d, want %d (cached prompt chain only)", live, base)
	}
}

// Two tenants with an identical prompt grow disjoint, request-private decode pages.
func TestInKernelPagedPrefixDecodePagesArePrivateAcrossTenants(t *testing.T) {
	cfg := tinyCfg()
	l, err := newInKernelPagedPrefixLedger(cfg, 64, 16)
	if err != nil {
		t.Fatal(err)
	}
	ids := synthIDs(cfg.VocabSize, 32, 1)
	a, b := l.acquire("tenant-a", ids), l.acquire("tenant-b", ids)
	if a == nil || b == nil {
		t.Fatal("missing lease")
	}
	a.grow(20)
	b.grow(20)
	if len(a.decode) != 2 || len(b.decode) != 2 {
		t.Fatalf("decode pages a=%v b=%v, want 2 each", a.decode, b.decode)
	}
	for _, x := range a.decode {
		for _, y := range b.decode {
			if x == y {
				t.Fatalf("decode page %d shared between tenants", x)
			}
		}
		if ref := pagedPrefixRefCount(t, l, x); ref != 1 {
			t.Fatalf("decode page %d refcount = %d, want 1 (request-private)", x, ref)
		}
	}
	a.release()
	if got := l.telemetry().ActiveDecodePages; got != 2 {
		t.Fatalf("active decode pages after releasing one tenant = %d, want 2", got)
	}
	b.release()
	if got := l.telemetry().ActiveDecodePages; got != 0 {
		t.Fatalf("active decode pages after releasing both = %d, want 0", got)
	}
}

// A full pool stops the lease growing without failing the request.
func TestInKernelPagedPrefixLeaseGrowthFailsOpen(t *testing.T) {
	l, err := newInKernelPagedPrefixLedger(tinyCfg(), 3, 16)
	if err != nil {
		t.Fatal(err)
	}
	lease := l.acquire("", synthIDs(tinyCfg().VocabSize, 40, 1)) // takes all 3 pages
	if lease == nil {
		t.Fatal("no lease")
	}
	lease.grow(30)
	lease.grow(31) // no retry after the first refusal
	if tel := l.telemetry(); tel.DecodeFailOpen != 1 || tel.ActiveDecodePages != 0 {
		t.Fatalf("telemetry = %+v, want one decode fail-open and no decode pages", tel)
	}
	lease.release()
}

// Drives the production decode path: a 40-token generation grows the lease while
// decoding and releases to zero.
func TestInKernelPlannerGenerationGrowsAndReleasesPageLease(t *testing.T) {
	t.Setenv("FAK_INKERNEL_RADIX", "on")
	cfg := tinyCfg()
	p := NewInKernelPlanner(model.NewSynthetic(cfg), nil, "synthetic", false, nil, false)
	p.quant = false
	l, err := newInKernelPagedPrefixLedger(cfg, 64, 16)
	if err != nil {
		t.Fatal(err)
	}
	p.pagedPrefix = l
	ids := synthIDs(cfg.VocabSize, 40, 1)
	var peak int64
	emit := func(int) bool {
		if a := l.telemetry().ActiveDecodePages; a > peak {
			peak = a
		}
		return false
	}
	if _, _, _, _, _, _, _, _, err := p.generateReusedContextWithBias(context.Background(), ids, 40, 0, 0, 0, nil, 0, 0, map[int]bool{}, emit); err != nil {
		t.Fatal(err)
	}
	tel, _ := p.PagedPrefixTelemetry()
	if peak < 2 || tel.DecodePages != 2 || tel.ActiveDecodePages != 0 || tel.ActiveLeases != 0 {
		t.Fatalf("peak active decode pages=%d telemetry=%+v, want peak>=2, 2 decode pages, all released", peak, tel)
	}
}
