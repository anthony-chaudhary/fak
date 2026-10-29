package agent

import (
	"context"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/model"
	"github.com/anthony-chaudhary/fak/internal/radixkv"
)

func pagedPrefixRefCount(t *testing.T, l *inKernelPagedPrefixLedger, id int) int32 {
	t.Helper()
	blk, err := l.pool.Pool().Allocator().GetBlock(id)
	if err != nil {
		t.Fatalf("GetBlock(%d): %v", id, err)
	}
	return blk.RefCount()
}

// TestInKernelPagedPrefixLedgerSharesPrefixPages pins the consumer contract: a second
// request whose prompt shares a cached prefix binds the SAME physical pages for every
// full prefix page (refcount++), allocates only its divergent tail, and releasing the
// requests leaves the cached chain owned by the tree alone.
func TestInKernelPagedPrefixLedgerSharesPrefixPages(t *testing.T) {
	l, err := newInKernelPagedPrefixLedger(tinyCfg(), 64, 16)
	if err != nil {
		t.Fatal(err)
	}
	prefix := synthIDs(tinyCfg().VocabSize, 32, 1) // exactly two pages
	req1 := append(append([]int{}, prefix...), synthIDs(tinyCfg().VocabSize, 8, 2)...)
	req2 := append(append([]int{}, prefix...), synthIDs(tinyCfg().VocabSize, 10, 3)...)

	first := l.acquire("", req1)
	if first == nil || first.sharedPages != 0 || first.freshPages != 3 {
		t.Fatalf("cold request lease = %+v, want 0 shared / 3 fresh pages", first)
	}
	second := l.acquire("", req2)
	if second == nil || second.sharedPages != 2 || second.freshPages != 1 {
		t.Fatalf("shared-prefix request lease = %+v, want 2 shared / 1 fresh pages", second)
	}
	p1, p2 := first.table.Pages(), second.table.Pages()
	for i := 0; i < 2; i++ {
		if p1[i] != p2[i] {
			t.Fatalf("prefix page %d: first=%d second=%d, want the same physical page", i, p1[i], p2[i])
		}
	}
	if p1[2] == p2[2] {
		t.Fatalf("divergent tail page %d shared between requests", p1[2])
	}
	if live := l.pool.Pool().PhysicalBlocksAllocated(); live != 4 {
		t.Fatalf("live pages with both requests active = %d, want 4 (2 shared + 1 tail each), not 6", live)
	}
	// Two tree chains + two stream leases reference each shared prefix page.
	for _, id := range p2[:2] {
		if ref := pagedPrefixRefCount(t, l, id); ref != 4 {
			t.Fatalf("shared page %d refcount = %d, want 4", id, ref)
		}
	}

	first.release()
	second.release()
	second.release() // idempotent
	for _, id := range p2[:2] {
		if ref := pagedPrefixRefCount(t, l, id); ref != 2 {
			t.Fatalf("shared page %d refcount after release = %d, want 2 (tree chains only)", id, ref)
		}
	}
	tel := l.telemetry()
	if tel.Leases != 2 || tel.ActiveLeases != 0 || tel.SharedPages != 2 || tel.AllocatedPages != 4 || tel.LivePages != 4 {
		t.Fatalf("telemetry = %+v", tel)
	}
	if tel.SharedKVElems != 2*tel.PageKVElems || tel.PageKVElems <= 0 {
		t.Fatalf("shared KV elems = %d, page elems = %d", tel.SharedKVElems, tel.PageKVElems)
	}

	// A third request over the same prefix reuses the published chain after release.
	third := l.acquire("", req2)
	if third == nil || third.sharedPages != 3 || third.freshPages != 0 {
		t.Fatalf("exact repeat lease = %+v, want 3 shared / 0 fresh", third)
	}
	third.release()

	// Pages never cross a tenant namespace.
	other := l.acquire("tenant-b", req2)
	if other == nil || other.sharedPages != 0 {
		t.Fatalf("cross-namespace lease = %+v, want no shared pages", other)
	}
	other.release()
}

// TestInKernelPagedPrefixLedgerFailsOpenOnExhaustion pins that a full pool never fails
// the request: the lease is nil and the fail-open counter moves.
func TestInKernelPagedPrefixLedgerFailsOpenOnExhaustion(t *testing.T) {
	l, err := newInKernelPagedPrefixLedger(tinyCfg(), 2, 16)
	if err != nil {
		t.Fatal(err)
	}
	held := l.acquire("", synthIDs(tinyCfg().VocabSize, 32, 7))
	if held == nil {
		t.Fatal("two-page request did not fit a two-page pool")
	}
	if lease := l.acquire("", synthIDs(tinyCfg().VocabSize, 32, 8)); lease != nil {
		t.Fatalf("exhausted pool returned lease %+v", lease)
	}
	if tel := l.telemetry(); tel.FailOpen != 1 || tel.ActiveLeases != 1 {
		t.Fatalf("telemetry = %+v", tel)
	}
	held.release()
}

// TestInKernelPagedPrefixLedgerWiredForDevicePlannersOnly pins the constructor seam: a
// device backend with radix reuse gets the ledger; a CPU planner, a disabled tree, or
// FAK_INKERNEL_PAGED_PREFIX=off does not.
func TestInKernelPagedPrefixLedgerWiredForDevicePlannersOnly(t *testing.T) {
	m := model.NewSynthetic(tinyCfg())
	tree := radixkv.New(160)
	if l := inKernelPagedPrefixLedgerFor(m, compute.Default(), tree); l == nil {
		t.Fatal("device-backed planner with radix reuse got no paged prefix ledger")
	} else if got := l.pool.Pool().Config().TotalPages; got != 10 {
		t.Fatalf("ledger pages = %d, want 10 (radix budget 160 / 16)", got)
	}
	if inKernelPagedPrefixLedgerFor(m, nil, tree) != nil {
		t.Fatal("CPU planner got a device paged prefix ledger")
	}
	if inKernelPagedPrefixLedgerFor(m, compute.Default(), nil) != nil {
		t.Fatal("planner without radix reuse got a paged prefix ledger")
	}
	t.Setenv("FAK_INKERNEL_PAGED_PREFIX", "off")
	if inKernelPagedPrefixLedgerFor(m, compute.Default(), tree) != nil {
		t.Fatal("FAK_INKERNEL_PAGED_PREFIX=off did not disable the ledger")
	}
}

// TestInKernelPlannerRequestsLeasePagedPrefixPages drives the production request path
// (generateReusedContextWithBias): each reuse-eligible request takes a page lease, a
// shared-prefix second request binds the first request's prefix pages, and every lease
// is released when the request returns.
func TestInKernelPlannerRequestsLeasePagedPrefixPages(t *testing.T) {
	t.Setenv("FAK_INKERNEL_RADIX", "on")
	cfg := tinyCfg()
	p := NewInKernelPlanner(model.NewSynthetic(cfg), nil, "synthetic", false, nil, false)
	p.quant = false
	l, err := newInKernelPagedPrefixLedger(cfg, 64, 16)
	if err != nil {
		t.Fatal(err)
	}
	p.pagedPrefix = l
	sys := synthIDs(cfg.VocabSize, 32, 1)
	for i, suffix := range [][]int{synthIDs(cfg.VocabSize, 8, 2), synthIDs(cfg.VocabSize, 10, 3)} {
		ids := append(append([]int{}, sys...), suffix...)
		if _, _, _, _, _, _, _, _, err := p.generateReusedContextWithBias(context.Background(), ids, 2, 0, 0, 0, nil, 0, 0, map[int]bool{}, nil); err != nil {
			t.Fatalf("request %d: %v", i, err)
		}
	}
	tel, ok := p.PagedPrefixTelemetry()
	if !ok {
		t.Fatal("PagedPrefixTelemetry reported no ledger")
	}
	if tel.Leases != 2 || tel.ActiveLeases != 0 || tel.SharedPages != 2 || tel.AllocatedPages != 4 {
		t.Fatalf("planner paged prefix telemetry = %+v, want 2 leases, 0 active, 2 shared, 4 allocated", tel)
	}
}
