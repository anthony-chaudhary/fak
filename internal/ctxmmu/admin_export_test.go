package ctxmmu_test

import (
	"context"
	"fmt"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	_ "github.com/anthony-chaudhary/fak/internal/blob" // registers the "blob" PageOut/Resolver backend
	"github.com/anthony-chaudhary/fak/internal/ctxmmu"
)

// admin_export_test.go - dead-island revival proofs for the ctxmmu admin export:
// zero-value/nil safety, counter monotonicity after forced ledger eviction, and
// the paged-store fold failing safe on nil receivers.

// TestSnapshotPagingZeroValueSafety covers the nil and unused-MMU guards.
func TestSnapshotPagingZeroValueSafety(t *testing.T) {
	var nilMMU *ctxmmu.MMU
	got := ctxmmu.SnapshotPaging(nilMMU)
	if got.HeldLen != 0 || got.Evicted != 0 || got.Total != 0 {
		t.Fatalf("nil MMU snapshot = %+v, want zero", got)
	}
	fresh := ctxmmu.SnapshotPaging(ctxmmu.NewWithLimit(4))
	if fresh.HeldLen != 0 || fresh.Evicted != 0 {
		t.Fatalf("fresh MMU snapshot = %+v, want zero ledger", fresh)
	}
	if fresh.Quarantined != 0 || fresh.Paged != 0 {
		t.Fatalf("fresh MMU counters must be zero, got %+v", fresh)
	}
}

// TestSnapshotPagingCounterMonotonicAfterEvict drives enough distinct quarantines
// through Admit to force FIFO ledger eviction, then asserts the exported counters
// moved strictly forward: HeldLen plateaus at the cap, Evicted accounts every drop,
// Quarantined reflects every hold, and re-snapshotting never rolls anything back.
func TestSnapshotPagingCounterMonotonicAfterEvict(t *testing.T) {
	ctx := context.Background()
	const cap = 4
	const n = 20
	m := ctxmmu.NewWithLimit(cap)

	before := ctxmmu.SnapshotPaging(m)
	for i := 0; i < n; i++ {
		c := call("read_file")
		r := result(c, poisonAdmin(i))
		if v := m.Admit(ctx, c, r); v.Kind != abi.VerdictQuarantine {
			t.Fatalf("admit %d: want Quarantine, got %v", i, v.Kind)
		}
	}
	after := ctxmmu.SnapshotPaging(m)

	if before.Evicted != 0 {
		t.Fatalf("before.Evicted = %d, want 0", before.Evicted)
	}
	if after.HeldLen != cap {
		t.Fatalf("after.HeldLen = %d, want plateau at cap %d", after.HeldLen, cap)
	}
	if want := int64(n - cap); after.Evicted != want {
		t.Fatalf("after.Evicted = %d, want %d (n-cap)", after.Evicted, want)
	}
	if after.Quarantined != int64(n) {
		t.Fatalf("after.Quarantined = %d, want %d", after.Quarantined, n)
	}
	if after.Total != int64(n) {
		t.Fatalf("after.Total = %d, want %d", after.Total, n)
	}
	if after.Cleared != 0 {
		t.Fatalf("after.Cleared = %d, want 0 (no witness Clears in this test)", after.Cleared)
	}
	// Read-only: re-snapshotting is stable and never rolls counters back.
	again := ctxmmu.SnapshotPaging(m)
	if again.Evicted != after.Evicted || again.HeldLen != after.HeldLen {
		t.Fatalf("snapshot mutated ledger: %+v then %+v", after, again)
	}
}

// TestSnapshotPagingReflectsClear marks a held id cleared and checks the Cleared
// export follows (cleared stays observable as a subset of held).
func TestSnapshotPagingReflectsClear(t *testing.T) {
	ctx := context.Background()
	m := ctxmmu.NewWithLimit(4)
	c := call("read_file")
	r := result(c, poisonAdmin(0))
	if v := m.Admit(ctx, c, r); v.Kind != abi.VerdictQuarantine {
		t.Fatalf("want Quarantine, got %v", v.Kind)
	}
	m.Clear(r.Meta["quarantine_id"])
	got := ctxmmu.SnapshotPaging(m)
	if got.HeldLen != 1 || got.Cleared != 1 {
		t.Fatalf("post-clear snapshot = %+v, want HeldLen 1 Cleared 1", got)
	}
}

// TestSnapshotPagingStoreZeroValueSafety proves the standalone paged-store fold
// is nil-safe and tracks Stage/Get/eviction counters on a real store. Keyed with
// non-64-hex digests so the integrity path stays inert and this test exercises
// only the staging counters (integrity is paged_store_test.go's job).
func TestSnapshotPagingStoreZeroValueSafety(t *testing.T) {
	var nilStore *ctxmmu.PagedStore
	if got := ctxmmu.SnapshotPagingStore(nilStore); got.StagedEntries != 0 || got.StagingBytes != 0 {
		t.Fatalf("nil store snapshot = %+v, want zero", got)
	}
	ps := ctxmmu.NewPagedStore(256, 4)
	if got := ctxmmu.SnapshotPagingStore(ps); got.StagedEntries != 0 {
		t.Fatalf("fresh store snapshot = %+v, want zero", got)
	}
	payload := []byte("payload bytes for staged-entry counting")
	for i := 0; i < 8; i++ {
		ps.Stage(fmt.Sprintf("digest-key-%03d-not-hex", i), payload)
	}
	got := ctxmmu.SnapshotPagingStore(ps)
	if got.StagedEntries != 4 {
		t.Fatalf("StagedEntries = %d, want plateau at 4 (entry bound)", got.StagedEntries)
	}
	if got.StagedEvicted != 4 {
		t.Fatalf("StagedEvicted = %d, want 4", got.StagedEvicted)
	}
	if got.StagingBytes <= 0 {
		t.Fatalf("StagingBytes = %d, want > 0", got.StagingBytes)
	}
	if _, ok := ps.Get("digest-key-007-not-hex"); !ok {
		t.Fatal("freshest entry should still resolve")
	}
	if got2 := ctxmmu.SnapshotPagingStore(ps); got2.StagedHits != 1 {
		t.Fatalf("StagedHits = %d, want 1", got2.StagedHits)
	}
}

// poisonAdmin mirrors leak_test.go poison: a body that trips ScreenBytes with a
// unique tail so every quarantine is a distinct payload.
func poisonAdmin(i int) []byte {
	return []byte(fmt.Sprintf("api_key=sk-abcdef0123456789abcdef%010d leaked", i))
}
