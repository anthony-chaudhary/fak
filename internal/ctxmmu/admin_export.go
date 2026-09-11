package ctxmmu

import "sync/atomic"

// admin_export.go - typed admin export for the context MMU: pure, read-only
// projections of the counters that already exist on MMU and its backing
// PagedStore (mmu.go Evicted/HeldLen/Screened/Digested/PollutionRate,
// paged_store.go Stats). Nothing here mutates engine state, adds work to the
// serving path, or widens the quarantine gate; it only folds the existing
// public accessors into one value operators can serialize. Nil and zero-value
// receivers answer with a zero snapshot instead of panicking.

// AdminPagingSnapshot is a point-in-time view of the MMU quarantine/paging
// state for cache-admin tooling. The Quarantine block mirrors mmu.go's
// bounded held ledger; the Results block mirrors the admission counters; the
// Staging block mirrors the backing PagedStore read-through store
// (paged_store.go). Worth reading together: HeldLen plateaus at the maxHeld
// cap while Evicted climbs - the leak-fix observability contract.
type AdminPagingSnapshot struct {
	// Quarantine ledger (bounded by maxHeld; oldest entries evicted FIFO).
	HeldLen  int   // quarantined results currently held (== len(held))
	Evicted  int64 // held entries dropped by the maxHeld bound over the lifetime
	Cleared  int   // held ids that have passed a witness Clear()
	Screened int64 // results additively quarantined by a registered SemanticScreen
	Digested int64 // oversize results paged out to a digest-bearing stub

	// Result-admission counters.
	Total       int64 // results admitted through the gate
	Quarantined int64 // results held out of context
	Paged       int64 // results paged out to pointer stubs

	// Read-through staging store (paged_store.go Stats()).
	StagedHits    int64 // staging hits
	StagedMisses  int64 // staging misses
	StagedEntries int64 // bodies currently resident in staging
	StagedStaged  int64 // bodies staged over the lifetime
	StagedEvicted int64 // bodies evicted from staging (byte/entry bound)
	StagingBytes  int64 // resident staged bytes
}

// SnapshotPaging folds the MMU quarantine/paging ledgers and its backing
// PagedStore counters into an AdminPagingSnapshot. It reads through the same
// accessors the recall/serving surfaces use (Held, Cleared, HeldLen, Evicted,
// PollutionRate) plus the paged-store Stats triple - no new locks, no
// mutation, no hot-path work. Copied maps make the read defensive: a caller
// can never mutate the live gate. A nil or zero-value MMU yields a zero
// snapshot, never a panic.
func SnapshotPaging(m *MMU) AdminPagingSnapshot {
	if m == nil {
		return AdminPagingSnapshot{}
	}
	q, total, _ := m.PollutionRate()
	out := AdminPagingSnapshot{
		HeldLen:     m.HeldLen(),
		Evicted:     m.Evicted(),
		Cleared:     len(m.Cleared()),
		Screened:    m.Screened(),
		Digested:    m.Digested(),
		Total:       total,
		Quarantined: q,
		Paged:       atomic.LoadInt64(&m.paged),
	}
	out.StagingBytes, out.StagedHits, out.StagedMisses, out.StagedEntries,
		out.StagedStaged, out.StagedEvicted = snapshotBackingStore(m.stagingStore())
	return out
}

// SnapshotPagingStore folds a standalone PagedStore into the same snapshot
// shape, with only the staging block populated. Nil/zero-value safe.
func SnapshotPagingStore(ps *PagedStore) AdminPagingSnapshot {
	if ps == nil {
		return AdminPagingSnapshot{}
	}
	var out AdminPagingSnapshot
	out.StagingBytes, out.StagedHits, out.StagedMisses, out.StagedEntries,
		out.StagedStaged, out.StagedEvicted = snapshotBackingStore(ps)
	return out
}

// snapshotBackingStore folds one PagedStore into the flat staging fields.
// Stats() already guards a nil receiver, but the explicit triage here keeps
// the sentinel zero-values exact for a nil store handed in directly.
func snapshotBackingStore(ps *PagedStore) (bytes, hits, misses, entries, staged, evicted int64) {
	if ps == nil {
		return
	}
	h, m, s, e := ps.Stats()
	return ps.Bytes(), h, m, int64(ps.Len()), s, e
}
