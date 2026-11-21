package ctxmmu

// eviction_page_aware.go implements the unreferenced-only session eviction
// rule (M3): a referenced session's warm prefix is never paged out or evicted
// while cold (unreferenced) sessions' KV can still be reclaimed. Under memory
// pressure the pool reclaims only blocks whose refCount <= 0 — a released
// session's KV — oldest-first, and stops at the first referenced block. A
// session actively holding warm-prefix KV (refCount > 0, routed per #58)
// therefore keeps its pages resident even when the pressure exceeds the
// unreferenced supply, so warm-TTFT p99 is held under eviction pressure.

// EvictUnreferencedSessions reclaims up to targetBlocks physical blocks under
// memory pressure, but ONLY from unreferenced sessions (blocks with
// refCount <= 0, i.e. a released session's KV). It NEVER swaps out or evicts a
// page belonging to a still-referenced session (refCount > 0) — even when the
// pressure exceeds the unreferenced supply. It returns the number of blocks
// reclaimed, capped at the unreferenced supply; when only referenced sessions
// remain it returns fewer than targetBlocks and leaves their warm KV resident.
func (p *KVPool) EvictUnreferencedSessions(targetBlocks int) (int, error) {
	if p == nil || p.allocator == nil || targetBlocks <= 0 {
		return 0, nil
	}
	reclaimed, err := p.allocator.RecycleLRU(targetBlocks)
	return len(reclaimed), err
}
