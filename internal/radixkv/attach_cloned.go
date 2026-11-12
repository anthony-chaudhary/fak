package radixkv

import "github.com/anthony-chaudhary/fak/internal/model"

// SharedDensePrefixes reports whether the shared-dense-prefix qualification
// scheme is armed on this tree. Callers that pre-clone payloads outside the
// tree lock must refuse that fast path when it is: the dense-owner materializer
// consumes the UNCLONED session cache and builds its own owned rows.
func (t *Tree) SharedDensePrefixes() bool { return t.sharedDensePrefixes }

// Evictions reports the LRU leaf-eviction counter without walking the tree.
// The attribution seams diff it around an admission to see how much
// budget-pressure bookkeeping the insert triggered.
func (t *Tree) Evictions() int { return t.evictions }

// ReserveCPUAdmission checks retained-storage room for an incoming full-prefix
// payload of payloadBytes plus logitsLen floats, evicting LRU leaves under the
// caller-held lock when a byte budget is armed. It is the off-lock companion
// of InsertCloneWithLogits' room check: the planner reserves room, releases the
// lock, clones the session cache, and re-acquires to AttachCloned. A second
// room check inside AttachCloned remains authoritative — another admission may
// have consumed the reserved room while the lock was released. An unbounded
// tree (maxCPUCacheBytes == 0) always fits.
func (t *Tree) ReserveCPUAdmission(payloadBytes int64, logitsLen int, boundary *node) bool {
	if t.maxCPUCacheBytes == 0 {
		return true
	}
	return t.makeCPUCacheRoom(cacheByteSum(payloadBytes, cpuLogitsBytes(logitsLen)), 0, boundary)
}

// AttachCloned attaches a caller-provided CLONE of the full-prefix KV payload to
// boundary, keyed by suffix, with the standard lease handoff, admission sketch,
// and budget enforcement. It is the off-lock companion of InsertCloneWithLogits
// for the demand-admission seam (ticket prefill-radix-cold-prefill-tax-20261108):
// the planner clones the request-owned session cache OUTSIDE the tree lock — the
// session is owned by the requesting goroutine until decode starts, so the copy
// is stable there — re-validates the lookup boundary under the lock, and hands
// this pre-cloned payload in so the locked section holds only the boundary walk,
// the attach, and the bookkeeping, never the O(prefix-bytes) copy.
//
// Contract versus InsertCloneWithLogits:
//   - suffix must be NON-EMPTY. The exact-hit logits-only update (empty suffix)
//     stays on the standard path; a misuse returns the boundary with its lease
//     intact and attaches nothing.
//   - cloned is the full-prefix payload the caller cloned from its session
//     cache; it is stored as-is (no second clone).
//   - the shared-dense-prefix scheme refuses the fast path (returns the
//     boundary, lease intact) because its materializer needs the uncloned
//     session cache; the caller falls back to InsertCloneWithLogits.
//   - on budget refusal (oversized, or full with everything pinned) the
//     boundary lease is preserved and the boundary returned, exactly like
//     InsertCloneWithLogits; the caller owns the unused clone and may drop it.
func (t *Tree) AttachCloned(boundary *node, suffix []int, cloned *model.KVCache, logits []float32) *node {
	if boundary == nil || len(suffix) == 0 || t.sharedDensePrefixes {
		return boundary
	}
	n, _ := t.insertWithLogitsAndChunk(boundary, suffix, cloned, logits, nil, 0)
	return n
}
