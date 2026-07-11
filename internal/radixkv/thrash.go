package radixkv

// thrash.go — the evict→reuse-gap thrash detector (#3393): a bounded, aged "just-evicted"
// side-map that makes PREMATURE eviction visible. The eviction plane already records what
// it evicts (recordEvictChoice keeps the LAST victim's stats), but nothing ever compares a
// victim against a LATER demand for the same prefix — so a key evicted under budget
// pressure and re-inserted moments later (thrash: the budget/policy was too aggressive) is
// invisible. This file closes that loop retrospectively, technique-inspired (clean-room)
// by LMCache's l1_lifecycle evict→reuse-gap subscriber.
//
// Mechanism — OBSERVE only, never steer. Victim choice is untouched: on every
// budget-pressure eviction (evictToBudget; POLICY evictions via EvictNode/EvictPrefix are
// deliberately excluded — a quarantined prefix returning is not a budget signal),
// noteEviction records the victim's full-prefix identity into a side-map stamped with the
// tree's existing logical clock (t.clock — the monotonic op counter Lookup/Insert already
// advance; no wall clock anywhere). On a later demand for EXACTLY that token path — a
// Lookup of the full path, or an attachLeaf re-creating it (Insert / WarmInsert) — the
// probe consumes the entry: within the window it counts one thrash and records the
// evict→reuse gap (current clock − evictedAt, in logical ticks); past the window it is a
// cold miss, not a thrash. An entry is consumed at most ONCE.
//
// BOUND (the documented contract, guarded by TestThrashSideMapBounded): the side-map holds
// at most thrashCap records — a FIFO ring of the most-recent budget evictions — and an
// entry older than thrashWindow logical ticks no longer counts as a thrash. Overflow ages
// out the OLDEST record (a re-insert of an aged-out key is a cold miss). Expired/overflow
// entries are swept lazily on record and on probe; nothing is ever scanned wholesale, and
// the hot path pays nothing while the side-map is empty (the probe is gated on len before
// any hashing happens).
//
// Identity: the tree keys nodes by token-id paths, so the side-map keys on an FNV-1a
// 64-bit fold of the victim's full root→leaf token path (O(cap) memory regardless of
// prefix length). A hash collision could at worst count one spurious thrash — this is an
// observability counter that never feeds victim choice, so exactness is traded for a hard
// memory bound. Matching is exact-path: a request that merely EXTENDS an evicted prefix is
// not detected (leaf-exact identity, same as the eviction grain).

const (
	// thrashCap is the hard entry bound on the just-evicted side-map: only the thrashCap
	// most-recent budget evictions are probe-able. Overflow pops the oldest record.
	thrashCap = 256
	// thrashWindow is the maximum evict→reuse gap, in logical-clock ticks, still counted
	// as a thrash. A reuse with gap > thrashWindow consumes the entry as a cold miss —
	// the eviction had a full window of operations to be proven premature and was not.
	thrashWindow uint64 = 4096
)

// fnv64Offset / fnv64Prime are the FNV-1a 64-bit parameters (inlined so the hot path
// needs no hash.Hash allocation per probe).
const (
	fnv64Offset uint64 = 14695981039346656037
	fnv64Prime  uint64 = 1099511628211
)

// thrashRecord is one just-evicted entry: which prefix (by path hash), when (logical
// clock at eviction), and how big (the victim's edge tokens — the recompute work that
// proves wasted if the key returns within the window).
type thrashRecord struct {
	keyHash   uint64
	evictedAt uint64
	tokens    int
}

// foldTokens extends an FNV-1a 64-bit hash over a run of token ids (each folded as 8
// little-endian bytes, so the fold over consecutive edge keys equals the fold over the
// concatenated path).
func foldTokens(h uint64, tokens []int) uint64 {
	for _, tok := range tokens {
		v := uint64(tok)
		for i := 0; i < 8; i++ {
			h ^= v & 0xff
			h *= fnv64Prime
			v >>= 8
		}
	}
	return h
}

// pathHash is the side-map identity of the token path root→n plus an optional suffix:
// the FNV-1a fold of every edge key in root→leaf order. pathHash(t.root, tokens) is
// therefore the identity of a raw request path, and pathHash(v, nil) of a resident node —
// the two forms the record and probe sites use, guaranteed to agree because both fold the
// same token sequence.
func pathHash(n *node, suffix []int) uint64 {
	var chain []*node
	for p := n; p != nil && p.parent != nil; p = p.parent {
		chain = append(chain, p)
	}
	h := fnv64Offset
	for i := len(chain) - 1; i >= 0; i-- {
		h = foldTokens(h, chain[i].key)
	}
	return foldTokens(h, suffix)
}

// noteEviction records a budget-pressure victim into the just-evicted side-map, stamped
// with the current logical clock. Called by evictToBudget for every leaf it removes —
// and ONLY there: EvictNode/EvictPrefix (policy eviction) never record, so a quarantined
// prefix returning is never miscounted as budget thrash. Before pushing, it lazily sweeps
// the ring head of expired records and, at capacity, ages out the oldest — enforcing the
// thrashCap bound.
func (t *Tree) noteEviction(v *node) {
	if t.thrashRing == nil {
		t.thrashRing = make([]thrashRecord, thrashCap)
		t.thrashIndex = make(map[uint64]thrashRecord, thrashCap)
	}
	for t.thrashCount > 0 {
		head := t.thrashRing[t.thrashHead]
		if t.thrashCount == thrashCap || t.clock-head.evictedAt > thrashWindow {
			t.popThrashHead()
			continue
		}
		break
	}
	rec := thrashRecord{keyHash: pathHash(v, nil), evictedAt: t.clock, tokens: len(v.key)}
	t.thrashRing[(t.thrashHead+t.thrashCount)%thrashCap] = rec
	t.thrashCount++
	// The index keeps only the LATEST record per key (a key re-evicted before reuse
	// supersedes its older record; the stale ring slot is popped harmlessly later).
	t.thrashIndex[rec.keyHash] = rec
}

// popThrashHead ages the oldest ring record out, dropping its index entry only when the
// index still points at THIS record (a newer re-eviction of the same key, or a consuming
// probe, may already have superseded/removed it).
func (t *Tree) popThrashHead() {
	head := t.thrashRing[t.thrashHead]
	t.thrashRing[t.thrashHead] = thrashRecord{}
	t.thrashHead = (t.thrashHead + 1) % thrashCap
	t.thrashCount--
	if cur, ok := t.thrashIndex[head.keyHash]; ok && cur.evictedAt == head.evictedAt {
		delete(t.thrashIndex, head.keyHash)
	}
}

// probeThrash checks whether the token path (root→boundary)+suffix is a just-evicted key
// coming back. Called on the demand paths — Lookup (full request path) and attachLeaf
// (the new leaf's full path, covering Insert and WarmInsert). On a live hit within
// thrashWindow it counts one thrash and accumulates the evict→reuse gap; past the window
// the entry is dropped as a cold miss. Either way the entry is CONSUMED — one eviction
// can be regretted at most once. The len gate keeps the un-armed hot path free: no
// hashing happens while the side-map is empty.
func (t *Tree) probeThrash(boundary *node, suffix []int) {
	if len(t.thrashIndex) == 0 {
		return
	}
	h := pathHash(boundary, suffix)
	rec, ok := t.thrashIndex[h]
	if !ok {
		return
	}
	delete(t.thrashIndex, h)
	gap := t.clock - rec.evictedAt
	if gap > thrashWindow {
		return // aged out: the key returned, but not soon enough to call the eviction premature
	}
	t.thrashReuses++
	t.thrashTokens += rec.tokens
	t.thrashGapTotal += gap
	t.thrashGapLast = gap
	if gap > t.thrashGapMax {
		t.thrashGapMax = gap
	}
}
