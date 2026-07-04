package compute

// kvcost_aging.go — the GDSF aging-clock term #2668 adds to #2239's cost-of-losing
// function: fak's KVEvictionCost is exactly GDSF's `(Frequency × Cost) ÷ Size` term but is
// missing GDSF's `Clock` (L) term, so a span that was hot LONG AGO (high Hits, stale
// LastUsed) keeps a permanently high cost-of-losing and holds memory forever against a
// scan of fresh one-shot spans — LastUsed only breaks ties (kvcost.go), never enters the
// score. This file adds the aging term as a pure extension of KVSpanStats/KVEvictionCost.
//
// WHY A PER-SPAN STAMP, NOT A SHARED SCALAR CLOCK. GDSF's L is a single monotonically
// increasing pool-wide clock, but a candidate's PRIORITY is computed and FROZEN at the
// clock value in effect the last time that candidate was referenced — not recomputed
// against the CURRENT clock at ranking time. Adding the same current L to every
// candidate's cost at ranking time would cancel out of the comparison entirely (it is a
// shared additive constant), so it could never change which span ranks lowest — the
// aging term would be provably inert. AgeStamp is that frozen per-span L: the resident
// pool stamps it with its current AgingClock on every reference (insert or hit), so a
// span untouched since an old, low clock value keeps a low stamp while spans referenced
// more recently carry a higher one — exactly what lets a stale-but-historically-hot span
// become the cheapest-to-lose once enough intervening evictions have inflated the clock
// past it.

// KVEvictionCostAged is KVEvictionCost plus the span's frozen aging-clock stamp:
//
//	cost = AgeStamp + KVEvictionCost(s)
//
// AgeStamp is the pool's AgingClock (L) as of this span's last reference (see
// KVSpanStats.AgeStamp). A span with AgeStamp == 0 (never stamped, or the pool does not
// age) scores exactly KVEvictionCost(s) — the reduction: an all-zero-stamp resident set
// ranks byte-identically to the de-aged function, so this is a strict generalization,
// never a divergence, on today's inputs. The existing +Inf fail-open on unknown Bytes
// (KVEvictionCost) is preserved automatically: AgeStamp (always finite/non-negative in
// practice) plus +Inf is still +Inf, so an unknown-footprint span never becomes cheaper
// merely by being long unstamped.
func KVEvictionCostAged(s KVSpanStats) float64 {
	return s.AgeStamp + KVEvictionCost(s)
}

// PickEvictionVictimAged is PickEvictionVictim scored by KVEvictionCostAged instead of
// KVEvictionCost: same Pinned/Leased hard exclusions, same LastUsed tie-break, same -1
// no-candidate signal. It additionally returns newClock, the AgingClock (L) the caller's
// resident pool should carry forward: the chosen victim's OWN aged cost at the moment it
// was evicted — GDSF's rule that L is raised to the evicted item's priority on every
// eviction. When no span is evictable, newClock is 0 (nothing to advance the clock to);
// callers should treat idx == -1 as "ignore newClock", mirroring PickEvictionVictim's -1
// contract.
//
// Threading newClock forward and re-stamping AgeStamp on every subsequent
// insert/hit (both on the CALLER's side — this package holds no state, per kvcost.go's
// discipline) is the R2 wiring into radixkv/the native preemptor; this function is the
// pure R1 decision primitive that wiring drives.
func PickEvictionVictimAged(spans []KVSpanStats) (idx int, newClock float64) {
	bestIdx := -1
	var bestCost float64
	for i := range spans {
		s := spans[i]
		if s.Pinned || s.Leased {
			continue
		}
		c := KVEvictionCostAged(s)
		switch {
		case bestIdx == -1:
			bestIdx, bestCost = i, c
		case c < bestCost:
			bestIdx, bestCost = i, c
		case c == bestCost && s.LastUsed < spans[bestIdx].LastUsed:
			// Equal aged cost → break the tie by recency (oldest first), matching
			// PickEvictionVictim's tie-break so a uniform-stamp resident set reduces
			// to the same ranking as the de-aged picker.
			bestIdx = i
		}
	}
	if bestIdx == -1 {
		return -1, 0
	}
	return bestIdx, bestCost
}
