package compute

import (
	"math"
	"testing"
)

// TestKVEvictionCostUniformBytesReducesToReuse proves the algebraic reduction in
// KVEvictionCost's doc: when Bytes is proportional to Tokens (one precision tier, uniform
// per-token byte cost), the byte/Tokens factors cancel and cost reduces to reuse alone —
// i.e. under uniform memory cost the evictor ranks spans purely by how often they were
// reused. Bytes only changes the ranking when spans carry DIFFERENT per-token costs.
func TestKVEvictionCostUniformBytesReducesToReuse(t *testing.T) {
	// Two spans at the SAME uniform per-token byte cost (4 bytes/token).
	long := KVSpanStats{Tokens: 100, Bytes: 400, Hits: 0} // cost = 100*1/400 = 0.25
	short := KVSpanStats{Tokens: 10, Bytes: 40, Hits: 0}  // cost = 10*1/40  = 0.25
	if cl, cs := KVEvictionCost(long), KVEvictionCost(short); cl != cs {
		t.Fatalf("uniform per-token cost must make length irrelevant: long=%v short=%v", cl, cs)
	}
	// Reuse dominates under uniform bytes: a hit span outranks a never-reused one regardless
	// of length. longReused (100 tok, 3 hits) vs shortOneShot (10 tok, 0 hits).
	longReused := KVSpanStats{Tokens: 100, Bytes: 400, Hits: 3} // 100*4/400 = 1.0
	shortOneShot := KVSpanStats{Tokens: 10, Bytes: 40, Hits: 0} // 10*1/40  = 0.25
	if got := KVEvictionCost(longReused); got != 1.0 {
		t.Fatalf("longReused cost = %v, want 1.0", got)
	}
	if got := KVEvictionCost(shortOneShot); got != 0.25 {
		t.Fatalf("shortOneShot cost = %v, want 0.25", got)
	}
	if KVEvictionCost(shortOneShot) >= KVEvictionCost(longReused) {
		t.Fatal("a one-shot span must be cheaper-to-lose than a reused span under uniform bytes")
	}
}

// TestKVEvictionCostCompactTierIsCheaperToKeep proves the byte-normalization matters: two
// spans of the SAME token length and SAME reuse, but one is a demoted/quantized tier
// (fewer bytes/token). The compact span has a HIGHER cost-of-losing → the evictor keeps it
// and prefers evicting the fat span. This is the value-aware distinction recency-only
// eviction cannot make and the reason #2239 divides by bytes.
func TestKVEvictionCostCompactTierIsCheaperToKeep(t *testing.T) {
	fat := KVSpanStats{Tokens: 50, Bytes: 400, Hits: 1}     // 50*2/400 = 0.25
	compact := KVSpanStats{Tokens: 50, Bytes: 100, Hits: 1} // 50*2/100 = 1.0
	if KVEvictionCost(compact) <= KVEvictionCost(fat) {
		t.Fatalf("a compact (demoted) span must be cheaper to KEEP per unit of recompute value: compact=%v fat=%v",
			KVEvictionCost(compact), KVEvictionCost(fat))
	}
}

// TestKVEvictionCostFailOpenOnUnknownBytes: a span of unknown footprint (Bytes <= 0) scores
// +Inf so it is never preferred as a victim — dropping a span of unknown size is conservative.
func TestKVEvictionCostFailOpenOnUnknownBytes(t *testing.T) {
	if got := KVEvictionCost(KVSpanStats{Tokens: 100, Bytes: 0, Hits: 5}); !math.IsInf(got, 1) {
		t.Fatalf("unknown footprint should score +Inf (never the preferred victim), got %v", got)
	}
	if got := KVEvictionCost(KVSpanStats{Tokens: 100, Bytes: -1, Hits: 5}); !math.IsInf(got, 1) {
		t.Fatalf("negative footprint should score +Inf, got %v", got)
	}
}

// TestPickEvictionVictimPrefersCheapestToLose: among three unpinned/unleased spans the
// picker returns the LOWEST cost (cheapest to lose), not the oldest — the core behavioral
// difference from pure LRU.
func TestPickEvictionVictimPrefersCheapestToLose(t *testing.T) {
	spans := []KVSpanStats{
		{Tokens: 100, Bytes: 400, Hits: 3, LastUsed: 10}, // cost 1.0 — expensive to lose, keep
		{Tokens: 10, Bytes: 40, Hits: 0, LastUsed: 5},    // cost 0.25 — cheapest, victim
		{Tokens: 50, Bytes: 400, Hits: 1, LastUsed: 1},   // cost 0.25 — ties on cost, but older LastUsed
	}
	idx := PickEvictionVictim(spans)
	if idx != 2 {
		t.Fatalf("victim = span %d, want 2 (cheapest cost 0.25, oldest LastUsed on the tie)", idx)
	}
}

// TestPickEvictionVictimReducesToLRUOnUniformCost: when every span has the same
// cost-of-losing (uniform bytes, equal reuse), the picker degenerates to pure LRU — it
// picks the oldest LastUsed. Cost-aware eviction is a strict generalization of LRU, never a
// divergence on equal-cost inputs.
func TestPickEvictionVictimReducesToLRUOnUniformCost(t *testing.T) {
	spans := []KVSpanStats{
		{Tokens: 10, Bytes: 40, Hits: 0, LastUsed: 9},
		{Tokens: 10, Bytes: 40, Hits: 0, LastUsed: 3}, // oldest — LRU victim
		{Tokens: 10, Bytes: 40, Hits: 0, LastUsed: 7},
	}
	if idx := PickEvictionVictim(spans); idx != 1 {
		t.Fatalf("uniform cost must reduce to LRU: victim = %d, want 1 (oldest LastUsed)", idx)
	}
}

// TestPickEvictionVictimSkipsPinnedAndLeased: a pinned (#2211/#2225) or in-flight-leased
// (#2225) span is a HARD exclusion — never a victim, regardless of how cheap it is to
// lose. Cost-aware eviction never violates a pin or an in-flight lease.
func TestPickEvictionVictimSkipsPinnedAndLeased(t *testing.T) {
	spans := []KVSpanStats{
		{Tokens: 10, Bytes: 40, Hits: 0, LastUsed: 1, Pinned: true}, // cheapest but pinned — skip
		{Tokens: 10, Bytes: 40, Hits: 0, LastUsed: 2, Leased: true}, // cheapest-but-one, leased — skip
		{Tokens: 100, Bytes: 40, Hits: 5, LastUsed: 9},              // expensive to lose, but only evictable one
	}
	if idx := PickEvictionVictim(spans); idx != 2 {
		t.Fatalf("victim = %d, want 2 (the only non-pinned non-leased span)", idx)
	}
}

// TestPickEvictionVictimReturnsMinusOneWhenAllLocked: every resident span pinned or leased
// — the budget-excess is fully locked, nothing can be reclaimed. Mirrors radixkv.lruLeaf
// returning nil so evictToBudget stops.
func TestPickEvictionVictimReturnsMinusOneWhenAllLocked(t *testing.T) {
	spans := []KVSpanStats{
		{Tokens: 10, Bytes: 40, Hits: 0, LastUsed: 1, Pinned: true},
		{Tokens: 10, Bytes: 40, Hits: 0, LastUsed: 2, Leased: true},
	}
	if idx := PickEvictionVictim(spans); idx != -1 {
		t.Fatalf("all-locked resident set must yield no victim, got %d", idx)
	}
	if idx := PickEvictionVictim(nil); idx != -1 {
		t.Fatalf("empty resident set must yield no victim, got %d", idx)
	}
}

// TestReplayCostAwareBeatsLRUOnHotSpanTrace is the #2239 witness: a replayed agent session
// where a HOT, frequently-reused span competes for a tight budget against one-shot COLD
// spans. Pure-LRU evicts by recency alone, so once the hot span is the oldest resident it
// gets evicted and its subsequent accesses are full misses (recompute). Cost-aware eviction
// reads the reuse signal and keeps the hot span resident, turning those accesses into hits.
// At the SAME fixed budget, cost-aware scores a STRICTLY higher hit rate than LRU. This is
// the "eviction-pressure trace where cost-aware beats LRU" the issue's witness checkbox
// names; the simulator (ReplayKVCache) is the committed artifact.
func TestReplayCostAwareBeatsLRUOnHotSpanTrace(t *testing.T) {
	trace, spanTokens, budget := hotColdKVReplayTrace()
	lruHits, accessed := ReplayKVCache(trace, budget, KVEvictLRU)
	costHits, _ := ReplayKVCache(trace, budget, KVEvictCostAware)

	if accessed == 0 {
		t.Fatal("trace accessed zero tokens — simulator did not run")
	}
	if lruHits >= costHits {
		t.Fatalf("cost-aware must STRICTLY beat LRU on a hot/cold trace at fixed budget: LRU hits=%d cost-aware hits=%d (accessed=%d)",
			lruHits, costHits, accessed)
	}
	// The divergence point is t6: cost-aware hits HOT, LRU misses it. So cost-aware must
	// have at least one more hot-span hit than LRU.
	if minExtra := spanTokens; costHits-lruHits < minExtra {
		t.Fatalf("cost-aware lead (%d) below the single t6 hot-hit divergence (%d): LRU=%d cost=%d",
			costHits-lruHits, minExtra, lruHits, costHits)
	}
}

func TestReplayKVCacheMultiScoresAgainstBeladyOracle(t *testing.T) {
	trace, _, budget := hotColdKVReplayTrace()
	rows := ReplayKVCacheMulti(trace, budget, KVEvictLRU, KVEvictCostAware)
	oracle := BeladyKVReplayOracle(trace, budget)
	if !oracle.Exact || oracle.HitTokens != 150 {
		t.Fatalf("Belady oracle = %+v, want exact 150 hit tokens", oracle)
	}
	lru := rows[KVEvictLRU]
	cost := rows[KVEvictCostAware]
	if lru.HitTokens != 100 || cost.HitTokens != 150 {
		t.Fatalf("policy hit tokens = LRU %d cost-aware %d, want 100/150", lru.HitTokens, cost.HitTokens)
	}
	if cost.GoodDecisionRatio != 1 {
		t.Fatalf("cost-aware good-decision ratio = %v, want 1 against oracle %+v", cost.GoodDecisionRatio, oracle)
	}
	if lru.GoodDecisionRatio >= cost.GoodDecisionRatio {
		t.Fatalf("LRU good-decision ratio should be below cost-aware: LRU=%+v cost=%+v", lru, cost)
	}
	if cost.EvictionsPerHit >= lru.EvictionsPerHit {
		t.Fatalf("cost-aware stability should beat LRU: LRU=%+v cost=%+v", lru, cost)
	}
}

func TestReplayKVCacheResultMatchesLegacyReplay(t *testing.T) {
	trace, _, budget := hotColdKVReplayTrace()
	for _, policy := range []KVEvictPolicy{KVEvictLRU, KVEvictCostAware} {
		hits, accessed := ReplayKVCache(trace, budget, policy)
		result := ReplayKVCacheResult(trace, budget, policy)
		if result.HitTokens != hits || result.AccessTokens != accessed {
			t.Fatalf("structured replay mismatch for policy %d: result=%+v legacy=%d/%d",
				policy, result, hits, accessed)
		}
		if result.Evictions == 0 {
			t.Fatalf("structured replay did not count evictions for policy %d: %+v", policy, result)
		}
	}
}

func TestBeladyKVReplayOracleUnboundedMatchesAllReuses(t *testing.T) {
	trace := []KVReplayEvent{
		{1, 10},
		{1, 10},
		{2, 20},
		{1, 10},
	}
	oracle := BeladyKVReplayOracle(trace, 0)
	if !oracle.Exact || oracle.HitTokens != 20 || oracle.AccessTokens != 50 {
		t.Fatalf("unbounded oracle = %+v, want exact 20/50", oracle)
	}
}

// TestReplayCostAwareEqualsLRUOnNoReuse confirms the reduction: a trace where no span is
// ever reused gives every span Hits=0, so cost-aware's ranking ties uniformly and
// degenerates to LRU. The two policies must score IDENTICALLY — cost-aware never does
// WORSE than LRU on a trace without hot/cold structure.
func TestReplayCostAwareEqualsLRUOnNoReuse(t *testing.T) {
	// Every span touched exactly once (no reuse). Budget holds 2 of the 3 spans.
	trace := []KVReplayEvent{
		{101, 30},
		{102, 20},
		{103, 50}, // evicts someone (all Hits=0)
		{104, 10}, // evicts someone
	}
	lruHits, accessed := ReplayKVCache(trace, 50, KVEvictLRU)
	costHits, _ := ReplayKVCache(trace, 50, KVEvictCostAware)
	if lruHits != costHits {
		t.Fatalf("on a no-reuse trace cost-aware must equal LRU: LRU=%d cost-aware=%d (accessed=%d)", lruHits, costHits, accessed)
	}
	if accessed == 0 {
		t.Fatal("trace accessed zero tokens — simulator did not run")
	}
}

// TestReplayUnboundedBudgetYieldsAllHits: a zero (unbounded) budget disables eviction, so
// after a span's first insert every later access to it is a hit. Both policies converge.
func TestReplayUnboundedBudgetYieldsAllHits(t *testing.T) {
	trace := []KVReplayEvent{
		{1, 10},
		{1, 10}, // hit
		{2, 20},
		{1, 10}, // hit
	}
	lruHits, accessed := ReplayKVCache(trace, 0, KVEvictLRU)
	costHits, _ := ReplayKVCache(trace, 0, KVEvictCostAware)
	wantHits := 20 // two hits on span 1, 10 tokens each
	if lruHits != wantHits || costHits != wantHits {
		t.Fatalf("unbounded budget must yield all-reuse hits: LRU=%d cost=%d want=%d (accessed=%d)", lruHits, costHits, wantHits, accessed)
	}
}

// TestReplaySkipsNonPositiveTokens: a non-positive token count on an event is skipped
// (defensive — incomplete trace row), so it contributes neither to the access total nor to
// the resident set.
func TestReplaySkipsNonPositiveTokens(t *testing.T) {
	trace := []KVReplayEvent{
		{1, 0},  // skipped
		{2, -5}, // skipped
		{3, 10}, // counted
	}
	hits, accessed := ReplayKVCache(trace, 0, KVEvictLRU)
	if hits != 0 || accessed != 10 {
		t.Fatalf("non-positive events skipped: hits=%d accessed=%d, want 0/10", hits, accessed)
	}
}

func hotColdKVReplayTrace() ([]KVReplayEvent, int, int) {
	const (
		hot   = 1
		cold1 = 2
		cold2 = 3
		cold3 = 4
		cold4 = 5
	)
	// Every span is 50 tokens; the budget holds exactly 2 spans resident (100 tokens).
	const spanTokens, budget = 50, 100
	return []KVReplayEvent{
		{hot, spanTokens},   // t1: insert HOT.            resident: {HOT}
		{cold1, spanTokens}, // t2: insert COLD1.          resident: {HOT,COLD1} (full)
		{hot, spanTokens},   // t3: HIT HOT (HOT.hits=1).  resident unchanged
		{cold2, spanTokens}, // t4: insert COLD2, evict 1.
		//   LRU:        HOT@3 newer than COLD1@2 -> evict COLD1. {HOT,COLD2}
		//   cost-aware: HOT.cost(Hits=1)=2 > COLD1.cost(Hits=0)=1 -> evict COLD1.
		{cold3, spanTokens}, // t5: insert COLD3, evict 1.
		//   LRU:        HOT@3 OLDER than COLD2@4 -> evict HOT!  {COLD2,COLD3}
		//   cost-aware: HOT.cost=2 > COLD2.cost=1 -> evict COLD2. {HOT,COLD3}
		{hot, spanTokens}, // t6: access HOT.
		//   LRU:        MISS (evicted@5), re-insert HOT, evict COLD2@4 (oldest). {HOT,COLD3}
		//   cost-aware: HIT (HOT.hits=2).
		{cold4, spanTokens}, // t7: insert COLD4, evict 1.
		//   LRU:        COLD3@5 older than HOT@6 -> evict COLD3. {HOT,COLD4}
		//   cost-aware: HOT.cost(Hits=2)=3 > COLD3.cost=1 -> evict COLD3. {HOT,COLD4}
		{hot, spanTokens}, // t8: access HOT.
		//   LRU:        HIT now (HOT resident since t6).
		//   cost-aware: HIT (HOT.hits=3).
	}, spanTokens, budget
}
