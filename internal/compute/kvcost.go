package compute

import "math"

// kvcost.go — cost-aware KV eviction: the value-of-keeping function #2239 names
// (recompute-cost × reuse-probability ÷ bytes) and the victim picker + replay harness
// that witness it. This is the POLICY half of #2239's R1 rung (the "cost function per
// resident span"), landing in the compute leaf alongside the other pure KV estimators
// (kvresidency.go, capacity.go): it performs no allocation and holds no model state. The
// wiring half — replacing pure-LRU victim choice in radixkv.EvictToBudget and
// modelengine.pickPreemptVictim with PickEvictionVictim behind FAK_NATIVE_KV_* flags — is
// the cross-lane follow-on tracked in #2239's remaining checkboxes.
//
// Dynamo's KVBM, vLLM's priority block pool, and SGLang's radix LRU all evict by recency
// or a single priority dimension. The fak differentiator #2239 targets is *value-aware*
// lifecycle: evict the span that is CHEAPEST TO LOSE per byte of memory it frees, where
// "cost to lose" folds in both the prefill work that would have to be redone (recompute
// cost) and how likely the span is to be needed again (reuse probability, read off the
// radix hit stats the cache already tracks). A span that is long, frequently reused, and
// compact (a quantized tier) is expensive to lose; a short, one-shot, fat span is the ideal
// victim. Pins (#2211 operator, #2225 agent-declared) and in-flight leases are hard
// exclusions — a pinned/leased span is never a candidate, regardless of cost.

// KVSpanStats is the per-resident-span telemetry a cost-aware KV cache tracks to score
// eviction candidates — the inputs to the value-of-keeping function #2239 specifies. Every
// field is something the cache layer either already maintains (radixkv keeps lastUsed +
// refs; the radix hit stats feed Hits) or derives from the geometry (Bytes from
// EstimateKVStoreBytes at the span's precision tier). The struct is pure data so a caller
// can build it from whatever bookkeeping the resident pool already carries.
type KVSpanStats struct {
	// Tokens is the span's cached length in positions — the recompute cost if it is
	// evicted (the prefill work that would have to be redone to restore it). It is the
	// "recompute-cost" factor of #2239's formula.
	Tokens int
	// Bytes is the span's resident memory footprint — the memory freed if it is evicted.
	// A demoted span (a quantized tier, or one pushed down the ladder by #2169) reports a
	// smaller Bytes for the same Tokens, which raises its cost-of-losing and makes the
	// evictor preferentially keep compacted spans. It is the "÷ bytes" factor.
	Bytes int64
	// Hits is the number of SUBSEQUENT accesses that found this span resident (prefix
	// reuse) — the observed reuse signal. The insertion that first cached the span does
	// not count; only later matches do. It feeds the reuse-probability factor.
	Hits int
	// LastUsed is the logical clock of the most recent access — the LRU key, used as the
	// tie-break among spans of equal cost and as the sole key under the pure-LRU policy.
	LastUsed uint64
	// Pinned reports an operator/agent-declared pin (#2211/#2225). A pinned span is never
	// an eviction candidate: pins are leases with TTL, never permanent, but while live they
	// are a hard exclusion. Cost-aware eviction never violates a pin.
	Pinned bool
	// Leased reports an in-flight request lease (the refs>0 contract radixkv already
	// enforces: a span being served cannot be reclaimed). A leased span is never a victim.
	Leased bool
	// AgeStamp is the GDSF-style aging-clock value (L) recorded the last time this span
	// was referenced (inserted or hit) — see KVEvictionCostAged in kvcost_aging.go. The
	// caller (the resident pool) stamps it with the pool's current AgingClock on every
	// reference; a span that has not been touched since keeps its OLD, smaller stamp while
	// AgingClock keeps advancing on later evictions, which is what lets recency re-enter
	// the score instead of only breaking ties (LastUsed's current role). Zero (the default)
	// makes KVEvictionCostAged reduce exactly to KVEvictionCost, so every existing caller
	// that never sets it is byte-identical to today.
	AgeStamp float64
	// PinBoost is a bounded, TTL-decaying lease strength (#2673) the caller derives from a
	// pin's remaining TTL via PinBoostFromTTL — the economics form of a pin that
	// KVEvictionCostPinned FOLDS INTO the cost-of-losing (additively) rather than bypassing
	// it. A live pin with ample TTL carries a large boost (effectively unevictable, today's
	// hard-pin behavior); as TTL → 0 the boost decays to 0 so an expiring pin on a cold span
	// releases gracefully instead of holding memory forever. Zero (the default) makes
	// KVEvictionCostPinned reduce exactly to KVEvictionCost, so every existing caller that
	// never sets it is byte-identical to today. The absolute-exclusion Pinned bool is the
	// PinBoost = +Inf limit of this term; both are honored by KVEvictionCostPinned.
	PinBoost float64
	// Hint is an adjudicated agent cache-control prior (#2225/#805) — a closed vocabulary
	// (HintNone/HintPrecious/HintEphemeral) that KVEvictionCostPinned applies as a BOUNDED
	// multiplier on the reuse term, never an override: `precious` raises keep-value,
	// `ephemeral` lowers it, both saturating so a hint can never fully defeat the value
	// function (an agent cannot game the cache into starvation — the adjudication fence).
	// HintNone (the default) leaves the reuse term untouched, so an unhinted span is
	// byte-identical to today.
	Hint KVCacheHint
	// Sharers is the count of live sessions/leases currently depending on this span as a
	// resident prefix (#2670) — the CONCURRENT-BREADTH signal the radix layer already tracks
	// via ref-count/child fan-out but that Hits never captures (Hits counts subsequent
	// lookups on one node, not how many distinct consumers currently share the node). A span
	// shared by many live sessions costs a re-prefill PER sharer if evicted, not one.
	// Sharers <= 1 (the zero value included) makes KVEvictionCostFanout reduce exactly to
	// KVEvictionCost, so every existing caller that never sets it is byte-identical to today.
	Sharers int
	// IntervalSum is the cumulative logical access gap across subsequent accesses (#2669).
	// Used by KVReuseEstimate to derive mean inter-arrival.
	IntervalSum uint64
	// IntervalCount is the number of access intervals recorded (#2669).
	IntervalCount int
}

// KVCacheHint is the closed vocabulary of adjudicated agent cache-control priors (#2225/#805
// intent conduit, default-deny adjudication) that KVEvictionCostPinned folds into the reuse
// term as a bounded multiplier. It is a PRIOR on keep-value, not an override: the multiplier
// is finite and strictly positive (see hintReuseMultiplier), so a hint biases the ranking
// but a genuinely hotter or colder span still dominates — an agent cannot pin the cache into
// starvation by declaring everything precious, nor evict a live hot span by declaring it
// ephemeral. The zero value HintNone is the identity (multiplier 1).
type KVCacheHint int

const (
	// HintNone is the absence of an agent cache-control hint — the reuse term is unscaled,
	// so a span with HintNone scores exactly as it does under KVEvictionCost. Default.
	HintNone KVCacheHint = iota
	// HintPrecious is an agent declaring a span worth keeping (#2225): it RAISES the reuse
	// term by a bounded factor, biasing the evictor to retain the span — but bounded, so a
	// precious hint on a genuinely cold span still loses to a much hotter unhinted span.
	HintPrecious
	// HintEphemeral is an agent declaring a span one-shot (#2225): it LOWERS the reuse term
	// by a bounded factor, biasing the evictor to reclaim the span first — but bounded, so
	// an ephemeral hint on a genuinely hot span still outranks a much colder unhinted span.
	HintEphemeral
)

// KVEvictionCost is #2239's value-of-keeping score for a resident KV span:
//
//	cost = recomputeCost × reuseProbability ÷ bytes
//
// read as "expected future recompute work per byte of memory freed if this span is
// evicted". A HIGHER cost means the span is more expensive to lose per byte gained, so a
// cost-aware evictor KEEPS it; the cheapest-to-lose span (LOWEST cost) is the victim.
//
//   - recomputeCost = Tokens (the span length — the prefill work evicted).
//   - reuseProbability = Hits + 1 (Laplace smoothing: a span never reused still carries a
//     residual probability of 1, so among equal-reuse spans the byte/recompute terms
//     decide; a span reused N times carries probability N+1, dominating). The +1 keeps a
//     zero-hit span's cost positive and well-defined, and means the function never
//     divides experience out of the ranking.
//   - bytes = Bytes (the resident footprint).
//
// The byte-normalization has a clean reduction: when every span pays the same per-token
// byte cost (Bytes = Tokens × k, one precision tier), cost = Tokens×(Hits+1)÷(Tokens×k) =
// (Hits+1)÷k — i.e. under uniform memory cost the evictor reduces to "evict the
// least-reused span". Bytes only changes the ranking when spans carry DIFFERENT
// per-token costs (a quantized tier demoted by #1474, or a span partly offloaded by
// #2169): then a compact span is cheaper to keep per unit of recompute value, and the
// evictor prefers evicting the fat one. That is exactly the value-aware distinction
// recency-only eviction cannot make.
//
// Fail-open: a span with non-positive Bytes (untracked footprint) scores +Inf so it is
// never preferred as a victim — dropping a span of unknown size is conservative.
func KVEvictionCost(s KVSpanStats) float64 {
	if s.Bytes <= 0 {
		return math.Inf(1)
	}
	reuseProbability := KVReuseTerm(s) // Unified reuse term seam (#3411)
	recomputeCost := float64(s.Tokens)
	return recomputeCost * reuseProbability / float64(s.Bytes)
}

// PickEvictionVictim selects the cheapest-to-lose resident span per KVEvictionCost: it
// returns the index (into spans) of the LOWEST-cost evictable candidate, skipping any span
// that is Pinned or Leased (the hard exclusions — a pinned/leased span is never a victim,
// matching the radixkv refs>0 rule). Ties break to the OLDEST LastUsed (LRU), so when
// costs are uniform the picker degenerates exactly to LRU victim selection — cost-aware
// eviction is a strict generalization of LRU, never a divergence from it on equal-cost
// inputs. Returns -1 when no candidate is evictable (everything resident is pinned or
// leased), the same signal radixkv.lruLeaf sends when budget-excess is fully locked.
//
// This is the function #2239's first checkbox names as the replacement for pure-LRU victim
// choice; wiring it into radixkv.EvictToBudget and modelengine.pickPreemptVictim behind
// FAK_NATIVE_KV_* flags is the cross-lane follow-on.
func PickEvictionVictim(spans []KVSpanStats) int {
	idx, _ := pickLowestCost(spans, KVEvictionCost)
	return idx
}

// pickLowestCost is the shared victim-ranking loop every PickEvictionVictim* variant runs:
// it scans spans for the cheapest-to-lose EVICTABLE candidate under the supplied per-span
// cost function, skipping any Pinned or Leased span (the hard exclusions), and breaks a cost
// tie toward the OLDEST LastUsed (LRU) so a uniform-cost resident set degenerates to pure LRU
// victim choice. It returns the winning index (or -1 when nothing is evictable) and that
// winner's cost. The variants differ ONLY in which cost function scores a span — KVEvictionCost
// (base), KVEvictionCostFanout, KVEvictionCostPinned, KVEvictionCostAged — so the ranking,
// exclusions, tie-break, and -1 no-candidate contract live here once. bestCost is the float64
// zero value when bestIdx is -1; callers that surface the cost (PickEvictionVictimAged) gate on
// the -1 index rather than the cost.
func pickLowestCost(spans []KVSpanStats, cost func(KVSpanStats) float64) (bestIdx int, bestCost float64) {
	bestIdx = -1
	for i := range spans {
		s := spans[i]
		if s.Pinned || s.Leased {
			continue
		}
		c := cost(s)
		switch {
		case bestIdx == -1:
			bestIdx, bestCost = i, c
		case c < bestCost:
			bestIdx, bestCost = i, c
		case c == bestCost && s.LastUsed < spans[bestIdx].LastUsed:
			// Equal cost-of-losing → break the tie by recency (oldest first), so a
			// uniform-cost resident set reduces to pure LRU.
			bestIdx = i
		}
	}
	return bestIdx, bestCost
}

// KVEvictPolicy selects the victim-ranking policy ReplayKVCache simulates.
type KVEvictPolicy int

const (
	// KVEvictLRU is pure recency-only eviction — least-recently-used victim, the policy
	// radixkv and modelengine run today and the baseline #2239's cost-aware mode must beat
	// on a trace with hot/cold structure.
	KVEvictLRU KVEvictPolicy = iota
	// KVEvictCostAware is #2239's value-aware eviction — cheapest-to-lose victim per
	// KVEvictionCost, LRU tie-break. On a uniform-cost resident set it is identical to
	// KVEvictLRU; it diverges (and wins) only when the reuse signal separates hot spans
	// from one-shot spans.
	KVEvictCostAware
)

// KVReplayEvent is one access in a replayed agent-session trace: SpanID names the prefix
// span touched (a stable id the simulator keys resident state on), Tokens is its length
// (the recompute cost and, at uniform per-token bytes in the replay, the footprint). A
// sequence of events replays a session's cache pressure; ReplayKVCache measures the hit
// rate each policy achieves at a fixed budget.
type KVReplayEvent struct {
	SpanID int
	Tokens int
}

// ReplayKVCache simulates a token-budgeted KV cache over a replayed access trace under
// either pure-LRU or cost-aware (#2239) eviction, returning the cache hit rate as
// (hitTokens, accessTokens): the total span tokens that found their prefix ALREADY
// resident over the total span tokens touched. It performs no model allocation — it is a
// pure accounting loop over KVSpanStats, the same discipline as the other compute
// estimators — and exists to WITNESS #2239's eviction-pressure claim: at a fixed budget,
// cost-aware eviction scores at least LRU's hit rate on any trace (it reduces to LRU when
// reuse is uniform) and STRICTLY higher on a trace where a hot, frequently-reused span
// competes with one-shot spans (the case recency-only eviction gets wrong).
//
// A budget of 0 disables eviction (unbounded); a non-positive per-event token count is
// skipped. No span is pinned or leased in the replay, so the policy difference is purely
// the victim ranking; the pin/lease exclusions are exercised directly by PickEvictionVictim's
// unit tests. This is the harness a bench child of #2236 drives against replayed agent
// sessions to land the R3 comparison.
func ReplayKVCache(events []KVReplayEvent, budget int, policy KVEvictPolicy) (hitTokens, accessTokens int) {
	type resident struct {
		tokens   int
		hits     int
		lastUsed uint64
	}
	residentSpans := map[int]*resident{}
	residentTokens := 0
	var clock uint64

	// victimUnderPolicy picks the resident span id to evict under the chosen policy. It
	// mirrors PickEvictionVictim's ranking but over the simulator's resident map; because
	// no replayed span is pinned/leased, the exclusion paths do not fire here.
	victimUnderPolicy := func() int {
		var victimID = -1
		var victimCost float64
		var minLastUsed uint64
		for id, r := range residentSpans {
			stats := KVSpanStats{Tokens: r.tokens, Bytes: int64(r.tokens), Hits: r.hits, LastUsed: r.lastUsed}
			switch policy {
			case KVEvictLRU:
				if victimID == -1 || r.lastUsed < minLastUsed {
					victimID, minLastUsed = id, r.lastUsed
				}
			case KVEvictCostAware:
				c := KVEvictionCost(stats)
				switch {
				case victimID == -1:
					victimID, victimCost, minLastUsed = id, c, r.lastUsed
				case c < victimCost:
					victimID, victimCost, minLastUsed = id, c, r.lastUsed
				case c == victimCost && r.lastUsed < minLastUsed:
					victimID, minLastUsed = id, r.lastUsed
				}
			}
		}
		return victimID
	}

	for _, ev := range events {
		if ev.Tokens <= 0 {
			continue
		}
		accessTokens += ev.Tokens
		clock++
		if r, ok := residentSpans[ev.SpanID]; ok {
			// Cache hit: the prefix was resident. Account the reused tokens and refresh
			// recency + the reuse counter that feeds cost-aware victim choice.
			r.hits++
			r.lastUsed = clock
			hitTokens += ev.Tokens
			continue
		}
		// Cache miss: insert the span, evicting cheapest-to-lose (or LRU) victims until
		// the budget holds the new span. A zero budget means unbounded (no eviction).
		for budget > 0 && residentTokens+ev.Tokens > budget && len(residentSpans) > 0 {
			vid := victimUnderPolicy()
			if vid < 0 {
				break
			}
			residentTokens -= residentSpans[vid].tokens
			delete(residentSpans, vid)
		}
		residentSpans[ev.SpanID] = &resident{tokens: ev.Tokens, lastUsed: clock}
		residentTokens += ev.Tokens
	}
	return hitTokens, accessTokens
}
