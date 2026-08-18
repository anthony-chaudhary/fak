package compute

import (
	"math"
	"math/bits"

	"github.com/anthony-chaudhary/fak/internal/mathx"
)

// KVReplayResult is the structured replay row #2675 needs for policy comparisons.
type KVReplayResult struct {
	Policy            KVEvictPolicy `json:"policy"`
	HitTokens         int           `json:"hit_tokens"`
	AccessTokens      int           `json:"access_tokens"`
	Evictions         int           `json:"evictions"`
	EvictionsPerHit   float64       `json:"evictions_per_hit"`
	GoodDecisionRatio float64       `json:"good_decision_ratio"`
}

// KVReplayOracleResult is the offline Belady-style upper bound for a finite replay.
type KVReplayOracleResult struct {
	HitTokens    int  `json:"hit_tokens"`
	AccessTokens int  `json:"access_tokens"`
	Exact        bool `json:"exact"`
}

// ReplayKVCacheResult replays one policy and reports hit, eviction, and stability counts.
func ReplayKVCacheResult(events []KVReplayEvent, budget int, policy KVEvictPolicy) KVReplayResult {
	type resident struct {
		tokens   int
		hits     int
		lastUsed uint64
	}
	residentSpans := map[int]*resident{}
	residentTokens := 0
	var clock uint64
	result := KVReplayResult{Policy: policy}

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
		result.AccessTokens += ev.Tokens
		clock++
		if r, ok := residentSpans[ev.SpanID]; ok {
			r.hits++
			r.lastUsed = clock
			result.HitTokens += ev.Tokens
			continue
		}
		for budget > 0 && residentTokens+ev.Tokens > budget && len(residentSpans) > 0 {
			vid := victimUnderPolicy()
			if vid < 0 {
				break
			}
			residentTokens -= residentSpans[vid].tokens
			delete(residentSpans, vid)
			result.Evictions++
		}
		residentSpans[ev.SpanID] = &resident{tokens: ev.Tokens, lastUsed: clock}
		residentTokens += ev.Tokens
	}
	result.EvictionsPerHit = evictionsPerHit(result.Evictions, result.HitTokens)
	return result
}

// ReplayKVCacheMulti replays every requested policy and scores it against the offline
// oracle. The old ReplayKVCache API remains the compatibility shim for callers that only
// need hit/access tokens.
func ReplayKVCacheMulti(events []KVReplayEvent, budget int, policies ...KVEvictPolicy) map[KVEvictPolicy]KVReplayResult {
	if len(policies) == 0 {
		policies = []KVEvictPolicy{KVEvictLRU, KVEvictCostAware}
	}
	oracle := BeladyKVReplayOracle(events, budget)
	out := make(map[KVEvictPolicy]KVReplayResult, len(policies))
	for _, policy := range policies {
		result := ReplayKVCacheResult(events, budget, policy)
		result.GoodDecisionRatio = mathx.AgainstOracle(result.HitTokens, oracle.HitTokens)
		out[policy] = result
	}
	return out
}

// BeladyKVReplayOracle computes the exact offline max-hit upper bound for a finite trace
// when the trace has at most 63 distinct spans. The DP state is (event index, resident
// bitset); on a miss it may keep any fitting subset of the current residents plus the
// accessed span, which is the offline optimum. Larger traces fall back to a farthest-next-use
// approximation and mark Exact=false.
func BeladyKVReplayOracle(events []KVReplayEvent, budget int) KVReplayOracleResult {
	filtered := validKVReplayEvents(events)
	access := 0
	for _, ev := range filtered {
		access += ev.Tokens
	}
	if len(filtered) == 0 {
		return KVReplayOracleResult{Exact: true}
	}
	if budget <= 0 {
		hits := 0
		seen := map[int]bool{}
		for _, ev := range filtered {
			if seen[ev.SpanID] {
				hits += ev.Tokens
			}
			seen[ev.SpanID] = true
		}
		return KVReplayOracleResult{HitTokens: hits, AccessTokens: access, Exact: true}
	}

	spanIndex := map[int]int{}
	var spanIDs []int
	for _, ev := range filtered {
		if _, ok := spanIndex[ev.SpanID]; !ok {
			spanIndex[ev.SpanID] = len(spanIDs)
			spanIDs = append(spanIDs, ev.SpanID)
		}
	}
	if len(spanIDs) > 63 {
		hits := beladyGreedyHits(filtered, budget)
		return KVReplayOracleResult{HitTokens: hits, AccessTokens: access, Exact: false}
	}

	sizes := make([]int, len(spanIDs))
	for _, ev := range filtered {
		idx := spanIndex[ev.SpanID]
		if sizes[idx] == 0 {
			sizes[idx] = ev.Tokens
		}
	}
	weights := make(map[uint64]int)
	var maskWeight func(uint64) int
	maskWeight = func(mask uint64) int {
		if w, ok := weights[mask]; ok {
			return w
		}
		total := 0
		m := mask
		for m != 0 {
			bit := bits.TrailingZeros64(m)
			total += sizes[bit]
			m &^= 1 << uint(bit)
		}
		weights[mask] = total
		return total
	}

	type key struct {
		pos  int
		mask uint64
	}
	memo := map[key]int{}
	var best func(int, uint64) int
	best = func(pos int, mask uint64) int {
		if pos >= len(filtered) {
			return 0
		}
		k := key{pos: pos, mask: mask}
		if v, ok := memo[k]; ok {
			return v
		}
		ev := filtered[pos]
		idx := spanIndex[ev.SpanID]
		bit := uint64(1) << uint(idx)
		if mask&bit != 0 {
			v := ev.Tokens + best(pos+1, mask)
			memo[k] = v
			return v
		}

		candidates := mask | bit
		bestFuture := best(pos+1, mask&^bit) // do not keep the missed span.
		for sub := candidates; ; sub = (sub - 1) & candidates {
			if sub&bit != 0 && maskWeight(sub) <= budget {
				if v := best(pos+1, sub); v > bestFuture {
					bestFuture = v
				}
			}
			if sub == 0 {
				break
			}
		}
		memo[k] = bestFuture
		return bestFuture
	}

	return KVReplayOracleResult{HitTokens: best(0, 0), AccessTokens: access, Exact: true}
}

func validKVReplayEvents(events []KVReplayEvent) []KVReplayEvent {
	out := make([]KVReplayEvent, 0, len(events))
	for _, ev := range events {
		if ev.Tokens > 0 {
			out = append(out, ev)
		}
	}
	return out
}

func evictionsPerHit(evictions, hitTokens int) float64 {
	if hitTokens <= 0 {
		if evictions == 0 {
			return 0
		}
		return math.Inf(1)
	}
	return float64(evictions) / float64(hitTokens)
}

func beladyGreedyHits(events []KVReplayEvent, budget int) int {
	resident := map[int]int{}
	residentTokens := 0
	hits := 0
	for i, ev := range events {
		if _, ok := resident[ev.SpanID]; ok {
			hits += ev.Tokens
			continue
		}
		for budget > 0 && residentTokens+ev.Tokens > budget && len(resident) > 0 {
			victim := farthestNextUse(events, i+1, resident)
			residentTokens -= resident[victim]
			delete(resident, victim)
		}
		if budget <= 0 || ev.Tokens <= budget {
			resident[ev.SpanID] = ev.Tokens
			residentTokens += ev.Tokens
		}
	}
	return hits
}

func farthestNextUse(events []KVReplayEvent, start int, resident map[int]int) int {
	victim := 0
	bestDistance := -1
	for id := range resident {
		distance := len(events) + 1
		for j := start; j < len(events); j++ {
			if events[j].SpanID == id {
				distance = j - start
				break
			}
		}
		if bestDistance < 0 || distance > bestDistance || (distance == bestDistance && id < victim) {
			victim, bestDistance = id, distance
		}
	}
	return victim
}
