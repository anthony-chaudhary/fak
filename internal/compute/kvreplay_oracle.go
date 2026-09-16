package compute

import (
	"math"

	"github.com/anthony-chaudhary/fak/internal/mathx"
	"github.com/anthony-chaudhary/fak/internal/replayoracle"
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
// when the trace has at most 63 distinct spans, and marks Exact=false when it falls back
// to a farthest-next-use approximation. It is a thin shim over replayoracle.Belady; the
// algorithm lives in the tier-1 oracle package so the model-side benches can share it.
func BeladyKVReplayOracle(events []KVReplayEvent, budget int) KVReplayOracleResult {
	r := replayoracle.Belady(toReplayEvents(events), budget)
	return KVReplayOracleResult{HitTokens: r.HitTokens, AccessTokens: r.AccessTokens, Exact: r.Exact}
}

// toReplayEvents projects compute's replay rows onto the shared oracle event type.
func toReplayEvents(events []KVReplayEvent) []replayoracle.Event {
	out := make([]replayoracle.Event, len(events))
	for i, ev := range events {
		out[i] = replayoracle.Event{SpanID: ev.SpanID, Tokens: ev.Tokens}
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
