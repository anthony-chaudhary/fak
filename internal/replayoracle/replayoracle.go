// Package replayoracle is the pure offline replay upper-bound core: given a finite
// access trace and a token budget it returns the exact Belady-style max-hit bound, or a
// farthest-next-use approximation above the exact-DP span limit. It imports nothing
// internal so both the serving compute layer and the model-side MoE benches can share it.
package replayoracle

import "math/bits"

// Event is one access in a replayed session trace: SpanID names the span touched (a stable
// id the simulator keys resident state on), Tokens is its length (the recompute cost and,
// at uniform per-token bytes in the replay, the footprint). A sequence of events replays a
// session's cache pressure.
type Event struct {
	SpanID int
	Tokens int
}

// Result is the offline Belady-style upper bound for a finite replay.
type Result struct {
	HitTokens    int  `json:"hit_tokens"`
	AccessTokens int  `json:"access_tokens"`
	Exact        bool `json:"exact"`
}

// Belady computes the exact offline max-hit upper bound for a finite trace when the trace
// has at most 63 distinct spans. The DP state is (event index, resident bitset); on a miss
// it may keep any fitting subset of the current residents plus the accessed span, which is
// the offline optimum. Larger traces fall back to a farthest-next-use approximation and
// mark Exact=false.
func Belady(events []Event, budget int) Result {
	filtered := validEvents(events)
	access := 0
	for _, ev := range filtered {
		access += ev.Tokens
	}
	if len(filtered) == 0 {
		return Result{Exact: true}
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
		return Result{HitTokens: hits, AccessTokens: access, Exact: true}
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
		return Result{HitTokens: hits, AccessTokens: access, Exact: false}
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

	return Result{HitTokens: best(0, 0), AccessTokens: access, Exact: true}
}

func validEvents(events []Event) []Event {
	out := make([]Event, 0, len(events))
	for _, ev := range events {
		if ev.Tokens > 0 {
			out = append(out, ev)
		}
	}
	return out
}

func beladyGreedyHits(events []Event, budget int) int {
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

func farthestNextUse(events []Event, start int, resident map[int]int) int {
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
