package replayoracle

import (
	"math/rand"
	"testing"
)

// TestBeladyExactOptimumOnWitnessFixture pins the #13001 witness. On a
// full-cache thrash the exact oracle must report the true optimum a real cache
// can reach, not the unachievable count the illegal miss-branch seed produced.
func TestBeladyExactOptimumOnWitnessFixture(t *testing.T) {
	witness := []Event{
		{SpanID: 0, Tokens: 6},
		{SpanID: 1, Tokens: 6},
		{SpanID: 0, Tokens: 6},
	}
	// Budget holds exactly one span. Admitting both is impossible, so the third
	// access can never hit: the reachable optimum is 0, not 6.
	for _, budget := range []int{6, 7, 8, 9, 10, 11} {
		got := Belady(witness, budget)
		if !got.Exact {
			t.Fatalf("budget=%d: witness must take the exact path", budget)
		}
		if got.HitTokens != 0 {
			t.Fatalf("budget=%d: oracle hits=%d, want the reachable optimum 0", budget, got.HitTokens)
		}
		if got.AccessTokens != 18 {
			t.Fatalf("budget=%d: access=%d, want 18", budget, got.AccessTokens)
		}
	}
	// With room for both spans every reuse is reachable: 6 hit tokens.
	if got := Belady(witness, 12); got.HitTokens != 6 || !got.Exact {
		t.Fatalf("budget=12: %+v, want hits=6 exact=true", got)
	}
}

// maskBruteOpt is an independent exhaustive search over the SAME state model the
// DP declares, used as ground truth: state is the resident bitmask, a span's
// footprint is its maximum token size over the trace, a hit scores the current
// touch (and is unreachable if that touch's size would not fit), and a miss may
// keep any budget-fitting subset of residents that contains the touched span. It
// has no memo and no pruning, so it cannot inherit a search bug from the DP.
func maskBruteOpt(events []Event, budget int) int {
	filtered := validEvents(events)
	if budget <= 0 {
		hits := 0
		seen := map[int]bool{}
		for _, ev := range filtered {
			if seen[ev.SpanID] {
				hits += ev.Tokens
			}
			seen[ev.SpanID] = true
		}
		return hits
	}
	idx := map[int]int{}
	var sizes []int
	for _, ev := range filtered {
		i, ok := idx[ev.SpanID]
		if !ok {
			i = len(sizes)
			idx[ev.SpanID] = i
			sizes = append(sizes, 0)
		}
		if ev.Tokens > sizes[i] {
			sizes[i] = ev.Tokens
		}
	}
	weight := func(mask uint64) int {
		w := 0
		for i := range sizes {
			if mask&(1<<uint(i)) != 0 {
				w += sizes[i]
			}
		}
		return w
	}
	var best func(pos int, mask uint64) int
	best = func(pos int, mask uint64) int {
		if pos >= len(filtered) {
			return 0
		}
		ev := filtered[pos]
		bit := uint64(1) << uint(idx[ev.SpanID])
		if mask&bit != 0 {
			return ev.Tokens + best(pos+1, mask)
		}
		bestFuture := -1
		cand := mask | bit
		for sub := cand; ; sub = (sub - 1) & cand {
			if sub&bit != 0 && weight(sub) <= budget {
				if v := best(pos+1, sub); v > bestFuture {
					bestFuture = v
				}
			}
			if sub == 0 {
				break
			}
		}
		if bestFuture < 0 {
			bestFuture = best(pos+1, mask)
		}
		return bestFuture
	}
	return best(0, 0)
}

// TestBeladyMatchesIndependentExhaustiveSearch proves the fixed exact DP returns
// the true optimum of its declared model, so an Exact=true result can never
// exceed what a real cache can reach.
func TestBeladyMatchesIndependentExhaustiveSearch(t *testing.T) {
	rng := rand.New(rand.NewSource(13001))
	for trial := 0; trial < 2000; trial++ {
		n := 1 + rng.Intn(9)
		spanN := 1 + rng.Intn(4)
		events := make([]Event, n)
		for i := range events {
			events[i] = Event{SpanID: rng.Intn(spanN), Tokens: 1 + rng.Intn(8)}
		}
		budget := rng.Intn(26)
		got := Belady(events, budget)
		if !got.Exact {
			t.Fatalf("trial %d: <=4 spans must take the exact path", trial)
		}
		want := maskBruteOpt(events, budget)
		if got.HitTokens != want {
			t.Fatalf("trial %d: events=%v budget=%d -> dp=%d, exhaustive=%d",
				trial, events, budget, got.HitTokens, want)
		}
	}
}
