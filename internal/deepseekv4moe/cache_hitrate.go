package deepseekv4moe

import (
	"errors"
	"math/bits"
)

// ExpertCacheHitRateWitnessSchema identifies the first-party expert-cache
// hit-rate witness wire format.
const ExpertCacheHitRateWitnessSchema = "fak.expert-cache-hitrate-witness/v1"

var (
	// ErrNoTrace means there is no expert-activation trace to witness. It fails
	// closed with a typed verdict rather than fabricating a rate.
	ErrNoTrace = errors.New("deepseekv4moe: no expert-activation trace to witness")
	// ErrInvalidHitRateWitnessBudget means the trace declares a byte budget that
	// does not afford even one whole routed (layer, expert) group.
	ErrInvalidHitRateWitnessBudget = errors.New("deepseekv4moe: invalid expert-cache hit-rate witness budget")
)

// ExpertActivationTrace is a captured or re-derived routed-expert activation
// trace. GroupBytes is the whole resident byte size of one (layer, expert)
// weight group; the byte budget is expressed against it.
type ExpertActivationTrace struct {
	Schema      string        `json:"schema"`
	Name        string        `json:"name"`
	Source      string        `json:"source"`
	Layers      int           `json:"layers"`
	Experts     int           `json:"experts"`
	TopK        int           `json:"top_k"`
	BudgetBytes int64         `json:"budget_bytes"`
	GroupBytes  int64         `json:"group_bytes"`
	Routes      []ExpertRoute `json:"routes"`
}

// ExpertCacheHitRateWitness is the deterministic receipt for one trace+budget.
// It is weight-free control-plane evidence: it proves neither model I/O,
// dequant/GPU execution, output parity, latency, nor throughput. Label is
// always "SW-VERIFIED". The Belady number is an offline upper bound, so
// BeladyRegretHits is clamped to >= 0 and is never a hardware claim.
type ExpertCacheHitRateWitness struct {
	Schema               string  `json:"schema"`
	Label                string  `json:"label"`
	TraceName            string  `json:"trace_name"`
	TraceSource          string  `json:"trace_source"`
	Policy               string  `json:"policy"`
	TopK                 int     `json:"top_k"`
	CacheGroups          int     `json:"cache_groups"`
	CacheBytes           int64   `json:"cache_bytes"`
	GroupBytes           int64   `json:"group_bytes"`
	Accesses             int64   `json:"accesses"`
	Hits                 int64   `json:"hits"`
	Misses               int64   `json:"misses"`
	HitRate              float64 `json:"hit_rate"`
	HitRateKnown         bool    `json:"hit_rate_known"`
	ByteWeightedHitRate  float64 `json:"byte_weighted_hit_rate"`
	ByteWeightedHitKnown bool    `json:"byte_weighted_hit_rate_known"`
	BytesReadResident    int64   `json:"bytes_read_resident"`
	BytesReadStreamed    int64   `json:"bytes_read_streamed"`
	BeladyOptimalHits    int64   `json:"belady_optimal_hits"`
	BeladyOptimalMisses  int64   `json:"belady_optimal_misses"`
	BeladyRegretHits     int64   `json:"belady_regret_hits"`
	BeladyRegretRatio    float64 `json:"belady_regret_ratio"`
	BeladyExact          bool    `json:"belady_exact"`
	PeakResident         int     `json:"peak_resident"`
}

// WitnessExpertCacheHitRate replays an expert-activation trace through the
// deterministic LRU simulator at the whole-group capacity the byte budget
// affords, and reports the LRU hit rate plus its regret against the offline
// Belady optimum.
//
// The Belady oracle maps each distinct (layer, expert) group to a stable
// SpanID (Layer*Experts+Expert+1) with Tokens=1, one unit per group access, and
// is budgeted in the same whole-group unit as CacheGroups. Because each event
// is one token, the oracle's HitTokens is a group-access hit count directly
// comparable to SimulateExpertCache.Hits. Traces with more than 63 distinct
// groups fall back to a farthest-next-use approximation (BeladyExact=false);
// its optimal may undershoot LRU, so the reported regret is clamped to 0 while
// BeladyExact stays false so the consumer knows.
//
// This is software-verified, weight-free control-plane evidence only.
func WitnessExpertCacheHitRate(trace ExpertActivationTrace) (ExpertCacheHitRateWitness, error) {
	if len(trace.Routes) == 0 {
		return ExpertCacheHitRateWitness{}, ErrNoTrace
	}
	if trace.Layers <= 0 || trace.Experts <= 0 || trace.TopK <= 0 || trace.TopK > trace.Experts || trace.GroupBytes <= 0 {
		return ExpertCacheHitRateWitness{}, ErrInvalidTraceShape
	}
	if trace.BudgetBytes <= 0 {
		return ExpertCacheHitRateWitness{}, ErrInvalidTraceCapacity
	}

	groups := trace.BudgetBytes / trace.GroupBytes
	if groups < 1 {
		return ExpertCacheHitRateWitness{}, ErrInvalidHitRateWitnessBudget
	}
	modelGroups := int64(trace.Layers) * int64(trace.Experts)
	if groups > modelGroups {
		groups = modelGroups
	}

	traceResult, err := SimulateExpertCache(trace.Routes, int(groups), trace.Layers, trace.Experts, trace.TopK)
	if err != nil {
		return ExpertCacheHitRateWitness{}, err
	}

	hits, misses := traceResult.Hits, traceResult.PageIns
	accesses := hits + misses
	hitRate, hitKnown := witnessHitRate(hits, misses)

	bytesResident := hits * trace.GroupBytes
	bytesStreamed := misses * trace.GroupBytes
	byteRate, byteKnown := witnessHitRate(bytesResident, bytesStreamed)

	optimalHits, exact := beladyOptimalHits(beladyEvents(trace), int(groups))
	optimalHitsInt := int64(optimalHits)
	optimalMisses := accesses - optimalHitsInt
	regret := optimalHitsInt - hits
	if regret < 0 {
		regret = 0
	}
	var regretRatio float64
	if optimalHits > 0 {
		regretRatio = float64(regret) / float64(optimalHits)
	}

	return ExpertCacheHitRateWitness{
		Schema:               ExpertCacheHitRateWitnessSchema,
		Label:                "SW-VERIFIED",
		TraceName:            trace.Name,
		TraceSource:          trace.Source,
		Policy:               "lru",
		TopK:                 trace.TopK,
		CacheGroups:          int(groups),
		CacheBytes:           groups * trace.GroupBytes,
		GroupBytes:           trace.GroupBytes,
		Accesses:             accesses,
		Hits:                 hits,
		Misses:               misses,
		HitRate:              hitRate,
		HitRateKnown:         hitKnown,
		ByteWeightedHitRate:  byteRate,
		ByteWeightedHitKnown: byteKnown,
		BytesReadResident:    bytesResident,
		BytesReadStreamed:    bytesStreamed,
		BeladyOptimalHits:    optimalHitsInt,
		BeladyOptimalMisses:  optimalMisses,
		BeladyRegretHits:     regret,
		BeladyRegretRatio:    regretRatio,
		BeladyExact:          exact,
		PeakResident:         traceResult.PeakResident,
	}, nil
}

func witnessHitRate(hits, misses int64) (float64, bool) {
	total := hits + misses
	if total <= 0 {
		return 0, false
	}
	return float64(hits) / float64(total), true
}

func beladyEvents(trace ExpertActivationTrace) []int {
	events := make([]int, 0, len(trace.Routes)*trace.TopK)
	for _, route := range trace.Routes {
		for _, expert := range route.Experts {
			events = append(events, route.Layer*trace.Experts+expert+1)
		}
	}
	return events
}

// beladyOptimalHits computes the offline Belady MIN hit count for a sequence of
// unit-sized group accesses and returns whether the result is exact. With at
// most 63 distinct spans it solves the exact DP over (position, resident bitset):
// on a miss it may retain ANY fitting subset of the current residents plus the
// accessed span, which is the true offline optimum. Larger traces fall back to a
// farthest-next-use approximation and report exact=false.
//
// This is deliberately self-contained rather than delegating to the shared KV
// oracle: the receipt's "regret vs the offline optimal" claim must be an honest
// upper bound, and the witness owns the exactness of the number it publishes.
func beladyOptimalHits(events []int, budget int) (int, bool) {
	if len(events) == 0 {
		return 0, true
	}
	if budget <= 0 {
		hits := 0
		seen := map[int]bool{}
		for _, span := range events {
			if seen[span] {
				hits++
			}
			seen[span] = true
		}
		return hits, true
	}

	index := map[int]int{}
	for _, span := range events {
		if _, ok := index[span]; !ok {
			index[span] = len(index)
		}
	}
	if len(index) > 63 {
		return beladyGreedyHits(events, budget), false
	}
	if len(index) <= budget {
		// Every distinct span fits at once: the optimum is every repeat access.
		seen := map[int]bool{}
		hits := 0
		for _, span := range events {
			if seen[span] {
				hits++
			}
			seen[span] = true
		}
		return hits, true
	}

	weights := map[uint64]int{}
	weight := func(mask uint64) int {
		if w, ok := weights[mask]; ok {
			return w
		}
		w := bits.OnesCount64(mask)
		weights[mask] = w
		return w
	}

	type key struct {
		pos  int
		mask uint64
	}
	memo := map[key]int{}
	var best func(pos int, mask uint64) int
	best = func(pos int, mask uint64) int {
		if pos >= len(events) {
			return 0
		}
		k := key{pos: pos, mask: mask}
		if v, ok := memo[k]; ok {
			return v
		}
		bit := uint64(1) << uint(index[events[pos]])
		var v int
		if mask&bit != 0 {
			v = 1 + best(pos+1, mask)
		} else {
			// A miss MUST admit the accessed span, so the next resident set is a
			// fitting subset of the current residents UNION the accessed span.
			candidates := mask | bit
			bestFuture := -1
			for sub := candidates; ; sub = (sub - 1) & candidates {
				if sub&bit != 0 && weight(sub) <= budget {
					if hv := best(pos+1, sub); hv > bestFuture {
						bestFuture = hv
					}
				}
				if sub == 0 {
					break
				}
			}
			v = bestFuture
		}
		memo[k] = v
		return v
	}
	return best(0, 0), true
}

// beladyGreedyHits is the farthest-next-use approximation for traces with more
// than 63 distinct spans, where the exact bitset DP is infeasible.
func beladyGreedyHits(events []int, budget int) int {
	resident := map[int]bool{}
	hits := 0
	for i, span := range events {
		if resident[span] {
			hits++
			continue
		}
		for len(resident) >= budget {
			victim := farthestNextUse(events, i+1, resident)
			delete(resident, victim)
		}
		resident[span] = true
	}
	return hits
}

func farthestNextUse(events []int, start int, resident map[int]bool) int {
	victim := 0
	bestDistance := -1
	for id := range resident {
		distance := len(events) + 1
		for j := start; j < len(events); j++ {
			if events[j] == id {
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
