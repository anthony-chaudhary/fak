package deepseekv4moe

import (
	"sort"

	"github.com/anthony-chaudhary/fak/internal/mathx"
	"github.com/anthony-chaudhary/fak/internal/replayoracle"
)

// CachePolicyBenchSchema is the receipt schema id for the expert-cache policy bench.
const CachePolicyBenchSchema = "fak.moe-expert-cache-policy/v1"

// CachePolicyMeasurementSimulated marks every receipt as synthetic evidence. The bench
// replays a modeled access stream; it is not a live measurement.
const CachePolicyMeasurementSimulated = "simulated"

// heatDecayEvery is how many accesses between two halvings of every resident group's
// decaying heat in the Heat arm. Chosen so a group touched in the last ~16 accesses keeps
// most of its heat while a cold group's signal fades.
const heatDecayEvery = 16

// CachePolicy names one eviction policy arm replayed over one expert access stream.
type CachePolicy int

const (
	// CachePolicyLRU evicts the least-recently-used resident group. It is the online
	// baseline and MUST reproduce SimulateExpertCache bit-for-bit on the same stream.
	CachePolicyLRU CachePolicy = iota
	// CachePolicyLFU evicts the resident group with the smallest lifetime access count.
	CachePolicyLFU
	// CachePolicyHeat evicts the resident group with the smallest decaying heat, retaining
	// ghost heat for evicted groups.
	CachePolicyHeat
	// CachePolicyRandom evicts a deterministic pseudo-random resident group; reproducible
	// run-to-run from a seed keyed on (seed const, capacity, access index).
	CachePolicyRandom
	// CachePolicyOPT is the offline Belady upper bound — a non-implementable oracle, not an
	// online policy, included only to normalize the other four.
	CachePolicyOPT
)

// String renders the arm name. Unknown values render "unknown".
func (p CachePolicy) String() string {
	switch p {
	case CachePolicyLRU:
		return "lru"
	case CachePolicyLFU:
		return "lfu"
	case CachePolicyHeat:
		return "heat"
	case CachePolicyRandom:
		return "random"
	case CachePolicyOPT:
		return "opt"
	default:
		return "unknown"
	}
}

// CachePolicyArms returns the five arms in a fixed deterministic order.
func CachePolicyArms() []CachePolicy {
	return []CachePolicy{
		CachePolicyLRU,
		CachePolicyLFU,
		CachePolicyHeat,
		CachePolicyRandom,
		CachePolicyOPT,
	}
}

// CachePolicyStream is a flat, deterministic access stream of unit-sized expert groups.
// Groups is processed left to right; each entry is one access.
type CachePolicyStream struct {
	Name   string
	Groups []ExpertGroup
}

// CachePolicyRow is one arm's result on one stream at one capacity.
type CachePolicyRow struct {
	Policy       CachePolicy `json:"policy"`
	Hits         int         `json:"hits"`
	Misses       int         `json:"misses"`
	Accesses     int         `json:"accesses"`
	HitRate      float64     `json:"hit_rate"`
	HitRateKnown bool        `json:"hit_rate_known"`
	Evictions    int         `json:"evictions"`
	VersusOPT    float64     `json:"versus_opt"`
}

// CachePolicyBenchReceipt is the per-arm L1 receipt for one stream at one capacity. It is
// weight-free synthetic evidence: it does NOT model transfer or dequant latency, GPU
// execution, I/O bandwidth, throughput, output parity, or any real routing trace. Every
// field is a modeled count over the supplied stream, marked
// Measurement == CachePolicyMeasurementSimulated. The shipped evictor default is
// unchanged by this harness — it only compares policies offline.
type CachePolicyBenchReceipt struct {
	Schema      string           `json:"schema"`
	Measurement string           `json:"measurement"`
	Stream      string           `json:"stream"`
	Capacity    int              `json:"capacity"`
	AccessCount int              `json:"access_count"`
	OPTExact    bool             `json:"opt_exact"`
	OPTHits     int              `json:"opt_hits"`
	Rows        []CachePolicyRow `json:"rows"`
}

// CachePolicyReplay replays one stream across the five arms at matched capacity and
// returns the per-arm receipt. A non-positive capacity fails closed with
// ErrInvalidTraceCapacity. The OPT arm is an offline Belady upper bound computed by
// replayoracle.Belady; its Evictions are left 0 by construction because an
// oracle does not pay online evictions, and VersusOPT for OPT is therefore 1.0.
//
// On a stream with more than 63 distinct groups the oracle falls back to a
// farthest-next-use greedy upper bound and sets OPTExact=false: VersusOPT is then a
// margin against that heuristic, not the exact offline optimum. Callers that require an
// exact bound must check OPTExact (or size the stream to <=63 distinct groups).
func CachePolicyReplay(stream CachePolicyStream, capacity int) (CachePolicyBenchReceipt, error) {
	if capacity <= 0 {
		return CachePolicyBenchReceipt{}, ErrInvalidTraceCapacity
	}

	events := cachePolicyEvents(stream)
	oracle := replayoracle.Belady(events, capacity)

	receipt := CachePolicyBenchReceipt{
		Schema:      CachePolicyBenchSchema,
		Measurement: CachePolicyMeasurementSimulated,
		Stream:      stream.Name,
		Capacity:    capacity,
		AccessCount: len(stream.Groups),
		OPTExact:    oracle.Exact,
		OPTHits:     oracle.HitTokens,
		Rows:        make([]CachePolicyRow, 0, len(CachePolicyArms())),
	}

	for _, policy := range CachePolicyArms() {
		row := cachePolicyReplayArm(policy, stream.Groups, capacity, events, oracle)
		row.VersusOPT = mathx.AgainstOracle(row.Hits, receipt.OPTHits)
		receipt.Rows = append(receipt.Rows, row)
	}
	return receipt, nil
}

// CachePolicyReplayOnline replays only the four implementable arms (LRU/LFU/heat/random),
// skipping the offline OPT oracle entirely. It returns the same rows, in the same order,
// and with the same Hits/Misses/Accesses/HitRate/Evictions as CachePolicyReplay would for
// those arms — but without paying replayoracle.Belady, whose exact DP is
// exponential in the distinct-group count. Use it to compare arms against each other (or
// against a shipped simulator) when no oracle-normalized margin is needed; use
// CachePolicyReplay when the margin versus OPT is the result.
//
// Each returned row's VersusOPT is ZERO because no oracle was computed; it is not the
// AgainstOracle ratio CachePolicyReplay would report. Callers needing VersusOPT must use
// CachePolicyReplay. A non-positive capacity fails closed with ErrInvalidTraceCapacity.
func CachePolicyReplayOnline(stream CachePolicyStream, capacity int) ([]CachePolicyRow, error) {
	if capacity <= 0 {
		return nil, ErrInvalidTraceCapacity
	}
	rows := make([]CachePolicyRow, 0, len(CachePolicyArms())-1)
	for _, policy := range CachePolicyArms() {
		if policy == CachePolicyOPT {
			continue
		}
		rows = append(rows, cachePolicyReplayArm(policy, stream.Groups, capacity, nil, replayoracle.Result{}))
	}
	return rows, nil
}

// cachePolicyEvents maps a stream to unit-sized replayoracle.Event rows, assigning a
// stable int SpanID per distinct ExpertGroup in first-touch order (so the OPT oracle and
// the LRU cross-check key on the same identity).
func cachePolicyEvents(stream CachePolicyStream) []replayoracle.Event {
	spanIDs := make(map[ExpertGroup]int)
	events := make([]replayoracle.Event, 0, len(stream.Groups))
	for _, group := range stream.Groups {
		id, ok := spanIDs[group]
		if !ok {
			id = len(spanIDs)
			spanIDs[group] = id
		}
		events = append(events, replayoracle.Event{SpanID: id, Tokens: 1})
	}
	return events
}

// cachePolicyReplayArm replays one arm over the unit-sized groups. Every victim choice
// uses an explicit sort or a deterministic scan with a documented tie-break; Go map
// iteration is never allowed to decide a victim.
func cachePolicyReplayArm(policy CachePolicy, groups []ExpertGroup, capacity int, events []replayoracle.Event, oracle replayoracle.Result) CachePolicyRow {
	row := CachePolicyRow{Policy: policy, Accesses: len(groups)}
	if policy == CachePolicyOPT {
		row.Hits = oracle.HitTokens
		row.Misses = len(groups) - oracle.HitTokens
		if row.Misses < 0 {
			row.Misses = 0
		}
		row.Evictions = 0
		row.HitRate, row.HitRateKnown = cachePolicyHitRate(row.Hits, row.Accesses)
		return row
	}

	switch policy {
	case CachePolicyLRU:
		hits, miss, evictions := replayOnlineLRU(groups, capacity)
		row.Hits, row.Misses, row.Evictions = hits, miss, evictions
	case CachePolicyLFU:
		hits, miss, evictions := replayOnlineLFU(groups, capacity)
		row.Hits, row.Misses, row.Evictions = hits, miss, evictions
	case CachePolicyHeat:
		hits, miss, evictions := replayOnlineHeat(groups, capacity)
		row.Hits, row.Misses, row.Evictions = hits, miss, evictions
	case CachePolicyRandom:
		hits, miss, evictions := replayOnlineRandom(groups, capacity)
		row.Hits, row.Misses, row.Evictions = hits, miss, evictions
	}
	row.HitRate, row.HitRateKnown = cachePolicyHitRate(row.Hits, row.Accesses)
	return row
}

// cachePolicyHitRate reports Hits/Accesses, and whether the rate is known. An empty
// stream is reported as unknown rather than a phantom 0% confidence.
func cachePolicyHitRate(hits, accesses int) (float64, bool) {
	if accesses <= 0 {
		return 0, false
	}
	return float64(hits) / float64(accesses), true
}

// residentEntry is the per-group resident bookkeeping every online arm maintains.
type residentEntry struct {
	group   ExpertGroup
	lastUse uint64
}

// replayOnlineLRU is the reference arm. It is structure-for-structure the same recency
// policy as SimulateExpertCache (smallest lastUse wins, groupLess breaks ties), so its
// Hits/Misses match that baseline exactly on the same single-group stream.
func replayOnlineLRU(groups []ExpertGroup, capacity int) (hits, misses, evictions int) {
	resident := make(map[ExpertGroup]residentEntry, capacity)
	var clock uint64
	for _, group := range groups {
		clock++
		if entry, ok := resident[group]; ok {
			hits++
			entry.lastUse = clock
			resident[group] = entry
			continue
		}
		misses++
		if len(resident) == capacity {
			victim := pickLeastByKey(resident, func(e residentEntry) uint64 { return e.lastUse })
			delete(resident, victim)
			evictions++
		}
		resident[group] = residentEntry{group: group, lastUse: clock}
	}
	return hits, misses, evictions
}

// replayOnlineLFU evicts the resident group with the smallest lifetime access count
// (counted whether or not the group is resident). Ties break on oldest lastUse, then
// groupLess.
func replayOnlineLFU(groups []ExpertGroup, capacity int) (hits, misses, evictions int) {
	resident := make(map[ExpertGroup]residentEntry, capacity)
	useCount := make(map[ExpertGroup]uint64)
	var clock uint64
	for _, group := range groups {
		clock++
		useCount[group]++
		if entry, ok := resident[group]; ok {
			hits++
			entry.lastUse = clock
			resident[group] = entry
			continue
		}
		misses++
		if len(resident) == capacity {
			victim := pickLeastByKey(resident, func(e residentEntry) uint64 { return useCount[e.group] })
			delete(resident, victim)
			evictions++
		}
		resident[group] = residentEntry{group: group, lastUse: clock}
	}
	return hits, misses, evictions
}

// replayOnlineHeat evicts the resident group with the smallest decaying heat. Heat starts
// at 0 for a never-seen group, gains +1 on every touch, and every heatDecayEvery accesses
// every group's heat is halved (integer halving). Ghost heat is retained for evicted
// groups, so a re-admitted group resumes with its prior signal. Ties break on oldest
// lastUse, then groupLess.
func replayOnlineHeat(groups []ExpertGroup, capacity int) (hits, misses, evictions int) {
	resident := make(map[ExpertGroup]residentEntry, capacity)
	heat := make(map[ExpertGroup]int)
	var clock uint64
	for _, group := range groups {
		clock++
		if clock > 1 && (clock-1)%heatDecayEvery == 0 {
			decayHeat(heat)
		}
		heat[group]++
		if entry, ok := resident[group]; ok {
			hits++
			entry.lastUse = clock
			resident[group] = entry
			continue
		}
		misses++
		if len(resident) == capacity {
			victim := pickLeastByKey(resident, func(e residentEntry) uint64 { return uint64(heat[e.group]) })
			delete(resident, victim)
			evictions++
		}
		resident[group] = residentEntry{group: group, lastUse: clock}
	}
	return hits, misses, evictions
}

// decayHeat halves every tracked group's heat. Iteration order is irrelevant: the
// transform is applied uniformly to every entry.
func decayHeat(heat map[ExpertGroup]int) {
	for group, h := range heat {
		heat[group] = h / 2
	}
}

// replayOnlineRandom evicts a deterministic pseudo-random resident group. The victim is
// chosen by hashing (cachePolicyRandomSeed, capacity, access index) into a uniformly
// sorted slice of resident groups; there is no global RNG, so the arm is reproducible
// run-to-run and across platforms.
func replayOnlineRandom(groups []ExpertGroup, capacity int) (hits, misses, evictions int) {
	resident := make(map[ExpertGroup]residentEntry, capacity)
	var clock uint64
	for _, group := range groups {
		clock++
		if entry, ok := resident[group]; ok {
			hits++
			entry.lastUse = clock
			resident[group] = entry
			continue
		}
		misses++
		if len(resident) == capacity {
			sorted := sortedResidentKeys(resident)
			h := mix(cachePolicyRandomSeed, uint64(capacity)+clock)
			victim := sorted[int(h%uint64(len(sorted)))]
			delete(resident, victim)
			evictions++
		}
		resident[group] = residentEntry{group: group, lastUse: clock}
	}
	return hits, misses, evictions
}

// cachePolicyRandomSeed is the fixed splitmix64 seed for the random arm, so a replay is
// reproducible run-to-run.
const cachePolicyRandomSeed uint64 = 0x9E3779B97F4A7C15

// pickLeastByKey scans resident and returns the group minimizing key(e). Ties break on
// the older lastUse, then groupLess — an explicit deterministic scan, never Go map order.
func pickLeastByKey(resident map[ExpertGroup]residentEntry, key func(residentEntry) uint64) ExpertGroup {
	var victim ExpertGroup
	var bestKey, bestLast uint64
	first := true
	for group, entry := range resident {
		k := key(entry)
		if first ||
			k < bestKey ||
			(k == bestKey && (entry.lastUse < bestLast ||
				(entry.lastUse == bestLast && groupLess(group, victim)))) {
			victim, bestKey, bestLast, first = group, k, entry.lastUse, false
		}
	}
	return victim
}

// sortedResidentKeys returns the resident groups sorted by groupLess.
func sortedResidentKeys(resident map[ExpertGroup]residentEntry) []ExpertGroup {
	out := make([]ExpertGroup, 0, len(resident))
	for group := range resident {
		out = append(out, group)
	}
	sort.Slice(out, func(i, j int) bool { return groupLess(out[i], out[j]) })
	return out
}

// CachePolicyMargin returns the spread of VersusOPT across the receipt's rows — the
// headroom the best arm has over the worst. It returns 0 when fewer than two rows exist.
func CachePolicyMargin(receipt CachePolicyBenchReceipt) float64 {
	if len(receipt.Rows) < 2 {
		return 0
	}
	min, max := receipt.Rows[0].VersusOPT, receipt.Rows[0].VersusOPT
	for _, row := range receipt.Rows[1:] {
		if row.VersusOPT < min {
			min = row.VersusOPT
		}
		if row.VersusOPT > max {
			max = row.VersusOPT
		}
	}
	return max - min
}

// SyntheticCachePolicyStream builds a deterministic Zipf-skewed stream of expert accesses
// over `layers` x `experts` groups, `accesses` accesses wide, each access a topK selection
// at one layer. It is seed-fixed: identical args yield an identical stream. Shape params
// are validated (layers/experts/topK/accesses positive, topK <= experts) and fail closed
// with ErrInvalidTraceShape. The stream is synthetic evidence only; it is not a real
// routing trace.
func SyntheticCachePolicyStream(name string, seed int64, layers, experts, topK, accesses int) (CachePolicyStream, error) {
	if layers <= 0 || experts <= 0 || topK <= 0 || topK > experts || accesses <= 0 {
		return CachePolicyStream{}, ErrInvalidTraceShape
	}

	stream := CachePolicyStream{Name: name, Groups: make([]ExpertGroup, 0, accesses*topK)}
	base := uint64(seed)
	// Precompute the Zipf(1) cumulative distribution over expert ranks: rank r is drawn
	// with probability ~ (1/(r+1)) / H, so low-index experts dominate. Inversion of one
	// uniform 64-bit hash gives a deterministic draw with no rejection loop.
	weights := make([]float64, experts)
	total := 0.0
	for e := 0; e < experts; e++ {
		weights[e] = 1.0 / float64(e+1)
		total += weights[e]
	}
	cumulative := make([]float64, experts)
	running := 0.0
	for e := 0; e < experts; e++ {
		running += weights[e] / total
		cumulative[e] = running
	}

	for a := 0; a < accesses; a++ {
		layer := int(mix(base^0xA5A5A5A5, uint64(a)+1) % uint64(layers))
		selected := make([]int, 0, topK)
		chosen := make(map[int]struct{}, topK)
		for k := 0; len(selected) < topK; k++ {
			u := float64(mix(base^uint64(k+1), uint64(a+1)*uint64(experts)+uint64(k+1))>>11) / float64(1<<53)
			rank := zipfRank(cumulative, u)
			// Probe forward deterministically on a duplicate draw; at most `experts`
			// steps, so the loop terminates. With topK <= experts a free rank exists.
			for steps := 0; steps < experts; steps++ {
				if _, dup := chosen[rank]; !dup {
					break
				}
				rank = (rank + 1) % experts
			}
			chosen[rank] = struct{}{}
			selected = append(selected, rank)
		}
		// Emit the selection sorted by expert index so access order is a stable function
		// of the selection set, not of the ranking scratch order.
		sort.Ints(selected)
		for _, expert := range selected {
			stream.Groups = append(stream.Groups, ExpertGroup{Layer: layer, Expert: expert})
		}
	}
	return stream, nil
}

// zipfRank inverts the cumulative Zipf distribution: it returns the smallest rank whose
// cumulative mass reaches u. cumulative is non-decreasing with cumulative[last] == 1.
func zipfRank(cumulative []float64, u float64) int {
	idx := sort.SearchFloat64s(cumulative, u)
	if idx >= len(cumulative) {
		return len(cumulative) - 1
	}
	return idx
}
