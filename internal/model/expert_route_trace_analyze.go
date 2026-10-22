package model

import (
	"bufio"
	"container/heap"
	"container/list"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math/bits"
	"sort"
	"strings"
)

// ExpertRouteAnalysisSchema names the analyzer's report.
const ExpertRouteAnalysisSchema = "fak.expert-route.analysis/v1"

// StrixMoECacheTraceSchema is the fak-strix moe-cache-bench --trace input
// format; ExpertRouteTraceToStrix emits it so a captured route trace replays
// through that bench unchanged.
const StrixMoECacheTraceSchema = "fak.strix_moe_cache_trace/v1"

// ErrExpertRouteTrace marks a malformed or empty route trace.
var ErrExpertRouteTrace = errors.New("model: invalid expert route trace")

// ExpertRouteTrace is a parsed JSONL route trace.
type ExpertRouteTrace struct {
	Header ExpertRouteTraceHeader
	Events []ExpertRouteEvent
}

// ReadExpertRouteTrace parses the JSONL written by the FAK_EXPERT_TRACE sink.
// Header lines (those carrying "schema") are optional and may repeat; the
// first one wins.
func ReadExpertRouteTrace(r io.Reader) (ExpertRouteTrace, error) {
	var out ExpertRouteTrace
	sc := bufio.NewScanner(r)
	sc.Buffer(make([]byte, 0, 64*1024), 16*1024*1024)
	line := 0
	for sc.Scan() {
		line++
		raw := strings.TrimSpace(sc.Text())
		if raw == "" {
			continue
		}
		var probe struct {
			Schema  string     `json:"schema"`
			Weights []*float32 `json:"weights"`
		}
		if err := json.Unmarshal([]byte(raw), &probe); err != nil {
			return out, fmt.Errorf("%w: line %d: %v", ErrExpertRouteTrace, line, err)
		}
		if probe.Schema != "" {
			if probe.Schema != ExpertRouteTraceSchema {
				return out, fmt.Errorf("%w: line %d: schema %q, want %q", ErrExpertRouteTrace, line, probe.Schema, ExpertRouteTraceSchema)
			}
			if out.Header.Schema == "" {
				if err := json.Unmarshal([]byte(raw), &out.Header); err != nil {
					return out, fmt.Errorf("%w: line %d: %v", ErrExpertRouteTrace, line, err)
				}
			}
			continue
		}
		var ev ExpertRouteEvent
		if err := json.Unmarshal([]byte(raw), &ev); err != nil {
			return out, fmt.Errorf("%w: line %d: %v", ErrExpertRouteTrace, line, err)
		}
		if ev.Layer < 0 || len(ev.Experts) == 0 {
			return out, fmt.Errorf("%w: line %d: layer=%d experts=%d", ErrExpertRouteTrace, line, ev.Layer, len(ev.Experts))
		}
		if probe.Weights != nil {
			if len(probe.Weights) != len(ev.Experts) {
				return out, fmt.Errorf("%w: line %d: weights=%d experts=%d", ErrExpertRouteTrace, line, len(probe.Weights), len(ev.Experts))
			}
			for _, weight := range probe.Weights {
				if weight == nil {
					return out, fmt.Errorf("%w: line %d: null gate weight", ErrExpertRouteTrace, line)
				}
			}
		}
		for _, e := range ev.Experts {
			if e < 0 {
				return out, fmt.Errorf("%w: line %d: negative expert id %d", ErrExpertRouteTrace, line, e)
			}
		}
		out.Events = append(out.Events, ev)
	}
	if err := sc.Err(); err != nil {
		return out, fmt.Errorf("%w: %v", ErrExpertRouteTrace, err)
	}
	if len(out.Events) == 0 {
		return out, fmt.Errorf("%w: no route events", ErrExpertRouteTrace)
	}
	return out, nil
}

// ExpertRouteAnalyzeOptions tunes the analyzer. Zero values select defaults.
type ExpertRouteAnalyzeOptions struct {
	Windows    []int   // token windows for unique-expert counts; default 1,8,32,128
	TopN       []int   // concentration cut-offs; default 8,16,32,64
	Capacities []int   // per-layer cache capacities in experts; default 1..256 by powers of two
	Coverage   float64 // hot-set pick coverage per layer; default 0.8
	Phase      string  // keep only this phase ("" keeps all)
	KeepRepeat bool    // keep re-routes of an already-seen (token, layer)
	Ranks      []int   // expert-parallel rank counts for per-step balance; none skips it
}

// ExpertRouteWindowStat is the unique-expert count over every sliding window of
// Window consecutive routed tokens.
type ExpertRouteWindowStat struct {
	Window     int     `json:"window"`
	MeanUnique float64 `json:"mean_unique"`
	MaxUnique  int     `json:"max_unique"`
}

// ExpertRouteTopNShare is the share of picks served by the N most-picked experts.
type ExpertRouteTopNShare struct {
	N     int     `json:"n"`
	Share float64 `json:"share"`
}

// ExpertRouteLayerStats is one layer's selection profile.
type ExpertRouteLayerStats struct {
	Layer       int                     `json:"layer"`
	Tokens      int                     `json:"tokens"`
	Picks       int                     `json:"picks"`
	Unique      int                     `json:"unique"`
	Windows     []ExpertRouteWindowStat `json:"windows"`
	TopNShare   []ExpertRouteTopNShare  `json:"top_n_share"`
	HotSetSize  int                     `json:"hot_set_size"`
	TopExperts  []int                   `json:"top_experts"`
	TopExpertsN []int                   `json:"top_experts_picks"`
}

// ExpertRouteReuseBucket counts re-picks whose reuse distance (routed tokens
// since the same layer last picked the expert) fell in [Min, Max].
type ExpertRouteReuseBucket struct {
	Min   int `json:"min"`
	Max   int `json:"max"`
	Count int `json:"count"`
}

// ExpertRouteCacheRow is the hit rate of a per-layer cache holding Capacity
// experts in every layer, under LRU and the offline optimum (Belady MIN, exact
// for uniform expert sizes).
type ExpertRouteCacheRow struct {
	CapacityPerLayer int     `json:"capacity_per_layer"`
	TotalExperts     int     `json:"total_experts"`
	LRUHitRate       float64 `json:"lru_hit_rate"`
	BeladyHitRate    float64 `json:"belady_hit_rate"`
}

// ExpertRouteAnalysis is the analyzer report.
type ExpertRouteAnalysis struct {
	Schema             string                   `json:"schema"`
	Phase              string                   `json:"phase,omitempty"`
	Routes             int                      `json:"routes"`
	RepeatsDropped     int                      `json:"repeats_dropped"`
	Picks              int                      `json:"picks"`
	Layers             int                      `json:"layers"`
	ExpertsPerLayer    int                      `json:"experts_per_layer,omitempty"`
	TopK               int                      `json:"top_k"`
	PhaseRoutes        map[string]int           `json:"phase_routes"`
	UniqueExperts      int                      `json:"unique_experts"`
	MeanUniquePerLayer float64                  `json:"mean_unique_per_layer"`
	MaxUniquePerLayer  int                      `json:"max_unique_per_layer"`
	Windows            []ExpertRouteWindowStat  `json:"windows"`
	TopNShare          []ExpertRouteTopNShare   `json:"top_n_share"`
	ColdPicks          int                      `json:"cold_picks"`
	Reuse              []ExpertRouteReuseBucket `json:"reuse_distance"`
	Cache              []ExpertRouteCacheRow    `json:"cache"`
	Coverage           float64                  `json:"hot_set_coverage"`
	HotSetSize         int                      `json:"hot_set_size"`
	HotSet             []ExpertUsageCount       `json:"hot_set"`
	PerLayer           []ExpertRouteLayerStats  `json:"per_layer"`
	RankBalance        []ExpertRouteRankBalance `json:"rank_balance,omitempty"`
}

func (o ExpertRouteAnalyzeOptions) withDefaults() ExpertRouteAnalyzeOptions {
	if len(o.Windows) == 0 {
		o.Windows = []int{1, 8, 32, 128}
	}
	if len(o.TopN) == 0 {
		o.TopN = []int{8, 16, 32, 64}
	}
	if len(o.Capacities) == 0 {
		o.Capacities = []int{1, 2, 4, 8, 16, 32, 64, 128, 256}
	}
	if o.Coverage <= 0 || o.Coverage > 1 {
		o.Coverage = 0.8
	}
	return o
}

// selectRoutes applies the phase filter and drops repeated (token, layer)
// re-routes from full-history recomputes so every routed token counts once.
func (t ExpertRouteTrace) selectRoutes(opts ExpertRouteAnalyzeOptions) ([]ExpertRouteEvent, int) {
	type key struct{ token, layer int }
	seen := map[key]bool{}
	out := make([]ExpertRouteEvent, 0, len(t.Events))
	dropped := 0
	for _, ev := range t.Events {
		if !opts.KeepRepeat && ev.Token >= 0 {
			k := key{ev.Token, ev.Layer}
			if seen[k] {
				dropped++
				continue
			}
			seen[k] = true
		}
		if opts.Phase != "" && ev.Phase != opts.Phase {
			continue
		}
		out = append(out, ev)
	}
	return out, dropped
}

// AnalyzeExpertRouteTrace reports how many routed experts a trace needs and
// which are prewarmable. It is pure: same trace and options, same report.
func AnalyzeExpertRouteTrace(t ExpertRouteTrace, opts ExpertRouteAnalyzeOptions) (ExpertRouteAnalysis, error) {
	opts = opts.withDefaults()
	routes, dropped := t.selectRoutes(opts)
	if len(routes) == 0 {
		return ExpertRouteAnalysis{}, fmt.Errorf("%w: no routes after phase filter %q", ErrExpertRouteTrace, opts.Phase)
	}
	rep := ExpertRouteAnalysis{
		Schema: ExpertRouteAnalysisSchema, Phase: opts.Phase, Routes: len(routes), RepeatsDropped: dropped,
		ExpertsPerLayer: t.Header.Experts, TopK: t.Header.TopK, PhaseRoutes: map[string]int{}, Coverage: opts.Coverage,
	}
	byLayer := map[int][][]int{}
	for _, ev := range routes {
		rep.PhaseRoutes[ev.Phase]++
		set := dedupeInts(ev.Experts)
		rep.Picks += len(set)
		byLayer[ev.Layer] = append(byLayer[ev.Layer], set)
		if len(set) > rep.TopK && t.Header.TopK == 0 {
			rep.TopK = len(set)
		}
	}
	layers := make([]int, 0, len(byLayer))
	for l := range byLayer {
		layers = append(layers, l)
	}
	sort.Ints(layers)
	rep.Layers = len(layers)

	winSum := make([]float64, len(opts.Windows))
	winMax := make([]int, len(opts.Windows))
	topHits := make([]int, len(opts.TopN))
	reuse := map[int]int{}
	lruHits := make([]int, len(opts.Capacities))
	minHits := make([]int, len(opts.Capacities))
	for _, l := range layers {
		seq := byLayer[l]
		st := ExpertRouteLayerStats{Layer: l, Tokens: len(seq)}
		counts := map[int]int{}
		last := map[int]int{}
		var flat []int
		for i, set := range seq {
			for _, e := range set {
				st.Picks++
				counts[e]++
				flat = append(flat, e)
				if p, ok := last[e]; ok {
					reuse[bits.Len(uint(i-p))-1]++
				} else {
					rep.ColdPicks++
				}
				last[e] = i
			}
		}
		st.Unique = len(counts)
		rep.UniqueExperts += st.Unique
		rep.MeanUniquePerLayer += float64(st.Unique)
		if st.Unique > rep.MaxUniquePerLayer {
			rep.MaxUniquePerLayer = st.Unique
		}
		for wi, w := range opts.Windows {
			mean, max := slidingUnique(seq, w)
			st.Windows = append(st.Windows, ExpertRouteWindowStat{Window: w, MeanUnique: mean, MaxUnique: max})
			winSum[wi] += mean
			if max > winMax[wi] {
				winMax[wi] = max
			}
		}
		ranked := rankExperts(counts)
		for ni, n := range opts.TopN {
			hit := 0
			for i := 0; i < n && i < len(ranked); i++ {
				hit += ranked[i].Count
			}
			topHits[ni] += hit
			st.TopNShare = append(st.TopNShare, ExpertRouteTopNShare{N: n, Share: routeRatio(hit, st.Picks)})
		}
		covered := 0
		for _, r := range ranked {
			if routeRatio(covered, st.Picks) >= opts.Coverage {
				break
			}
			covered += r.Count
			st.HotSetSize++
			rep.HotSet = append(rep.HotSet, ExpertUsageCount{Layer: l, Expert: r.Expert, Count: float64(r.Count)})
		}
		rep.HotSetSize += st.HotSetSize
		for i := 0; i < len(ranked) && i < 8; i++ {
			st.TopExperts = append(st.TopExperts, ranked[i].Expert)
			st.TopExpertsN = append(st.TopExpertsN, ranked[i].Count)
		}
		for ci, c := range opts.Capacities {
			lruHits[ci] += lruUnitHits(flat, c)
			minHits[ci] += beladyUnitHits(flat, c)
		}
		rep.PerLayer = append(rep.PerLayer, st)
	}
	rep.MeanUniquePerLayer /= float64(rep.Layers)
	for wi, w := range opts.Windows {
		rep.Windows = append(rep.Windows, ExpertRouteWindowStat{Window: w, MeanUnique: winSum[wi] / float64(rep.Layers), MaxUnique: winMax[wi]})
	}
	for ni, n := range opts.TopN {
		rep.TopNShare = append(rep.TopNShare, ExpertRouteTopNShare{N: n, Share: routeRatio(topHits[ni], rep.Picks)})
	}
	buckets := make([]int, 0, len(reuse))
	for b := range reuse {
		buckets = append(buckets, b)
	}
	sort.Ints(buckets)
	for _, b := range buckets {
		rep.Reuse = append(rep.Reuse, ExpertRouteReuseBucket{Min: 1 << b, Max: 1<<(b+1) - 1, Count: reuse[b]})
	}
	for ci, c := range opts.Capacities {
		rep.Cache = append(rep.Cache, ExpertRouteCacheRow{
			CapacityPerLayer: c, TotalExperts: c * rep.Layers,
			LRUHitRate: routeRatio(lruHits[ci], rep.Picks), BeladyHitRate: routeRatio(minHits[ci], rep.Picks),
		})
	}
	if len(opts.Ranks) > 0 {
		rb, err := expertRouteRankBalance(byLayer, expertRouteExperts(t, routes), opts.Ranks)
		if err != nil {
			return ExpertRouteAnalysis{}, err
		}
		rep.RankBalance = rb
	}
	return rep, nil
}

type expertRank struct{ Expert, Count int }

func rankExperts(counts map[int]int) []expertRank {
	out := make([]expertRank, 0, len(counts))
	for e, c := range counts {
		out = append(out, expertRank{e, c})
	}
	sort.Slice(out, func(i, j int) bool {
		if out[i].Count != out[j].Count {
			return out[i].Count > out[j].Count
		}
		return out[i].Expert < out[j].Expert
	})
	return out
}

func dedupeInts(in []int) []int {
	out := make([]int, 0, len(in))
	seen := make(map[int]bool, len(in))
	for _, v := range in {
		if !seen[v] {
			seen[v] = true
			out = append(out, v)
		}
	}
	return out
}

func routeRatio(n, d int) float64 {
	if d <= 0 {
		return 0
	}
	return float64(n) / float64(d)
}

// slidingUnique is the mean and max distinct-expert count over every window of
// w consecutive routed tokens; a sequence shorter than w is one window.
func slidingUnique(seq [][]int, w int) (float64, int) {
	if w <= 0 || len(seq) == 0 {
		return 0, 0
	}
	if w > len(seq) {
		w = len(seq)
	}
	counts := map[int]int{}
	add := func(set []int, d int) {
		for _, e := range set {
			counts[e] += d
			if counts[e] == 0 {
				delete(counts, e)
			}
		}
	}
	for i := 0; i < w; i++ {
		add(seq[i], 1)
	}
	sum, max, n := len(counts), len(counts), 1
	for i := w; i < len(seq); i++ {
		add(seq[i], 1)
		add(seq[i-w], -1)
		u := len(counts)
		sum += u
		n++
		if u > max {
			max = u
		}
	}
	return float64(sum) / float64(n), max
}

func lruUnitHits(accesses []int, capacity int) int {
	if capacity <= 0 {
		return 0
	}
	order := list.New()
	at := map[int]*list.Element{}
	hits := 0
	for _, e := range accesses {
		if el, ok := at[e]; ok {
			hits++
			order.MoveToFront(el)
			continue
		}
		if order.Len() >= capacity {
			victim := order.Back()
			delete(at, victim.Value.(int))
			order.Remove(victim)
		}
		at[e] = order.PushFront(e)
	}
	return hits
}

type nextUseItem struct{ next, expert int }
type nextUseHeap []nextUseItem

func (h nextUseHeap) Len() int           { return len(h) }
func (h nextUseHeap) Less(i, j int) bool { return h[i].next > h[j].next }
func (h nextUseHeap) Swap(i, j int)      { h[i], h[j] = h[j], h[i] }
func (h *nextUseHeap) Push(x any)        { *h = append(*h, x.(nextUseItem)) }
func (h *nextUseHeap) Pop() any {
	old := *h
	it := old[len(old)-1]
	*h = old[:len(old)-1]
	return it
}

// beladyUnitHits is Belady's MIN (evict the resident used furthest in the
// future), optimal for demand paging with uniform item sizes.
func beladyUnitHits(accesses []int, capacity int) int {
	if capacity <= 0 {
		return 0
	}
	n := len(accesses)
	next := make([]int, n)
	seen := map[int]int{}
	for i := n - 1; i >= 0; i-- {
		if j, ok := seen[accesses[i]]; ok {
			next[i] = j
		} else {
			next[i] = n + accesses[i]
		}
		seen[accesses[i]] = i
	}
	resident := map[int]int{}
	h := &nextUseHeap{}
	hits := 0
	for i, e := range accesses {
		if _, ok := resident[e]; ok {
			hits++
		} else if len(resident) >= capacity {
			for {
				top := heap.Pop(h).(nextUseItem)
				if cur, ok := resident[top.expert]; ok && cur == top.next {
					delete(resident, top.expert)
					break
				}
			}
		}
		resident[e] = next[i]
		heap.Push(h, nextUseItem{next: next[i], expert: e})
	}
	return hits
}

// StrixMoECacheEvent and StrixMoECacheTrace mirror fak-strix moe-cache-bench's
// --trace input.
type StrixMoECacheEvent struct {
	Layer  int   `json:"layer"`
	Expert int   `json:"expert"`
	Bytes  int64 `json:"bytes"`
}

type StrixMoECacheTrace struct {
	Schema      string               `json:"schema"`
	Layers      int                  `json:"layers"`
	Experts     int                  `json:"experts"`
	TopK        int                  `json:"top_k"`
	MoeInter    int                  `json:"moe_inter"`
	ExpertBytes int64                `json:"expert_bytes"`
	Events      []StrixMoECacheEvent `json:"events"`
}

// ExpertRouteTraceToStrix flattens the selected routes, in access order, into
// the moe-cache-bench trace with every expert costing expertBytes.
func ExpertRouteTraceToStrix(t ExpertRouteTrace, opts ExpertRouteAnalyzeOptions, expertBytes int64) (StrixMoECacheTrace, error) {
	if expertBytes <= 0 {
		return StrixMoECacheTrace{}, fmt.Errorf("%w: expert bytes must be positive", ErrExpertRouteTrace)
	}
	routes, _ := t.selectRoutes(opts)
	out := StrixMoECacheTrace{Schema: StrixMoECacheTraceSchema, Layers: t.Header.Layers, Experts: t.Header.Experts, TopK: t.Header.TopK, ExpertBytes: expertBytes}
	maxLayer, maxExpert := -1, -1
	for _, ev := range routes {
		maxLayer = maxInt(maxLayer, ev.Layer)
		set := dedupeInts(ev.Experts)
		for _, e := range set {
			maxExpert = maxInt(maxExpert, e)
			out.Events = append(out.Events, StrixMoECacheEvent{Layer: ev.Layer, Expert: e, Bytes: expertBytes})
		}
		out.TopK = maxInt(out.TopK, len(set))
	}
	if len(out.Events) == 0 {
		return StrixMoECacheTrace{}, fmt.Errorf("%w: no routes after phase filter %q", ErrExpertRouteTrace, opts.Phase)
	}
	out.Layers = maxInt(out.Layers, maxLayer+1)
	out.Experts = maxInt(out.Experts, maxExpert+1)
	return out, nil
}
