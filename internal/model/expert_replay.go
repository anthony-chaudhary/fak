package model

import (
	"errors"
	"fmt"
	"math"
	"math/rand"
	"sort"
	"sync"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// ExpertReplayTraceSchema is the host-free schema for expert-residency policy replay.
const ExpertReplayTraceSchema = "fak.expert-residency.trace/v1"

// ExpertRouteObserver receives the final routed expert ids for one layer invocation.
// experts is a fresh copy owned by the observer. The observer cannot mutate route weights
// or expert execution order.
type ExpertRouteObserver func(layer int, experts []int)

// SetExpertRouteObserver installs or clears the observed-routing trace seam. It is not
// safe to change concurrently with a forward pass; install it before the measured window.
func (m *Model) SetExpertRouteObserver(obs ExpertRouteObserver) { m.expertRouteObs = obs }

// ExpertRouteObserverSet reports whether observed routing is currently being captured.
func (m *Model) ExpertRouteObserverSet() bool { return m.expertRouteObs != nil }

func (m *Model) emitExpertRoute(layer int, picks []routePick) {
	if m == nil || m.expertRouteObs == nil {
		return
	}
	experts := make([]int, len(picks))
	for i, pick := range picks {
		experts[i] = pick.expert
	}
	m.expertRouteObs(layer, experts)
}

// ExpertResidentWeightBytes reports the resident quantized footprint of one routed
// expert's gate/up/down projections. It deliberately does not fall back to the f32
// manifest: the gauge models the Q4_K/Q5_K/Q6_K/Q8 bytes pagedRing would budget, not a
// four-byte-per-weight expansion. Zero means no supported quantized representation exists.
func (m *Model) ExpertResidentWeightBytes(layer, expert int) int64 {
	if m == nil || layer < 0 || expert < 0 {
		return 0
	}
	var total int64
	for _, suffix := range []string{"gate_proj.weight", "up_proj.weight", "down_proj.weight"} {
		name := expertName(layer, expert, suffix)
		switch {
		case m.q4kw != nil && m.q4kw[name] != nil:
			q := m.q4kw[name]
			total += int64(q.out) * int64(q.nblk) * int64(q4kBlockBytes)
		case m.kqw != nil && m.kqw[name] != nil:
			q := m.kqw[name]
			total += int64(q.out) * int64(q.nblk) * int64(q.kind.blockBytes())
		case m.q8w != nil && m.q8w[name] != nil:
			q := m.q8w[name]
			total += int64(len(q.q)) + int64(len(q.d))*4
		}
	}
	return total
}

// ExpertAccessTraceEvent is one routed-expert touch. Identity is (Layer, Expert):
// expert 7 in adjacent layers names different resident weights.
type ExpertAccessTraceEvent struct {
	Layer       int   `json:"layer"`
	Expert      int   `json:"expert"`
	WeightBytes int64 `json:"weight_bytes"`
}

// ExpertAccessTrace is a replayable expert-residency corpus. UnsizedTouches makes an
// unsupported/f32-only observation visible rather than silently inventing a byte size.
type ExpertAccessTrace struct {
	Schema           string                   `json:"schema"`
	Name             string                   `json:"name"`
	Source           string                   `json:"source"`
	Policy           ExpertRingEvictPolicy    `json:"policy"`
	PolicyGeneration uint64                   `json:"policy_generation"`
	BudgetBytes      int64                    `json:"budget_bytes"`
	Events           []ExpertAccessTraceEvent `json:"events"`
	UnsizedTouches   int                      `json:"unsized_touches,omitempty"`
}

// ExpertLocalityReport tests the prior behind LRU. Positive correlation means weights
// used recently also tend to be reused sooner. The null preserves layer order and touch
// volume while deterministically removing observed expert-identity ordering.
type ExpertLocalityReport struct {
	Samples                        int     `json:"samples"`
	RecencyNextUseCorrelation      float64 `json:"recency_next_use_correlation"`
	LayerSequentialNullSamples     int     `json:"layer_sequential_null_samples"`
	LayerSequentialNullCorrelation float64 `json:"layer_sequential_null_correlation"`
	ObservedMinusNull              float64 `json:"observed_minus_null"`
}

// ExpertReplayReport is the quality-not-rate gauge for pagedRing's LRU policy.
// PagedRingLRU is explicit so consumers need not know compute's policy enum.
type ExpertReplayReport struct {
	Name           string                                           `json:"name"`
	Source         string                                           `json:"source"`
	BudgetBytes    int64                                            `json:"budget_bytes"`
	PagedRingLRU   compute.KVReplayResult                           `json:"paged_ring_lru"`
	Policies       map[compute.KVEvictPolicy]compute.KVReplayResult `json:"policies"`
	Oracle         compute.KVReplayOracleResult                     `json:"oracle"`
	Locality       ExpertLocalityReport                             `json:"locality"`
	UnsizedTouches int                                              `json:"unsized_touches,omitempty"`
}

// ExpertTraceRecorder turns the live router observer into an immutable replay trace.
type ExpertTraceRecorder struct {
	mu    sync.Mutex
	model *Model
	trace ExpertAccessTrace
}

// NewExpertTraceRecorder creates an observed-routing recorder. Install Observer with
// SetExpertRouteObserver for the measured forward window, then call Snapshot.
func NewExpertTraceRecorder(m *Model, name string, budgetBytes int64) *ExpertTraceRecorder {
	if name == "" {
		name = "observed-expert-routing"
	}
	return &ExpertTraceRecorder{model: m, trace: ExpertAccessTrace{
		Schema: ExpertReplayTraceSchema, Name: name, Source: "observed-routing", BudgetBytes: budgetBytes,
	}}
}

// Observer records an emitted route and sizes it from resident quantized payloads.
func (r *ExpertTraceRecorder) Observer(layer int, experts []int) {
	if r == nil {
		return
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	for _, expert := range experts {
		bytes := r.model.ExpertResidentWeightBytes(layer, expert)
		if bytes <= 0 {
			r.trace.UnsizedTouches++
			continue
		}
		r.trace.Events = append(r.trace.Events, ExpertAccessTraceEvent{Layer: layer, Expert: expert, WeightBytes: bytes})
	}
}

// Snapshot returns a deep copy safe to retain after capture ends.
func (r *ExpertTraceRecorder) Snapshot() ExpertAccessTrace {
	if r == nil {
		return ExpertAccessTrace{}
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	out := r.trace
	out.Events = append([]ExpertAccessTraceEvent(nil), r.trace.Events...)
	return out
}

// ExpertReplaySyntheticOptions controls the deterministic layer-sequential CI corpus.
type ExpertReplaySyntheticOptions struct {
	Name        string
	Seed        int64
	Tokens      int
	Layers      int
	Experts     int
	TopK        int
	WeightBytes int64
	BudgetBytes int64
}

// GenerateExpertReplaySyntheticTrace builds a stable Zipf-skewed, bimodal trace for CI.
func GenerateExpertReplaySyntheticTrace(opts ExpertReplaySyntheticOptions) ExpertAccessTrace {
	if opts.Name == "" {
		opts.Name = "synthetic-zipf-bimodal-experts"
	}
	if opts.Tokens <= 0 {
		opts.Tokens = 12
	}
	if opts.Layers <= 0 {
		opts.Layers = 3
	}
	if opts.Experts <= 0 {
		opts.Experts = 6
	}
	if opts.TopK <= 0 {
		opts.TopK = 2
	}
	if opts.TopK > opts.Experts {
		opts.TopK = opts.Experts
	}
	if opts.WeightBytes <= 0 {
		opts.WeightBytes = q4kBlockBytes
	}
	if opts.BudgetBytes <= 0 {
		opts.BudgetBytes = opts.WeightBytes * int64(opts.TopK*2)
	}

	rng := rand.New(rand.NewSource(opts.Seed))
	var zipf *rand.Zipf
	if opts.Experts > 1 {
		zipf = rand.NewZipf(rng, 1.25, 1, uint64(opts.Experts-1))
	}
	events := make([]ExpertAccessTraceEvent, 0, opts.Tokens*opts.Layers*opts.TopK)
	for token := 0; token < opts.Tokens; token++ {
		for layer := 0; layer < opts.Layers; layer++ {
			seen := map[int]bool{}
			for pick := 0; pick < opts.TopK; pick++ {
				expert := 0
				if pick > 0 || (token+layer)%5 == 3 {
					if zipf != nil {
						expert = int(zipf.Uint64())
					}
				}
				for seen[expert] {
					expert = (expert + 1) % opts.Experts
				}
				seen[expert] = true
				events = append(events, ExpertAccessTraceEvent{Layer: layer, Expert: expert, WeightBytes: opts.WeightBytes})
			}
		}
	}
	return ExpertAccessTrace{
		Schema: ExpertReplayTraceSchema, Name: opts.Name,
		Source:      fmt.Sprintf("synthetic-zipf-bimodal seed=%d", opts.Seed),
		BudgetBytes: opts.BudgetBytes, Events: events,
	}
}

// ReplayExpertAccessTrace maps resident bytes onto the existing KV simulator: its Tokens
// field is an integer capacity unit, so here one unit is one resident byte. The Belady DP,
// fallback, policy replay, and GoodDecisionRatio are reused unchanged.
func ReplayExpertAccessTrace(trace ExpertAccessTrace, policies ...compute.KVEvictPolicy) (ExpertReplayReport, error) {
	events, err := trace.replayEvents()
	if err != nil {
		return ExpertReplayReport{}, err
	}
	budget, err := replayInt(trace.BudgetBytes, "budget_bytes")
	if err != nil {
		return ExpertReplayReport{}, err
	}
	if len(policies) == 0 {
		policies = []compute.KVEvictPolicy{compute.KVEvictLRU, compute.KVEvictCostAware}
	} else {
		hasLRU := false
		for _, policy := range policies {
			hasLRU = hasLRU || policy == compute.KVEvictLRU
		}
		if !hasLRU {
			policies = append([]compute.KVEvictPolicy{compute.KVEvictLRU}, policies...)
		}
	}
	rows := compute.ReplayKVCacheMulti(events, budget, policies...)
	return ExpertReplayReport{
		Name: trace.Name, Source: trace.Source, BudgetBytes: trace.BudgetBytes,
		PagedRingLRU: rows[compute.KVEvictLRU], Policies: rows,
		Oracle: compute.BeladyKVReplayOracle(events, budget), Locality: expertLocality(trace.Events),
		UnsizedTouches: trace.UnsizedTouches,
	}, nil
}

func (t ExpertAccessTrace) replayEvents() ([]compute.KVReplayEvent, error) {
	if t.Schema != ExpertReplayTraceSchema {
		return nil, fmt.Errorf("model: expert replay schema %q, want %q", t.Schema, ExpertReplayTraceSchema)
	}
	if t.Name == "" {
		return nil, errors.New("model: expert replay name is required")
	}
	if t.Source == "" {
		return nil, errors.New("model: expert replay source is required")
	}
	if t.BudgetBytes <= 0 {
		return nil, errors.New("model: expert replay budget_bytes must be positive")
	}
	if len(t.Events) == 0 {
		return nil, errors.New("model: expert replay trace must contain a sized event")
	}
	type key struct{ layer, expert int }
	ids, sizes := map[key]int{}, map[key]int64{}
	out := make([]compute.KVReplayEvent, 0, len(t.Events))
	for i, event := range t.Events {
		if event.Layer < 0 || event.Expert < 0 || event.WeightBytes <= 0 {
			return nil, fmt.Errorf("model: invalid expert replay event %d: layer=%d expert=%d weight_bytes=%d", i, event.Layer, event.Expert, event.WeightBytes)
		}
		k := key{event.Layer, event.Expert}
		if prior, ok := sizes[k]; ok && prior != event.WeightBytes {
			return nil, fmt.Errorf("model: expert replay weight size changed for layer=%d expert=%d: %d -> %d", event.Layer, event.Expert, prior, event.WeightBytes)
		}
		sizes[k] = event.WeightBytes
		id, ok := ids[k]
		if !ok {
			id = len(ids) + 1
			ids[k] = id
		}
		units, err := replayInt(event.WeightBytes, fmt.Sprintf("events[%d].weight_bytes", i))
		if err != nil {
			return nil, err
		}
		out = append(out, compute.KVReplayEvent{SpanID: id, Tokens: units})
	}
	return out, nil
}

func replayInt(value int64, field string) (int, error) {
	maxInt := int64(^uint(0) >> 1)
	if value <= 0 || value > maxInt {
		return 0, fmt.Errorf("model: %s %d cannot be represented as a positive replay unit", field, value)
	}
	return int(value), nil
}

func expertLocality(events []ExpertAccessTraceEvent) ExpertLocalityReport {
	observed, samples := recencyNextUseCorrelation(events)
	nullCorrelation, nullSamples := recencyNextUseCorrelation(layerSequentialNull(events))
	return ExpertLocalityReport{
		Samples: samples, RecencyNextUseCorrelation: observed,
		LayerSequentialNullSamples: nullSamples, LayerSequentialNullCorrelation: nullCorrelation,
		ObservedMinusNull: observed - nullCorrelation,
	}
}

func recencyNextUseCorrelation(events []ExpertAccessTraceEvent) (float64, int) {
	type key struct{ layer, expert int }
	previous := make([]int, len(events))
	next := make([]int, len(events))
	for i := range previous {
		previous[i], next[i] = -1, -1
	}
	last := map[key]int{}
	for i, event := range events {
		k := key{event.Layer, event.Expert}
		if p, ok := last[k]; ok {
			previous[i] = p
		}
		last[k] = i
	}
	last = map[key]int{}
	for i := len(events) - 1; i >= 0; i-- {
		k := key{events[i].Layer, events[i].Expert}
		if n, ok := last[k]; ok {
			next[i] = n
		}
		last[k] = i
	}
	var xs, ys []float64
	for i := range events {
		if previous[i] >= 0 && next[i] >= 0 {
			xs = append(xs, float64(i-previous[i]))
			ys = append(ys, float64(next[i]-i))
		}
	}
	return expertPearson(xs, ys), len(xs)
}

func expertPearson(xs, ys []float64) float64 {
	if len(xs) < 2 || len(xs) != len(ys) {
		return 0
	}
	var sx, sy float64
	for i := range xs {
		sx += xs[i]
		sy += ys[i]
	}
	mx, my := sx/float64(len(xs)), sy/float64(len(ys))
	var covariance, vx, vy float64
	for i := range xs {
		dx, dy := xs[i]-mx, ys[i]-my
		covariance += dx * dy
		vx += dx * dx
		vy += dy * dy
	}
	if vx == 0 || vy == 0 {
		return 0
	}
	return covariance / math.Sqrt(vx*vy)
}

func layerSequentialNull(events []ExpertAccessTraceEvent) []ExpertAccessTraceEvent {
	byLayer := map[int]map[int]bool{}
	for _, event := range events {
		if byLayer[event.Layer] == nil {
			byLayer[event.Layer] = map[int]bool{}
		}
		byLayer[event.Layer][event.Expert] = true
	}
	experts := map[int][]int{}
	for layer, set := range byLayer {
		for expert := range set {
			experts[layer] = append(experts[layer], expert)
		}
		sort.Ints(experts[layer])
	}
	counts := map[int]int{}
	out := make([]ExpertAccessTraceEvent, len(events))
	for i, event := range events {
		ids := experts[event.Layer]
		event.Expert = ids[counts[event.Layer]%len(ids)]
		counts[event.Layer]++
		out[i] = event
	}
	return out
}
