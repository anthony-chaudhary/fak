// Package deepseekv4moe is a pure, weight-free synthetic model of DeepSeek V4 Pro's
// all-MoE dispatch, used to lock the dispatch contract and compare naive per-expert
// scheduling against grouped/fused scheduling.
//
// It is the witness named by docs/notes/DEEPSEEK-V4-MOE-DISPATCH-BASELINE-2026-07-08.md
// (issue #3018, epic #3006). It runs NO V4 weights and issues NO kernel: it drives a
// deterministic router at V4's shape (384 routed experts, top-6, 1 shared) and reasons
// in WORK-UNITS (kernel launches, token-expert GEMM rows, all-to-all token movements),
// never in fabricated wall-clock milliseconds. The per-layer "cost" figures are an
// explicitly modeled work proxy, not a measurement.
//
// What it asserts with confidence: the dispatch contract and its failure modes (bad
// top-k width, missing router weight, shared-expert mis-offload), expert-parallel
// partition correctness (an exact, disjoint cover — bit-exact at ranks=1), and that
// grouped scheduling never launches more kernels than the naive per-expert loop.
package deepseekv4moe

import (
	"errors"
	"fmt"
	"sort"
	"strings"
)

// Published V4 Pro MoE constants (DeepSeek V4 technical report; carried in the MoE
// dispatch baseline note #3018).
const (
	NumRoutedExperts = 384  // routed experts per MoE layer
	NumSharedExperts = 1    // always-on shared expert (hit rate 1.0)
	TopK             = 6    // routed experts activated per token
	ExpertHidden     = 3072 // per-expert intermediate hidden size
)

// Config is the dispatch contract a benchmark row is validated against. The zero value
// is invalid on purpose; use V4Config for the published shape.
type Config struct {
	NumRoutedExperts int
	NumSharedExperts int
	TopK             int
	HasRouterWeight  bool // false models a checkpoint missing the gate/router tensor
	// SharedTreatedAsRouted models the moe_offload mis-fit: the always-on shared expert
	// offloaded on the same LRU path as routed experts. It must stay resident, so a true
	// here is a contract violation the validator rejects.
	SharedTreatedAsRouted bool
}

// V4Config returns the published DeepSeek V4 Pro MoE shape, correctly configured.
func V4Config() Config {
	return Config{
		NumRoutedExperts:      NumRoutedExperts,
		NumSharedExperts:      NumSharedExperts,
		TopK:                  TopK,
		HasRouterWeight:       true,
		SharedTreatedAsRouted: false,
	}
}

// Failure-mode sentinels — the closed set of dispatch-contract violations the fixture
// drives on purpose.
var (
	// ErrTopKWidth indicates the top-k routed expert width is zero or exceeds routed expert count.
	ErrTopKWidth = errors.New("deepseekv4moe: top-k width out of range (need 1..NumRoutedExperts)")
	// ErrMissingRouter indicates the router/gate weight tensor is absent.
	ErrMissingRouter = errors.New("deepseekv4moe: router/gate weight missing")
	// ErrSharedMisOffload indicates the always-on shared expert was improperly flagged as offloaded.
	ErrSharedMisOffload = errors.New("deepseekv4moe: shared expert must stay resident, not offloaded as routed")
	// ErrExpertCount indicates the routed expert count is non-positive.
	ErrExpertCount = errors.New("deepseekv4moe: routed-expert count must be positive")
	// ErrSharedCount indicates the shared expert count is negative.
	ErrSharedCount = errors.New("deepseekv4moe: shared-expert count must be non-negative")
)

// Validate returns the specific contract violation, or nil when the config is a legal
// dispatch shape. This is the failure-mode gate the ticket asks the witness to lock.
func (c Config) Validate() error {
	if c.NumRoutedExperts <= 0 {
		return ErrExpertCount
	}
	if c.NumSharedExperts < 0 {
		return ErrSharedCount
	}
	if c.TopK <= 0 || c.TopK > c.NumRoutedExperts {
		return ErrTopKWidth
	}
	if !c.HasRouterWeight {
		return ErrMissingRouter
	}
	if c.SharedTreatedAsRouted {
		return ErrSharedMisOffload
	}
	return nil
}

// Routing is the outcome of driving the router over a batch of tokens: per-expert load
// counts plus the batch shape. Loads has length NumRoutedExperts.
type Routing struct {
	Tokens int
	TopK   int
	Loads  []int // routed-expert selection counts; sum == Tokens*TopK
	Shared int   // shared-expert firings == Tokens (hit rate 1.0)
}

// Route drives a deterministic top-k router over numTokens synthetic tokens at the
// config's shape. skew >= 0 biases selection toward low-index experts (0 = near
// uniform); it exists so the fixture can exercise both a balanced and an imbalanced
// assignment without any RNG — the result is a pure function of (numTokens, cfg, skew).
func Route(numTokens int, c Config, skew float64) (Routing, error) {
	if err := c.Validate(); err != nil {
		return Routing{}, err
	}
	if numTokens < 0 {
		numTokens = 0
	}
	loads := make([]int, c.NumRoutedExperts)
	scored := make([]scoredExpert, c.NumRoutedExperts)
	for t := 0; t < numTokens; t++ {
		for e := 0; e < c.NumRoutedExperts; e++ {
			base := float64(mix(uint64(t)+1, uint64(e)+1)>>11) / float64(1<<53)
			bias := skew * float64(c.NumRoutedExperts-e) / float64(c.NumRoutedExperts)
			scored[e] = scoredExpert{expert: e, score: base + bias}
		}
		// Top-k by score; expert index breaks ties deterministically (matches a stable
		// torch.topk tie-break: lower index wins).
		sort.Slice(scored, func(i, j int) bool {
			if scored[i].score != scored[j].score {
				return scored[i].score > scored[j].score
			}
			return scored[i].expert < scored[j].expert
		})
		for k := 0; k < c.TopK; k++ {
			loads[scored[k].expert]++
		}
	}
	return Routing{Tokens: numTokens, TopK: c.TopK, Loads: loads, Shared: numTokens}, nil
}

type scoredExpert struct {
	expert int
	score  float64
}

// ActiveExperts counts routed experts that received at least one token.
func (r Routing) ActiveExperts() int {
	n := 0
	for _, l := range r.Loads {
		if l > 0 {
			n++
		}
	}
	return n
}

// LoadBalance is max(load)/mean(load) over routed experts: 1.0 is perfect balance,
// larger is more skewed. Returns 1.0 for an empty batch.
func (r Routing) LoadBalance() float64 {
	if len(r.Loads) == 0 {
		return 1.0
	}
	sum, max := 0, 0
	for _, l := range r.Loads {
		sum += l
		if l > max {
			max = l
		}
	}
	if sum == 0 {
		return 1.0
	}
	mean := float64(sum) / float64(len(r.Loads))
	return float64(max) / mean
}

// ScheduleMode selects the dispatch strategy the cost model prices.
type ScheduleMode int

const (
	// NaivePerExpert launches one kernel per routed expert every layer, even for
	// experts that received zero tokens — the unfused baseline (moeFFN.apply).
	NaivePerExpert ScheduleMode = iota
	// GroupedFused launches one grouped kernel only per ACTIVE expert (tokens grouped
	// into per-expert batches) — the MegaMoE-shaped target's scheduling.
	GroupedFused
)

func (m ScheduleMode) String() string {
	if m == GroupedFused {
		return "grouped-fused"
	}
	return "naive-per-expert"
}

// Schedule is the work a single MoE layer performs under a mode. WorkRows (the
// token-expert GEMM rows) is identical across modes — grouping changes only how the
// work is launched, never how much there is. Launches is the scheduling cost that
// differs: the naive loop pays for every expert; grouped pays only for active ones.
type Schedule struct {
	Mode     ScheduleMode
	Launches int // kernel launches (routed + shared) for one layer
	WorkRows int // token-expert GEMM rows processed (routed) + shared firings
}

// Cost prices one layer of dispatch under the given mode.
func (r Routing) Cost(mode ScheduleMode) Schedule {
	routedRows := 0
	for _, l := range r.Loads {
		routedRows += l
	}
	var routedLaunches int
	switch mode {
	case GroupedFused:
		routedLaunches = r.ActiveExperts()
	default:
		routedLaunches = len(r.Loads) // one launch per expert, empty or not
	}
	return Schedule{
		Mode:     mode,
		Launches: routedLaunches + NumSharedExperts, // shared always fires
		WorkRows: routedRows + r.Shared,
	}
}

// ExpertParallelPlan partitions the routed experts into contiguous bands, one per rank
// (mirroring internal/model/expert_parallel.go). It returns an error unless the bands
// are an exact, disjoint cover of every expert — the invariant that keeps EP bit-exact
// at ranks=1.
func ExpertParallelPlan(numExperts, ranks int) ([][]int, error) {
	if numExperts <= 0 {
		return nil, ErrExpertCount
	}
	if ranks <= 0 || ranks > numExperts {
		return nil, fmt.Errorf("deepseekv4moe: ranks must be 1..%d, got %d", numExperts, ranks)
	}
	bands := make([][]int, ranks)
	perRank := numExperts / ranks
	rem := numExperts % ranks
	next := 0
	for rk := 0; rk < ranks; rk++ {
		n := perRank
		if rk < rem {
			n++ // spread the remainder across the low ranks
		}
		band := make([]int, 0, n)
		for i := 0; i < n; i++ {
			band = append(band, next)
			next++
		}
		bands[rk] = band
	}
	if next != numExperts {
		return nil, fmt.Errorf("deepseekv4moe: partition covered %d of %d experts", next, numExperts)
	}
	return bands, nil
}

// CommUnits is the all-to-all token-movement count a DeepEP-style dispatcher would move
// for this batch: Tokens*TopK routed tokens cross the fabric (independent of the 384
// expert count — traffic scales with the top-k width, not the pool). It is 0 at ranks=1
// (no fabric). This is a shape proxy, not a measured byte count.
func (r Routing) CommUnits(ranks int) int {
	if ranks <= 1 {
		return 0
	}
	return r.Tokens * r.TopK
}

// Metrics is the minimum metric set the issue names, computed from the synthetic model.
// The per-layer figures are modeled WORK-UNITS (launch cost + row cost), explicitly not
// milliseconds — a weight-free fixture cannot measure time without fabricating it.
type Metrics struct {
	Mode              ScheduleMode
	ExpertLoadBalance float64 // max/mean routed load (1.0 = perfect)
	ActiveExperts     int
	WorkRows          int
	Launches          int
	CommUnits         int
	P50PerLayerWork   float64
	P95PerLayerWork   float64
}

// ComputeMetrics prices `layers` MoE layers and folds them into the metric set. A small
// deterministic per-layer perturbation makes p50/p95 meaningful without any RNG.
func (r Routing) ComputeMetrics(mode ScheduleMode, layers, ranks int) Metrics {
	if layers < 1 {
		layers = 1
	}
	sched := r.Cost(mode)
	const launchWork = 8.0 // modeled fixed cost per kernel launch (work-units)
	const rowWork = 1.0    // modeled cost per token-expert GEMM row
	baseWork := float64(sched.Launches)*launchWork + float64(sched.WorkRows)*rowWork

	perLayer := make([]float64, layers)
	for l := 0; l < layers; l++ {
		// +/- up to ~2% deterministic wobble keyed on the layer index.
		frac := float64(mix(uint64(l)+1, 0xA5A5)>>12) / float64(1<<52) // [0,1)
		perLayer[l] = baseWork * (0.98 + 0.04*frac)
	}
	return Metrics{
		Mode:              mode,
		ExpertLoadBalance: r.LoadBalance(),
		ActiveExperts:     r.ActiveExperts(),
		WorkRows:          sched.WorkRows,
		Launches:          sched.Launches,
		CommUnits:         r.CommUnits(ranks),
		P50PerLayerWork:   percentile(perLayer, 50),
		P95PerLayerWork:   percentile(perLayer, 95),
	}
}

// FormatMetrics renders the naive-vs-grouped comparison for a runbook or CI log.
func FormatMetrics(numTokens, layers, ranks int) string {
	r, err := Route(numTokens, V4Config(), 0)
	if err != nil {
		return "route error: " + err.Error()
	}
	var b strings.Builder
	fmt.Fprintf(&b, "DeepSeek V4 MoE dispatch (synthetic, weight-free): %d tokens, %d layers, %d ranks\n",
		numTokens, layers, ranks)
	fmt.Fprintf(&b, "shape: %d routed experts, top-%d, %d shared; load-balance %.3f, active experts %d\n",
		NumRoutedExperts, TopK, NumSharedExperts, r.LoadBalance(), r.ActiveExperts())
	fmt.Fprintf(&b, "%-18s %10s %10s %10s %14s %14s\n", "mode", "launches", "workRows", "comm", "p50-work", "p95-work")
	for _, m := range []ScheduleMode{NaivePerExpert, GroupedFused} {
		mm := r.ComputeMetrics(m, layers, ranks)
		fmt.Fprintf(&b, "%-18s %10d %10d %10d %14.1f %14.1f\n",
			m, mm.Launches, mm.WorkRows, mm.CommUnits, mm.P50PerLayerWork, mm.P95PerLayerWork)
	}
	return b.String()
}

// percentile returns the p-th percentile (nearest-rank) of xs. xs is copied, not mutated.
func percentile(xs []float64, p int) float64 {
	if len(xs) == 0 {
		return 0
	}
	cp := append([]float64(nil), xs...)
	sort.Float64s(cp)
	rank := (p * len(cp)) / 100
	if rank >= len(cp) {
		rank = len(cp) - 1
	}
	return cp[rank]
}

// mix is a splitmix64-style deterministic hash — the fixture's stand-in for a router
// score, so routing is reproducible across runs and platforms.
func mix(a, b uint64) uint64 {
	x := a*0x9E3779B97F4A7C15 + b*0xBF58476D1CE4E5B9
	x ^= x >> 30
	x *= 0xBF58476D1CE4E5B9
	x ^= x >> 27
	x *= 0x94D049BB133111EB
	x ^= x >> 31
	return x
}
