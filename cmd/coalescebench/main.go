// Command coalescebench projects the aggregate net-tok/s(B) curve for an SSD-offloaded
// MoE served to B concurrent agents, by driving the deterministic cross-agent expert-cache
// coalescing simulator (deepseekv4moe.SimulateExpertCacheBatch, #5244) over a SYNTHETIC
// GLM-5.2-shaped router and applying the §2 roofline of
// docs/notes/MOE-SSD-MULTI-AGENT-NET-TOKS-2026-07-18.md:
//
//	t_agent(B)  = max( SSD_term(B), RAM_term(B), FLOP_term )
//	SSD_term(B) = PageIns_per_step · e / (B · BW_ssd)   — the coalesced expert stream (§2's Σ_ℓ M_ℓ,
//	              the simulator's Misses: distinct groups NOT already resident)
//	RAM_term(B) = ( NR + U_step(B) · e ) / BW_ram       — the dense/resident roofline
//	FLOP_term   = active_flops_per_token / FLOPS        — the compute roofline
//	net_toks(B) = B / t_agent(B)
//
// The whole thesis lives in the SSD term: B agents' per-step top-K selections union to
// U(B) ≪ B·K distinct (layer, expert) groups, so the SSD bytes PER AGENT-TOKEN fall as B
// grows (the coalescing ratio C(B) = B·K/U(B)), until the working set goes resident and the
// box rides the RAM/FLOP roofline — the regime transition B* this bench detects and prints.
//
// EVERY emitted row is PROJECTED: a roofline over a synthetic, deterministically seeded
// router (tunable uniform→Zipf skew), not a served measurement. It proves no model I/O, no
// dequant, no GPU execution, and no latency — it counts distinct-group page-ins under a
// stated capacity and divides labelled bandwidths into labelled byte counts. The real-trace
// witness (captured GLM-5.2 top-K routes) is #5251. Two baselines are reported side by side
// per §2.1 so the multiple is never stated without its denominator:
//
//	×vs-uncoalesced = net_toks(B) ÷ (B · net_toks(1)) — vs B independent un-coalesced
//	                  streams, EACH granted the full SSD bandwidth (B separate boxes): an
//	                  OPTIMISTIC upper-baseline, which is why it dips below 1× once the
//	                  batch goes RAM-bound (#5245 review). The same-box physics is the
//	                  shared-SSD un-coalesced floor printed under the table:
//	                  BW_ssd/(L·K·e), constant in B — B un-coalesced, un-cached streams
//	                  time-sharing one SSD. The defensible headline stays C(B) itself.
//	×vs-1agent      = net_toks(B) ÷ net_toks(1)       — AGGREGATE over single-agent latency;
//	                  latency-tolerant only, never a speedup a single user feels.
//
// Flag defaults are GLM-5.2 GGUF-header-derived where a header truth exists
// (docs/notes/GLM52-DGX-THEORETICAL-CEILING-2026-07-06.md §2) and labelled PLACEHOLDER
// where it does not (bandwidths, cache size, skew). Same flags → identical table:
// the router PRNG is a seeded splitmix64, no time, no math/rand default source.
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"math"
	"os"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/benchcli"
	"github.com/anthony-chaudhary/fak/internal/deepseekv4moe"
	"github.com/anthony-chaudhary/fak/internal/intlist"
)

const (
	mib = 1024.0 * 1024.0
	gib = 1024.0 * 1024.0 * 1024.0
)

// benchConfig is the full deterministic input set: same config → identical output.
type benchConfig struct {
	Seed         int64   `json:"seed"`
	Steps        int     `json:"steps"`            // decode steps replayed per batch point
	Layers       int     `json:"layers"`           // L — MoE layers
	Experts      int     `json:"experts"`          // N — experts per layer
	TopK         int     `json:"top_k"`            // K — experts per token
	Skew         float64 `json:"router_zipf_skew"` // Zipf exponent s over per-layer expert popularity; 0 = uniform
	ExpertMiB    float64 `json:"expert_mib"`       // e — bytes per routed (layer,expert) group, in MiB
	NonRoutedGiB float64 `json:"non_routed_gib"`   // NR — non-routed active bytes per token, in GiB
	BWSSDGiB     float64 `json:"bw_ssd_gib_s"`     // BW_ssd — SSD/host read bandwidth, GiB/s
	BWRAMGiB     float64 `json:"bw_ram_gib_s"`     // BW_ram — resident-memory bandwidth, GiB/s
	ActiveGFLOP  float64 `json:"active_gflop"`     // active GFLOP per token
	GFLOPS       float64 `json:"device_gflop_s"`   // device GFLOP/s
	CacheGiB     float64 `json:"cache_gib"`        // resident expert-cache budget (the RAM tier over SSD), GiB
	Batches      []int   `json:"batches"`
}

func (c benchConfig) validate() error {
	switch {
	case c.Steps <= 0:
		return fmt.Errorf("steps must be positive (got %d)", c.Steps)
	case c.Layers <= 0 || c.Experts <= 0 || c.TopK <= 0 || c.TopK > c.Experts:
		return fmt.Errorf("invalid model shape: layers=%d experts=%d topk=%d", c.Layers, c.Experts, c.TopK)
	case c.Skew < 0:
		return fmt.Errorf("skew must be >= 0 (got %g)", c.Skew)
	case c.ExpertMiB <= 0 || c.NonRoutedGiB < 0:
		return fmt.Errorf("invalid byte inputs: expert-mib=%g nonrouted-gib=%g", c.ExpertMiB, c.NonRoutedGiB)
	case c.BWSSDGiB <= 0 || c.BWRAMGiB <= 0 || c.GFLOPS <= 0 || c.ActiveGFLOP < 0:
		return fmt.Errorf("invalid rate inputs: bw-ssd=%g bw-ram=%g gflops=%g active-gflop=%g",
			c.BWSSDGiB, c.BWRAMGiB, c.GFLOPS, c.ActiveGFLOP)
	case c.CacheGiB <= 0:
		return fmt.Errorf("cache-gib must be positive (got %g)", c.CacheGiB)
	case len(c.Batches) == 0:
		return fmt.Errorf("batches must name at least one batch size")
	}
	for _, b := range c.Batches {
		if b <= 0 {
			return fmt.Errorf("batch sizes must be positive (got %d)", b)
		}
	}
	return nil
}

// capacityGroups converts the cache budget into whole (layer,expert) groups.
func (c benchConfig) capacityGroups() (int, error) {
	groups := int(c.CacheGiB * gib / (c.ExpertMiB * mib))
	if groups < 1 {
		return 0, fmt.Errorf("cache-gib %.2f holds less than one %.2f MiB expert group", c.CacheGiB, c.ExpertMiB)
	}
	return groups, nil
}

// rng is a splitmix64 PRNG: deterministic across platforms and Go releases, unlike
// relying on any library default source. Seeded once per (config seed, batch size).
type rng struct{ s uint64 }

func newRNG(seed uint64) *rng { return &rng{s: seed} }

func (r *rng) next() uint64 {
	r.s += 0x9e3779b97f4a7c15
	z := r.s
	z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9
	z = (z ^ (z >> 27)) * 0x94d049bb133111eb
	return z ^ (z >> 31)
}

// float64 returns a uniform draw in (0,1] — never 0, so -log(u) is always finite.
func (r *rng) float64() float64 {
	f := float64(r.next()>>11) / (1 << 53)
	if f == 0 {
		return 1.0 / (1 << 53)
	}
	return f
}

// mixSeed derives an independent stream per batch size from the one config seed, so
// changing -batches never perturbs the draws of the sizes that stay in the sweep.
func mixSeed(seed int64, agents int) uint64 {
	return uint64(seed)*0x9e3779b97f4a7c15 ^ uint64(agents)*0xd1b54a32d192ed03
}

// layerWeights builds the per-layer expert-popularity weights for the synthetic router:
// weight(expert) = 1/(rank+1)^skew with the rank order rotated per layer so hot experts
// are not the same ids at every layer. skew=0 collapses to uniform.
func layerWeights(layers, experts int, skew float64) [][]float64 {
	w := make([][]float64, layers)
	for l := 0; l < layers; l++ {
		row := make([]float64, experts)
		for i := 0; i < experts; i++ {
			rank := (i + l*97) % experts
			row[i] = math.Pow(float64(rank+1), -skew)
		}
		w[l] = row
	}
	return w
}

// sampleTopK draws k distinct experts weighted by w via Efraimidis–Spirakis exponential
// keys (key_i = -ln(u_i)/w_i, keep the k smallest): exact weighted sampling WITHOUT
// replacement, bounded work, fully deterministic given the rng state.
func sampleTopK(r *rng, w []float64, k int) []int {
	bestKey := make([]float64, 0, k)
	bestIdx := make([]int, 0, k)
	for i, wi := range w {
		key := -math.Log(r.float64()) / wi
		if len(bestKey) < k {
			bestKey = append(bestKey, key)
			bestIdx = append(bestIdx, i)
		} else if key >= bestKey[k-1] {
			continue
		} else {
			bestKey[k-1] = key
			bestIdx[k-1] = i
		}
		// Insertion-sort the new entry into position (k is tiny — K=8 for GLM-5.2).
		for j := len(bestKey) - 1; j > 0 && bestKey[j] < bestKey[j-1]; j-- {
			bestKey[j], bestKey[j-1] = bestKey[j-1], bestKey[j]
			bestIdx[j], bestIdx[j-1] = bestIdx[j-1], bestIdx[j]
		}
	}
	return bestIdx
}

// unionStats accumulates the within-step coalescing evidence: the summed distinct-expert
// union sizes over every (decode step, layer) entry.
type unionStats struct {
	total   int64
	entries int
}

// uMean is the mean distinct experts per (layer, decode-step) — the U(B) of §2.
func (u unionStats) uMean() float64 {
	if u.entries == 0 {
		return 0
	}
	return float64(u.total) / float64(u.entries)
}

// unionSize counts distinct expert ids across the batch's per-agent selections using a
// stamp array (scratch, stamped with a monotonically increasing epoch — no clearing).
func unionSize(perAgent [][]int, stamp []int, epoch int) int {
	n := 0
	for _, sel := range perAgent {
		for _, e := range sel {
			if stamp[e] != epoch {
				stamp[e] = epoch
				n++
			}
		}
	}
	return n
}

// synthRoutes generates the deterministic synthetic router trace for one batch point:
// steps × layers BatchStep entries, each with `agents` independent top-K draws from the
// per-layer skewed popularity, plus the exact within-step union stats.
func synthRoutes(cfg benchConfig, agents int) ([]deepseekv4moe.BatchStep, unionStats) {
	r := newRNG(mixSeed(cfg.Seed, agents))
	weights := layerWeights(cfg.Layers, cfg.Experts, cfg.Skew)
	steps := make([]deepseekv4moe.BatchStep, 0, cfg.Steps*cfg.Layers)
	stamp := make([]int, cfg.Experts)
	var stats unionStats
	epoch := 0
	for s := 0; s < cfg.Steps; s++ {
		for l := 0; l < cfg.Layers; l++ {
			perAgent := make([][]int, agents)
			for a := 0; a < agents; a++ {
				perAgent[a] = sampleTopK(r, weights[l], cfg.TopK)
			}
			epoch++
			stats.total += int64(unionSize(perAgent, stamp, epoch))
			stats.entries++
			steps = append(steps, deepseekv4moe.BatchStep{Layer: l, PerAgent: perAgent})
		}
	}
	return steps, stats
}

// row is one PROJECTED point of the net-tok/s(B) curve.
type row struct {
	B              int     `json:"batch"`
	UMean          float64 `json:"mean_distinct_experts"`  // U(B): mean distinct experts per (layer, decode-step)
	C              float64 `json:"coalescing_ratio"`       // C(B) = B·K / U(B) — within-step coalescing ratio
	PageInsPerStep float64 `json:"page_ins_per_step"`      // simulated SSD page-in groups (trace.Misses — §2's M) per decode step, all layers
	TraceRatio     float64 `json:"trace_coalescing_ratio"` // simulator CoalesceRatio: NaiveStreamed/DistinctStreamed
	PeakResident   int     `json:"peak_resident_groups"`
	SSDTerm        float64 `json:"ssd_term_seconds"` // seconds per agent-token, §2 as written
	RAMTerm        float64 `json:"ram_term_seconds"`
	FLOPTerm       float64 `json:"flop_term_seconds"`
	T              float64 `json:"projected_agent_seconds"` // max of the three terms
	NetToks        float64 `json:"projected_net_tokens_s"`  // B / T
	Binding        string  `json:"binding_roofline"`        // which roofline binds: SSD | RAM | FLOP
}

// benchRun is the computed portion of one report. Keeping it separate from the
// Markdown renderer lets the optional JSON artifact describe the exact same rows
// without parsing or recomputing the operator-facing output.
type benchRun struct {
	Capacity int
	Rows     []row
	Base     row
}

const coalesceArtifactSchema = "fak-coalescebench/1"

// coalesceArtifactReport is the additive machine-readable companion to the
// historical Markdown stdout. ProjectedMarkdown is retained byte-for-byte so
// adding the shared benchcli envelope cannot silently rewrite a headline or
// baseline while the typed points make the same calculation indexable.
type coalesceArtifactReport struct {
	Schema  string                 `json:"schema"`
	Label   string                 `json:"label"`
	Model   string                 `json:"model"`
	Config  benchConfig            `json:"config"`
	Results coalesceArtifactResult `json:"results"`
}

type coalesceArtifactResult struct {
	Points                   []row   `json:"points"`
	RegimeTransitionBatch    *int    `json:"regime_transition_batch"`
	UncoalescedSharedSSDTokS float64 `json:"uncoalesced_shared_ssd_tokens_s"`
	UncoalescedBaseline      string  `json:"uncoalesced_baseline"`
	SingleAgentBaseline      string  `json:"single_agent_baseline"`
	ProjectedMarkdown        string  `json:"projected_markdown"`
}

func artifactReport(cfg benchConfig, run benchRun, markdown string) coalesceArtifactReport {
	var transition *int
	if star, ok := regimeTransition(run.Rows); ok {
		b := star.B
		transition = &b
	}
	return coalesceArtifactReport{
		Schema: coalesceArtifactSchema,
		Label:  "PROJECTED",
		Model:  "synthetic GLM-5.2-shaped router",
		Config: cfg,
		Results: coalesceArtifactResult{
			Points:                   run.Rows,
			RegimeTransitionBatch:    transition,
			UncoalescedSharedSSDTokS: uncoalescedFloor(cfg),
			UncoalescedBaseline:      "B independent un-coalesced streams, each granted full SSD bandwidth (optimistic B-box baseline)",
			SingleAgentBaseline:      "aggregate net tokens/s divided by the B=1 projected point; not single-user latency",
			ProjectedMarkdown:        markdown,
		},
	}
}

// analyticalEvidence declares exactly what coalescebench's deterministic
// router replay plus roofline can prove. In particular, its projected tok/s
// fields remain bottleneck hypotheses and can never be promoted to achieved or
// competitive throughput by a downstream artifact consumer.
func analyticalEvidence(cfg benchConfig, wall time.Duration, reportBytes int64) (benchcli.SimulationEvidence, error) {
	configJSON, err := json.Marshal(cfg)
	if err != nil {
		return benchcli.SimulationEvidence{}, fmt.Errorf("marshal evidence config: %w", err)
	}
	configDigest := benchcli.SHA256Digest(configJSON)
	engineRevision := benchcli.Stamp().GitCommit
	if strings.TrimSpace(engineRevision) == "" || strings.EqualFold(strings.TrimSpace(engineRevision), "unknown") {
		return benchcli.SimulationEvidence{}, fmt.Errorf("engine revision unresolved: %q", engineRevision)
	}
	workloadDigest := benchcli.SHA256Digest(append([]byte(engineRevision+"\n"), configJSON...))
	seed := cfg.Seed
	return benchcli.SimulationEvidence{
		Schema:       benchcli.SimulationEvidenceSchema,
		EvidenceType: benchcli.EvidenceAnalyticalBound,
		ClaimCeiling: benchcli.ClaimBottleneckOnly,
		Engine: benchcli.SimulationEngine{
			Name:         "coalescebench synthetic-router roofline",
			Revision:     engineRevision,
			ConfigDigest: configDigest,
		},
		Workload: benchcli.WorkloadProvenance{
			Name:   "synthetic GLM-5.2-shaped MoE routes",
			Source: "cmd/coalescebench splitmix64 weighted-without-replacement generator",
			Digest: workloadDigest,
		},
		ValidityEnvelope: benchcli.ValidityEnvelope{
			Description: "deterministic mechanism counts and roofline bottleneck bounds for the declared synthetic router, cache capacity, model shape, and placeholder rates",
			Dimensions: map[string]string{
				"batches":             fmt.Sprint(cfg.Batches),
				"cache":               fmt.Sprintf("%.6g GiB LRU over %.6g MiB expert groups", cfg.CacheGiB, cfg.ExpertMiB),
				"model_shape":         fmt.Sprintf("layers=%d experts=%d top_k=%d", cfg.Layers, cfg.Experts, cfg.TopK),
				"router_distribution": fmt.Sprintf("synthetic Zipf exponent %.6g", cfg.Skew),
				"seed":                fmt.Sprint(cfg.Seed),
				"steps":               fmt.Sprint(cfg.Steps),
			},
		},
		ExcludedEffects: []string{
			"real-model routing distribution and the pending #5251 trace",
			"served model I/O, dequantization, kernel execution, scheduling, and contention",
			"achieved latency, throughput, energy, cost, and tail behavior",
			"fak-vs-llama.cpp or native-faster competitive claims",
			"cross-seed uncertainty and statistical confidence; this is one frozen fixed-seed fixture",
			"host CPU time is not sampled by this portable in-process runner (recorded as 0 ms)",
		},
		Replay: benchcli.ReplaySpec{
			Seed:               &seed,
			Stream:             "one frozen fixture; splitmix64 substreams keyed by (seed,batch) are sweep inputs, not independent replications",
			Repetitions:        1,
			IndependentStreams: 1,
			Stochastic:         false,
			Uncertainty:        "not estimated; the fixed seed establishes replay only",
		},
		Cost: benchcli.SimulationCost{
			HostWallTimeMS: float64(wall) / float64(time.Millisecond),
			HostCPUTimeMS:  0,
			Bytes:          reportBytes,
		},
	}, nil
}

func writeArtifact(path string, cfg benchConfig, run benchRun, markdown string, wall time.Duration) error {
	report := artifactReport(cfg, run, markdown)
	reportJSON, err := json.Marshal(report)
	if err != nil {
		return fmt.Errorf("marshal report for cost accounting: %w", err)
	}
	evidence, err := analyticalEvidence(cfg, wall, int64(len(reportJSON)))
	if err != nil {
		return err
	}
	if err := benchcli.WriteReportWithEvidence(path, report, evidence); err != nil {
		return fmt.Errorf("write artifact: %w", err)
	}
	return nil
}

// computeRow applies the §2 roofline exactly as written to one batch point.
// pageInsPerStep is the simulator's Misses per decode step — §2's Σ_ℓ M_ℓ(B), the groups
// NOT already resident that actually stream from SSD. (The landed #5244 API's
// DistinctStreamed counts Hits+Misses — every distinct group per step, resident or not —
// so the faithful SSD stream is the Misses field, per §2 and the issue's "actually paged
// in" intent.)
func computeRow(cfg benchConfig, b int, u unionStats, pageInsPerStep float64) row {
	e := cfg.ExpertMiB * mib
	uMean := u.uMean()
	ssd := pageInsPerStep * e / (float64(b) * cfg.BWSSDGiB * gib)
	ram := (cfg.NonRoutedGiB*gib + uMean*float64(cfg.Layers)*e) / (cfg.BWRAMGiB * gib)
	flop := cfg.ActiveGFLOP / cfg.GFLOPS
	t, binding := ssd, "SSD"
	if ram > t {
		t, binding = ram, "RAM"
	}
	if flop > t {
		t, binding = flop, "FLOP"
	}
	c := 0.0
	if uMean > 0 {
		c = float64(b*cfg.TopK) / uMean
	}
	return row{
		B: b, UMean: uMean, C: c, PageInsPerStep: pageInsPerStep,
		SSDTerm: ssd, RAMTerm: ram, FLOPTerm: flop, T: t, NetToks: float64(b) / t, Binding: binding,
	}
}

// uncoalescedFloor is the same-box, shared-SSD un-coalesced aggregate rate: B un-coalesced,
// un-cached streams time-sharing ONE SSD each pay the full L·K·e expert bytes per token, so
// the aggregate is BW_ssd/(L·K·e) tok/s at ANY B — the physical floor the #5245 landing
// review asked to state beside the optimistic B-separate-boxes ×vs-uncoalesced column.
func uncoalescedFloor(cfg benchConfig) float64 {
	return cfg.BWSSDGiB * gib / (float64(cfg.Layers) * float64(cfg.TopK) * cfg.ExpertMiB * mib)
}

// regimeTransition returns the smallest swept B where the coalesced SSD stream stops
// binding (SSD_term < RAM_term) — the point the box goes effectively resident.
func regimeTransition(rows []row) (row, bool) {
	best, found := row{}, false
	for _, r := range rows {
		if r.SSDTerm < r.RAMTerm && (!found || r.B < best.B) {
			best, found = r, true
		}
	}
	return best, found
}

// benchPoint runs one batch size end to end: synthetic routes → the #5244 simulator →
// the §2 roofline row.
func benchPoint(cfg benchConfig, capacity, b int) (row, error) {
	steps, stats := synthRoutes(cfg, b)
	trace, err := deepseekv4moe.SimulateExpertCacheBatch(steps, capacity, cfg.Layers, cfg.Experts, cfg.TopK, b)
	if err != nil {
		return row{}, fmt.Errorf("B=%d: SimulateExpertCacheBatch: %w", b, err)
	}
	// Cross-check: the generator's within-step unions and the simulator's DistinctStreamed
	// count the same thing (distinct groups per BatchStep entry). Disagreement means the
	// bench's U(B) column would not describe the trace the simulator replayed — fail closed.
	if trace.DistinctStreamed != stats.total {
		return row{}, fmt.Errorf("B=%d: generator union total %d != simulator DistinctStreamed %d", b, stats.total, trace.DistinctStreamed)
	}
	r := computeRow(cfg, b, stats, float64(trace.Misses)/float64(cfg.Steps))
	r.TraceRatio = trace.CoalesceRatio
	r.PeakResident = trace.PeakResident
	return r, nil
}

// runSweep computes the deterministic points shared by the Markdown and JSON
// renderers. diag receives per-point progress lines (os.Stderr in main,
// io.Discard in tests).
func runSweep(cfg benchConfig, diag io.Writer) (benchRun, error) {
	if err := cfg.validate(); err != nil {
		return benchRun{}, err
	}
	capacity, err := cfg.capacityGroups()
	if err != nil {
		return benchRun{}, err
	}

	rows := make([]row, 0, len(cfg.Batches))
	var base row
	haveBase := false
	for _, b := range cfg.Batches {
		r, err := benchPoint(cfg, capacity, b)
		if err != nil {
			return benchRun{}, err
		}
		rows = append(rows, r)
		if b == 1 {
			base, haveBase = r, true
		}
		fmt.Fprintf(diag, "[coalescebench] B=%-4d U=%6.1f C=%5.2f page-ins/step=%8.1f trace-C=%5.2f peak-resident=%d net=%8.2f tok/s (%s-bound) PROJECTED\n",
			r.B, r.UMean, r.C, r.PageInsPerStep, r.TraceRatio, r.PeakResident, r.NetToks, r.Binding)
	}
	if !haveBase {
		// The §2.1 baselines divide by net_toks(1); compute the reference point even when
		// B=1 is not in the sweep (it is, at the defaults).
		base, err = benchPoint(cfg, capacity, 1)
		if err != nil {
			return benchRun{}, err
		}
	}
	return benchRun{Capacity: capacity, Rows: rows, Base: base}, nil
}

// runBench produces the full deterministic stdout report for a config. The
// returned string remains the legacy artifact: same cfg -> identical bytes.
func runBench(cfg benchConfig, diag io.Writer) (string, error) {
	run, err := runSweep(cfg, diag)
	if err != nil {
		return "", err
	}
	return render(cfg, run.Capacity, run.Rows, run.Base), nil
}

func render(cfg benchConfig, capacity int, rows []row, base row) string {
	var sb strings.Builder
	fmt.Fprintf(&sb, "# coalescebench — PROJECTED net-tok/s(B) roofline (synthetic GLM-5.2-shaped router, seed=%d)\n", cfg.Seed)
	fmt.Fprintf(&sb, "# model shape (GLM-5.2 GGUF-header-derived; docs/notes/GLM52-DGX-THEORETICAL-CEILING-2026-07-06.md §2):\n")
	fmt.Fprintf(&sb, "#   L=%d MoE layers, N=%d experts/layer, K=%d experts/token, e=%.2f MiB/(layer,expert) group, NR=%.2f GiB/token\n",
		cfg.Layers, cfg.Experts, cfg.TopK, cfg.ExpertMiB, cfg.NonRoutedGiB)
	fmt.Fprintf(&sb, "# machine inputs (PLACEHOLDER, not measured): BW_ssd=%.2f GiB/s, BW_ram=%.2f GiB/s, FLOPS=%.0f GFLOP/s, active=%.0f GFLOP/tok, expert-cache=%.1f GiB (%d groups), skew=%.2f (Zipf exponent; PLACEHOLDER), steps=%d\n",
		cfg.BWSSDGiB, cfg.BWRAMGiB, cfg.GFLOPS, cfg.ActiveGFLOP, cfg.CacheGiB, capacity, cfg.Skew, cfg.Steps)
	fmt.Fprintf(&sb, "# U(B) = mean distinct experts per (layer, decode-step); C(B) = B*K/U(B). SSD_term streams the SIMULATED\n")
	fmt.Fprintf(&sb, "# page-ins (#5244 SimulateExpertCacheBatch Misses — groups not already resident): Misses/step * e / (B*BW_ssd).\n")
	fmt.Fprintf(&sb, "# NOT a served measurement: every row is PROJECTED from the stated inputs (roofline; real-trace witness pending #5251).\n")
	fmt.Fprintf(&sb, "\n")
	fmt.Fprintf(&sb, "| B | U(B) | C(B) | SSD_term | net_toks | ×vs-uncoalesced | ×vs-1agent | binding-roofline | label |\n")
	fmt.Fprintf(&sb, "|--:|-----:|-----:|---------:|---------:|----------------:|-----------:|:-----------------|:------|\n")
	for _, r := range rows {
		fmt.Fprintf(&sb, "| %d | %.1f | %.2f | %.4f s | %.2f | %.2f× | %.2f× | %s | PROJECTED |\n",
			r.B, r.UMean, r.C, r.SSDTerm, r.NetToks,
			r.NetToks/(float64(r.B)*base.NetToks), r.NetToks/base.NetToks, r.Binding)
	}
	fmt.Fprintf(&sb, "\n")
	if star, ok := regimeTransition(rows); ok {
		fmt.Fprintf(&sb, "regime transition: B* = %d — first swept B where SSD_term (%.4f s) drops below RAM_term (%.4f s): the coalesced expert stream stops binding and the box rides the resident roofline. PROJECTED.\n",
			star.B, star.SSDTerm, star.RAMTerm)
	} else {
		fmt.Fprintf(&sb, "regime transition: not reached within the swept batch sizes — SSD_term >= RAM_term throughout (the expert stream still binds). PROJECTED.\n")
	}
	fmt.Fprintf(&sb, "un-coalesced shared-SSD floor (PROJECTED): BW_ssd/(L·K·e) = %.2f tok/s aggregate at ANY B — B un-coalesced, un-cached streams time-sharing ONE SSD each stream the full L·K·e expert bytes per token. This is the same-box physics baseline; the coalesced curve's rise above it is the coalescing+cache win.\n", uncoalescedFloor(cfg))
	fmt.Fprintf(&sb, "baselines (§2.1): ×vs-uncoalesced = net_toks(B) ÷ (B · net_toks(1)) — vs B independent un-coalesced streams EACH granted the full SSD bandwidth (B separate boxes): an OPTIMISTIC upper-baseline, so it dips below 1× once the batch goes RAM-bound; the same-box comparison is the shared-SSD floor above, and the defensible headline is C(B) itself. ×vs-1agent = net_toks(B) ÷ net_toks(1) — AGGREGATE over single-agent latency; latency-tolerant only, never a speedup a single user feels.\n")
	return sb.String()
}

func main() {
	seed := flag.Int64("seed", 1, "synthetic-router PRNG seed (same seed+flags → identical table)")
	steps := flag.Int("steps", 32, "decode steps replayed per batch point")
	layers := flag.Int("layers", 76, "L: MoE layer count (GLM-5.2 GGUF header: 79 layers = 3 dense + 76 MoE; GLM52-DGX-THEORETICAL-CEILING §2)")
	experts := flag.Int("experts", 256, "N: experts per layer (GLM-5.2 GGUF header)")
	topk := flag.Int("topk", 8, "K: experts per token (GLM-5.2 GGUF header)")
	skew := flag.Float64("skew", 1.0, "Zipf exponent over per-layer expert popularity; 0 = uniform. PLACEHOLDER — real GLM-5.2 router skew is the #5251 real-trace witness")
	expertMiB := flag.Float64("expert-mib", 21.81, "e: bytes per routed (layer,expert) group, MiB. Header-derived: 1.619 GiB/expert across 76 MoE layers = 1.619*1024/76 (GLM52-DGX-THEORETICAL-CEILING §2)")
	nonRoutedGiB := flag.Float64("nonrouted-gib", 19.31, "NR: non-routed active bytes/token, GiB (header-derived; GLM52-DGX-THEORETICAL-CEILING §2)")
	bwSSD := flag.Float64("bw-ssd-gib", 7.0, "BW_ssd: SSD/host read bandwidth, GiB/s. PLACEHOLDER (PCIe-Gen4 NVMe class; the striped-read hardware witness is operator-gated)")
	bwRAM := flag.Float64("bw-ram-gib", 200.0, "BW_ram: resident-memory bandwidth, GiB/s. PLACEHOLDER (server-class host)")
	activeGFLOP := flag.Float64("active-gflop", 64.0, "active GFLOP per token (~2 x 32B active params; GLM52-DGX-THEORETICAL-CEILING §3.2)")
	gflops := flag.Float64("gflops", 1000.0, "device GFLOP/s for FLOP_term. PLACEHOLDER (host-GEMM ~1 TFLOP/s class)")
	cacheGiB := flag.Float64("cache-gib", 128.0, "resident expert-cache budget (RAM tier over SSD), GiB. PLACEHOLDER — the epic's premise is host RAM smaller than the 414.5 GiB routed band")
	batchesArg := flag.String("batches", "1,4,16,64,128", "comma-separated batch sizes (agent counts) to sweep")
	artifactOut := flag.String("artifact-out", "", "optional shared benchcli JSON artifact path; stdout remains the historical PROJECTED Markdown")
	flag.Parse()

	cfg := benchConfig{
		Seed: *seed, Steps: *steps, Layers: *layers, Experts: *experts, TopK: *topk,
		Skew: *skew, ExpertMiB: *expertMiB, NonRoutedGiB: *nonRoutedGiB,
		BWSSDGiB: *bwSSD, BWRAMGiB: *bwRAM, ActiveGFLOP: *activeGFLOP, GFLOPS: *gflops,
		CacheGiB: *cacheGiB, Batches: intlist.Parse(*batchesArg),
	}
	started := time.Now()
	run, err := runSweep(cfg, os.Stderr)
	if err != nil {
		fmt.Fprintln(os.Stderr, "coalescebench:", err)
		os.Exit(1)
	}
	simulationWall := time.Since(started)
	out := render(cfg, run.Capacity, run.Rows, run.Base)
	if path := strings.TrimSpace(*artifactOut); path != "" {
		if err := writeArtifact(path, cfg, run, out, simulationWall); err != nil {
			fmt.Fprintln(os.Stderr, "coalescebench:", err)
			os.Exit(1)
		}
	}
	fmt.Print(out)
}
