package bench

// fanscale.go — the N=100..1000 fan-out SCALE harness for issue #11 (D-001).
//
// Issue #11 asks to extend `fanbench` (the one-master-goal → N-subagent topology that
// lives in internal/turnbench) into the N=100/500/1000 regime and record, per scale
// point:
//
//   - COORDINATION OVERHEAD vs the N=1 single-agent baseline — the synchronous-join /
//     fold tax that grows with N (the literature's "the lead waits on the slowest of N",
//     the term that makes throughput saturate), priced against doing the goal with ONE
//     agent so a "fan-out win" is always a budget-controlled comparison; and
//   - CROSS-AGENT REUSE UPLIFT — the fan-out-only sibling tool-result dedup the kernel
//     MEASURES (cross_uplift = SHARED-world fan-out − ISOLATED-world sub-agents), the
//     EXACT (N−1)·prefix shared-prefix-KV-reuse geometry, and the MODELED prompt-cache
//     tax clawback — kept in their own fields so the measured/modeled split stays honest.
//
// as HARNESS OUTPUTS. This file is the host-runnable harness plus a tiny N=8 smoke
// (fanscale_test.go) that proves it builds and the trends hold. It is >=1024 capable:
// pass any grid (CanonicalFanScaleGrid is the documented {1,100,500,1000} default).
//
// PLACEMENT LAW (why this is "the harness, not the headline run"). The harness COMPUTES
// the geometry above deterministically and cheaply — turnbench scoring is in-process
// kernel arithmetic, not a model call — so even N=1000 runs on the agent host. What is
// DEFERRED to a bench node is the published HEADLINE run: the live-model wall-clock, the
// LangGraph/AutoGen/CrewAI comparison, and the published results table. Running THAT here
// would starve the agent seat and skew the number, so the report names it as deferred
// (DeferredRun) rather than fabricating it.
//
// It wraps the already-witnessed fan-out engine (turnbench.RunFanoutCell, grounded by
// internal/turnbench/fanout_test.go and cmd/fanbench's TestPrefixReuseFanoutWitness)
// rather than re-deriving the geometry, so the measured/modeled honesty line is identical
// to `fanbench`.

import (
	"context"
	"runtime"
	"sort"

	"github.com/anthony-chaudhary/fak/internal/appversion"
	"github.com/anthony-chaudhary/fak/internal/turnbench"
)

// CanonicalFanScaleGrid is the issue-#11 scale ladder: the N=1 single-agent baseline plus
// the N=100/500/1000 points the acceptance criteria name. The harness accepts any grid
// and is >=1024 capable; this is just the documented default. Always sorted, unique, and
// includes 1 (the baseline) so every overhead/uplift number is priced against one agent.
var CanonicalFanScaleGrid = []int{1, 100, 500, 1000}

// Defaults for a scale sweep. Trials is kept modest because the scale grid multiplies the
// per-cell work by N; the smoke and the published bench-node run can both override it.
const (
	defaultFanScaleSubTurns = 4
	defaultFanScaleTrials   = 8
	// fanScaleSeed makes the whole sweep reproducible (a fixed (profile,N,subTurns,trials,
	// seed) yields the identical surface — the turnbench determinism discipline).
	fanScaleSeed = 0x5EED_FA11

	deferredRunNote = "published headline run deferred to a bench node (placement law): " +
		"live-model wall-clock, LangGraph/AutoGen/CrewAI comparison, and the published " +
		"results table. This harness emits the host-computed coordination overhead + " +
		"cross-agent reuse geometry; run the N=1000 headline off the agent host."
)

// FanScalePoint is one N on the scale ladder: the fan-out width, the coordination
// overhead vs the N=1 baseline, and the cross-agent reuse uplift. The measured and
// modeled halves are kept in separate fields (the fanbench honesty line).
type FanScalePoint struct {
	Agents   int `json:"agents"`    // N: fan-out width (sub-agents spawned for the one goal)
	SubTurns int `json:"sub_turns"` // turns per sub-agent
	Trials   int `json:"trials"`
	Calls    int `json:"calls"` // kernel calls scored per trial in the SHARED arm (≈ plan + N·subTurns)

	// ---- COORDINATION OVERHEAD (vs the N=1 single-agent baseline) ----
	FoldTurns          float64 `json:"fold_turns"`           // master synthesis turns (grows with N: the join tax)
	CriticalPathTurns  float64 `json:"critical_path_turns"`  // plan + slowest sub-agent + fold (depth latency)
	TotalWorkTurns     float64 `json:"total_work_turns"`     // plan + N·sub-agent + fold (sum of work)
	CoordOverheadTurns float64 `json:"coord_overhead_turns"` // critical_path(N) − critical_path(1): the fan-out's added depth
	CoordOverheadFrac  float64 `json:"coord_overhead_frac"`  // coord_overhead_turns ÷ baseline critical path
	ParallelSpeedup    float64 `json:"parallel_speedup"`     // total_work ÷ critical_path

	// ---- CROSS-AGENT REUSE UPLIFT ----
	CrossUpliftP50    int     `json:"cross_uplift_p50"`     // MEASURED fan-out-only sibling tool-result dedup (SHARED − ISOLATED)
	PrefixTokensSaved int     `json:"prefix_tokens_saved"`  // EXACT geometry: (N−1)·prefix prefill the kernel does not redo
	TaxClawedBackFrac float64 `json:"tax_clawed_back_frac"` // MODELED prompt-cache clawback of the (naive−1) token tax
	DedupTokensSaved  int     `json:"dedup_tokens_saved"`   // MEASURED cross_uplift priced (kept apart from the modeled saving)
	NetTokensSaved    int     `json:"net_tokens_saved"`     // modeled prefix-cache saving + measured dedup saving
}

// FanScaleReport is the harness artifact: provenance, the run knobs, the N=1 baseline,
// and one FanScalePoint per N on the scale grid. JSON-able so the bench-node run can emit
// it next to the published table.
type FanScaleReport struct {
	AppVersion  string `json:"app_version"`
	GoVersion   string `json:"go_version"`
	OS          string `json:"os"`
	GeneratedBy string `json:"generated_by"`

	Profile  string `json:"profile"`
	Seed     int64  `json:"seed"`
	Trials   int    `json:"trials"`
	SubTurns int    `json:"sub_turns"`
	Grid     []int  `json:"grid"`

	Baseline FanScalePoint   `json:"baseline"` // the N=1 single-agent control
	Points   []FanScalePoint `json:"points"`   // one per N in Grid (includes the baseline point)

	// DeferredRun is the honest RED flag: the published headline run lives on a bench node.
	DeferredRun string `json:"deferred_run"`
}

// FanScaleOptions configure a scale sweep. The zero value is valid: every field falls back
// to a documented default (CanonicalFanScaleGrid, FanoutResearch, DefaultFanoutCostModel).
type FanScaleOptions struct {
	Profile  turnbench.FanoutProfile   // workload profile (default turnbench.FanoutResearch)
	Cost     turnbench.FanoutCostModel // cost model (default turnbench.DefaultFanoutCostModel)
	Grid     []int                     // fan-out widths to score (default CanonicalFanScaleGrid)
	SubTurns int                       // turns per sub-agent (default defaultFanScaleSubTurns)
	Trials   int                       // seeded trials per cell (default defaultFanScaleTrials)
	Seed     int64                     // root seed (default fanScaleSeed)
}

// normalizeFanScaleGrid returns a sorted, unique, strictly-positive grid that always
// includes 1 (the single-agent baseline that prices every overhead/uplift number).
func normalizeFanScaleGrid(grid []int) []int {
	seen := map[int]bool{1: true}
	out := []int{1}
	for _, n := range grid {
		if n < 1 || seen[n] {
			continue
		}
		seen[n] = true
		out = append(out, n)
	}
	sort.Ints(out)
	return out
}

// fanScalePoint maps a scored turnbench cell into a scale point, pricing the coordination
// overhead against the baseline's critical path.
func fanScalePoint(c turnbench.FanoutCell, baseCritical float64) FanScalePoint {
	p := FanScalePoint{
		Agents:            c.Agents,
		SubTurns:          c.SubTurns,
		Trials:            c.Trials,
		Calls:             c.Calls,
		FoldTurns:         c.FoldTurns,
		CriticalPathTurns: c.CriticalPathTurns,
		TotalWorkTurns:    c.TotalWorkTurns,
		ParallelSpeedup:   c.ParallelSpeedup,
		CrossUpliftP50:    c.CrossUplift.P50,
		PrefixTokensSaved: c.PrefixTokensSaved,
		TaxClawedBackFrac: c.TaxClawedBack,
		DedupTokensSaved:  c.DedupTokensSaved,
		NetTokensSaved:    c.NetTokensSaved,
	}
	p.CoordOverheadTurns = c.CriticalPathTurns - baseCritical
	if baseCritical > 0 {
		p.CoordOverheadFrac = p.CoordOverheadTurns / baseCritical
	}
	return p
}

// RunFanScale scores the fan-out scale ladder and assembles the report. It always scores
// the N=1 baseline first (so coordination overhead is priced against one agent), then one
// cell per N in the normalized grid, reusing the witnessed turnbench fan-out engine. The
// run is deterministic in (profile, grid, subTurns, trials, seed).
//
// This is the harness, not the published headline run: it computes the geometry above for
// any grid (including N>=1024), and names the live-model/SOTA-comparison headline as
// deferred to a bench node (see DeferredRun and the package doc's placement-law note).
func RunFanScale(ctx context.Context, opt FanScaleOptions) FanScaleReport {
	if opt.SubTurns <= 0 {
		opt.SubTurns = defaultFanScaleSubTurns
	}
	if opt.Trials <= 0 {
		opt.Trials = defaultFanScaleTrials
	}
	if opt.Seed == 0 {
		opt.Seed = fanScaleSeed
	}
	grid := normalizeFanScaleGrid(opt.Grid)
	if len(grid) == 1 { // only the implicit baseline survived (nil/empty input)
		grid = normalizeFanScaleGrid(CanonicalFanScaleGrid)
	}

	prof := opt.Profile
	if prof.Name == "" {
		prof = turnbench.FanoutResearch
	}
	cm := opt.Cost
	if cm.PrefixTokens == 0 {
		cm = turnbench.DefaultFanoutCostModel()
	}

	// Baseline N=1: prices the coordination overhead (and is reused as the grid's N=1 point
	// so the surface is internally consistent).
	baseCell := turnbench.RunFanoutCell(ctx, prof, 1, opt.SubTurns, opt.Trials, opt.Seed, cm)
	baseCritical := baseCell.CriticalPathTurns
	baseline := fanScalePoint(baseCell, baseCritical)

	points := make([]FanScalePoint, 0, len(grid))
	for _, n := range grid {
		cell := baseCell
		if n != 1 {
			cell = turnbench.RunFanoutCell(ctx, prof, n, opt.SubTurns, opt.Trials, opt.Seed, cm)
		}
		points = append(points, fanScalePoint(cell, baseCritical))
	}

	return FanScaleReport{
		AppVersion:  appversion.Current(),
		GoVersion:   runtime.Version(),
		OS:          runtime.GOOS,
		GeneratedBy: "fak/internal/bench (fanscale)",
		Profile:     prof.Name,
		Seed:        opt.Seed,
		Trials:      opt.Trials,
		SubTurns:    opt.SubTurns,
		Grid:        grid,
		Baseline:    baseline,
		Points:      points,
		DeferredRun: deferredRunNote,
	}
}
