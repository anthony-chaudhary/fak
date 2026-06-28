// toposearch.go — the RSI loop that searches the FLEET TOPOLOGY (issue #541), the
// orthogonal STRUCTURE-search axis to policysearch.go's policy-genome search (#503).
// Where policysearch.go hill-climbs the ADJUDICATOR-POLICY genome, this file evolves the
// SHAPE of the fleet — the master->N-worker graph fanout.go hardcodes — against the SAME
// model-free replay oracle and behind the SAME divergence-frontier honesty fence. The
// search spends ZERO model calls and cannot win on a branch the frozen corpus never
// witnessed. (EvoAgentX evolves a WorkFlowGraph with a model-in-the-loop evaluator; the
// fak twist is that the evaluator is model-free measured replay and the keep/revert is a
// git-evidence witness the search did not author.)
//
// THE GENOME IS THE FLEET GRAPH, NOT THE POLICY (the decoupled-optimizer split #503 makes
// explicit). A TopologyGenome is three searchable levers fanout.go / dos.toml freeze by
// hand today:
//
//   - Width N — the fan-out width (sub-agents spawned for the one master goal). fanout.go
//     treats N as a swept INPUT; here it is a SEARCHED variable.
//   - SubTurns — the orchestrator->worker depth (turns per worker before the master folds).
//   - Lanes L — the agent->lane assignment: the N workers partitioned across L distinct
//     leaf-lanes (the `dos.toml [lanes.trees]` hand-authored partition, here searched).
//
// THE FITNESS ORACLE IS MODEL-FREE MEASURED REPLAY (the same substrate, the same $0). Each
// genome is scored by RunFanoutCell — the blessed fan-out replay path that reads MEASURED
// kernel events only (scoreWorld / k.Syscall tier-2, never an engine decode). Two honest,
// replayable savings axes (the measured halves fanout.go already computes):
//
//   - PrefixTokensSaved = (N-1)*prefix_tokens — EXACT geometry over the shipped
//     NewBatchFromPrefix clone (the prefill the kernel does not redo for N siblings).
//   - dedup turns — the MEASURED cross-agent tier-2 dedup (real k.Syscall hits) the
//     fan-out structure buys over the same workers run isolated (CrossUplift.P50).
//
// priced into a single measured-savings token figure — and the COST axis a topology
// induces on the arbiter:
//
//   - ArbiterCollisionCost — the count of worker PAIRS forced to SERIALIZE because the
//     lane assignment routes them onto the same leaf-lane. This is the exact rule
//     dos_arbitrate enforces: an exclusive holder conflicts with any intersecting tree, so
//     two workers on one lane cannot run concurrently. It is a pure-data, deterministic
//     function of (Width, Lanes): Σ C(size_j, 2) over the L lanes — 0 when every worker has
//     its own lane (fully parallel), and C(N,2) (the worst) for the hand-frozen
//     N-workers-one-lane shape. Lanes affect ONLY this cost, never the measured savings:
//     the tier-2 cache is process-global keyed by (tool, args, world-version), so siblings
//     share regardless of lane partition — the honest orthogonality that makes the Pareto
//     surface two-dimensional.
//
// RESOLVE-RATE / COMPLETION IS NOT A FITNESS TERM (the first hard fence, inherited). The
// objective is MAXIMIZE measured savings vs MINIMIZE arbiter collision; it never reads a
// task-completion or resolve-rate. A wider fan-out that "gets more done" earns nothing here
// unless its savings are MEASURED kernel events on a witnessed structure.
//
// THE DIVERGENCE GATE (the second hard fence — the whole point, #505). The frozen corpus
// records measured fan-out runs up to a FRONTIER width Wmax. Savings GROW with width (more
// siblings = more prefix reuse + more dedup), but the corpus can only WITNESS savings at a
// structure it actually recorded. A genome whose width N <= Wmax is EXACT — its measured
// savings is credited. A genome whose width N > Wmax extrapolates PAST the frontier: the
// increment it claims by going wider than anything recorded is post-frontier projection the
// frozen corpus cannot produce, so the divergence gate REFUSES to credit it (CreditedSavings
// is capped at the frontier; the refused delta is surfaced apart). A topology whose
// advantage lives only past the frontier therefore gains NOTHING in CreditedSavings and
// cannot be crowned best — the search cannot win by extrapolating to an unrecorded scale.
//
// ZERO MODEL CALLS (the third hard fence). RunFanoutCell is scoreWorld replay end to end;
// no engine decode runs during the search. ModelCallsSpent is 0 and the tests assert it.
// The search is DETERMINISTIC: a fixed (corpus, grids, seed) yields byte-identical frontiers.
//
// LIVE RE-VALIDATION IS A FLAG, NOT A RUN. A genome past the frontier is FLAGGED
// NeedsLiveRevalidation — its savings past Wmax needs a live re-run (and the #388 keep/revert
// witness AND-gate + the #387 ship-stamp referee decide from git evidence whether the
// topology is KEPT). The search PROPOSES; the kernel, from evidence the search did not
// author, decides. A self-certified topology win stays NOT_SHIPPED.
//
// SCOPE / FENCES. Additive: built ON the exported RunFanoutCell / FanoutProfile /
// FanoutCostModel (the model-free fan-out replay) — it adds the optimizer OVER that
// representation, touching no kernel, adjudicator, or shipped turnbench file. Like the rest
// of the package's replay drivers it is NOT safe to interleave with another replay
// in-process (RunFanoutSweep is serial for the same process-global-world reason).
package turnbench

import (
	"context"
	"fmt"
	"math/rand"
	"runtime"
	"sort"

	"github.com/anthony-chaudhary/fak/internal/appversion"
)

// TopologyGenome is one searched fleet-graph shape — the three levers fanout.go/dos.toml
// freeze by hand. Width and SubTurns drive the MEASURED savings (via RunFanoutCell); Lanes
// drives ONLY the arbiter collision cost (the lane partition the workers serialize on).
type TopologyGenome struct {
	Width    int `json:"width"`     // N: fan-out width (sub-agents for the one master goal)
	SubTurns int `json:"sub_turns"` // orchestrator->worker depth (turns per worker)
	Lanes    int `json:"lanes"`     // distinct leaf-lanes the N workers are partitioned across
}

// TopologyFitness is one genome's MEASURED fitness over the frozen corpus — the honest,
// replayable axes ONLY. No field is a resolve-rate or completion estimate (the first hard
// fence). Savings are token counts off RunFanoutCell's measured halves; collision is a
// pure-data count of arbiter serialization pairs.
type TopologyFitness struct {
	// CreditedSavingsTokens is the savings the divergence gate CREDITS: the measured
	// prefix-reuse + dedup token saving at the genome's width CAPPED at the corpus frontier
	// width (min(Width, Wmax)). For a witnessed genome (Width<=Wmax) this equals
	// MeasuredSavingsTokens; for one past the frontier it is held to what the corpus
	// witnessed. HIGHER is better. This is the axis the search ranks on, so an extrapolated
	// topology cannot be crowned by a saving the corpus never recorded.
	CreditedSavingsTokens int `json:"credited_savings_tokens"`
	// MeasuredSavingsTokens is the raw measured saving at the genome's FULL width (reported
	// for transparency; for a past-frontier genome it is a projection, not credited).
	MeasuredSavingsTokens int `json:"measured_savings_tokens"`
	// RefusedProjectedSavingsTokens is MeasuredSavings - CreditedSavings: the past-frontier
	// portion the divergence gate REFUSED to credit (0 for a witnessed genome). A genome
	// whose only advantage is here gains nothing in CreditedSavings — the honesty witness
	// that the search did not crown an unrecorded-scale projection.
	RefusedProjectedSavingsTokens int `json:"refused_projected_savings_tokens"`

	// ArbiterCollisionCost is the count of worker PAIRS forced to serialize because the
	// lane assignment routes them onto the same leaf-lane (Σ C(size_j,2) over L lanes) — the
	// exact dos_arbitrate tree-disjointness serialization rule, priced. LOWER is better.
	// It is the COST axis of the Pareto frontier (a width-1 topology trivially has 0
	// collision but saves nothing), so the frontier is savings-vs-collision, never a scalar.
	ArbiterCollisionCost int `json:"arbiter_collision_cost"`

	// The MEASURED components behind CreditedSavingsTokens (at the credited width), surfaced
	// apart so a "savings win" is always decomposable into the exact geometry and the real
	// kernel dedup. PrefixTokensSaved = (creditedWidth-1)*prefix_tokens (exact);
	// DedupTurns is the measured cross-agent CrossUplift.P50.
	PrefixTokensSaved int `json:"prefix_tokens_saved"`
	DedupTurns        int `json:"dedup_turns"`

	// Bounded is true iff the genome's width EXCEEDS the corpus frontier (Width>Wmax), so its
	// savings past the frontier is counterfactual and needs a live re-run. Witnessed is its
	// negation. These govern the revalidation flag; the collision cost stands regardless.
	Bounded       bool `json:"bounded"`
	CreditedWidth int  `json:"credited_width"` // min(Width, Wmax): the width the corpus witnessed
	FrontierWidth int  `json:"frontier_width"` // Wmax: the widest recorded run in the corpus
}

// TopologyCandidate is one searched fleet-graph genome plus its measured fitness and its
// live-revalidation flag — the structural dual of policysearch.go's SearchCandidate.
type TopologyCandidate struct {
	Name    string            `json:"name"`
	Genome  TopologyGenome    `json:"genome"`
	Summary map[string]string `json:"summary"` // human-readable summary of the searched levers
	Fitness TopologyFitness   `json:"fitness"`

	// NeedsLiveRevalidation is the FLAG (never an executed model run): true iff the genome's
	// width exceeds the corpus frontier, so its savings past Wmax needs a live re-run before
	// the operator trusts it (gated by the #388 keep/revert witness + #387 ship-stamp). The
	// search sets the flag; it does NOT run a model.
	NeedsLiveRevalidation bool   `json:"needs_live_revalidation"`
	RevalidationNote      string `json:"revalidation_note,omitempty"`

	// OnFrontier marks a genome Pareto-non-dominated on (CreditedSavings, ArbiterCollision).
	OnFrontier bool `json:"on_frontier"`
}

// TopologySearchReport is the issue-#541 artifact, mirroring PolicySearchReport: the search
// result over a frozen corpus of recorded fan-out runs — the hand-frozen baseline topology,
// every scored genome, the Pareto frontier of measured-savings-vs-arbiter-collision, and the
// frontier flags for live re-validation. All at $0 model spend. Deterministic for a fixed
// (corpus, grids, seed).
type TopologySearchReport struct {
	Provenance Provenance      `json:"provenance"`
	Cost       FanoutCostModel `json:"cost_model"`

	Seed          int64 `json:"seed"`           // the fixed math/rand seed (reproducible)
	Iterations    int   `json:"iterations"`     // genomes evaluated beyond the baseline
	FrontierWidth int   `json:"frontier_width"` // Wmax: the widest recorded run the corpus witnessed

	// ModelCallsSpent is ZERO — the whole point. Every genome is scored as model-free replay.
	ModelCallsSpent int64 `json:"model_calls_spent"`

	// Baseline is the hand-frozen topology (the master->N-identical-worker, single-lane shape
	// fanout.go/dos.toml encode) — the bar the structure-search beats.
	Baseline TopologyCandidate `json:"baseline"`

	// Best is the genome with the highest CreditedSavings (ties -> lower collision -> lower
	// width -> name) — the headline structure improvement. Because credit is capped at the
	// frontier, Best can never be an unrecorded-scale projection.
	Best TopologyCandidate `json:"best"`

	// NamedTopology is an optional score for a declared topology supplied by cmd/topobench.
	// It is scored by the same model-free replay oracle and frontier gate as searched
	// genomes, but is not inserted into the search frontier unless the caller also puts its
	// equivalent width/lane shape in the search grid.
	NamedTopology *TopologyCandidate `json:"named_topology,omitempty"`

	// Candidates are every scored genome (sorted by name for a stable artifact).
	Candidates []TopologyCandidate `json:"candidates"`

	// Frontier is the Pareto-non-dominated set on (CreditedSavings up, ArbiterCollision down),
	// sorted by CreditedSavings descending — the honest trade-off surface, not a scalar.
	Frontier []TopologyCandidate `json:"frontier"`

	// FlaggedForRevalidation is the subset of the frontier whose savings needs a live re-run
	// (width past the frontier). A FLAG list, never an executed model run.
	FlaggedForRevalidation []string `json:"flagged_for_live_revalidation"`
	CompletionNote         string   `json:"completion_note"`
}

// JSON renders the report (stable indentation, trailing newline) for an artifact file.
func (r *TopologySearchReport) JSON() []byte { return marshalArtifact(r) }

// CSV renders the searched genomes as a flat grid for curve-fitting the
// savings-vs-arbiter-collision frontier (one row per scored topology) — the structural dual
// of FanoutSweep.CSV and the consumable Pareto surface #541 specifies. The columns are the
// headline fitness values: the divergence-gated CREDITED savings and its measured/refused
// split first, then the arbiter collision cost, the measured components (exact prefix geometry
// + real dedup), and the divergence/frontier flags last. Rows are sorted by (width, lanes) so
// the grid reads as a surface. The two summands stay separately visible (credited vs the
// refused post-frontier projection) exactly as the JSON keeps them apart — the CSV blends
// nothing the report does not.
func (r *TopologySearchReport) CSV() []byte {
	var b []byte
	b = append(b, "name,width,sub_turns,lanes,credited_savings_tokens,measured_savings_tokens,refused_projected_savings_tokens,arbiter_collision_cost,prefix_tokens_saved,dedup_turns,credited_width,frontier_width,bounded,needs_live_revalidation,on_frontier\n"...)
	rows := append([]TopologyCandidate(nil), r.Candidates...)
	sort.Slice(rows, func(i, j int) bool {
		if rows[i].Genome.Width != rows[j].Genome.Width {
			return rows[i].Genome.Width < rows[j].Genome.Width
		}
		if rows[i].Genome.Lanes != rows[j].Genome.Lanes {
			return rows[i].Genome.Lanes < rows[j].Genome.Lanes
		}
		return rows[i].Name < rows[j].Name
	})
	for _, c := range rows {
		f := c.Fitness
		b = append(b, fmt.Sprintf("%s,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%v,%v,%v\n",
			c.Name, c.Genome.Width, c.Genome.SubTurns, c.Genome.Lanes,
			f.CreditedSavingsTokens, f.MeasuredSavingsTokens, f.RefusedProjectedSavingsTokens,
			f.ArbiterCollisionCost, f.PrefixTokensSaved, f.DedupTurns,
			f.CreditedWidth, f.FrontierWidth,
			f.Bounded, c.NeedsLiveRevalidation, c.OnFrontier)...)
	}
	return b
}

// arbiterCollisionCost prices the serialization a (width, lanes) lane assignment induces on
// the arbiter: round-robin `width` workers across `lanes` leaf-lanes, then count the worker
// PAIRS forced to serialize because they share a lane — Σ C(size_j, 2). This is exactly
// dos_arbitrate's rule (an exclusive holder conflicts with any intersecting tree, so two
// workers on one lane cannot run concurrently). 0 when every worker has its own lane
// (lanes>=width, fully parallel); C(width,2) (the worst) at lanes==1 (the hand-frozen
// N-workers-one-lane shape). Pure data — no corpus, no replay, always honest.
func arbiterCollisionCost(width, lanes int) int {
	if width < 1 {
		width = 1
	}
	if lanes < 1 {
		lanes = 1
	}
	if lanes > width {
		lanes = width
	}
	base := width / lanes
	extra := width % lanes
	cost := 0
	for j := 0; j < lanes; j++ {
		size := base
		if j < extra {
			size++
		}
		cost += size * (size - 1) / 2
	}
	return cost
}

// cellSavingsTokens is the measured savings (in tokens) a fan-out cell yields: the EXACT
// prefix-reuse geometry (N-1)*prefix_tokens PLUS the MEASURED cross-agent dedup priced into
// tokens (DedupTokensSaved). Both are read off RunFanoutCell's measured halves — no model.
func cellSavingsTokens(c FanoutCell) int {
	return c.PrefixTokensSaved + c.DedupTokensSaved
}

// scoreTopology is the FITNESS ORACLE: score ONE topology genome over the frozen corpus as
// model-free replay (RunFanoutCell), crediting savings only at-or-within the corpus frontier
// (the divergence gate) and pricing the arbiter collision the lane assignment induces. ZERO
// model calls (RunFanoutCell runs scoreWorld, never an engine decode).
func scoreTopology(ctx context.Context, p FanoutProfile, g TopologyGenome, frontierWidth, trials int, seed int64, cm FanoutCostModel) TopologyCandidate {
	creditedWidth := g.Width
	bounded := false
	if creditedWidth > frontierWidth {
		creditedWidth = frontierWidth
		bounded = true
	}

	// Credited savings: the measured saving at the WITNESSED width (capped at the frontier).
	credCell := RunFanoutCell(ctx, p, creditedWidth, g.SubTurns, trials, seed, cm)
	credited := cellSavingsTokens(credCell)

	// Raw projection at the genome's FULL width (reported; refused-credited if past frontier).
	measured := credited
	if bounded {
		projCell := RunFanoutCell(ctx, p, g.Width, g.SubTurns, trials, seed, cm)
		measured = cellSavingsTokens(projCell)
	}

	fit := TopologyFitness{
		CreditedSavingsTokens:         credited,
		MeasuredSavingsTokens:         measured,
		RefusedProjectedSavingsTokens: measured - credited,
		ArbiterCollisionCost:          arbiterCollisionCost(g.Width, g.Lanes),
		PrefixTokensSaved:             credCell.PrefixTokensSaved,
		DedupTurns:                    credCell.CrossUplift.P50,
		Bounded:                       bounded,
		CreditedWidth:                 creditedWidth,
		FrontierWidth:                 frontierWidth,
	}

	out := TopologyCandidate{
		Name:   fmt.Sprintf("topo-w%04d-l%03d", g.Width, g.Lanes),
		Genome: g,
		Summary: map[string]string{
			"width":     fmt.Sprintf("%d", g.Width),
			"sub_turns": fmt.Sprintf("%d", g.SubTurns),
			"lanes":     fmt.Sprintf("%d", g.Lanes),
		},
		Fitness: fit,
	}
	if bounded {
		out.NeedsLiveRevalidation = true
		out.RevalidationNote = fmt.Sprintf(
			"width %d exceeds the corpus frontier %d — the credited savings (%d tokens) is real "+
				"measured geometry at the witnessed width %d, but the extra %d tokens claimed by "+
				"fanning out wider than anything recorded is post-frontier projection the frozen "+
				"corpus cannot produce; it needs a LIVE re-run from the frontier (a flag, not an "+
				"executed model run) gated by the #388 keep/revert witness",
			g.Width, frontierWidth, credited, creditedWidth, measured-credited)
	}
	return out
}

// ScoreTopology scores one genome with the same model-free replay oracle, corpus-frontier
// gate, and arbiter-collision cost that RunTopologySearch uses. It is the adapter surface
// for callers that already have a declared topology instead of asking the search to propose
// one.
func ScoreTopology(ctx context.Context, p FanoutProfile, g TopologyGenome, frontierWidth, trials int, seed int64, cm FanoutCostModel) TopologyCandidate {
	return scoreTopology(ctx, withFanoutProfileVersion(p), g, frontierWidth, trials, seed, withFanoutCostModelVersion(cm))
}

// TopologySearchConfig configures the structure-search. The corpus frontier (the widest
// recorded run) sets the divergence gate; the width/lane grids are the searchable genome
// levers; Seed makes the search deterministic.
type TopologySearchConfig struct {
	// Profile is the fan-out workload mix the genomes are scored under (the goal pool, the
	// shared-read mix) — the same FanoutProfile RunFanoutCell uses.
	Profile FanoutProfile

	// FrontierWidth is Wmax: the widest fan-out the frozen corpus actually RECORDED. A genome
	// whose width exceeds it is past the divergence frontier (bounded; savings refused credit).
	// Must be >= 1.
	FrontierWidth int

	// Baseline is the HAND-FROZEN topology (the bar the search beats): the master->N-worker,
	// single-lane shape fanout.go/dos.toml encode today.
	Baseline TopologyGenome

	// WidthGrid / LaneGrid are the candidate fan-out widths and lane partitions the search
	// explores (the genome levers). The search evaluates their product (in a seeded order).
	WidthGrid []int
	LaneGrid  []int

	// Trials is the per-cell trial count handed to RunFanoutCell (the measured savings are a
	// distribution; P50 is used). Seed seeds both the proposal order and RunFanoutCell, so the
	// whole search and its frontier are byte-identical every run.
	Trials int
	Seed   int64

	// TopK bounds how many frontier genomes are flagged for live re-validation.
	TopK int
}

// RunTopologySearch drives a deterministic search over the fleet-topology genome whose
// fitness oracle is model-free fan-out replay over the frozen corpus (issue #541). It scores
// the hand-frozen baseline plus every (width × lanes) genome in the grids via scoreTopology
// (ZERO model calls), credits savings only at-or-within the corpus frontier (the divergence
// gate), and reports the Pareto frontier of credited-savings-vs-arbiter-collision with the
// frontier genomes past the corpus boundary flagged for live re-validation.
//
// The model is invoked ZERO times: scoreTopology is RunFanoutCell (scoreWorld) only. Only the
// FRONTIER genomes past the corpus width are flagged for a live re-run — the search never runs
// a model itself, and a self-certified topology win stays NOT_SHIPPED until the #388/#387
// git-evidence witness confirms it.
func RunTopologySearch(ctx context.Context, cfg TopologySearchConfig, cm FanoutCostModel) (*TopologySearchReport, error) {
	if len(cfg.WidthGrid) == 0 {
		return nil, fmt.Errorf("turnbench: RunTopologySearch needs a non-empty WidthGrid")
	}
	if len(cfg.LaneGrid) == 0 {
		return nil, fmt.Errorf("turnbench: RunTopologySearch needs a non-empty LaneGrid")
	}
	if cfg.FrontierWidth < 1 {
		return nil, fmt.Errorf("turnbench: RunTopologySearch needs FrontierWidth >= 1 (the widest recorded run)")
	}
	cm = withFanoutCostModelVersion(cm)
	p := withFanoutProfileVersion(cfg.Profile)
	trials := cfg.Trials
	if trials <= 0 {
		trials = 1
	}

	// Baseline: the hand-frozen topology, scored as-is.
	base := cfg.Baseline
	if base.Width < 1 {
		base.Width = 1
	}
	if base.Lanes < 1 {
		base.Lanes = 1
	}
	baseline := scoreTopology(ctx, p, base, cfg.FrontierWidth, trials, cfg.Seed, cm)
	baseline.Name = "baseline"

	candidates := []TopologyCandidate{baseline}

	// Deterministic proposal order: a seeded permutation of the (width × lanes) product.
	type cell struct{ w, l int }
	grid := make([]cell, 0, len(cfg.WidthGrid)*len(cfg.LaneGrid))
	for _, w := range cfg.WidthGrid {
		for _, l := range cfg.LaneGrid {
			grid = append(grid, cell{w, l})
		}
	}
	rng := rand.New(rand.NewSource(cfg.Seed))
	order := rng.Perm(len(grid))

	seen := map[string]bool{baseline.Name: true}
	for _, idx := range order {
		c := grid[idx]
		g := TopologyGenome{Width: c.w, SubTurns: base.SubTurns, Lanes: c.l}
		sc := scoreTopology(ctx, p, g, cfg.FrontierWidth, trials, cfg.Seed, cm)
		if seen[sc.Name] {
			continue
		}
		seen[sc.Name] = true
		candidates = append(candidates, sc)
	}

	// Pareto frontier on (CreditedSavings up, ArbiterCollision down); best is highest
	// CreditedSavings, ties -> lower collision -> lower width -> name (the frontier credit cap
	// means an extrapolated genome can never out-credit a witnessed one). finalizeSearch runs
	// the shared sort/stamp/best/baseline-refresh/top-k tail; only the frontier oracle and the
	// headline tie-break are topology specific.
	candidates, frontier, best, baseline, flagged := finalizeSearch(
		candidates, baseline,
		func(c TopologyCandidate) string { return c.Name },
		func(c *TopologyCandidate, on bool) { c.OnFrontier = on },
		topoParetoFrontier,
		func(c, incumbent TopologyCandidate) bool {
			return betterTopology(c.Fitness, c.Genome, incumbent.Fitness, incumbent.Genome) ||
				(equalTopology(c.Fitness, incumbent.Fitness) && c.Name < incumbent.Name)
		},
		func(c TopologyCandidate) bool { return c.NeedsLiveRevalidation },
		cfg.TopK,
	)

	return &TopologySearchReport{
		Provenance: Provenance{
			AppVersion:  appversion.Current(),
			Command:     "turnbench.RunTopologySearch",
			GoVersion:   runtime.Version(),
			OS:          runtime.GOOS,
			GeneratedBy: "fak/internal/turnbench (fleet-topology genome search over model-free replay)",
		},
		Cost:                   cm,
		Seed:                   cfg.Seed,
		Iterations:             len(candidates) - 1,
		FrontierWidth:          cfg.FrontierWidth,
		ModelCallsSpent:        0, // the whole point: the structure-search is model-free replay
		Baseline:               baseline,
		Best:                   best,
		Candidates:             candidates,
		Frontier:               frontier,
		FlaggedForRevalidation: flagged,
		CompletionNote: "a topology's CREDITED savings is sound ONLY at-or-within the corpus frontier " +
			"width; every flagged genome fans out WIDER than anything recorded, so its extra savings is " +
			"post-frontier projection that needs a LIVE re-run from the frontier (gated by the #388 " +
			"keep/revert witness + the #387 ship-stamp) before it is trusted — the credited savings and " +
			"the arbiter collision cost are real measured/pure-data quantities and stand",
	}, nil
}

// betterTopology reports whether genome a is a strictly better headline than b: more credited
// savings, ties broken by lower arbiter collision, then lower width (the cheaper structure).
func betterTopology(a TopologyFitness, ag TopologyGenome, b TopologyFitness, bg TopologyGenome) bool {
	if a.CreditedSavingsTokens != b.CreditedSavingsTokens {
		return a.CreditedSavingsTokens > b.CreditedSavingsTokens
	}
	if a.ArbiterCollisionCost != b.ArbiterCollisionCost {
		return a.ArbiterCollisionCost < b.ArbiterCollisionCost
	}
	return ag.Width < bg.Width
}

// equalTopology reports whether two fitnesses tie on the two headline axes.
func equalTopology(a, b TopologyFitness) bool {
	return a.CreditedSavingsTokens == b.CreditedSavingsTokens &&
		a.ArbiterCollisionCost == b.ArbiterCollisionCost
}

// topoParetoFrontier returns the Pareto-non-dominated genomes on (CreditedSavings up,
// ArbiterCollision down). A genome is dominated iff some other is >= on savings AND <= on
// collision, with strict inequality on at least one. Sorted by CreditedSavings descending
// (ties by collision ascending, then name) for a stable artifact.
func topoParetoFrontier(cands []TopologyCandidate) []TopologyCandidate {
	return paretoFrontierBy(cands,
		func(a, b TopologyCandidate) bool {
			af, bf := a.Fitness, b.Fitness
			geq := af.CreditedSavingsTokens >= bf.CreditedSavingsTokens && af.ArbiterCollisionCost <= bf.ArbiterCollisionCost
			gt := af.CreditedSavingsTokens > bf.CreditedSavingsTokens || af.ArbiterCollisionCost < bf.ArbiterCollisionCost
			return geq && gt
		},
		func(a, b TopologyCandidate) bool {
			if a.Fitness.CreditedSavingsTokens != b.Fitness.CreditedSavingsTokens {
				return a.Fitness.CreditedSavingsTokens > b.Fitness.CreditedSavingsTokens
			}
			if a.Fitness.ArbiterCollisionCost != b.Fitness.ArbiterCollisionCost {
				return a.Fitness.ArbiterCollisionCost < b.Fitness.ArbiterCollisionCost
			}
			return a.Name < b.Name
		})
}
