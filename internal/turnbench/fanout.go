package turnbench

// fanout.go — the ONE-MASTER-GOAL → N-SUBAGENT fan-out sweep.
//
// turnbench.Run prices ONE agent on ONE trace. fleet.go (RunFleetSweep) adds the
// second axis — A *independent* agents interleaved in one world — and measures the
// cross-agent tier-2 dedup that buys. This file adds the topology that neither
// captures and that no public benchmark maps: a SINGLE master goal that decomposes
// into N sub-agents (the orchestrator-worker / lead-subagent pattern), swept from
// N=1 to N=1000+. (See experiments/fanout/RESEARCH-BRIEF-fanout-2026-06-17.md for the
// cited state-of-the-art survey this design rests on — the headline finding is that
// the entire N≥50 regime is unbenchmarked, and shared-prefix KV reuse across N
// siblings has never been quantified for this topology.)
//
// Why the fan-out is the IDEAL case for fak's two levers. A flat fleet (fleet.go) is
// A *different* agents that happen to overlap on a shared catalog. A fan-out is N
// sub-agents decomposing the SAME goal, so the sharing is structurally higher: every
// sub-agent reads the goal's reference data (the GoalPool), and a master "plan" warms
// that catalog before the wave. That is exactly the regime where the kernel's two
// real levers pay off the most:
//
//   - Shared-prefix KV reuse (MEASURED as exact geometry). N sub-agents share the
//     master-goal prefix (system + goal + tool schemas). model.NewBatchFromPrefix
//     prefills that prefix ONCE and clones its KV into all N sub-agents (bit-identical
//     — model/kvreuse_test.go:TestKVPrefixReuseMatchesRecompute proves a clone prefills
//     only the SUFFIX, "skipping the prefix's prefill FLOPs"; cmd/fleetserve wall-clocks
//     the reuse-vs-no-reuse win). The prefill the kernel does NOT redo is therefore
//     prefix_tokens_saved = (N−1)·prefix_tokens — exact, not modeled.
//   - Cross-agent tool-result dedup (MEASURED as real kernel events). The kernel's
//     tier-2 vDSO cache is keyed (tool, args-sha256, world-version) and process-global,
//     so a sub-agent's read of goal data an earlier sub-agent already fetched is a
//     genuine tier-2 hit the kernel counts itself. Each cell is ablated SHARED (the
//     whole fan-out in one world epoch) vs ISOLATED (the plan + one sub-agent in its
//     own world, repeated for every sub-agent), and cross_uplift = shared − isolated
//     is the fan-out-ONLY sibling dedup — a measured path-swap, identical in
//     discipline to fleet.go.
//
// The honest line (same as TURN-TAX §3.2 / FLEET-SWEEP). The dedup turns are REAL
// kernel events. The prefix-reuse saving is EXACT geometry over a shipped kernel
// property. Everything downstream — the token multiplier, the critical-path-vs-total-
// work latency, throughput, the saturation knee — is a TRANSPARENT, knobbed cost model
// (FanoutCostModel), reported separately and never blended into the measured halves.
// Determinism is the fleet.go discipline: a fixed (profile, N, sub-turns, trials, seed)
// yields the identical surface. Every sweep includes the N=1 single-agent control, so
// a "fan-out win" is always priced against doing the goal with one agent (the
// budget-controlled comparison the literature insists on).

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"math/rand"
	"sort"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/appversion"
)

// FanoutProfile is the per-sub-agent workload mix for ONE master goal fanned out to N
// sub-agents. It reuses the fleet per-turn class mix (so the kernel events are scored
// by the SAME path fleet.go uses), but adds the master-goal structure: a GoalPool the
// sub-agents all draw shared reads from, and a PlanReads warm-up the lead issues before
// the wave.
type FanoutProfile struct {
	Version string `json:"version,omitempty"`
	Name    string `json:"name,omitempty"`

	// GoalPool is the number of DISTINCT reference reads the master goal's context spans
	// — the catalog every sub-agent's shared reads are drawn from. A small pool warms
	// fast, so cross-agent dedup saturates at a lower N (the knob that bends the curve).
	GoalPool int `json:"goal_pool"`

	// PlanReads is the count of master-decomposition reads issued BEFORE the fan-out
	// (the lead reading the goal's reference data to split it into sub-tasks). In the
	// SHARED arm these warm the goal cache for every sub-agent; in ISOLATED each
	// sub-agent gets its own plan warm-up, so cross_uplift excludes the single-agent
	// plan-to-sub-agent benefit and measures sibling sharing only.
	PlanReads int `json:"plan_reads"`

	// Per-sub-agent-turn class mix (independent Bernoulli draws; the remainder is a
	// fresh private read that saves/warms nothing). Mirrors FleetProfile exactly.
	PShared  float64 `json:"p_shared"`  // shared goal-catalog read -> cross-agent tier-2 dedup
	PPrivate float64 `json:"p_private"` // repeat of this sub-agent's own earlier read -> intra-agent tier-2
	PAlias   float64 `json:"p_alias"`   // aliased convert_currency -> grammar TRANSFORM (per-agent)
	PPure    float64 `json:"p_pure"`    // calculate -> tier-1 pure local serve (per-agent)
	PStatic  float64 `json:"p_static"`  // list_all_airports -> tier-3 static local serve (per-agent)
	PWrite   float64 `json:"p_write"`   // book_flight -> engine pass + world bump (invalidation)
}

// fleet projects the fan-out profile onto a FleetProfile so the SAME agentSession
// generator (fleet.go) produces each sub-agent's turns — guaranteeing the kernel
// events are scored identically to the fleet sweep.
func (p FanoutProfile) fleet() FleetProfile {
	return FleetProfile{
		Version: p.Version, Name: p.Name, SharedPool: p.GoalPool,
		PShared: p.PShared, PPrivate: p.PPrivate, PAlias: p.PAlias,
		PPure: p.PPure, PStatic: p.PStatic, PWrite: p.PWrite,
	}
}

// Named fan-out profiles.
var (
	// FanoutResearch is the headline profile: the orchestrator-worker research goal
	// (Anthropic's lead-researcher → 3–5 subagent pattern, scaled to 1000+). Sub-agents
	// mostly read the goal's shared sources (high PShared — they decompose ONE goal), a
	// small GoalPool so the cross-agent curve has a visible knee, a plan warm-up, and no
	// writes — the upper bound where cross-agent dedup is cleanly positive.
	FanoutResearch = FanoutProfile{
		Version: BenchmarkConceptVersion, Name: "research-goal", GoalPool: 8, PlanReads: 4,
		PShared: 0.55, PPrivate: 0.15, PAlias: 0.06, PPure: 0.06, PStatic: 0.04, PWrite: 0.0,
	}
	// FanoutWriteHeavy stresses the invalidation tension: sub-agents that write (book /
	// modify) bump the global world, clearing every sibling's warmed goal reads — so the
	// cross-agent uplift is throttled and can go negative, exactly as in the flat fleet.
	FanoutWriteHeavy = FanoutProfile{
		Version: BenchmarkConceptVersion, Name: "write-goal", GoalPool: 8, PlanReads: 4,
		PShared: 0.40, PPrivate: 0.10, PAlias: 0.05, PPure: 0.05, PStatic: 0.05, PWrite: 0.25,
	}
	// FanoutNoShare is the anti-inflation control: no shared goal reads, no plan, no
	// writes — there is nothing for cross-agent dedup to serve, so the uplift MUST be
	// exactly 0 at every N (running N sub-agents together buys nothing over apart). A
	// non-zero uplift here is a harness bug, not a result.
	FanoutNoShare = FanoutProfile{
		Version: BenchmarkConceptVersion, Name: "no-share", GoalPool: 8, PlanReads: 0,
		PShared: 0.0, PPrivate: 0.20, PAlias: 0.06, PPure: 0.06, PStatic: 0.04, PWrite: 0.0,
	}
)

// FanoutCostModel converts a fan-out run into the MODELED headline numbers — the token
// multiplier (vs a single agent), the prompt-cache prefix economics, and the
// critical-path-vs-total-work latency. Every field is a knob; the defaults are
// documented and illustrative, not billed. The MEASURED halves (cross-agent dedup
// turns, prefix-reuse geometry) are computed elsewhere; this model only PRICES them,
// exactly as turnbench.CostModel prices the single-agent turn-tax.
type FanoutCostModel struct {
	Version             string  `json:"version,omitempty"`
	PrefixTokens        int     `json:"prefix_tokens"`          // master-goal shared prefix P (system + goal + tool schemas)
	SuffixTokens        int     `json:"suffix_tokens"`          // per-sub-agent private prompt (the sub-task slice)
	DecodeTokensPerTurn int     `json:"decode_tokens_per_turn"` // assistant tokens generated per sub-agent turn
	FoldTokensPerResult int     `json:"fold_tokens_per_result"` // master fold: tokens to ingest one sub-result
	FoldTurnTokenBudget int     `json:"fold_turn_token_budget"` // tokens the master synthesizes per fold turn
	DollarsPerMTokIn    float64 `json:"dollars_per_mtok_in"`
	DollarsPerMTokOut   float64 `json:"dollars_per_mtok_out"`
	CacheReadMult       float64 `json:"cache_read_mult"`  // cached-input price multiple (Anthropic ~0.1 = 90% off)
	CacheWriteMult      float64 `json:"cache_write_mult"` // cache-write price multiple (Anthropic ~1.25)
	TurnLatencyMs       float64 `json:"turn_latency_ms"`  // one model round-trip
}

// DefaultFanoutCostModel is a documented, conservative orchestrator-worker turn: a
// ~2K-token shared goal prefix (the regime where prefix-reuse pays), a short per-agent
// suffix, ~120-tok answers, a 200-tok-per-result fold, at the repo's blended $3/$15
// rate with Anthropic's 0.1×-read / 1.25×-write prompt-cache multiples and a ~1.5s
// round-trip. All overridable on the CLI.
func DefaultFanoutCostModel() FanoutCostModel {
	return FanoutCostModel{
		Version:      FanoutCostModelVersion,
		PrefixTokens: 2048, SuffixTokens: 256, DecodeTokensPerTurn: 120,
		FoldTokensPerResult: 200, FoldTurnTokenBudget: 4000,
		DollarsPerMTokIn: 3.0, DollarsPerMTokOut: 15.0,
		CacheReadMult: 0.1, CacheWriteMult: 1.25, TurnLatencyMs: 1500,
	}
}

// fanoutProjection is the modeled cost/latency block for one cell.
type fanoutProjection struct {
	tokenMultNaive, tokenMultReuse, taxClawedBack                                   float64
	criticalPathTurns, totalWorkTurns, parallelSpeedup, foldTurns, agentTurnsPerSec float64
	dedupTokensSaved, netTokensSaved                                                int
	netDollarsSaved                                                                 float64
}

// project prices one (N, subTurns) cell. The two levers are kept STRICTLY APART (the
// honesty line in this file's package doc): the token multiplier / tax-clawed-back is
// the MODELED prefix-cache lever ALONE — smooth and monotone, never contaminated by the
// measured dedup — while dedupTurnsSaved (the MEASURED cross_uplift.P50, fan-out-only so
// a single agent's own intra-agent dedup is never double-counted) is priced into its OWN
// field, and only the explicitly-named net field sums the two.
func (cm FanoutCostModel) project(N, subTurns, dedupTurnsSaved int) fanoutProjection {
	if N < 1 {
		N = 1
	}
	P := float64(cm.PrefixTokens)
	S := float64(cm.SuffixTokens)
	D := float64(subTurns) * float64(cm.DecodeTokensPerTurn) // per-agent decode output
	foldTok := float64(N) * float64(cm.FoldTokensPerResult)
	foldTurns := math.Ceil(foldTok / math.Max(1, float64(cm.FoldTurnTokenBudget)))

	// --- token multiplier (vs the N=1 single-agent control): PREFIX-CACHE LEVER ONLY ---
	single := P + S + D
	// Naive multi-agent: every sub-agent materializes the full prefix; plus the fold.
	naive := float64(N)*(P+S+D) + foldTok
	// Reuse: the master-goal prefix is written to cache ONCE (CacheWriteMult) and read
	// cheap by the other N−1 sub-agents (CacheReadMult) — the NewBatchFromPrefix lever
	// priced as Anthropic prompt-cache $/token. NO measured dedup is blended in here, so
	// this stays a smooth, monotone, fully-modeled curve.
	reuse := P*(cm.CacheWriteMult+float64(N-1)*cm.CacheReadMult) + float64(N)*(S+D) + foldTok

	var pr fanoutProjection
	if single > 0 {
		pr.tokenMultNaive = naive / single
		pr.tokenMultReuse = reuse / single
	}
	if pr.tokenMultNaive > 1 {
		pr.taxClawedBack = (pr.tokenMultNaive - pr.tokenMultReuse) / (pr.tokenMultNaive - 1)
	}
	if pr.taxClawedBack < 0 {
		pr.taxClawedBack = 0
	}
	if pr.taxClawedBack > 1 {
		pr.taxClawedBack = 1
	}

	// --- latency: critical path (depth) vs total work (sum) ---
	// The N sub-agents run in parallel, so the wave costs ONE sub-agent's turns; the
	// plan is one turn and the fold grows with N (the synchronous-join coordination tax,
	// the term that makes throughput saturate — the literature's "lead waits on the
	// slowest of N" and "coordination overhead grows with N").
	pr.foldTurns = foldTurns
	pr.criticalPathTurns = 1 + float64(subTurns) + foldTurns
	pr.totalWorkTurns = 1 + float64(N*subTurns) + foldTurns
	if pr.criticalPathTurns > 0 {
		pr.parallelSpeedup = pr.totalWorkTurns / pr.criticalPathTurns
		pr.agentTurnsPerSec = pr.totalWorkTurns / (pr.criticalPathTurns * cm.TurnLatencyMs / 1000.0)
	}

	// --- MEASURED dedup lever, priced into its OWN field (conservative per-turn marginal
	// S+decode, never the cached prefix); can be negative for a write-mixed fan-out where
	// cross-agent invalidation makes sharing a net cost. ---
	pr.dedupTokensSaved = dedupTurnsSaved * (cm.SuffixTokens + cm.DecodeTokensPerTurn)

	// --- combined headline: the MODELED prefix-cache token saving (naive − reuse) PLUS
	// the MEASURED dedup saving. Reported as one number, but the two summands stay
	// separately visible (tax_clawed_back / token_mult_* are pure-modeled;
	// dedup_tokens_saved is pure-measured). ---
	prefixCacheSaved := int(naive - reuse)
	pr.netTokensSaved = prefixCacheSaved + pr.dedupTokensSaved
	pr.netDollarsSaved = float64(pr.netTokensSaved) / 1e6 * cm.DollarsPerMTokIn
	return pr
}

// FanoutCell is one (N, subTurns) point: the dedup the fan-out deletes vs the same
// sub-agents run isolated, the exact prefix-reuse geometry, and the modeled cost/
// latency projection.
type FanoutCell struct {
	Agents       int `json:"agents"`        // N: fan-out width (sub-agents spawned for the one goal)
	SubTurns     int `json:"sub_turns"`     // turns per sub-agent
	PrefixTokens int `json:"prefix_tokens"` // master-goal shared prefix tokens P for this cell
	Trials       int `json:"trials"`
	Calls        int `json:"calls"` // total kernel calls scored per trial in the SHARED arm (≈ plan + N·subTurns)

	// ---- MEASURED: cross-agent tool-result dedup (real k.Syscall tier-2 events) ----
	SharedSaved   Dist `json:"shared_saved"`   // turns the fan-out deletes in ONE world epoch
	IsolatedSaved Dist `json:"isolated_saved"` // each sub-agent run solo (its own plan + own world), summed
	CrossUplift   Dist `json:"cross_uplift"`   // shared − isolated: the fan-out-ONLY sibling dedup

	// ---- MEASURED (exact geometry): shared-prefix KV reuse ----
	// (N−1)·PrefixTokens prefill positions the kernel does NOT redo, because
	// NewBatchFromPrefix prefills the master-goal prefix once and clones it into all N
	// sub-agents (bit-identical; witnessed by model.TestKVPrefixReuseMatchesRecompute and
	// cmd/fanbench's TestPrefixReuseFanoutWitness, wall-clocked by cmd/fleetserve).
	PrefixTokensSaved int `json:"prefix_tokens_saved"`

	// ---- MODELED: cost / latency projection (transparent FanoutCostModel) ----
	TokenMultNaive    float64 `json:"token_mult_naive"`     // naive multi-agent input+output cost ÷ single agent
	TokenMultReuse    float64 `json:"token_mult_reuse"`     // with the prefix-cache lever ALONE (pure-modeled, no dedup blended)
	TaxClawedBack     float64 `json:"tax_clawed_back_frac"` // fraction of the (naive−1) token tax the PREFIX-CACHE lever removes
	CriticalPathTurns float64 `json:"critical_path_turns"`  // plan + slowest sub-agent + fold (depth latency)
	TotalWorkTurns    float64 `json:"total_work_turns"`     // plan + N·sub-agent + fold (sum)
	ParallelSpeedup   float64 `json:"parallel_speedup"`     // total ÷ critical
	FoldTurns         float64 `json:"fold_turns"`           // master synthesis turns (grows with N: the coordination tax)
	AgentTurnsPerSec  float64 `json:"agent_turns_per_sec"`  // total work ÷ critical-path wall-time
	DedupTokensSaved  int     `json:"dedup_tokens_saved"`   // MEASURED cross_uplift.P50 priced (the dedup lever, kept apart)
	NetTokensSaved    int     `json:"net_tokens_saved"`     // modeled prefix-cache saving + measured dedup saving
	NetDollarsSaved   float64 `json:"net_dollars_saved"`
}

// genGoal builds the master plan + N sub-agent sessions for ONE trial. The plan is
// PlanReads shared goal-catalog reads (the lead warming the goal's reference data);
// each sub-agent is a fleet-style subTurns-turn session over the SAME goal catalog, so
// its shared reads collide with the plan's and with sibling sub-agents'.
func genGoal(p FanoutProfile, N, subTurns int, rng *rand.Rand) (plan []Call, subs [][]Call) {
	for i := 0; i < p.PlanReads; i++ {
		ri := i % max1(p.GoalPool)
		plan = append(plan, Call{Tool: "search_direct_flight", Args: sharedRoute(ri), Meta: roMeta, Class: "plan"})
	}
	fp := p.fleet()
	subs = make([][]Call, N)
	for a := 0; a < N; a++ {
		subs[a] = agentSession(fp, subTurns, a, rng)
	}
	return plan, subs
}

func sumLen(sessions [][]Call) int {
	n := 0
	for _, s := range sessions {
		n += len(s)
	}
	return n
}

// RunFanoutCell scores one (N, subTurns) cell over `trials` seeded trials. Each trial:
// generate the plan + N sub-agent sessions; score the SHARED arm (plan + sub-agents
// interleaved in one world epoch) and the ISOLATED arm (plan + one sub-agent in its
// own world, summed across the wave); record the two turn-tax totals and their
// difference. Per-trial seeds derive from `seed` deterministically, so the whole cell
// is reproducible.
func RunFanoutCell(ctx context.Context, p FanoutProfile, N, subTurns, trials int, seed int64, cm FanoutCostModel) FanoutCell {
	if trials <= 0 {
		trials = 1
	}
	if N < 1 {
		N = 1
	}
	agent.Configure()

	shared := make([]int, 0, trials)
	isolated := make([]int, 0, trials)
	cross := make([]int, 0, trials)
	totalCalls := 0

	root := rand.New(rand.NewSource(seed ^ (int64(N) << 20) ^ (int64(subTurns) << 8)))
	seeds := make([]int64, trials)
	for i := range seeds {
		seeds[i] = root.Int63()
	}

	for i := 0; i < trials; i++ {
		trng := rand.New(rand.NewSource(seeds[i]))
		plan, subs := genGoal(p, N, subTurns, trng)

		// SHARED: plan + all sub-agents interleaved round-robin in ONE world epoch, so a
		// later sub-agent's read of goal data an earlier one fetched is a genuine tier-2
		// hit.
		stream := make([]Call, 0, len(plan)+sumLen(subs))
		stream = append(stream, plan...)
		stream = append(stream, interleave(subs)...)
		totalCalls += len(stream)
		sSaved := scoreWorld(ctx, stream).turnsSaved()

		// ISOLATED: each sub-agent gets its own plan warm-up in its own epoch, so the
		// N=1 control has zero cross uplift and the measured delta is sibling-only.
		iSaved := 0
		for a := 0; a < N; a++ {
			alone := make([]Call, 0, len(plan)+len(subs[a]))
			alone = append(alone, plan...)
			alone = append(alone, subs[a]...)
			iSaved += scoreWorld(ctx, alone).turnsSaved()
		}

		shared = append(shared, sSaved)
		isolated = append(isolated, iSaved)
		cross = append(cross, sSaved-iSaved)
	}

	cell := FanoutCell{
		Agents: N, SubTurns: subTurns, PrefixTokens: cm.PrefixTokens, Trials: trials,
		Calls:             totalCalls / trials,
		SharedSaved:       distOf(shared),
		IsolatedSaved:     distOf(isolated),
		CrossUplift:       distOf(cross),
		PrefixTokensSaved: (N - 1) * cm.PrefixTokens,
	}
	pr := cm.project(N, subTurns, cell.CrossUplift.P50)
	cell.TokenMultNaive = pr.tokenMultNaive
	cell.TokenMultReuse = pr.tokenMultReuse
	cell.TaxClawedBack = pr.taxClawedBack
	cell.CriticalPathTurns = pr.criticalPathTurns
	cell.TotalWorkTurns = pr.totalWorkTurns
	cell.ParallelSpeedup = pr.parallelSpeedup
	cell.FoldTurns = pr.foldTurns
	cell.AgentTurnsPerSec = pr.agentTurnsPerSec
	cell.DedupTokensSaved = pr.dedupTokensSaved
	cell.NetTokensSaved = pr.netTokensSaved
	cell.NetDollarsSaved = pr.netDollarsSaved
	return cell
}

// FanoutSweep is the full artifact: the profile, the cost model, the sampled grids,
// and one FanoutCell per (subTurns, N) point.
type FanoutSweep struct {
	AppVersion  string          `json:"app_version"`
	Profile     FanoutProfile   `json:"profile"`
	Cost        FanoutCostModel `json:"cost_model"`
	Seed        int64           `json:"seed"`
	Trials      int             `json:"trials"`
	AgentGrid   []int           `json:"agent_grid"`
	SubTurnGrid []int           `json:"sub_turn_grid"`
	PrefixGrid  []int           `json:"prefix_grid,omitempty"`
	Cells       []FanoutCell    `json:"cells"`
	GeneratedBy string          `json:"generated_by"`
}

// RunFanoutSweep sweeps the (subTurnGrid × agentGrid) product. It is SERIAL by
// construction: the vDSO world version is process-global, so two cells cannot score
// concurrently without contaminating each other's epoch (parallelism is across
// PROCESSES — shard the agent grid; see cmd/fanbench).
func RunFanoutSweep(ctx context.Context, p FanoutProfile, agentGrid, subTurnGrid []int, trials int, seed int64, cm FanoutCostModel, progress func(done, total int, c FanoutCell)) FanoutSweep {
	return RunFanoutPrefixSweep(ctx, p, agentGrid, subTurnGrid, []int{cm.PrefixTokens}, trials, seed, cm, progress)
}

// RunFanoutPrefixSweep sweeps the (prefixGrid × subTurnGrid × agentGrid) product. The
// per-cell prefix dimension lets the same stochastic fan-out surface be priced from
// smoke-sized prompts through near-full-context prompts without hiding P in a global
// cost block.
func RunFanoutPrefixSweep(ctx context.Context, p FanoutProfile, agentGrid, subTurnGrid, prefixGrid []int, trials int, seed int64, cm FanoutCostModel, progress func(done, total int, c FanoutCell)) FanoutSweep {
	p = withFanoutProfileVersion(p)
	cm = withFanoutCostModelVersion(cm)
	if len(prefixGrid) == 0 {
		prefixGrid = []int{cm.PrefixTokens}
	}
	sw := FanoutSweep{
		AppVersion: appversion.Current(),
		Profile:    p, Cost: cm, Seed: seed, Trials: trials,
		AgentGrid: agentGrid, SubTurnGrid: subTurnGrid, PrefixGrid: prefixGrid,
		GeneratedBy: "fak/internal/turnbench (fanout)",
	}
	total := len(prefixGrid) * len(subTurnGrid) * len(agentGrid)
	done := 0
	for _, P := range prefixGrid {
		cellCost := cm
		cellCost.PrefixTokens = P
		for _, T := range subTurnGrid {
			for _, N := range agentGrid {
				c := RunFanoutCell(ctx, p, N, T, trials, seed, cellCost)
				sw.Cells = append(sw.Cells, c)
				done++
				if progress != nil {
					progress(done, total, c)
				}
			}
		}
	}
	return sw
}

func withFanoutProfileVersion(p FanoutProfile) FanoutProfile {
	if p.Version == "" {
		p.Version = BenchmarkConceptVersion
	}
	return p
}

func withFanoutCostModelVersion(cm FanoutCostModel) FanoutCostModel {
	if cm.Version == "" {
		cm.Version = FanoutCostModelVersion
	}
	return cm
}

// JSON renders the sweep artifact.
func (s *FanoutSweep) JSON() []byte {
	b, _ := json.MarshalIndent(s, "", "  ")
	return append(b, '\n')
}

// CSV renders the sweep as a flat grid for curve-fitting (one row per cell). The
// columns are the headline values: the MEASURED dedup + prefix geometry first, then
// the MODELED projection.
func (s *FanoutSweep) CSV() []byte {
	var b []byte
	b = append(b, "agents,sub_turns,prefix_tokens,calls,shared_saved_p50,isolated_saved_p50,cross_uplift_p50,prefix_tokens_saved,token_mult_naive,token_mult_reuse,tax_clawed_back,critical_path_turns,total_work_turns,parallel_speedup,agent_turns_per_sec,dedup_tokens_saved,net_tokens_saved,net_dollars_saved\n"...)
	rows := append([]FanoutCell(nil), s.Cells...)
	sort.Slice(rows, func(i, j int) bool {
		if rows[i].PrefixTokens != rows[j].PrefixTokens {
			return rows[i].PrefixTokens < rows[j].PrefixTokens
		}
		if rows[i].SubTurns != rows[j].SubTurns {
			return rows[i].SubTurns < rows[j].SubTurns
		}
		return rows[i].Agents < rows[j].Agents
	})
	for _, c := range rows {
		b = append(b, fmt.Sprintf("%d,%d,%d,%d,%d,%d,%d,%d,%.4f,%.4f,%.4f,%.1f,%.1f,%.3f,%.4f,%d,%d,%.6f\n",
			c.Agents, c.SubTurns, c.PrefixTokens, c.Calls,
			c.SharedSaved.P50, c.IsolatedSaved.P50, c.CrossUplift.P50,
			c.PrefixTokensSaved,
			c.TokenMultNaive, c.TokenMultReuse, c.TaxClawedBack,
			c.CriticalPathTurns, c.TotalWorkTurns, c.ParallelSpeedup, c.AgentTurnsPerSec,
			c.DedupTokensSaved, c.NetTokensSaved, c.NetDollarsSaved)...)
	}
	return b
}
