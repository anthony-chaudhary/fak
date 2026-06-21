package turnbench

// fleet.go — the TWO-DIMENSIONAL turn-tax sweep: turns-per-agent (T) × agents (A).
//
// turnbench.Run prices ONE agent running ONE frozen trace. stochastic.go turns
// that point into a distribution over an error mix, but it is still ONE agent.
// This file adds the missing axis the fleet thesis lives on: what does running A
// agents TOGETHER buy, and how does it scale with both the session length (T) and
// the fleet size (A)?
//
// The grounded agent-count lever (NOT a model). The kernel's tier-2 vDSO cache is
// keyed on (tool, args-sha256, world-version) — it is process-global, so it is
// SHARED across every agent in a world epoch. When A agents in one fleet all read
// the same reference data (a popular flight route, a policy doc), the FIRST agent
// pays a cold engine round-trip and every other agent's identical read is served
// from that one agent's result as a tier-2 hit. That cross-agent dedup is a real
// kernel event (Counters.VDSOHits / a tier-2 tag), not arithmetic — exactly as the
// single-agent turn-tax is. The "shared tool-result context across agents" claim is
// therefore MEASURABLE here, on the real k.Syscall, not just modeled in
// inline_tool_roi.py.
//
// The honest tension (why this is not just "linear win in A"). A write-shaped
// completion BUMPS the world version (vdso.Emit), which invalidates the shared read
// cache. In a fleet, one agent's book_flight invalidates the reads every OTHER
// agent had warmed. So the cross-agent benefit is a fight between dedup (more agents
// reading the same data => more hits) and invalidation (more agents writing => more
// world bumps). The net can even go NEGATIVE for a write-heavy fleet — and the
// benchmark reports that faithfully rather than assuming a monotone win.
//
// The ablation that proves cross-agent dedup is real (the path-swap analogue).
// Each cell is scored TWICE through the live kernel over the SAME generated work:
//
//   - SHARED arm:   the A agents are interleaved round-robin (the concurrent-fleet
//                   model) and replayed in ONE world epoch, so a later agent's read
//                   of an earlier agent's reference data is a genuine tier-2 hit.
//   - ISOLATED arm: each agent is replayed ALONE in its OWN world epoch (a world
//                   bump between agents), so only WITHIN-session dedup can fire.
//
// cross_uplift = shared_saved − isolated_saved is the turns the fleet deletes that
// A independent agents could not — and it equals the extra VDSOHits the shared
// epoch produced. This mirrors turnbench's vDSO ON/OFF path swap: a measured swap,
// not a subtraction of two cost models.
//
// Scope (same two-axes discipline as stochastic.go). Only the happy-path turn-tax
// classes are in the sweep — shared/private reads (tier-2 dedup), grammar repair,
// tier-1 pure, tier-3 static, and writes (which bump the world). The safety floor
// (quarantine / deny) is a completion-integrity delta on its own axis and is kept
// OUT of the fleet workload, so a fleet cell can never inflate the turn count with a
// safety event. Determinism is the same math/rand discipline: a fixed (profile, T,
// A, trials, seed) yields the identical surface.

import (
	"context"
	"encoding/json"
	"fmt"
	"math/rand"
	"sort"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/appversion"
	"github.com/anthony-chaudhary/fak/internal/ifc"
	"github.com/anthony-chaudhary/fak/internal/kernel"
	"github.com/anthony-chaudhary/fak/internal/vdso"
)

// FleetProfile is the per-turn workload mix for a fleet sweep — the probability
// that a given agent turn is each class. The remainder (1 − sum of the read/serve/
// write rates) is a FRESH private read: a first-occurrence engine pass that saves
// nothing and warms nothing cross-agent (the realistic "every loop does some
// genuinely novel lookup" filler). Every rate is an independent Bernoulli draw, so
// a profile is a compact model of how a fleet's agents actually spend their turns.
type FleetProfile struct {
	Version string `json:"version,omitempty"`
	Name    string `json:"name,omitempty"`

	// SharedPool is the number of DISTINCT reference reads the fleet shares — the
	// catalog a shared read is drawn from (popular routes / docs). Smaller pool =>
	// faster cache warm-up => cross-agent dedup saturates sooner. This is the knob
	// that makes the agent-count curve SATURATE: once A is large enough to keep the
	// whole pool warm within an epoch, each extra agent's shared reads are ~all hits.
	SharedPool int `json:"shared_pool"`

	PShared  float64 `json:"p_shared"`  // P(turn = shared-catalog read) -> cross-agent tier-2 dedup
	PPrivate float64 `json:"p_private"` // P(turn = repeat of THIS agent's own earlier read) -> intra-agent tier-2
	PAlias   float64 `json:"p_alias"`   // P(turn = aliased convert_currency) -> grammar TRANSFORM (per-agent)
	PPure    float64 `json:"p_pure"`    // P(turn = calculate{a,b}) -> tier-1 local serve (per-agent)
	PStatic  float64 `json:"p_static"`  // P(turn = list_all_airports) -> tier-3 local serve (per-agent)
	PWrite   float64 `json:"p_write"`   // P(turn = book_flight) -> engine pass + world bump (invalidation)
}

// Named fleet profiles. The default sweep profile is a read-heavy tool-use loop
// (the common shape: agents mostly look things up, occasionally book), with a
// small shared catalog so the cross-agent curve has a visible knee.
var (
	// FleetReadHeavy is the headline profile: a read/retrieval fleet (research,
	// monitoring, support-lookup) that mostly reads shared reference data, repeats a
	// few of its own lookups, and does a little arithmetic — and does NOT write. The
	// zero write rate is deliberate: the kernel bumps the GLOBAL world version on any
	// write, so a write by ANY agent invalidates EVERY agent's shared cache. A
	// read-only fleet is therefore where cross-agent dedup is cleanly positive and
	// saturating — the upper bound the "shared tool-result context" thesis predicts.
	// The write-rate axis (FleetWriteHeavy + --write-rate) maps how fast that gain
	// erodes once writes re-enter (the honest tension; see fleet.go's package doc).
	FleetReadHeavy = FleetProfile{
		Version: BenchmarkConceptVersion, Name: "read-fleet", SharedPool: 8,
		PShared: 0.45, PPrivate: 0.15, PAlias: 0.08, PPure: 0.08, PStatic: 0.05, PWrite: 0.0,
	}
	// FleetWriteHeavy stresses the invalidation tension: frequent writes keep bumping
	// the global world, so the shared cache rarely survives and the cross-agent
	// uplift is throttled — it goes NEGATIVE, because in the interleaved fleet every
	// agent's write also clears every OTHER agent's warmed reads (a cost an isolated
	// agent never pays). This is the real architectural finding: global-world-bump
	// invalidation is too coarse for a write-mixed fleet.
	FleetWriteHeavy = FleetProfile{
		Version: BenchmarkConceptVersion, Name: "write-heavy", SharedPool: 8,
		PShared: 0.35, PPrivate: 0.10, PAlias: 0.05, PPure: 0.05, PStatic: 0.05, PWrite: 0.30,
	}
	// FleetNoShare is the anti-inflation control: no shared reads AND no writes, so
	// there is nothing for cross-agent dedup to serve and no write to perturb the two
	// arms differently — the uplift MUST be exactly 0 (running A private-read agents
	// together buys nothing over running them apart). A non-zero uplift here is a
	// harness bug, not a result.
	FleetNoShare = FleetProfile{
		Version: BenchmarkConceptVersion, Name: "no-share", SharedPool: 8,
		PShared: 0.0, PPrivate: 0.20, PAlias: 0.08, PPure: 0.08, PStatic: 0.05, PWrite: 0.0,
	}
)

// routeEndpoints returns the origin/destination of the i-th catalog route. Reads
// (search_direct_flight) and writes (book_flight) both name routes through this one
// helper, so a booking and a search of the same route derive the SAME resource tag
// (flights:O-D) under scoped invalidation — the property the finer eraser needs.
func routeEndpoints(i int) (string, string) {
	origins := []string{"SFO", "LAX", "BOS", "JFK", "ORD", "SEA", "ATL", "DFW"}
	dests := []string{"JFK", "ORD", "SEA", "SFO", "LAX", "BOS", "DFW", "ATL"}
	if i < len(origins) {
		return origins[i], dests[i]
	}
	// Beyond the 8 canonical airports, synthesize DISTINCT route codes so a larger
	// shared catalog (SharedPool > 8) is genuinely distinct reads rather than args
	// that collide modulo 8 (which silently capped the old pool override at 8 real
	// routes). The first 8 routes stay byte-identical to the v0.1 catalog, so every
	// default-pool sweep is unchanged; only SharedPool > 8 now scales as intended —
	// the axis the finer eraser's ~1/pool damage-reduction needs to be measured on.
	return fmt.Sprintf("R%03dA", i), fmt.Sprintf("R%03dB", i)
}

// sharedRoute returns the canonical args for the i-th route in the shared catalog.
// Identical bytes for the same i across every agent (after canonicalJSON in the
// vDSO), so two agents reading route i collide on the tier-2 key.
func sharedRoute(i int) json.RawMessage {
	o, d := routeEndpoints(i)
	b, _ := json.Marshal(map[string]any{"origin": o, "destination": d, "date": "2026-07-01"})
	return json.RawMessage(b)
}

var (
	roMeta   = map[string]string{"readOnlyHint": "true", "idempotentHint": "true"}
	wrMeta   = map[string]string{"readOnlyHint": "false", "idempotentHint": "false", "destructive": "true"}
	aliasKey = []struct{ a, b string }{{"from", "to"}, {"source", "target"}}
)

// agentSession builds ONE agent's T-turn session deterministically from rng. The
// agent index makes private reads agent-specific (so they never collide across
// agents in tier-2), while shared reads draw from the fleet-wide catalog (so they
// DO collide). A handful of the agent's own reads are remembered and occasionally
// repeated to seed intra-agent tier-2 hits the isolated arm can also capture.
func agentSession(p FleetProfile, T, agentIdx int, rng *rand.Rand) []Call {
	calls := make([]Call, 0, T)
	priorPrivate := make([]json.RawMessage, 0, 4) // this agent's own earlier reads (intra-dedup anchors)

	privateRead := func() Call {
		uid := fmt.Sprintf("agent_%d_user_%d", agentIdx, rng.Intn(3)) // 3 users/agent => some intra repeats
		args, _ := json.Marshal(map[string]any{"user_id": uid})
		c := Call{Tool: "get_user_details", Args: json.RawMessage(args), Meta: roMeta, Class: "private"}
		priorPrivate = append(priorPrivate, c.Args)
		return c
	}

	for t := 0; t < T; t++ {
		x := rng.Float64()
		switch {
		case x < p.PShared:
			// A shared-catalog route read: the cross-agent dedup lever.
			ri := rng.Intn(max1(p.SharedPool))
			calls = append(calls, Call{Tool: "search_direct_flight", Args: sharedRoute(ri), Meta: roMeta, Class: "shared"})
		case x < p.PShared+p.PPrivate:
			// Repeat one of this agent's OWN earlier reads if it has any (intra-agent
			// tier-2 hit, visible in BOTH arms); else a fresh private read.
			if len(priorPrivate) > 0 && rng.Float64() < 0.7 {
				dup := priorPrivate[rng.Intn(len(priorPrivate))]
				calls = append(calls, Call{Tool: "get_user_details", Args: dup, Meta: roMeta, Class: "private-dup"})
			} else {
				calls = append(calls, privateRead())
			}
		case x < p.PShared+p.PPrivate+p.PAlias:
			ap := aliasKey[rng.Intn(len(aliasKey))]
			amt := 50 + rng.Intn(900)
			args, _ := json.Marshal(map[string]any{ap.a: "USD", ap.b: "EUR", "amount": amt})
			calls = append(calls, Call{Tool: "convert_currency", Args: json.RawMessage(args), Meta: roMeta, Class: "grammar"})
		case x < p.PShared+p.PPrivate+p.PAlias+p.PPure:
			args, _ := json.Marshal(map[string]any{"a": rng.Intn(1000), "b": rng.Intn(1000)})
			calls = append(calls, Call{Tool: "calculate", Args: json.RawMessage(args), Meta: roMeta, Class: "pure"})
		case x < p.PShared+p.PPrivate+p.PAlias+p.PPure+p.PStatic:
			calls = append(calls, Call{Tool: "list_all_airports", Args: json.RawMessage(`{}`), Meta: roMeta, Class: "static"})
		case x < p.PShared+p.PPrivate+p.PAlias+p.PPure+p.PStatic+p.PWrite:
			// A booking targets one of the popular catalog routes (agents book the
			// routes they search). It carries origin/destination so scoped invalidation
			// can name the route it changes (flights:O-D) — a write that touches seat
			// availability on exactly that route, sparing every OTHER route's warmed
			// reads. Under Global invalidation the same booking still strands the whole
			// board; under Resource it strands only this route's reads.
			ri := rng.Intn(max1(p.SharedPool))
			o, d := routeEndpoints(ri)
			args, _ := json.Marshal(map[string]any{
				"user_id": fmt.Sprintf("agent_%d", agentIdx), "flight_id": fmt.Sprintf("FL%d", rng.Intn(9000)),
				"origin": o, "destination": d,
			})
			calls = append(calls, Call{Tool: "book_flight", Args: json.RawMessage(args), Meta: wrMeta, Class: "write"})
		default:
			// Fresh private read: a genuinely novel lookup. Engine pass, saves nothing,
			// warms nothing cross-agent — the realistic remainder.
			calls = append(calls, privateRead())
		}
	}
	return calls
}

func max1(n int) int {
	if n < 1 {
		return 1
	}
	return n
}

// interleave round-robins the per-agent sessions into one call stream — the
// concurrent-fleet model (turn t of every agent, then turn t+1 of every agent...).
// This is what lets agent j's early reference reads land in the same world epoch as
// agent i's, so the cross-agent tier-2 hit is real. (Agent-SERIAL batching would
// place every agent's trailing writes before the next agent's reads, bumping the
// world and destroying cross-agent sharing — the wrong model for a live fleet.)
func interleave(sessions [][]Call) []Call {
	maxLen := 0
	total := 0
	for _, s := range sessions {
		if len(s) > maxLen {
			maxLen = len(s)
		}
		total += len(s)
	}
	out := make([]Call, 0, total)
	for t := 0; t < maxLen; t++ {
		for _, s := range sessions {
			if t < len(s) {
				out = append(out, s[t])
			}
		}
	}
	return out
}

// scoreWorld replays a call slice through ONE fresh kernel in ONE world epoch and
// returns the live-kernel turn-tax breakdown. It bumps the vDSO world and resets
// the IFC ledger ONCE at the start (isolating this epoch from any prior run), then
// runs every call WITHOUT a further bump — so within this slice, a duplicate
// reference read is a genuine tier-2 hit and a write advances the world mid-slice
// exactly as it would in production. The caller must have run agent.Configure().
func scoreWorld(ctx context.Context, calls []Call) ClassBreakdown {
	vdso.Default.BumpWorld()
	ifc.Default.Reset("")

	k := kernel.New("localtools")
	k.SetVDSO(true)
	res := k.Resolver()
	var cb ClassBreakdown
	var sf rawSafety
	for _, c := range calls {
		args := []byte(c.Args)
		if len(args) == 0 {
			args = []byte("{}")
		}
		ref, err := res.Put(ctx, args)
		if err != nil {
			continue
		}
		tc := &abi.ToolCall{Tool: c.Tool, Args: ref, Meta: c.Meta}
		r, v := k.Syscall(ctx, tc)
		classify(&cb, &sf, c, r, v)
	}
	return cb
}

// FleetCell is one (T, A) point: the turn-tax the fleet deletes in the shared epoch
// vs the same agents run in isolation, plus the cross-agent uplift the sharing buys.
type FleetCell struct {
	Turns  int `json:"turns"`  // T: turns per agent
	Agents int `json:"agents"` // A: fleet size
	Trials int `json:"trials"`
	Calls  int `json:"calls"` // total calls scored per trial (≈ T·A), provenance

	// Distributions across the seeded trials (p50 is the headline).
	SharedSaved   Dist `json:"shared_saved"`   // turns the interleaved fleet deletes (one world)
	IsolatedSaved Dist `json:"isolated_saved"` // sum over agents of each agent alone (own world)
	CrossUplift   Dist `json:"cross_uplift"`   // shared − isolated: the fleet-only turns

	// Per-agent normalization, ×1000 so it stays an integer-friendly Dist (the
	// surface is easier to fit on the normalized value; divide by 1000 to read it).
	SharedPerAgentMilli Dist `json:"shared_saved_per_agent_milli"`

	// Median-cell cost-model conversion of the SHARED headline (priced once on the
	// p50 so the artifact carries tokens/$/latency without re-deriving the cost).
	NetShared Net `json:"net_shared"`
	NetCross  Net `json:"net_cross"`
}

// RunFleetCell scores one (T, A) cell over `trials` seeded trials and returns the
// distributions. Each trial: generate A agent sessions, score the interleaved fleet
// in one world (SHARED), score each agent alone in its own world and sum (ISOLATED),
// and record the two totals + their difference. Per-trial seeds derive from `seed`
// deterministically, so the whole cell is reproducible.
func RunFleetCell(ctx context.Context, p FleetProfile, T, A, trials int, seed int64, cm CostModel) FleetCell {
	if trials <= 0 {
		trials = 1
	}
	agent.Configure()

	shared := make([]int, 0, trials)
	isolated := make([]int, 0, trials)
	cross := make([]int, 0, trials)
	perAgentMilli := make([]int, 0, trials)
	totalCalls := 0

	root := rand.New(rand.NewSource(seed ^ (int64(T) << 32) ^ (int64(A) << 16)))
	seeds := make([]int64, trials)
	for i := range seeds {
		seeds[i] = root.Int63()
	}

	for i := 0; i < trials; i++ {
		trng := rand.New(rand.NewSource(seeds[i]))
		sessions := make([][]Call, A)
		for a := 0; a < A; a++ {
			sessions[a] = agentSession(p, T, a, trng)
		}

		// SHARED: one interleaved epoch.
		fleet := interleave(sessions)
		totalCalls += len(fleet)
		sCB := scoreWorld(ctx, fleet)
		sSaved := sCB.turnsSaved()

		// ISOLATED: each agent its own epoch; sum the per-agent savings.
		iSaved := 0
		for a := 0; a < A; a++ {
			iSaved += scoreWorld(ctx, sessions[a]).turnsSaved()
		}

		shared = append(shared, sSaved)
		isolated = append(isolated, iSaved)
		cross = append(cross, sSaved-iSaved)
		if A > 0 {
			perAgentMilli = append(perAgentMilli, sSaved*1000/A)
		}
	}

	cell := FleetCell{
		Turns: T, Agents: A, Trials: trials,
		Calls:               totalCalls / trials,
		SharedSaved:         distOf(shared),
		IsolatedSaved:       distOf(isolated),
		CrossUplift:         distOf(cross),
		SharedPerAgentMilli: distOf(perAgentMilli),
	}
	cell.NetShared = netFor(cell.SharedSaved.P50, cm)
	cell.NetCross = netFor(cell.CrossUplift.P50, cm)
	return cell
}

// FleetSweep is the full 2-D artifact: the profile, the cost model, the sampled
// turn/agent grids, and one FleetCell per (T, A) sampled point.
type FleetSweep struct {
	AppVersion   string       `json:"app_version"`
	Profile      FleetProfile `json:"profile"`
	Cost         CostModel    `json:"cost_model"`
	Seed         int64        `json:"seed"`
	Trials       int          `json:"trials"`
	Invalidation string       `json:"invalidation"` // vDSO eraser granularity this sweep ran under
	TurnGrid     []int        `json:"turn_grid"`
	AgentGrid    []int        `json:"agent_grid"`
	Cells        []FleetCell  `json:"cells"`
	GeneratedBy  string       `json:"generated_by"`
}

// SetInvalidation selects the process-global vDSO invalidation granularity the fleet
// harness scores under (Global = the v0.1 full-flush eraser; Namespace / Resource =
// the finer erasers). The fleet sweep scores through vdso.Default, so this configures
// the eraser for the whole sweep. Returns the previous setting so a caller (e.g. a
// test) can restore it.
func SetInvalidation(g vdso.Granularity) vdso.Granularity {
	prev := vdso.Default.GranularityOf()
	vdso.Default.SetGranularity(g)
	return prev
}

// RunFleetSweep sweeps the (turnGrid × agentGrid) product, scoring each cell. It is
// SERIAL by construction: the vDSO world version is process-global, so two cells can
// not score concurrently without contaminating each other's epoch. Parallelism, if
// needed, is across PROCESSES (each has its own vdso.Default) — see cmd/fleetbench.
func RunFleetSweep(ctx context.Context, p FleetProfile, turnGrid, agentGrid []int, trials int, seed int64, cm CostModel, progress func(done, total int, c FleetCell)) FleetSweep {
	p = withFleetProfileVersion(p)
	cm = withCostModelVersion(cm)
	sw := FleetSweep{
		AppVersion: appversion.Current(),
		Profile:    p, Cost: cm, Seed: seed, Trials: trials,
		Invalidation: vdso.Default.GranularityOf().String(),
		TurnGrid:     turnGrid, AgentGrid: agentGrid,
		GeneratedBy: "fak/internal/turnbench (fleet)",
	}
	total := len(turnGrid) * len(agentGrid)
	done := 0
	for _, T := range turnGrid {
		for _, A := range agentGrid {
			c := RunFleetCell(ctx, p, T, A, trials, seed, cm)
			sw.Cells = append(sw.Cells, c)
			done++
			if progress != nil {
				progress(done, total, c)
			}
		}
	}
	return sw
}

func withFleetProfileVersion(p FleetProfile) FleetProfile {
	if p.Version == "" {
		p.Version = BenchmarkConceptVersion
	}
	return p
}

// JSON renders the sweep artifact.
func (s *FleetSweep) JSON() []byte {
	b, _ := json.MarshalIndent(s, "", "  ")
	return append(b, '\n')
}

// CSV renders the sweep as a flat grid for curve-fitting (one row per cell). The
// columns are the headline medians a model is fit against.
func (s *FleetSweep) CSV() []byte {
	var b []byte
	b = append(b, "turns,agents,calls,shared_saved_p50,isolated_saved_p50,cross_uplift_p50,shared_per_agent_p50,shared_saved_mean,cross_uplift_mean,tokens_saved_shared,dollars_saved_shared\n"...)
	rows := append([]FleetCell(nil), s.Cells...)
	sort.Slice(rows, func(i, j int) bool {
		if rows[i].Turns != rows[j].Turns {
			return rows[i].Turns < rows[j].Turns
		}
		return rows[i].Agents < rows[j].Agents
	})
	for _, c := range rows {
		b = append(b, fmt.Sprintf("%d,%d,%d,%d,%d,%d,%.3f,%.3f,%.3f,%d,%.6f\n",
			c.Turns, c.Agents, c.Calls,
			c.SharedSaved.P50, c.IsolatedSaved.P50, c.CrossUplift.P50,
			float64(c.SharedPerAgentMilli.P50)/1000.0,
			c.SharedSaved.Mean, c.CrossUplift.Mean,
			c.NetShared.TokensSaved, c.NetShared.DollarsSaved)...)
	}
	return b
}
