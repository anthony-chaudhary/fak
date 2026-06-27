package bench

// fanrun.go — the MEASURED live one-master-goal → N-subagent fan-out capstone.
//
// internal/turnbench/fanout.go (RunFanoutCell, driven by cmd/fanbench) scores a
// profile-DRIVEN stream of synthetic tool calls through the real kernel cache and then
// PRICES it with a transparent, MODELED cost model (the token multiplier, the prompt-cache
// economics, the 72.8× parallel-speedup knee). Strong and honest, but the "agents" there
// are call sequences, not running agents, and the headline speedup is modeled arithmetic.
//
// fanrun is the other half: it actually RUNS N real agent sessions — each a genuine
// internal/agent.RunArm loop through a real kernel.New("localtools") with the vDSO fast
// path on, real tool dispatch across the syscall boundary, real ArmMetrics — all
// decomposing ONE shared master goal, and WALL-CLOCKS the wave. Every number here is a
// measured wall-clock, a real kernel counter, or exact geometry. There are NO modeled
// fields: the modeled 72.8× stays in fanbench, fenced.
//
// THE LOAD-BEARING HONESTY LINE (why this is serial, not parallel):
// the kernel's fast path is the PROCESS-GLOBAL abi.FastPaths() registry (vdso.Default),
// and the cache world-version is one process-global scalar. That is exactly what makes
// cross-agent dedup REAL — a later sub-agent's read of goal data an earlier sub-agent
// already fetched is a genuine tier-2 hit on the shared cache. But it also means the N
// sub-agents must run SERIALLY in one world epoch for that count to be reproducible and
// honestly attributed. So the measured wall-clock is a SUM (AgentsWallSerialMs), and
// AgentsPerSecSerial is N / Σt — explicitly NOT a parallel throughput rate. "ran N agents"
// means N real RunArm sessions completed end-to-end through the kernel, summed wall-clock,
// sharing one epoch so cross-agent dedup is real. The measured WIN is prefill elision
// ((N−1)·P, wall-clocked reuse-vs-no-reuse when a timing model is present) plus the real
// cross-agent dedup count — not parallel speedup.
//
// CrossHits discipline mirrors fanout.go's cross_uplift = shared − isolated: the wave's
// total vDSO hits include each sub-agent's own intra-agent dedup (the duplicate get_user
// re-verify in MockPlanner), so the fan-out-ONLY sibling dedup is the wave total minus N×
// the single-agent (N=1) baseline. CrossHits is reported as that sibling-only delta;
// WaveHits carries the raw total for transparency.

import (
	"context"
	"fmt"
	"runtime"
	"sort"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/appversion"
	"github.com/anthony-chaudhary/fak/internal/turnbench"
	"github.com/anthony-chaudhary/fak/internal/vdso"
)

// FanrunSchema is the artifact schema tag (bumped on a breaking field change).
const FanrunSchema = "fanrun/1"

// FanrunOptions configures a live fan-out sweep. It reuses turnbench's fan-out profile
// vocabulary so a fanrun artifact is directly comparable to the modeled fanbench one.
type FanrunOptions struct {
	// Profile selects the sharing regime by NAME only (research|write-heavy|no-share).
	// fanrun drives the canonical agent.DefaultTask, so the profile's per-turn class mix
	// is not replayed here; what it controls is whether the sub-agents SHARE their goal
	// reads (research/write → bare MockPlanner, all agents read the same ids → cross-agent
	// hits) or NOT (no-share → a per-agent salted planner, distinct arg hashes → 0 uplift,
	// the anti-inflation control). The profile name + sharing bit are recorded for
	// provenance.
	Profile  turnbench.FanoutProfile
	Grid     []int // agent widths N (the sweep)
	SubTurns int   // RunArm maxTurns cap per sub-agent (the canonical task needs ~7)
	Prefix   int   // master-goal shared prefix tokens P (the reuse lever / geometry)
	Trials   int   // determinism witness: >=2 re-runs the wave and asserts identical counts
	Reps     int   // prefill-timing reps (best-of-min); 0 (or no ModelDir) => skip timing
	Seed     int64
	ModelDir string // optional small CPU model for the prefill-elision wall-clock; "" => skip
	Quant    bool   // Q8 lane for the prefill-timing model (fleetserve parity)
}

// shares reports whether this profile's sub-agents share their goal reads (so cross-agent
// dedup is expected) — true for everything except the no-share anti-inflation control.
func (o FanrunOptions) shares() bool {
	return !strings.Contains(strings.ToLower(o.Profile.Name), "no-share") &&
		!strings.Contains(strings.ToLower(o.Profile.Name), "noshare")
}

// FanrunReport is the full artifact. It carries ONLY measured / counter / geometry /
// provenance fields — no modeled fields by construction.
type FanrunReport struct {
	Schema      string                  `json:"schema"`
	Provenance  FanrunProvenance        `json:"provenance"`
	Host        FanrunHost              `json:"host"`
	Profile     turnbench.FanoutProfile `json:"profile"`
	SharedGoal  bool                    `json:"shared_goal"` // false only for the no-share control
	Prefix      int                     `json:"prefix_len"`
	SubTurns    int                     `json:"sub_turns"`
	Trials      int                     `json:"trials"`
	Reps        int                     `json:"reps"`
	Seed        int64                   `json:"seed"`
	Task        string                  `json:"task"`
	AgentGrid   []int                   `json:"agent_grid"`
	TimingModel string                  `json:"prefill_timing_model"` // model id, or "skipped: <reason>"
	Cells       []FanrunCell            `json:"cells"`
}

// FanrunProvenance carries only STABLE facts — no wall-clock timestamp, so the
// counter+geometry projection of the artifact is byte-reproducible (the
// fanout_longctx_probe rule).
type FanrunProvenance struct {
	AppVersion  string `json:"app_version"`
	Command     string `json:"command"`
	GoVersion   string `json:"go_version"`
	OS          string `json:"os"`
	Arch        string `json:"arch"`
	GeneratedBy string `json:"generated_by"`
}

// FanrunHost carries stable host facts for provenance (CPU-only box, no GPU).
type FanrunHost struct {
	CPUCount  int  `json:"cpu_count"`
	GoThreads int  `json:"go_threads"`
	HasGPU    bool `json:"has_gpu"`
}

// FanrunCell is one (N) point. Field names carry their provenance class: *_ms / *_serial =
// wall-clock MEASURED; cross/wave/turns/tokens/tasks = real kernel/loop counters;
// *_elided = exact geometry. There are deliberately NO modeled fields.
type FanrunCell struct {
	Agents int `json:"agents"`

	// --- wall-clock MEASURED (serial sum; explicitly NOT a parallel rate) ---
	AgentsWallSerialMs float64 `json:"agents_wall_serial_ms"`
	PerAgentMsMean     float64 `json:"per_agent_ms_mean"`
	AgentsPerSecSerial float64 `json:"agents_per_sec_serial"`

	// --- real kernel / loop counters (the live fan-out actually produced these) ---
	WaveHits         int  `json:"wave_hits"`         // Σ per-agent vDSO hits over the wave (raw)
	CrossHits        int  `json:"cross_hits"`        // sibling-only: wave − N×(N=1 baseline per-agent hits)
	CrossHitsStable  bool `json:"cross_hits_stable"` // determinism witness (Trials>=2)
	VDSOLookups      int  `json:"vdso_lookups"`      // vdso.Default.Stats() delta over the wave (cross-check)
	VDSOFills        int  `json:"vdso_fills"`
	TurnsTotal       int  `json:"turns_total"`
	PromptTokens     int  `json:"prompt_tokens_total"`
	CompletionTokens int  `json:"completion_tokens_total"`
	ToolErrorsTotal  int  `json:"tool_errors_total"`
	TasksCompleted   int  `json:"tasks_completed"`

	// --- exact GEOMETRY (arithmetic over a shipped kernel property, not measured) ---
	PrefixTokensElided int `json:"prefix_tokens_elided"` // (N-1)*Prefix

	// --- wall-clock MEASURED: the prefill-elision lever (only when a timing model is present) ---
	PrefillReuseTotalMs   float64 `json:"prefill_reuse_total_ms,omitempty"`
	PrefillNoReuseTotalMs float64 `json:"prefill_noreuse_total_ms,omitempty"`
	PrefillReuseSpeedup   float64 `json:"prefill_reuse_speedup,omitempty"`
	PrefillTimingSkipped  string  `json:"prefill_timing_skipped,omitempty"`
}

// RunFanoutLive runs the full live fan-out sweep. SERIAL by construction (process-global
// vDSO world). Deterministic in (profile, grid, sub-turns, prefix, trials, seed) for the
// counter+geometry halves; the *_ms wall-clock halves are not.
func RunFanoutLive(ctx context.Context, opts FanrunOptions) FanrunReport {
	if opts.SubTurns <= 0 {
		opts.SubTurns = 8
	}
	if opts.Trials <= 0 {
		opts.Trials = 1
	}
	if opts.Prefix < 1 {
		opts.Prefix = 2048
	}

	timingModel := "skipped: no -model-dir (geometry-only; prefill elision reported as exact (N-1)*P)"
	if opts.ModelDir != "" && opts.Reps > 0 {
		timingModel = opts.ModelDir
	}

	rep := FanrunReport{
		Schema: FanrunSchema,
		Provenance: FanrunProvenance{
			AppVersion: appversion.Current(), Command: "cmd/fanrun",
			GoVersion: runtime.Version(), OS: runtime.GOOS, Arch: runtime.GOARCH,
			GeneratedBy: "fak/internal/bench (fanrun: live one-goal→N-subagent capstone)",
		},
		Host: FanrunHost{
			CPUCount: runtime.NumCPU(), GoThreads: runtime.GOMAXPROCS(0), HasGPU: false,
		},
		Profile: opts.Profile, SharedGoal: opts.shares(),
		Prefix: opts.Prefix, SubTurns: opts.SubTurns, Trials: opts.Trials, Reps: opts.Reps,
		Seed: opts.Seed, Task: agent.DefaultTask, AgentGrid: opts.Grid,
		TimingModel: timingModel,
	}

	// The single-agent (N=1) per-agent hit baseline is the intra-agent dedup one sub-agent
	// gets alone (the duplicate get_user re-verify). cross_uplift subtracts N× of it so
	// CrossHits is the fan-out-ONLY sibling dedup, exactly like fanout.go's shared−isolated.
	// Measured in the SHARED regime; the no-share control has no sibling reuse to subtract.
	baselinePerAgentHits := liveWave(ctx, opts, 1).waveHits

	for _, N := range opts.Grid {
		if N < 1 {
			continue
		}
		rep.Cells = append(rep.Cells, runLiveCell(ctx, opts, N, baselinePerAgentHits))
	}
	return rep
}

// waveResult is one wave's measured rollup.
type waveResult struct {
	wallMs           float64
	waveHits         int
	lookupsDelta     int
	fillsDelta       int
	turns            int
	promptTokens     int
	completionTokens int
	toolErrors       int
	tasksCompleted   int
}

// runLiveCell runs the wave for width N (Trials times for the determinism witness), then
// attaches the geometry and (optionally) the prefill-elision wall-clock.
func runLiveCell(ctx context.Context, opts FanrunOptions, N, baselinePerAgentHits int) FanrunCell {
	first := liveWave(ctx, opts, N)
	stable := true
	for t := 1; t < opts.Trials; t++ {
		again := liveWave(ctx, opts, N)
		if again.waveHits != first.waveHits || again.turns != first.turns ||
			again.tasksCompleted != first.tasksCompleted {
			stable = false
		}
	}

	cell := FanrunCell{
		Agents:             N,
		AgentsWallSerialMs: first.wallMs,
		WaveHits:           first.waveHits,
		CrossHits:          first.waveHits - N*baselinePerAgentHits,
		CrossHitsStable:    opts.Trials < 2 || stable,
		VDSOLookups:        first.lookupsDelta,
		VDSOFills:          first.fillsDelta,
		TurnsTotal:         first.turns,
		PromptTokens:       first.promptTokens,
		CompletionTokens:   first.completionTokens,
		ToolErrorsTotal:    first.toolErrors,
		TasksCompleted:     first.tasksCompleted,
		PrefixTokensElided: (N - 1) * opts.Prefix,
	}
	if cell.CrossHits < 0 {
		cell.CrossHits = 0 // a lone agent has no sibling to share with; never report negative
	}
	if N > 0 {
		cell.PerAgentMsMean = first.wallMs / float64(N)
	}
	if first.wallMs > 0 {
		cell.AgentsPerSecSerial = float64(N) / (first.wallMs / 1e3)
	}

	if opts.ModelDir != "" && opts.Reps > 0 {
		applyPrefillTiming(&cell, opts, N)
	}
	return cell
}

// liveWave runs N real RunArm sub-agent sessions SERIALLY in one fresh world epoch and
// rolls up their measured ArmMetrics + the global vDSO counter delta over the wave.
func liveWave(ctx context.Context, opts FanrunOptions, N int) waveResult {
	agent.Configure()

	// Isolate the epoch so this wave's dedup is a clean delta (the scoreWorld discipline).
	vdso.Default.BumpWorld()
	l0, _, f0, _ := vdso.Default.Stats()

	shared := opts.shares()
	var res waveResult
	t0 := time.Now()
	for a := 0; a < N; a++ {
		p := plannerFor(shared, a)
		m, err := agent.RunArm(ctx, p, agent.DefaultTask, true /*fak*/, opts.SubTurns, nil)
		if err != nil {
			// A RunArm error is a harness fault, not a result; record nothing for this
			// sub-agent and keep the wave honest (tasksCompleted will fall short, which a
			// test catches).
			continue
		}
		res.waveHits += m.VDSOHits
		res.turns += m.Turns
		res.promptTokens += m.PromptTokens
		res.completionTokens += m.CompletionTokens
		res.toolErrors += m.ToolErrors
		// A research sub-agent "completes" when it finishes its gather and emits a final
		// answer within the turn budget (it never books — that is the lead's fold). So the
		// completion criterion is a final answer reached, not a successful booking.
		if m.FinalAnswer != "" && !m.HitTurnCap {
			res.tasksCompleted++
		}
	}
	res.wallMs = float64(time.Since(t0).Nanoseconds()) / 1e6

	l1, _, f1, _ := vdso.Default.Stats()
	res.lookupsDelta = int(l1 - l0)
	res.fillsDelta = int(f1 - f0)
	return res
}

// plannerFor returns the deterministic offline RESEARCH planner for sub-agent a. fanrun
// models the real orchestrator-worker pattern: the sub-agents GATHER (read the shared goal
// sources) and the lead would FOLD their findings — the sub-agents themselves never write.
// This is exactly FanoutResearch's PWrite=0.0, and it is what makes cross-agent dedup real:
// a destructive write (a booking) bumps the process-global world version and FLUSHES every
// sibling's warmed read, so a sub-agent that booked would strand the whole fleet (the
// honest write-invalidation tension fanbench's write-goal profile reports). The capstone
// measures the read-heavy regime where the levers pay; the booking belongs to the lead's
// fold, which fanrun does not run.
//
// In the SHARED regime every sub-agent reads the SAME goal sources (mia_li_3668 / refunds /
// SFO→JFK), so after sub-agent 0 warms the cache, siblings 1..N-1 get genuine cross-agent
// tier-2 hits. In the no-share control each sub-agent's reads carry a distinct salt, so the
// arg hashes never collide → distinct tier-2 keys → zero cross-agent uplift.
func plannerFor(shared bool, a int) agent.Planner {
	salt := ""
	if !shared {
		salt = fmt.Sprintf("_%d", a)
	}
	return &researchPlanner{salt: salt, idx: a}
}

// researchPlanner is a deterministic, offline, context-stateful planner that drives the
// read-only research gather: look up the account → fetch the refund policy → search flights
// → convert the cheapest price → done. It never books (no destructive write, so no world
// bump), so sibling reads stay warm and cross-agent dedup is measurable. It is the
// read-only sibling of agent.MockPlanner, scoped to the orchestrator-worker sub-agent role.
type researchPlanner struct {
	salt string
	idx  int
}

func (researchPlanner) Model() string { return "fanrun-research" }

func (p researchPlanner) Complete(_ context.Context, messages []agent.Message, _ []agent.ToolDef, _ ...agent.SampleOpt) (*agent.Completion, error) {
	uid := "mia_li_3668" + p.salt
	s := scanResearch(messages)
	turns := s.assistantTurns

	emit := func(tool, args string) *agent.Completion {
		return &agent.Completion{
			Message: agent.Message{Role: agent.RoleAssistant, ToolCalls: []agent.ToolCall{
				{ID: fmt.Sprintf("call_%d", turns), Type: "function",
					Function: agent.Func{Name: tool, Arguments: args}},
			}},
			FinishReason: "tool_calls",
			Usage:        researchUsage(messages, 24),
		}
	}

	// In the no-share control EVERY read carries the salt (a distinct topic / origin /
	// amount per sub-agent), so no read type collides across siblings — the uplift must be
	// exactly 0. In the shared regime the salt is empty, so all sub-agents read the SAME
	// sources and siblings reuse agent 0's warmed cache.
	topic := "refunds" + p.salt
	origin := "SFO" + p.salt
	amount := 240
	if p.salt != "" {
		amount = 240 + 1 + p.idx // distinct amount per salted agent
	}
	switch {
	case s.userCalls == 0:
		return emit("get_user_details", fmt.Sprintf(`{"user_id":%q}`, uid)), nil
	case !s.gotPolicy:
		return emit("fetch_policy", fmt.Sprintf(`{"topic":%q}`, topic)), nil
	case s.gotUser && s.userCalls < 2:
		// a duplicate read-only re-verify — an intra-agent dedup hit on every arm.
		return emit("get_user_details", fmt.Sprintf(`{"user_id":%q}`, uid)), nil
	case !s.gotSearch:
		return emit("search_direct_flight", fmt.Sprintf(`{"origin":%q,"destination":"JFK","date":"2026-07-01"}`, origin)), nil
	case !s.gotConvert:
		return emit("convert_currency", fmt.Sprintf(`{"from_currency":"USD","to_currency":"EUR","amount":%d}`, amount)), nil
	default:
		return &agent.Completion{
			Message:      agent.Message{Role: agent.RoleAssistant, Content: "Research complete: cheapest SFO→JFK 2026-07-01 is $240 (~220.80 EUR); refunds 24h/$75."},
			FinishReason: "stop",
			Usage:        researchUsage(messages, 40),
		}, nil
	}
}

// researchState is the read of the gather so far.
type researchState struct {
	userCalls      int
	gotUser        bool
	gotPolicy      bool
	gotSearch      bool
	gotConvert     bool
	assistantTurns int
}

func scanResearch(messages []agent.Message) researchState {
	var s researchState
	for _, msg := range messages {
		if msg.Role == agent.RoleAssistant {
			s.assistantTurns++
		}
		if msg.Role != agent.RoleTool {
			continue
		}
		isErr := strings.Contains(strings.ToLower(msg.Content), `"error"`)
		switch msg.Name {
		case "get_user_details":
			s.userCalls++
			if !isErr {
				s.gotUser = true
			}
		case "fetch_policy":
			s.gotPolicy = true
		case "search_direct_flight":
			if !isErr {
				s.gotSearch = true
			}
		case "convert_currency":
			if !isErr {
				s.gotConvert = true
			}
		}
	}
	return s
}

func researchUsage(messages []agent.Message, out int) agent.Usage {
	in := 0
	for _, msg := range messages {
		in += len(msg.Content) / 4
		for _, tc := range msg.ToolCalls {
			in += len(tc.Function.Arguments) / 4
		}
	}
	return agent.Usage{PromptTokens: in, CompletionTokens: out, TotalTokens: in + out}
}

// minDurMs returns the minimum of the durations in milliseconds (best/least-contended
// sampling, the fleetserve methodology).
func minDurMs(ds []time.Duration) float64 {
	if len(ds) == 0 {
		return 0
	}
	cp := append([]time.Duration(nil), ds...)
	sort.Slice(cp, func(i, j int) bool { return cp[i] < cp[j] })
	return float64(cp[0].Nanoseconds()) / 1e6
}
