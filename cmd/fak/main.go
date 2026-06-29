// Command fak is the Fused Agent Kernel: one statically-linked Go binary that
// runs an agentic tool loop where every tool call passes through one in-process
// policy and quarantine boundary (adjudicate -> vDSO -> pre-flight/grammar ->
// dispatch -> context-MMU admit). Verbs:
//
//	fak run        -  replay a trace (or a single call) through the kernel
//	fak preflight  -  run only the pre-flight + grammar rungs over a call
//	fak bench      -  A/B ablate the vDSO over a frozen trace, emit report.json
//	fak policy     -  dump / validate the deployable capability-floor manifest
//	fak hook       -  spawned-hook mode: decide one call from stdin (the baseline)
//
// The single blank import of internal/registrations is what wires every leaf
// subsystem into the frozen ABI before the kernel boots.
package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/a2achan"
	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/adjudicator"
	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/bench"
	"github.com/anthony-chaudhary/fak/internal/gateway"
	"github.com/anthony-chaudhary/fak/internal/ggufload"
	"github.com/anthony-chaudhary/fak/internal/grammar"
	"github.com/anthony-chaudhary/fak/internal/guard"
	"github.com/anthony-chaudhary/fak/internal/ifc"
	"github.com/anthony-chaudhary/fak/internal/kernel"
	"github.com/anthony-chaudhary/fak/internal/metrics"
	"github.com/anthony-chaudhary/fak/internal/newmodel"
	"github.com/anthony-chaudhary/fak/internal/policy"
	"github.com/anthony-chaudhary/fak/internal/ratelimit"
	"github.com/anthony-chaudhary/fak/internal/session"
	"github.com/anthony-chaudhary/fak/internal/sessionreset"
	"github.com/anthony-chaudhary/fak/internal/tokenizer"
	"github.com/anthony-chaudhary/fak/internal/toollint"
	"github.com/anthony-chaudhary/fak/internal/turnbench"

	_ "github.com/anthony-chaudhary/fak/internal/registrations"
)

func main() {
	if len(os.Args) < 2 {
		usage()
		os.Exit(2)
	}
	switch os.Args[1] {
	case "run":
		cmdRun(os.Args[2:])
	case "replay":
		// Explicit, unambiguous spelling of the trace-replay path (`fak run --trace`).
		cmdRunTrace(os.Args[2:])
	case "commit":
		cmdCommit(os.Args[2:])
	case "sweep":
		cmdSweep(os.Args[2:])
	case "affected":
		cmdAffected(os.Args[2:])
	case "preflight":
		cmdPreflight(os.Args[2:])
	case "attest":
		cmdAttest(os.Args[2:])
	case "bench":
		cmdBench(os.Args[2:])
	case "benchmarks":
		cmdBenchmarks(os.Args[2:])
	case "ablate":
		cmdAblate(os.Args[2:])
	case "turntax":
		cmdTurnTax(os.Args[2:])
	case "agent":
		cmdAgent(os.Args[2:])
	case "recall":
		cmdRecall(os.Args[2:])
	case "session":
		cmdSession(os.Args[2:])
	case "resume":
		cmdResume(os.Args[2:])
	case "dispatch":
		cmdDispatch(os.Args[2:])
	case "ps":
		cmdPS(os.Args[2:])
	case "top":
		cmdTop(os.Args[2:])
	case "signal":
		cmdSignal(os.Args[2:])
	case "task":
		cmdTask(os.Args[2:])
	case "c":
		cmdTUI(append([]string{"agent"}, os.Args[2:]...))
	case "console":
		cmdTUI(os.Args[2:])
	case "tui":
		cmdTUI(os.Args[2:])
	case "chatrelay":
		cmdChatRelay(os.Args[2:])
	case "claude-mac-fak":
		cmdClaudeMacFak(os.Args[2:])
	case "loop":
		cmdLoop(os.Args[2:])
	case "cron":
		cmdCron(os.Args[2:])
	case "snapshot":
		cmdSnapshot(os.Args[2:])
	case "traj":
		cmdTraj(os.Args[2:])
	case "dream":
		cmdDream(os.Args[2:])
	case "memory":
		cmdMemory(os.Args[2:])
	case "debug":
		cmdDebug(os.Args[2:])
	case "policy":
		cmdPolicy(os.Args[2:])
	case "egress":
		cmdEgress(os.Args[2:])
	case "lint":
		cmdLint(os.Args[2:])
	case "codelint":
		cmdCodelint(os.Args[2:])
	case "answer-shape":
		cmdAnswerShape(os.Args[2:])
	case "claim-check":
		cmdClaimCheck(os.Args[2:])
	case "doctor":
		cmdDoctor(os.Args[2:])
	case "index":
		cmdIndex(os.Args[2:])
	case "tree-doctor":
		cmdTreeDoctor(os.Args[2:])
	case "self-update":
		cmdSelfUpdate(os.Args[2:])
	case "slack":
		cmdSlack(os.Args[2:])
	case "release-staleness":
		cmdReleaseStaleness(os.Args[2:])
	case "serve":
		cmdServe(os.Args[2:])
	case "serve-wiring":
		cmdServeWiring(os.Args[2:])
	case "guard":
		cmdGuard(os.Args[2:])
	case "guard-precompact":
		// Hidden: Claude Code PreCompact hook actuator installed by `fak guard`.
		cmdGuardPreCompact(os.Args[2:])
	case "guard-stophook":
		// Hidden: Claude Code Stop hook actuator installed by `fak guard`. Reads the
		// gateway's deny-all consecutive gauge and, in enforce mode, blocks an unchosen
		// end_turn so the agent auto-continues past a fully-refused turn (bounded).
		cmdGuardStopHook(os.Args[2:])
	case guard.TrampolineVerb:
		// Hidden: the Landlock hook-floor re-exec trampoline (Linux). `fak guard
		// --landlock-hooks` re-execs itself into this verb, which applies the
		// read-only-.git/hooks ruleset to itself and then execs the real agent. Not
		// listed in usage() — an internal implementation detail of the spawn seam.
		if err := guard.LandlockTrampoline(os.Args[2:]); err != nil {
			fmt.Fprintf(os.Stderr, "fak: %v\n", err)
			os.Exit(127)
		}
	case "audit":
		cmdAudit(os.Args[2:])
	case "headroom":
		cmdHeadroom(os.Args[2:])
	case "vcache":
		cmdVCache(os.Args[2:])
	case "hook":
		cmdHook()
	case "hooks":
		cmdHooks(os.Args[2:])
	case "hygiene":
		cmdHygiene(os.Args[2:])
	case "rungstats":
		cmdRungStats(os.Args[2:])
	case "swebench":
		cmdSwebench(os.Args[2:])
	case "webbench":
		cmdWebbench(os.Args[2:])
	case "ailuminate":
		cmdAILuminate(os.Args[2:])
	case "model":
		cmdModel(os.Args[2:])
	case "new-model":
		cmdNewModel(os.Args[2:])
	case "pull":
		// Top-level alias for `fak model pull`: the Ollama-style run-by-name download.
		cmdModelPull(os.Args[2:])
	case "ls":
		// Top-level alias for `fak model ls`: list known model aliases + cache status.
		cmdModelLs(os.Args[2:])
	case "route":
		cmdRoute(os.Args[2:])
	case "routebench":
		cmdRoutebench(os.Args[2:])
	case "accounts":
		cmdAccounts(os.Args[2:])
	case "garden":
		cmdGarden(os.Args[2:])
	case "cadence":
		cmdCadence(os.Args[2:])
	case "rollup":
		cmdRollup(os.Args[2:])
	case "nightrun":
		cmdNightrun(os.Args[2:])
	case "sessions":
		cmdSessions(os.Args[2:])
	case "loop-index-scorecard":
		cmdLoopIndexScorecard(os.Args[2:])
	case "loop-map":
		cmdLoopMap(os.Args[2:])
	case "experiments":
		cmdExperiments(os.Args[2:])
	case "coverage-matrix":
		cmdCoverageMatrix(os.Args[2:])
	case "support-maturity-scorecard":
		cmdSupportMaturityScorecard(os.Args[2:])
	case "dojo":
		cmdDojo(os.Args[2:])
	case "dojo-rsi":
		cmdDojoRSI(os.Args[2:])
	case "guard-verdict-rsi":
		cmdGuardVerdictRSI(os.Args[2:])
	case "guard-rsi-scorecard":
		cmdGuardRSIScorecard(os.Args[2:])
	case "dogfood-score":
		cmdDogfoodScore(os.Args[2:])
	case "concept-usage-score":
		cmdConceptUsageScore(os.Args[2:])
	case "loop-score":
		cmdLoopScore(os.Args[2:])
	case "token-defaults-scorecard":
		cmdTokenDefaultsScorecard(os.Args[2:])
	case "skill-effectiveness-scorecard":
		cmdSkillEffectivenessScorecard(os.Args[2:])
	case "conflation-scorecard":
		cmdConflationScorecard(os.Args[2:])
	case "scoreboard":
		cmdScoreboard(os.Args[2:])
	case "steering":
		cmdSteering(os.Args[2:])
	case "blockers":
		cmdBlockers(os.Args[2:])
	case "product":
		cmdProduct(os.Args[2:])
	case "grafana":
		cmdGrafana(os.Args[2:])
	case "cachevalue":
		cmdCachevalue(os.Args[2:])
	case "marketing":
		cmdMarketing(os.Args[2:])
	case "nodeusage":
		cmdNodeUsage(os.Args[2:])
	case "callavoid":
		cmdCallavoid(os.Args[2:])
	case "savings-vector":
		cmdSavingsVector(os.Args[2:])
	case "horizon-recovery":
		cmdHorizonRecovery(os.Args[2:])
	case "dogfood-issues":
		cmdDogfoodIssues(os.Args[2:])
	case "learning-debt-dispatch":
		cmdLearningDebtDispatch(os.Args[2:])
	case "stopfailure":
		cmdStopFailure(os.Args[2:])
	case "cluster":
		cmdCluster(os.Args[2:])
	case "leaseref":
		cmdLeaseref(os.Args[2:])
	case "node":
		cmdNode(os.Args[2:])
	case "lab":
		cmdLab(os.Args[2:])
	case "version", "-v", "--version":
		cmdVersion(os.Stdout)
	case "-h", "--help", "help":
		usage()
	default:
		fmt.Fprintf(os.Stderr, "fak: unknown verb %q\n", os.Args[1])
		usage()
		os.Exit(2)
	}
}

func ctx() context.Context { return context.Background() }

// fak run  -  two modes, split on argv[0] BEFORE any flag parse so the two
// parsers never collide:
//
//   - argv[0] is a non-flag (a model alias, an hf:// URI, or a .gguf path) ->
//     CHAT mode: load that model into the in-kernel engine and run a one-shot
//     completion (or an interactive REPL with no prompt). The Ollama `run` analog.
//   - argv[0] starts with '-' (or is empty) -> the existing TRACE-replay mode,
//     parsed exactly as before (--trace still required). Every documented
//     `fak run --trace ...` caller is flag-first, so it is unaffected.
//
// `fak replay` is an explicit, unambiguous alias for the trace path.
func cmdRun(argv []string) {
	if len(argv) > 0 && !strings.HasPrefix(argv[0], "-") {
		runChatModel(argv)
		return
	}
	cmdRunTrace(argv)
}

// cmdRunTrace replays a trace through the kernel (the original `fak run`).
func cmdRunTrace(argv []string) {
	fs := flag.NewFlagSet("run", flag.ExitOnError)
	trace := fs.String("trace", "", "path to a trace JSON file")
	engineID := fs.String("engine", "inkernel", "engine id (inkernel: the fused in-kernel model; mock; cassette)")
	vdso := fs.Bool("vdso", true, "enable the vDSO fast path")
	policyPath := fs.String("policy", "", "load the capability floor from a manifest (default: the built-in adjudicator floor — the tau2 airline-demo tools, NOT the `fak guard` coding floor; see `fak policy --dump`)")
	_ = fs.Parse(argv)

	if *trace == "" {
		fmt.Fprintln(os.Stderr, "fak run: --trace is required")
		os.Exit(2)
	}
	applyPolicy(*policyPath)
	t, err := bench.LoadTrace(*trace)
	must(err)
	k := kernel.New(*engineID)
	k.SetVDSO(*vdso)
	res := abi.ActiveResolver()
	for i, c := range t.Calls {
		args := []byte(c.Args)
		if len(args) == 0 {
			args = []byte("{}")
		}
		ref, err := res.Put(ctx(), args)
		must(err)
		tc := &abi.ToolCall{Tool: c.Tool, Args: ref, Meta: c.Meta}
		r, v := k.Syscall(ctx(), tc)
		fmt.Printf("[%2d] %-28s verdict=%-9s by=%-9s status=%v\n",
			i, c.Tool, verdictName(v.Kind), v.By, statusName(r.Status))
	}
	cc := k.Counters()
	fmt.Printf("\nsummary: submits=%d vdso_hits=%d engine_calls=%d denies=%d transforms=%d quarantines=%d\n",
		cc.Submits, cc.VDSOHits, cc.EngineCalls, cc.Denies, cc.Transforms, cc.Quarantines)
}

// fak preflight  -  run only the pre-flight/grammar rungs over one call.
func cmdPreflight(argv []string) {
	fs := flag.NewFlagSet("preflight", flag.ExitOnError)
	tool := fs.String("tool", "", "tool name")
	args := fs.String("args", "{}", "tool args as JSON")
	policyPath := fs.String("policy", "", "load the capability floor from a manifest (default: the built-in adjudicator floor — the tau2 airline-demo tools, NOT the `fak guard` coding floor; see `fak policy --dump`)")
	grammarSchema := fs.String("grammar-schema", "", "load a JSON Schema grammar for --tool before adjudication (demo/debug witness)")
	showDispatchedArgs := fs.Bool("show-dispatched-args", false, "print post-transform args that would be dispatched (may include raw arg values; demo/debug only)")
	explain := fs.Bool("explain", false, "print the full decision trace: every rung folded, what each returned, which won, and why")
	asJSON := fs.Bool("json", false, "emit the decision trace as JSON (safe to log: args digest only, never raw args)")
	_ = fs.Parse(argv)
	if *tool == "" {
		fmt.Fprintln(os.Stderr, "fak preflight: --tool is required")
		os.Exit(2)
	}
	if *asJSON && *showDispatchedArgs {
		fmt.Fprintln(os.Stderr, "fak preflight: --json cannot be combined with --show-dispatched-args")
		os.Exit(2)
	}
	must(loadPreflightGrammar(*tool, *grammarSchema))
	applyPolicy(*policyPath)
	res := abi.ActiveResolver()
	ref, err := res.Put(ctx(), []byte(*args))
	must(err)
	tc := &abi.ToolCall{Tool: *tool, Args: ref}
	// --explain/--json fold the SAME chain to the SAME verdict but additionally
	// surface the per-rung Decision trace (the eight rungs preflight actually folds
	// are invisible in the default one-liner). Default output is unchanged.
	if *explain || *asJSON {
		v, d := kernel.FoldExplain(ctx(), abi.AdjudicatorsFor(tc), tc)
		if *asJSON {
			fmt.Println(d.JSON())
		} else {
			fmt.Print(d.Text())
			if *showDispatchedArgs {
				printDispatchedArgs(ctx(), v)
			}
		}
		return
	}
	v := kernel.Fold(ctx(), abi.AdjudicatorsFor(tc), tc)
	fmt.Printf("verdict=%s reason=%s by=%s\n", verdictName(v.Kind), abi.ReasonName(v.Reason), v.By)
	if *showDispatchedArgs {
		printDispatchedArgs(ctx(), v)
	}
}

func loadPreflightGrammar(tool, schemaPath string) error {
	if strings.TrimSpace(schemaPath) == "" {
		return nil
	}
	b, err := os.ReadFile(schemaPath)
	if err != nil {
		return fmt.Errorf("load grammar schema: %w", err)
	}
	if err := grammar.Default.LoadFromJSONSchema(tool, b); err != nil {
		return fmt.Errorf("load grammar schema: %w", err)
	}
	return nil
}

func printDispatchedArgs(ctx context.Context, v abi.Verdict) {
	line, ok, err := dispatchedArgsLine(ctx, v)
	must(err)
	if ok {
		fmt.Println(line)
	}
}

func dispatchedArgsLine(ctx context.Context, v abi.Verdict) (string, bool, error) {
	if v.Kind != abi.VerdictTransform {
		return "", false, nil
	}
	tp, ok := v.Payload.(abi.TransformPayload)
	if !ok {
		return "", false, fmt.Errorf("transform payload type %T, want abi.TransformPayload", v.Payload)
	}
	b, err := abi.ActiveResolver().Resolve(ctx, tp.NewArgs)
	if err != nil {
		return "", false, fmt.Errorf("resolve transformed args: %w", err)
	}
	var compact bytes.Buffer
	if err := json.Compact(&compact, b); err == nil {
		b = compact.Bytes()
	}
	return "dispatched_args=" + string(b), true, nil
}

// fak bench  -  A/B ablate the vDSO over a frozen trace.
func cmdBench(argv []string) {
	// `fak bench post|request` are the outbound bench-CHANNEL surface (post rollups /
	// run-requests to Slack); anything else falls through to the benchmark RUNNER below.
	if len(argv) > 0 {
		switch argv[0] {
		case "post":
			os.Exit(runBenchPost(os.Stdout, os.Stderr, argv[1:]))
		case "request":
			os.Exit(runBenchRequest(os.Stdout, os.Stderr, argv[1:]))
		}
	}
	fs := flag.NewFlagSet("bench", flag.ExitOnError)
	suite := fs.String("suite", "tau2-smoke", "trace suite name (under testdata/tau2)")
	out := fs.String("out", "report.json", "report output path")
	tracePath := fs.String("trace", "", "explicit trace path (overrides --suite)")
	baselineN := fs.Int("baseline-n", 30, "spawned-hook baseline samples")
	noBaseline := fs.Bool("no-baseline", false, "skip the spawned baseline (RED gate)")
	_ = fs.Parse(argv)

	path := *tracePath
	if path == "" {
		path = resolveSuite(traceDir(), *suite)
	}
	t, err := bench.LoadTrace(path)
	must(err)

	opt := bench.Options{EngineID: "mock", EngineModel: "mock-offline", BaselineN: *baselineN}
	if !*noBaseline {
		if bin, err := os.Executable(); err == nil {
			opt.BinPath = bin
		}
	}
	rep, err := bench.Run(ctx(), t, opt)
	must(err)

	must(os.WriteFile(*out, rep.JSON(), 0o644))
	// also dump the standalone baseline witness (unit 23)
	if rep.Baseline.Calls > 0 {
		bj, _ := json.MarshalIndent(map[string]any{
			"source": rep.Baseline.Source, "p50_ns": rep.Baseline.P50Ns,
			"p50_ms": float64(rep.Baseline.P50Ns) / 1e6, "calls": rep.Baseline.Calls,
		}, "", "  ")
		_ = os.WriteFile(filepath.Join(filepath.Dir(*out), "baseline.json"), bj, 0o644)
	}

	printReport(rep, *out)
}

// fak turntax  -  the TURN-TAX A/B. Replays a class-labeled trace through the real
// kernel and prices the extra error-code MODEL TURNS the SOTA baseline fires
// (malformed args, duplicate read, poison) that fak's 1-shot path deletes  -
// keeping the deterministic safety floor on its own axis.
func cmdTurnTax(argv []string) {
	fs := flag.NewFlagSet("turntax", flag.ExitOnError)
	suite := fs.String("suite", "turntax-airline", "trace suite under testdata/turntax")
	out := fs.String("out", "turntax-report.json", "report output path")
	tracePath := fs.String("trace", "", "explicit trace path (overrides --suite)")
	breakeven := fs.Bool("breakeven", false, "emit the hit-rate -> turns-saved -> amortization curve (turntax-breakeven.json) instead of a single-trace report")
	trials := fs.Int("trials", 200, "trials per hit-rate point (--breakeven only)")
	seed := fs.Int64("seed", 0x7A8BC0DE, "RNG seed for the break-even sweep (--breakeven only)")
	cm := turnbench.DefaultCostModel()
	promptTok := fs.Int("prompt-tokens", cm.PromptTokensPerTurn, "model prompt tokens per turn (cost knob)")
	complTok := fs.Int("completion-tokens", cm.CompletionTokensPerTurn, "model completion tokens per turn (cost knob)")
	latencyMs := fs.Float64("turn-latency-ms", cm.ModelTurnLatencyMs, "model round-trip latency per turn, ms (cost knob)")
	_ = fs.Parse(argv)

	cm.PromptTokensPerTurn = *promptTok
	cm.CompletionTokensPerTurn = *complTok
	cm.ModelTurnLatencyMs = *latencyMs

	if *breakeven {
		bePath := *out
		if bePath == "turntax-report.json" { // user kept the default; pick the break-even default
			bePath = "turntax-breakeven.json"
		}
		rep := turnbench.RunBreakEvenSweep(ctx(), turnbench.BaseTrace(), nil, nil, cm, *trials, *seed)
		must(os.WriteFile(bePath, rep.JSON(), 0o644))
		printBreakEven(os.Stdout, &rep)
		fmt.Printf("\nbreak-even curve written    : %s\n", bePath)
		return
	}

	path := *tracePath
	if path == "" {
		path = resolveSuite(turnTaxDir(), *suite)
	}
	t, err := turnbench.LoadTrace(path)
	must(err)

	rep, err := turnbench.Run(ctx(), t, cm)
	must(err)

	must(os.WriteFile(*out, rep.JSON(), 0o644))
	turnbench.PrintReport(os.Stdout, rep)
	fmt.Printf("\nreport written              : %s\n", *out)
}

// printBreakEven renders the hit-rate curve as an operator-readable table: the
// expected turns saved per session at each addressable rate, the priced net, and
// the §3.1 self-host amortization (the real ~0.7% row is the honest answer to "is
// 9 cherry-picked").
func printBreakEven(w io.Writer, r *turnbench.BreakEvenReport) {
	fmt.Fprintf(w, "== fak turntax break-even: %s  (base %d calls, %d trials/point, hash %s) ==\n",
		r.SliceID, r.BaseCalls, r.Trials, r.BaseHash)
	fmt.Fprintf(w, "real-world addressable hit-rate (TURN-TAX §3.1, tau2-airline): %.1f%%\n\n", r.RealWorldHitRate*100)
	fmt.Fprintf(w, "%6s  %11s  %4s  %4s  %9s  %9s  %8s  %16s\n",
		"h", "mean_turns", "p50", "p90", "tok/sess", "$/sess", "lat(s)", "self-host_sess")
	for _, p := range r.Points {
		self := "n/a"
		for _, rg := range p.Regimes {
			if rg.Name == "self-host-fork" {
				if rg.SessionsToBreakEven == turnbench.NeverAmortizes {
					self = "never"
				} else {
					self = fmt.Sprintf("%.0f", rg.SessionsToBreakEven)
				}
			}
		}
		mark := ""
		if p.HitRate == r.RealWorldHitRate {
			mark = "  <- real-world rate"
		}
		fmt.Fprintf(w, "%6.3f  %11.4f  %4d  %4d  %9.0f  %9.5f  %8.2f  %16s%s\n",
			p.HitRate, p.MeanTurnsSaved, p.TurnsSaved.P50, p.TurnsSaved.P90,
			p.TokensSavedMean, p.DollarsSavedMean, p.LatencySavedSecMean, self, mark)
	}
	fmt.Fprintf(w, "\nThe turn-saving is small at the real ~%.1f%% rate (the safety floor is the reason to run the kernel there);\n", r.RealWorldHitRate*100)
	fmt.Fprintln(w, "it only becomes large in error/dup-rich regimes. The airline demo slice (9/14) is the far high end of this curve.")
}

// fak agent  -  the LIVE agentic loop. A real model (or the offline mock) drives a
// multi-turn tool-calling conversation TWICE over the same task: once with every
// tool call mediated by the in-process kernel (fak arm), once naive (the "now"
// baseline). It reports turns, tokens, in-syscall repairs, vDSO dedup hits,
// adjudicator denies, and MMU quarantines for each arm  -  the real turn-use-vs-now
// measurement the static bench could not produce.
func cmdAgent(argv []string) {
	fs := flag.NewFlagSet("agent", flag.ExitOnError)
	task := fs.String("task", agent.DefaultTask, "the user task the agent must complete")
	provider := fs.String("provider", "openai", "provider transcript wire: openai, anthropic, gemini, or xai")
	baseURL := fs.String("base-url", "", "provider base URL (OpenAI-compatible: .../v1; Gemini native: .../v1beta; Anthropic native: https://api.anthropic.com)")
	model := fs.String("model", "gemini-2.5-flash", "model id")
	apiKeyEnv := fs.String("api-key-env", "GEMINI_API_KEY", "env var holding the API key")
	offline := fs.Bool("offline", false, "use the deterministic mock planner (no network)")
	maxTurns := fs.Int("max-turns", 10, "max model turns per arm")
	out := fs.String("out", "agent-report.json", "report output path")
	logOut := fs.String("log", "", "optional path to write the per-call trace log")
	policyPath := fs.String("policy", "", "load the capability floor from a manifest (default: the built-in adjudicator floor — the tau2 airline-demo tools, NOT the `fak guard` coding floor; see `fak policy --dump`)")
	_ = fs.Parse(argv)
	applyPolicy(*policyPath)

	var planner agent.Planner
	if *offline || *baseURL == "" {
		if !*offline {
			fmt.Fprintln(os.Stderr, "fak agent: no --base-url given; using the offline mock planner (pass --base-url for a live run)")
		}
		planner = agent.NewMockPlanner(*model)
	} else {
		key := os.Getenv(*apiKeyEnv)
		if key == "" {
			// A local endpoint (e.g. the transformers shim) needs no key; a remote
			// one will return 401, which the planner surfaces clearly. Warn, proceed.
			fmt.Fprintf(os.Stderr, "fak agent: env %s is empty  -  proceeding with no auth header (fine for a local endpoint)\n", *apiKeyEnv)
		}
		p, err := agent.NewProviderHTTPPlanner(*provider, *baseURL, *model, key)
		must(err)
		planner = p
	}

	res, trace, err := agent.Run(ctx(), planner, *task, *maxTurns)
	must(err)

	must(os.WriteFile(*out, jsonIndent(res), 0o644))
	if *logOut != "" {
		_ = os.WriteFile(*logOut, agent.RenderTrace(trace), 0o644)
	}
	agent.PrintReport(os.Stdout, res, trace, *out)
}

func jsonIndent(v any) []byte {
	b, _ := json.MarshalIndent(v, "", "  ")
	return b
}

// applyPolicy loads a capability-floor manifest and swaps it into the registered
// adjudicator before the kernel runs. Empty path = keep the built-in
// DefaultPolicy. A bad manifest is fatal: a misconfigured floor must fail loudly
// at startup, never silently fall back to a more permissive default.
func applyPolicy(path string) {
	if path == "" {
		return
	}
	_, err := reloadPolicy(path)
	must(err)
	fmt.Fprintf(os.Stderr, "fak: loaded capability floor from %s\n", path)
}

func reloadPolicy(path string) (policy.Runtime, error) {
	if path == "" {
		return policy.Runtime{}, errors.New("policy reload requires --policy FILE")
	}
	rt, err := policy.LoadRuntime(path)
	if err != nil {
		return policy.Runtime{}, err
	}
	adjudicator.Default.SetPolicy(rt.Adjudicator)
	applyRuntime(rt)
	return rt, nil
}

func policyReloader(path string) gateway.PolicyReloadFunc {
	if path == "" {
		return nil
	}
	return func(context.Context) (gateway.PolicyReloadResponse, error) {
		rt, err := reloadPolicy(path)
		if err != nil {
			return gateway.PolicyReloadResponse{}, err
		}
		return gateway.PolicyReloadResponse{
			Reloaded: true,
			Source:   path,
			Summary:  policy.SummaryRuntime(rt),
		}, nil
	}
}

func resetTrace(_ context.Context, traceID string) error {
	traceID = strings.TrimSpace(traceID)
	if traceID == "" {
		return errors.New("trace_id is required")
	}
	ifc.Default.Reset(traceID)
	return nil
}

// observeTrace is the read-only complement of resetTrace (#411): it reports the
// live IFC taint high-water mark for one trace so the gateway can serve
// GET /v1/fak/trace/{trace_id} without importing IFC internals. An unseen trace
// reads "trusted"  -  the ledger's own clean default.
func observeTrace(_ context.Context, traceID string) (string, bool) {
	lvl := ifc.Default.Level(strings.TrimSpace(traceID))
	return taintLevelName(lvl), ifc.Dangerous(lvl)
}

// serveSessions is the process-local per-session DRIVE-state table shared by the
// gateway session routes (observe/control) and any in-process agent loop. It is the
// structural twin of ifc.Default: TraceID-keyed, bounded-LRU, live-mutable  -  widened
// from the single taint bit to a small drive struct (run-state/budget/priority/pace).
// Constructed once at process start; the gateway holds it by injected closure, never
// by import, so the gateway stays session-internals-blind the way it stays
// IFC-internals-blind for the trace routes.
var serveSessions = session.NewTable()

// observeSession is the read side of the /v1/fak/session control surface (#620): it
// returns one served session's current DRIVE state so an operator can read how hard
// a live session is running without reconstructing it from git + a process scan. An
// unseen trace reads its default  -  Running, unbounded budget  -  the table's own safe
// default, never a phantom Stopped.
func observeSession(_ context.Context, traceID string) gateway.SessionState {
	return toGatewaySessionState(serveSessions.Get(strings.TrimSpace(traceID)))
}

// listSessions is the multi-session read side of the /v1/fak/session control surface:
// it projects the WHOLE live drive table (Snapshot order  -  by priority, lower yields
// first) into the gateway wire DTO so an operator can see what every session is doing
// right now in one read, instead of reconstructing liveness from git + a process scan
// (docs/dispatch-loop.md). Snapshot already returns a fresh, sorted copy.
func listSessions(_ context.Context) []gateway.SessionState {
	snap := serveSessions.Snapshot()
	out := make([]gateway.SessionState, 0, len(snap))
	for _, s := range snap {
		out = append(out, toGatewaySessionState(s))
	}
	return out
}

// decideSession is the served request-boundary hook: it applies session.Table.Decide
// to the process-local DRIVE table so served model requests honor run-state,
// turn-budget, token-budget, and pace controls before the upstream request runs.
func decideSession(_ context.Context, traceID string) gateway.SessionVerdict {
	return toGatewaySessionVerdict(serveSessions.Decide(strings.TrimSpace(traceID)))
}

// debitSession records post-response token usage for a served request. The next
// Decide observes normal token-budget exhaustion at the following turn boundary;
// context-budget exhaustion drains immediately with a continuation id.
func debitSession(_ context.Context, traceID string, usage gateway.SessionUsage) gateway.SessionState {
	return toGatewaySessionState(serveSessions.DebitUsage(strings.TrimSpace(traceID), session.Usage{
		OutputTokens:  usage.CompletionTokens,
		ContextTokens: usage.ContextTokens,
	}))
}

// resetServedSessionOnBudget is the host-owned "human-like reset" hook the gateway
// calls after a context-budget drain. It distills the refused request transcript into
// a compact carryover seed, re-arms the continuation trace with a fresh context budget,
// and hands both back to the gateway so the live request can continue transparently.
func resetServedSessionOnBudget(freshContextTokens int) gateway.ResetOnBudgetFunc {
	if freshContextTokens <= 0 {
		return nil
	}
	return func(_ context.Context, traceID string, messages []agent.Message) (string, []agent.Message, bool) {
		traceID = strings.TrimSpace(traceID)
		st := serveSessions.Get(traceID)
		child := strings.TrimSpace(st.ContinuationID)
		if traceID == "" || child == "" {
			return "", nil, false
		}
		seed := sessionreset.BuildSeed(sessionreset.Input{
			Trace:          traceID,
			Messages:       resetMsgs(messages),
			FreshBudgetTok: freshContextTokens,
		})
		if strings.TrimSpace(seed.Recap) == "" {
			return "", nil, false
		}
		serveSessions.Recontinue(traceID, child, session.Budget{
			TurnsLeft:         session.Unbounded,
			TokensLeft:        session.Unbounded,
			ContextTokensLeft: freshContextTokens,
		})
		return child, []agent.Message{{Role: agent.RoleSystem, Content: seed.Recap}}, true
	}
}

func resetMsgs(messages []agent.Message) []sessionreset.Msg {
	out := make([]sessionreset.Msg, 0, len(messages))
	for _, m := range messages {
		out = append(out, sessionreset.Msg{Role: m.Role, Content: m.Content})
	}
	return out
}

// budgetWebhookObserver returns the session.BudgetObserver that wires the operator
// webhook (#743): it POSTs each pre-exhaustion warning and each exhaustion (reset-trigger)
// event to rawURL as JSON, so an external monitor is notified BEFORE a served session
// drains, not only after. It is fire-and-forget and fail-open  -  the POST runs on its own
// goroutine under a short timeout, and any transport error is logged to stderr but never
// blocks or fails the served turn that produced the event. An empty URL returns nil (the
// no-op seam: behavior is byte-identical to today).
func budgetWebhookObserver(rawURL string) session.BudgetObserver {
	rawURL = strings.TrimSpace(rawURL)
	if rawURL == "" {
		return nil
	}
	return func(ev session.BudgetEvent) {
		body, err := json.Marshal(ev)
		if err != nil {
			fmt.Fprintf(os.Stderr, "fak: budget webhook encode failed: %v\n", err)
			return
		}
		webhookPOST("budget webhook", rawURL, body, "application/json")
	}
}

// controlSession is the write side of /v1/fak/session (#620): it applies one control
// verb (run|budget|pace|priority) to a live session's DRIVE state. The (ok=false,
// err=nil) return is a terminal/CAS refusal (the route maps it to 409); a non-nil
// err is a malformed verb/body (the route maps it to 400). if_rev, when non-zero,
// is an optimistic-concurrency guard: the write is taken only if the session's
// current Rev matches (read-then-CompareAndSet; a lost race returns ok=false).
func controlSession(_ context.Context, traceID, verb string, req gateway.SessionControlRequest) (gateway.SessionState, bool, error) {
	traceID = strings.TrimSpace(traceID)
	st, ok, err := applySessionControl(serveSessions, traceID, verb, req)
	if err != nil {
		return gateway.SessionState{}, false, err
	}
	return toGatewaySessionState(st), ok, nil
}

// steerSession enqueues an operator steer onto the process-global a2achan bus so a RUNNING
// detached session can receive the input at its next turn boundary (#760). The serve
// process owns the bus, so the in-process Send happens HERE (the CLI is a separate process
// that POSTs HTTP; only the server can enqueue onto the bus it shares with the served loop).
//
// The body rides the a2achan floor: "operator" is a different principal from the target
// trace, so a Private (ScopeAgent) body would be refused — Shared (ScopeFleet) is the
// auditable cross-principal widening the operator must make, and it stays Tainted (operator
// input is untrusted, screened on ingress). A tainted/over-scoped/uncapped Send is refused
// by the SAME default-deny floor that gates a tool call; that deny-as-value becomes the
// error the route maps to 422 — "a tainted/over-scoped steer is refused", mechanically.
func steerSession(ctx context.Context, traceID, text string) error {
	traceID = strings.TrimSpace(traceID)
	if traceID == "" {
		return errors.New("trace_id is required")
	}
	key := a2achan.ChannelKey{Locale: a2achan.Session, ID: traceID}
	v := a2achan.Default.Send(ctx, "operator", key, a2achan.Shared([]byte(text)), a2achan.CapA2ASend)
	if v.Kind != abi.VerdictAllow {
		return fmt.Errorf("a2a floor refused (%s)", abi.ReasonName(v.Reason))
	}
	return nil
}

// applySessionControl routes one control verb to the matching Table write. It is the
// single place that knows the verb set, so the verb vocabulary lives with the table
// owner (cmd/fak), not the gateway. CAS (if_rev>0) reads the current record, mutates
// the one field, and CompareAndSets; a concurrent write between read and CAS loses
// the race and returns ok=false for the client to retry.
func applySessionControl(tbl *session.Table, traceID, verb string, req gateway.SessionControlRequest) (session.State, bool, error) {
	switch verb {
	case "run":
		run, ok := session.ParseRunState(req.Run)
		if !ok {
			return session.State{}, false, fmt.Errorf("unknown run-state %q (want running|throttled|paused|draining|stopped)", req.Run)
		}
		if req.IfRev > 0 {
			return casApply(tbl, traceID, req.IfRev, func(s *session.State) {
				s.Run = run
				s.Reason = transitionReason(run, req.Reason)
			})
		}
		st, ok := tbl.Transition(traceID, run, req.Reason)
		return st, ok, nil
	case "budget":
		if req.Budget == nil {
			return session.State{}, false, errors.New("budget is required")
		}
		b := session.Budget{
			TurnsLeft:         req.Budget.TurnsLeft,
			TokensLeft:        req.Budget.TokensLeft,
			ContextTokensLeft: req.Budget.ContextTokensLeft,
		}
		if req.IfRev > 0 {
			return casApply(tbl, traceID, req.IfRev, func(s *session.State) { s.Budget = b })
		}
		st, ok := tbl.SetBudget(traceID, b)
		return st, ok, nil
	case "pace":
		if req.Pace == nil {
			return session.State{}, false, errors.New("pace is required")
		}
		p := session.Pace{MaxTokensPerTurn: req.Pace.MaxTokensPerTurn, MinTurnGapMs: req.Pace.MinTurnGapMs}
		if req.IfRev > 0 {
			return casApply(tbl, traceID, req.IfRev, func(s *session.State) { s.Pace = p })
		}
		st, ok := tbl.SetPace(traceID, p)
		return st, ok, nil
	case "priority":
		if req.Priority == nil {
			return session.State{}, false, errors.New("priority is required")
		}
		pri := *req.Priority
		if req.IfRev > 0 {
			return casApply(tbl, traceID, req.IfRev, func(s *session.State) { s.Priority = pri })
		}
		st, ok := tbl.SetPriority(traceID, pri)
		return st, ok, nil
	}
	return session.State{}, false, fmt.Errorf("unknown verb %q (want run|budget|pace|priority)", verb)
}

// casApply reads the current drive record, mutates it in place, and CompareAndSets
// it against expectRev. It is the optimistic-concurrency form of every verb. A lost
// race (the table moved between read and CAS) returns ok=false; the caller maps that
// to 409 and the client re-reads and retries.
func casApply(tbl *session.Table, traceID string, expectRev uint64, apply func(*session.State)) (session.State, bool, error) {
	cur := tbl.Get(traceID)
	apply(&cur)
	st, ok := tbl.CompareAndSet(traceID, expectRev, cur)
	return st, ok, nil
}

// transitionReason mirrors Table.Transition's reason bookkeeping so a CAS run-state
// change records/clears the reason token identically to the direct path: Throttled
// and Stopped carry the reason, Running clears it.
func transitionReason(to session.RunState, reason string) string {
	switch to {
	case session.Throttled, session.Stopped:
		return reason
	case session.Running:
		return ""
	}
	return ""
}

// toGatewaySessionState projects internal/session.State into the gateway's
// session-internals-blind wire DTO. Run becomes its lowercase token; everything
// else is a 1:1 field copy.
func toGatewaySessionState(s session.State) gateway.SessionState {
	return gateway.SessionState{
		TraceID: s.TraceID,
		Run:     s.Run.String(),
		Budget: gateway.SessionBudget{
			TurnsLeft:         s.Budget.TurnsLeft,
			TokensLeft:        s.Budget.TokensLeft,
			ContextTokensLeft: s.Budget.ContextTokensLeft,
		},
		Priority:       s.Priority,
		Pace:           gateway.SessionPace{MaxTokensPerTurn: s.Pace.MaxTokensPerTurn, MinTurnGapMs: s.Pace.MinTurnGapMs},
		Reason:         s.Reason,
		ContinuationID: s.ContinuationID,
		ParentTrace:    s.ParentTrace,
		Generation:     s.Generation,
		Rev:            s.Rev,
	}
}

func toGatewaySessionVerdict(v session.Verdict) gateway.SessionVerdict {
	return gateway.SessionVerdict{
		Proceed:   v.Proceed,
		MaxTokens: v.MaxTokens,
		MinGapMs:  v.MinGapMs,
		State:     toGatewaySessionState(v.State),
		Stop:      v.Stop,
		Reason:    v.Reason,
	}
}

// taintLevelName renders an abi.TaintLabel as its stable wire name. It mirrors
// ifc's unexported taintName (the enum is not ordered by restrictiveness, so it is
// switched, never formatted).
func taintLevelName(t abi.TaintLabel) string {
	switch t {
	case abi.TaintTrusted:
		return "trusted"
	case abi.TaintTainted:
		return "tainted"
	case abi.TaintQuarantined:
		return "quarantined"
	}
	return "unknown"
}

func applyRuntime(rt policy.Runtime) {
	policy.ApplySources(rt)
	ifc.ConfigureDefaultPolicy(ifcPolicy(rt))
	applyRateLimit(rt.RateLimit)
}

// applyRateLimit pushes the manifest-declared rate_limit into the governor singleton
// (issue #699, Epic 8), mirroring how SafeSinks/Authorize reach ifc. A present block
// installs the cap (authoritative over the FAK_RATELIMIT_* env fallback); an absent
// block resets the limiter to inert  -  so editing the cap out of the file on
// --policy hot-reload removes it. Config and accrued counters are separate
// (SetLimit does not wipe budgets), exactly as a mid-flight env change behaves.
func applyRateLimit(r *policy.RateLimitRule) {
	if r == nil {
		ratelimit.Default.SetLimit(ratelimit.Limit{}, ratelimit.KeyPerTrace) // unlimited/inert
		return
	}
	ratelimit.Default.SetLimit(ratelimit.Limit{
		MaxCalls:   r.MaxCalls,
		MaxCost:    r.MaxCost,
		RetryAfter: time.Duration(r.RetryAfterMS) * time.Millisecond,
	}, rateLimitKeyMode(r.Key))
}

// rateLimitKeyMode maps the manifest key string to the governor's KeyMode. The
// manifest validator already guaranteed trace|tool|global (or empty); "" and "trace"
// both mean per-trace (the governor's default dimension).
func rateLimitKeyMode(key string) ratelimit.KeyMode {
	switch strings.ToLower(strings.TrimSpace(key)) {
	case "tool":
		return ratelimit.KeyPerTool
	case "global":
		return ratelimit.KeyGlobal
	default:
		return ratelimit.KeyPerTrace
	}
}

func ifcPolicy(rt policy.Runtime) ifc.Policy {
	p := ifc.Policy{}
	if len(rt.SafeSinks) > 0 || len(rt.AuthorizeRules) > 0 || len(rt.Sources) > 0 {
		p.GatedSinks = ifc.StrictGatedSinks()
	}
	if len(rt.SafeSinks) > 0 {
		p.SafeSinks = make(map[string]bool, len(rt.SafeSinks))
		for _, tool := range rt.SafeSinks {
			p.SafeSinks[tool] = true
		}
	}
	type rule struct {
		tool string
		sink ifc.SinkClass
	}
	rules := make([]rule, 0, len(rt.AuthorizeRules))
	for _, r := range rt.AuthorizeRules {
		rules = append(rules, rule{tool: r.Tool, sink: sinkClass(r.Sink)})
	}
	if len(rules) > 0 {
		p.Authorize = func(c *abi.ToolCall, into ifc.SinkClass) bool {
			if c == nil {
				return false
			}
			for _, r := range rules {
				if c.Tool == r.tool && into == r.sink {
					return true
				}
			}
			return false
		}
	}
	return p
}

func sinkClass(name string) ifc.SinkClass {
	switch strings.ToUpper(strings.TrimSpace(name)) {
	case "EGRESS":
		return ifc.SinkEgress
	case "EXEC":
		return ifc.SinkExec
	case "DESTRUCTIVE":
		return ifc.SinkDestructive
	default:
		return ifc.SinkNone
	}
}

// fak policy  -  author and validate the deployable capability floor. --dump emits
// the built-in DefaultPolicy as a manifest (the starting point an adopter edits);
// --check validates a manifest against the closed refusal vocabulary and prints
// the floor it admits, so a misconfigured policy is caught BEFORE it gates a run.
func cmdPolicy(argv []string) {
	fs := flag.NewFlagSet("policy", flag.ExitOnError)
	dump := fs.Bool("dump", false, "write the built-in DefaultPolicy as a manifest to stdout")
	check := fs.String("check", "", "validate a manifest file and print the floor it admits")
	_ = fs.Parse(argv)

	switch {
	case *dump:
		os.Stdout.Write(policy.FromPolicy(adjudicator.DefaultPolicy()).JSON())
	case *check != "":
		rt, err := policy.LoadRuntime(*check)
		if err != nil {
			fmt.Fprintln(os.Stderr, "fak policy:", err)
			os.Exit(1)
		}
		fmt.Printf("OK  %s  (manifest valid; every deny cites a closed-vocabulary reason)\n\n%s", *check, policy.SummaryRuntime(rt))
	default:
		fmt.Fprintln(os.Stderr, "fak policy: pass --dump (emit the default manifest) or --check FILE (validate one)")
		os.Exit(2)
	}
}

// fak lint  -  the STATIC tool linter. The kernel never trusts a tool's self-declared
// annotations: it re-checks them every call and silently does the safe thing (the
// vDSO overrides a lying readOnlyHint from the name, pre-flight re-validates args).
// This verb is the definition-time DUAL of those call-time re-checks: it runs once
// over the configured tool surface and says OUT LOUD what the runtime would only
// ever whisper to itself  -  a dead cache hint, an unreachable pure registration, a
// canned answer for a write-shaped tool, a schema the model is shown but the kernel
// never enforces. Exit 1 on an error-severity finding (or any finding with
// --strict), so it can gate a build.
func cmdLint(argv []string) {
	fs := flag.NewFlagSet("lint", flag.ExitOnError)
	asJSON := fs.Bool("json", false, "emit findings as JSON")
	strict := fs.Bool("strict", false, "exit non-zero on ANY finding (info/warn too), not just errors")
	kernelOnly := fs.Bool("kernel-only", false, "lint only the kernel registries (skip the agent hint classifier + model-facing catalog)")
	_ = fs.Parse(argv)

	var facts []toollint.ToolFacts
	if *kernelOnly {
		facts = toollint.FromKernel()
	} else {
		agent.Configure() // register the agent's schemas, grammar, and engine first
		facts = agent.LintFacts()
	}
	rep := toollint.Lint(facts)

	if *asJSON {
		type jf struct {
			Code      string `json:"code"`
			Severity  string `json:"severity"`
			Tool      string `json:"tool"`
			Message   string `json:"message"`
			Mechanism string `json:"mechanism"`
		}
		rows := make([]jf, 0, len(rep.Findings))
		for _, f := range rep.Findings {
			rows = append(rows, jf{string(f.Code), f.Severity.String(), f.Tool, f.Message, f.Mechanism})
		}
		b, _ := json.MarshalIndent(map[string]any{
			"tools":    len(facts),
			"findings": rows,
			"errors":   rep.Errors(),
			"warnings": rep.Warnings(),
			"infos":    rep.Infos(),
		}, "", "  ")
		fmt.Println(string(b))
	} else {
		for _, f := range rep.Findings {
			fmt.Printf("%s  %-5s  %-22s  %s\n          %s\n", f.Code, f.Severity.String(), f.Tool, f.Message, f.Mechanism)
		}
		if rep.Clean() {
			fmt.Printf("lint clean: %d tool(s), no findings\n", len(facts))
		} else {
			fmt.Printf("\n%d tool(s): %d error, %d warn, %d info\n", len(facts), rep.Errors(), rep.Warnings(), rep.Infos())
		}
	}

	if code := lintExitCode(rep, *strict); code != 0 {
		os.Exit(code)
	}
}

// lintExitCode is the PURE exit-code contract for `fak lint`, factored out so it is
// unit-testable without os.Exit: 1 on any error-severity finding, or  -  under
// --strict  -  on ANY finding at all (the "gate a build on a clean surface" mode the
// help text and cmdLint doc both promise). 0 otherwise.
func lintExitCode(rep toollint.Report, strict bool) int {
	if rep.Errors() > 0 || (strict && !rep.Clean()) {
		return 1
	}
	return 0
}

// fak serve  -  the GATEWAY. It fronts the kernel over an OpenAI-compatible HTTP
// surface and MCP so an agent in ANY language can route its tool calls through the
// in-process syscall boundary without writing Go. The gateway is Go and ON the
// request path (it adjudicates)  -  in-direction; non-Go CLIENTS live in the
// adopter's repo. Construction mirrors cmdAgent: registrations is already imported
// (so the resolver + full adjudicator chain are wired), the capability floor is
// installed fail-loud, and the kernel is built bound to a registered engine.
// resolveRequiredKey resolves a secret the operator REQUIRED by naming an env
// var via a --...-key-env flag. When the flag is unset (empty name) auth was not
// requested, so it returns ok=true with an empty key. But when the flag names an
// env var that is unset or empty, it returns ok=false: the operator asked for
// auth and the secret did not land (typo, un-propagated CI env, k8s Secret
// mis-mount, pod restarted without it). For an agent kernel the safe
// default is to fail CLOSED  -  refuse to start  -  not to warn and silently serve
// unauthenticated. The lookup is injected so the decision is unit-testable
// without touching process env. (issue #213-class fail-open fix; see #255.)
func resolveRequiredKey(envName string, lookup func(string) string) (key string, ok bool) {
	if envName == "" {
		return "", true // flag not set: auth not requested.
	}
	v := lookup(envName)
	if v == "" {
		return "", false // requested but missing: caller must fail closed.
	}
	return v, true
}

// fak hook  -  the spawned-hook decide transport (A/B baseline). Reads one call
// from stdin, folds the adjudicator chain, writes the verdict to stdout.
func cmdHook() {
	var c bench.Call
	if err := json.NewDecoder(os.Stdin).Decode(&c); err != nil {
		// an empty/invalid call still exercises the spawn+decide path
		c = bench.Call{Tool: "noop"}
	}
	res := abi.ActiveResolver()
	args := []byte(c.Args)
	if len(args) == 0 {
		args = []byte("{}")
	}
	ref, _ := res.Put(ctx(), args)
	tc := &abi.ToolCall{Tool: c.Tool, Args: ref, Meta: c.Meta}
	v := kernel.Fold(ctx(), abi.AdjudicatorsFor(tc), tc)
	_ = json.NewEncoder(os.Stdout).Encode(map[string]any{
		"kind": verdictName(v.Kind), "reason": abi.ReasonName(v.Reason), "by": v.By,
	})
}

func printReport(rep *metrics.Report, path string) {
	fmt.Printf("== fak bench: %s ==\n", rep.Provenance.SliceID)
	fmt.Printf("in-process adjudication p50 : %d ns\n", rep.On.P50Ns)
	fmt.Printf("spawned-hook        p50     : %d ns (%.3f ms, n=%d)\n",
		rep.Baseline.P50Ns, float64(rep.Baseline.P50Ns)/1e6, rep.Baseline.Calls)
	if rep.Baseline.P50Ns > 0 && rep.On.P50Ns > 0 {
		fmt.Printf("fusion speedup (p50)        : %.0fx\n", float64(rep.Baseline.P50Ns)/float64(rep.On.P50Ns))
	}
	fmt.Printf("PRIMARY GATE                : %s  (%s)\n", rep.GatePrimary, rep.PrimaryDetail)
	fmt.Printf("secondary token delta       : %.2f%% (soft, never gates)\n", rep.TokenDeltaPct)
	fmt.Printf("vdso hit-rate               : %.3f   pollution-rate: %.3f\n",
		rep.KPIs.VDSOHitRate, rep.KPIs.ContextPollutionRate)
	fmt.Printf("workload hash               : %s   live seam: %s\n",
		rep.Provenance.WorkloadHash, rep.LiveSeam)
	fmt.Printf("report written              : %s\n", path)
}

func traceDir() string { return testdataDir("tau2") }

func turnTaxDir() string { return testdataDir("turntax") }

// testdataDir resolves the testdata/<name> directory relative to cwd first, then
// the executable dir; it falls back to the cwd-relative path when neither exists.
// testdata sits next to the module root.
func testdataDir(name string) string {
	if _, err := os.Stat(filepath.Join("testdata", name)); err == nil {
		return filepath.Join("testdata", name)
	}
	if exe, err := os.Executable(); err == nil {
		d := filepath.Join(filepath.Dir(exe), "testdata", name)
		if _, err := os.Stat(d); err == nil {
			return d
		}
	}
	return filepath.Join("testdata", name)
}

// resolveSuite turns a --suite NAME into its trace path under dir, but FAILS LOUD
// and actionable when the file is absent: a cold-start user (or agent) who follows
// the help's `--suite NAME` and guesses a name (e.g. "default") otherwise hits a raw
// `open testdata\...\NAME.json: cannot find the file specified` with no hint of what
// IS valid. Instead we list the available suites (the *.json basenames in dir) so the
// next command is obvious. An explicit --trace PATH bypasses this (the caller owns it).
// Returns the path unchanged when the file exists; exits 2 with the suite list when not.
func resolveSuite(dir, suite string) string {
	path := filepath.Join(dir, suite+".json")
	if _, err := os.Stat(path); err == nil {
		return path
	}
	avail := availableSuites(dir)
	if len(avail) == 0 {
		fmt.Fprintf(os.Stderr, "fak: unknown suite %q — no suites found under %s (pass --trace PATH to load a trace directly)\n", suite, dir)
	} else {
		fmt.Fprintf(os.Stderr, "fak: unknown suite %q — available: %s (or pass --trace PATH)\n", suite, strings.Join(avail, ", "))
	}
	os.Exit(2)
	return path // unreachable
}

// availableSuites lists the suite NAMES (the *.json basenames, extension stripped)
// in dir, sorted, so an unknown-suite error can name the real choices. Empty on a
// missing/unreadable dir (the caller reports that case).
func availableSuites(dir string) []string {
	ents, err := os.ReadDir(dir)
	if err != nil {
		return nil
	}
	var out []string
	for _, e := range ents {
		if name := e.Name(); !e.IsDir() && strings.HasSuffix(name, ".json") {
			out = append(out, strings.TrimSuffix(name, ".json"))
		}
	}
	sort.Strings(out)
	return out
}

func verdictName(k abi.VerdictKind) string {
	switch k {
	case abi.VerdictAllow:
		return "ALLOW"
	case abi.VerdictDeny:
		return "DENY"
	case abi.VerdictTransform:
		return "TRANSFORM"
	case abi.VerdictQuarantine:
		return "QUARANTINE"
	case abi.VerdictRequireWitness:
		return "WITNESS"
	case abi.VerdictDefer:
		return "DEFER"
	case abi.VerdictIndeterminate:
		return "INDETERMINATE"
	}
	return "K" + strconv.Itoa(int(k))
}

func statusName(s abi.Status) string {
	switch s {
	case abi.StatusOK:
		return "OK"
	case abi.StatusError:
		return "ERR"
	case abi.StatusPending:
		return "PEND"
	}
	return "?"
}

func must(err error) {
	if err != nil {
		fmt.Fprintln(os.Stderr, "fak:", err)
		os.Exit(1)
	}
}

// embeddedGGUFTokenizer builds a tokenizer straight from the GGUF's own
// tokenizer.ggml.* metadata, mirroring cmd/simpledemo's embedded path. It lets
// `fak serve --gguf` serve real in-kernel chat without a separate tokenizer.json.
// Returns an error (not a panic) when the checkpoint embeds no usable BPE tokenizer,
// so the caller can fall back to the MockPlanner instead of aborting startup.
func embeddedGGUFTokenizer(ggufPath string) (*tokenizer.Tokenizer, error) {
	f, err := ggufload.Open(ggufPath)
	if err != nil {
		return nil, err
	}
	gt, ok := f.GGMLTokenizer()
	if !ok {
		return nil, fmt.Errorf("no embedded BPE tokenizer in %s", filepath.Base(ggufPath))
	}
	return tokenizer.FromGGML(gt.Tokens, gt.Merges, gt.TokenTypes, gt.Pre)
}

func cmdNewModel(argv []string) {
	fs := flag.NewFlagSet("new-model", flag.ExitOnError)
	fs.SetOutput(os.Stderr)
	family := fs.String("family", "", "family name, lowercase (e.g. myfamily)")
	topology := fs.String("topology", "identity", "topology: prenorm, postnorm, parallel, or identity")
	dryRun := fs.Bool("dry-run", false, "print scaffold without writing files")
	asJSON := fs.Bool("json", false, "emit the result as JSON")
	if err := fs.Parse(argv); err != nil {
		os.Exit(2)
	}

	if *family == "" {
		fmt.Fprintln(os.Stderr, "fak new-model: --family is required")
		fmt.Fprintln(os.Stderr, "usage: fak new-model --family <name> [--topology <topology>] [--dry-run] [--json]")
		os.Exit(2)
	}

	res, err := newmodel.Run(newmodel.Scaffold{
		Family:   *family,
		Topology: *topology,
		DryRun:   *dryRun,
	})
	if err != nil {
		fmt.Fprintf(os.Stderr, "fak new-model: %v\n", err)
		os.Exit(1)
	}

	if *asJSON {
		enc := json.NewEncoder(os.Stdout)
		enc.SetIndent("", "  ")
		enc.Encode(res)
		return
	}

	fmt.Printf("=== Scaffolding model family '%s' (topology: %s) ===\n\n", res.Family, res.Topology)
	fmt.Println("Files to edit:")
	for _, e := range res.Edits {
		fmt.Printf("  - %s\n", e)
	}
	fmt.Println("\nNext steps:")
	for _, s := range res.NextSteps {
		fmt.Printf("%s\n", s)
	}
}
