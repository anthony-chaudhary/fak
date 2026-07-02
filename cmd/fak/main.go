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
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/a2achan"
	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/adjudicator"
	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/bench"
	"github.com/anthony-chaudhary/fak/internal/benchcli"
	"github.com/anthony-chaudhary/fak/internal/gateway"
	"github.com/anthony-chaudhary/fak/internal/grammar"
	"github.com/anthony-chaudhary/fak/internal/guard"
	"github.com/anthony-chaudhary/fak/internal/ifc"
	"github.com/anthony-chaudhary/fak/internal/kernel"
	"github.com/anthony-chaudhary/fak/internal/policy"
	"github.com/anthony-chaudhary/fak/internal/ratelimit"
	"github.com/anthony-chaudhary/fak/internal/session"
	"github.com/anthony-chaudhary/fak/internal/sessionreset"
	"github.com/anthony-chaudhary/fak/internal/turnbench"

	_ "github.com/anthony-chaudhary/fak/internal/registrations"
)

func main() {
	start := time.Now()
	var verb string
	var argv []string
	if len(os.Args) >= 2 {
		verb = os.Args[1]
	}
	if len(os.Args) > 2 {
		argv = os.Args[2:]
	}
	defer func() {
		if r := recover(); r != nil {
			recordUsage(verb, argv, 2, start)
			panic(r)
		}
	}()

	if len(os.Args) < 2 {
		usage()
		recordUsage(verb, argv, 2, start)
		os.Exit(2)
	}
	// C2 of epic #2228 (#2231): `fak dev <verb>` is the canonical namespace of the
	// dev tier. It resolves BEFORE the dispatch switch by rewriting os.Args to the
	// underlying verb, so the very same case arm runs — byte-identical dispatch, no
	// re-exec — and the 200-case switch (plus the devindex scanner keyed on its
	// `switch os.Args[1]` header) stays untouched. The usage journal records the
	// composite verb ("dev commit" vs bare "commit"): the bare-vs-namespaced
	// adoption evidence the C5 enforcement flip is gated on.
	if os.Args[1] == "dev" {
		v, rest, code := resolveDevVerb(os.Args[2:], os.Stdout, os.Stderr)
		if code >= 0 {
			recordUsage(verb, argv, code, start)
			os.Exit(code)
		}
		verb = "dev " + v
		argv = rest
		os.Args = append([]string{os.Args[0], v}, rest...)
	}
	switch os.Args[1] {
	case "run":
		cmdRun(os.Args[2:])
	case "replay":
		// Explicit, unambiguous spelling of the trace-replay path (`fak run --trace`).
		cmdRunTrace(os.Args[2:])
	case "commit":
		cmdCommit(os.Args[2:])
	case "edit-tx":
		cmdEditTx(os.Args[2:])
	case "sweep":
		cmdSweep(os.Args[2:])
	case "sync":
		cmdSync(os.Args[2:])
	case "merge":
		cmdMerge(os.Args[2:])
	case "whats-changed":
		cmdWhatsChanged(os.Args[2:])
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
	case "frontierswe":
		cmdFrontierswe(os.Args[2:])
	case "sota":
		cmdSota(os.Args[2:])
	case "sota-coverage-scorecard":
		cmdSOTACoverageScorecard(os.Args[2:])
	case "bench-runs":
		cmdBenchRuns(os.Args[2:])
	case "bench-loop", "benchloop":
		cmdBenchLoop(os.Args[2:])
	case "amd-gpu-facts":
		cmdAMDGPUFacts(os.Args[2:])
	case "commit-subject-coverage":
		cmdCommitSubjectCoverage(os.Args[2:])
	case "ablate":
		cmdAblate(os.Args[2:])
	case "ablate-arm":
		// Hidden: the rung-2 arm-mode child `fak ablate` re-execs (one fresh process per
		// env-gated arm, each reading its own FAK_* at start). Reads an arm request on
		// stdin, writes one AblationRun on stdout. Not listed in usage() — an internal seam
		// of the ablate subprocess fan-out.
		cmdAblateArm(os.Args[2:])
	case "turntax":
		cmdTurnTax(os.Args[2:])
	case "hooklat":
		cmdHookLat(os.Args[2:])
	case "agent":
		cmdAgent(os.Args[2:])
	case "api-host":
		cmdAPIHost(os.Args[2:])
	case "recall":
		cmdRecall(os.Args[2:])
	case "recover":
		cmdRecover(os.Args[2:])
	case "session":
		cmdSession(os.Args[2:])
	case "session-audit":
		cmdSessionAudit(os.Args[2:])
	case "resume":
		cmdResume(os.Args[2:])
	case "dispatch":
		cmdDispatch(os.Args[2:])
	case "process-guard":
		cmdProcessGuard(os.Args[2:])
	case "windowgate":
		cmdWindowgate(os.Args[2:])
	case "ps":
		cmdPS(os.Args[2:])
	case "top":
		cmdTop(os.Args[2:])
	case "signal":
		cmdSignal(os.Args[2:])
	case "task":
		cmdTask(os.Args[2:])
	case "toolproc":
		cmdToolproc(os.Args[2:])
	case "test":
		cmdTest(os.Args[2:])
	case "done":
		cmdDone(os.Args[2:])
	case "profile":
		cmdProfile(os.Args[2:])
	case "c":
		cmdTUI(append([]string{"agent"}, os.Args[2:]...))
	case "console":
		cmdTUI(os.Args[2:])
	case "chat":
		cmdChat(os.Args[2:])
	case "chatrelay":
		cmdChatRelay(os.Args[2:])
	case "relay":
		cmdRelay(os.Args[2:])
	case "claude-mac-fak":
		cmdClaudeMacFak(os.Args[2:])
	case "codex":
		cmdCodex(os.Args[2:])
	case "codex-mcp-health":
		cmdCodexMCPHealth(os.Args[2:])
	case "loop":
		cmdLoop(os.Args[2:])
	case "bgloop":
		cmdBgloop(os.Args[2:])
	case "loop-score":
		cmdLoopScore(os.Args[2:])
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
	case "tool-coverage-audit":
		cmdToolCoverageAudit(os.Args[2:])
	case "answer-shape":
		cmdAnswerShape(os.Args[2:])
	case "claim-check":
		cmdClaimCheck(os.Args[2:])
	case "check-tool-failure":
		cmdCheckToolFailure(os.Args[2:])
	case "doctor":
		cmdDoctor(os.Args[2:])
	case "feature":
		cmdFeature(os.Args[2:])
	case "index":
		cmdIndex(os.Args[2:])
	case "orient":
		cmdOrient(os.Args[2:])
	case "workflow":
		cmdWorkflow(os.Args[2:])
	case "workflow-audit":
		cmdWorkflowAudit(os.Args[2:])
	case "tree-doctor":
		cmdTreeDoctor(os.Args[2:])
	case "self-update":
		cmdSelfUpdate(os.Args[2:])
	case "slack":
		cmdSlack(os.Args[2:])
	case "release":
		cmdRelease(os.Args[2:])
	case "release-lock":
		cmdReleaseLock(os.Args[2:])
	case "release-staleness":
		cmdReleaseStaleness(os.Args[2:])
	case "serve":
		cmdServe(os.Args[2:])
	case "serve-wiring":
		cmdServeWiring(os.Args[2:])
	case "guard":
		cmdGuard(os.Args[2:])
	case "info":
		// The live fak-info overlay: poll a fak guard/serve gateway's /debug/vars and print
		// one compact line per tick (cache economy + floor safety + liveness). This is the 20%
		// pane `fak guard --split` opens; also runnable by hand in a second pane.
		cmdInfo(os.Args[2:])
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
	case "usage":
		cmdUsage(os.Args[2:])
	case "headroom":
		cmdHeadroom(os.Args[2:])
	case "fleetcap":
		cmdFleetcap(os.Args[2:])
	case "vcache":
		cmdVCache(os.Args[2:])
	case "hook":
		cmdHook()
	case "hooks":
		cmdHooks(os.Args[2:])
	case "hygiene":
		cmdHygiene(os.Args[2:])
	case "public-scrub":
		cmdPublicScrub(os.Args[2:])
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
	case "new-leaf":
		cmdNewLeaf(os.Args[2:])
	case "pull":
		// Top-level alias for `fak model pull`: the Ollama-style run-by-name download.
		cmdModelPull(os.Args[2:])
	case "ls":
		// Top-level alias for `fak model ls`: list known model aliases + cache status.
		cmdModelLs(os.Args[2:])
	case "route":
		cmdRoute(os.Args[2:])
	case "llmd-smoke", "llm-d-smoke":
		cmdLLMDSmoke(os.Args[2:])
	case "routebench":
		cmdRoutebench(os.Args[2:])
	case "accounts":
		cmdAccounts(os.Args[2:])
	case "fleet-accounts":
		cmdFleetAccounts(os.Args[2:])
	case "fleet":
		cmdFleet(os.Args[2:])
	case "garden":
		cmdGarden(os.Args[2:])
	case "cadence":
		cmdCadence(os.Args[2:])
	case "operator":
		cmdOperatorHeaviness(os.Args[2:])
	case "milestone":
		cmdMilestone(os.Args[2:])
	case "milestone-scorecard":
		cmdMilestoneScorecard(os.Args[2:])
	case "program":
		cmdProgram(os.Args[2:])
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
	case "superloop":
		cmdSuperloop(os.Args[2:])
	case "fused":
		cmdFused(os.Args[2:])
	case "experiments":
		cmdExperiments(os.Args[2:])
	case "coverage-matrix":
		cmdCoverageMatrix(os.Args[2:])
	case "support-maturity-scorecard":
		cmdSupportMaturityScorecard(os.Args[2:])
	case "support":
		cmdSupport(os.Args[2:])
	case "dojo":
		cmdDojo(os.Args[2:])
	case "dojo-rsi":
		cmdDojoRSI(os.Args[2:])
	case "guard-verdict-rsi":
		cmdGuardVerdictRSI(os.Args[2:])
	case "guard-rsi-scorecard":
		cmdGuardRSIScorecard(os.Args[2:])
	case "opt":
		cmdOpt(os.Args[2:])
	case "dogfood-score":
		cmdDogfoodScore(os.Args[2:])
	case "concept-usage-score":
		cmdConceptUsageScore(os.Args[2:])
	case "propagation-scorecard":
		cmdPropagationScorecard(os.Args[2:])
	case "propagation-debt-dispatch":
		cmdPropagationDebtDispatch(os.Args[2:])
	case "maturity":
		cmdMaturity(os.Args[2:])
	case "token-defaults-scorecard":
		cmdTokenDefaultsScorecard(os.Args[2:])
	case "skill-effectiveness-scorecard":
		cmdSkillEffectivenessScorecard(os.Args[2:])
	case "conflation-scorecard":
		cmdConflationScorecard(os.Args[2:])
	case "scorecard":
		cmdScorecardPane(os.Args[2:])
	case "repo-hygiene-scorecard":
		cmdRepoHygieneScorecard(os.Args[2:])
	case "ui-quality-scorecard":
		cmdUIQualityScore(os.Args[2:])
	case "scoreboard":
		cmdScoreboard(os.Args[2:])
	case "steering":
		cmdSteering(os.Args[2:])
	case "blockers":
		cmdBlockers(os.Args[2:])
	case "product":
		cmdProduct(os.Args[2:])
	case "product-scorecard":
		os.Exit(runProductScorecard(os.Stdout, os.Stderr, os.Args[2:]))
	case "grafana":
		cmdGrafana(os.Args[2:])
	case "cachevalue":
		cmdCachevalue(os.Args[2:])
	case "marketing":
		cmdMarketing(os.Args[2:])
	case "news":
		cmdNews(os.Args[2:])
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
	case "issue":
		cmdIssue(os.Args[2:])
	case "complain":
		cmdComplain(os.Args[2:])
	case "learning-debt-dispatch":
		cmdLearningDebtDispatch(os.Args[2:])
	case "stopfailure":
		cmdStopFailure(os.Args[2:])
	case "cluster":
		cmdCluster(os.Args[2:])
	case "leaseref":
		cmdLeaseref(os.Args[2:])
	case "intent":
		cmdIntent(os.Args[2:])
	case "memgate":
		cmdMemgate(os.Args[2:])
	case "memory-read":
		cmdMemoryRead(os.Args[2:])
	case "memory-stability-governor":
		cmdMemoryStabilityGovernor(os.Args[2:])
	case "node":
		cmdNode(os.Args[2:])
	case "node-compare":
		cmdNodeCompare(os.Args[2:])
	case "plan-audit":
		cmdPlanAudit(os.Args[2:])
	case "qwen36-node-reports":
		cmdQwen36NodeReports(os.Args[2:])
	case "qwen36-parity-witness-gate":
		cmdQwen36ParityWitnessGate(os.Args[2:])
	case "lab":
		cmdLab(os.Args[2:])
	case "fleet-trend":
		cmdFleetTrend(os.Args[2:])
	case "issue-contract-repair":
		cmdIssueContractRepair(os.Args[2:])
	case "popularization-tickets":
		cmdPopularizationTickets(os.Args[2:])
	case "readme-visual-audit":
		cmdReadmeVisualAudit(os.Args[2:])
	case "codex-memory":
		cmdCodexMemory(os.Args[2:])
	case "version", "-v", "--version":
		cmdVersion(os.Stdout)
	case "-h", "--help", "help":
		cmdHelp(os.Args[2:])
	default:
		fmt.Fprintf(os.Stderr, "fak: unknown verb %q\n", os.Args[1])
		if s := suggestVerb(os.Args[1]); s != "" {
			fmt.Fprintf(os.Stderr, "  did you mean 'fak %s'?\n", s)
		}
		fmt.Fprintln(os.Stderr, "  'fak help' shows the overview; 'fak help --all' lists every verb.")
		recordUsage(verb, argv, 2, start)
		os.Exit(2)
	}
	recordUsage(verb, argv, 0, start)
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

	must(benchcli.WriteReport(*out, rep.JSON()))
	// also dump the standalone baseline witness (unit 23)
	if rep.Baseline.Calls > 0 {
		must(benchcli.WriteReport(filepath.Join(filepath.Dir(*out), "baseline.json"), map[string]any{
			"source": rep.Baseline.Source, "p50_ns": rep.Baseline.P50Ns,
			"p50_ms": float64(rep.Baseline.P50Ns) / 1e6, "calls": rep.Baseline.Calls,
		}))
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
		must(benchcli.WriteReport(bePath, rep.JSON()))
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

	must(benchcli.WriteReport(*out, rep.JSON()))
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

// resetTransactions is the process-local, append-only audit trail of every
// budget-triggered reset resetServedSessionOnBudget performs (issue #1582): a
// ResetTransactionLog row is appended immediately after each successful
// Table.RecontinueWithTransaction call, so the full reset history for the process
// survives independent of any single trace's State (which only carries the LATEST
// transaction in its own lineage). Safe for concurrent resets across traces — see
// ResetTransactionLog's own locking.
var resetTransactions session.ResetTransactionLog

// observeSession is the read side of the /v1/fak/session control surface (#620): it
// returns one served session's current DRIVE state so an operator can read how hard
// a live session is running without reconstructing it from git + a process scan. An
// unseen trace reads its default  -  Running, unbounded budget  -  the table's own safe
// default, never a phantom Stopped.
func observeSession(_ context.Context, traceID string) gateway.SessionState {
	traceID = strings.TrimSpace(traceID)
	if serveSessionDurability != nil && !sessionTableHas(serveSessions, traceID) {
		if st, ok, err := serveSessionDurability.lookupState(traceID); err == nil && ok {
			return toGatewaySessionState(st)
		}
	}
	return toGatewaySessionState(serveSessions.Get(traceID))
}

// listSessions is the multi-session read side of the /v1/fak/session control surface:
// it projects the WHOLE live drive table (Snapshot order  -  by priority, lower yields
// first) into the gateway wire DTO so an operator can see what every session is doing
// right now in one read, instead of reconstructing liveness from git + a process scan
// (docs/dispatch-loop.md). Snapshot already returns a fresh, sorted copy.
func listSessions(_ context.Context) []gateway.SessionState {
	snap := serveSessions.Snapshot()
	out := make([]gateway.SessionState, 0, len(snap))
	seen := make(map[string]bool, len(snap))
	for _, s := range snap {
		out = append(out, toGatewaySessionState(s))
		seen[s.TraceID] = true
	}
	if serveSessionDurability != nil {
		if persisted, err := serveSessionDurability.snapshotStates(); err == nil {
			for _, s := range persisted {
				if seen[s.TraceID] {
					continue
				}
				out = append(out, toGatewaySessionState(s))
			}
		}
	}
	return out
}

// decideSession is the served request-boundary hook: it applies session.Table.Decide
// to the process-local DRIVE table so served model requests honor run-state,
// turn-budget, token-budget, and pace controls before the upstream request runs.
func decideSession(ctx context.Context, traceID string) gateway.SessionVerdict {
	traceID = strings.TrimSpace(traceID)
	v := serveSessions.Decide(traceID)
	persistServeSessionRevision(ctx, traceID, v.State)
	return toGatewaySessionVerdict(v)
}

// debitSession records post-response token usage for a served request. The next
// Decide observes normal token-budget exhaustion at the following turn boundary;
// context-budget exhaustion drains immediately with a continuation id, and a
// priced spend-ceiling crossing drains immediately with BUDGET_SPEND_EXHAUSTED
// (the turn cost comes from the host spend meter — see session_spend.go).
func debitSession(ctx context.Context, traceID string, usage gateway.SessionUsage) gateway.SessionState {
	traceID = strings.TrimSpace(traceID)
	st := serveSessions.DebitUsage(traceID, session.Usage{
		OutputTokens:   usage.CompletionTokens,
		ContextTokens:  usage.ContextTokens,
		CostMicroCents: servedTurnSpendMicroCents(usage),
	})
	persistServeSessionRevision(ctx, traceID, st)
	return toGatewaySessionState(st)
}

// resetServedSessionOnBudget is the host-owned "human-like reset" hook the gateway
// calls after a context-budget drain. It distills the refused request transcript into
// a compact carryover seed, re-arms the continuation trace with a fresh context budget,
// and hands both back to the gateway so the live request can continue transparently.
func resetServedSessionOnBudget(freshContextTokens int) gateway.ResetOnBudgetFunc {
	if freshContextTokens <= 0 {
		return nil
	}
	return func(ctx context.Context, traceID string, messages []agent.Message) (string, []agent.Message, bool) {
		traceID = strings.TrimSpace(traceID)
		st := serveSessions.Get(traceID)
		child := strings.TrimSpace(st.ContinuationID)
		if traceID == "" || child == "" {
			return "", nil, false
		}
		resetInput := sessionreset.Input{
			Trace:          traceID,
			Messages:       resetMsgs(messages),
			FreshBudgetTok: freshContextTokens,
		}
		seed := sessionreset.BuildSeed(resetInput)
		if strings.TrimSpace(seed.Recap) == "" {
			return "", nil, false
		}
		resetTx := sessionreset.BuildResetTransaction(resetInput, child, seed)
		childState := serveSessions.RecontinueWithTransaction(traceID, child, session.Budget{
			TurnsLeft:         session.Unbounded,
			TokensLeft:        session.Unbounded,
			ContextTokensLeft: freshContextTokens,
		}, resetTx)
		resetTransactions.Append(childState.ResetTransaction)
		persistServeSessionRevision(ctx, child, childState)
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
func controlSession(ctx context.Context, traceID, verb string, req gateway.SessionControlRequest) (gateway.SessionState, bool, error) {
	traceID = strings.TrimSpace(traceID)
	st, ok, err := applySessionControlDurable(ctx, serveSessions, serveSessionDurability, traceID, verb, req)
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
	return toGatewaySessionStateAt(s, time.Now())
}

// toGatewaySessionStateAt is the clock-injectable core of toGatewaySessionState: the
// only time-dependent projection is the wall-clock TimeBudget (elapsed keeps ticking in
// real time even when no token is spent), so `now` is threaded explicitly here — matching
// the session package's clock-injection discipline — and the process-boundary wrapper
// above supplies time.Now(). A deterministic `now` lets a test assert the projected
// elapsed/remaining without a sleep.
func toGatewaySessionStateAt(s session.State, now time.Time) gateway.SessionState {
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
		CacheAffinity: gateway.SessionCacheAffinity{
			Action:      s.CacheAffinity.Action,
			AffinityKey: s.CacheAffinity.AffinityKey,
			FromTraceID: s.CacheAffinity.FromTraceID,
			ToTraceID:   s.CacheAffinity.ToTraceID,
			Reason:      s.CacheAffinity.Reason,
		},
		ResetTransaction: toGatewayResetTransaction(s.ResetTransaction),
		Assumptions:      toGatewaySessionAssumptions(s.Assumptions),
		Time:             toGatewaySessionTime(s.Time, now),
		Rev:              s.Rev,
	}
}

// toGatewaySessionTime projects a session's wall-clock TimeBudget into the read-only wire
// form `fak session status` renders — the field that finally makes `--max-duration`
// legible (it was armed and enforced, but never observable). It surfaces the budget
// whenever a wall-clock envelope is configured OR the clock has ticked at all, so an
// UNBOUNDED-but-running guard session ("--max-duration 0 … still tracked for session
// status") still reports its elapsed time. A never-started, unconfigured TimeBudget
// projects to the zero SessionTime, which omitzero drops from the wire entirely.
func toGatewaySessionTime(tb session.TimeBudget, now time.Time) gateway.SessionTime {
	q := tb.Query(now)
	elapsed := tb.Elapsed(now)
	if !q.Bounded && elapsed <= 0 {
		return gateway.SessionTime{}
	}
	return gateway.SessionTime{
		Bounded:          q.Bounded,
		Exceeded:         q.Exceeded,
		ElapsedSeconds:   int64(elapsed / time.Second),
		RemainingSeconds: int64(q.Remaining / time.Second),
		LimitSeconds:     int64(q.Limit / time.Second),
	}
}

func toGatewaySessionAssumptions(in []session.Assumption) []gateway.SessionAssumption {
	if len(in) == 0 {
		return nil
	}
	out := make([]gateway.SessionAssumption, 0, len(in))
	for _, a := range in {
		out = append(out, gateway.SessionAssumption{
			Key:        a.Key,
			Statement:  a.Statement,
			Source:     a.Source,
			Confidence: a.Confidence,
			Expiry:     a.Expiry,
			SourceRef:  a.SourceRef,
		})
	}
	return out
}

func toGatewayResetTransaction(tx session.ResetTransaction) gateway.SessionResetTransaction {
	out := gateway.SessionResetTransaction{
		Schema:       tx.Schema,
		OldTrace:     tx.OldTrace,
		NewTrace:     tx.NewTrace,
		SeedDigest:   tx.SeedDigest,
		Contributors: append([]string(nil), tx.Contributors...),
		BudgetRearm: gateway.SessionResetBudgetRearm{
			TurnsLeft:         tx.BudgetRearm.TurnsLeft,
			TokensLeft:        tx.BudgetRearm.TokensLeft,
			ContextTokensLeft: tx.BudgetRearm.ContextTokensLeft,
			ContextTokensCap:  tx.BudgetRearm.ContextTokensCap,
		},
		WarmPrefixDigest: tx.WarmPrefixDigest,
	}
	if len(tx.OmittedSpans) > 0 {
		out.OmittedSpans = make([]gateway.SessionResetOmittedSpan, 0, len(tx.OmittedSpans))
		for _, span := range tx.OmittedSpans {
			out.OmittedSpans = append(out.OmittedSpans, gateway.SessionResetOmittedSpan{
				Index:  span.Index,
				Role:   span.Role,
				Digest: span.Digest,
				Reason: span.Reason,
			})
		}
	}
	return out
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
