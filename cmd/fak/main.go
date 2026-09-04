// Command fak is the Fused Agent Kernel: one statically-linked Go binary that
// runs an agentic tool loop where every tool call passes through one in-process
// policy and quarantine boundary (adjudicate -> vDSO -> pre-flight/grammar ->
// dispatch -> context-MMU admit). Verbs:
//
//	fak run        -  replay a trace (or a single call) through the kernel
//	fak preflight  -  run only the pre-flight + grammar rungs over a call
//	fak bench      -  A/B ablate the vDSO over a frozen trace, emit report.json
//	fak policy     -  dump / validate the deployable capability-floor manifest
//	fak scratch-janitor - scan and optionally remove stale session scratchpads
//	fak temp-artifacts  - preview or safely reap stale direct fak OS-temp files
//	fak hook       -  spawned-hook mode: decide one call from stdin (the baseline)
//
// The single blank import of internal/registrations is what wires every leaf
// subsystem into the frozen ABI before the kernel boots.
package main

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/adjudicator"
	"github.com/anthony-chaudhary/fak/internal/bench"
	"github.com/anthony-chaudhary/fak/internal/benchcli"
	"github.com/anthony-chaudhary/fak/internal/gateway"
	"github.com/anthony-chaudhary/fak/internal/grammar"
	"github.com/anthony-chaudhary/fak/internal/guard"
	"github.com/anthony-chaudhary/fak/internal/ifc"
	"github.com/anthony-chaudhary/fak/internal/journal"
	"github.com/anthony-chaudhary/fak/internal/kernel"
	"github.com/anthony-chaudhary/fak/internal/policy"
	"github.com/anthony-chaudhary/fak/internal/turnbench"

	_ "github.com/anthony-chaudhary/fak/internal/registrations"
)

func main() {
	start := time.Now()
	verb, argv := parseVerbArgv()
	defer recoverUsage(&verb, &argv, start)
	// The two pre-switch gates (empty verb, `fak dev <verb>` namespace) live in
	// resolveEarlyDispatch so main() stays at the routing table; a true return means
	// the call was fully handled there.
	if resolveEarlyDispatch(&verb, &argv, start) {
		return
	}
	if dispatchPrimaryVerb(os.Args[1], os.Args[2:], start, &verb) {
		recordUsage(verb, argv, 0, start)
		return
	}
	if dispatchCoreVerbA(os.Args[1], os.Args[2:]) || dispatchCoreVerbB(os.Args[1], os.Args[2:]) ||
		dispatchExtendedVerbA(os.Args[1], os.Args[2:]) || dispatchExtendedVerbB(os.Args[1], os.Args[2:]) {
		recordUsage(verb, argv, 0, start)
		return
	}
	fmt.Fprintf(os.Stderr, "fak: unknown verb %q\n", os.Args[1])
	if s := suggestVerbSpelling(os.Args[1]); s != "" {
		fmt.Fprintf(os.Stderr, "  did you mean 'fak %s'?\n", s)
	}
	fmt.Fprintln(os.Stderr, "  'fak help' shows the overview; 'fak help --all' lists every verb.")
	recordUsage(verb, argv, 2, start)
	os.Exit(2)
}

func dispatchCoreVerbA(name string, args []string) bool {
	switch name {
	case "agent":
		cmdAgent(args)
	case "agentic":
		cmdAgentic(args)
	case "harness":
		cmdHarnessCommand(args)
	case "armbench":
		cmdArmbench(args)
	case "api-host":
		cmdAPIHost(args)
	case "question-ledger":
		cmdQuestionLedger(args)
	case "disambiguation":
		cmdDisambiguation(args)
	case "trunk-build-probe":
		cmdTrunkBuildProbe(args)
	case "godsplit-plan":
		cmdGodsplitPlan(args)
	case "glm52-prefill-sweep":
		cmdGLM52PrefillSweep(args)
	case "recall":
		cmdRecall(args)
	case "recover":
		cmdRecover(args)
	case "concept":
		os.Exit(runConceptCLI(os.Stdout, os.Stderr, args))
	case "config":
		cmdConfig(args)
	case "coordinate":
		cmdCoordinate(args)
	case "rename-concept":
		cmdRenameConcept(args)
	case "session":
		cmdSession(args)
	case "session-audit":
		cmdSessionAudit(args)
	case "scratch-janitor":
		os.Exit(runScratchJanitor(os.Stdout, os.Stderr, args))
	case "temp-artifacts":
		os.Exit(runTempArtifacts(os.Stdout, os.Stderr, args))
	case "workspin":
		cmdWorkspin(args)
	case "causal-receipt", "causalreceipt":
		cmdCausalReceipt(args)
	case "codex-resume":
		cmdCodexResume(args)
	case "sessionjournal":
		cmdSessionJournal(args)
	case "tier-calibrate":
		cmdTierCalibrate(args)
	case "resume":
		cmdResume(args)
	case "dispatch":
		cmdDispatch(args)
	case "dispatch-conservation":
		cmdDispatchConservation(args)
	case "dispatch-aging":
		cmdDispatchAging(args)
	case "knownbad":
		cmdKnownBad(args)
	case "process-guard":
		cmdProcessGuard(args)
	case "windowgate":
		cmdWindowgate(args)
	case "conpty":
		cmdConPTY(args)
	case "agents":
		cmdAgents(args)
	case "lifecycle":
		cmdLifecycle(args)
	case "ps":
		cmdPS(args)
	case "top":
		cmdTop(args)
	case "signal":
		cmdSignal(args)
	case "task":
		cmdTask(args)
	case "tasks":
		cmdTasks(args)
	case "toolproc":
		cmdToolproc(args)
	case "stallscan":
		cmdStallscan(args)
	case "terminal-relief":
		cmdTerminalRelief(args)
	case "schedule-held":
		cmdScheduleHeld(args)
	case "learning-observation":
		cmdLearningObservation(args)
	case "learning-mesh":
		os.Exit(runLearningMesh(os.Stdout, os.Stderr, args))
	case "host-crash":
		cmdHostCrash(args)
	case "hostdiag":
		cmdHostdiag(args)
	case "shellprov":
		cmdShellprov(args)
	case "host-relaunch-broker":
		os.Exit(runHostRelaunchBroker(os.Stdout, os.Stderr, args))
	case "watchdog-audit-run":
		os.Exit(runWatchdogAuditRunner(os.Stdout, os.Stderr, args))
	case "watchdog-audit-health":
		os.Exit(runWatchdogAuditHealth(os.Stdout, os.Stderr, args, time.Now))
	case "schedscan":
		cmdSchedScan(args)
	case "growthgate":
		cmdGrowthgate(args)
	case "egresslist":
		cmdEgresslist(args)
	case "test":
		cmdTest(args)
	case "done":
		cmdDone(args)
	case "profile":
		cmdProfile(args)
	case "multisubmit":
		cmdMultiSubmit(args)
	case "c":
		cmdTUI(append([]string{"agent"}, args...))
	case "console":
		cmdTUI(args)
	case "chat":
		cmdChat(args)
	case "chatrelay":
		cmdChatRelay(args)
	case "relay":
		cmdRelay(args)
	case "claude-mac-fak", "mac":
		// `fak mac` is the crisp, memorable handle; `fak claude-mac-fak` is the long
		// form kept working byte-for-byte. Both spellings route to the one handler.
		cmdClaudeMacFak(args)
	case "macbench":
		cmdMacBench(args)
	case "macfit":
		cmdMacFit(args)
	case "capabilities":
		os.Exit(runCapabilities(os.Stdout, os.Stderr, args))
	case "runtime-capabilities":
		os.Exit(runRuntimeCapabilities(os.Stdout, os.Stderr, args))
	case "codex":
		cmdCodex(args)
	case "opencode":
		cmdOpencode(args)
	case "codex-mcp-health":
		cmdCodexMCPHealth(args)
	case "loop":
		cmdLoop(args)
	case "bgloop":
		cmdBgloop(args)
	default:
		return false
	}
	return true
}

func dispatchCoreVerbB(name string, args []string) bool {
	switch name {
	case "compute-trace":
		os.Exit(runComputeTrace(os.Stdout, os.Stderr, args))
	case "loop-score":
		cmdLoopScore(args)
	case "waiting":
		cmdWaiting(args)
	case "cron":
		cmdCron(args)
	case "logvault":
		cmdLogvault(args)
	case "guard-audit":
		cmdGuardAudit(args)
	case "snapshot":
		cmdSnapshot(args)
	case "traj":
		cmdTraj(args)
	case "trajectory":
		cmdTrajectory(args)
	case "quantwatch":
		cmdQuantwatch(args)
	case "cubicquanteval":
		cmdCubicQuantEval(args)
	case "bitnetmeta":
		cmdBitnetmeta(args)
	case "codebookmeta":
		cmdCodebookmeta(args)
	case "work-delivery":
		cmdWorkDelivery(args)
	case "worktype":
		if len(args) > 0 && args[0] == "attribution" {
			os.Exit(runWorktypeSpend(os.Stdout, os.Stderr, args[1:]))
		}
		os.Exit(2)
	case "workpattern":
		if err := cmdWorkpattern(args); err != nil {
			fmt.Fprintln(os.Stderr, "fak workpattern:", err)
			os.Exit(2)
		}
	case "trajctl":
		cmdTrajctl(args)
	case "shadowgit":
		cmdShadowGit(args)
	case "signals":
		cmdSignals(args)
	case "trajquery":
		cmdTrajQuery(args)
	case "dup":
		cmdDup(args)
	case "dream":
		cmdDream(args)
	case "memory":
		cmdMemory(args)
	case "debug":
		cmdDebug(args)
	case "policy":
		cmdPolicy(args)
	case "enroll":
		// Pin (or show, or revoke) this box's org trust anchor — the opt-in door to
		// the org-policy plane, #5323. See cmd/fak/enroll.go.
		cmdEnroll(args)
	case "org":
		// Read the org-policy plane: posture, staleness, and which channel owns each
		// capability knob — the legibility half of centralized control, #5322. See
		// cmd/fak/org.go.
		cmdOrg(args)
	case "egress":
		cmdEgress(args)
	case "eve":
		// The Eve integration bridge (#2600): mechanical security preflight over
		// eve's MCP/OpenAPI connections (#2602). See cmd/fak/eve.go.
		cmdEve(args)
	case "doomloop":
		// The two-axis doom-loop guard: classifies live workers on effort vs
		// verified progress and wires the reversible-first correction. See
		// cmd/fak/doomloop.go and internal/doomloop.
		cmdDoomloop(args)
	case "lint":
		cmdLint(args)
	case "codelint":
		cmdCodelint(args)
	case "lsp":
		cmdLSP(args)
	case "breath":
		cmdBreath(args)
	case "answer-shape":
		cmdAnswerShape(args)
	case "negate":
		// The negation operator (#4461/#4472, negframe L2): `fak negate detect|resolve|reframe`
		// exposes detect (Classify), resolve (positive-complement over the L2 registry), and
		// reframe (emit-time Reframe) as one callable primitive over internal/negframe.
		cmdNegate(args)
	case "claim-check":
		cmdClaimCheck(args)
	case "citeverify":
		cmdCiteverify(args)
	case "native-first-lint":
		os.Exit(cmdNativeFirstLint(args))
	case "headless-lint":
		cmdHeadlessLint(args)
	case "hwgate-lint":
		cmdHwGateLint(args)
	case "ctxplans":
		cmdCtxPlans(args)
	case "check-tool-failure":
		cmdCheckToolFailure(args)
	case "doctor":
		cmdDoctor(args)
	case "init":
		// Emit a minimal, valid fak.toml deployment manifest (#3421).
		cmdInit(args)
	case "workflow":
		cmdWorkflow(args)
	case "fanout":
		cmdFanout(args)
	case "slack":
		cmdSlack(args)
	case "chatops":
		// The inbound read-only Slack control door (chatops.go). docs/cli-reference.md has
		// documented `fak chatops` since it landed, but the verb was never routed here, so
		// every invocation fell through to the unknown-verb path.
		cmdChatOps(args)
	case "release":
		cmdRelease(args)
	case "release-lock":
		cmdReleaseLock(args)
	case "release-staleness":
		cmdReleaseStaleness(args)
	case "watchdog":
		cmdWatchdog(args)
	case "service":
		cmdService(args)
	case "ops":
		os.Exit(runOps(os.Stdout, os.Stderr, args))
	case "micro":
		// The native in-process Go microagent runtime front door (see cmdMicro).
		cmdMicro(args)
	case "microbench":
		// Per-agent RSS + CPU density witness for the in-process microagent host
		// vs the guarded-CLI baseline (#2008; mock engine, no spend).
		cmdMicroBench(args)
	case "token-profile":
		cmdTokenProfile(args)
	case "up":
		cmdUp(args)
	case "serve":
		cmdServe(args)
	case "serve-wiring":
		cmdServeWiring(args)
	case "manage", "m":
		cmdManage(args)
	case "guard":
		cmdGuard(args)
	case "launchguard":
		cmdLaunchguard(args)
	case "goal-park":
		cmdGoalPark(args)
	case "progress":
		cmdProgress(args)
	case "info":
		// The live fak-info overlay: poll a fak guard/serve gateway's /debug/vars and print
		// one compact line per tick (cache economy + floor safety + liveness). This is the 20%
		// pane `fak guard --split` opens; also runnable by hand in a second pane.
		cmdInfo(args)
	case "demo":
		// The zero-flag 60-second proof: fak's offline scenario through the REAL kernel,
		// one live verdict per call class (see cmdDemo). No agent, no key.
		cmdDemo(args)
	case "guard-precompact":
		// Hidden: Claude Code PreCompact hook actuator installed by `fak guard`.
		cmdGuardPreCompact(args)
	case "guard-stophook":
		// Hidden: Claude Code Stop hook actuator installed by `fak guard` (see
		// cmdGuardStopHook: blocks an unchosen end_turn past a fully-refused turn).
		cmdGuardStopHook(args)
	case "guard-goal-question":
		// Hidden: active-goal AskUserQuestion PreToolUse boundary.
		cmdGuardGoalQuestionHook(args)
	case "guard-commit-gate":
		// Hidden: Claude Code PreToolUse boundary for git commit stamp/path binding.
		cmdGuardCommitGate(args)
	case "guard-sessionstart":
		// Hidden: Claude Code SessionStart hook actuator installed by `fak guard` (#3092).
		cmdGuardSessionStart(args)
	case "guard-stops":
		// Operator-facing: fold the typed Stop-hook decision ledger into a tally (clean
		// stops, bounded stand-downs, and the fail-open stops that are otherwise invisible).
		cmdGuardStops(args)
	case "guard-stops-slack":
		// Durable update-in-place scoreboard feeder for the Stop decision ledger.
		cmdGuardStopsSlack(args)
	case "trunk-red":
		// Operator-facing: fold the pre-existing trunk-red witness ledger into the distinct
		// shared breaks the build gates admitted over (a peer's red, not yours) — worst (most
		// clones stuck) first, so a break the whole fleet inherits gets fixed at its source.
		cmdTrunkRed(args)
	case guard.TrampolineVerb:
		// Hidden internal seam: the Landlock hook-floor re-exec trampoline, Linux (see
		// guard.LandlockTrampoline).
		if err := guard.LandlockTrampoline(args); err != nil {
			fmt.Fprintf(os.Stderr, "fak: %v\n", err)
			os.Exit(127)
		}
	default:
		return false
	}
	return true
}

func dispatchExtendedVerbA(name string, args []string) bool {
	switch name {
	case "audit":
		cmdAudit(args)
	case "value-chain":
		cmdValueChain(args)
	case "usage":
		cmdUsage(args)
	case "provider-cost":
		cmdProviderCost(args)
	case "goal":
		cmdGoal(args)
	case "headroom":
		cmdHeadroom(args)
	case "vcache":
		cmdVCache(args)
	case "hook":
		cmdHook()
	case "hooks":
		cmdHooks(args)
	case "hygiene":
		cmdHygiene(args)
	case "idempotency":
		cmdIdempotency(args)
	case "public-scrub":
		cmdPublicScrub(args)
	case "rungstats":
		cmdRungStats(args)
	case "swebench":
		cmdSwebench(args)
	case "webbench":
		cmdWebbench(args)
	case "ailuminate":
		cmdAILuminate(args)
	case "model":
		cmdModel(args)
	case "model-default":
		os.Exit(runModelDefault(os.Stdout, os.Stderr, args))
	case "new-model":
		cmdNewModel(args)
	case "architecture":
		cmdArchitecture(args)
	case "new-leaf":
		cmdNewLeaf(args)
	case "pull":
		// Top-level alias for `fak model pull`: the Ollama-style run-by-name download.
		cmdModelPull(args)
	case "ls":
		// Top-level alias for `fak model ls`: list known model aliases + cache status.
		cmdModelLs(args)
	case "route":
		cmdRoute(args)
	case "execution-route":
		cmdExecutionRoute(args)
	case "llmd-smoke", "llm-d-smoke":
		cmdLLMDSmoke(args)
	case "routebench":
		cmdRoutebench(args)
	case "deepseekbench":
		cmdDeepSeekBench(args)
	case "assume":
		cmdAssume(args)
	case "accounts":
		cmdAccounts(args)
	case "fleet-accounts":
		cmdFleetAccounts(args)
	case "fleet":
		cmdFleet(args)
	case "garden":
		cmdGarden(args)
	case "stale-work":
		cmdStaleWork(args)
	case "cadence":
		cmdCadence(args)
	case "operator":
		cmdOperatorHeaviness(args)
	case "milestone":
		cmdMilestone(args)
	case "milestone-scorecard":
		cmdMilestoneScorecard(args)
	case "project":
		// The ProjectsV2 board control-pane fold — makes the board an operator-visible
		// dimension (report/verdict/Slack-ready) instead of a write-only sync target.
		cmdProject(args)
	case "mlp-score":
		// The witnessed grade of the "first lovable cut" for the all-in-one agent
		// runtime epic (#3256, milestone #17): per-criterion PASS/not-yet (#3284).
		cmdMLPScore(args)
	case "program":
		cmdProgram(args)
	case "rollup":
		cmdRollup(args)
	case "spend":
		cmdSpend(args)
	case "budget":
		cmdBudget(args)
	case "sidecar":
		cmdSidecar(args)
	case "nightrun":
		cmdNightrun(args)
	case "sessions":
		cmdSessions(args)
	case "search":
		cmdSearch(args)
	case "loop-index-scorecard":
		cmdLoopIndexScorecard(args)
	case "loop-map":
		cmdLoopMap(args)
	case "superloop":
		cmdSuperloop(args)
	case "superstream":
		cmdSuperstream(args)
	case "arch":
		cmdArch(args)
	case "fused":
		cmdFused(args)
	case "experiments":
		cmdExperiments(args)
	case "coverage-matrix":
		cmdCoverageMatrix(args)
	case "conformance":
		cmdConformance(args)
	case "support-maturity-scorecard":
		cmdSupportMaturityScorecard(args)
	case "support":
		cmdSupport(args)
	case "component":
		cmdComponent(args)
	case "dojo":
		cmdDojo(args)
	case "dojo-rsi":
		cmdDojoRSI(args)
	case "guard-verdict-rsi":
		cmdGuardVerdictRSI(args)
	case "guard-rsi-scorecard":
		cmdGuardRSIScorecard(args)
	case "opt":
		cmdOpt(args)
	case "dogfood-score":
		cmdDogfoodScore(args)
	case "concept-usage-score":
		cmdConceptUsageScore(args)
	case "propagation-scorecard":
		cmdPropagationScorecard(args)
	default:
		return false
	}
	return true
}

func dispatchExtendedVerbB(name string, args []string) bool {
	if name == "orientation" {
		cmdOrientation(args)
		return true
	}
	if name == "centrality-audit" {
		cmdCentralityAudit(args)
		return true
	}
	switch name {
	case "agent-queue":
		cmdAgentQueue(args)
	case "propagation-debt-dispatch":
		cmdPropagationDebtDispatch(args)
	case "unwired-scorecard":
		cmdUnwiredScorecard(args)
	case "unwired-debt-dispatch":
		cmdUnwiredDebtDispatch(args)
	case "qa-process-debt-dispatch":
		cmdQAProcessDebtDispatch(args)
	case "mode-debt-dispatch":
		// The permission-regime fan-out (#4416, epic #4397): consume the sibling
		// mode-debt scorecard and file one deduped issue per HARD un-lifted dial,
		// routed to the permission-regime backlog (#2389 / #2405).
		cmdModeDebtDispatch(args)
	case "checkpoint-scorecard":
		cmdCheckpointScorecard(args)
	case "checkpoint-debt-dispatch":
		cmdCheckpointDebtDispatch(args)
	case "antipattern-scorecard":
		// The unifying work-loss card (docs/notes/AGENTIC-DEV-ANTIPATTERNS-2026-07-02.md
		// spine): folds REDUNDANT_REWORK + UNWIRED_PKG + ORPHAN_FUNC into one antipattern_debt.
		cmdAntipatternScorecard(args)
	case "debt-lanes":
		cmdDebtLanes(args)
	case "maturity":
		cmdMaturity(args)
	case "balance":
		// The night-balance readout (#3128): resume recovery-vs-stranding and
		// gardening-vs-throughput folded side by side; exit non-zero on an
		// underwater recovery budget so a night gate reading $? sees it.
		cmdBalance(args)
	case "token-defaults-scorecard":
		cmdTokenDefaultsScorecard(args)
	case "skill-effectiveness-scorecard":
		cmdSkillEffectivenessScorecard(args)
	case "mcp-filter-proof":
		os.Exit(runMCPFilterProof(os.Stdout, os.Stderr, args))
	case "footprint":
		// The always-sent MCP tool-schema floor scorecard (epic #3229, #3230):
		// price fak's registered tools/list floor offline via internal/mcpfootprint.
		os.Exit(runMCPFootprint(os.Stdout, os.Stderr, args))
	case "skill":
		// The queried skill loader operator surface (epic #1103, C7 / #1110):
		// `fak skill query|residency|swap` over .claude/skills (+ MCP resolver).
		cmdSkill(args)
	case "conflation-scorecard":
		cmdConflationScorecard(args)
	case "test-quality":
		cmdTestQuality(args)
	case "quality":
		// The missing-middle quality ladder spine (epic #4509): `fak quality run|explain`
		// runs one versioned case through a reference path and an engine path, applies a
		// deterministic comparator + a rubric scorer, and emits a machine-readable result
		// with a replayable failure bundle localized to the first divergence.
		cmdQuality(args)
	case "score":
		// Parent verb grouping the meta-scorecards / RSI loops (#1505): `fak score <name>` routes
		// to the same handler each legacy top-level *-scorecard/*-score/*-rsi verb ran (see
		// score.go). Behavior-preserving; the legacy verbs stay wired below as thin aliases.
		cmdScore(args)
	case "scorecard":
		cmdScorecardPane(args)
	case "performance-rsi-scorecard":
		cmdPerformanceRSIScorecard(args)
	case "repo-hygiene-scorecard":
		cmdRepoHygieneScorecard(args)
	case "ui-quality-scorecard":
		cmdUIQualityScore(args)
	case "scoreboard":
		cmdScoreboard(args)
	case "steering":
		cmdSteering(args)
	case "steer":
		cmdSteer(args)
	case "blockers":
		cmdBlockers(args)
	case "product":
		cmdProduct(args)
	case "product-scorecard":
		os.Exit(runProductScorecard(os.Stdout, os.Stderr, args))
	case "grafana":
		cmdGrafana(args)
	case "cachevalue":
		cmdCachevalue(args)
	case "cachesweep":
		os.Exit(runCachesweep(os.Stdout, os.Stderr, args))
	case "savings":
		cmdSavings(args)
	case "marketing":
		cmdMarketing(args)
	case "news":
		cmdNews(args)
	case "nodeusage":
		cmdNodeUsage(args)
	case "callavoid":
		cmdCallavoid(args)
	case "turnavoid":
		cmdTurnavoid(args)
	case "savings-vector":
		cmdSavingsVector(args)
	case "horizon-recovery":
		cmdHorizonRecovery(args)
	case "dogfood-issues":
		cmdDogfoodIssues(args)
	case "study":
		// Local content-addressed source-to-decision receipts.
		os.Exit(runStudy(os.Stdout, os.Stderr, args))
	case "complain":
		cmdComplain(args)
	case "learning-debt-dispatch":
		cmdLearningDebtDispatch(args)
	case "code-debt":
		cmdCodeDebt(args)
	case "cloud-handoff", "cloudhandoff":
		cmdCloudHandoff(args)
	case "harness-debt-dispatch":
		// The harness-strength fan-out (#4414, epic #4396, under self-ablation #607 /
		// open ablation registry #2828): consume the sibling model-strength
		// classifier's --json verdict and file one deduped deletion issue per HARD
		// scaffold graded REDUNDANT or HOBBLING (LOAD_BEARING files nothing).
		cmdHarnessDebtDispatch(args)
	case "stopfailure":
		cmdStopFailure(args)
	case "cluster":
		cmdCluster(args)
	case "leaseref":
		cmdLeaseref(args)
	case "intent":
		cmdIntent(args)
	case "memgate":
		cmdMemgate(args)
	case "memory-read":
		cmdMemoryRead(args)
	case "memory-stability-governor":
		cmdMemoryStabilityGovernor(args)
	case "node":
		cmdNode(args)
	case "node-compare":
		cmdNodeCompare(args)
	case "qwen36-node-reports":
		cmdQwen36NodeReports(args)
	case "qwen36-parity-witness-gate":
		cmdQwen36ParityWitnessGate(args)
	case "lab":
		cmdLab(args)
	case "fleet-trend":
		cmdFleetTrend(args)
	case "popularization-tickets":
		cmdPopularizationTickets(args)
	case "dormancy":
		cmdDormancy(args)
	case "version", "-v", "--version":
		cmdVersion(os.Stdout)
	case "-h", "--help", "help":
		cmdHelp(args)
	default:
		return false
	}
	return true
}

func dispatchPrimaryVerb(name string, args []string, start time.Time, verb *string) bool {
	switch name {
	case "run":
		cmdRun(args)
	case "launch":
		cmdLaunch(args)
	case "tool-width":
		cmdToolWidth(args)
	case "replay":
		// Explicit, unambiguous spelling of the trace-replay path (`fak run --trace`).
		cmdRunTrace(args)
	case "commit":
		*verb = gitOperationName(*verb, args)
		os.Exit(runObservedGitOperation(start, *verb, args, func() int {
			return runCommitCommand(os.Stdout, os.Stderr, args)
		}))
	case "edit-tx":
		cmdEditTx(args)
	case "sweep":
		*verb = gitOperationName(*verb, args)
		os.Exit(runObservedGitOperation(start, *verb, args, func() int {
			return runSweep(os.Stdout, os.Stderr, args)
		}))
	case "sync":
		*verb = gitOperationName(*verb, args)
		os.Exit(runObservedGitOperation(start, *verb, args, func() int {
			return runSync(os.Stdout, os.Stderr, args)
		}))
	case "merge":
		cmdMerge(args)
	case "affected":
		cmdAffected(args)

	case "go":
		cmdGoShim(args)
	case "worktree":
		cmdWorktreeVerb(args)
	case "wip":
		cmdWip(args)
	case "tree-doctor":
		cmdTreeDoctor(args)
	case "storage-pressure":
		cmdStoragePressure(args)
	case "git-maint":
		cmdGitMaint(args)
	case "git-daily":
		cmdGitDaily(args)
	case "gitd":
		cmdGitd(args)
	case "clean-bins":
		cmdCleanBins(args)
	case "self-update":
		cmdSelfUpdate(args)
	case "preflight":
		cmdPreflight(args)
	case "validate":
		cmdValidate(args)
	case "llms-full":
		cmdLLMSFull(args)
	case "attest":
		cmdAttest(args)
	case "speed-ab":
		cmdSpeedAB(os.Args[2:])
	case "model-observe":
		cmdModelObserve(args)
	case "bench":
		cmdBench(args)
	case "benchmarks":
		cmdBenchmarks(args)
	case "native-benchmarks":
		cmdNativeBenchmarks(args)
	case "native-performance":
		cmdNativePerformance(args)
	case "quantbench":
		cmdQuantbench(args)
	case "bitnetruntime":
		cmdBitnetRuntime(args)
	case "frontierswe":
		cmdFrontierswe(args)
	case "sota":
		cmdSota(args)
	case "kvbm":
		cmdKVBM(args)
	case "sota-coverage-scorecard":
		cmdSOTACoverageScorecard(args)
	case "bench-runs":
		cmdBenchRuns(args)
	case "bench-loop", "benchloop":
		cmdBenchLoop(args)
	case "bench-ingest":
		cmdBenchIngest(args)
	case "ablate":
		cmdAblate(args)
	case "ablate-arm":
		// Hidden internal seam: the ablate arm-mode re-exec child (see cmdAblateArm).
		cmdAblateArm(args)
	case "turntax":
		if len(args) > 0 && args[0] == "visual" {
			cmdTurnTaxVisual(args[1:])
		} else {
			cmdTurnTax(args)
		}
	case "hooklat":
		cmdHookLat(args)
	case "dispatchlat":
		cmdDispatchLat(args)
	default:
		return false
	}
	return true
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
	verbFlagUsage(fs, "run")
	trace := fs.String("trace", "", "path to a trace JSON file")
	engineID := fs.String("engine", "mock", "engine id (default: mock; inkernel: the explicit fak-native model path; cassette)")
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
	verbFlagUsage(fs, "preflight")
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
		case "gitspawn":
			// #5620: git process spawns per unit of work on the three hot paths.
			os.Exit(runGitSpawnBench(os.Stdout, os.Stderr, argv[1:]))
		case "system-baseline":
			os.Exit(runBenchSystemBaseline(os.Stdout, os.Stderr, argv[1:]))
		case "local":
			os.Exit(runBenchLocal(os.Stdout, os.Stderr, argv[1:]))
		case "tb4":
			os.Exit(runBenchTB4(os.Stdout, os.Stderr, argv[1:]))
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

func jsonIndent(v any) []byte {
	b, _ := json.MarshalIndent(v, "", "  ")
	return b
}

// applyPolicy loads a capability-floor manifest and swaps it into the registered
// adjudicator before the kernel runs. Empty path = keep the built-in
// DefaultPolicy. A bad manifest is fatal: a misconfigured floor must fail loudly
// at startup, never silently fall back to a more permissive default.
// policyReloadMu serializes the complete multi-singleton floor swap.
var policyReloadMu sync.Mutex

func applyPolicy(path string) {
	if path == "" {
		return
	}
	policyReloadMu.Lock()
	_, _, err := loadAndApplyPolicyLocked(path, false)
	policyReloadMu.Unlock()
	must(err)
	fmt.Fprintf(os.Stderr, "fak: loaded capability floor from %s\n", path)
}

func reloadPolicy(path string) (policy.Runtime, string, error) {
	policyReloadMu.Lock()
	defer policyReloadMu.Unlock()
	return loadAndApplyPolicyLocked(path, true)
}

func reloadPolicyWithPrior(path string) (policy.Runtime, adjudicator.Policy, string, error) {
	policyReloadMu.Lock()
	defer policyReloadMu.Unlock()
	prior := adjudicator.Default.PolicySnapshot()
	rt, warning, err := loadAndApplyPolicyLocked(path, true)
	return rt, prior, warning, err
}

func loadAndApplyPolicyLocked(path string, enforceWideningGate bool) (policy.Runtime, string, error) {
	if path == "" {
		return policy.Runtime{}, "", errors.New("policy reload requires --policy FILE")
	}
	rt, err := policy.LoadRuntime(path)
	if err != nil {
		// A rejected floor swap is exactly what an auditor asks about (an operator
		// trying to widen or break the capability floor with a malformed edit), so
		// record the refused attempt — with the digest of whatever bytes we could
		// read — before returning. journal.Active() is nil on an unjournaled run and
		// AppendConfigSwap no-ops, keeping that run byte-identical.
		journal.Active().AppendConfigSwap(journal.ConfigSwapFloor, path, configFileDigest(path), journal.ConfigSwapRejected, err.Error())
		return policy.Runtime{}, "", err
	}
	// Re-apply the operator overlays before comparing effective floors. This keeps
	// the gate from mistaking a persisted always-allow entry for a manifest widening.
	denyPath := guardDenyOverlayPath()
	overlayWarning := ""
	if ov, _, ovErr := loadGuardAllowOverlayLayers(); ovErr == nil {
		guardApplyAllowOverlay(&rt, ov)
	} else {
		overlayWarning = "overlay_error: " + ovErr.Error()
	}
	applyLaunchToolGrant(&rt)
	if ov, ovErr := loadGuardDenyOverlay(denyPath); ovErr == nil {
		guardApplyDenyOverlay(&rt, ov)
	} else if overlayWarning == "" {
		overlayWarning = "deny_overlay_error: " + ovErr.Error()
	} else {
		overlayWarning += "\ndeny_overlay_error: " + ovErr.Error()
	}
	rt = protectGuardPolicyConfig(rt, append(guardAllowOverlayLayerPaths(), denyPath, path)...)

	digest := configFileDigest(path)
	overlayWarning, err = applyPolicyRuntimeLocked(rt, path, digest, overlayWarning, enforceWideningGate)
	if err != nil {
		return policy.Runtime{}, "", err
	}
	return rt, overlayWarning, nil
}

// configFileDigest returns the sha256 (as "sha256:<hex>") of a config manifest's
// bytes, for the over-which-bytes field of a CONFIG_SWAP audit row. It reads the
// file fresh — the swap installs whatever is on disk now — and yields "" on an
// unreadable file, so a rejected swap still records the attempt, just without a
// digest.
func configFileDigest(path string) string {
	b, err := os.ReadFile(path)
	if err != nil {
		return ""
	}
	sum := sha256.Sum256(b)
	return "sha256:" + hex.EncodeToString(sum[:])
}

func policyReloader(path string) gateway.PolicyReloadFunc {
	if path == "" {
		return nil
	}
	return func(context.Context) (gateway.PolicyReloadResponse, error) {
		rt, prior, overlayWarning, err := reloadPolicyWithPrior(path)
		if err != nil {
			return gateway.PolicyReloadResponse{}, err
		}
		summary := policy.SummaryRuntime(rt)
		if overlayWarning != "" {
			fmt.Fprintln(os.Stderr, "fak policy reload warning:", overlayWarning)
			summary += "\n" + overlayWarning
		}
		policyBytes, readErr := os.ReadFile(path)
		if readErr != nil {
			return gateway.PolicyReloadResponse{}, readErr
		}
		rollback := func() {
			policyReloadMu.Lock()
			defer policyReloadMu.Unlock()
			adjudicator.Default.SetPolicy(prior)
		}
		return gateway.PolicyReloadResponse{Reloaded: true, Source: path, Summary: summary, EffectiveDigest: effectiveDigestWithLaunchGrant(policyBytes), Rollback: rollback}, nil
	}
}

// guardPolicyReloader is the reloader `fak guard` wires (guard.go). For an explicit
// --policy FILE it is exactly policyReloader (the file-reload path, unchanged). For the
// default built-in floor (empty path) it returns a working reloader that re-derives the
// embedded floor + operator allow overlay — instead of policyReloader's nil, which had
// disabled POST /v1/fak/policy/reload (404) on the most common guard launch (#3957).
// `fak serve` keeps calling policyReloader directly, so its non-guard no-`--policy`
// default stays byte-identical, as does `--policy` everywhere.
func guardPolicyReloader(policyPath string) gateway.PolicyReloadFunc {
	if strings.TrimSpace(policyPath) != "" {
		return policyReloader(policyPath)
	}
	return func(context.Context) (gateway.PolicyReloadResponse, error) {
		rt, overlayWarning, err := guardReloadDefaultFloor()
		if err != nil {
			return gateway.PolicyReloadResponse{}, err
		}
		summary := policy.SummaryRuntime(rt)
		if overlayWarning != "" {
			fmt.Fprintln(os.Stderr, "fak guard reload warning:", overlayWarning)
			summary += "\n" + overlayWarning
		}
		return gateway.PolicyReloadResponse{Reloaded: true, Source: launchGrantSource("built-in guard floor + operator allow overlay"), Summary: summary, EffectiveDigest: effectiveDigestWithLaunchGrant(guardDefaultPolicyJSON)}, nil
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
