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
	"github.com/anthony-chaudhary/fak/internal/journal"
	"github.com/anthony-chaudhary/fak/internal/kernel"
	"github.com/anthony-chaudhary/fak/internal/modelroute"
	"github.com/anthony-chaudhary/fak/internal/policy"
	"github.com/anthony-chaudhary/fak/internal/session"
	"github.com/anthony-chaudhary/fak/internal/sessionreset"
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
	switch os.Args[1] {
	case "run":
		cmdRun(os.Args[2:])
	case "replay":
		// Explicit, unambiguous spelling of the trace-replay path (`fak run --trace`).
		cmdRunTrace(os.Args[2:])
	case "commit":
		verb = gitOperationName(verb, argv)
		os.Exit(runObservedGitOperation(start, verb, argv, func() int {
			return runCommitCommand(os.Stdout, os.Stderr, argv)
		}))
	case "edit-tx":
		cmdEditTx(os.Args[2:])
	case "sweep":
		verb = gitOperationName(verb, argv)
		os.Exit(runObservedGitOperation(start, verb, argv, func() int {
			return runSweep(os.Stdout, os.Stderr, argv)
		}))
	case "sync":
		verb = gitOperationName(verb, argv)
		os.Exit(runObservedGitOperation(start, verb, argv, func() int {
			return runSync(os.Stdout, os.Stderr, argv)
		}))
	case "merge":
		cmdMerge(os.Args[2:])
	case "whats-changed":
		cmdWhatsChanged(os.Args[2:])
	case "affected":
		cmdAffected(os.Args[2:])
	case "blast":
		cmdBlast(os.Args[2:])
	case "buildcheck":
		cmdBuildCheck(os.Args[2:])
	case "go":
		cmdGoShim(os.Args[2:])
	case "worktree":
		cmdWorktreeVerb(os.Args[2:])
	case "wip":
		cmdWip(os.Args[2:])
	case "preflight":
		cmdPreflight(os.Args[2:])
	case "ci-preflight":
		cmdCIPreflight(os.Args[2:])
	case "attest":
		cmdAttest(os.Args[2:])
	case "backend":
		cmdBackend(os.Args[2:])
	case "bench":
		cmdBench(os.Args[2:])
	case "benchmarks":
		cmdBenchmarks(os.Args[2:])
	case "frontierswe":
		cmdFrontierswe(os.Args[2:])
	case "sota":
		cmdSota(os.Args[2:])
	case "kvbm":
		cmdKVBM(os.Args[2:])
	case "sota-coverage-scorecard":
		cmdSOTACoverageScorecard(os.Args[2:])
	case "bench-runs":
		cmdBenchRuns(os.Args[2:])
	case "bench-loop", "benchloop":
		cmdBenchLoop(os.Args[2:])
	case "bench-ingest":
		cmdBenchIngest(os.Args[2:])
	case "amd-gpu-facts":
		cmdAMDGPUFacts(os.Args[2:])
	case "commit-subject-coverage":
		cmdCommitSubjectCoverage(os.Args[2:])
	case "ablate":
		cmdAblate(os.Args[2:])
	case "ablate-arm":
		// Hidden internal seam: the ablate arm-mode re-exec child (see cmdAblateArm).
		cmdAblateArm(os.Args[2:])
	case "turntax":
		cmdTurnTax(os.Args[2:])
	case "hooklat":
		cmdHookLat(os.Args[2:])
	case "dispatchlat":
		cmdDispatchLat(os.Args[2:])
	case "agent":
		cmdAgent(os.Args[2:])
	case "api-host":
		cmdAPIHost(os.Args[2:])
	case "question-ledger":
		cmdQuestionLedger(os.Args[2:])
	case "trunk-build-probe":
		cmdTrunkBuildProbe(os.Args[2:])
	case "godsplit-plan":
		cmdGodsplitPlan(os.Args[2:])
	case "refactor-verify":
		cmdRefactorVerify(os.Args[2:])
	case "glm52-prefill-sweep":
		cmdGLM52PrefillSweep(os.Args[2:])
	case "recall":
		cmdRecall(os.Args[2:])
	case "recover":
		cmdRecover(os.Args[2:])
	case "concept":
		os.Exit(runConcept(os.Stdout, os.Stderr, os.Args[2:]))
	case "rename-concept":
		cmdRenameConcept(os.Args[2:])
	case "session":
		cmdSession(os.Args[2:])
	case "session-audit":
		cmdSessionAudit(os.Args[2:])
	case "sessionjournal":
		cmdSessionJournal(os.Args[2:])
	case "tier-calibrate":
		cmdTierCalibrate(os.Args[2:])
	case "resume":
		cmdResume(os.Args[2:])
	case "dispatch":
		cmdDispatch(os.Args[2:])
	case "dispatch-conservation":
		cmdDispatchConservation(os.Args[2:])
	case "dispatch-aging":
		cmdDispatchAging(os.Args[2:])
	case "knownbad":
		cmdKnownBad(os.Args[2:])
	case "process-guard":
		cmdProcessGuard(os.Args[2:])
	case "windowgate":
		cmdWindowgate(os.Args[2:])
	case "conpty":
		cmdConPTY(os.Args[2:])
	case "ps":
		cmdPS(os.Args[2:])
	case "top":
		cmdTop(os.Args[2:])
	case "signal":
		cmdSignal(os.Args[2:])
	case "task":
		cmdTask(os.Args[2:])
	case "tasks":
		cmdTasks(os.Args[2:])
	case "toolproc":
		cmdToolproc(os.Args[2:])
	case "stallscan":
		cmdStallscan(os.Args[2:])
	case "host-crash":
		cmdHostCrash(os.Args[2:])
	case "host-relaunch-broker":
		os.Exit(runHostRelaunchBroker(os.Stdout, os.Stderr, os.Args[2:]))
	case "schedscan":
		cmdSchedScan(os.Args[2:])
	case "growthgate":
		cmdGrowthgate(os.Args[2:])
	case "test":
		cmdTest(os.Args[2:])
	case "done":
		cmdDone(os.Args[2:])
	case "profile":
		cmdProfile(os.Args[2:])
	case "multisubmit":
		cmdMultiSubmit(os.Args[2:])
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
	case "claude-mac-fak", "mac":
		// `fak mac` is the crisp, memorable handle; `fak claude-mac-fak` is the long
		// form kept working byte-for-byte. Both spellings route to the one handler.
		cmdClaudeMacFak(os.Args[2:])
	case "macbench":
		cmdMacBench(os.Args[2:])
	case "macfit":
		cmdMacFit(os.Args[2:])
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
	case "waiting":
		cmdWaiting(os.Args[2:])
	case "cron":
		cmdCron(os.Args[2:])
	case "logvault":
		cmdLogvault(os.Args[2:])
	case "snapshot":
		cmdSnapshot(os.Args[2:])
	case "traj":
		cmdTraj(os.Args[2:])
	case "trajctl":
		cmdTrajctl(os.Args[2:])
	case "shadowgit":
		cmdShadowGit(os.Args[2:])
	case "signals":
		cmdSignals(os.Args[2:])
	case "trajquery":
		cmdTrajQuery(os.Args[2:])
	case "dup":
		cmdDup(os.Args[2:])
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
	case "eve":
		// The Eve integration bridge (#2600): mechanical security preflight over
		// eve's MCP/OpenAPI connections (#2602). See cmd/fak/eve.go.
		cmdEve(os.Args[2:])
	case "doomloop":
		// The two-axis doom-loop guard: classifies live workers on effort vs
		// verified progress and wires the reversible-first correction. See
		// cmd/fak/doomloop.go and internal/doomloop.
		cmdDoomloop(os.Args[2:])
	case "lint":
		cmdLint(os.Args[2:])
	case "codelint":
		cmdCodelint(os.Args[2:])
	case "tool-coverage-audit":
		cmdToolCoverageAudit(os.Args[2:])
	case "answer-shape":
		cmdAnswerShape(os.Args[2:])
	case "negate":
		// The negation operator (#4461/#4472, negframe L2): `fak negate detect|resolve|reframe`
		// exposes detect (Classify), resolve (positive-complement over the L2 registry), and
		// reframe (emit-time Reframe) as one callable primitive over internal/negframe.
		cmdNegate(os.Args[2:])
	case "claim-check":
		cmdClaimCheck(os.Args[2:])
	case "headless-lint":
		cmdHeadlessLint(os.Args[2:])
	case "hwgate-lint":
		cmdHwGateLint(os.Args[2:])
	case "check-tool-failure":
		cmdCheckToolFailure(os.Args[2:])
	case "doctor":
		cmdDoctor(os.Args[2:])
	case "init":
		// Emit a minimal, valid fak.toml deployment manifest (#3421).
		cmdInit(os.Args[2:])
	case "feature":
		cmdFeature(os.Args[2:])
	case "index", "devindex":
		cmdIndex(os.Args[2:])
	case "orient":
		cmdOrient(os.Args[2:])
	case "wiki":
		cmdWiki(os.Args[2:])
	case "workflow":
		cmdWorkflow(os.Args[2:])
	case "workflow-audit":
		cmdWorkflowAudit(os.Args[2:])
	case "tree-doctor":
		cmdTreeDoctor(os.Args[2:])
	case "git-maint":
		cmdGitMaint(os.Args[2:])
	case "clean-bins":
		cmdCleanBins(os.Args[2:])
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
	case "watchdog":
		cmdWatchdog(os.Args[2:])
	case "service":
		cmdService(os.Args[2:])
	case "micro":
		// The native in-process Go microagent runtime front door (see cmdMicro).
		cmdMicro(os.Args[2:])
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
	case "demo":
		// The zero-flag 60-second proof: fak's offline scenario through the REAL kernel,
		// one live verdict per call class (see cmdDemo). No agent, no key.
		cmdDemo(os.Args[2:])
	case "guard-precompact":
		// Hidden: Claude Code PreCompact hook actuator installed by `fak guard`.
		cmdGuardPreCompact(os.Args[2:])
	case "guard-stophook":
		// Hidden: Claude Code Stop hook actuator installed by `fak guard` (see
		// cmdGuardStopHook: blocks an unchosen end_turn past a fully-refused turn).
		cmdGuardStopHook(os.Args[2:])
	case "guard-commit-gate":
		// Hidden: Claude Code PreToolUse boundary for git commit stamp/path binding.
		cmdGuardCommitGate(os.Args[2:])
	case "guard-sessionstart":
		// Hidden: Claude Code SessionStart hook actuator installed by `fak guard` (#3092).
		cmdGuardSessionStart(os.Args[2:])
	case "guard-stops":
		// Operator-facing: fold the typed Stop-hook decision ledger into a tally (clean
		// stops, bounded stand-downs, and the fail-open stops that are otherwise invisible).
		cmdGuardStops(os.Args[2:])
	case "guard-stops-slack":
		// Durable update-in-place scoreboard feeder for the Stop decision ledger.
		cmdGuardStopsSlack(os.Args[2:])
	case "trunk-red":
		// Operator-facing: fold the pre-existing trunk-red witness ledger into the distinct
		// shared breaks the build gates admitted over (a peer's red, not yours) — worst (most
		// clones stuck) first, so a break the whole fleet inherits gets fixed at its source.
		cmdTrunkRed(os.Args[2:])
	case guard.TrampolineVerb:
		// Hidden internal seam: the Landlock hook-floor re-exec trampoline, Linux (see
		// guard.LandlockTrampoline).
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
	case "idempotency":
		cmdIdempotency(os.Args[2:])
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
	case "deepseekbench":
		cmdDeepSeekBench(os.Args[2:])
	case "assume":
		cmdAssume(os.Args[2:])
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
	case "project":
		// The ProjectsV2 board control-pane fold — makes the board an operator-visible
		// dimension (report/verdict/Slack-ready) instead of a write-only sync target.
		cmdProject(os.Args[2:])
	case "mlp-score":
		// The witnessed grade of the "first lovable cut" for the all-in-one agent
		// runtime epic (#3256, milestone #17): per-criterion PASS/not-yet (#3284).
		cmdMLPScore(os.Args[2:])
	case "program":
		cmdProgram(os.Args[2:])
	case "rollup":
		cmdRollup(os.Args[2:])
	case "spend":
		cmdSpend(os.Args[2:])
	case "budget":
		cmdBudget(os.Args[2:])
	case "sidecar":
		cmdSidecar(os.Args[2:])
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
	case "conformance":
		cmdConformance(os.Args[2:])
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
	case "unwired-scorecard":
		cmdUnwiredScorecard(os.Args[2:])
	case "unwired-debt-dispatch":
		cmdUnwiredDebtDispatch(os.Args[2:])
	case "qa-process-debt-dispatch":
		cmdQAProcessDebtDispatch(os.Args[2:])
	case "mode-debt-dispatch":
		// The permission-regime fan-out (#4416, epic #4397): consume the sibling
		// mode-debt scorecard and file one deduped issue per HARD un-lifted dial,
		// routed to the permission-regime backlog (#2389 / #2405).
		cmdModeDebtDispatch(os.Args[2:])
	case "checkpoint-scorecard":
		cmdCheckpointScorecard(os.Args[2:])
	case "checkpoint-debt-dispatch":
		cmdCheckpointDebtDispatch(os.Args[2:])
	case "antipattern-scorecard":
		// The unifying work-loss card (docs/notes/AGENTIC-DEV-ANTIPATTERNS-2026-07-02.md
		// spine): folds REDUNDANT_REWORK + UNWIRED_PKG + ORPHAN_FUNC into one antipattern_debt.
		cmdAntipatternScorecard(os.Args[2:])
	case "maturity":
		cmdMaturity(os.Args[2:])
	case "balance":
		// The night-balance readout (#3128): resume recovery-vs-stranding and
		// gardening-vs-throughput folded side by side; exit non-zero on an
		// underwater recovery budget so a night gate reading $? sees it.
		cmdBalance(os.Args[2:])
	case "token-defaults-scorecard":
		cmdTokenDefaultsScorecard(os.Args[2:])
	case "skill-effectiveness-scorecard":
		cmdSkillEffectivenessScorecard(os.Args[2:])
	case "footprint":
		// The always-sent MCP tool-schema floor scorecard (epic #3229, #3230):
		// price fak's registered tools/list floor offline via internal/mcpfootprint.
		os.Exit(runMCPFootprint(os.Stdout, os.Stderr, os.Args[2:]))
	case "skill":
		// The queried skill loader operator surface (epic #1103, C7 / #1110):
		// `fak skill query|residency|swap` over .claude/skills (+ MCP resolver).
		cmdSkill(os.Args[2:])
	case "conflation-scorecard":
		cmdConflationScorecard(os.Args[2:])
	case "quality":
		// The missing-middle quality ladder spine (epic #4509): `fak quality run|explain`
		// runs one versioned case through a reference path and an engine path, applies a
		// deterministic comparator + a rubric scorer, and emits a machine-readable result
		// with a replayable failure bundle localized to the first divergence.
		cmdQuality(os.Args[2:])
	case "score":
		// Parent verb grouping the meta-scorecards / RSI loops (#1505): `fak score <name>` routes
		// to the same handler each legacy top-level *-scorecard/*-score/*-rsi verb ran (see
		// score.go). Behavior-preserving; the legacy verbs stay wired below as thin aliases.
		cmdScore(os.Args[2:])
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
	case "cachesweep":
		os.Exit(runCachesweep(os.Stdout, os.Stderr, os.Args[2:]))
	case "savings":
		cmdSavings(os.Args[2:])
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
		if s := suggestVerbSpelling(os.Args[1]); s != "" {
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
	verbFlagUsage(fs, "run")
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
	routeManifest := fs.String("route-manifest", "", "model-routing policy to install for the fak arm; each tool call is classified and a single-model PICK binds abi.ToolCall.Engine before kernel submit")
	_ = fs.Parse(argv)
	applyPolicy(*policyPath)
	loadedRoute, runOpts, err := loadAgentRouteOptions(*routeManifest)
	must(err)
	if loadedRoute != nil {
		fmt.Fprintf(os.Stderr, "fak agent: loaded model-routing policy from %s\n", *routeManifest)
	}

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

	res, trace, err := agent.Run(ctx(), planner, *task, *maxTurns, runOpts...)
	must(err)

	must(os.WriteFile(*out, jsonIndent(res), 0o644))
	if *logOut != "" {
		_ = os.WriteFile(*logOut, agent.RenderTrace(trace), 0o644)
	}
	agent.PrintReport(os.Stdout, res, trace, *out)
}

func loadAgentRouteOptions(path string) (*modelroute.Manifest, []agent.RunOption, error) {
	path = strings.TrimSpace(path)
	if path == "" {
		return nil, nil, nil
	}
	manifest, err := modelroute.LoadManifest(path)
	if err != nil {
		return nil, nil, fmt.Errorf("fak agent: --route-manifest: %w", err)
	}
	return &manifest, []agent.RunOption{agent.WithRouteManifest(&manifest)}, nil
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
	_, _, err := reloadPolicy(path)
	must(err)
	fmt.Fprintf(os.Stderr, "fak: loaded capability floor from %s\n", path)
}

func reloadPolicy(path string) (policy.Runtime, string, error) {
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
	// Re-apply the operator allow overlay on hot-reload so a `--policy` swap never
	// silently drops the out-of-band always-allow list (`fak guard allow`). A missing
	// overlay is the common no-op; a malformed one is tolerated on reload rather than
	// wedging a live gateway (the loud failure already fired at launch).
	denyPath := guardDenyOverlayPath()
	overlayWarning := ""
	if ov, _, ovErr := loadGuardAllowOverlayLayers(); ovErr == nil {
		guardApplyAllowOverlay(&rt, ov)
	} else {
		overlayWarning = "overlay_error: " + ovErr.Error()
	}
	if ov, ovErr := loadGuardDenyOverlay(denyPath); ovErr == nil {
		guardApplyDenyOverlay(&rt, ov)
	} else if overlayWarning == "" {
		overlayWarning = "deny_overlay_error: " + ovErr.Error()
	} else {
		overlayWarning += "\ndeny_overlay_error: " + ovErr.Error()
	}
	rt = protectGuardPolicyConfig(rt, append(guardAllowOverlayLayerPaths(), denyPath, path)...)
	adjudicator.Default.SetPolicy(rt.Adjudicator)
	applyRuntime(rt)
	// Record the now-live capability floor as a durable CONFIG_SWAP row (#3959): the
	// security boundary just changed, so an auditor must be able to see which bytes
	// (source path + sha256) became authoritative and when.
	journal.Active().AppendConfigSwap(journal.ConfigSwapFloor, path, configFileDigest(path), journal.ConfigSwapOK, "")
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
		rt, overlayWarning, err := reloadPolicy(path)
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
		return gateway.PolicyReloadResponse{Reloaded: true, Source: path, Summary: summary, EffectiveDigest: guardCurrentEffectivePolicyDigest(policyBytes)}, nil
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
		return gateway.PolicyReloadResponse{Reloaded: true, Source: "built-in guard floor + operator allow overlay", Summary: summary, EffectiveDigest: guardCurrentEffectivePolicyDigest(guardDefaultPolicyJSON)}, nil
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
		rearmTokens := freshContextTokens
		if st.Budget.ContextTokensCap > 0 {
			rearmTokens = st.Budget.ContextTokensCap
		}
		resetInput := sessionreset.Input{
			Trace:          traceID,
			Messages:       resetMsgs(messages),
			FreshBudgetTok: rearmTokens,
		}
		seed := sessionreset.BuildSeed(resetInput)
		if strings.TrimSpace(seed.Recap) == "" {
			return "", nil, false
		}
		resetTx := sessionreset.BuildResetTransaction(resetInput, child, seed)
		childState := serveSessions.RecontinueWithTransaction(traceID, child, session.Budget{
			TurnsLeft:         session.Unbounded,
			TokensLeft:        session.Unbounded,
			ContextTokensLeft: rearmTokens,
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
func steerSession(ctx context.Context, traceID, principal, text string) error {
	traceID = strings.TrimSpace(traceID)
	if traceID == "" {
		return errors.New("trace_id is required")
	}
	// `from` is the attribution of the steer source, not a trust grant: the a2achan floor
	// gates on caps+taint+scope (Shared ⇒ Tainted/ScopeFleet), never on the from-string, so
	// naming a machine principal (e.g. "doomloop-guard", #3529) truthfully labels the source
	// without buying it more authority than an operator steer. Empty ⇒ the human default.
	from := strings.TrimSpace(principal)
	if from == "" {
		from = "operator"
	}
	key := a2achan.ChannelKey{Locale: a2achan.Session, ID: traceID}
	v := a2achan.Default.Send(ctx, from, key, a2achan.Shared([]byte(text)), a2achan.CapA2ASend)
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
			return session.State{}, false, fmt.Errorf("unknown run-state %q (want running|throttled|paused|draining|terminating|stopped)", req.Run)
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
	case session.Throttled, session.Stopped, session.Terminating:
		return reason
	case session.Running:
		return ""
	}
	return ""
}
