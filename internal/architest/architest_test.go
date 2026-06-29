package architest

import (
	"go/ast"
	"go/build"
	"go/parser"
	"go/token"
	"io/fs"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"testing"
)

// modPrefix is the import-path prefix for every internal leaf.
const modPrefix = "github.com/anthony-chaudhary/fak/internal/"

// tier assigns every internal package a layer. THE import rule (TestNoUpwardImports):
// a package may import only packages whose tier is <= its own. Upward imports are the
// layer inversion that turns the DAG into spaghetti; Go already forbids cycles, so the
// two together pin the graph to a clean layered DAG.
//
// The tiers were set AT REALITY on 2026-06-17 from `go list` of the real tree: every
// one of the ~30 current cross-package edges satisfies the rule (proven by
// TestNoUpwardImports passing). The gate is deliberately seeded at the status quo so it
// has ZERO false positives today; it is tightened over time, never loosened to admit a
// new violation.
//
//	0 root        the frozen ABI — the one tree everyone imports; imports nothing internal.
//	1 foundation  pure primitives: content store, canonicalizer, taint/provenance vocab,
//	              metrics, the model tensor runtime. Import only the root.
//	2 mechanism   single-purpose kernel mechanisms & engines: the adjudicator, context-MMU,
//	              vDSO, grammar, pre-flight, the kernel walker, the engine client, witness,
//	              plan-CFI, stewards, harvest, ship-gate, policy.
//	3 composer    security behaviors composed FROM mechanisms: IFC, normgate, recall, the
//	              KV-MMU, the context debugger, the AgentDojo harness.
//	4 integrator  top-level wiring: the agent harness, benches, the gateway, and the
//	              built-in driver list (registrations / defconfig).
//
// A NEW package MUST be added here (TestEveryPackageDeclaresTier fails otherwise) at the
// LOWEST tier whose role it fits. That forced choice is the review gate: it makes adding
// a leaf a conscious layering decision instead of an accident.
var tier = map[string]int{
	"abi": 0,

	"agenticbench":     1, // pure #868 artifact rollup gate over committed benchmark evidence; stdlib-only, off the hot path.
	"ailuminate":       1, // pure MLCommons-AILuminate benchmark-entry scoping/go-no-go contract (#1070); stdlib-only, off the hot path.
	"benchcatalog":     1, // pure benchmark registry used by fak benchmarks and scorecards; stdlib-only, off the hot path.
	"cachevalueledger": 1, // durable, append-only cache-value observation ledger for fak sessions; JSONL persistence over cacheobs stats.
	"benchcli":         1, // shared helpers the bench-CLI mains (cmd/*bench) had copy-pasted; imports model(1) only, off the hot path.
	"benchids":         1, // pure deterministic synthetic-token-ID generator for the bench mains (#776); stdlib-only, off the hot path.
	"benchscore":       1, // pure benchmark score artifact validator/renderer; stdlib-only, off the hot path.
	"callavoid":        1, // pure avoided-call economics/accounting primitive; stdlib-only, folded by higher layers.
	"accounts":         1, "appversion": 1, "blob": 1, "boundarylint": 1, "cachemeta": 1, "cacheobs": 1, "canon": 1, "compute": 1, "deletioncert": 1, "demoui": 1, "ggufload": 1, "gpulease": 1, "hfhub": 1, "intlist": 1, "leakcheck": 1, "metalgemm": 1, "metrics": 1, "model": 1, "pathlint": 1, "pathutil": 1, "provenance": 1, "swebench": 1, "urllint": 1, "webbench": 1,
	// stdlib-only foundation leaves (import nothing internal); off the hot path.
	"bgloop": 1, "binstamp": 1, "cachewitness": 1, "covmatrix": 1, "defaultvaluescore": 1, "dojocal": 1, "experiments": 1, "flock": 1, "guardtrace": 1, "maputil": 1, "mathx": 1, "newmodel": 1, "numfmt": 1, "selfinstall": 1,
	"supportmaturityscore": 1,                // pure scorecard over internal/covmatrix; off the hot path.
	"supportmaturity":      1,                // the closed M0–M7 support-maturity ladder vocabulary (#1244); lowers covmatrix(1)+ggufload(1)+compute(1) onto one ordered enum, off the hot path.
	"releasestale":         1,                // pure publish-staleness verdict (latest tag vs HEAD, in commits+days) + a thin git Gather shell; the publish-axis dual of binstamp's source-axis freshness. Stdlib-only, imports nothing internal, off the hot path.
	"affectedtests":        1,                // pure reverse-dependency closure for the `fak affected` fast test gate: changed packages + every package that (transitively, test imports included) imports one; stdlib-only, imports nothing internal, off the hot path.
	"modelladder":          2,                // model-ladder selector; imports benchcli(1)+model(1)+stdlib, off the hot path.
	"modelreg":             2,                // model registry; imports hfhub(1)+stdlib, off the hot path.
	"skillenv":             4,                // skill virtual-env composer; imports ctxmmu(2)+ctxresidency(3)+kvmmu(3)+stdlib.
	"guardroute":           4,                // guard RSI worst-bucket auto-router to a finding+gh issue; imports dogfoodissues(3)+guardrsi(1)+stdlib, off the hot path.
	"conflationscore":      1,                // pure Go port of tools/conflation_scorecard.py (provenance-honesty stick); stdlib-only, off the hot path.
	"scoreboard":           1,                // outbound Slack publisher for scorecard/score/run-event status posts; stdlib-only, off the hot path.
	"benchpost":            1,                // outbound Slack publisher for bench-channel rollups/run-requests; folds catalog/baseline/plan JSON, reuses scoreboard(1) transport, off the hot path.
	"blockerpost":          1,                // outbound Slack publisher for the central #blockers channel: severity-driven (background status vs surfaced operator page); reuses scoreboard(1) transport, off the hot path.
	"dispatchpost":         1,                // outbound Slack publisher for background code-dispatch run RESULTS; reuses scoreboard(1) transport, off the hot path.
	"dojopost":             1,                // outbound Slack publisher for dojo rollups/trends; folds dojo(1) reports, reuses scoreboard(1) transport, off the hot path.
	"marketing":            1,                // completion-driven marketing subsystem: witnessed-ship(hooks) -> claim/artifact, CLAIMS.md honesty gate, AEO/AgentEO refresh; imports hooks(1)+scoreboard(1)+stdlib, off the hot path.
	"fleet":                1,                // fleet-roster snapshot fold for the #node-usage feeder; stdlib-only, imports nothing internal, off the hot path.
	"nodeusagepost":        1,                // outbound Slack publisher for the #node-usage feeder; folds fleet(1), reuses scoreboard(1) transport, off the hot path.
	"blobfs":               1, "blobhttp": 1, // durable on-disk / remote-HTTP content-addressed Ref backends; attach to abi like blob (Resolver+PageOutBackend), import only abi+blob+stdlib.
	"xenginekv":  1, // cross-engine zero-copy KV co-residence arena (#448): a RefRegion-issuing Resolver+RegionBackend+PageOutBackend; attaches to abi like blob, imports only abi+blob+stdlib (FAK_XENGINE_KV-gated).
	"secretload": 1, // first-class secret/config loader (#887/#889): SecretSource priority list + os-env/encrypted-file/.env backends + Require checklist + Redact; imports canon(1)+stdlib, off the hot path.
	"windowgate": 1, // no-desktop-popup ratchet: scans tracked .ps1 task installers + window-suppressing .py for console-window flashes; stdlib-only, off the hot path.

	"adjudicator": 2, "ctxmmu": 2, "engine": 2, "enginecache": 2, "grammar": 2, "kernel": 2,
	"preflight": 2, "vdso": 2, "plancfi": 2, "steward": 2, "witness": 2,
	"harvest": 2, "shipgate": 2, "policy": 2, "modelengine": 2, "ratelimit": 2,
	"journal": 2, "gitgate": 2, "safecommit": 2, "spec": 2, // spec: the ProvisionalSink/OpsSpec speculation mechanism; composes model+polymodel under abi (off-defconfig, gated by FAK_POLYMODEL).
	"storedrv": 2, // content-addressed storage ROUTER: composes the blob/blobfs/blobhttp (tier-1) drivers into one namespace; the abi RegionBackend only when FAK_STORE opts in.
	"capindex": 2, // protocol-blind capability keystone (#1104 C1): CapRef/Capability/Index/Resolver + skill resolver, imports only abi(0). The gateway-backed MCP/A2A resolvers live in capindexgw(4) so the core stays importable by the tier-3 skill-loader (ctxresidency/ctxmmu, #1106).

	"ifc": 3, "normgate": 3, "secretgate": 3, "recall": 3, "kvmmu": 3, "radixkv": 3, "cdb": 3, "contextq": 3, "agentdojo": 3, "toollint": 3, "toolsandbox": 3, "terminalbench": 3,
	"agentdemo":     3,                // agentic "try-it" demo spine (epic #1167): a deterministic, no-key tool-using agent loop that folds the REAL kernel per call — the live-loop dual of turnbench's trace replay. Composer: imports abi(0)+adjudicator(2)+kernel(2), off the hot path.
	"browseraction": 3,                // browser/computer-use action-mediation harness: composes webbench actions with policy/adjudicator, off the live request path.
	"memq":          3, "headroom": 3, // memq: the memory-operation algebra composed over recall (tier 3). headroom: the context-compression seam over ctxmmu/abi (its doc.go declares composer/3).

	"agent": 4, "bench": 4, "turnbench": 4, "gateway": 4, "registrations": 4, "rsiloop": 4,
	"capindexgw": 4, // gateway-backed capindex resolvers (MCP tools / A2A methods): the adapter that couples capindex(2) to gateway(4). It lives at the higher tier so the capindex keystone itself stays tier-2 and importable by the tier-3 skill-loader.
	"tracesink":  4, // imports agent/turnbench/registrations (tier 4) — tier forced to 4
	"agenttest":  4, // public agent-workflow TEST harness (#238, D-008): deterministic fixtures + tool-call assertion library + mock tool responses + reproduce-from-transcript replay; imports agent(4), off the hot path.
	"ablate":     4, // the N-arm self-ablation sweep: a bench sibling; imports bench(4)+registrations(4)+metrics(1), off the hot path.

	"tokenizer":       1,
	"answershape":     1, // pure degeneration/verbosity metric over text; stdlib-only, imports nothing internal.
	"codelint":        1,
	"polymodel":       1, // multi-model residency + serial-decode-lane + cache-led MTP accept core; stdlib-only, imports nothing internal.
	"rulesynth":       3, // refusal-log rule synthesizer (#537): composes harvest/policy/adjudicator/shipgate to propose+gate a new structural rule as a reviewable diff; imports tier-2 mechanisms, never the hot path.
	"residency":       2,
	"ctxresidency":    3,
	"ctxplan":         1, // context planner: cost-based, forecast-driven O(1) view over a lossless history store; stdlib-only, imports nothing internal.
	"session":         1, // per-session DRIVE state: a TraceID-keyed, bounded-LRU, live-mutable control-state value (run-state/budget/priority/pace), the structural twin of ifc.Ledger widened past one value; stdlib-only, imports nothing internal.
	"wirescreen":      2, // local-model-on-the-wire proposer spine: registers an abi.SemanticScreen that ctxmmu consults after its regex floor (#569) + the ScreenDigest useful-page-out (#570) + the pre-send redactor (#572); imports only abi by default — the -tags fakwiremodel model arm (#569) adds model/tokenizer/ggufload (all tier-1).
	"advmodel":        2,
	"modelroute":      1, // per-aspect + ensemble model-routing policy spine (Route + Combine); pure, stdlib-only, imports nothing internal.
	"simhash":         1, // reference vector-similarity primitive (embed/cosine/top-k); the observability layer's near-duplicate / outlier-query substrate. Deterministic, stdlib-only, imports nothing internal.
	"trajectory":      3, // trajectory data plane: folds the abi event stream into per-trace Turn rows + JSONL export; an abi.Emitter that optionally stamps a simhash query embedding. Imports abi+simhash.
	"trajhook":        3, // pluggable trajectory scorer/tap seam (the "trivial skill does gardening" enabler): app code registers Scorers over Turn rows without a core edit. Imports trajectory+simhash.
	"sessionimage":    4, // portable, model-agnostic SESSION image: composes recall(3)+session(1)+trajectory(3)+ctxplan(1) into one versioned, sha256-integrity bundle + a .faksession tar (dump/pack/unpack/rehydrate across hosts/users/VMs/model changes). Integrator: imports tier-3 composers, off the hot path.
	"a2achan":         2, // in-kernel agent-to-agent message channel: a process-global, capability-floored, Ref-backed mailbox (Send/Recv adjudicated by a registered a2aGate + a2aIngress; Taint/Scope enforced). Mechanism: imports only abi, off the hot path.
	"region":          2, // typed one-sided shared Ref window: adjudicated Put/Get/Accumulate over abi.Resolver with ScopeFleet ceiling and vDSO coherence bumps.
	"snapshot":        3, // uniform DUMP/RESTORE seam over any primitive (turn/tool/session/fleet/rsi): a sha256-integrity envelope (Marshal/Parse over any body) + a ladder registry + typed codecs for trace(trajectory) and fleet(session.Table). Imports session(1)+trajectory(3); off the hot path.
	"rungobs":         2, // passive rung-decision distribution counter: an abi.Emitter (subscribed to EvDecide/EvDeny/EvVDSOHit) that re-folds each call's chain off the hot path via kernel.FoldExplain and bumps a per-(rung,kind,reason) histogram. Mechanism: imports kernel(2)+abi(0); runs synchronously in emit but adds 0 adjudication rungs and never touches the verdict or Counters.
	"sharedtask":      2, // in-memory collaborative task-record fold: user patches, conflicts, held verdicts, event rows, scoped views, and a2achan live subscriptions. Mechanism, off the hot path.
	"vcachegov":       2, // vCache M5 Governor (#720): the steady-state policy over the vCache warm set — pin/lazy/evict (§5.4), rate-limit warm budget (§5.5), cross-shard affinity routing + rehash/burst guards (§9/D3), and the Law-D4 secret classifier. Pure decision layer: imports cachemeta(1)+stdlib, off the hot path (NOT registered; M1–M3 wire the live loop).
	"vcachechain":     2, // vCache M4 chains & recall (#719): prefix DAG + topological replay (send-one-then-fan) + 20-block breakpoints + the §11.0 cost-gated rebuild (refuses single-unit chain rebuilds, allows amortized fan-out). Pure decision layer: imports cachemeta(1)+vcachegov(2)+stdlib, off the hot path (NOT registered; gated OFF by default).
	"vcachecal":       2, // vCache M1 observe & calibrate (#716): the warmth-belief estimator (§7) over cachemeta.Lifecycle at TierProvider + the offline probe harness that fits T/M_min/r (Law D2) + the LRU probe budget (observer-perturbs-state) + the Zipf-s concentration gate (§5.2) + the false-warm/false-cold prediction-error report. Pure decision layer: imports cachemeta(1)+stdlib only, off the hot path (NOT registered; observe-only — no warming in M1).
	"vcachescore":     2, // vCache operator scorecard: composes vcachecal/vcachechain/vcachegov proof leaves into the offline 2x readiness gate and hot-anchor index artifact; pure off-path decision layer.
	"vcachestar":      2, // vCache M2 star anchors (#717): canonicalizer-as-gate, wire-byte manifest keying, first-natural-request anchor warming, telemetry demotion, and uncached-first cost booking. Pure decision layer: imports cachemeta(1)+stdlib only, off the hot path.
	"vcachewarm":      2, // vCache M3 dedicated warming (#718): Anthropic max_tokens:0 vs decode-1 decision gates, byte-identical prefix guard, send-one-then-fan barrier, and wasted-warm accounting. Pure decision layer, off the hot path, no live transport claim.
	"sessionreset":    2, // budget-reset carryover builder: a pluggable Contributor registry that folds a drained session's transcript into the "human-like" seed a fresh session is re-armed with (durable facts via ctxmmu's shipped prior + task recap + warm-prefix descriptor via vcachechain + verbatim tail). Mechanism: imports ctxmmu(2)+vcachechain(2)+stdlib, NOT the wire agent type; off the hot path, registers nothing into the kernel.
	"taskmgr":         1, // process-local task/step/resource/ETA snapshot fold; stdlib-only, off the hot path.
	"stopfailure":     1, // pure StopFailure marker planner/settler over JSON files and transcript existence; stdlib-only, off the hot path.
	"dogfoodscore":    1, // pure dogfood-loop scorecard over transcripts/markers; imports stopfailure, off the hot path.
	"dropin":          1,
	"comm":            2,
	"cohort":          2, // fail-closed cohort shrink/agree over comm.Group + modelroute vote fold.
	"agenttopo":       2, // declared agent communication DAG over comm.Group + modelroute folds.
	"promptmmu":       1, // cache-prefix-preserving inbound prompt MMU: splices tools[] past the last cache_control breakpoint; stdlib-only, off the hot path, no agent/gateway import (decode is a callback).
	"loopmgr":         1, // durable loop-event JSONL ledger + read fold: SHA-256 hash chain over armed/fire/admit/start/heartbeat/end/witness/notify events. stdlib-only, off the hot path; schedules/spawns/notifies/authorizes nothing — those stay in the producers.
	"leaseref":        1, // cross-machine lease VISIBILITY substrate (#825): persists a lease record under refs/fak/locks/<id> so lease state rides ordinary git fetch/push between clones. Distribution, NOT atomic acquisition. Shells to git off the hot path through one Runner seam; imports only dormancy(1) for the lease's LastActiveAt clock (#1179).
	"guard":           1, // agent-spawn containment seam (#824): the Linux Landlock read-only-.git/hooks hook-floor for the child `fak guard` spawns, via a re-exec trampoline. Pure spec/resolution core + raw-syscall linux impl + no-op twin; opt-in, off by default, fails open; imports only stdlib (syscall/unsafe on linux), nothing internal.
	"pythongate":      2, // NEW-PYTHON-TOOL de-Python ratchet: scans tracked tools/*.py (git ls-files) against a frozen grandfathered baseline and refuses any new .py (NEW_PYTHON_TOOL). A tool-shaped witness leaf (reads tree, folds, emits offenses); shells to git off the hot path, imports nothing internal.
	"treedoctor":      2, // tree-hygiene doctor over safecommit's lock seam plus git worktree reads; mechanism/tool leaf, off the hot path.
	"gardenbundle":    3, // the garden bundle: a read-only fold-over-folds that runs the grandfathered Python gardening passes (scorecard control pane + fresh status, +loop-audit under --deep), reads each control-pane payload, and folds one schema/ok/verdict/finding envelope. Composer: composes other tools' outputs (shelling out off the hot path), imports nothing internal.
	"savingsvector":   1, // pure four-account saving-decomposition lens over a turnbench Report's already-measured fields (local_cpu/gpu_prefill/context_window/wall_clock, labeled per axis); stdlib-only, imports nothing internal, off the hot path.
	"swebenchsota":    2, // SWE-bench SOTA leaderboard snapshot: a tool-shaped leaf that extracts the embedded leaderboard JSON (regex+unescape), folds the per-group SOTA, and emits a versioned snapshot. net/http fetch off the hot path; imports nothing internal.
	"dogfoodissues":   3, // dogfood-action-issues backlog bridge: folds a recent-feature dogfood report.json into scorecard ACTION items, derives a stable dedup key per item, renders the marker-stamped issue body, and (only on --live) composes the external `gh` CLI to create/update one issue per item. Composer: shells out off the hot path, imports nothing internal.
	"horizonrecovery": 1, // pure budget-recovery (term r) grounding lens over a ctxplanbench report's already-measured real-transcript fields: surfaces the recovery ratio + its fault-rate FENCE co-located, structurally refuses to emit r/horizon_multiplier; stdlib-only, imports nothing internal, off the hot path.
	"guardrsi":        1, // pure guard RSI journal fold + scorecard: reads guard-audit bytes, computes deterministic verdict quality, and validates keep/revert iterations; stdlib-only, off the hot path.
	"repoguard":       1, // pure repo-containment classifier: resolves write/delete targets against a workspace root and emits OUT_OF_TREE_WRITE; stdlib-only, shared by the hook binary and loop driver.
	"egressfloor":     1, // pure network-egress destination classifier (cloud-metadata SSRF floor): names a tool call reaching the cloud-instance metadata / link-local family (169.254.169.254, metadata.google.internal, fd00:ec2::254, ...). Imports only abi(0)+net+net/url; the adjudicator's egress rung folds it on the live path.
	"hooks":           1, // commit-boundary gates run in ONE process: the Go port of the tools/check_*.py git-hook checkers (PUBLIC_LEAK/SECRET_SHAPE/DOC_PLACEMENT/BROKEN_LINK/FILE_ADMISSION/INDEX_SYNC/PROVENANCE_LABEL + commit-msg), folding one staged-diff read. stdlib-only, imports nothing internal, off the hot path.
	"workflow":        1, // pure DAG/map-reduce/fan-out orchestration core (#245, D-005): JSON-DSL compiler + topo-validated executor + retry/fail-fast fault tolerance; stdlib-only, imports nothing internal, off the hot path.
	"l3region":        1, // L3 disaggregated-cache child B Stage-1 seam (#77, epic #504): an abi.RegionBackend over a fake in-memory page-keyed L3 store — a Ref.Digest resolves to a page-key set (mget/mset), region round-trips bit-exact + verify-don't-trust. Imports only abi+stdlib; NOT registered (library leaf), off the hot path.
	"lifecycle":       1, // canonical shared run-state vocabulary; stdlib-only, imported by session/loopmgr to avoid token drift.
	"epochbridge":     1, // explicit session generation <-> abi speculation epoch converter; imports only abi/session and owns neither type.
	"lifebridge":      1, // explicit session.RunState <-> loopmgr.LoopState converter over lifecycle; imports only tier-1 leaves.
	"memview":         2, // typed virtual-view contract over canonical raw memory cells (#904): MemoryViewRecord binds a derived view (snippet/summary/qa/fact) to its source by a digest + byte span, inherits the source taint, and is invalidated when the source digest changes; a materialized view carries an abi.Verdict and re-enters adjudication before any effect. Mechanism: imports only abi(0)+stdlib, defines a RawPage interface so recall.Page adapts without an upward import; off the hot path, registers nothing.
	"fakrpc":          1, // disaggregated agent-RPC contract (#930): the pure Request envelope + the FAKRES nonce/sha frame (encode/decode/verify) a resident worker (cmd/fakrpcd) and pluggable text-only bridges build on. stdlib-only, imports nothing internal, off the hot path — the same frame tools/dgx_witness_run.sh emits.
	"resume":          1, // deterministic resume-cache decision (#745/#774 family): prices RESUME_FULL/CUT/RESET against the projected cold/warm prompt-cache posture at the resume boundary and recommends a cut-by-default re-entry; pure Plan(Input) Report, stdlib-only, imports nothing internal, off the hot path. The computable answer to "resume a 250k session — what happens to the cache".
	"vcacheobserve":   2,
	"vcachesnapshot":  2, // vCache observed-cache-window snapshot bridge (#827d882f): folds vcacheobserve's realized traffic into the read-side snapshot the score consumes; imports vcacheobserve(2)+stdlib only, off the hot path.
	"cadencereport":   3, // the consolidated regular-cadence report: a read-only fold-over-folds that distills the scorecard control pane (scores), git (work-done), and release-status (releases) into one schema/ok/verdict/finding envelope + a durable JSONL trend ledger. Composer (like gardenbundle): shells to the Python folds + git off the hot path, imports nothing internal.
	"dispatchorder":   1, // pure dispatch-ordering helper; stdlib-only, imports nothing internal, off the hot path.
	"dojo":            1, // the prediction-vs-reality gym's pure scoring/fold/ledger/board core: Prediction/Outcome/Episode scoring + the cross-lever leaderboard fold; stdlib-only, imports nothing internal (the corpus-scanning levers live in cmd/fak), off the hot path.
	"looprecover":     1, // pure loop-recovery decision helper; stdlib-only, imports nothing internal, off the hot path.
	"nightrun":        1, // RUN-IT-ALL-NIGHT local-capability data-collection planner: probes the box + ranks feasible-here collection tasks over the benchmark grid; imports benchcatalog(1)+stdlib, off the hot path.
	"claimcheck":      1, // pure net-true-value claim grader; stdlib-only, off the hot path.
	"loopindex":       1, // pure S0 agentic-loop scorecard: folds orient->plan->act->verify->ship->learn probes into loop-index + loopindex_debt; stdlib-only, off the hot path.
	"loopmap":         1, // queryable loop-stage -> tool map over loopindex(1); off the hot path.
	"sessionobs":      1, // SESSION-OBSERVABILITY-for-RSI scorecard: the value-side complement to tools/session_audit.py — grades how far our coding-session data has climbed the capture->structure->link->aggregate->learn ladder, folding the missing rungs into one sessionobs_debt integer. Pure scorer (Record/Outcome/Pipeline/Score), stdlib-only, imports nothing internal, off the hot path.
	"compactcohere":   1, // fak<->harness context-manager COHERENCE policy (#1131): attributes a served turn's prefix event (stable/fak_cut/fak_world_break/harness_rewrite/cold_ttl) + a standing PreCompact block/allow posture to suppress Claude Code's cache-destroying auto-compaction while fak's cache-preserving compaction copes. Pure sensor+policy, stdlib-only, imports nothing internal, off the hot path.
	"loopdrive":       1,
	"loopgate":        1, // pure loop exit gate: maps a claimed-done turn plus a witness criterion to WITNESSED/NOT_YET/REFUSED; witness execution is caller-injected.
	"turntaxmeter":    1, // pure observer-effect sampling and overhead-budget meter; stdlib-only, off the hot path.
	"slackenv":        1, // the ONE .env.slack.local token/channel resolver every outbound Slack publisher (scoreboard/blockerpost/benchpost/dispatchpost/dojopost/marketing/nodeusagepost) and chatrelay delegates to; pure stdlib, off the hot path.
	"dormancy":        1, // dormancy clock + horizon bucketer (#1179, epic #1178): a durable monotonic LastActiveAt Stamp + a pure Horizon(gap) -> {warm,cool,cold,frozen,ancient} bucketer (thresholds anchored to the resume cache TTLs); stdlib-only, imports nothing internal, off the hot path. Surfaced on session/loop/lease as the shared "how long dormant" field.
	"syspromptmmu": 2, // system-prompt MMU Rung 1 (#1259): emits fak's ordered base-context plan — the SegStable spine + versioned policy floor as []cachemeta.PromptSegment, each with a content-derived Witness. Pure authorship/decision layer: imports only cachemeta(1)+stdlib, off the hot path, no wire mutation.
	// new-leaf:tier — `python tools/new_leaf.py <name> --tier <name>` inserts the
	// declaration for a generated leaf immediately ABOVE this line. Keep the marker last.
}

var tierName = []string{"root", "foundation", "mechanism", "composer", "integrator"}

// hotPath is the set of packages on a live tool-call decision. None of them may import
// os/exec: the per-decide subprocess boundary is exactly the microsecond-vs-232ms cost
// fak exists to remove (DIRECTION.md, reviewer's grep #1). os/exec is fine OFF the path
// (bench, shipgate, tests) — only these packages are checked.
var hotPath = []string{"adjudicator", "kernel", "vdso", "grammar", "preflight", "ctxmmu", "ratelimit"}

// internalDir returns the absolute path of fak/internal. This test file lives in
// internal/architest, so internal is its parent.
func internalDir(t *testing.T) string {
	t.Helper()
	_, self, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("runtime.Caller failed — cannot locate the internal/ tree")
	}
	return filepath.Dir(filepath.Dir(self))
}

// goPackageDirs returns the short names of every directory under internal/ that contains
// at least one non-test .go file (i.e. is a real Go package), excluding architest itself.
func goPackageDirs(t *testing.T, internal string) []string {
	t.Helper()
	entries, err := os.ReadDir(internal)
	if err != nil {
		t.Fatalf("read internal dir: %v", err)
	}
	var pkgs []string
	for _, e := range entries {
		if !e.IsDir() || e.Name() == "architest" {
			continue
		}
		files, err := os.ReadDir(filepath.Join(internal, e.Name()))
		if err != nil {
			t.Fatalf("read %s: %v", e.Name(), err)
		}
		for _, f := range files {
			if !f.IsDir() && strings.HasSuffix(f.Name(), ".go") && !strings.HasSuffix(f.Name(), "_test.go") {
				pkgs = append(pkgs, e.Name())
				break
			}
		}
	}
	sort.Strings(pkgs)
	return pkgs
}

// imports parses every non-test .go file in internal/<pkg> and returns its full import
// paths. Build tags are intentionally ignored (every file is parsed): an upward import
// hidden behind a GOOS/GOARCH tag is still an upward import. Only the import block is
// read (parser.ImportsOnly), so this is fast.
func imports(t *testing.T, internal, pkg string) []string {
	t.Helper()
	fset := token.NewFileSet()
	parsed, err := parser.ParseDir(fset, filepath.Join(internal, pkg),
		func(fi fs.FileInfo) bool { return !strings.HasSuffix(fi.Name(), "_test.go") },
		parser.ImportsOnly)
	if err != nil {
		t.Fatalf("parse %s: %v", pkg, err)
	}
	seen := map[string]bool{}
	var out []string
	for _, p := range parsed {
		for _, f := range p.Files {
			for _, spec := range f.Imports {
				path := strings.Trim(spec.Path.Value, `"`)
				if !seen[path] {
					seen[path] = true
					out = append(out, path)
				}
			}
		}
	}
	return out
}

// TestEveryPackageDeclaresTier fails if a package on disk is missing from the tier table
// (a new leaf that forgot to take a layering position) or if the table names a package
// that no longer exists (a stale entry). This is what forces every future leaf through a
// conscious tier decision.
func TestEveryPackageDeclaresTier(t *testing.T) {
	internal := internalDir(t)
	onDisk := goPackageDirs(t, internal)

	diskSet := map[string]bool{}
	for _, p := range onDisk {
		diskSet[p] = true
		if _, ok := tier[p]; !ok {
			t.Errorf("package internal/%s has no declared tier.\n"+
				"Add it to the architest tier table at the LOWEST layer whose role it fits "+
				"(root<foundation<mechanism<composer<integrator). This forced choice is the "+
				"review gate that keeps the layered-DAG contract honest as the kernel grows.", p)
		}
	}
	for p := range tier {
		if p == "abi" {
			continue // abi is a package; keep it explicit even though it has its own dir
		}
		if !diskSet[p] {
			t.Errorf("tier table names internal/%s, but no such package is on disk "+
				"(stale entry — remove it).", p)
		}
	}
}

// TestNoUpwardImports is THE layering gate. For every internal cross-package edge it
// asserts tier(importer) >= tier(imported). A failure means a lower layer now reaches UP
// into a higher one — fix the dependency direction (usually: invert it through a
// registration seam, or move the shared type down into the foundation), do not relax the
// tier table to admit it.
func TestNoUpwardImports(t *testing.T) {
	internal := internalDir(t)
	var violations []string
	for _, pkg := range goPackageDirs(t, internal) {
		from, ok := tier[pkg]
		if !ok {
			continue // reported by TestEveryPackageDeclaresTier
		}
		for _, imp := range imports(t, internal, pkg) {
			if !strings.HasPrefix(imp, modPrefix) {
				continue
			}
			dep := strings.TrimPrefix(imp, modPrefix)
			dep = strings.SplitN(dep, "/", 2)[0] // collapse any future sub-packages
			to, ok := tier[dep]
			if !ok {
				continue // the missing-tier case is the other test's job
			}
			if to > from {
				violations = append(violations, fmtEdge(pkg, from, dep, to))
			}
		}
	}
	if len(violations) > 0 {
		sort.Strings(violations)
		t.Fatalf("layered-DAG import rule violated (a lower layer imports a higher one).\n"+
			"Rule: a package may import only packages whose tier is <= its own.\n  %s\n"+
			"Invert the dependency (registration seam) or push the shared type down a layer; "+
			"do not loosen the tier table.", strings.Join(violations, "\n  "))
	}
}

// TestHotPathHasNoExec turns DIRECTION.md's reviewer-grep #1 into a gate: no package on a
// live tool-call decision may import os/exec (or os/exec via a wrapper path). Keeps the
// adjudication path a single in-process chokepoint, never a per-decide subprocess.
func TestHotPathHasNoExec(t *testing.T) {
	internal := internalDir(t)
	for _, pkg := range hotPath {
		for _, imp := range imports(t, internal, pkg) {
			if imp == "os/exec" {
				t.Errorf("hot-path package internal/%s imports os/exec — the request path must "+
					"stay interpreter/subprocess-free (DIRECTION.md). Move the off-path work to a "+
					"non-hot-path package.", pkg)
			}
		}
	}
}

func fmtEdge(from string, ft int, to string, tt int) string {
	return from + " (" + tierName[ft] + ") -> " + to + " (" + tierName[tt] + ")"
}

// selfRegisters reports whether a leaf's NON-TEST code contains an actual call to
// abi.Register*(...). It walks the AST for the call expression (not a text grep), so a
// doc comment mentioning abi.Register or a Register call that lives only in _test.go is
// correctly NOT counted — a driver only loads if its production init() registers.
func selfRegisters(t *testing.T, internal, pkg string) bool {
	t.Helper()
	fset := token.NewFileSet()
	parsed, err := parser.ParseDir(fset, filepath.Join(internal, pkg),
		func(fi fs.FileInfo) bool { return !strings.HasSuffix(fi.Name(), "_test.go") }, 0)
	if err != nil {
		t.Fatalf("parse %s: %v", pkg, err)
	}
	found := false
	for _, p := range parsed {
		for _, f := range p.Files {
			ast.Inspect(f, func(n ast.Node) bool {
				call, ok := n.(*ast.CallExpr)
				if !ok {
					return true
				}
				sel, ok := call.Fun.(*ast.SelectorExpr)
				if !ok {
					return true
				}
				if id, ok := sel.X.(*ast.Ident); ok && id.Name == "abi" && strings.HasPrefix(sel.Sel.Name, "Register") {
					found = true
				}
				return true
			})
		}
	}
	return found
}

// regOffList names leaves that self-register but are intentionally wired NOT through the
// defconfig (internal/registrations). `agent` registers the "localtools" engine from its
// init() and is pulled in directly by cmd/fak, never blank-imported. `gateway` registers
// a per-Server metrics observer from New(), and the server itself is wired by cmd/fak.
// `spec` (the speculation ProvisionalSink + OpsSpec ops) registers ONLY from its
// Enabled()-gated Install(), never from init() and never from the defconfig — that
// off-defconfig absence IS the strongest of its two safety gates (the kernel never even
// links the poly-model lane until a rung flips FAK_POLYMODEL and calls Install; epic
// #529). A leaf added here is a conscious "wired elsewhere" decision, the same review
// chokepoint as the tier table.
var regOffList = map[string]bool{"agent": true, "gateway": true, "spec": true}

// TestRequestPathLeavesRegistered closes the registration-completeness hole: a leaf whose
// production init() calls abi.Register* MUST be either blank-imported by the defconfig
// (internal/registrations) or on the off-list — otherwise its init() never fires and the
// driver silently never loads (the worst failure: a security rung that isn't there). The
// set is re-derived from the AST on every run, so a new self-registering leaf that nobody
// wired in fails the build instead of vanishing.
func TestRequestPathLeavesRegistered(t *testing.T) {
	internal := internalDir(t)

	registered := map[string]bool{}
	for _, imp := range imports(t, internal, "registrations") {
		if strings.HasPrefix(imp, modPrefix) {
			registered[strings.SplitN(strings.TrimPrefix(imp, modPrefix), "/", 2)[0]] = true
		}
	}

	for _, pkg := range goPackageDirs(t, internal) {
		if pkg == "registrations" {
			continue
		}
		if selfRegisters(t, internal, pkg) && !registered[pkg] && !regOffList[pkg] {
			t.Errorf("leaf internal/%s self-registers (init() -> abi.Register*) but is neither "+
				"blank-imported in internal/registrations (the defconfig) nor on regOffList — so "+
				"its init() never fires and the driver silently never loads. Add\n"+
				"    _ \"%s%s\"\nto internal/registrations, or add %q to regOffList if it is "+
				"wired elsewhere (as agent is, via cmd/fak).", pkg, modPrefix, pkg, pkg)
		}
	}
}

// chatEndpointRole names every internal package allowed to reference the OpenAI
// chat-completions endpoint path in its NON-TEST source, each paired with the role
// that earns it. The kernel must not grow accidental duplicate chat clients: a
// degenerate `engine.HTTPEngine` once duplicated the live planner (spoke a bespoke
// `tool=X args=Y` prompt, never wired, never spoke real tool-calling) and the seam
// *entrenched* over time before it was deleted. `agent` owns the general outbound
// planner (`HTTPPlanner`); `gateway` is the inbound SERVER of that route (the
// adjudication proxy), not a client; and off-path witnesses may replay or benchmark
// the same wire. cmd/fak's help text also names it but lives outside internal/, so it
// is not scanned.
var chatEndpointRole = map[string]string{
	"agent":      "the single outbound chat-completions client (HTTPPlanner)",
	"gateway":    "the inbound /v1/chat/completions server route (adjudication proxy)",
	"webbench":   "the off-path serving-parity benchmark client (not a live planner)",
	"guardtrace": "the off-path trace-replay upstream fake (OpenAI/Anthropic provider replay, not a live planner)",
}

// TestSingleOpenAIChatClient pins the T4 fix as an architecture invariant: the
// OpenAI chat-completions endpoint path appears, as a string literal in non-test
// code, in EXACTLY the declared set of packages. A new package referencing it is a
// re-grown duplicate client (fail: re-home the call in `agent`); a declared package
// that stops referencing it is a client/route that went missing (fail: a required
// seam vanished, or update the table if the topology deliberately changed). Reading
// the AST — not a text grep — means a doc comment that merely mentions the path is
// correctly not counted.
func TestSingleOpenAIChatClient(t *testing.T) {
	internal := internalDir(t)
	got := pkgsReferencingChatEndpoint(t, internal)

	for pkg := range got {
		if _, ok := chatEndpointRole[pkg]; !ok {
			t.Errorf("package internal/%s references the OpenAI chat-completions endpoint as a "+
				"string literal, but is not a declared holder. The live client must live ONLY in "+
				"internal/agent (HTTPPlanner) — a second one is the TICKETS-T4 duplicate-seam "+
				"regression. Re-home the call in agent, or, if this is a deliberate new role, add "+
				"%q to chatEndpointRole with the role it serves.", pkg, pkg)
		}
	}
	for pkg, role := range chatEndpointRole {
		if !got[pkg] {
			t.Errorf("internal/%s is declared as %q but no longer references the chat-completions "+
				"endpoint. If that seam was intentionally moved, update chatEndpointRole; otherwise "+
				"a required client/route went missing.", pkg, role)
		}
	}
}

// pkgsReferencingChatEndpoint returns the set of internal packages whose NON-TEST
// source contains the OpenAI chat-completions endpoint path ("chat/completions") as
// a string literal. Only real string constants count (parsed from the AST), so a
// comment or identifier mentioning the path does not.
func pkgsReferencingChatEndpoint(t *testing.T, internal string) map[string]bool {
	t.Helper()
	got := map[string]bool{}
	for _, pkg := range goPackageDirs(t, internal) {
		fset := token.NewFileSet()
		parsed, err := parser.ParseDir(fset, filepath.Join(internal, pkg),
			func(fi fs.FileInfo) bool { return !strings.HasSuffix(fi.Name(), "_test.go") }, 0)
		if err != nil {
			t.Fatalf("parse %s: %v", pkg, err)
		}
		for _, p := range parsed {
			for _, f := range p.Files {
				ast.Inspect(f, func(n ast.Node) bool {
					if got[pkg] {
						return false
					}
					if lit, ok := n.(*ast.BasicLit); ok && lit.Kind == token.STRING &&
						strings.Contains(lit.Value, "chat/completions") {
						got[pkg] = true
						return false
					}
					return true
				})
			}
		}
	}
	return got
}

// requestPathClosure returns the set of internal package short-names transitively
// reachable from internal/registrations — the live request-path closure. It is the
// root the kernel boots from: registrations blank-imports every leaf that ships
// enabled, so anything it can reach is on a live tool-call decision's dependency
// graph. The walk reuses imports() (parser.ImportsOnly) and the same modPrefix-trim
// + sub-package collapse as TestNoUpwardImports.
//
// imports() is build-tag-blind BY DESIGN (see its comment): every file is parsed
// regardless of GOOS/GOARCH tags. So this closure is the UNION over all build
// constraints — a strict SUPERSET of `go list -deps`, which honors only the host's
// tags. For a ban-gate that is the safe direction: the closure can over-include a
// package, never silently drop one a tagged build would pull onto the path. Do not
// "fix" imports() to honor tags here — that would narrow the gate.
func requestPathClosure(t *testing.T, internal string) map[string]bool {
	t.Helper()
	reached := map[string]bool{}
	queue := []string{"registrations"}
	for len(queue) > 0 {
		pkg := queue[len(queue)-1]
		queue = queue[:len(queue)-1]
		if reached[pkg] {
			continue
		}
		reached[pkg] = true
		for _, imp := range imports(t, internal, pkg) {
			if !strings.HasPrefix(imp, modPrefix) {
				continue
			}
			dep := strings.SplitN(strings.TrimPrefix(imp, modPrefix), "/", 2)[0]
			if !reached[dep] {
				queue = append(queue, dep)
			}
		}
	}
	return reached
}

// interpreterExec is the closed denylist of script-interpreter program names a
// request-path package may NOT exec. Matched against filepath.Base(arg), lowercased,
// with a trailing .exe stripped — so "/usr/bin/python3" and "C:\\py\\python.exe" both
// resolve to a denied basename. Execing a COMPILED binary (git, go, the fak binary)
// is allowed; only an interpreter that would re-introduce an untyped runtime on the
// decision path is banned (DIRECTION.md: the binary adjudicates with zero runtime
// dependency on Python/shell/node).
var interpreterExec = map[string]bool{
	"python": true, "python3": true, "python2": true, "py": true,
	"node": true, "nodejs": true, "deno": true, "bun": true,
	"ruby": true, "perl": true, "php": true,
	"sh": true, "bash": true, "zsh": true, "dash": true,
	"pwsh": true, "powershell": true, "osascript": true,
}

// interpreterSuffix flags a literal arg whose extension names a script body, even if
// the program name itself was missed (e.g. exec.Command("./run.sh")).
var interpreterSuffix = []string{".py", ".sh", ".bash", ".ps1", ".rb", ".pl", ".mjs"}

// interpreterExecAllow is the conscious escape hatch: a request-path package that
// legitimately execs something the gate cannot prove is a compiled binary (a
// non-literal program arg), keyed to a one-line justification — the same review
// chokepoint as regOffList. Each entry is a deliberate, reviewed decision, not a
// silent pass.
var interpreterExecAllow = map[string]string{
	"witness": "execution witnesses run caller-declared selector argv; the selector is evidence-gated and not a script interpreter dependency of the kernel itself",
}

// oracleSeamFiles names the off-path Python oracle/baseline seam scripts (DIRECTION.md
// seam table row 1). They are the independent ML-ecosystem witness the Go model is
// measured against — legitimate at a typed, off-path boundary, reachable ONLY from
// _test.go and off-path bench/commands. A live string reference to one from a non-test
// file on the registrations closure would put an untyped seam on the request path.
var oracleSeamFiles = []string{
	"export_oracle.py", "bench_hf.py", "bench_hf_quant.py",
	"bench_llamacpp.py", "compare.py", "local_shim.py",
}

// execProgramArg returns the program-name argument of an exec.Command /
// exec.CommandContext call expression, and whether the node is such a call at all.
// CommandContext's program is Args[1] (Args[0] is the context); Command's is Args[0].
func execProgramArg(call *ast.CallExpr) (ast.Expr, bool) {
	sel, ok := call.Fun.(*ast.SelectorExpr)
	if !ok {
		return nil, false
	}
	id, ok := sel.X.(*ast.Ident)
	if !ok || id.Name != "exec" {
		return nil, false
	}
	switch sel.Sel.Name {
	case "Command":
		if len(call.Args) >= 1 {
			return call.Args[0], true
		}
	case "CommandContext":
		if len(call.Args) >= 2 {
			return call.Args[1], true
		}
	}
	return nil, true // a recognized exec.* with too few args is still an exec we saw
}

// stringLit returns the unquoted value of a string-literal expression, or ("", false)
// if the expression is not a plain string constant the gate can read statically.
func stringLit(e ast.Expr) (string, bool) {
	lit, ok := e.(*ast.BasicLit)
	if !ok || lit.Kind != token.STRING {
		return "", false
	}
	v, err := strconv.Unquote(lit.Value)
	if err != nil {
		return "", false
	}
	return v, true
}

// looksLikeInterpreter reports whether a literal program/arg names a banned interpreter.
func looksLikeInterpreter(arg string) bool {
	base := strings.ToLower(filepath.Base(arg))
	base = strings.TrimSuffix(base, ".exe")
	if interpreterExec[base] {
		return true
	}
	low := strings.ToLower(arg)
	for _, suf := range interpreterSuffix {
		if strings.HasSuffix(low, suf) {
			return true
		}
	}
	return false
}

// TestRequestPathInterpreterFree is DIRECTION.md's foundational thesis turned into a
// gate: "the binary adjudicates with ZERO runtime dependency on Python, a shell, or a
// node runtime. If the binary needs an interpreter to adjudicate, the direction is
// broken." No package transitively reachable from internal/registrations (the live
// request path) may exec a script interpreter. Execing a COMPILED binary (git, the fak
// binary) is fine — that is not the per-decide untyped-runtime boundary fak exists to
// remove; execing python/node/sh IS.
//
// Seeded at green (2026-06-18): every in-closure exec.Command in the tree is a literal
// "git" (shipgate/witness), which is a compiled binary and passes. The one variable
// program-name exec (bench, exec.Command(binPath, "hook")) is OFF the registrations
// closure, so the fail-closed branch is dormant — the seed is clean, the allow-list is
// empty. Like every architest gate it is tightened over time, never loosened to admit a
// new violation.
//
// Fail-closed polarity: this is a BAN, so a program arg the gate cannot prove is a
// compiled binary (a non-literal — variable, concat, call) FAILS rather than passes.
// An interpreter name built dynamically on the path is exactly the case the thesis
// forbids; make the program a string literal, move the exec off-path, or add the
// package to interpreterExecAllow with a justification.
func TestRequestPathInterpreterFree(t *testing.T) {
	internal := internalDir(t)
	closure := requestPathClosure(t, internal)
	for pkg := range closure {
		execImported := false
		for _, imp := range imports(t, internal, pkg) {
			if imp == "os/exec" {
				execImported = true
				break
			}
		}

		fset := token.NewFileSet()
		parsed, err := parser.ParseDir(fset, filepath.Join(internal, pkg),
			func(fi fs.FileInfo) bool { return !strings.HasSuffix(fi.Name(), "_test.go") }, 0)
		if err != nil {
			t.Fatalf("parse %s: %v", pkg, err)
		}

		sawExecCall := false
		for _, p := range parsed {
			for _, f := range p.Files {
				ast.Inspect(f, func(n ast.Node) bool {
					call, ok := n.(*ast.CallExpr)
					if !ok {
						return true
					}
					prog, isExec := execProgramArg(call)
					if !isExec {
						return true
					}
					sawExecCall = true
					if prog == nil {
						return true // recognized exec with too few args; nothing to classify
					}
					if _, allowed := interpreterExecAllow[pkg]; allowed {
						return true
					}
					lit, isLit := stringLit(prog)
					if !isLit {
						t.Errorf("request-path package internal/%s calls exec with a non-literal "+
							"program argument the gate cannot prove is a compiled binary. On the live "+
							"path (reachable from internal/registrations) the program name must be a "+
							"string literal so the interpreter-free invariant is checkable (DIRECTION.md). "+
							"Make it a literal, move the exec off-path (a non-registered command/bench "+
							"package), or add %q to interpreterExecAllow with a one-line justification "+
							"(as regOffList does).", pkg, pkg)
						return true
					}
					if looksLikeInterpreter(lit) {
						t.Errorf("request-path package internal/%s execs the script interpreter %q "+
							"(reachable from internal/registrations). DIRECTION.md: the binary "+
							"adjudicates with ZERO runtime dependency on Python/shell/node — an "+
							"interpreter exec on the path breaks the direction. Move it off-path; if "+
							"%q is a compiled binary the gate misclassified, that is a gate bug.",
							pkg, lit, lit)
					}
					// also scan later args for a script-body suffix (e.g. exec.Command("python", "x.py"))
					for _, a := range call.Args {
						if s, ok := stringLit(a); ok && a != prog && looksLikeInterpreter(s) {
							t.Errorf("request-path package internal/%s execs a script body %q "+
								"(reachable from internal/registrations) — a script interpreter on the "+
								"decision path breaks DIRECTION.md. Move it off-path.", pkg, s)
						}
					}
					return true
				})
			}
		}

		// Belt rung: a closure package that imports os/exec but exposed zero recognized
		// exec.Command/CommandContext calls is execing via an aliased import or an
		// unrecognized form the AST walk missed — fail closed rather than wave it through.
		if execImported && !sawExecCall {
			if _, allowed := interpreterExecAllow[pkg]; allowed {
				continue
			}
			t.Errorf("request-path package internal/%s imports os/exec but the gate classified "+
				"zero exec.Command/CommandContext calls — exec via an aliased import or an "+
				"unrecognized form the interpreter-free gate cannot read. Surface the call in a "+
				"recognized form, move it off-path, or add %q to interpreterExecAllow with a "+
				"justification.", pkg, pkg)
		}
	}
}

// TestOracleSeamStaysOffPath is the companion to TestRequestPathInterpreterFree: it
// catches the Python oracle seam leaking onto the request path AS DATA (a live string
// reference) even without an exec. Per DIRECTION.md's seam table the ML-ecosystem
// oracle (export_oracle.py and friends) is OFF-path measurement only, reachable from
// _test.go and off-path commands — never named as a live string in a non-test file on
// the registrations closure.
//
// The scan reads only string LITERALS (BasicLit) from the AST, so a doc comment that
// mentions a seam file — e.g. model/weights.go's "reads a directory produced by
// export_oracle.py" — is correctly NOT a violation, exactly as TestSingleOpenAIChatClient
// distinguishes a literal from a comment. Seeded at green: today the only references are
// comments. A real os.ReadFile("…/export_oracle.py") on the path would fail.
func TestOracleSeamStaysOffPath(t *testing.T) {
	internal := internalDir(t)
	closure := requestPathClosure(t, internal)
	for pkg := range closure {
		fset := token.NewFileSet()
		parsed, err := parser.ParseDir(fset, filepath.Join(internal, pkg),
			func(fi fs.FileInfo) bool { return !strings.HasSuffix(fi.Name(), "_test.go") }, 0)
		if err != nil {
			t.Fatalf("parse %s: %v", pkg, err)
		}
		for _, p := range parsed {
			for _, f := range p.Files {
				ast.Inspect(f, func(n ast.Node) bool {
					lit, ok := n.(*ast.BasicLit)
					if !ok || lit.Kind != token.STRING {
						return true
					}
					val, err := strconv.Unquote(lit.Value)
					if err != nil {
						return true
					}
					for _, seam := range oracleSeamFiles {
						if strings.Contains(val, seam) {
							t.Errorf("request-path package internal/%s references the Python oracle "+
								"seam %q as a live string (not a comment) in non-test code, reachable "+
								"from internal/registrations. Per DIRECTION.md's seam table the oracle is "+
								"OFF-path measurement only; a live reference on the request-path closure "+
								"puts an untyped seam on the decision path. Move it to a _test.go or an "+
								"off-path command package.", pkg, seam)
						}
					}
					return true
				})
			}
		}
	}
}

// pkgCallsSelector reports whether package internal/<pkg>'s NON-TEST source contains
// a call of the form <recv>.<name>(...) — read from the AST, not a text grep, so a
// doc comment or a call living only in _test.go does NOT count. It is the general
// form of selfRegisters (which is hard-wired to abi.Register*).
func pkgCallsSelector(t *testing.T, internal, pkg, recv, name string) bool {
	t.Helper()
	fset := token.NewFileSet()
	parsed, err := parser.ParseDir(fset, filepath.Join(internal, pkg),
		func(fi fs.FileInfo) bool { return !strings.HasSuffix(fi.Name(), "_test.go") }, 0)
	if err != nil {
		t.Fatalf("parse %s: %v", pkg, err)
	}
	for _, p := range parsed {
		for _, f := range p.Files {
			found := false
			ast.Inspect(f, func(n ast.Node) bool {
				call, ok := n.(*ast.CallExpr)
				if !ok {
					return true
				}
				sel, ok := call.Fun.(*ast.SelectorExpr)
				if !ok {
					return true
				}
				id, ok := sel.X.(*ast.Ident)
				if ok && id.Name == recv && sel.Sel.Name == name {
					found = true
					return false
				}
				return true
			})
			if found {
				return true
			}
		}
	}
	return false
}

// TestRecallReadmissionFoldsRegistry is the readmission-gate-strength gate
// (GROWTH.md). recall's page-in re-screen MUST enforce the kernel's REGISTERED
// ResultAdmitter chain — the same fold kvmmu.FoldedGate uses — so a rank-5+ detector
// (normgate today, anything the fleet adds later) that quarantines a payload also
// catches it on reload. The shipped construction backs the Session re-screen with a
// bare ctxmmu gate; folding `abi.ResultAdmittersFor` is what makes readmission
// inherit the stronger chain.
//
// WHY A GATE (and why it BITES): the compiler cannot enforce this. A bare
// ctxmmu.New() gate is valid Go, and the gate field is a concrete *ctxmmu.MMU with
// no interface that forces the fold — so a future edit could silently revert recall
// to bare-gate readmission and Go would wave it through, re-opening the documented
// weakening. This is the difference from the REJECTED reason-closure gate, which was
// vacuous because Verdict.Reason is already a typed enum the compiler enforces. Here
// the AST gate catches a real construction regression the type system permits.
//
// Seeded GREEN: recall/recall.go's reScreen now folds abi.ResultAdmittersFor
// (the fix this gate guards). A revert to bare-gate-only readmission drops that call
// and turns the gate RED — exactly the regression it exists to catch.
func TestRecallReadmissionFoldsRegistry(t *testing.T) {
	internal := internalDir(t)
	if !pkgCallsSelector(t, internal, "recall", "abi", "ResultAdmittersFor") {
		t.Error("internal/recall does not call abi.ResultAdmittersFor in non-test code — its " +
			"page-in re-screen has regressed to a bare single-driver gate that ignores the kernel's " +
			"registered ResultAdmitter chain (the GROWTH.md readmission-gate-strength weakening). A " +
			"rank-5+ detector (normgate) that quarantines a payload would then NOT catch it on reload. " +
			"Fold abi.ResultAdmittersFor in reScreen, most-restrictive-wins, exactly as kvmmu.FoldedGate does.")
	}
}

// foldSites names every internal package whose NON-TEST source folds a verdict chain
// most-restrictive-wins, paired with the role that earns it. The fold's ordering MUST
// come from abi.FoldRank — the single restrictiveness-lattice authority — never from a
// raw comparison of VerdictKind values. This matters because a VerdictKind's numeric
// value is its REGISTRATION-BLOCK id, not its restrictiveness: VerdictDeny is kind 1
// numerically but FoldRank 100 (most restrictive), while a registered combinator-verdict
// in the 1024+ vendor block can carry any declared rank. A fold that wrote
// `if v.Kind > best.Kind` would compile and silently mis-order — picking a numerically-
// larger but LESS restrictive verdict and dropping a real Deny/Quarantine on the floor.
// abi.FoldRank is the indirection that lets the FROZEN fold order a NEW kind without a
// core edit (registry.go); every fold site must route through it.
//
// Set AT REALITY (2026-06-20) from an AST scan of `abi.FoldRank(` callers: these four
// are exactly the packages that fold a verdict chain by restrictiveness — the kernel's
// pre-call fold + result-side admitResult (kernel), the KV-level result gate (kvmmu),
// the session-boundary readmission re-screen (recall), and the transcript-replay
// admission fold (agent). A NEW caller is conforming and does not fail; a DECLARED site
// that stops calling abi.FoldRank has regressed to a hand-rolled comparison and fails.
var foldSites = map[string]string{
	"kernel": "the pre-call Fold + result-side admitResult (most-restrictive-wins)",
	"kvmmu":  "the KV-level FoldedGate result-admission fold",
	"recall": "the session-boundary readmission re-screen fold",
	"agent":  "the transcript-replay result-admission fold",
}

// TestFoldSitesOrderByFoldRank is the ordering-primitive dual of
// TestRecallReadmissionFoldsRegistry: that gate pins WHICH chain a site folds
// (abi.ResultAdmittersFor); this one pins HOW it orders the fold (abi.FoldRank). Every
// declared fold site MUST call abi.FoldRank in non-test code, so the most-restrictive-wins
// decision comes from the registry's restrictiveness lattice and not from a raw VerdictKind
// comparison the type system would happily accept.
//
// WHY A GATE (and why it BITES): the compiler cannot enforce this. `if v.Kind > best.Kind`
// is valid Go — VerdictKind is a uint16, fully ordered — but it orders by registration-block
// id, not restrictiveness, so it silently picks the wrong verdict (see foldSites). There is
// no interface that forces a fold loop to consult abi.FoldRank, so a future edit could revert
// any site to a bare comparison and Go would wave it through, re-opening a most-restrictive-
// wins violation on the live decision path. Like TestRecallReadmissionFoldsRegistry, the AST
// gate catches a real construction regression the type system permits; unlike the REJECTED
// reason-closure gate, it is not vacuous — nothing in the type system already guarantees it.
//
// Seeded GREEN (2026-06-20): all four sites call abi.FoldRank today. Dropping the call at any
// of them — the regression to a hand-rolled .Kind comparison — turns the gate RED.
func TestFoldSitesOrderByFoldRank(t *testing.T) {
	internal := internalDir(t)
	for pkg, role := range foldSites {
		if !pkgCallsSelector(t, internal, pkg, "abi", "FoldRank") {
			t.Errorf("fold site internal/%s (%s) does not call abi.FoldRank in non-test code — its "+
				"most-restrictive-wins fold has regressed to a raw VerdictKind comparison. A VerdictKind's "+
				"numeric value is its registration-block id, not its restrictiveness (VerdictDeny is kind 1 "+
				"but FoldRank 100), so `v.Kind > best.Kind` silently mis-orders and can drop a real "+
				"Deny/Quarantine. Order the fold by abi.FoldRank(v.Kind), exactly as kernel.admitResult and "+
				"kvmmu.FoldedGate do; if this site genuinely no longer folds a verdict chain, remove it from "+
				"the foldSites table with that justification.", pkg, role)
		}
	}
}

// defaultBuildContext is the portable pure-Go artifact's build context. It honors
// //go:build constraints (CgoEnabled true so an untagged cgo file is surfaced, not
// silently ignored) but pins a non-Apple-Silicon target and sets NO opt-in tags: CUDA,
// Vulkan, and Apple-Silicon Metal cgo files are constraint-excluded. The context is
// constructed explicitly (rather than reading the host's CGO_ENABLED / GOOS / GOARCH)
// so the gate's verdict is the same on every node — a Mac, a GPU server, and this
// Windows box all judge the SAME pure-Go closure.
func defaultBuildContext() build.Context {
	ctx := build.Default
	ctx.CgoEnabled = true // surface a tag-passing cgo file in CgoFiles; do not drop it
	ctx.GOOS = "linux"
	ctx.GOARCH = "amd64"
	ctx.BuildTags = nil // no cuda/vulkan tags; Apple-Silicon Metal requires darwin/arm64+cgo
	return ctx
}

// TestRequestPathDefaultBuildIsCgoFree is DIRECTION.md's static-binary thesis turned
// into a gate (the architest gap-(a) knife): the portable pure-Go artifact — the
// binary that adjudicates a live tool call without a device runtime — must link NO cgo. A cgo import pulls a C
// toolchain and a dynamically-linked C runtime onto the decision path, voiding the
// "one static Go binary, no external runtime" property the whole direction rests on
// (the same thesis TestRequestPathInterpreterFree enforces for a script interpreter;
// this is its compile-time, in-process twin — cgo is the untyped runtime that creeps
// in at LINK time rather than via exec).
//
// The scan is over requestPathClosure (everything transitively reachable from
// internal/registrations). For each package it asks go/build — NOT a shelled-out
// `go build` — which files compile under defaultBuildContext, and fails if any of
// them is a cgo file (import "C"). go/build.ImportDir resolves //go:build
// constraints, so this is the one architest gate that is deliberately build-tag-AWARE
// rather than tag-blind: the cgo files legitimately EXIST in the tree (the GPU
// backends), gated out of the default binary by opt-in tags. A tag-blind scan (like
// imports()) would see compute/cuda.go's import "C" and fire RED at seed; the
// default-context scan correctly sees only what a real `go build ./cmd/fak` links.
//
// WHY A GATE (and why it BITES): nothing in the type system or `go vet` stops an
// agent from adding an UNTAGGED `import "C"` (or a cgo file whose //go:build clause
// matches the default context) to a request-path leaf — it compiles fine on a box
// with a C toolchain, and the regression (the shipped fak binary silently became a
// cgo binary) is invisible until a from-source build on a toolchain-less node fails,
// or until the static-binary release artifact stops being static. go/build sorts a
// tag-passing cgo file into CgoFiles; a tag-gated one into IgnoredGoFiles. So
// len(CgoFiles)>0 on the closure is exactly "the default build links cgo."
//
// Seeded GREEN (2026-06-20, updated for #62): the cgo files in the module —
// compute/{cuda,metal,vulkan}.go, metalgemm/metalgemm.go, model/awq_cuda.go — are
// all behind either explicit device tags or Apple-Silicon+cgo constraints, so under
// defaultBuildContext every request-path package reports zero CgoFiles. Adding an
// untagged import "C" to any in-closure package turns the gate RED — exactly the
// "cgo crept onto the pure-Go request path" regression it exists to catch. Like every
// architest gate it is tightened over time, never loosened to admit a new violation.
func TestRequestPathDefaultBuildIsCgoFree(t *testing.T) {
	internal := internalDir(t)
	ctx := defaultBuildContext()
	closure := requestPathClosure(t, internal)
	// Deterministic order so a multi-offender failure reads the same on every run.
	pkgs := make([]string, 0, len(closure))
	for pkg := range closure {
		pkgs = append(pkgs, pkg)
	}
	sort.Strings(pkgs)

	for _, pkg := range pkgs {
		dir := filepath.Join(internal, pkg)
		bp, err := ctx.ImportDir(dir, 0)
		if err != nil {
			// A closure package with no files that satisfy the DEFAULT context (e.g. an
			// all-tagged package) is not a cgo violation — go/build reports NoGoError.
			// Anything else is a real read failure the gate must not swallow.
			if _, noGo := err.(*build.NoGoError); noGo {
				continue
			}
			t.Fatalf("go/build ImportDir(internal/%s) under the default build context: %v", pkg, err)
		}
		if len(bp.CgoFiles) > 0 {
			t.Errorf("request-path package internal/%s links cgo in the DEFAULT build "+
				"(`go build ./cmd/fak`): file(s) %v use import \"C\" with a build constraint that "+
				"the default tag set satisfies. A cgo import puts a C toolchain + dynamically-linked "+
				"C runtime on the live tool-call decision path, voiding the single-static-Go-binary "+
				"property DIRECTION.md rests on. Put the cgo file behind an opt-in //go:build tag "+
				"(as compute/cuda.go uses `cuda`, compute/metal.go and metalgemm use "+
				"`darwin && arm64 && cgo`), or move it to an off-path package not "+
				"reachable from internal/registrations.", pkg, bp.CgoFiles)
		}
	}
}

// TestMetalComputeSharesMetalgemmDeviceSeam pins issue #61's architecture fix: the
// compute registry's Metal backend must not create its own MTLDevice/command queue.
// internal/metalgemm owns the process-wide Metal singleton, and compute/metal_shim.m
// binds to that singleton before compiling its compute kernels.
func TestMetalComputeSharesMetalgemmDeviceSeam(t *testing.T) {
	internal := internalDir(t)

	fset := token.NewFileSet()
	computeFile, err := parser.ParseFile(fset, filepath.Join(internal, "compute", "metal.go"), nil, parser.ImportsOnly)
	if err != nil {
		t.Fatalf("parse internal/compute/metal.go: %v", err)
	}
	hasMetalgemmImport := false
	for _, spec := range computeFile.Imports {
		if strings.Trim(spec.Path.Value, `"`) == modPrefix+"metalgemm" {
			hasMetalgemmImport = true
			break
		}
	}
	if !hasMetalgemmImport {
		t.Fatalf("internal/compute/metal.go must import internal/metalgemm so compute and model Metal paths share one device seam")
	}

	metalgemmFile, err := parser.ParseFile(fset, filepath.Join(internal, "metalgemm", "metalgemm.go"), nil, 0)
	if err != nil {
		t.Fatalf("parse internal/metalgemm/metalgemm.go: %v", err)
	}
	funcs := map[string]bool{}
	for _, decl := range metalgemmFile.Decls {
		if fn, ok := decl.(*ast.FuncDecl); ok {
			funcs[fn.Name.Name] = true
		}
	}
	for _, name := range []string{"MPSAvailable", "DeviceName", "DeviceMemoryTotal"} {
		if !funcs[name] {
			t.Fatalf("internal/metalgemm must expose %s for the compute Metal backend's shared-device probe", name)
		}
	}

	shimBytes, err := os.ReadFile(filepath.Join(internal, "compute", "metal_shim.m"))
	if err != nil {
		t.Fatalf("read internal/compute/metal_shim.m: %v", err)
	}
	shim := string(shimBytes)
	for _, want := range []string{
		"extern id<MTLDevice> gDev;",
		"extern id<MTLCommandQueue> gQueue;",
		"int mg_init(void);",
		"mg_init()",
	} {
		if !strings.Contains(shim, want) {
			t.Fatalf("internal/compute/metal_shim.m must bind through metalgemm's shared Metal seam; missing %q", want)
		}
	}
	for _, oldLane := range []string{
		"MTLCreateSystemDefaultDevice()",
		"static id<MTLDevice> g_device;",
		"static id<MTLCommandQueue> g_queue;",
	} {
		if strings.Contains(shim, oldLane) {
			t.Fatalf("internal/compute/metal_shim.m still contains the old independent Metal lane %q", oldLane)
		}
	}

	metalBytes, err := os.ReadFile(filepath.Join(internal, "metalgemm", "metal.m"))
	if err != nil {
		t.Fatalf("read internal/metalgemm/metal.m: %v", err)
	}
	metal := string(metalBytes)
	for _, want := range []string{
		"id<MTLDevice>",
		"gDev = nil",
		"id<MTLCommandQueue>",
		"gQueue = nil",
		"int mg_mps_available(void)",
		"int mg_device_name(char *name, int namelen)",
		"int mg_device_memory_total(unsigned long long *total)",
	} {
		if !strings.Contains(metal, want) {
			t.Fatalf("internal/metalgemm/metal.m must own and expose the shared Metal seam; missing %q", want)
		}
	}
}

// regionBackendRole names every internal package allowed to register the singleton
// Ref/Resolver backend (abi.RegisterRegionBackend) in its NON-TEST source, paired with
// the role that earns it. Unlike the KEYED registries (RegisterEngine(id,…),
// RegisterPageOutBackend(id,…), RegisterWitnessResolver(id,…)) — which are maps where
// multiple drivers coexist — RegisterRegionBackend is a SINGLETON setter: registry.go
// does a bare `reg.regionBackend = b`, so "the last registration wins" (its own doc) and
// every registrant after the first SILENTLY overrides the one before it, with the survivor
// decided by blank-import order. ARCHITECTURE.md frames this seam as a single Resolver
// swap (v0.1 = the content-addressed blob store copy; "zero-copy KV co-residence … is a
// `RegionBackend` swap behind Capability \"zerocopy\""), i.e. exactly ONE backend is live
// at a time. `blob` ships the v0.1 default; it is the only legitimate registrant today.
var regionBackendRole = map[string]string{
	"blob": "the v0.1 default Ref/Resolver backend (content-addressed blob store)",
	// The DELIBERATE, reviewed swap (storedrv/config.go init): the storage-driver router
	// becomes the single live Ref backend, fanning across its blob/blobfs/blobhttp tiers.
	// It registers ONLY when FAK_STORE opts in (unset => inert, blob stays live), and
	// imports blob so blob's init runs first — the last-wins override is order-deterministic.
	"storedrv": "the FAK_STORE-gated storage-driver router (fans Ref resolution across tiers; blob remains the unset default)",
	// The DELIBERATE, reviewed zero-copy override (#448, xenginekv/register.go init): the
	// cross-engine KV co-residence arena becomes the single live Ref backend, handing out
	// RefRegion handles that resolve to VIEWS (Capability "zerocopy"). It registers ONLY when
	// FAK_XENGINE_KV opts in (unset => inert, blob stays live), and imports blob so blob's
	// init runs first — the last-wins override is order-deterministic. This is the documented
	// zerocopy swap this gate's own comment anticipates.
	"xenginekv": "the FAK_XENGINE_KV-gated cross-engine KV co-residence arena (zero-copy RefRegion views; blob remains the unset default)",
}

// TestSingleRegionBackendRegistrant turns ARCHITECTURE.md's "Ref backend is a single
// Resolver swap" seam into a gate, the singleton dual of TestSingleOpenAIChatClient: the
// set of non-test packages that call abi.RegisterRegionBackend MUST be exactly
// regionBackendRole. A NEW registrant is a second backend silently flipping Ref
// resolution by blank-import order (the last-wins corruption); a DECLARED registrant that
// stops calling it is the v0.1 default gone missing — boot with no Ref/Resolver backend.
//
// WHY A GATE (and why it BITES): the compiler cannot enforce this. RegisterRegionBackend
// takes any RegionBackend and assigns it unconditionally (`reg.regionBackend = b`); a
// second call in some other leaf's init() compiles cleanly and just overwrites the field,
// so a regression to two registrants is invisible until the wrong Resolver wins at
// runtime on a particular import order. There is no interface or type that forces
// singularity — this is the same construction-regression class as
// TestRecallReadmissionFoldsRegistry/TestFoldSitesOrderByFoldRank, not the vacuous
// already-typed-enum class of the rejected reason-closure gate.
//
// This is NOT a blunt "≤1 forever" cap: the documented zerocopy override (ship a
// RegionBackend whose Resolver hands out shared-arena handles, advertise "zerocopy") is a
// deliberate, reviewed swap — when it lands, the new singleton owner is added to
// regionBackendRole (and the old default removed if it is being replaced, since only one
// is live), the same conscious review chokepoint as adding a tier or a chatEndpointRole
// holder. The gate forces that swap to be explicit instead of an accidental second init().
//
// Seeded GREEN (2026-06-20): an AST scan of non-test callers finds exactly {blob}
// (blob/store.go's init()); every other RegisterRegionBackend call in the tree is a
// _test.go fixture registering an inlineBackend and is correctly not counted. Adding a
// second non-test registrant — or blob dropping the call — turns the gate RED.
func TestSingleRegionBackendRegistrant(t *testing.T) {
	internal := internalDir(t)
	got := pkgsCallingSelector(t, internal, "abi", "RegisterRegionBackend")

	for pkg := range got {
		if _, ok := regionBackendRole[pkg]; !ok {
			t.Errorf("package internal/%s registers a RegionBackend (abi.RegisterRegionBackend) in "+
				"non-test code, but is not a declared holder. The Ref/Resolver backend is a SINGLETON "+
				"(registry.go: `reg.regionBackend = b`, last-wins) — a second registrant silently flips "+
				"Ref resolution by blank-import order. If this is the deliberate zero-copy override "+
				"(ARCHITECTURE.md), add %q to regionBackendRole with its role and remove the backend it "+
				"replaces; otherwise move the registration off the live path or drop it.", pkg, pkg)
		}
	}
	for pkg, role := range regionBackendRole {
		if !got[pkg] {
			t.Errorf("internal/%s is declared as %q but no longer calls abi.RegisterRegionBackend in "+
				"non-test code. If the singleton Ref/Resolver backend was deliberately moved, update "+
				"regionBackendRole; otherwise the v0.1 default Resolver went missing and the kernel boots "+
				"with no Ref backend.", pkg, role)
		}
	}
}

// pkgsCallingSelector is the set-returning plural of pkgCallsSelector: it returns the set
// of internal packages whose NON-TEST source contains a call of the form
// <recv>.<name>(...) — read from the AST, so a doc comment or a call living only in
// _test.go does not count. Used by the singleton-registrant gates to find WHICH packages
// register a last-wins backend, not just whether a single named one does.
func pkgsCallingSelector(t *testing.T, internal, recv, name string) map[string]bool {
	t.Helper()
	got := map[string]bool{}
	for _, pkg := range goPackageDirs(t, internal) {
		if pkgCallsSelector(t, internal, pkg, recv, name) {
			got[pkg] = true
		}
	}
	return got
}

// pkgsCallingSelectorWithStringArg is the id-scoped refinement of pkgsCallingSelector:
// it returns the set of internal packages whose NON-TEST source contains a call of the
// form <recv>.<name>(<arg0>, ...) where the FIRST argument is the exact string literal
// `arg0`. It is the tool the KEYED-registry gates need: pkgsCallingSelector answers
// "who calls RegisterPageOutBackend at all" (every id), but a map-keyed registry is
// plural by design — multiple distinct ids legitimately coexist — so the load-bearing
// invariant is per-ID ("who registers id X"), not per-symbol. Reading the AST (not a
// text grep) means a doc comment or a _test.go fixture naming the same id does not count;
// matching arg0 as a BasicLit string means RegisterPageOutBackend("blob", …) is found
// while RegisterPageOutBackend(someVar, …) or a different literal id is not.
func pkgsCallingSelectorWithStringArg(t *testing.T, internal, recv, name, arg0 string) map[string]bool {
	t.Helper()
	got := map[string]bool{}
	for _, pkg := range goPackageDirs(t, internal) {
		fset := token.NewFileSet()
		parsed, err := parser.ParseDir(fset, filepath.Join(internal, pkg),
			func(fi fs.FileInfo) bool { return !strings.HasSuffix(fi.Name(), "_test.go") }, 0)
		if err != nil {
			t.Fatalf("parse %s: %v", pkg, err)
		}
		for _, p := range parsed {
			for _, f := range p.Files {
				ast.Inspect(f, func(n ast.Node) bool {
					if got[pkg] {
						return false
					}
					call, ok := n.(*ast.CallExpr)
					if !ok || len(call.Args) < 1 {
						return true
					}
					sel, ok := call.Fun.(*ast.SelectorExpr)
					if !ok {
						return true
					}
					id, ok := sel.X.(*ast.Ident)
					if !ok || id.Name != recv || sel.Sel.Name != name {
						return true
					}
					if lit, ok := stringLit(call.Args[0]); ok && lit == arg0 {
						got[pkg] = true
						return false
					}
					return true
				})
			}
		}
	}
	return got
}

// pageOutDefaultRole names every internal package allowed to register the v0.1 DEFAULT
// page-out codec — the backend stored under id "blob" — in its NON-TEST source, paired
// with the role that earns it. Unlike abi.RegisterRegionBackend (a SINGLETON setter,
// pinned by TestSingleRegionBackendRegistrant), abi.RegisterPageOutBackend is a KEYED
// map: registry.go does `reg.pageOut[id] = b`, so distinct ids coexist by design (the
// doc-promised "headroom sidecar later" is a SECOND, differently-keyed codec, perfectly
// legitimate). What is NOT legitimate is a second registrant of the SAME id "blob": the
// map assignment is unconditional and last-write-wins, so a second `RegisterPageOutBackend
// ("blob", …)` in another leaf's init() silently swaps the process-wide default page-out
// codec, with the survivor decided by blank-import order. ARCHITECTURE.md frames this row
// as "Go blob store default; headroom sidecar later" — exactly one default per id, and the
// default id is "blob". `blob` ships it (blob/store.go init, the same object it registers
// as the RegionBackend); it is the only legitimate registrant of id "blob" today.
var pageOutDefaultRole = map[string]string{
	"blob": "the v0.1 default page-out codec registered under id \"blob\" (content-addressed blob store)",
}

// pageOutDefaultID is the id literal the v0.1 default page-out codec is registered under.
const pageOutDefaultID = "blob"

// TestSinglePageOutBlobRegistrant is the keyed-registry analogue of
// TestSingleRegionBackendRegistrant: where that gate pins a SINGLETON setter, this one
// pins a single ID within a plural, map-keyed registry. The set of non-test packages that
// call abi.RegisterPageOutBackend with the literal first argument "blob" MUST be exactly
// pageOutDefaultRole. A NEW registrant of id "blob" is a second default codec silently
// flipping page-out by blank-import order (the last-write-wins map corruption); a DECLARED
// registrant that stops registering "blob" is the v0.1 default page-out backend gone
// missing — the context-MMU pages a quarantined result out to a handle with no codec under
// the default id.
//
// WHY A GATE (and why it BITES): the compiler cannot enforce this. RegisterPageOutBackend
// takes any (string, PageOutBackend) and does `reg.pageOut[id] = b` unconditionally; a
// second call with id "blob" in some other leaf's init() compiles cleanly and just
// overwrites the map entry, so a regression to two "blob" registrants is invisible until
// the wrong codec wins at runtime on a particular import order. There is no interface or
// type that forces per-id singularity — this is the same construction-regression class as
// TestSingleRegionBackendRegistrant / TestFoldSitesOrderByFoldRank, not the vacuous
// already-typed class of the rejected reason-closure gate. It is DISTINCT from the region-
// backend gate: that one is symbol-scoped (RegisterRegionBackend is a singleton, so any
// second caller fails); this one is (symbol, id)-scoped, because the page-out registry is
// plural — only a second caller of the SAME default id is the regression, and a sidecar
// codec under a new id is correctly allowed.
//
// This is NOT a blunt "≤1 PageOutBackend forever" cap: the documented headroom sidecar
// (a second page-out codec under a DIFFERENT id) is a deliberate, designed extension and
// does not trip this gate at all — only a re-registration of the "blob" default does. A
// deliberate swap of the default itself (ship a new codec under id "blob") is the same
// conscious review chokepoint as adding a tier or a regionBackendRole holder: update
// pageOutDefaultRole to name the new owner and remove the one it replaces.
//
// Seeded GREEN (2026-06-20): an AST scan of non-test callers with arg0=="blob" finds
// exactly {blob} (blob/store.go's init(), which registers the same object as both the
// RegionBackend singleton and the "blob" page-out default); every other
// RegisterPageOutBackend call in the tree is a _test.go fixture and is correctly not
// counted. Adding a second non-test registrant of id "blob" — or blob dropping the call —
// turns the gate RED.
func TestSinglePageOutBlobRegistrant(t *testing.T) {
	internal := internalDir(t)
	got := pkgsCallingSelectorWithStringArg(t, internal, "abi", "RegisterPageOutBackend", pageOutDefaultID)

	for pkg := range got {
		if _, ok := pageOutDefaultRole[pkg]; !ok {
			t.Errorf("package internal/%s registers the page-out codec under id %q "+
				"(abi.RegisterPageOutBackend(%q, …)) in non-test code, but is not a declared holder. The "+
				"page-out registry is a map keyed by id (registry.go: `reg.pageOut[id] = b`, last-write-"+
				"wins), so a second registrant of the DEFAULT id silently flips the process-wide page-out "+
				"codec by blank-import order. If this is a deliberate swap of the v0.1 default, add %q to "+
				"pageOutDefaultRole and remove the codec it replaces; if it is a NEW sidecar codec, register "+
				"it under a DIFFERENT id (the map is plural by design) instead of %q.",
				pkg, pageOutDefaultID, pageOutDefaultID, pkg, pageOutDefaultID)
		}
	}
	for pkg, role := range pageOutDefaultRole {
		if !got[pkg] {
			t.Errorf("internal/%s is declared as %q but no longer registers the page-out codec under id "+
				"%q in non-test code. If the default page-out backend was deliberately moved, update "+
				"pageOutDefaultRole; otherwise the v0.1 default codec went missing and the context-MMU pages "+
				"out to the default id with no backend.", pkg, role, pageOutDefaultID)
		}
	}
}

// witnessResolverRole names every internal package allowed to register the witness
// resolver under id "dos_verify" — the one that backs the require-witness verdict — in
// its NON-TEST source, paired with the role that earns it. Like RegisterPageOutBackend,
// abi.RegisterWitnessResolver is a KEYED map: registry.go does `reg.witnesses[id] = w`
// UNCONDITIONALLY (no clash panic, unlike RegisterOp/RegisterVerdictKind), so distinct
// ids coexist by design (a future "dos_status" or "external_anchor" resolver is a SECOND,
// differently-keyed witness, perfectly legitimate). What is NOT legitimate is a second
// registrant of the SAME id "dos_verify": the map assignment is last-write-wins, so a
// second `RegisterWitnessResolver("dos_verify", …)` in another leaf's init() silently
// swaps the resolver the kernel consults on a RequireWitness verdict — the in-process
// git-evidence effect-verify that turns a claimed effect into a corroborated one — with
// the survivor decided by blank-import order. That resolver is the T4 effect-verification
// seam (SECURITY-DRIVERS): the "don't believe the agent" mechanism itself. `witness`
// ships the v0.1 default (witness.go init); it is the only legitimate registrant of id
// "dos_verify" today.
var witnessResolverRole = map[string]string{
	"witness": "the v0.1 require-witness resolver registered under id \"dos_verify\" (in-process git-evidence effect-verify)",
}

// witnessResolverID is the id literal the v0.1 require-witness resolver is registered under.
const witnessResolverID = "dos_verify"

// TestSingleWitnessResolverRegistrant is the witness-seam sibling of
// TestSinglePageOutBlobRegistrant: a single ID within a plural, map-keyed registry. The
// set of non-test packages that call abi.RegisterWitnessResolver with the literal first
// argument "dos_verify" MUST be exactly witnessResolverRole. A NEW registrant of id
// "dos_verify" is a second resolver silently flipping require-witness resolution by
// blank-import order (the last-write-wins map corruption); a DECLARED registrant that
// stops registering "dos_verify" is the v0.1 effect-verifier gone missing — the kernel
// consults abi.Witnesses() on a RequireWitness verdict and finds no resolver under the id
// the require-witness gate names.
//
// WHY A GATE (and why it BITES): the compiler cannot enforce this. RegisterWitnessResolver
// takes any (string, WitnessResolver) and does `reg.witnesses[id] = w` unconditionally —
// no clash panic — so a second call with id "dos_verify" in some other leaf's init()
// compiles cleanly and just overwrites the map entry, and the regression to two
// "dos_verify" registrants is invisible until the wrong resolver wins at runtime on a
// particular import order. There is no interface or type that forces per-id singularity —
// this is the same construction-regression class as TestSinglePageOutBlobRegistrant /
// TestSingleRegionBackendRegistrant, not the vacuous already-typed class of the rejected
// reason-closure gate. It is (symbol, id)-scoped, not symbol-scoped: the witness registry
// is plural by design (a sidecar resolver under a NEW id is correctly allowed), so only a
// second caller of the SAME id "dos_verify" is the regression.
//
// This is NOT a blunt "≤1 WitnessResolver forever" cap: the documented DOS read-back /
// external-anchor resolvers (a second witness under a DIFFERENT id) are deliberate, designed
// extensions and do not trip this gate at all. A deliberate swap of the "dos_verify" default
// itself is the same conscious review chokepoint as adding a tier or a pageOutDefaultRole
// holder: update witnessResolverRole to name the new owner and remove the one it replaces.
//
// Seeded GREEN (2026-06-20): an AST scan of non-test callers with arg0=="dos_verify" finds
// exactly {witness} (witness.go's init()); the kernel's witness tests register id "test"
// (a different id) and are correctly not counted. Adding a second non-test registrant of id
// "dos_verify" — or witness dropping the call — turns the gate RED.
func TestSingleWitnessResolverRegistrant(t *testing.T) {
	internal := internalDir(t)
	got := pkgsCallingSelectorWithStringArg(t, internal, "abi", "RegisterWitnessResolver", witnessResolverID)

	for pkg := range got {
		if _, ok := witnessResolverRole[pkg]; !ok {
			t.Errorf("package internal/%s registers a witness resolver under id %q "+
				"(abi.RegisterWitnessResolver(%q, …)) in non-test code, but is not a declared holder. The "+
				"witness registry is a map keyed by id (registry.go: `reg.witnesses[id] = w`, last-write-"+
				"wins, no clash panic), so a second registrant of the require-witness id silently flips the "+
				"resolver the kernel consults on a RequireWitness verdict by blank-import order. If this is a "+
				"deliberate swap of the v0.1 default, add %q to witnessResolverRole and remove the resolver it "+
				"replaces; if it is a NEW sidecar resolver, register it under a DIFFERENT id (the map is plural "+
				"by design) instead of %q.",
				pkg, witnessResolverID, witnessResolverID, pkg, witnessResolverID)
		}
	}
	for pkg, role := range witnessResolverRole {
		if !got[pkg] {
			t.Errorf("internal/%s is declared as %q but no longer registers the witness resolver under id "+
				"%q in non-test code. If the default require-witness resolver was deliberately moved, update "+
				"witnessResolverRole; otherwise the v0.1 effect-verifier went missing and the kernel resolves a "+
				"RequireWitness verdict to no resolver under the id the gate names.", pkg, role, witnessResolverID)
		}
	}
}

// adjudicatorExecAllow is the declared escape-list for TestEveryAdjudicatorIsExecFree: a
// package that registers an Adjudicator (so it runs on the live decision chain) AND
// legitimately imports os/exec, keyed to a one-line justification — the same conscious
// review chokepoint as regOffList / interpreterExecAllow. `shipgate` registers a rank-40
// Adjudicator but is the RSI/ship harness (git + worktree orchestration), explicitly NOT
// the dispatch hot path — its own source says so (shipgate.go: "this is the RSI harness,
// NOT the dispatch hot path, so the os/exec-absence proof … does not apply here"). A new
// entry here is a deliberate, reviewed decision that a given Adjudicator's exec is off the
// live tool-call decision path, not a silent pass.
var adjudicatorExecAllow = map[string]string{
	"shipgate": "RSI/ship harness — git + worktree exec, registered as a rank-40 Adjudicator but off the dispatch hot path (shipgate.go)",
}

// TestEveryAdjudicatorIsExecFree is the "any Adjudicator" clause of DIRECTION.md's
// reviewer-grep #1 turned into a gate, and the AST-derived superset of
// TestHotPathHasNoExec. That gate checks a HAND-MAINTAINED hotPath list of seven leaves
// for os/exec; this one derives the set of packages that actually register an Adjudicator
// (abi.RegisterAdjudicator, read from the AST) and asserts NONE of them imports os/exec —
// except a declared escape-list. DIRECTION.md: os/exec "must never appear in adjudicator,
// kernel, vdso, ctxmmu, or any Adjudicator." A registered Adjudicator is, by definition, a
// rung on the live PDP/PEP decision chain; an os/exec import there puts a per-decide
// subprocess on the exact path fak exists to keep interpreter-free.
//
// WHY A GATE (and why it BITES): the hotPath list is manually curated, so it silently lags
// reality. Today four packages register an Adjudicator but are NOT in hotPath — engine
// (rank 12), ifc (rank 30), plancfi (rank 25), shipgate (rank 40) — so TestHotPathHasNoExec
// does not check them at all. A NEW Adjudicator rung (or one of those four growing an
// os/exec import) would pass TestHotPathHasNoExec untouched while quietly adding a
// subprocess to the live decision chain. Deriving the set from RegisterAdjudicator callers
// closes that lag: a new rung is checked the moment it registers, with no hotPath edit
// required. This is the same construction-regression class as the registrant-singleton
// gates — nothing in the type system stops an Adjudicator-registering package from importing
// os/exec — not the vacuous already-typed class of the rejected reason-closure gate.
//
// Seeded GREEN (2026-06-20): eight packages register an Adjudicator (adjudicator, engine,
// grammar, ifc, plancfi, preflight, ratelimit, shipgate); seven import no os/exec, and the
// only one that does — shipgate, the RSI/ship harness off the dispatch path — is on
// adjudicatorExecAllow with that justification. Adding os/exec to any other Adjudicator
// package, or a new exec-importing Adjudicator rung, turns the gate RED.
func TestEveryAdjudicatorIsExecFree(t *testing.T) {
	internal := internalDir(t)
	adjudicators := pkgsCallingSelector(t, internal, "abi", "RegisterAdjudicator")

	// Deterministic order so a multi-offender failure reads the same on every run.
	pkgs := make([]string, 0, len(adjudicators))
	for pkg := range adjudicators {
		pkgs = append(pkgs, pkg)
	}
	sort.Strings(pkgs)

	for _, pkg := range pkgs {
		if _, allowed := adjudicatorExecAllow[pkg]; allowed {
			continue
		}
		for _, imp := range imports(t, internal, pkg) {
			if imp == "os/exec" {
				t.Errorf("Adjudicator-registering package internal/%s imports os/exec — it runs on the "+
					"live PDP/PEP decision chain (abi.RegisterAdjudicator), and DIRECTION.md forbids os/exec "+
					"in any Adjudicator: a per-decide subprocess is exactly the boundary fak exists to remove. "+
					"Move the exec to an off-path (non-Adjudicator) package, or, if this Adjudicator is "+
					"genuinely off the dispatch hot path (as shipgate's RSI harness is), add %q to "+
					"adjudicatorExecAllow with a one-line justification.", pkg, pkg)
			}
		}
	}

	// The escape-list must not rot: an allow entry for a package that no longer registers
	// an Adjudicator (or no longer imports os/exec) is a stale exception masking nothing —
	// remove it so the list keeps naming only real, current exceptions.
	for pkg, why := range adjudicatorExecAllow {
		if !adjudicators[pkg] {
			t.Errorf("adjudicatorExecAllow names internal/%s (%q) but it no longer registers an "+
				"Adjudicator — a stale exception. Remove it from adjudicatorExecAllow.", pkg, why)
			continue
		}
		importsExec := false
		for _, imp := range imports(t, internal, pkg) {
			if imp == "os/exec" {
				importsExec = true
				break
			}
		}
		if !importsExec {
			t.Errorf("adjudicatorExecAllow names internal/%s (%q) but it no longer imports os/exec — a "+
				"stale exception that grants an exemption nothing uses. Remove it from adjudicatorExecAllow.",
				pkg, why)
		}
	}
}

// resultAdmitterExecAllow is the declared escape-list for TestEveryResultAdmitterIsExecFree,
// the result-side twin of adjudicatorExecAllow. A package that registers a ResultAdmitter
// (so it runs on the live WRITE-TIME result-admission chain) AND legitimately imports
// os/exec would be keyed here with a one-line justification. EMPTY at green: today every
// ResultAdmitter (ctxmmu, ifc, normgate) is exec-free, so no exception is needed — unlike
// the Adjudicator side, where shipgate's off-path RSI harness earns one. A future entry is
// a deliberate, reviewed decision that a given ResultAdmitter's exec is off the live path.
var resultAdmitterExecAllow = map[string]string{}

// TestEveryResultAdmitterIsExecFree is the result-admission dual of
// TestEveryAdjudicatorIsExecFree: it extends DIRECTION.md's no-os/exec-on-the-decision-path
// thesis from the pre-call Adjudicator chain to the WRITE-TIME ResultAdmitter chain. A
// ResultAdmitter (abi.RegisterResultAdmitter) is the post-tool dual of an Adjudicator —
// after the engine produces a Result, the kernel folds these to decide whether it may enter
// context (Allow), must be held out (Quarantine), or rewritten to a pointer (Transform).
// That fold runs on every tool-call result, so it is as much a live decision rung as the
// pre-call chain; an os/exec import there puts a per-decide subprocess on the result side of
// the exact boundary fak exists to keep interpreter-free. The set of ResultAdmitter
// registrants is derived from the AST (not a hand-kept list), so a new result-admission rung
// is checked the moment it registers — the same lag-closing move as the Adjudicator gate.
//
// WHY A GATE (and why it BITES): the compiler cannot enforce this. Nothing stops a
// ResultAdmitter-registering package from importing os/exec; the regression (a subprocess on
// the result-admission path) is invisible until it runs. TestHotPathHasNoExec does not cover
// these — its hand-kept hotPath list names none of ctxmmu's result side, ifc, or normgate as
// result-admitters — so without this gate the entire write-time chain is unchecked for exec.
// Same construction-regression class as the Adjudicator/registrant-singleton gates, not the
// vacuous already-typed class of the rejected reason-closure gate.
//
// Seeded GREEN (2026-06-20): three packages register a ResultAdmitter (ctxmmu rank 10, ifc
// rank 20, normgate rank 5); all three import no os/exec, so resultAdmitterExecAllow is
// empty. Adding os/exec to any of them — or a new exec-importing ResultAdmitter rung — turns
// the gate RED.
func TestEveryResultAdmitterIsExecFree(t *testing.T) {
	internal := internalDir(t)
	admitters := pkgsCallingSelector(t, internal, "abi", "RegisterResultAdmitter")

	// Deterministic order so a multi-offender failure reads the same on every run.
	pkgs := make([]string, 0, len(admitters))
	for pkg := range admitters {
		pkgs = append(pkgs, pkg)
	}
	sort.Strings(pkgs)

	for _, pkg := range pkgs {
		if _, allowed := resultAdmitterExecAllow[pkg]; allowed {
			continue
		}
		for _, imp := range imports(t, internal, pkg) {
			if imp == "os/exec" {
				t.Errorf("ResultAdmitter-registering package internal/%s imports os/exec — it runs on the "+
					"live write-time result-admission chain (abi.RegisterResultAdmitter), the post-tool dual "+
					"of the Adjudicator chain, so DIRECTION.md's no-os/exec-on-the-decision-path rule applies "+
					"to it too: a per-decide subprocess is the boundary fak exists to remove. Move the exec to "+
					"an off-path (non-ResultAdmitter) package, or, if this admitter is genuinely off the live "+
					"path, add %q to resultAdmitterExecAllow with a one-line justification.", pkg, pkg)
			}
		}
	}

	// The escape-list must not rot (mirrors the Adjudicator gate): an allow entry for a
	// package that no longer registers a ResultAdmitter — or no longer imports os/exec — is a
	// stale exception masking nothing.
	for pkg, why := range resultAdmitterExecAllow {
		if !admitters[pkg] {
			t.Errorf("resultAdmitterExecAllow names internal/%s (%q) but it no longer registers a "+
				"ResultAdmitter — a stale exception. Remove it from resultAdmitterExecAllow.", pkg, why)
			continue
		}
		importsExec := false
		for _, imp := range imports(t, internal, pkg) {
			if imp == "os/exec" {
				importsExec = true
				break
			}
		}
		if !importsExec {
			t.Errorf("resultAdmitterExecAllow names internal/%s (%q) but it no longer imports os/exec — a "+
				"stale exception that grants an exemption nothing uses. Remove it from resultAdmitterExecAllow.",
				pkg, why)
		}
	}
}

// selfModifyWitnessTrees is the set of trees that, if an RSI loop could quietly self-edit
// them, would let the loop grade its own homework — so the adjudicator's self-modify floor
// MUST guard every one. Each entry is paired with a sample write target inside it; coverage
// is checked against that target exactly the way Adjudicate matches at runtime (matchGlob:
// the first SelfModifyGlobs fragment CONTAINED IN the path). The witness machinery is:
//
//   - internal/adjudicator — the policy-decision point itself (the guard's own code).
//   - internal/architest   — these gates; if self-edited, the loop could delete the floor.
//   - internal/shipgate    — the RSI/ship harness that lands the candidate.
//   - dos.toml             — the workspace lane taxonomy + stamp grammar the referee binds to.
//
// These are exactly the trees #172 Hole 2 (commit ff2dda6) added to SelfModifyGlobs. That
// commit's regression floor (TestSelfModifyGuardsWitnessMachinery) lives IN the adjudicator
// package and asserts the running policy denies a write into each tree — the behavioral
// half. This gate is the CROSS-PACKAGE half ff2dda6 deliberately left separate: it reads the
// adjudicator SOURCE (not a constructed Policy) and asserts the literal SelfModifyGlobs set
// in BOTH deployable policy constructors still spells out every witness tree. The two are
// independent on purpose — the in-package test could be deleted by the very self-edit it
// guards against (internal/adjudicator is itself a witness tree), so an out-of-package gate
// that re-derives the floor from source is the belt to its suspenders.
var selfModifyWitnessTrees = map[string]string{
	"internal/adjudicator/": "internal/adjudicator/decide.go",
	"internal/architest/":   "internal/architest/architest_test.go",
	"internal/shipgate/":    "internal/shipgate/gate.go",
	"dos.toml":              "dos.toml",
}

// selfModifyPolicyCtors are the adjudicator constructors that return a DEPLOYABLE Policy —
// the ones whose SelfModifyGlobs actually guard a running agent. Every one must cover the
// witness trees; a future third deployable policy is added here so it is held to the same
// floor (the forced-choice review gate, like the tier table).
var selfModifyPolicyCtors = []string{"DefaultPolicy", "DevAgentPolicy"}

// selfModifyGlobsFromCtor parses internal/adjudicator's non-test source and returns the
// string-literal SelfModifyGlobs slice assigned inside the named constructor (e.g.
// "DefaultPolicy"). It reads the SOURCE rather than importing adjudicator and calling the
// constructor on purpose: the gate must witness what the code literally declares, so a glob
// built at runtime from a variable (which a self-editing loop could make say anything) can't
// satisfy the floor — only a visible string literal in the constructor does. Returns nil if
// the constructor or its SelfModifyGlobs field is absent (the caller turns that into a
// failure naming which constructor went missing).
func selfModifyGlobsFromCtor(t *testing.T, internal, ctor string) []string {
	t.Helper()
	fset := token.NewFileSet()
	parsed, err := parser.ParseDir(fset, filepath.Join(internal, "adjudicator"),
		func(fi fs.FileInfo) bool { return !strings.HasSuffix(fi.Name(), "_test.go") }, 0)
	if err != nil {
		t.Fatalf("parse adjudicator: %v", err)
	}
	for _, p := range parsed {
		for _, f := range p.Files {
			for _, decl := range f.Decls {
				fn, ok := decl.(*ast.FuncDecl)
				if !ok || fn.Name.Name != ctor || fn.Recv != nil || fn.Body == nil {
					continue
				}
				var globs []string
				ast.Inspect(fn.Body, func(n ast.Node) bool {
					kv, ok := n.(*ast.KeyValueExpr)
					if !ok {
						return true
					}
					key, ok := kv.Key.(*ast.Ident)
					if !ok || key.Name != "SelfModifyGlobs" {
						return true
					}
					comp, ok := kv.Value.(*ast.CompositeLit)
					if !ok {
						return true
					}
					for _, elt := range comp.Elts {
						if s, ok := stringLit(elt); ok {
							globs = append(globs, s)
						}
					}
					return false
				})
				if globs != nil {
					return globs
				}
			}
		}
	}
	return nil
}

// TestSelfModifyFloorCoversWitnessTrees is the cross-package half of the #172 Hole 2
// self-modify floor. The self-improve loop edits fak's own kernel, so the self-modify guard
// is load-bearing: it is what stops a self-improving agent from quietly rewriting the very
// invariants that witness it. Hole 2 extended adjudicator.Policy.SelfModifyGlobs to cover the
// WITNESS machinery — internal/adjudicator (the guard), internal/architest (these gates),
// internal/shipgate (the ship harness), and dos.toml (the taxonomy/stamp grammar) — so a loop
// that can edit its own witness can't launder a bad change past the gate.
//
// WHY A GATE (and why it BITES): nothing in the type system keeps a witness tree in
// SelfModifyGlobs; it is a hand-written []string in two constructors. A refactor that drops
// "internal/architest/" (or a self-editing loop that does so deliberately) silently reopens
// the hole — the floor would compile and every other test would pass while the loop regained
// the ability to grade its own homework. ff2dda6 already added a behavioral floor
// (TestSelfModifyGuardsWitnessMachinery) IN the adjudicator package, but that test is itself
// inside a witness tree (internal/adjudicator), so the same self-edit that drops a tree from
// the floor could drop the test that guards it. This gate closes that circularity: it lives
// OUTSIDE adjudicator, re-derives the floor from adjudicator's source, and fails if either
// deployable policy constructor stops naming a witness tree — the belt to ff2dda6's suspenders.
//
// Seeded GREEN (2026-06-20): DefaultPolicy and DevAgentPolicy both enumerate every witness
// tree in selfModifyWitnessTrees (DefaultPolicy via explicit entries plus the fak/internal/
// catch-all; both name dos.toml explicitly). Dropping any witness tree from either
// constructor's SelfModifyGlobs literal turns this gate RED.
func TestSelfModifyFloorCoversWitnessTrees(t *testing.T) {
	internal := internalDir(t)

	// Deterministic tree order so a multi-gap failure reads the same on every run.
	trees := make([]string, 0, len(selfModifyWitnessTrees))
	for tree := range selfModifyWitnessTrees {
		trees = append(trees, tree)
	}
	sort.Strings(trees)

	for _, ctor := range selfModifyPolicyCtors {
		globs := selfModifyGlobsFromCtor(t, internal, ctor)
		if globs == nil {
			t.Errorf("adjudicator.%s has no SelfModifyGlobs string-literal slice in its source — "+
				"the self-modify floor is the guard that stops a self-improving loop from rewriting the "+
				"invariants that witness it; a deployable Policy constructor MUST declare it as a visible "+
				"[]string literal (not a runtime-built value a self-edit could make say anything). If %s "+
				"was renamed or is no longer deployable, update selfModifyPolicyCtors.", ctor, ctor)
			continue
		}
		// matchGlob's runtime semantics: a glob covers a target if the glob fragment is
		// CONTAINED IN the target path. Mirror that here against each witness tree's sample.
		covers := func(target string) bool {
			for _, g := range globs {
				if g != "" && strings.Contains(target, g) {
					return true
				}
			}
			return false
		}
		for _, tree := range trees {
			target := selfModifyWitnessTrees[tree]
			if !covers(target) {
				t.Errorf("adjudicator.%s SelfModifyGlobs does not cover witness tree %q "+
					"(no glob fragment is contained in %q).\nSelfModifyGlobs=%v\n"+
					"This is the #172 Hole 2 floor: %s is part of the witness machinery — if a self-improving "+
					"loop can write there, it can grade its own homework. Re-add a glob covering %q to %s "+
					"(the in-package TestSelfModifyGuardsWitnessMachinery is its behavioral twin).",
					ctor, tree, target, globs, tree, tree, ctor)
			}
		}
	}
}

// shellSelfModifyCallee is the decide-path function that gates a SHELL/Bash write into a
// guarded tree (#172 Hole 1). decide.go calls it as `commandSelfModify(args, p.SelfModifyGlobs)`
// and denies on a non-empty result. The exact name is the wiring this gate witnesses: a
// self-edit that deletes the call re-opens the hole, and renaming the function is a visible,
// reviewable edit that must update this constant too.
const shellSelfModifyCallee = "commandSelfModify"

// bodyCallsFunc reports whether the named top-level function/method (matched by name across
// every non-test file of the package at dir) contains a call to callee somewhere in its body.
// Like selfModifyGlobsFromCtor it reads SOURCE, not a built value: the gate must witness that
// the decide path literally invokes the shell-write guard, so a runtime-built or reflected
// dispatch (which a self-editing loop could make say anything) cannot satisfy it - only a
// visible call expression does. Returns false if the function is absent (the caller turns that
// into a failure naming which function went missing).
func bodyCallsFunc(t *testing.T, dir, fnName, callee string) bool {
	t.Helper()
	fset := token.NewFileSet()
	parsed, err := parser.ParseDir(fset, dir,
		func(fi fs.FileInfo) bool { return !strings.HasSuffix(fi.Name(), "_test.go") }, 0)
	if err != nil {
		t.Fatalf("parse %s: %v", dir, err)
	}
	found := false
	for _, p := range parsed {
		for _, f := range p.Files {
			for _, decl := range f.Decls {
				fn, ok := decl.(*ast.FuncDecl)
				if !ok || fn.Name.Name != fnName || fn.Body == nil {
					continue
				}
				ast.Inspect(fn.Body, func(n ast.Node) bool {
					call, ok := n.(*ast.CallExpr)
					if !ok {
						return true
					}
					if id, ok := call.Fun.(*ast.Ident); ok && id.Name == callee {
						found = true
						return false
					}
					return true
				})
			}
		}
	}
	return found
}

// TestShellSelfModifyGuardWiredInDecide is the cross-package half of the #172 Hole 1 self-modify
// floor - the Bash-write twin of TestSelfModifyFloorCoversWitnessTrees. Hole 1 is that the
// file-write self-modify guard never fires for a SHELL tool: a `Bash` call carries its write
// target INSIDE the command string (`sed -i internal/abi/x.go`, `> .dos/state`, `git apply`),
// not a `path` arg, and the tool NAME contains no write-word, so the targetPath guard skips it.
// decide.go closes that hole by ALSO calling commandSelfModify(args, p.SelfModifyGlobs) on the
// decision path and denying with ReasonSelfModify when a write-shaped shell verb targets a
// guarded glob.
//
// WHY A GATE (and why it BITES): nothing in the type system keeps that call on the decide path.
// A refactor - or a self-editing loop that does so deliberately - could delete the
// commandSelfModify call from decide.go and the kernel would compile and pass every other test
// while the Bash write path silently regained the ability to launder a self-edit (a shell `sed
// -i` into internal/abi/) past the floor. #172 Hole 1 (commit 811beea) added a behavioral floor
// (TestSelfModifyGuardsShellWritePath) IN the adjudicator package, but that test is itself inside
// a witness tree (internal/adjudicator), so the same self-edit that drops the shell guard could
// drop the test that guards it - the exact circularity TestSelfModifyFloorCoversWitnessTrees was
// written to break for the glob set. This gate is the belt to that test's suspenders: it lives
// OUTSIDE adjudicator, re-derives the wiring from adjudicator's source, and fails if decide.go
// stops invoking the shell-write guard.
//
// Seeded GREEN (2026-06-20): decide.go's Adjudicate method calls commandSelfModify (decide.go:234).
// Deleting that call - re-opening Hole 1 - turns this gate RED from outside every guarded tree.
func TestShellSelfModifyGuardWiredInDecide(t *testing.T) {
	internal := internalDir(t)
	adjDir := filepath.Join(internal, "adjudicator")

	// The decision entry point is the adjudicator's Adjudicate method (the PDP/PEP rung every
	// tool call traverses). It is the body that MUST consult the shell-write guard; if the
	// method is renamed, update the name here (a visible, reviewable edit).
	const decideFn = "Adjudicate"
	if !bodyCallsFunc(t, adjDir, decideFn, shellSelfModifyCallee) {
		t.Errorf("adjudicator.%s does not call %s - the #172 Hole 1 shell-write self-modify guard "+
			"is NOT on the decision path. A Bash/exec tool carries its write target inside the "+
			"command string (e.g. `sed -i internal/abi/x.go`, `> .dos/state`), so the file-write "+
			"guard (targetPath) never fires for it; decide MUST also call %s(args, p.SelfModifyGlobs) "+
			"so a shell write into a guarded tree is denied. Without that call a self-improving loop "+
			"can launder a self-edit past the floor through the shell. Re-add the call in decide.go "+
			"(the in-package TestSelfModifyGuardsShellWritePath is its behavioral twin), or, if %s was "+
			"renamed, update decideFn/shellSelfModifyCallee in this gate.",
			decideFn, shellSelfModifyCallee, shellSelfModifyCallee, decideFn)
	}
}

// engineDriverRole names every internal package allowed to register an inference-engine
// driver under a given id (abi.RegisterEngine(id, …)) in its NON-TEST source, keyed by the
// engine id, paired with the role that earns it. Like RegisterPageOutBackend and
// RegisterWitnessResolver, abi.RegisterEngine is a KEYED map: registry.go does
// `reg.engines[id] = d` UNCONDITIONALLY (no clash panic, unlike RegisterOp /
// RegisterVerdictKind), so distinct ids coexist by design — the kernel SELECTS one by id at
// construction (kernel.New("inkernel")). What is NOT legitimate is a second registrant of the
// SAME id: the map assignment is last-write-wins, so a second `RegisterEngine("inkernel", …)`
// in another leaf's init() silently swaps the engine the kernel runs, with the survivor
// decided by blank-import order. The three v0.1 engines each own a distinct id:
//
//   - "inkernel"   — modelengine: the in-kernel Go model-fusion engine (the dogfood default).
//   - "localtools" — agent: the local tool-call engine cmd/fak wires directly.
//   - "fakread"    — agent: the read-only engine for fak_read gateway calls.
//   - "mock"       — engine: the routing/mock engine used by the engine-route capability.
//
// A NEW engine under a NEW id is correctly allowed (the map is plural by design); only a
// second registrant of an EXISTING id is the regression this gate catches.
var engineDriverRole = map[string]map[string]string{
	"inkernel":   {"modelengine": "the in-kernel Go model-fusion engine (dogfood default)"},
	"fakread":    {"agent": "the read-only engine for fak_read gateway calls"},
	"localtools": {"agent": "the local tool-call engine wired by cmd/fak"},
	"mock":       {"engine": "the routing/mock engine behind the engine.route capability"},
	"vllm":       {"engine": "the vLLM V1 public HTTP/KV-events/metrics adapter"},
}

// resolveEngineIDArg returns the engine-id string a RegisterEngine call's first argument
// names, resolving BOTH a direct string literal (RegisterEngine("localtools", …)) AND a
// same-package string const (RegisterEngine(EngineID, …) where `const EngineID = "inkernel"`).
// The const arm matters: modelengine registers via the const EngineID, so a literal-only
// scan (like pkgsCallingSelectorWithStringArg) would be BLIND to the "inkernel" registrant —
// the exact engine the dogfood path selects — and could not enforce its singularity. consts
// is the package's string-const map (name -> value); a first arg that is neither a literal
// nor a known same-package const returns ("", false), which the caller treats as an
// unreadable id (fail-closed: a dynamically-built engine id on a registrant is itself a
// regression, since the gate cannot prove singularity for it).
func resolveEngineIDArg(arg ast.Expr, consts map[string]string) (string, bool) {
	if s, ok := stringLit(arg); ok {
		return s, true
	}
	if id, ok := arg.(*ast.Ident); ok {
		if v, ok := consts[id.Name]; ok {
			return v, true
		}
	}
	return "", false
}

// stringConsts parses internal/<pkg>'s non-test source and returns its package-level
// string-valued consts as a name->value map (e.g. {"EngineID": "inkernel"}). Only plain
// string-literal const declarations are read — a const built from an expression is not a
// stable id the gate can resolve, and is correctly omitted (resolveEngineIDArg then treats a
// registration keyed on it as an unreadable id). Reads SOURCE via the AST, never a built
// value, so the resolved id is what the code literally declares.
func stringConsts(t *testing.T, internal, pkg string) map[string]string {
	t.Helper()
	out := map[string]string{}
	fset := token.NewFileSet()
	parsed, err := parser.ParseDir(fset, filepath.Join(internal, pkg),
		func(fi fs.FileInfo) bool { return !strings.HasSuffix(fi.Name(), "_test.go") }, 0)
	if err != nil {
		t.Fatalf("parse %s: %v", pkg, err)
	}
	for _, p := range parsed {
		for _, f := range p.Files {
			for _, decl := range f.Decls {
				gen, ok := decl.(*ast.GenDecl)
				if !ok || gen.Tok != token.CONST {
					continue
				}
				for _, spec := range gen.Specs {
					vs, ok := spec.(*ast.ValueSpec)
					if !ok {
						continue
					}
					for i, name := range vs.Names {
						if i >= len(vs.Values) {
							continue
						}
						if v, ok := stringLit(vs.Values[i]); ok {
							out[name.Name] = v
						}
					}
				}
			}
		}
	}
	return out
}

// engineRegistrantsByID scans every non-test internal package for abi.RegisterEngine(id, …)
// calls and returns id -> set-of-packages-that-register-it, resolving the id via
// resolveEngineIDArg (literal OR same-package string const). A call whose id the gate cannot
// read statically is recorded under the sentinel id "" so the caller can fail closed on it.
func engineRegistrantsByID(t *testing.T, internal string) map[string]map[string]bool {
	t.Helper()
	byID := map[string]map[string]bool{}
	record := func(id, pkg string) {
		if byID[id] == nil {
			byID[id] = map[string]bool{}
		}
		byID[id][pkg] = true
	}
	for _, pkg := range goPackageDirs(t, internal) {
		consts := stringConsts(t, internal, pkg)
		fset := token.NewFileSet()
		parsed, err := parser.ParseDir(fset, filepath.Join(internal, pkg),
			func(fi fs.FileInfo) bool { return !strings.HasSuffix(fi.Name(), "_test.go") }, 0)
		if err != nil {
			t.Fatalf("parse %s: %v", pkg, err)
		}
		for _, p := range parsed {
			for _, f := range p.Files {
				ast.Inspect(f, func(n ast.Node) bool {
					call, ok := n.(*ast.CallExpr)
					if !ok || len(call.Args) < 1 {
						return true
					}
					sel, ok := call.Fun.(*ast.SelectorExpr)
					if !ok {
						return true
					}
					recv, ok := sel.X.(*ast.Ident)
					if !ok || recv.Name != "abi" || sel.Sel.Name != "RegisterEngine" {
						return true
					}
					id, ok := resolveEngineIDArg(call.Args[0], consts)
					if !ok {
						record("", pkg) // unreadable id — fail closed
						return true
					}
					record(id, pkg)
					return true
				})
			}
		}
	}
	return byID
}

// TestSingleEngineDriverPerID is the keyed-registry sibling of TestSinglePageOutBlobRegistrant
// and TestSingleWitnessResolverRegistrant, extended to the inference-engine registry. For each
// engine id the non-test tree registers, the set of packages that register THAT id MUST be
// exactly the declared holder in engineDriverRole. A NEW registrant of an existing id is a
// second engine silently swapping the kernel's selected backend by blank-import order (the
// last-write-wins map corruption); a DECLARED registrant that stops registering its id is an
// engine gone missing — the kernel constructs New("inkernel") and finds no driver under the id.
//
// WHY A GATE (and why it BITES): the compiler cannot enforce this. RegisterEngine takes any
// (string, EngineDriver) and does `reg.engines[id] = d` unconditionally — no clash panic — so a
// second call with an existing id in some other leaf's init() compiles cleanly and just
// overwrites the map entry; the regression is invisible until the wrong engine wins at runtime
// on a particular import order. There is no interface or type that forces per-id singularity —
// the same construction-regression class as the three sibling registrant gates, not the vacuous
// already-typed class of the rejected reason-closure gate.
//
// It is STRONGER than a literal-only id scan (pkgsCallingSelectorWithStringArg): modelengine
// registers via the const EngineID ("inkernel"), so a literal-only gate would be BLIND to the
// dogfood engine's registrant and could not enforce its singularity. resolveEngineIDArg resolves
// the same-package const, so all three v0.1 engines are covered. An id the gate cannot read
// statically (neither a literal nor a known const — e.g. a runtime-built id) is recorded under
// the sentinel "" and fails closed: a dynamically-keyed engine registrant defeats the very
// singularity proof this gate exists to make.
//
// This is NOT a blunt "one engine forever" cap: a NEW engine under a NEW id (a remote driver, a
// second model backend) is correctly allowed — add it to engineDriverRole with its role, the
// same conscious review chokepoint as adding a tier or a chatEndpointRole holder. A deliberate
// swap of an existing id's owner updates that id's entry. Only an UNDECLARED second registrant
// of an existing id, or a declared one going missing, turns the gate RED.
//
// Seeded GREEN (2026-06-20): an AST scan of non-test callers finds exactly {inkernel→modelengine,
// localtools→agent, mock→engine}; every other RegisterEngine call in the tree is a _test.go
// fixture (ids "test"/"e"/"local"/"remote"/"") and is correctly not counted. Adding a second
// non-test registrant of any of those three ids — or one of them dropping the call — turns the
// gate RED.
func TestSingleEngineDriverPerID(t *testing.T) {
	internal := internalDir(t)
	byID := engineRegistrantsByID(t, internal)

	// Fail closed on any registration whose id the gate could not read statically.
	if unreadable := byID[""]; len(unreadable) > 0 {
		pkgs := make([]string, 0, len(unreadable))
		for pkg := range unreadable {
			pkgs = append(pkgs, pkg)
		}
		sort.Strings(pkgs)
		t.Errorf("package(s) %v call abi.RegisterEngine with a first argument the gate cannot resolve "+
			"to a static id (not a string literal nor a same-package string const). A dynamically-built "+
			"engine id defeats the per-id singularity proof: the gate cannot tell whether two leaves "+
			"register the same engine and silently swap the kernel's backend by import order. Register "+
			"the engine under a string-literal or const id so the invariant stays checkable.", pkgs)
	}

	// Every registered id must match its declared holder set exactly.
	ids := make([]string, 0, len(byID))
	for id := range byID {
		if id == "" {
			continue // handled above
		}
		ids = append(ids, id)
	}
	sort.Strings(ids)
	for _, id := range ids {
		got := byID[id]
		want := engineDriverRole[id]
		if want == nil {
			pkgs := make([]string, 0, len(got))
			for pkg := range got {
				pkgs = append(pkgs, pkg)
			}
			sort.Strings(pkgs)
			t.Errorf("engine id %q is registered by %v in non-test code but is not a declared id in "+
				"engineDriverRole. abi.RegisterEngine is a last-write-wins map (registry.go: "+
				"`reg.engines[id] = d`), so if two leaves register %q the kernel's selected engine flips "+
				"by blank-import order. If this is a deliberate NEW engine, add %q to engineDriverRole with "+
				"its single registrant and role; if it is a second registrant of an existing engine, re-home "+
				"it.", id, pkgs, id, id)
			continue
		}
		for pkg := range got {
			if _, ok := want[pkg]; !ok {
				t.Errorf("engine id %q is registered by internal/%s, which is not its declared holder. The "+
					"engine registry is last-write-wins keyed by id, so a second registrant of %q silently "+
					"swaps the kernel's selected backend by import order. Re-home the registration, or, if "+
					"this is a deliberate ownership change, update engineDriverRole[%q].", id, pkg, id, id)
			}
		}
		for pkg, role := range want {
			if !got[pkg] {
				t.Errorf("internal/%s is declared in engineDriverRole as the holder of engine id %q (%q) but "+
					"no longer registers it in non-test code. If the engine was deliberately moved or removed, "+
					"update engineDriverRole; otherwise the kernel constructs New(%q) and finds no driver under "+
					"that id.", pkg, id, role, id)
			}
		}
	}

	// A declared id that nothing registers is a stale table entry (the engine went missing
	// entirely, not just changed hands) — surface it so the table keeps naming real engines.
	for id, holders := range engineDriverRole {
		if _, present := byID[id]; !present {
			roles := make([]string, 0, len(holders))
			for pkg := range holders {
				roles = append(roles, pkg)
			}
			sort.Strings(roles)
			t.Errorf("engineDriverRole declares engine id %q (holder(s) %v) but NO non-test package "+
				"registers it — the engine went missing. Remove the stale entry, or restore the "+
				"registration if it was dropped by mistake.", id, roles)
		}
	}
}

// TestRootImportsNothingInternal gates the tier-0 root-purity invariant that the tier table
// (line "0 root … imports nothing internal"), doc.go, GROWTH.md ("the one tree everyone
// imports"), and PARTITION.md ("internal/abi … wave-0, human-owned, and unleasable") all
// assert in prose but no test enforces: internal/abi imports ZERO other internal packages.
//
// WHY THIS IS NOT ALREADY COVERED BY TestNoUpwardImports. The layering gate enforces the
// RELATIVE rule tier(importer) >= tier(imported). For abi (tier 0) that catches an import of
// any tier-1+ package (to>0 is an upward edge) — but it is BLIND to abi importing another
// tier-0 package, because to(0) > from(0) is false, so the edge passes. The prose invariant is
// ABSOLUTE ("imports nothing internal", "the one tree everyone imports"), strictly stronger
// than "may import packages at its own tier or below." abi is the single shared, human-owned,
// unleasable tree at the bottom of the DAG; its zero-internal-imports property is what lets
// every other leaf import abi without creating a cycle and what keeps it wave-0 separable. A
// `<=`-tier proxy cannot express "exactly zero."
//
// WHY IT BITES (and is not vacuous). Seeded GREEN today: an AST scan of abi's non-test source
// finds no internal import. It turns RED the moment abi grows an import of ANY internal
// package — including a second package promoted to tier 0, the one case TestNoUpwardImports
// structurally cannot catch. An internal import inside abi is a real regression: it makes the
// frozen root depend on a higher-churn leaf, voids "everyone imports abi, abi imports no one,"
// and can reintroduce an import cycle the moment that leaf imports abi back (which, being the
// ABI, it almost certainly does). This is the construction-regression class the sibling gates
// catch, not the vacuous already-typed class — the compiler permits abi to import any internal
// package; only this gate forbids it.
//
// SCOPE: non-test source only (abi's own _test.go may import internal fixtures freely — that is
// off the request path and does not couple the shipped root). A future deliberate decision to
// let abi depend on something internal would be a direction change recorded by editing this
// gate, the same conscious chokepoint as the tier table.
func TestRootImportsNothingInternal(t *testing.T) {
	internal := internalDir(t)
	var internalImports []string
	for _, imp := range imports(t, internal, "abi") {
		if strings.HasPrefix(imp, modPrefix) {
			internalImports = append(internalImports, imp)
		}
	}
	if len(internalImports) > 0 {
		sort.Strings(internalImports)
		t.Fatalf("internal/abi (tier 0, the frozen root) imports %d internal package(s):\n  %s\n"+
			"The root must import NOTHING internal — it is the one tree everyone imports, so any "+
			"internal dependency of abi inverts the DAG (a higher-churn leaf now sits UNDER the ABI) "+
			"and risks an import cycle the moment that leaf imports abi back. TestNoUpwardImports does "+
			"NOT catch this when the dependency is another tier-0 package (to(0) > from(0) is false). "+
			"Push the shared type the other way (the leaf imports abi, never the reverse), or, if this "+
			"is a deliberate direction change, edit this gate consciously — do not loosen it silently.",
			len(internalImports), strings.Join(internalImports, "\n  "))
	}
}

// TestKernelImportsOnlyAbi gates the kernel's "driver-blind integrator" contract: package
// kernel "never imports a leaf package; it only WALKS the abi registries" (kernel.go's own
// package doc). The kernel is the one concrete abi.Kernel; it reaches every subsystem through
// the frozen abi registries (Adjudicators, FastPaths, ResultAdmitters, engines, emitters),
// which is exactly what lets the leaves be built in disjoint trees and linked by a single
// blank-import line in internal/registrations. A direct `import ".../internal/<leaf>"` in the
// kernel voids that: it hard-wires the kernel to one leaf at compile time, re-couples two
// trees that fleet leases assume are disjoint, and opens an import cycle the moment that leaf
// imports the kernel back.
//
// TestNoUpwardImports does NOT catch this: the kernel and every mechanism leaf (vdso, grammar,
// preflight, ctxmmu, adjudicator, …) are all tier 2, so a sideways kernel->vdso edge satisfies
// to(2) <= from(2). This is the tier-2 analogue of TestRootImportsNothingInternal's tier-0
// floor — a named-package import-purity gate the relative DAG rule structurally cannot express.
//
// Seeded GREEN (2026-06-20): internal/kernel's only non-test internal import is abi (kernel.go).
func TestKernelImportsOnlyAbi(t *testing.T) {
	internal := internalDir(t)
	var leafImports []string
	for _, imp := range imports(t, internal, "kernel") {
		if !strings.HasPrefix(imp, modPrefix) {
			continue // stdlib / external — out of scope
		}
		if imp == modPrefix+"abi" {
			continue // the one permitted internal import: the frozen registry seam
		}
		leafImports = append(leafImports, imp)
	}
	if len(leafImports) > 0 {
		sort.Strings(leafImports)
		t.Fatalf("internal/kernel (the driver-blind integrator) imports %d leaf package(s) other than abi:\n  %s\n"+
			"The kernel must reach every subsystem through the abi registries (Adjudicators, FastPaths, "+
			"ResultAdmitters, engines, emitters), NEVER by importing a leaf directly — that is what lets the "+
			"leaves live in disjoint trees linked by one blank-import line in internal/registrations. A direct "+
			"leaf import hard-wires the kernel to that leaf at compile time, re-couples two trees fleet leases "+
			"assume disjoint, and risks an import cycle the moment the leaf imports kernel back. TestNoUpwardImports "+
			"does NOT catch this: kernel and the mechanism leaves are all tier 2, so a sideways edge satisfies the "+
			"DAG rule. Move the shared type into abi and walk it from the registry, or, if this is a deliberate "+
			"direction change, edit this gate consciously — do not loosen it silently.",
			len(leafImports), strings.Join(leafImports, "\n  "))
	}
}
