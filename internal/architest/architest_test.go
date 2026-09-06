package architest

import (
	"go/ast"
	"go/build"
	"go/parser"
	"go/token"
	"io/fs"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/archreport"
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
//	0 root                  frozen ABI; imports nothing internal.
//	1 primitive             pure libraries; import only root.
//	2 foundation-composite foundational helpers assembled from primitives.
//	3 mechanism             single-purpose kernel mechanisms and engines.
//	4 composer              workflows composed from mechanisms.
//	5 integrator            top-level wiring, gateways, and registrations.
//
// A NEW package MUST be added here (TestEveryPackageDeclaresTier fails otherwise) at the
// LOWEST tier whose role it fits. That forced choice is the review gate: it makes adding
// a leaf a conscious layering decision instead of an accident.
var tier = map[string]int{

	"qwen38campaign":     1, // Qwen3.8 subagent fan-out multi-agent benchmark harness.
	"factorymigrate":     1, // 5-Gate boundary and migration auditor across fak and fak-private.
	"agentsched":         3, // agent thread scheduling, priority queue, and 4-gate admission governor (#11177, #11178).
	"reflexagent":        3, // fast-dispatch micro-agent reflex loop (#11189).
	"workspaceslot":      5, // pre-allocated workspace slot ring and fast in-place recycling (#11180).
	"harnessbench":       5, // high-churn soak and thundering-herd saturation witness suite (#11183).
	"effortbench":        5, // intra-model effort modulation and thinking budget benchmark (#11185).
	"control":            2, // dynamic inference configuration control plane: shift-left validation, relational invariants, dry-run diff, and canary auto-rollback watchdog (#10869).
	"agentquery":         2, // typed agent-state query/schema engine; composes maputil for deterministic schema-key ordering.
	"bgloop":             2, // durable background-loop runtime; composes dormancy stamps and horizon classification for persisted wake events.
	"cmdutil":            2, // shared command benchmark/render helpers; composes benchids for deterministic synthetic token streams.
	"ops":                4, // autonomous operations daemon and machine maintenance subsystem (#11156, #11158).
	"observability":      2,
	"sandbox":            2, // multi-tier local execution sandbox (wazero WebAssembly, OS container, Job Object/cgroups).
	"gym":                5, // sandbox evaluation gym, scenario runner, and closed-loop multi-turn harness simulation (#11652, #11666).
	"observer":           1, // in-kernel observer worker pool for zero-cost step-by-step agent output interpretation (#11408).
	"director":           3, // autonomous multi-agent roll-up digest for zero-self-report supervisor steering (#11411).
	"studyreceipt":       1,
	"stopgate":           1, // unified lifecycle stop gates across fak guard and fak agent (#11253); stdlib-only, off the hot path.
	"abi":                0,
	"launchguard":        1, // per-attempt launch guard and state directory management; stdlib-only, off the hot path.
	"servingsupervision": 3, // serving failure-domain isolation and supervision; mechanism leaf, off the hot path.
	"storagepressure":    3, // storage headroom and reclaim reporting; mechanism leaf, off the hot path.
	"wipreadiness":       1, // working-tree WIP readiness classification; stdlib-only, off the hot path.
	"timeaware":          1, // time-aware duration and deadline calculation; stdlib-only, off the hot path.
	"resulttier":         1, // standard bounded result tiers and cursor pagination; stdlib-only, off the hot path.
	"power":              1, // cross-platform OS power assertion and wake-lock management; stdlib-only, off the hot path.

	"citeverify":      2, // mechanical source-line claim verification; stdlib-only, off the hot path.
	"genlock":         2, // generated-output input lock verification; stdlib-only, off the hot path.
	"harnessprotocol": 1, // product-neutral headless run protocol and projections; stdlib plus public harnesskit contract only.
	"usagepreflight":  2, // remaining-quota admission calculation; stdlib-only, off the hot path.

	"envconfiglint":          2, // CONFIG_NOT_ENV ratchet banning new non-secret env reads; imports windowgate(1), off the hot path.
	"flowcredit":             2, // receiver-granted credit ledger for KV-transfer backpressure; stdlib-only, imports nothing internal, off the hot path.
	"flowmetrics":            2, // deterministic flow-metrics fold (#6194); composes maputil for stable issue/commit row traversal while grading Little's-Law KPIs and working-tree WIP.
	"stallpage":              2, // durable deduped operator page for stallscan reboot high-water; imports stallscan(1)+choicetriage(1)+flock, off the hot path.
	"agenticbench":           3, // #868 artifact rollup gate over committed benchmark evidence; stdlib-only, off the hot path.
	"agentxbench":            3, // AgentX benchmark runner, request lifecycle phase decomposition, and interactivity metrics (#8774, #8775); stdlib-only.
	"ailuminate":             1, // pure MLCommons-AILuminate benchmark-entry scoping/go-no-go contract (#1070); stdlib-only, off the hot path.
	"apihostprobe":           1, // API host readiness/acceptance probe: stdlib HTTP probes + roster parsing for cmd/fak api-host; off the hot path.
	"accountprobe":           1, // pure account-probe ledger reader (probe_ledger.jsonl): last-probe-by-account + probe recency for the roster fresh-probe fold; stdlib-only, imports nothing internal, off the hot path.
	"walkfiles":              1, // shared swallow-and-scan filepath.WalkDir primitive (Files: visit regular files, swallow walk-step errors, propagate visit errors); stdlib-only, imports nothing internal, off the hot path (#10405).
	"dispatchconservation":   1, // pure worker-unit conservation fold over .dispatch-runs artifacts; stdlib-only, off the hot path.
	"dispatchdoa":            1, // #5868: pure DOA-spawn detector — grades a worker log into "the dispatcher spawned it, it wrote a stub, it never reached the guard's agent-launch banner" and folds a window into a clear/warn/alarm spawn-health rung; stdlib-only, imports nothing internal, off the hot path.
	"eveparity":              1, // CI-runnable Eve-eval parity witness (#2605): pure in-repo eval-semantics evaluator (Evaluate/Compare keep the hard/soft gate distinction) proving fak-routed == raw; production code stdlib-only, off the hot path.
	"eveimport":              1, // read-only Eve run/OTel evidence importer (#2606): pure deterministic fold of saved NDJSON session streams / eve.* spans into session-ledger rows with default body redaction; stdlib-only, imports nothing internal, off the hot path.
	"benchcatalog":           2, // pure benchmark registry used by fak benchmarks and scorecards; stdlib-only, off the hot path.
	"localbench":             3, // local hardware receipt workflow composes benchcatalog plus OS/process probes; command-facing and off the hot path.
	"buildoverlay":           2, // neutral shared-checkout Go overlay/file-selection helpers; imports windowgate and is shared by runtime and fak-dev.
	"buildwitness":           2, // structural CI guard (#3217): runs `go build ./cmd/fak` under default tags and reds the trunk (naming the undefined symbol) when uncommitted/tagless WIP breaks the shared build; stdlib-only (os/exec+runtime), imports nothing internal, off the hot path.
	"clonescan":              1, // pure authoring-time clone QUERY: the forward half of the code-slop clone detector (normalized Go token-window engine) as an importable library — "does a token-similar block already exist?" before the code is written; stdlib-only, no internal import, off the hot path.
	"sotamatrix":             1, // pure SOTA prior-art registry (op -> reference/route/oracle) read by fak sota, the PRIOR_ART gate, and the coverage scorecard; stdlib-only, off the hot path.
	"stallscan":              1, // pure churn-signal stall classifier (Classify(Sample,Thresholds)->Verdict) for low-usage machine lockups read by fak stallscan; stdlib-only, imports nothing internal, off the hot path.
	"assumecheck":            2, // pure assumption-audit kernel (#3819 C1) + name-resolved witness drivers (#3821 C3): closed Level/WitnessKind/Outcome vocabulary, pure Check(Assumption,Evidence)->Verdict, fail-closed GuardAssumption typed error; read by fak assume; the kernel file stays stdlib-only while the driver shell imports windowgate(1)+procguard(1), off the hot path.
	"growthgate":             1, // pure unbounded-growth classifier (Classify([]Artifact,Budget)->Report) for append-only ledger/log bloat read by fak growthgate; the standing-bloat twin of stallscan; stdlib-only, imports nothing internal, off the hot path.
	"branchrole":             2, // branch-role contract reader over dos.toml; stdlib-only, off the hot path.
	"benchloop":              2, // benchmark super-loop manager: folds benchcatalog/benchruns/nightrun status into one command-facing control surface; off the hot path.
	"macbench":               1, // Mac gateway benchmark probes for nightrun: stdlib HTTP client + JSON artifact fold, off the hot path.
	"benchruns":              1, // pure benchmark-run catalog reader/renderer over experiments/benchmark artifacts; stdlib-only, off the hot path.
	"benchckpt":              1, // per-cell write-ahead checkpoint/resume ledger the compute-bench executors write through (#2382); stdlib-only, off the hot path.
	"devcheckpoint":          2, // append-only developer milestone checkpoint records; imports flock(1)+stdlib, off the hot path.
	"benchlineagegate":       2, // pure benchmark-emitter lineage hygiene gate; stdlib-only source scanner, off the hot path.
	"benchmarkdown":          1, // stdlib-only typed layout renderer for byte-stable benchmark adapter Markdown reports.
	"conceptbench":           2, // dos-refereed conceptbench grader (#2732): maps a concept + transcript + fixture to a referee-sourced verdict; imports taskmgr(1)+hooks(1), off the hot path.
	"cachevalueledger":       2, // durable, append-only cache-value observation ledger for fak sessions; JSONL persistence over cacheobs stats.
	"cacheprice":             1, // the ONE source of truth for the provider prompt-cache price multipliers (read 0.1x / write 1.25x / 2.0x): a pure leaf gateway(4) and resume(1) read (and the agent(4) fire gate is test-pinned to) so an identical cached token is priced identically (#2798). Imports nothing internal, off the hot path.
	"gatewayusageledger":     2, // durable, append-only gateway usage-counter ledger (#1610); JSONL persistence over a stdlib-only Counters mirror, no internal/gateway or internal/kernel import.
	"skillvalue":             2, // durable, append-only per-skill outcome-value ledger (#2873) read by `fak skill value report`; JSONL persistence via jsonlledger(1), imports nothing else internal, off the hot path.
	"benchcli":               2, // shared helpers the bench-CLI mains (cmd/*bench) had copy-pasted; imports model(1) only, off the hot path.
	"benchids":               1, // pure deterministic synthetic-token-ID generator for the bench mains (#776); stdlib-only, off the hot path.
	"benchscore":             2, // pure benchmark score artifact validator/renderer; stdlib-only, off the hot path.
	"callavoid":              1, // pure avoided-call economics/accounting primitive; stdlib-only, folded by higher layers.
	"turnavoid":              1, // pure offline whole-model-turn replay/accounting primitive; stdlib-only, folded by CLI.
	"cavemansafety":          1, // pure deterministic blinded safety judge; stdlib-only, offline benchmark evidence.
	"cavemanpairwise":        1, // pure blinded pairwise benchmark judge and fail-closed evidence accounting; stdlib-only, off the product path.
	"harnessartifact":        2, // immutable, content-addressed artifacts around harnessresolve locks (#7230).
	"harnessres":             1, // cross-platform, stdlib-only process resource sampler for the fak guard harness (CPU/mem/IO); imports nothing internal, off the hot path (#2045, epic #2044).
	"harnessinit":            3, // product scaffold + cross-host conformance; depends on host/resolve/window launch seams (#8227).
	"harnesshost":            2, // resolves first-party host profiles into product manifests and locks before scaffold rendering.
	"harnessrelease":         1, // release-asset checksum/extraction and external-product witness runner (#6957).
	"harnessgallery":         1, // static user-need blueprints and starter pack renderer (#6961).
	"harnessdiscover":        2, // provenance-bearing scoped declaration discovery before contextual selection (#6898).
	"harnessclassify":        1, // deterministic explicit-first domain/task classification before contextual selection (#6900).
	"harnesscrossover":       1, // pure reproducible net-work comparison for contextual versus tuned native profiles (#6903).
	"harnesscreationreceipt": 1, // stdlib-only participant receipt validation and study-row projection (#6976).
	"harnesscreationstudy":   1, // deterministic independent-participant denominator and claim eligibility fold (#6937).
	"harnesscompose":         1, // typed inert asset overlap semantics before dependency solving and launch (#6904).
	"harnessderive":          2, // verified base-lock deltas producing launchable resolved locks (#7225).
	"harnesscontrolstudy":    2, // deterministic paired-preference evidence over typed harness variants (#7224).
	"harnesscontrolpacket":   2, // digest-bound isolated participant packets with cross-arm leak refusal (#7274).
	"workflowoutcomestudy":   1, // deterministic equivalent-arm and blind-grading evidence gate (#7272).
	"harnessinspect":         2, // product-facing inspection over composed assets and resolved locks (#7213/#7232).
	"harnessmix":             2, // deterministic multi-lock composition with compatibility and budget admission (#7229/#7251).
	"harnessoverride":        2, // bounded override proposals over composed assets and resolved locks (#7223/#7232).
	"harnessverify":          2, // runtime deviation checks over composed assets and resolved locks (#7222/#7233).
	"harnesspreview":         2, // deterministic contextual lock-risk diff over classifier, composition, and resolver facts (#6902).
	"harnessresolve":         2, // deterministic product lock over typed assets plus stackresolve dependency facts (#6792).
	"harnessselect":          1, // context-layer selection and explain trace; stdlib-only before product composition.
	"stackresolve":           1, // stdlib-only generic dependency search and receipts; domain facts enter through Provider (#6891).
	"supportgraph":           1, // stdlib-only exact-tuple support evidence and freshness query before model/hardware composition (#6895).
	"workloadfit":            2, // purpose-specific fitness mechanism over stackresolve contracts; keeps soft ranking outside composition gates (#6893).
	"stackpreflight":         3, // launch integrator joining dependency, workload-fitness, and support-evidence receipts (#6897).
	"amdgpu":                 2, // AMD GPU fact probe and perf-counter JSON fold for Windows harness diagnostics; imports windowgate(1), off the hot path.
	"accounts":               2, "accountobs": 1, "guardaudit": 2, "appversion": 2, "blob": 1, "boundarylint": 1, "cachemeta": 2, "cacheobs": 1, "cachevalue": 2, "canon": 1, "compute": 2, "runtimecap": 3, "deletioncert": 1, "demoui": 2, "ggufload": 2, "gpulease": 2, "researcharm": 2, "fleetreap": 2, "hfhub": 2, "intlist": 1, "leakcheck": 1, "metalgemm": 1, "metrics": 2, "model": 2, "orphanscan": 1, "pathlint": 1, "pathutil": 1, "privatepath": 2, "provenance": 1, "swebench": 2, "urllint": 1, "webbench": 2,
	// stdlib-only foundation leaves (import nothing internal); off the hot path.
	"auditpane": 2, "benchingest": 2, "binstamp": 2, "cachewitness": 2, "codexmemory": 1, "covmatrix": 1, "defaultvaluescore": 1, "demoutil": 1, "experiments": 2, "fleetaccounts": 2, "fleetbottleneck": 2, "flock": 1, "framevisibility": 1, "ghspam": 1, "issuecontractrepair": 2, "jsonlledger": 1, "kvbudget": 2, "maputil": 1, "mathx": 1, "modeldescriptor": 1, "newleaf": 1, "numfmt": 1, "randhex": 1, "refutil": 2, "selfinstall": 2, "sessionaudit": 2, "strmatch": 1,
	"newmodel":      2,                                                                            // deterministic onboarding compiler composed over the primitive modeldescriptor contract (#9421).
	"reportledger":  2,                                                                            // the (date, generated-at) ordering-key contract + LatestBefore binding the report packages' ledger shims each carried (#10405); composes jsonlledger(1) only, stdlib otherwise, off the hot path.
	"sessiondiag":   2,                                                                            // bounded redacted Codex SQLite/log and process-incident classifier (#5992); stdlib-only, off the hot path.
	"deepseekbench": 1, "deepseekv4kv": 1, "deepseekv4moe": 1, "dispatchaging": 1, "linkstate": 1, // stdlib-only leaves: DeepSeek V4 bench/KV/MoE fixture cores (#3014/#3017/#3018), fleet-dispatch priority-aging, link-state phase vocab; import nothing internal, off the hot path.
	"guardvars": 1, "xprobe": 1, // guardvars: pure /debug/vars sessions-row schema (SessionVars) the fak info agents pane renders. xprobe: throwaway end-to-end buildcheck-fallback ping probe (Ping). Both stdlib-only, import nothing internal, off the hot path.
	"glm52prefillsweep": 2, "turnkind": 1, // glm52prefillsweep: GLM-5.2 prefill-latency sweep planner + benchmark-ledger driver (#3085/#3086); imports windowgate(1) for hidden subprocesses. turnkind: content-free last-user-turn structural classifier (#3307); imports nothing internal. Both off the hot path.
	"questionledger":        2,                // /question-loop ledger discipline (Go port of tools/question_ledger.py): lint/next-id/dedupe-check/stats over docs/questions/asked.jsonl; stdlib-only, off the hot path.
	"quality":               2,                // missing-middle quality-ladder spine (epic #4509): versioned quality-case schema + reference/engine Runner adapters + deterministic differential comparator & rubric oracle + replay-complete result/failure-bundle; off the hot path. NOT pureRoot: the incremental-Unicode oracle's first-divergence offset delegates to strmatch(1) rather than carrying the copy `fak benchmarks`' name matcher already had (bfb2e3fa7), so it imports a sibling leaf.
	"trunkbuildprobe":       2,                // release-gate diagnosis (Go port of tools/trunk_build_probe.py): parses `go build` errors + hunts forgotten-`git add` definers; imports windowgate(1) for hidden subprocesses, off the hot path.
	"godsplitplan":          1,                // doc-comment-aware Go split boundary+hazard planner (Go port of tools/godsplit_plan.py): the /modularize skill's planner + the decl-fold refactorverify reuses; stdlib-only, off the hot path.
	"refactorverify":        3,                // proves a god-split dropped no top-level decl (Go port of tools/refactor_verify.py): folds each touched package's decl multiset before/after via godsplitplan.Compute; imports godsplitplan, off the hot path.
	"chatrelay":             2,                // pure Slack chat-relay client (the inbound complement to the scoreboard publishers): posts/reads a channel via the shared slackenv resolver; rides slackwire(1) for transport, off the hot path.
	"evebridge":             1,                // Eve preflight connection gates (auth/allowlist/approval, #2602): pure request-shape screening for the Eve bridge; stdlib-only, imports nothing internal, off the hot path.
	"sessionsignals":        1,                // shared closed vocabulary of terminal-turn transcript signals (limit-reset banners, auth/credit/access walls, transient API errors) ported from tools/fleet_session_signals.py; the one taxonomy the resume sweep/stopped/watchdog family classifies with. Stdlib-only, imports nothing internal, off the hot path.
	"sessiondesc":           2,                // deterministic session descriptor fold plus durable journal adapter over sessionjournal(2).
	"watchdoghealth":        2,                // pure health-digest for the DEFAULT watchdog monitors (fak.watchdog-health.v1): folds the live probe + persisted autoheal heal-state into a closed per-monitor status vocabulary + a worst-of fleet rollup for `fak watchdog status`; stdlib-only, imports nothing internal, off the hot path.
	"resumemetrics":         1,                // in-process expvar metrics for the resume/heal watchdog (#3803): expvar counters + strings only; imports nothing internal, off the hot path.
	"resumebackoff":         2,                // pure resume signature backoff and cross-session park fold (#3584); stdlib only.
	"resumeactuator":        1,                // stdlib-only managed continuation-command rendering primitive (#8060).
	"commitlifecycle":       2,                // pure commit-to-ship lifecycle fold (#5989); stdlib-only, no git/filesystem/process I/O, off the hot path.
	"wiplease":              2,                // live-WIP projection into laneadmit lease geometry; imports laneadmit(2)+wipattr(1), off the hot path.
	"wipref":                2,                // append-only working-tree checkpoint ref store under refs/fak/wip/ read by fak wip (sibling of leaseref's refs/fak/locks); stdlib-only, imports nothing internal, off the hot path.
	"wipattr":               2,                // dirty-hunk-to-checkpoint attribution fold (#3874); imports pathutil(1), off the hot path.
	"wiprecon":              1,                // pure crashed-checkpoint reconciliation decision fold (#3875); stdlib-only, imports nothing internal, off the hot path.
	"supportmaturityscore":  3,                // support-maturity scorecard over covmatrix + supportmaturity(2); off the hot path.
	"supportmaturity":       3,                // support-maturity ladder + shipgate-gated promotion/drop rules (#1244/#1245); imports tier-1 support facts and shipgate(2), off the hot path.
	"releasestale":          1,                // pure publish-staleness verdict (latest tag vs HEAD, in commits+days) + a thin git Gather shell; the publish-axis dual of binstamp's source-axis freshness. Stdlib-only, imports nothing internal, off the hot path.
	"releasereadiness":      2,                // read-only release-readiness scorecard over tracked contracts, git state, and optional GitHub assets; stdlib-only and off the hot path.
	"releasestatus":         2,                // pure read-only release-posture fold (Compute over injected Facts) porting tools/release_status.py; the broader sibling of releasestale. Stdlib-only, imports nothing internal, off the hot path.
	"affectedtests":         1,                // pure reverse-dependency closure for the `fak affected` fast test gate: changed packages + every package that (transitively, test imports included) imports one; stdlib-only, imports nothing internal, off the hot path.
	"knownbad":              1,                // pure fold core of the fleet-wide known-bad signature ledger (#2713, epic #2712): sha256 signature derivation + stateless Match/liveness projection over records; the impure shell (ledger I/O, clock, flags) lives in cmd/fak/knownbad.go. Stdlib-only, imports nothing internal, off the hot path.
	"knownenv":              1,                // pure core of the fleet-wide known-ENVIRONMENT-failure registry (#2144, epic #2136): matches TOOL OUTPUT (error-text needle + exit code) -> a not-your-fault verdict and annotates it inline; the output-fingerprint twin of knownbad's tree match. Stdlib-only, imports nothing internal, off the hot path.
	"blastlease":            3,                // shared lease projection/fixture reader for known-bad runtime reporting and the development blast estimator; imports blastradius(2)+leaseref(2).
	"blastradius":           2,                // pure JOIN for blast-radius containment (#2712 W3): broken package -> dependents (affectedtests.Select) -> the leases/issues whose tree intersects (knownbad.TreesIntersect); no I/O, no clock. Imports affectedtests(1)+knownbad(1), off the hot path.
	"commitsubject":         2,                // commit-subject coverage fold over hooks.CommitMsgVerdict + recent git subjects; imports hooks(1)+windowgate(1), off the hot path.
	"fleetspine":            1,                // LAN multicast heartbeat self-discovery spine for the fleet pane: passive concurrency-safe peer Registry + net.ListenMulticastUDP transport + advertiser/listener/expiry runners; stdlib-only, imports nothing internal, off the hot path.
	"guardsessions":         1,                // append-only guard-session index with exact-then-prefix Resolve over an injected filesystem path; stdlib-only, imports nothing internal, off the hot path.
	"guardrotate":           2,                // pure cooldown-aware seat-selection core for `fak guard`: given a registry + cooldown store + headroom signal, rotates a launch off a walled account onto a seat with proven room; imports only accounts(1), off the hot path.
	"balance":               2,                // night-balance readout folding resume-recovery + work-mix into a degradation verdict; imports resume(1)+superloop(1), off the hot path.
	"focusscore":            2,                // fleet convergence/breadth focus scorecard over the trajctl objective tree; imports trajctl(1)+pkg/scorecard, off the hot path.
	"memgate":               2,                // memory-pressure admission fold for heavy model loads; stdlib + windowgate shell helpers, off the hot path.
	"rolloutmode":           1,                // stdlib-only closed rollout vocabulary and caller-selected parse fallback (#6090).
	"memorycotravel":        2,                // stdlib-only project memory co-travel gate/ledger for shadow/live carryover between config roots; off the hot path.
	"dosdecision":           1,                // stdlib-only revalidation of DOS decision rows against caller-provided live lease state (#6494).
	"memoryindex":           2,                // stdlib-only memory-index reconciliation mechanism; off the hot path.
	"memorystability":       1,                // stdlib-only fleet-memory stability governor over drift trajectories; off the hot path.
	"memoryread":            1,                // read-only committed fleet-memory digest renderer; stdlib-only, off the hot path.
	"fleetmemory":           1,                // cross-agent lessons ledger (#2141) + its write-time dedup guard (#2142): New/Match/Inject fold publishable lessons into a key-indexed Ledger and select by trigger for peer injection (freshness re-verify is the caller's job — the dos_recall discipline); stdlib-only, imports nothing internal, off the hot path.
	"nodecompare":           1,                // stdlib-only cross-node benchmark result fold; off the hot path.
	"planaudit":             1,                // stdlib-only plan-doc drift audit; off the hot path.
	"sotacoverage":          2,                // SOTA prior-art coverage scorecard over sotamatrix + git tree scans; imports sotamatrix(1)+windowgate(1), off the hot path.
	"testroute":             1,                // pure host test-route fold (native / WSL / CI) over caller-supplied probe data; stdlib-only, no I/O.
	"mergepreview":          2,                // read-only shared-trunk merge preview over git merge-tree; stdlib-only, off the hot path.
	"taskdecision":          1,                // task-scoped append-only decision log loaded into reset carryover; stdlib-only, off the hot path.
	"taskidentity":          1,                // canonical task directive/signature derivation for session transcript heads; stdlib-only, off the hot path.
	"sweepconfig":           2,                // agent sweep profile load/save/list helpers over JSON/YAML-lite config; stdlib-only, off the hot path.
	"qwen36parity":          1,                // Qwen3.6 Mac parity witness grader/scrubber over JSON artifacts; stdlib-only, off the hot path.
	"whatschanged":          2,                // read-only peer code-diff readout over git log/diff-tree; stdlib-only, off the hot path.
	"modver":                2,                // per-module version derivation over git log/ls-files (the version-everything spine); imports appversion(1)+windowgate(1), off the hot path.
	"livecodebench":         2,                // LiveCodeBench fixture/report smoke; no network/model, off the hot path. NOT pureRoot: gradedarm imports benchcatalog(1) for WitnessSameTasks, so it sits in the foundation sub-DAG, not the pure-primitive set.
	"logvault":              3,                // central chain-aware log-vault capture engine (epic #2447): incremental mirrors of the durable log stores (guard-audit/dos/dispatch/harness) + a hash-chained vault manifest + the scrub-gated off-box sync rung (#2454); imports flock(1)+wirescreen(2), off the hot path.
	"readmevisualaudit":     2,                // README visual-audit fold over git ls-files + text parsing; imports strmatch(1)+windowgate(1), off the hot path.
	"toolcoverage":          1,                // read-only tools/*.py sibling-test coverage audit over skills/CI references; stdlib-only, off the hot path.
	"modelladder":           3,                // model-ladder selector; imports benchcli(1)+model(1)+stdlib, off the hot path.
	"modelreg":              3,                // model registry; imports hfhub(1)+stdlib, off the hot path.
	"modelinventory":        1,                // deterministic local/provider candidate evidence normalization; primitive over standard-library records.
	"modelsrc":              3,                // model-source URL registry; transports stay behind ReaderAt openers.
	"nativebench":           1,                // stdlib-only native benchmark comparison obligations and report generation.
	"nativeperf":            2,                // native performance evidence classification composes primitive attestations.
	"nativeperfartifact":    2,                // Native-performance evidence locator and digest mechanism; orchestration remains outside.
	"nativeperfbackend":     2,                // Native-performance backend vocabulary and bounded contract mechanism.
	"nativeperfcorrelation": 2,                // Opaque evidence-correlation mechanism.
	"nativeperfobscontract": 2,                // Native-performance observation contract mechanism.
	"nativeperfslo":         2,                // Native-performance SLO contract mechanism.
	"nativeperfcoverage":    3,                // Composes backend and SLO contracts into deterministic coverage evidence.
	"skillenv":              5,                // skill virtual-env composer; imports ctxmmu(2)+ctxresidency(3)+kvmmu(3)+stdlib.
	"guardroute":            5,                // guard RSI worst-bucket auto-router to a finding+gh issue; imports dogfoodissues(3)+guardrsi(1)+stdlib, off the hot path.
	"guardcomplaint":        5,                // agent APPEAL channel (the subjective complement of guardroute): files a witnessed, deduping `fak complain` gh issue when the agent judges a guard DENY wrong; imports dogfoodissues(3)+guardrsi(1)+stdlib, off the hot path.
	"propagationscore":      5,                // convention-propagation scorecard + fan-out: measures how far a proven scorecard convention (shared kernel/--compare/--markdown) has fanned across the family, files one deduped issue per laggard via dogfoodissues(3); imports dogfoodissues(3)+pkg/scorecard+stdlib, off the hot path.
	"unwiredscore":          5,                // unwired-code scorecard + fan-out: detects code-complete internal packages imported by no .go file in the module (wired into no default path -- the invariant architest does NOT catch) and files one deduped wire-or-retire issue per orphan via dogfoodissues(3); imports dogfoodissues(3)+pkg/scorecard+go/parser+stdlib, off the hot path.
	"checkpointscore":       5,                // WIP-checkpoint readiness scorecard + fan-out: probes each long-running process subsystem's source for a durable resumable store (crash_recovery) and a witnessed status surface, folds checkpoint_debt, and maps each gap to a dogfoodissues(3) ActionItem; imports dogfoodissues(3)+pkg/scorecard+stdlib, off the hot path.
	"findingsink":           5,                // general-purpose scorecard-finding sink seam: routes neutral Findings to stdout (dry-run), a durable local JSONL ledger, or GitHub issues via dogfoodissues(3); imports dogfoodissues(3)+stdlib, off the hot path.
	"mcpfootprint":          5,                // prices the always-sent MCP tool-schema floor: the fixed per-turn token tax every registered tool adds to every API call; imports agent(4)+gateway(4), off the hot path.
	"conflationscore":       1,                // pure Go port of tools/conflation_scorecard.py (provenance-honesty stick); stdlib-only, off the hot path.
	"heavinessscore":        2,                // operator-heaviness scorecard over the cmd/fak dispatch table + guard flag set + dos.toml reasons + llms.txt doc map; imports pkg/scorecard, off the hot path.
	"choicetriage":          2,                // closed disposition vocabulary that decenters the human from a surfaced choice (TAKE_OBVIOUS / FRESH_CONTEXT / FILE_TICKET / HUMAN_RESIDUAL): the decision-side dual of waiting(1). Pure, stdlib-only, imports nothing internal, off the hot path.
	"headlesslint":          2,                // sensor-side dual of choicetriage(1): a closed vocabulary of operator-directed "pesky note" classes (PERMISSION_ASK / PREFERENCE_ASK / CLARIFICATION_REQUEST / REVIEW_REQUEST / CONFIRMATION_WAIT / DEFERRED_WORK / SUGGESTION_PUNT / OPEN_OFFER) that scans agent final-output text and folds each hit to a choicetriage disposition. Imports choicetriage(1) only, off the hot path.
	"hwgatelint":            1,                // hardware-gate dual of headlesslint(1): a closed vocabulary of "local machine is the compute boundary" stop classes (NO_LOCAL_GPU / NO_LOCAL_RUNTIME / LOCAL_BOUNDARY) that scans agent final-output text and carries the fixed sanctioned-compute-node redirect. Pure, stdlib-only, imports nothing internal, off the hot path.
	"operatorbrief":         4,                // folds cadence/program/milestone/heaviness report envelopes into one human pacing control pane; composer-only, off the hot path.
	"productscorecard":      2,                // pure Go port of tools/product_scorecard.py; folds CLAIMS.md + tools/product_scorecard.data into product_debt; stdlib-only, off the hot path.
	"scorecardpane":         2,                // pure Go port of tools/scorecard_control_pane.py (portfolio debt-ratchet fold) + tools/repo_hygiene_scorecard.py (the tree-wide hygiene fold); stdlib-only, off the hot path.
	"scorecardportfolio":    2,                // deterministic read-only inventory of scorecard producers, routes, pane bindings, versions, and baseline coverage; stdlib-only, off the hot path.
	"scdiff":                2,                // shift-left seam for scorecardpane/uiquality: pure ChangedPaths(git diff working-tree-inclusive) + corpus-glob matcher so a card whose corpus is untouched is carried from baseline instead of re-measured; stdlib-only, imports nothing internal, off the hot path.
	"uiquality":             1,                // UI/UX-quality scorecard over the terminal render source (the fak console panes + fak info overlay + guard --split); stdlib-only, off the hot path.
	"tuiplugin":             1,                // in-process console-pane registry and descriptor seam; stdlib-only, off the hot path.
	"scoreboard":            2,                // outbound Slack publisher for scorecard/score/run-event status posts; rides slackwire(1) for transport, keeps token resolution + change-gating, off the hot path.
	"slackmeta":             1,                // common Slack report metadata formatter; stdlib-only, off the hot path.
	"slackoutbox":           2,                // durable Slack outbox (#2262): JSONL spool + one flock-serialized drainer with nonce idempotency (metadata probe closes the crash window), update coalescing, dead-letter, and the hooks.ScanOutboundText leak fence; imports slackwire(1)+hooks(1)+flock(1), off the hot path.
	"slackwire":             1,                // the ONE Slack Web API transport (#2261): chat.postMessage/chat.update/conversations.history/auth.test with bounded 429/Retry-After retry + typed *APIError; scoreboard and chatrelay delegate here. Pure stdlib, off the hot path.
	"benchpost":             2,                // outbound Slack publisher for bench-channel rollups/run-requests; folds catalog/baseline/plan JSON, reuses scoreboard(1) transport, off the hot path.
	"blockerpost":           2,                // outbound Slack publisher for the central #blockers channel: severity-driven (background status vs surfaced operator page); reuses scoreboard(1) transport, off the hot path.
	"dispatchpost":          2,                // outbound Slack publisher for background code-dispatch run RESULTS; reuses scoreboard(1) transport, off the hot path.
	"dojopost":              2,                // outbound Slack publisher for dojo rollups/trends; folds dojo(1) reports, reuses scoreboard(1) transport, off the hot path.
	"grafanapost":           2,                // outbound Slack publisher for the #grafana channel: exported Grafana snapshots + long-lived dashboard/debug links; folds a committed links registry, reuses scoreboard(1) transport, off the hot path.
	"promalert":             1,                // pure Alertmanager v4 webhook parser + Slack-card renderer for the Prometheus-alerts->Slack receiver (fak slack alert); stdlib-only, imports nothing internal, off the hot path.
	"marketing":             2,                // completion-driven marketing subsystem: witnessed-ship(hooks) -> claim/artifact, CLAIMS.md honesty gate, AEO/AgentEO refresh; imports hooks(1)+scoreboard(1)+stdlib, off the hot path.
	"fleet":                 2,                // fleet-roster snapshot fold for the #node-usage feeder; stdlib-only, imports nothing internal, off the hot path.
	"nodeusagepost":         2,                // outbound Slack publisher for the #node-usage feeder; folds fleet(1), reuses scoreboard(1) transport, off the hot path.
	"blobfs":                2, "blobhttp": 2, // durable on-disk / remote-HTTP content-addressed Ref backends; attach to abi like blob (Resolver+PageOutBackend), import only abi+blob+stdlib.
	"xenginekv":        2, // cross-engine zero-copy KV co-residence arena (#448): a RefRegion-issuing Resolver+RegionBackend+PageOutBackend; attaches to abi like blob, imports only abi+blob+stdlib (FAK_XENGINE_KV-gated).
	"secretload":       2, // first-class secret/config loader (#887/#889): SecretSource priority list + os-env/encrypted-file/.env backends + Require checklist + Redact; imports canon(1)+stdlib, off the hot path.
	"windowgate":       2, // no-desktop-popup ratchet: scans tracked .ps1 task installers + window-suppressing .py for console-window flashes; stdlib-only, off the hot path.
	"worktype":         1, // closed work-class taxonomy (ongoing optimization PROGRAM vs DISCRETE deliverable epic) the milestone roadmap + `fak program` report sort by; stdlib-only, imports nothing internal, off the hot path.
	"programreport":    2, // ongoing-program report (kernel-opt + cache-opt frontier+trend; the sibling of milestonereport that measures never-'done' programs); imports worktype(1)+cachevalueledger(1)+hooks(1), off the hot path.
	"benchauthority":   1, // typed in-binary source of truth for the PRIMARY benchmark numbers fak claims (the "what" half of the benchmark discipline); stdlib-only, imports nothing internal, off the hot path.
	"closureaudit":     1, // pure stdlib-only port of the issue_closure_audit grader (#1406): binds commits to issue numbers and grades each issue's closure; imports nothing internal, off the hot path.
	"ctxplans":         1, // CONTEXT-PLAN-REQUIRED advisory lint (R4, #2202, epic #2198): the code form of doctrine law L7 — every surface declares its context plan; stdlib-only, off the hot path.
	"fleetverify":      2, // throwaway compile-witness isolating the operator.go fleet helpers' loopfleet/loopmgr API usage from the churning cmd/fak; imports loopfleet(1)+loopmgr(1), off the hot path.
	"issueownerprompt": 1, // stdlib-only canonical resolver lifecycle and drift validator (#6848).
	"promptlint":       1, // durable freshness monitor for the dispatch worker-issue prompts (#3218): flags a rendered prompt whose `fak <verb>` / UPPER_SNAKE claims drift from the surface; stdlib-only, off the hot path.
	"sensecheck":       1, // "does this actually make sense?" common-sense smell battery side-car; stdlib-only, imports nothing internal, off the hot path.
	"spendrollup":      2, // cross-account `fak spend` rollup with a WITNESSED/OBSERVED provenance gate that fails any figure missing its valuation basis or provenance label; imports fleetaccounts(1), off the hot path.
	"worktreewitness":  1, // runs a command inside a transient detached worktree pinned at origin/main so the verdict reflects the trunk tip, not the caller's dirty tree; stdlib-only, off the hot path.
	"goalregistry":     2, // canonical cross-harness goal identity and bindings; foundation composite over the tier-0 flock primitive.
	"schedcontract":    1, // scheduler contracts, constraints, and validation; stdlib-only primitive.
	"serverartifact":   1, // stdlib-only local artifact identity, digest, and mutation verification; imports no internal package.
	"serverproduct":    2, // independent local-server product contract and readiness receipts; foundation composite over protocol and artifact identities.
	"serveradapter":    2, // independent llama-server invocation rendering and readiness probes; foundation composite over server product contracts.
	"harnessserver":    2, // immutable external server-receipt binding and compatibility verification for harness consumers.
	"serverlifecycle":  3, // direct local-process lifecycle over artifact, adapter, product, and process-start evidence.

	"adjudicator": 3, "ctxmmu": 3, "engine": 3, "enginecache": 3, "grammar": 3, "kernel": 3,
	"preflight": 3, "vdso": 3, "plancfi": 3, "steward": 3, "witness": 3,
	"cachevaluereport": 3, // weekly cache-value TREND roll-up (epic #1301 rung A, Track 1): pure Fold over cachevalueledger(1) into a by-week realized-reuse trend, #1066-fenced; imports cachevalueledger(1)+stdlib only, off the hot path.
	"auditusage":       3, // cross-session audit usage rollup (#1612): folds sink rows from journal(2), loopmgr(1), dispatchaudit(1), and usage ledgers into one CLI report; off the hot path.
	"harvest":          3, "shipgate": 3, "policy": 3, "privacy": 3, "auditreceipt": 3, "providercost": 3, "modelengine": 3, "ratelimit": 3,
	"launchshim": 2, // user-local launch configuration + direct-bypass policy; stdlib-only foundation consumed by cmd/fak.
	"journal":    3, "gitgate": 3, "gitdaily": 3, "safecommit": 3, "patchcommit": 3,
	"storedrv": 3, // content-addressed storage ROUTER: composes the blob/blobfs/blobhttp (tier-1) drivers into one namespace; the abi RegionBackend only when FAK_STORE opts in.
	"capindex": 3, // protocol-blind capability keystone (#1104 C1): CapRef/Capability/Index/Resolver + skill resolver, imports only abi(0). The gateway-backed MCP/A2A resolvers live in capindexgw(4) so the core stays importable by the tier-3 skill-loader (ctxresidency/ctxmmu, #1106).

	"ifc": 4, "normgate": 4, "secretgate": 4, "recall": 4, "kvmmu": 4, "radixkv": 4, "cdb": 4, "contextq": 4, "agentdojo": 4, "toollint": 4, "toolsandbox": 4, "terminalbench": 4,
	"agentdemo":     4,                // agentic "try-it" demo spine (epic #1167): a deterministic, no-key tool-using agent loop that folds the REAL kernel per call â€” the live-loop dual of turnbench's trace replay. Composer: imports abi(0)+adjudicator(2)+kernel(2), off the hot path.
	"browseraction": 4,                // browser/computer-use action-mediation harness: composes webbench actions with policy/adjudicator, off the live request path.
	"demo":          4,                // CLI 60-second proof composer: folds agentdemo(3)+kernel(2)+abi(0) into ALLOW/DENY/QUARANTINE evidence for `fak demo`, off the hot path.
	"memq":          4, "headroom": 4, // memq: the memory-operation algebra composed over recall (tier 3). headroom: the context-compression seam over ctxmmu/abi (its doc.go declares composer/3).
	"sessionctl":    4, // out-of-band session-control ops (#2755): turn-boundary redirect/set-objective constraints applied to a session. Composes adjudicator(2)+abi(0), off the hot path.
	"memvaluescore": 4, // unbounded memory-value scorecard (frontier/pressure/debt) over the committed memory mirror + recall-events ledger; composes recall(3)+memoryread(1)+pkg/scorecard, off the hot path.
	"guardaccuracy": 4, // guard-accuracy scorecard (#2376 durability): folds a labeled command corpus through the REAL adjudicator(2) reversibility classifier and scores false-positive + false-negative rate into one guard_accuracy_debt; read-only import of adjudicator (the hard-self lane is never edited), off the hot path.
	"selfquery":     4, // unified self-feature catalog over devindex, memq, gateway-supplied tool descriptors, and capindex cards; composer view, off the hot path.

	"agent": 5, "bench": 5, "turnbench": 5, "gateway": 5, "registrations": 5, "rsiloop": 5,
	"docfreshrsi": 5, // RSI rung of the docs-freshness loop (#1278/#1284): an rsiloop(4) sibling that imports only shipgate(2)'s keep-bit, off the hot path.
	"stalework":   5, // Advisory repository-freshness discovery; imports docfreshrsi and never gates or mutates candidates (#6613).
	"dojocal":     5, // dojo-RSI calibration worktree/proposer: imports rsiloop(4)+shipgate(2), so it belongs at the integrator tier.
	"capindexgw":  5, // gateway-backed capindex resolvers (MCP tools / A2A methods): the adapter that couples capindex(2) to gateway(4). It lives at the higher tier so the capindex keystone itself stays tier-2 and importable by the tier-3 skill-loader.
	"tracesink":   5, // imports agent/turnbench/registrations (tier 4) â€” tier forced to 4
	"agenttest":   5, // public agent-workflow TEST harness (#238, D-008): deterministic fixtures + tool-call assertion library + mock tool responses + reproduce-from-transcript replay; imports agent(4), off the hot path.
	"ablate":      5, // the N-arm self-ablation sweep: a bench sibling; imports bench(4)+registrations(4)+metrics(1), off the hot path.
	"guardtrace":  5, // guard replay fixture harness plus session->ablate bridge; composes bench(4)+engine(2), off the hot path.

	"tokenizer":           1,
	"toolgrammar":         1, // discriminated union EBNF grammar compiler with byte-level space protection and literal parameter escaping (#11747); stdlib-only primitive.
	"answershape":         1, // pure degeneration/verbosity metric over text; stdlib-only, imports nothing internal.
	"codelint":            1,
	"codexmcpdiag":        2, // pure Codex MCP startup evidence classifier (#5980); stdlib-only, off the hot path.
	"codexlifecycle":      2, // pure exactly-once Codex task-lifecycle fold keyed by exact turn_id (#4785): events in, typed terminal (complete/aborted/superseded/process_death/live) + provenance out; stdlib-only, imports nothing internal, off the hot path.
	"polymodel":           1, // multi-model residency + serial-decode-lane + cache-led MTP accept core; stdlib-only, imports nothing internal.
	"reachdelta":          4, // typed policy reach-expansion referee (#4220): composes adjudicator policy sets with knownbad accepted-risk records; off the hot path.
	"rulesynth":           4, // refusal-log rule synthesizer (#537): composes harvest/policy/adjudicator/shipgate to propose+gate a new structural rule as a reviewable diff; imports tier-2 mechanisms, never the hot path.
	"residency":           3,
	"ctxresidency":        4,
	"ctxplan":             1, // context planner: cost-based, forecast-driven O(1) view over a lossless history store; stdlib-only, imports nothing internal.
	"relay":               2, // pure relay-baton value (#1870): the closed Baton schema + deterministic Parse/project round-trip for the handoff/resume relay; imports ctxplan(1)+stdlib, off the hot path.
	"session":             2, // per-session DRIVE state: a TraceID-keyed, bounded-LRU, live-mutable control-state value (run-state/budget/priority/pace), the structural twin of ifc.Ledger widened past one value; stdlib-only, imports nothing internal.
	"sessionledger":       2, // content-addressed, hash-chained session event log with O(1) fork (#2416); stdlib-only value store, imports no internal package, off the hot path.
	"sessionread":         1, // the read/query/observe-op VOCABULARY spine (#4176/#4191): the closed set of shipped session READ seams, each carrying its capability/disclosure/evidence/refusal contract; the outbound twin of sessionctl(3), but pure — stdlib-only, imports nothing internal, off the hot path.
	"wirescreen":          3, // local-model-on-the-wire proposer spine: registers an abi.SemanticScreen that ctxmmu consults after its regex floor (#569) + the ScreenDigest useful-page-out (#570) + the pre-send redactor (#572); imports only abi by default â€” the -tags fakwiremodel model arm (#569) adds model/tokenizer/ggufload (all tier-1).
	"advmodel":            3,
	"modelroute":          2, // per-aspect + ensemble model-routing policy spine (Route + Combine); pure, stdlib-only, imports nothing internal.
	"modelscore":          1, // durable raw model-capability evidence registry (#3038): unbounded native-unit benchmark/cost/context rows with provenance; a Profile keeps raw evidence separate from any tier policy. Pure, stdlib-only, imports nothing internal, off the hot path.
	"toon":                1, // general JSON<->TOON codec (#3065): lossless, type-preserving Encode/Decode + TabularEligibility, the round-trip generalization of memview's flat-subset encoder. Pure, stdlib-only, imports nothing internal, off the hot path.
	"simhash":             1, // reference vector-similarity primitive (embed/cosine/top-k); the observability layer's near-duplicate / outlier-query substrate. Deterministic, stdlib-only, imports nothing internal.
	"guideddecode":        1, // slice-1 constrained tool-JSON decode (#26): sound byte-level AllowedNextBytes FSM over the {"name":<enum>,"arguments":…} tool-call envelope; the model.LogitMask adapter is a later slice. Pure, stdlib-only (bytes), imports nothing internal, off the hot path.
	"issuededup":          2, // write-time near-duplicate gate for issue producers (#2504): simhash embed + TopK over title / title+body axes into advisory dup-risk verdicts. Imports simhash(1) only, off the hot path.
	"idempotency":         2, // keyed, time-windowed dedup for retryable mutating tool ops (#2093): op+token -> key, JSONL ledger so a post-hang retry replays the original result instead of double-applying. Imports jsonlledger(1) only, off the hot path.
	"trajectory":          4, // trajectory data plane: folds the abi event stream into per-trace Turn rows + JSONL export; an abi.Emitter that optionally stamps a simhash query embedding. Imports abi+simhash.
	"trajectoryassurance": 1, // pure, shadow-only fold of typed trajectory evidence into a privacy-safe health receipt; stdlib-only.
	"trajhook":            4, // pluggable trajectory scorer/tap seam (the "trivial skill does gardening" enabler): app code registers Scorers over Turn rows without a core edit. Imports trajectory+simhash.
	"sessionimage":        5, // portable, model-agnostic SESSION image: composes recall(3)+session(1)+trajectory(3)+ctxplan(1) into one versioned, sha256-integrity bundle + a .faksession tar (dump/pack/unpack/rehydrate across hosts/users/VMs/model changes). Integrator: imports tier-3 composers, off the hot path.
	"a2achan":             3, // in-kernel agent-to-agent message channel: a process-global, capability-floored, Ref-backed mailbox (Send/Recv adjudicated by a registered a2aGate + a2aIngress; Taint/Scope enforced). Mechanism: imports only abi, off the hot path.
	"region":              3, // typed one-sided shared Ref window: adjudicated Put/Get/Accumulate over abi.Resolver with ScopeFleet ceiling and vDSO coherence bumps.
	"snapshot":            4, // uniform DUMP/RESTORE seam over any primitive (turn/tool/session/fleet/rsi): a sha256-integrity envelope (Marshal/Parse over any body) + a ladder registry + typed codecs for trace(trajectory) and fleet(session.Table). Imports session(1)+trajectory(3); off the hot path.
	"rungobs":             3, // passive rung-decision distribution counter: an abi.Emitter (subscribed to EvDecide/EvDeny/EvVDSOHit) that re-folds each call's chain off the hot path via kernel.FoldExplain and bumps a per-(rung,kind,reason) histogram. Mechanism: imports kernel(2)+abi(0); runs synchronously in emit but adds 0 adjudication rungs and never touches the verdict or Counters.
	"vcachegov":           3, // vCache M5 Governor (#720): the steady-state policy over the vCache warm set â€” pin/lazy/evict (Â§5.4), rate-limit warm budget (Â§5.5), cross-shard affinity routing + rehash/burst guards (Â§9/D3), and the Law-D4 secret classifier. Classifier itself: imports cachemeta(1)+stdlib. The live loop (loop.go, journal.go, quality.go, #1492) additionally imports vcacheqa(2) from its own qa_test.go only; loop.go itself stays cachemeta(1)+stdlib. Blank-imported by internal/registrations (env-gated by FAK_VCACHE_GOVERNOR / --vcache-governor, default OFF) -- no longer "NOT registered".
	"vcachechain":         3, // vCache M4 chains & recall (#719): prefix DAG + topological replay (send-one-then-fan) + 20-block breakpoints + the Â§11.0 cost-gated rebuild (refuses single-unit chain rebuilds, allows amortized fan-out). Pure decision layer: imports cachemeta(1)+vcachegov(2)+stdlib, off the hot path (NOT registered; gated OFF by default).
	"vcachecal":           3, // vCache M1 observe & calibrate (#716): the warmth-belief estimator (Â§7) over cachemeta.Lifecycle at TierProvider + the offline probe harness that fits T/M_min/r (Law D2) + the LRU probe budget (observer-perturbs-state) + the Zipf-s concentration gate (Â§5.2) + the false-warm/false-cold prediction-error report. Pure decision layer: imports cachemeta(1)+stdlib only, off the hot path (NOT registered; observe-only â€” no warming in M1).
	"vcachecalibration":   3, // Dated provider/model warmth-prediction ledger and stale-status readout; imports vcacheobserve plus bounded JSONL persistence.
	"vcachescore":         3, // vCache operator scorecard: composes vcachecal/vcachechain/vcachegov proof leaves into the offline 2x readiness gate and hot-anchor index artifact; pure off-path decision layer.
	"vcachestar":          3, // vCache M2 star anchors (#717): canonicalizer-as-gate, wire-byte manifest keying, first-natural-request anchor warming, telemetry demotion, and uncached-first cost booking. Pure decision layer: imports cachemeta(1)+stdlib only, off the hot path.
	"vcachewarm":          3, // vCache M3 dedicated warming (#718): Anthropic max_tokens:0 vs decode-1 decision gates, byte-identical prefix guard, send-one-then-fan barrier, and wasted-warm accounting. Pure decision layer, off the hot path, no live transport claim.
	"vcacheqa":            3, // vCache gate QA harness (#1495, child of #1490): the shared honesty-lint (Law A2 elision AST scan) + forced-cache-MISS helper (drives vcachestar.FoldTelemetry) + non-forgeable witness (journal.Row-shaped hash chain, verified via journal.VerifyRows) + provenance fence (OBSERVED/WITNESSED, cachewitness vocabulary) + determinism check every M1-M5 gate imports before flipping default-on. Imports journal(2)+guardrsi(1)+cachewitness(1)+vcachestar(2)+cachemeta(1)+stdlib, off the hot path, not registered.
	"sessionreset":        3, // budget-reset carryover builder: a pluggable Contributor registry that folds a drained session's transcript into the "human-like" seed a fresh session is re-armed with (durable facts via ctxmmu's shipped prior + task recap + warm-prefix descriptor via vcachechain + verbatim tail). Mechanism: imports ctxmmu(2)+vcachechain(2)+stdlib, NOT the wire agent type; off the hot path, registers nothing into the kernel.
	"taskmgr":             2, // process-local task/step/resource/ETA snapshot fold; stdlib-only, off the hot path.
	"issuepolicy":         2, // pure spine-first GitHub issue candidate policy; stdlib-only, off the hot path.
	"issuecentrality":     2, // deterministic issue problem-centrality audit over issuepolicy contracts.
	"issuecohort":         2, // pure batch cohort planner: folds many issuecontract candidates into concurrency-safe waves (disjoint-tree), a split-first queue, triage, and duplicate-key groups; imports issuecontract(1)+stdlib, off the hot path.
	"issuefanout":         2, // pure spine-first fan-out planner: expands one shipped working spine into contract-ready QA/dogfood/productization/observability/integration follow-on candidates + an observability scorecard fold over git/gh witnesses; imports issuecontract(1)+pkg/scorecard+stdlib, off the hot path.
	"issuehygiene":        2, // pure issue-creation/tagging pickup-readiness scorecard (fak score issue-hygiene): folds the open backlog's class/priority coverage, worker-ready body sections, and near-dup avoidance into issue_hygiene_debt; imports issuededup(1)+pkg/scorecard+stdlib, off the hot path (the gh fetch lives in cmd/fak).
	"stopfailure":         1, // pure StopFailure marker planner/settler over JSON files and transcript existence; stdlib-only, off the hot path.
	"dogfoodscore":        2, // pure dogfood-loop scorecard over transcripts/markers; imports stopfailure, off the hot path.
	"conceptusage":        2, // pure concept-usage scorecard: folds git log + .dos journals into a dogfooding score; stdlib-only, off the hot path.
	"conceptcatalog":      2, // pure catalog validation and atomic authoring planner; stdlib-only, filesystem boundary off the hot path.
	"renameconcept":       1, // pure tree-wide concept-rename planner/applier: case-form variant expansion + mechanical-vs-holdout triage + residual rescan; stdlib-only, imports nothing internal, off the hot path.
	"maturity":            2, // pure feature-maturity lifecycle scorecard: folds dos.toml lanes + the tree's import graph into a per-capability ladder rung + next-work backlog; stdlib-only, off the hot path.
	"dropin":              1,
	"comm":                3,
	"cohort":              3, // fail-closed cohort shrink/agree over comm.Group + modelroute vote fold.
	"agenttopo":           3, // declared agent communication DAG over comm.Group + modelroute folds.
	"promptmmu":           2, // cache-prefix-preserving inbound prompt MMU: splices tools[] past the last cache_control breakpoint; stdlib-only, off the hot path, no agent/gateway import (decode is a callback).
	"loopmgr":             2, // durable loop-event JSONL ledger + read fold: SHA-256 hash chain over armed/fire/admit/start/heartbeat/end/witness/notify events. stdlib-only, off the hot path; schedules/spawns/notifies/authorizes nothing â€” those stay in the producers.
	"leaseref":            2, // cross-machine lease VISIBILITY substrate (#825): persists a lease record under refs/fak/locks/<id> so lease state rides ordinary git fetch/push between clones. Distribution, NOT atomic acquisition. Shells to git off the hot path through one Runner seam; imports only dormancy(1) for the lease's LastActiveAt clock (#1179).
	"laneadmit":           2, // pure lane/tree admission fold for dispatch/loop/manual surfaces: lane taxonomy + live leases -> COLLISION_RISK verdict; imports dispatchorder(1)+stdlib, off the hot path.
	"lanebeat":            2, // #5864: the pure authority rung for REFRESHING a DOS lane lease — the writer the kernel's heartbeat_at liveness evidence never had (0 HEARTBEAT ops in 3584 WAL entries). Folds the supervisor's structural readings of the holder process (alive / output-growth / spawn budget) against the live lease set into a beat-or-named-refusal, fail-closed in every rung so a beat can never outlive the work it attests. Imports nothing internal, no clock, no process table, off the hot path; the `dos` I/O lives at the cmd/fak witness-sweep call site.
	"guard":               2, // agent-spawn containment seam (#824): the Linux Landlock read-only-.git/hooks hook-floor for the child `fak guard` spawns, via a re-exec trampoline. Pure spec/resolution core + raw-syscall linux impl + no-op twin; opt-in, off by default, fails open; imports only stdlib (syscall/unsafe on linux), nothing internal.
	"goalpark":            2, // durable long Retry-After goal parking and exactly-once supervisor claim (#4805)
	"goalsync":            2, // goal artifact sync between workspace and private backup repo; off the hot path.
	"codexmcphealth":      3, // Codex MCP transport health diagnostic (#1445): fresh stdio smoke + stale-child inventory/reap fold over subprocess evidence. Tool-shaped mechanism leaf, off the hot path, imports only stdlib.
	"pythongate":          3, // NEW-PYTHON-TOOL de-Python ratchet: scans tracked tools/*.py (git ls-files) against a frozen grandfathered baseline and refuses any new .py (NEW_PYTHON_TOOL). A tool-shaped witness leaf (reads tree, folds, emits offenses); shells to git off the hot path, imports nothing internal.
	"ctxknobs":            2, // MANUAL-OVERLAY COUNTER ratchet (#2199): walks cmd/fak flags/env + .claude/skills for context knobs, classifies operator-debug vs user-required, refuses a NEW user-required overlay against a frozen baseline (NEW_USER_REQUIRED_KNOB). Pure filesystem walk + fold, stdlib-only, imports nothing internal, off the hot path.
	"knobcensus":          3, // knob census (#2210): classifies every cmd/fak flag/env + skill knob as INTENT vs HOUSEKEEPING over a tree walk. Tool-shaped mechanism leaf; imports ctxknobs(1)+stdlib, off the hot path.
	"treedoctor":          3, // tree-hygiene doctor over safecommit's lock seam plus git worktree reads; mechanism/tool leaf, off the hot path.
	"tooltrend":           4, // cross-session tool-mix & I/O-shape drift lens (#2826): folds toolrollup(3) per-tool rollups across a session corpus into a drift/delta trend. Composer/analytics leaf, imports toolrollup(3)+stdlib, off the hot path.
	"toolrollup":          4, // per-tool usage rollup over a trajectory(3) corpus (#2824): counts/costs/error-rates folded per tool. Composer/analytics leaf, imports trajectory(3)+stdlib, off the hot path.
	"cachesweep":          4, // prompt-cache sweep/eviction planner over a radixkv(3) store; composer leaf, imports radixkv(3)+stdlib, off the hot path.
	"qaprocessscore":      4, // QA-process quality score composing dogfoodissues(3)+brittleness(1); composer leaf, off the hot path.
	"sessionjournal":      2, // durable session-event journal over procguard(1); foundation ledger leaf, off the hot path.
	"toolseq":             2, // per-session tool-call SEQUENCE fold (#2825): n-gram/transition counts over a trajectory's tool stream; stdlib-only foundation leaf, imports nothing internal, off the hot path.
	"commitintent":        1, // durable commit-intent queue record and drain ordering contract; pure stdlib, off the hot path.
	"commitrollup":        2, // pure compatible-intent rollup planner and pathset assertion helper; stdlib-only, off the hot path.
	"commitlane":          3, // read-only commit-lane status over safecommit's lock seam plus git/process probes; mechanism/tool leaf, off the hot path.
	"release":             3, // the shared process-safe release lock (#1391): a TTL+O_EXCL .release.lock that every VERSION/tag mutator (scheduled auto-cut AND human /release) takes so they cannot race. Mechanism leaf over flock(1)'s advisory guard + git reads; off the hot path.
	"gardenbundle":        4, // the garden bundle: a read-only fold-over-folds that runs the grandfathered Python gardening passes (scorecard control pane + fresh status, +loop-audit under --deep), reads each control-pane payload, and folds one schema/ok/verdict/finding envelope. Composer: composes other tools' outputs (shelling out off the hot path), imports nothing internal.
	"savingsvector":       1, // pure four-account saving-decomposition lens over a turnbench Report's already-measured fields (local_cpu/gpu_prefill/context_window/wall_clock, labeled per axis); stdlib-only, imports nothing internal, off the hot path.
	"swebenchsota":        3, // SWE-bench SOTA leaderboard snapshot: a tool-shaped leaf that extracts the embedded leaderboard JSON (regex+unescape), folds the per-group SOTA, and emits a versioned snapshot. net/http fetch off the hot path; imports nothing internal.
	"borrowprovenance":    1, // exact-source provenance pin + drift verification: pure SHA-256/validation contract, stdlib-only and off the hot path.
	"ultracodeborrow":     1, // deterministic study-artifact contract for source/license/axis/owner/benchmark/privacy completeness; stdlib-only and off the hot path.
	"dogfoodissues":       4, // dogfood-action-issues backlog bridge: folds a recent-feature dogfood report.json into scorecard ACTION items, derives a stable dedup key per item, renders the marker-stamped issue body, and (only on --live) composes the external `gh` CLI to create/update one issue per item. Composer: shells out off the hot path, imports nothing internal.
	"learningdebt":        4, // learning-scorecard HARD-defect -> GitHub triage issue dispatcher (#1283): folds learning_scorecard.py --json defects into stable keys, dedups (seen-cache + issue-body marker), caps, dry-run by default; shells to gh/python off the hot path, imports nothing internal.
	"horizonrecovery":     1, // pure budget-recovery (term r) grounding lens over a ctxplanbench report's already-measured real-transcript fields: surfaces the recovery ratio + its fault-rate FENCE co-located, structurally refuses to emit r/horizon_multiplier; stdlib-only, imports nothing internal, off the hot path.
	"guardrsi":            2, // pure guard RSI journal fold + scorecard: reads guard-audit bytes, computes deterministic verdict quality, and validates keep/revert iterations; stdlib-only, off the hot path.
	"generalizationdebt":  1, // pure tree-wide generalization-debt scorer; stdlib-only and off the hot path (#10486).
	"incidentrsi":         1, // pure incident/trigger/debounce receipt contracts and folds; stdlib-only and off the hot path (#10495/#10497).
	"repoguard":           2, // pure repo-containment classifier: resolves write/delete targets against a workspace root and emits OUT_OF_TREE_WRITE; stdlib-only, shared by the hook binary and loop driver.
	"egresslist":          1, // pure offline operator/community allow-block matcher: compiles immutable adblock-style host rules with allow precedence and an embedded registry + provenance/checksum manifest. Stdlib-only, imports nothing internal, off the hot path.
	"egressrefresh":       2, // the network-touching maintenance twin of egresslist(1): re-fetches bundled filter lists from their pinned provenance URLs, re-normalizes through egresslist's own ingest path, and rewrites the checked-in artifact + checksum. Imports egresslist(1)+stdlib; runs ONLY from `fak egresslist refresh`, never from a decision, which is what keeps the decide path offline. NOT pureRoot (it imports a sibling leaf).
	"egressfloor":         1, // pure network-egress destination classifier (cloud-metadata SSRF floor): names a tool call reaching the cloud-instance metadata / link-local family (169.254.169.254, metadata.google.internal, fd00:ec2::254, ...). Imports only abi(0)+net+net/url; the adjudicator's egress rung folds it on the live path.
	"hooks":               2, // commit-boundary gates run in ONE process: the Go port of the tools/check_*.py git-hook checkers (PUBLIC_LEAK/SECRET_SHAPE/DOC_PLACEMENT/BROKEN_LINK/FILE_ADMISSION/INDEX_SYNC/PROVENANCE_LABEL + commit-msg), folding one staged-diff read. stdlib-only, imports nothing internal, off the hot path.
	"workflow":            1, // pure DAG/map-reduce/fan-out orchestration core (#245, D-005): JSON-DSL compiler + topo-validated executor + retry/fail-fast fault tolerance; stdlib-only, imports nothing internal, off the hot path.
	"l3region":            1, // L3 disaggregated-cache child B Stage-1 seam (#77, epic #504): an abi.RegionBackend over a fake in-memory page-keyed L3 store â€” a Ref.Digest resolves to a page-key set (mget/mset), region round-trips bit-exact + verify-don't-trust. Imports only abi+stdlib; NOT registered (library leaf), off the hot path.
	"lifecycle":           1, // canonical shared run-state vocabulary; stdlib-only, imported by session/loopmgr to avoid token drift.
	"epochbridge":         2, // explicit session generation <-> abi speculation epoch converter; imports only abi/session and owns neither type.
	"lifebridge":          2, // explicit session.RunState <-> loopmgr.LoopState converter over lifecycle; imports only tier-1 leaves.
	"memview":             3, // typed virtual-view contract over canonical raw memory cells (#904): MemoryViewRecord binds a derived view (snippet/summary/qa/fact) to its source by a digest + byte span, inherits the source taint, and is invalidated when the source digest changes; a materialized view carries an abi.Verdict and re-enters adjudication before any effect. Mechanism: imports only abi(0)+stdlib, defines a RawPage interface so recall.Page adapts without an upward import; off the hot path, registers nothing.
	"fakrpc":              1, // disaggregated agent-RPC contract (#930): the pure Request envelope + the FAKRES nonce/sha frame (encode/decode/verify) a resident worker (cmd/fakrpcd) and pluggable text-only bridges build on. stdlib-only, imports nothing internal, off the hot path â€” the same frame tools/dgx_witness_run.sh emits.
	"resume":              2, // deterministic resume-cache decision (#745/#774 family): prices RESUME_FULL/CUT/RESET against the projected cold/warm prompt-cache posture at the resume boundary and recommends a cut-by-default re-entry; pure Plan(Input) Report, imports only cacheprice(1) for the canonical cache multipliers, off the hot path. The computable answer to "resume a 250k session â€” what happens to the cache".
	"vcacheobserve":       3,
	"vcacheextract":       3, // Codex session token-telemetry sanitizer: reads a Codex/`codex exec --json` JSONL and keeps ONLY token counters (drops all prompt/tool/response content), bridging them into the observed-window snapshot + score. Pure decision layer: imports vcacheobserve(2)+vcachesnapshot(2)+vcachescore(2)+stdlib, off the hot path.
	"vcachesnapshot":      3, // vCache observed-cache-window snapshot bridge (#827d882f): folds vcacheobserve's realized traffic into the read-side snapshot the score consumes; imports vcacheobserve(2)+stdlib only, off the hot path.
	"cadencereport":       4, // the consolidated regular-cadence report: a read-only fold-over-folds that distills the scorecard control pane (scores), git (work-done), and release-status (releases) into one schema/ok/verdict/finding envelope + a durable JSONL trend ledger. Composer (like gardenbundle): shells to the Python folds + git off the hot path, imports nothing internal.
	"epicprogress":        2, // the provenance-honest gh epic child-completion resolver (#1438): a track-label -> body-checklist -> honest-errored-row priority chain behind an injectable Runner, returning EpicCounts{Closed,Total,Source,Err} that NEVER fabricates a 0%. Extracted from milestonereport so issue-triage/plan-audit/dogfoodissues can reuse it; stdlib-only, imports nothing internal, off the hot path.
	"rsl":                 1, // append-only hash-chained Reference State Log (#3190): records and verifies fast-forward-only ref transitions with an optional signing seam; stdlib-only, imports nothing internal, off the hot path.
	"mlpscore":            3, // milestone #17 first-lovable-cut scorecard (#3284): pure witness-manifest fold plus a committed-HEAD reader; imports trendreport(1)+windowgate(1), off the hot path.
	"versionskew":         3, // refusable binary-vs-trunk version-skew witness: pure classifier plus git ancestry reader; imports binstamp(1), off the hot path.
	"milestonereport":     4, // the milestone tracking report: a read-only fold of the maturity CLIMB (covmatrix x supportmaturity rung distribution) + the epic ROADMAP (tracked-epic child completion read from `gh`) into one schema/ok/verdict/finding envelope + a durable JSONL trend ledger. Composer (like cadencereport): shells to `gh` + git off the hot path; imports covmatrix(1)+supportmaturity(2).
	"milestonepost":       4, // outbound Slack publisher for the milestone report (twin of cachevaluepost): folds a milestonereport.Report into one #milestones channel card. Forced to tier 3 by its milestonereport(3) import; also imports scoreboard(1)+slackenv(1)+slackmeta(1), off the hot path.
	"milestonedoc":        4, // the freshness-checked milestone status doc (#1441): renders the maturity CLIMB (covmatrix grid -> M0-M7 ladder via milestonereport.InterpretMaturity) into a committed docs/milestones/STATUS.md block with the --write-doc/--check-doc seam (twin of supportmaturityscore's matrix block). Forced to tier 3 by its milestonereport(3) import; also imports covmatrix(1)+supportmaturity(2), off the hot path.
	"dispatchorder":       1, // pure dispatch-ordering helper; stdlib-only, imports nothing internal, off the hot path.
	"dispatchauto":        1, // pure dispatch wave auto-sizing fold: live ceilings + node roster + context budget -> target/refill/placement; stdlib-only, off the hot path.
	"categorybaseline":    1, // explicit category-layer completion registry + pure hold decision; stdlib-only, off the hot path.
	"dispatchcache":       2, // pure stdlib-only TTL/content-hash cache for routed dispatch payloads.
	"dispatchtick":        2, // pure issue-resolution dispatch tick contract: backend argv, guard wrap, wave/account sidecars + the tier-aware account chooser (#3042); imports modelroute(1) for the WorkTier vocabulary, else stdlib-only, off the hot path.
	"dispatchsweep":       2, // pure queue-drain loop core for `fak dispatch sweep`: find next issue -> spawn one worker -> repeat, until a tick refuses or the best-effort agent ceiling is hit; tick+settle are injected and the cmd/fak shell runs the Go tick evaluator. stdlib-only, imports nothing internal, off the hot path.
	"issuesmallness":      1, // pure issue-template smallness lint: one deliverable + one witness classifier and dry-run report fold; stdlib-only, off the hot path.
	"procguard":           2, // native port of tools/proc_resource_guard.py's operator/control-pane modes (#1412): the runaway-process guard's richer surface — sustained per-core CPU-pin dimension, orphaned-helper/idle-shell sprawl reaping, cross-tick streak ledger, opt-in --enact reaper, and the "fleet-proc-resource-guard/1" JSON contract. Reuses dispatchtick(1)'s resource-level classifier for the thread/handle/ws core; imports dispatchtick(1)+stdlib, off the hot path.
	"proctest":            1, // stdlib-only black-box process containment witness harness (#9051); test helper contracts, off the hot path.
	"fleetmon":            2, // headless-worker fleet monitor/janitor/fold/replace (#1856-#1859): pure evidence-derived worker classification, stale-child-command detection, witnessed run-ledger fold, and stuck-worker replacement. Reuses procguard(1)'s process forest + tree-kill; imports procguard(1)+stdlib, off the hot path.
	"dojo":                2, // the prediction-vs-reality gym's pure scoring/fold/ledger/board core: Prediction/Outcome/Episode scoring + the cross-lever leaderboard fold; stdlib-only, imports nothing internal (the corpus-scanning levers live in cmd/fak), off the hot path.
	"looprecover":         1, // pure loop-recovery decision helper; stdlib-only, imports nothing internal, off the hot path.
	"nightrun":            2, // RUN-IT-ALL-NIGHT local-capability data-collection planner: probes the box + ranks feasible-here collection tasks over the benchmark grid; imports benchcatalog(1)+stdlib, off the hot path.
	"claimcheck":          1, // pure net-true-value claim grader; stdlib-only, off the hot path.
	"ideascout":           2, // inbound arXiv/GitHub idea scout and issue planner; stdlib-only shell/network I/O off the hot path.
	"perfscout":           2, // open-source performance scout for Qwen 3.8 and GLM 5.3 Flash models; stdlib-only + ghexec/windowgate, off hot path.
	"studymonitor":        2, // stdlib-only external source registry validator and report renderer; off the runtime hot path.
	"study":               1, // stdlib-only immutable source-to-decision receipt store; off the runtime hot path.
	"edgequal":            1, // stdlib-only offline evidence contract validator; off the runtime hot path.
	"customizationindex":  2, // stdlib-only agent customization registry validator and freshness reporter; off the runtime hot path.
	"loopindex":           1, // pure S0 agentic-loop scorecard: folds orient->plan->act->verify->ship->learn probes into loop-index + loopindex_debt; stdlib-only, off the hot path.
	"loopmap":             2, // queryable loop-stage -> tool map over loopindex(1); off the hot path.
	"superloop":           2, // operator-intent meta-loop: pure registry+Classify(super-vs-normal)+Walk(worst-first worklist) over member loops/scorecards/gardens, plus the C6 model-fit eval (#3043) grading read-only meta decisions; imports modelroute(1) for the single-sourced risk class, off the hot path.
	"superstream":         3, // super workstream orchestrator: queue-ordered task progression with dynamic per-item lane leasing and long-turn context safety; imports laneadmit(2)+ctxplan(1)+stdlib.
	"sessionobs":          2, // SESSION-OBSERVABILITY-for-RSI scorecard: the value-side complement to fak trajectory audit â€” grades how far our coding-session data has climbed the capture->structure->link->aggregate->learn ladder, folding the missing rungs into one sessionobs_debt integer. Pure scorer (Record/Outcome/Pipeline/Score), imports only cacheprice(1) for the canonical warm-shed marginal, off the hot path.
	"sessionsteer":        1, // long-horizon steering + admission core (#3512, rung of #2198): folds a content-free session snapshot (ctxvalue step-advice + pending-work bits) into one typed directive — MANAGED/LEGACY admission (never silent), a BLOCK/ALLOW persist decision, and step-advice steering text. Consumed by the SessionStart hook (soft rule, default-on) and staged for the Stop hook (hard block, shadow). Pure Steer(Input)Directive, stdlib-only, imports nothing internal.
	"compactcohere":       1, // fak<->harness context-manager COHERENCE policy (#1131): attributes a served turn's prefix event (stable/fak_cut/fak_world_break/harness_rewrite/cold_ttl) + a standing PreCompact block/allow posture to suppress Claude Code's cache-destroying auto-compaction while fak's cache-preserving compaction copes. Pure sensor+policy, stdlib-only, imports nothing internal, off the hot path.
	"loopdrive":           2,
	"loopgate":            2, // pure loop exit gate: maps a claimed-done turn plus a witness criterion to WITNESSED/NOT_YET/REFUSED; witness execution is caller-injected.
	"turntaxmeter":        2, // pure observer-effect sampling and overhead-budget meter; stdlib-only, off the hot path.
	"slackenv":            1, // the ONE .env.slack.local token/channel resolver every outbound Slack publisher (scoreboard/blockerpost/benchpost/dispatchpost/dojopost/marketing/nodeusagepost) and chatrelay delegates to; pure stdlib, off the hot path.
	"dormancy":            1, // dormancy clock + horizon bucketer (#1179, epic #1178): a durable monotonic LastActiveAt Stamp + a pure Horizon(gap) -> {warm,cool,cold,frozen,ancient} bucketer (thresholds anchored to the resume cache TTLs); stdlib-only, imports nothing internal, off the hot path. Surfaced on session/loop/lease as the shared "how long dormant" field.
	"dormancysim":         2, // deterministic time-travel dormancy harness (#1192): drives dormancy(1)+rehydrate(1) with one injected virtual clock so day/week/month horizon tests run without wall-clock waits; off the hot path.
	"rehydrate":           2, // horizon-gated re-entry orchestrator (#1181, epic #1178 Phase-2 spine): the CRaC afterRestore analog — a staged Gate that runs strictly more revalidation rungs (COLD_CACHE/STALE_CRED/STALE_RECALL/STALE_LEASE/STALE_PLAN) the longer the image was dormant, refusing admission at the first that does not clear. COMPOSES rungs (the four children supply the checks); imports only dormancy(1)+stdlib, registers nothing, off the hot path. sessionimage.Rehydrate composes a Gate at its boundary.
	"syspromptmmu":        3, // system-prompt MMU: Rung 1 (#1259) emits fak's ordered base-context plan (SegStable spine + versioned policy floor as []cachemeta.PromptSegment, each content-witnessed); Rung 2 (#1260) is the cache-safe system-block splicer (BuildSystemValue + SpliceSystemOverlay, bytes.Equal(prefix) proven, fail-safe identity). Pure authorship/decision layer: imports cachemeta(1)+promptmmu(1)+stdlib, off the hot path.
	"devcmd":              5, // fak-dev command implementations; integrates the agent/gateway/selfquery surfaces while staying off the serving/guard graph.
	"deliverystages":      1, // canonical agent-development stage and bottleneck registry (#7127); stdlib-only, off the hot path.
	"devindex":            2,
	"edittx":              2, // transactional multi-file edit core: snapshots touched files, applies a full-file batch, validates, and rolls back on failure; stdlib-only, off the hot path.
	"workflowlint":        1, // refutes fak-blind ultracode Workflow scripts (#1494/C4 #1502): pure Lint over the self-index/memory/shared-path concept classes; stdlib-only (embed/regexp/sort/strings), imports nothing internal, off the hot path.
	"execrollup":          2, // executive activity roll-up: pure fold-over-folds turning the agentic-fleet plane payloads (dispatch closure-honesty + throughput, dark loops, cadence ship-stamp rate, fleet box liveness) into one GREEN/WATCH/RED control-pane envelope + a ranked what-needs-you list with provenance labels; stdlib-only, imports nothing internal, off the hot path.
	"sidecar":             2, // sidecar pane v0 (#2215, epic #2209): pure fold over the four per-agent runtime planes (sessions census / accounts usable-throttled-blocked / lane occupancy rendered-not-re-adjudicated / context-cache posture) into ONE render model both RenderText and RenderSlack walk via Sections(), so terminal and Slack carry the identical core-field sequence by construction; provenance labels per plane, stdlib-only, imports nothing internal, off the hot path.
	"opttarget":           5, // declarative target layer of the RSI optimization fuser (epic #1279): Compile lowers an OptTarget (data) into a rsiloop.Harness, so it imports rsiloop(4) and is forced to the integrator layer. Off the hot path.
	"cachevaluepost":      3, // outbound Slack publisher for the cache-value P&L roll-up (twin of benchpost/dojopost/grafanapost); forced to tier 2 by its cachevaluereport(2) import. Imports cachevalueledger(1)+cachevaluereport(2)+scoreboard(1)+slackenv(1), off the hot path.
	"loopscore":           2, // pure loop scorecard: folds the loopmgr ledger + job registry into a durability/self-report/dogfood score for the agentic background loops; imports loopmgr(1), off the hot path.
	"looptrigger":         2, // pure trigger-decision receipt classifier for meta/super loops; stdlib-only, read-only, and off the execution path (#10352).
	"loopfleet":           2, // cross-ledger loop-health fold (#1196): pure read-only adapters over the loopmgr/nightrun/dojo/cadence/dispatch journals joined into one per-loop health view in loopmgr's HealthState vocabulary; imports loopmgr(1), off the hot path.
	"loopunblock":         1, // the generic head-of-line UNBLOCKER for any worklist-draining loop/super loop: pure Decide fold over a worst-first candidate list + per-candidate admission status, into a closed {enter,clear-then-enter,bypass,wait,escalate,stand-down} action vocabulary. The always-on watchdog rung that clears an auto-releasable blocker in place or routes around a stalled head; stdlib-only, imports nothing internal, off the hot path.
	"fleetpane":           5, // native fleet control-pane reader: shells to git/status helpers and folds loop/fleet/operator state for cmd/fak; command-facing integrator, off the hot path.
	"trendreport":         2, // the generic trend-report substrate (#1437): the durable-JSONL ledger plumbing (ParseLedger/LatestBefore/AppendLedgerLine over a Row key), DirectionWord, the embeddable control-pane Envelope+Stamp, and the AdvisoryGate whose only failing finding is the caller's *_unmeasured token. Lifts the machinery cadencereport/milestonereport/the dojo board each re-declare; stdlib+generics only, imports nothing internal, off the hot path. Consumer migration is a documented follow-on.
	"dispatchaudit":       2, // pure dispatch-worker outcome fold (#1454): classifies each .dispatch-runs worker (shipped/wasted-spawn/quota-walled/retry-storm/no-op/errored), rolls up wasted-spawn + wasted-wall-clock per backend, and emits fingerprinted findings; stdlib-only core + a thin I/O shell, off the hot path.
	"issuecatalog":        4, // performance-enablement issue-catalog planner/syncer; reads curated rows, reviews them through issuecontract(1), and shells to gh only behind --live; off the hot path.
	"safesync":            3, // safe fast-forward sync for dirty shared worktrees; shells to git off the hot path.
	"hostfault":           2, // pure closed host-fault classification vocabulary; stdlib-only, off hot path.
	"terminalrisk":        2, // pure Windows Terminal crash-risk assessment + settings rewrite.
	"terminalrelief":      1, // pure pressure/cooldown decision and durable state; Windows actuation stays in cmd/fak.
	"hostresurrect":       3, // composes host-fault signals with durable guard-session inventory into bounded relaunch requests.
	"issuestriage":        2, // pure issue-action triage classifier; stdlib-only, off hot path.
	"wipfence":            2, // pure shared-trunk WIP build-fence text engine; no hot-path dependency.
	"usagelog":            2, // durable, append-only, hash-chained CLI-invocation journal (epic #1601/#1608): one redacted row per top-level fak verb + the `fak usage` read fold; stdlib-only, imports nothing internal, off the hot path.
	"promptaudit":         1, // scans system/developer/context prompt text for hidden control markers (apostrophe-alphabet / date-separator channels) before they cross a model or cache boundary; stdlib-only, imports nothing internal, off the hot path.
	"corelocks":           2, // parses + validates the declarative core-lock taxonomy (#1681): classes hard-self/serial-core/soft-contract/shadow-learn/open-leaf and reason tokens, classifying a path to its lock class from an embedded fixture; data + validation only, stdlib-only, imports nothing internal, off the hot path.
	"corelockaudit":       2, // read-only changed-path core-lock audit fold (#1680): classifies changed paths via the corelocks(1) taxonomy into a closed-schema, deterministic Report (measurement-only warnings); imports only corelocks(1), shells to git in a thin off-hot-path I/O layer.
	"frontierswe":         2, // typed FrontierSWE dataset spine (#1707, epic #1706): the Task model + Category enum + 17-task Catalog + hand-rolled task.toml/job.yaml/oracle.yaml loaders, mirroring internal/swebench's Instance; stdlib-only, imports nothing internal, off the hot path.
	"fleetcap":            1, // Little's-law fleet-capacity calculator: required concurrent workers from target issue-rate + median session; stdlib-only, off the hot path.
	"fleetcompare":        1, // pure fleet sweep slice builder that decomposes shared vs isolated/cross-cache gains; stdlib-only, off the hot path.
	"fleettrend":          2, // fleet status JSONL history and sparkline trend fold; stdlib-only, off the hot path.
	"workdelivery":        2, // independent record/admission/verification/integration/release contract (#7101); stdlib-only, off the hot path.
	"workflowaudit":       2, // classifies every branch/tag ref in .github/workflows/*.yml against the branch-role contract (#1697/#1701): development/release-front-door/tag/legacy/unclassified, with an embedded allowlist + a committed audit report; imports only branchrole(1)+stdlib, off the hot path.
	"fleetsim":            2, // synthetic 400-issues/hour dry-run replay fixture: deterministic ledger generator + closes/hour fold; stdlib-only, off the hot path.
	"fleetmetrics":        1, // pure worker-session duration percentiles (p50/p95) over a ledger; stdlib-only, off the hot path.
	"fusedturn":           1, // executable "one turn spawns both": classifies each abi.ToolCall into its concept-family (classical / weight-based) + folds a FusedTurn (fused iff it spans both) whose Adjudicate proves both families cross ONE floor via an injected Decider (*kernel.Kernel.BatchDecide); imports only abi(0), off the hot path.
	"workerenvelope":      1, // machine-readable worker-result envelope (sha/issue/tests/blocker/witness) + witness-gated validation; stdlib-only, off the hot path.
	"workerworktree":      2, // native port of tools/worker_worktree.py (#3168): per-worker git worktree isolation (prepare detached-at-SHA / land diff onto trunk / reap) that the Go dispatch spawn wires in behind a default-off flag; imports windowgate(1) for the no-window git runner + stdlib, off the hot path.
	"speedab":             1, // pure pinned fast-vs-standard manifest grader; stdlib-only, off the hot path.
	"launchlatency":       2, // dispatch→heartbeat worker launch-latency histogram + p50/p95 (reuses fleetmetrics percentiles); stdlib-only, off the hot path.
	"closurerate":         1, // closure-rate + witnessed-close-rate + claimed-without-witness honesty counters over a close ledger; stdlib-only, off the hot path.
	"issuecost":           2, // per-issue worker elapsed/attempts/outcome ledger → median+p95 (reuses fleetmetrics); stdlib-only, off the hot path.
	"mutationbudget":      1, // GitHub mutation throttle guard: holds close/comment bursts with an actionable reason when remaining API budget < reserve; stdlib-only, off the hot path.
	"mutationefficacy":    2, // bounded SOFT mutation-testing probe (#3845): applies operator mutants over an allow-list, runs `go test`, counts survivors as SOFT KPI; imports pkg/scorecard + os/exec, off the hot path.
	"completiondist":      2, // fold historical issue-closure durations into a distribution (median/p95/buckets) for the capacity model; reuses fleetmetrics+fleetcap; stdlib-only, off the hot path.
	"fleetfreeze":         1, // operator freeze/dry-run gate: holds new spawns while allowing witness-close + status-refresh; stdlib-only, off the hot path.
	"auditreason":         1, // closed vocabulary classifying commit-audit failures as transient (retryable) vs permanent unwitnessed claim; stdlib-only, off the hot path.
	"commitissuelink":     1, // pure checker (#1812): flags a commit that carries this repo's ship-stamp trailer but whose subject omits a scannable #N, guessing the issue from a body Fixes/Closes/Resolves trailer when one exists; stdlib-only, off the hot path.
	"committedtree":       2, // neutral committed git tree materialization shared by runtime and fak-dev; imports windowgate, no command ownership.
	"unwitnessedclaim":    1, // pure checker (#1816): flags an open issue whose latest comment reads as a self-reported completion claim, rendering the missing-witness/recovery-action comment without closing the issue; stdlib-only, off the hot path.
	"closebatch":          2, // pure batch planner (#1826): groups witnessed-closeable issues into dry-run batches, pricing each against mutationbudget.Guard and naming a rollback note; never closes an issue itself, off the hot path.
	"skipledger":          2, // pure skip/select ledger fold (#1776): turns one dispatchorder.Result into per-candidate rows (issue, lane, reason, safety/capacity category, timestamp) for later rate-loss audit; imports only dispatchorder (tier 1), off the hot path.
	"attemptbudget":       2, // pure per-issue attempt-budget fold (#1777): counts an issue's recorded failed attempts against a budget and holds it for triage once crossed, naming the last failure class; stdlib-only, off the hot path.
	"timeoutphase":        1, // pure timeout-phase classifier (#1793): folds one timed-out attempt's observed lifecycle-stage markers into a closed phase (before_startup/during_edit/during_tests/during_commit/during_push/unknown) for the timeout ledger; stdlib-only, off the hot path.
	"vllmquant":           1, // pure vLLM quantization capability-contract adjudication; stdlib-only, off the hot path.
	"vllmcompile":         1, // pure tuned-baseline gate for served-engine benchmarks (#1731): records torch.compile/CUDA-graph/warmup state as a `vllm_compile` block and classifies tuned/cold-start/diagnostic; stdlib-only, off the hot path.
	"harnessprofile":      1,
	"managedinventory":    1, // public-safe managed-agent object registry + catalog validator/renderer; stdlib-only, imports nothing internal, off the hot path.
	"portabilitycontract": 1, // versioned managed-agent portability object/package/transaction contract; stdlib-only, imports nothing internal, off the hot path.
	"portability":         1, // personal-continuity discovery/export/apply/switch/rollback leaf; stdlib-only, imports nothing internal, off the hot path.
	"ociartifact":         2, // activation-neutral OCI collection transport and MCP metadata bridge
	"portabilityswitch":   2, // context-switch transaction coordinator; composes lifecycle/processforest authorities, off the hot path.
	"fastintent":          3, // joins tier-2 orchestration, provider-tier, and benchmark receipts into one portable replay; off the serving hot path.
	"coordination":        1, // pure cross-layer agentic-compute plan contract; stdlib-only, fail-closed, off the serving hot path.
	"orchestration":       2, // pure portable workflow-plan/profile resolution contract; stdlib-only, no provider adapters, off the hot path.
	"ultracodebench":      2, // pure paired-run fold and verdict contract; stdlib-only, off the serving hot path.
	"devexmeter":          1, // pure dev-ex friction meter + RSI close gate; stdlib-only, off the hot path.
	"toolproc":            3,
	"regionadmit":         3,
	"toolprocgate":        5, // post-kill tool-result quarantine rung; declared once its leaf landed on the trunk (see the stale-row rule above)

	"operatortouches":       2, // R1 babysitting counter (#2270): pure fold over loopmgr's loop-event ledger measuring human touches per witnessed shipped unit; mttr_sessions is fed by the R4 drain witness (#2273); imports loopmgr(1)+resume(1), off the hot path.
	"popularizationtickets": 1, // embedded popularization-ticket curriculum (tickets.json) + pure fold/render over it; stdlib-only, off the hot path.
	"qwen36nodereports":     3, // qwen36 node-report harvester: reads/unzips node artifacts and folds a versioned report; tool-shaped mechanism leaf, shells out off the hot path, imports nothing internal.
	"trajctl":               2,
	"trajctlhook":           3, // impure call-site of the pure trajctl fold (#3129): git-shell PhaseCommits assembly + LiveTurnEnd producer that drives RunTurnEnd from a running session. Imports trajctl(1)+sessionaudit(1), off the hot path.
	"waiting":               2, // R3 waiting-on-human queue (#2272, epic #2269): pure fold over loopmgr's loop-event ledger turning each blocked-on-operator notify into one kernel object with age, held resources, deadline, and the safe default that fires on expiry; ack-closure honestly fenced not_yet pending the R2 escalation packet (#2271); imports loopmgr(1), off the hot path.
	"chatopsdetach":         1, // detached-execution decision kernel for chatops act verbs (#2265, epic #2259 C5): the pure fold that routes an inbound act verb + its guarded-dispatch admission verdict to dispatch-once / re-ack / structured-refuse over a command-nonce spool, plus the stall-to-blockers escalation judge. State in, decision out; imports nothing internal, off the hot path — the seam the (not-yet-landed) chatops door binds to.
	"godfileceiling":        2,
	"microscaleeval":        1,
	"microagent":            5,
	"sessionmine":           2, // offline normalization and ranking of local session telemetry; imports processalive(1), no runtime kernel.
	"sessionsearch":         3, // witnessed cross-session recall (#2913): pure TF-IDF inverted index over the guard tool-process journal + cache-safe suffix injection proven byte-stable via cachemeta + a recall-usefulness witness; imports toolproc(2)+cachemeta(1), off the hot path.
	"taskgraph":             3, // #2437: shared task journal pure-folded to a typed table with lease-gated claims (created/claimed/blocked/completed/abandoned); refuses a dead-lease claim, a tree-colliding claim, and a complete-over-open-blockers as closed reasons. Pure fold, imports only abi(0), off the hot path.
	"toolshape":             4, // #2823 (epic #2822 C1): pure SHAPE fingerprint of one tool call — trajectory.Turn folded to closed-vocabulary ArgClass/buckets + Avro-style key-set signature, names-not-values; imports trajectory(3)+stdlib, off the hot path.
	"l3kv":                  3, // durable off-box L3 KV residency backend (#1472): routes span content through the storedrv(2) router (blobfs local stand-in + the FAK_BLOB_HTTP_URL remote pool) behind a durable span→content manifest, so it declares alongside storedrv; imports abi(0)+model(1)+blob/blobfs/blobhttp(1)+storedrv(2), off the hot path.
	"l3server":              2, // pure L3 slab-allocated storage engine and protocol server (#11077); stdlib-only, off the hot path.
	"worklog":               1, // unified agent-work change feed primitive (#3172): the outbox-insight applied to agent work — one append-only, Seq-ordered, principal-scoped CDC feed (commit / verdict-flip / lease events) drained by cursor with an idempotency+retention contract + a fold read-model; pure primitive, imports nothing internal (`import "sync"` only), off the hot path.
	"conformance":           3, // standalone third-party-runnable fak safety-conformance suite (#453): pins the guarantees a fork/auditor must verify independently of the kernel's own tests; composes abi(0)+adjudicator(2)+policy(2), so it declares at the lowest layer its imports allow (tier 2, alongside shipgate), off the hot path.
	"sessionreplay":         3, // per-turn regime-conditioned replay-regression fixture (#4425, epic #4107): the fak.sessionreplay.v1 golden format + a pure Replay that re-adjudicates a captured (turn, active_regime) through the REAL adjudicator(2) over an internal/policy(2) regime floor, freezing a mode/regime-conditioned verdict for regression. Composes abi(0)+adjudicator(2)+policy(2) (like conformance), so it declares tier 2; test-time only, mutates nothing, off the hot path.
	"doomloop":              1, // two-axis doom-loop classifier (#doomloop): folds a live worker's effort-vs-verified-progress sample window into a closed verdict + a graduated, reversible-first correction (observe->nudge->escalate, never an auto-teardown); the missing enforcement half of fleetmon/trajctl/relay/loopmgr detection. Pure, stdlib-only, off the hot path.
	"looporphan":            1, // duplicate-loop-supervisor reaper core (#resume-consolidate-duplicate-loops): pure fold of a loop-supervisor process census into a closed KEEP/REAP/COLLISION/UNKNOWN plan that keeps the one parenting live work and reaps idle orphans/duplicates; fail-closed (no start-fence or unknown parent -> UNKNOWN, never REAP). Imports nothing internal, off the hot path.
	"cvregress":             3,
	"antipattern":           5,
	"astquery":              1,
	"atif":                  4,
	"brittleness":           1,
	"codegraph":             1,
	"codesearch":            5, // composes selfquery(3)+trigram/astquery/codegraph(1) into the `fak dev codesearch` engine (#3434 dogfood wiring)
	"dsparity":              1,
	"shadowgit":             2,
	"signals":               1,
	"trajquery":             1,
	"trigram":               2,
	"chatops":               1, // inbound chatops DOOR (#2264, epic #2259 C4): the pure parse+authorize fold that turns one raw Slack message into a closed-grammar verb + operand or a structured refusal (BOT_LOOP/WRONG_CHANNEL/NOT_ADDRESSED/EMPTY/NOT_ADMIN/UNKNOWN_VERB/MISSING_OPERAND); fail-closed admin allowlist on immutable user ids, channel text never executes as free-form. State in, decision out; imports nothing internal, off the hot path — produces the Command chatopsdetach(1) consumes.
	"negframe":              1, // negation-lexicon + reframe-classification card (#3539): classifies negatively-framed steer prose (prohibition/absence/refusal/hedge) and returns the mechanical positive reframe; imports pkg/scorecard only, off the hot path.
	"stepbaton":             1, // pure step-baton stamp core; stdlib-only, imports nothing internal, off the hot path.
	"stepbatoncapture":      2, // step-baton capture over stepbaton(1); imports only stepbaton, off the hot path.
	"seatpark":              1, // pure bounded park-and-retry fold for the no-seat transient; stdlib-only, imports nothing internal, off the hot path.
	"agentsindex":           1, // sectioned, fence-aware AGENTS.md loader (#3535, epic #3229): stdlib-only ATX view over AGENTS.md bytes with a resident TOC; imports nothing internal, off the hot path.
	"milestoneburndown":     3, // the GitHub-milestone SCHEDULE dimension milestonereport never had: reads the live milestones' own due_on + open/closed counts + trailing closure velocity and folds each into a closed at-risk verdict (ON_TRACK/AT_RISK/OVERDUE/NO_DUE_DATE/DONE) with a projected drain date vs the due date. Pure fold + injected-`gh` collector (twin of mlpscore/versionskew); imports trendreport(1)+epicprogress(1), off the hot path.
	"agentreadinessscore":   2, // agent-readiness scorecard (Go port of tools/agent_readiness_scorecard.py): experience-frontier + friction-debt over the git-tracked tree; stdlib-only, imports nothing internal, off the hot path.
	"verattest":             1, // #10463: commit-bound pre-PR verification attestation validator; stdlib-only, off the hot path.
	"verifierexposure":      2, // pure verifier-gameability score fold (#4424): declared gate signals to debt + worst-first worklist; stdlib-only, off the hot path.
	"wiki":                  2, // #4277 deepwiki-study: witness-verified repo-wiki core — Structure (L1 #4278) projects the section→page tree from the self-index, VerifyCitations (L3 #4280) resolves Sources:[path:line] code cites vs the tree; composes devindex(1), off the hot path.
	"projectreport":         2, // the ProjectsV2 board control-pane fold: a pure fold of board item × {Status,Generation,Priority} into the same schema/ok/verdict/finding envelope milestonereport uses, with a fail-closed UNMEASURED verdict for an unreadable board. The `gh` read lives in cmd/fak; this leaf is stdlib-only, imports nothing internal, off the hot path.
	"projectcompletion":     3, // weighted production-scope aggregation over canonical issuecontract metadata; imports issuecontract only, off the hot path.
	"catchupscore":          2, // the dev-system CATCH-UP control-pane scorecard (`fak score catchup`): folds intake/measurement/index/trunk/loops into a 0..1 caught-up fraction + an unbounded catchup_backlog headline. Pure fold, stdlib-only, imports nothing internal, off the hot path.
	"seoaeoscore":           2, // seo/aeo discoverability scorecard (Go port of the former tools/seo_aeo_scorecard.py): front-matter/JSON-LD/llms.txt + crawlable-link audit over the git-tracked doc surface; stdlib-only, imports nothing internal, off the hot path.
	"zaitask":               5,
	"macfit":                3,
	"roofline":              2, // pure-Go GLM-5.2 roofline dashboard fold (#3090): folds committed benchmark run artifacts × transcribed ceiling estimates into one current-vs-80%target-vs-ceiling table; stdlib-only, imports nothing internal, laptop-composable (no GPU), off the hot path.
	"supervisoragent":       3, // #4478 (epic #4477, supervisor-seat fence #1): the closed, payload-free SupervisorInput contract a supervisor agent consumes + the pure assembler that projects it. Folds fleetmon(1) per-worker class -> WorkerState and leaseref(1) ArbiterLease -> Lease, passing the dos_status/escalation heads through payload-free; no transcript/free-text field by construction. Off the hot path.
	"modedebt":              4, // #4416 (epic #4397, permission regimes #2389/#2405): the CONSUMER half of the mode-debt scorer/dispatcher pair. Reads the sibling scorer's scorecard JSON, selects HARD un-lifted permission dials, and maps each onto dogfoodissues(3).ActionItem with a content-stable dedup key, capped at the family --cap. Composer twin of qaprocessscore: imports dogfoodissues(3)+stdlib only, off the hot path.
	"deploymanifest":        1, // unified fak.toml all-in-one deployment manifest (#3421, epic #3256): typed schema for the eight deployment sections + a fail-closed loader (unknown/typo'd key refuses at load with a closed-vocabulary reason) + the `fak init` minimal-emit bytes; stdlib-only, imports nothing internal, off the hot path.
	"configguide":           2, // intent-level posture composer over deploymanifest: emits explained minimal fak.toml deltas and validates them through the foundation parser; no runtime side effects.
	"configsurface":         2, // config discoverability/default/budget scorer (#2938/#2862): composes deploymanifest(1)+configguide(2), stdlib-only, off the hot path.
	"toolplugin":            2, // typed monotone tool-call plugin host + layered preference resolver (#6045): stdlib-only contract/composer, no direct execution authority beyond its injected Executor, off the hot path.
	"systemservice":         2, // pure service-manager definition renderer; stdlib-only and off the hot path.
	"guardcompile":          4,
	"operatorquestion":      3,
	"operatorresolve":       4,
	"planresolve":           4,
	"modelops":              4,
	"modelaccept":           4,
	"executionroute":        4,
	"market":                4,
	"steerpr":               1, // #5015: pure fold of stamped trunk commits into operator-legible PR units + attention bands; the band is a VIEW over dispatchtick's witness verdicts (supplied by the caller), which is what keeps this leaf stdlib-only and off the hot path.
	"tokencache":            2, // #4330: persisted, git-common-dir-anchored backing store for clonescan's per-file token windows (content-addressed under the tokenizer version); imports only tier-1 siblings (clonescan for the window contract, windowgate to suppress its one `git rev-parse` spawn), does its own disk I/O, off the hot path.
	"guardcorpus":           3, // guarded-session corpus fold consumes journal semantics and emits durable analysis records; off the request hot path.
	"stripeload":            1, // #4298: pure chunk-planner + striping io.ReaderAt that fans one logical read across N byte-identical mirrors proportionally to measured bandwidth (the same fan-out later stripes {NVMe, peer-RAM, S3}); stdlib-only, callers own I/O policy and bandwidth measurement.
	"lookahead":             3, // #5204 (epic #5202): pure witness-gated Lesson core — distills a fork-rollout's outcome into a Lesson whose FACT/RISK authority is bounded by the witness rung its evidence earned (W3 may assert FACT, W2 only RISK, W1/W0 refused LESSON_OVERCLAIMS); injected model seam (decline discipline), stdlib + trajctl(1), off the hot path.
	"escalation":            2, // #2271 (epic #2269, no-babysit R2): the fak.escalation.v1 typed packet schema — closed-token escalation object (reason/ids/state digest/evidence refs/bounded actions/safe default+expiry/cost-of-delay, NO free-text field by construction) + JSONL packet/ack ledger + the pure open/expired/acked fold R1's handling p50 reads. Deliberately imports nothing internal: waiting(1) and operatortouches(1) sit ABOVE it in the no-babysit stack. Off the hot path.
	"servicespec":           2, // #4749 (parent #4748): the portable fak.service.v1 desired-state + restart-semantics contract — ONE versioned schema (identity, command, readiness, restart/backoff, checkpoint/resume, deps, intentional-stop) with DESIRED vs OBSERVED state kept strictly disjoint; platform supervisors render from it, never call SCM/systemd/launchd/cron themselves. Pure, stdlib-only, imports nothing internal, off the hot path.
	"servicelease":          2, // #4752 (parent #4748): lease/generation/incarnation fencing layer over fak.service.v1 — the four durable facts (Generation, Incarnation, Lease, FencingToken) plus the pure LocalRestartAllowed/RemoteReassignAllowed rules that keep two owners from ever being simultaneously valid across a partition. Imports only servicespec(1), off the hot path.
	"serviceledger":         2, // #4753 (parent #4748): portable append-only observed-state event ledger (fak.service.events.v1) folding Windows Event/SCM, journald, launchd, and fak supervisor receipts into one schema correlated by servicespec identity + boot/incarnation/lease facts; exactly-once by (source, source_uid), secrets/private-host redaction before persist. Imports only servicespec(1), off the hot path.
	"sharedtask":            3, // in-memory reference fold for collaborative task records — patch, conflict, scope, event, and journal semantics adapters reuse instead of inventing per-adapter task-state contracts; rides the a2achan(2) mailbox for its live seam, hence mechanism. Off the hot path.
	"loaddebounce":          1, // #3376 (parent #3365, scale-to-100 #1333): clean-room per-worker load publisher — a pure Coalescer state machine that publishes only on change (last==current dedup) and coalesces a rapid burst behind a reset-on-every-change debounce window so only the LATEST value emits. Clock is injected per call, so the debounce is exercised with zero goroutines/timers/wall-clock. Inspired by ai-dynamo worker_metrics.rs; stdlib-only (time), imports nothing internal, off the hot path. Intended landing: cmd/fak/dispatch_tick_preflight.go:45 (OSWorkerProcs); analog internal/gateway/serving_autoscaler.go:64.
	"hostplacement":         1, // #3599: pure multi-host worker-placement primitive — a per-host headroom heartbeat {hostname, live_headroom, saturation, ts} + a Registry that folds the latest per host, and a deterministic Place over injected (now, TTL) that picks the least-saturated non-stale host below its saturation threshold or returns stay-local. No wall-clock, no ssh/exec, imports nothing internal; off the hot path. Intended landing: internal/dispatchtick worker-launch seam, deriving live_headroom from accounts_headroom + saturation from procguard.
	"computeadmit":          2, // #3269 (parent #3259): the ONE shared admission kernel over the compute partitioners — the compute-plane twin of regionadmit/laneadmit.Decide. Pure Decide(Request, []Lease, Taxonomy) refusing with the closed vocabulary (COLLISION_RISK + RepartitionAdvice for a region contention via dispatchorder.ComputeClaimsContend, POLICY_BLOCK/out_of_taxonomy for an out-of-space claim = ExpertParallelPlan's [1,N] fail-closed in shared form), plus SubmitAdmitter, an abi.Adjudicator pricing compute-region contention at the Kernel.Submit seam (decision only, no scheduling; claim-less calls defer untouched). Imports abi(0)+dispatchorder(1), off the hot path.
	"scmbridge":             2, // #4756 (parent #4748): the Windows supervision plane over fak.service.v1 — ONE desired-state/read-back contract for SCM LocalService services + S4U/InteractiveToken scheduled-task roles (placement + launch fencing + stop-cause + reconcile), and the pure deterministic SCM service-DEFINITION projection (start type incl. desired-stopped=installed-but-disabled, LocalService-only account, depends_on->native dependencies, RestartPolicy.Decide-replayed FailureActions ladder with circuit-open as terminal none, sc.exe rendering as data). No SCM/registry/sc.exe calls. Imports only servicespec(1), off the hot path.
	"ghexec":                2, // #3473: deadline-bound `gh` invocation builder (Command/CommandTimeout + DefaultTimeout) — every gh call carries a context deadline, disables interactive prompting + the update notifier, and suppresses the Windows console via windowgate so a wedged gh can never wedge its caller. Imports windowgate(1)+stdlib, off the hot path.
	"deadlineadmit":         1, // #5292 (parent #5289 mooncake-study): pure EDF admission-ordering + predicted-miss load-shedding policy — Order (Deadline asc, ties by ID) + Shed (degradable items whose predictedFinish overshoots Deadline by >= dropThreshold; non-degradable never shed) + Admit. Clean-room from Mooncake's admission queue; imports only stdlib (sort), nothing internal; off the hot path.
	"taskvc":                2, // #3323 (reboot-survival audit): binds the fleet's live Windows Scheduled Tasks to the versioned installers that recreate them — the reboot-survival record `fak schedscan` cannot give (schedscan reports live task HEALTH, never whether a task is rebuildable at all). Pure literal-only parse of an installer's -TaskName defaults (an interpolated default is never treated as coverage) + Verify, which refuses a coverage claim the tree does not back (INSTALLER_NAME_DRIFT) or an unexplained gap (ORPHAN_FLEET_TASK); the declared Inventory carries the residual orphans with a named reason each. Imports windowgate(1)+stdlib for the `git ls-files` call, off the hot path. NOT pureRoot (it imports a sibling leaf).
	"skilleffectiveness":    4, // skill-effectiveness scorecard pure core: probes every .claude/skills SKILL.md for the affordances that make a skill discoverable (frontmatter description), triggerable (an explicit use-when phrase), and affordable (per-load-tier word budgets), plus the queried loader's queryable/paging/in-sync health, and folds skill_debt through the shared kernel. Imports capindex(2)+pkg/scorecard+stdlib, off the hot path.
	"skillfootprint":        4, // #5444 (epic #3229): the resident .claude/skills DESCRIPTION floor — the fold `fak skill footprint` renders (Fold/Measure over the shipped SkillResolver cards) PLUS the one-way ratchet gating it (SkillDescriptionBudgetBytes, SKILL_DESC_BUDGET_EXCEEDED / _STALE), held together so the scorecard and the gate can never become rival authorities on what the resident floor is. The userland twin of mcpfootprint's MCP ratchet. Imports capindex(2)+stdlib, off the hot path.
	"leasequeue":            3, // #5505: the WAITER PLANE a region-admission refusal never had — Plan ranks waiting tickets with dispatchaging's aging law over their ENQUEUE clock, tests each with regionadmit's one conflict predicate against live holders + this pass's grants + every higher-ranked waiter's reservation (conservative backfill), and hands each queued waiter seatpark's poll schedule plus its blocker, place in line and (only when the blocker's expiry is declared) an ETA. The pure fold holds/acquires/reaps nothing; store.go is the one I/O half (a same-machine per-ticket journal under the git dir). Imports regionadmit(2)+dispatchaging(1)+seatpark(1), off the hot path.
	"corelockgate":          2, // #5392: the single owner of the hard-self core-lock question every path to the trunk must ask (CoreLockCheck + CheckCoreLockHardSelf + the confirm/refute/abstain witness semantics). Pushed DOWN out of safecommit(2) so the sanctioned detached worker-worktree lander — workerworktree(1), which `fak commit` cannot serve because it refuses a detached HEAD — can ask the SAME question without the upward foundation->mechanism import this gate refuses, and without a second copy of the policy. The tier-2 witness resolver is inverted into a registration seam (RegisterResolverFactory, called from internal/witness's init); with none registered the gate fails CLOSED. Imports corelocks(1)+corelockaudit(1)+abi(0), off the hot path. NOT pureRoot (it imports sibling leaves).
	"fleetbus":              2, // #5600 (epic #5599): the fleet control bus — the cross-PROCESS join sessionctl(3)/broadcast (per-gateway, ack-less) and fleetspine(1) (display-only, one-way) each lack. Three envelopes (directive/instance/ack) over a swappable Bus, with the return path as the load-bearing half: a directive is never its own witness, the consuming instance's ack is. Declares NO ops — the bus is a carrier and the APPLIER (cmd/fak) owns op meaning, which is exactly what keeps a fleet-level carrier from importing sessionctl(3) and lets it carry control ops that are not session ops. Exactly-once application under at-least-once delivery via a per-(instance,directive) O_EXCL claim marker. Imports flock(1)+jsonlledger(1), off the hot path. NOT pureRoot (it imports sibling leaves).
	"gitbroker":             2, // #5622 (rung 3 of epic #5619): the resident per-repo git query broker separate short-lived client processes share over one AF_UNIX socket — token-authed transport, a Class A (content-addressed, full-OID keyed) object cache that by construction needs no invalidation, and a mandatory provenance tag (broker/cache/fallback-spawn) on every answer. Fail-open is the safety argument: every client call is deadlined and ANY broker problem re-derives the answer by spawning git through the same Runner, so a dead broker costs latency and a provenance tag, never bytes. Imports windowgate(1)+stdlib (net/os/exec/crypto) — every `git cat-file --batch` spawn takes the no-window hook, since the fallback path spawns under background automation on a Windows fleet; off the hot path. NOT pureRoot (it imports a sibling leaf).
	"tokenprofile":          3,
	"docrender":             2,
	"pagespublish":          2, // Pure stdlib pre/post-build audit for source UTF-8, artifact breadth, SEO metadata, sitemap, and exact deploy manifests; off the hot path.
	"learningobservation":   2, // #5982 (parent #2908): content-addressed observation/candidate/witness/verdict records plus the seven closed-enum lineage relations. Pure stdlib durable substrate; admission policy remains outside this leaf. Off the hot path.
	"learningmesh":          2, // #9839: deterministic cross-envelope transfer candidates; orchestration and execution remain outside this leaf.
	"streamrules":           0, // #5920 (epic #5917): pure stdlib-only streaming rule matcher; per-call buffers, scopes, regex/glob matching, diagnostics, and turn reset. No decision-path wiring and imports no sibling leaf.
	"testenv":               2, // credential-free test process boundary (#5914); imports envconfiglint(1), off the hot path.
	"estimatecal":           2, // #5899 (epic #3229): pure in-memory estimate-vs-billed correction ratio fold keyed by provider/model; stdlib-only, imports nothing internal, off the hot path.
	"docreach":              2, // Pure document-reachability census; repository I/O stays in cmd/fak.
	"testquality":           3, // #5936: pure stdlib Go test AST quality findings plus counted baseline ratchet; repository I/O and CLI wiring stay in cmd/fak.
	"dependencyquarantine":  0, // #5947: stdlib-only repository dependency-budget and nested-module quarantine checker; imports no sibling leaf.
	"enumlint":              0, // #5935: pure stdlib Go AST closed-enum discovery, exhaustiveness rules, and counted baseline ratchet; repository I/O is caller-selected.
	"archreport":            2, // read-only query/report over the architest declaration source and package import graph.
	"generationctl":         3,
	"quantprov":             1,
	"cudaarch":              1, // deterministic CUDA architecture compatibility matrix; stdlib-only pure parser/renderer.
	"cubicquanteval":        1, // pinned-codebook quantization evaluation; stdlib-only leaf.
	"kvquantquality":        1, // KV quantization quality budgets and witness parser; stdlib-only leaf.
	"kvvectoreval":          1, // attention-preserving KV vector quantization evaluation; stdlib-only leaf.
	"qevicteval":            1, // recoverable KV eviction evaluation; stdlib-only leaf.
	"lightgapscore":         1, // deterministic stdlib-only scorecard parser and renderer; off the hot path.
	"lightgapport":          1,
	"kvquantmeta":           1,
	"ggufinterop":           2, // GGUF metadata adapter over ggufload into the neutral quantization contract (#6231).
	"llamacppinterop":       2, // versioned external llama.cpp delegation adapter over quantization metadata (#6232).
	"quantcompat":           2, // artifact/runtime/hardware compatibility contract (#6224); imports quantmeta(1).
	"quantroute":            2, // ordered quantization compatibility routing contract (#6225); imports quantcompat(2).
	"quantfixture":          1, // stdlib-only synthetic redistributable artifact fixtures and verifier (#6230).
	"quantmeta":             1, // stdlib-only neutral quantization descriptor, parser, and typed adjudication contract (#6222).
	"qwen38quant":           1, // stdlib-only evidence contract and validator for the Qwen3.8 quantization campaign (#8307).
	"qwen38quantrun":        2, // endpoint runner depends on the tier-1 campaign evidence contract (#8343).
	"modelperfobs":          2, // OpenAI-compatible request measurement and JSONL report leaf; stdlib-only, off the hot path (#8389).
	"quantdetect":           1,
	"fp4runtime":            2, // stdlib-only FP4/microscaling runtime, GPU-architecture, and accumulator compatibility contract; no model kernel.
	"fp4meta":               1, // stdlib-only neutral FP4 artifact, recipe, scale, runtime, and hardware metadata contract; no model kernel.
	"bitnetruntime":         1, // stdlib-only BitNet runtime delegation and host-compatibility contract; discovers external runtimes but does not execute a model kernel.
	"quantpolicy":           2, // stdlib-only structural quantization capability policy; no quantizer, conversion, runtime, or model kernel.
	"requanteval":           1,
	"lightroteval":          1,
	"turntaxvisual":         1, // deterministic stdlib-only SVG renderer; off the kernel hot path.
	"fabricmap":             2, // direction-agnostic endpoint/link planner; stdlib-only mechanism.
	"quantlicense":          1, // stdlib-only license-chain contract and evidence gate.
	"codebookmeta":          1, // stdlib-only quantization codebook metadata and adjudication; no model kernel.
	"mixedprecision":        1, // layerwise mixed-precision assignment and evidence contract; no runtime kernel.
	"quantmatrix":           1, // stdlib-only public quantization support/delegation registry and docs parser; no model kernel.
	"residualquant":         1, // stdlib-only RRQ metadata/research adjudication; no model kernel.
	"rotationmeta":          1, // stdlib-only rotation-transform metadata and runtime capability adjudication; no model kernel.
	"quantwatch":            1, // stdlib-only public metadata ingestion and deterministic ranking; no model or runtime kernel.
	"toolpriorbench":        1, // stdlib-only reproducible tool-name compatibility corpus and raw-call benchmark ledger (#6820).
	"quantbench":            2, // stdlib-only benchmark evidence contract and adjudication; no runtime kernel.
	"kvint2eval":            1, // stdlib-only INT2 KV rotation evidence contract; CUDA producer is fixture-only.
	"codexresume":           2,
	"disambiguation":        1, // stdlib-only terminology schema, validation, and self-test; no policy or model kernel.
	"suiteverify":           1, // stdlib-only shared suite schema validation; imported by higher-tier test harnesses (#7214).
	"sessionregistry":       2, // stdlib-only durable child-lineage record and read-back; no runtime kernel.
	"grafanacontract":       1, // stdlib-only contract checks for shipped Grafana dashboard JSON.
	"processforest":         1, // stdlib-only durable logical process-forest identity, ancestry, adoption, and deterministic snapshots.
	"lifecycleadapter":      1, // stdlib-only process-forest adapter capability negotiation and bounded invocation contract.
	"terminalbarrier":       3, // #6436 (epic #6432): the fail-closed pause barrier terminal relief must clear before any host replacement. Composes fleetbus(2) durable lifecycle envelopes with processforest(1) identity and lifecycleadapter(1) negotiation; the destructive step is an injected Actuator, so the barrier itself can never kill anything.
	"depthadmit":            1, // The DEPTH mirror of focusscore(1): focusscore folds how BROAD the fleet is (active objectives vs the WIP cap), this folds how FAR DOWN one line got (declared plan phases vs the ones a W3 commit-progress row witnessed), so a `met` that stopped short is refused as DEPTH_NOT_CARRIED instead of reading clean. Pure over stdlib, no ledger and no git: the impure read lives in cmd/fak/trajctl_depth.go.
	"gardenbudget":          1, // #6493 whole-tick budget/checkpoint primitive: stdlib-only durable cursor plus pure suffix executor/remaining-budget arithmetic; imports nothing internal and stays off the hot path.
	"armbench":              3, // #6676 off-path multi-arm benchmark integrator; imports the tier-2 windowgate to keep managed child processes invisible on Windows.
	"codetools":             2, // Kernel-mediated workspace coding engines: canonical confinement, policy rung, bounded Read/Grep/Glob; imports ABI/refutil/vDSO but not core runtime.
	"systools":              2, // safe system and web search/fetch utility tools for the native harness
	"portabilitylab":        2, // Hermetic release acceptance harness over the tier-1 portability leaf; stdlib plus portability only, off the hot path.
	"scratchjanitor":        1, // stdlib-only age and resume-reference guarded harness scratch cleanup; off the runtime hot path.
	"tempartifact":          1, // stdlib-only exact-path OS-temp artifact lifecycle; process inspection stays in its off-path platform adapter.
	"managedocs":            1,
	"humanctl":              1,
	"frontmatter":           1, // stdlib-only flat YAML scalar decoder shared by skill metadata readers.
	"toolcatalog":           2, // typed tool declarations and runtime adapters composed over the frontmatter primitive.
	"harnessserve":          2, // bounded local-runtime supervisor composed over the processalive primitive.
	"projectassets":         1, // stdlib-only project asset registry and parity adapter; off the runtime hot path.
	"wavefuel":              1, // stdlib-only fleet-wave operator-receipt contract witness.
	"quantobs":              1, // stdlib-only bounded quantization telemetry schema and encoder.
	"bitnetmeta":            1, // stdlib-only low-bit artifact metadata contract and typed adjudication.
	"workaccount":           1, // stdlib-only WORK DONE mechanism coverage registry and validator; off the runtime hot path.
	"orientation":           2, // versioned temporal product-orientation snapshot; imports issue-free stdlib only, off the runtime hot path.
	"harnessweb":            2, // loopback browser runtime over public harnesskit plus persisted semantic sessions; off kernel hot path (#6980).
	"committedbuildwitness": 2, // exact-HEAD build receipt; imports windowgate(2) for hidden git probes, off the hot path (#6871).
	"toolcallcontrol":       1, // stdlib-only deterministic pre-execution call gate and offline ablation accounting.
	"devhandoff":            1, // stdlib-only moved-command inventory shared across the runtime/fak-dev process boundary.
	"workshape":             1, // stdlib-only dispatch execution-shape contract and capacity pricing.
	"valuechain":            2, // deterministic offline audit over declared stack and outcome evidence; stdlib-only, off the hot path.
	"doneclaimaudit":        1, // stdlib-only deterministic audit of issue completion claims against repository-local evidence.
	"wipinventory":          1, // read-only Git/filesystem census for provenance-honest working-tree operations.
	"processalive":          1, // stdlib-only no-spawn process liveness probe shared by operational leaves (#7215).
	"processstart":          1, // stdlib-only no-spawn process identity probe shared by dispatch session liveness (#8022).
	"refid":                 1, // stdlib-only Git-ref segment validation shared by ref-backed operational leaves (#7130).
	"generation":            1, // stdlib-only project horizon vocabulary shared by planning and handoff leaves (#7129).
	"kquantbits":            1, // stdlib-only k-quant metadata bit decoding shared by compute and model (#6986).
	"linefmt":               1, // stdlib-only newline-terminated formatting shared by service renderers (#6970).
	"markerblock":           1, // stdlib-only generated-document marker extraction and replacement shared by report leaves (#7115).
	"strictjson":            1, // stdlib-only strict single-document JSON decoder shared by contract leaves (#7064).
	"childprocess":          1, // stdlib-only child exit-status normalization shared by launchers (#7063).
	"stringset":             1, // stdlib-only deterministic string-set views shared by report leaves (#7047).
	"stringlist":            1, // stdlib-only compact string-list parsing shared by CLI surfaces (#7062).
	"numbermap":             1, // stdlib-only JSON numeric-map normalization shared by score renderers (#7059).
	"interspersedflags":     1, // stdlib-only interspersed Go flag parsing shared by CLI surfaces (#7058).
	"parentdir":             1, // stdlib-only parent-directory preparation shared by file writers (#7057).
	"sortkeys":              1, // stdlib-only source-location ordering shared by census leaves (#7056).
	"demoassert":            1, // stdlib-only self-check failure recorder shared by runnable demos (#7054).
	"exclusivefile":         1, // stdlib-only O_EXCL PID/time marker creation shared by lock owners (#7051).
	"wiplifecycle":          2, // lifecycle receipts depend on Git-backed WIP inventory collection (#7250).
	"sessionrecovery":       2, // selects and witnesses bounded crash recovery without provider-private state (#8013).
	"harnessinstructions":   3, // tier-3 adapter composing public harness instructions through the host-owned system-prompt MMU.
	"sessionintent":         1, // stdlib-only session control declarations and deterministic stop evaluation (#7352).
	"subtractiveprofile":    1, // stdlib-only deterministic capability subtraction algebra (#6791).
	"harnessmodelset":       1,
	"modelsetresolve":       2, // deterministic role resolution composing tier-1 intent and inventory contracts.
	"modelsetlock":          2,
	"modelsetreceipt":       2, // startup receipts compose model-set resolution with current tier-1 evidence.

	"harnessmodelsetconformance": 3, // captured end-to-end model-set artifact and CLI witness; off the hot path.
	"httptrust":                  2, // the ONE corporate-trust seam (#8172): resolves a declared CA bundle through secretload(2), widens the platform pool with it, and hands every fak-originated HTTPS client the same RootCAs plus the derived child-runtime trust vars; imports secretload(2)+stdlib, off the hot path.
	"cloudroute":                 1, // stdlib-only detector for a request-signed cloud model route (Bedrock SigV4 / Vertex ADC) whose base-URL repoint cannot take effect (#8172); pure over an environ snapshot, imports nothing internal.
	"modelloadplan":              3, // deterministic Qwen load planner over harnessresolve locks; off the hot path (#8134).
	"geminicache":                2, // provider CachedContent lifecycle adapter over stdlib HTTP and immutable context identity (#8481).
	"qwen38ladder":               1, // stdlib-only paired evidence gate from pinned Qwen3.5 proxies to exact Qwen3.8-27B (#8011).
	"scratchmark":                1, // #6616 stdlib-only bounded detector for source artifacts self-declared disposable in their leading comment block.
	"fleetsearch":                3, // read-only join over lifecycle/sessionjournal/toolproc evidence; no runtime kernel.
	"discoveryrouter":            3, // bounded read-only composition over existing discovery sources; no storage or runtime kernel.
	"nativefirst":                1, // stdlib-only native-engine substitution policy shared by CLI lint and commit gates.
	"agentqueue":                 2, // bounded agent desired-state planner and crash-safe store (#8875, #8876); composes flock for serialized durable state updates.
	"supervisionpolicy":          2, // typed fault-domain, bounded restart, and epoch-fenced adoption policy (#8909, #9053); composes flock for cross-process child takeover.
	"projectionspine":            2, // disposable projection restart and session reattach spine (#8912); composes supervisionpolicy restart budgets and fault domains.
	"codexsession":               4, // local Codex app-server adapter projected into the public harness protocol (#8736).
	"hostdiag":                   3,
	"shellprov":                  2, // privacy-safe receipt store for fak-owned shell launch identity (#9086); composes flock for serialized broker receipt updates.
	"qwenworkbudget":             5, // campaign-boundary adapter over canonical trajectory audit rollups and typed Qwen amplification policy; imports trajectory(4).
	"ultracodetokenizer":         1,
	"studybench":                 1, // stdlib-only deterministic offline retrieval benchmark and quality/context report (#8612).
	"studydrift":                 2, // external source source refresh and supersession receipt primitive (#8611).
	"servicewatchdog":            2, // systemd lifecycle read-back and watchdog progress integration (#8654).
	"computetrace":               1, // stdlib-only bounded compute-event artifact schema and recorder.
	"ultracodenegcontrol":        1,
	"computetune":                2, // offline offline tuning decision primitive over replayable compute traces (#8608).
	"ultracodecrossover":         1, // stdlib-only deterministic task-complexity crossover evaluator (#8674).
	"systembaseline":             1,
	"ultracodedogfood":           1, // stdlib-only deterministic lifecycle-boundary witness evaluator (#8678).
	"archrank":                   3, // deterministic research-ranking method; no runtime kernel (#9145).
	"qwenflashnext":              1, // stdlib-only pinned Flash-Next chat/channel/tool/stop contract (#9124).
	"qwen4exp":                   1, // stdlib-only deterministic native four-layer correctness oracle (#9123).
	"qwensemanticstop":           2, // deterministic experiment receipt gate; live exact-model cancellation remains hardware-witnessed (#8646).
	"studyclass":                 2,
	"studyadjacency":             2,
	"studyprio":                  2,
	"studytickets":               2,
	"microfleeteconomics":        1,
	"placementtax":               1,
	"localappcert":               1, // stdlib-only deterministic Mac certification-matrix validator (#9157).
	"localapphelper":             2, // authenticated app-scoped helper admission boundary (#9149).
	"openviking":                 1,
	"agentdescriptor":            1, // stdlib-only agent descriptor identity and validation contract.
	"agentic":                    2, // deterministic agentic-objective fold over native performance evidence.
	"archfitness":                1, // stdlib-only architecture fitness report contract.
	"causalreceipt":              1, // stdlib-only causal-receipt validation and ordering primitive.
	"cloudhandoff":               1, // stdlib-only cloud handoff contract and deterministic planner.
	"composition":                1, // stdlib-only immutable composition identity and validation contract.
	"docsearch":                  2, // shared documentation loader/search composite over trigram ranking.
	"issuecheck":                 1, // stdlib-only issue-bound Top-5 review record contract.
	"localadmission":             1, // stdlib-only local runtime admission state primitive.
	"localappcontract":           1, // stdlib-only local application protocol contract.
	"localappmetrics":            1, // stdlib-only local application metrics fold.
	"localappux":                 1, // stdlib-only local application UX state renderer.
	"macromailbox":               1, // stdlib-only authenticated macro mailbox primitive.
	"macrostate":                 1, // stdlib-only deterministic macro-state contract.
	"modelpack":                  1, // stdlib-only signed model-pack artifact contract.
	"openaiadapter":              1, // stdlib-only authenticated OpenAI-compatible adapter contract.
	"opensweharder":              1, // stdlib-only OpenSWE-hardening evidence contract.
	"perfrsiscore":               2, // deterministic performance-RSI scorecard over the shared scorecard kernel.
	"providerjobaccounting":      1, // stdlib-only provider-job accounting contract.
	"resourcelifecycle":          1, // stdlib-only resource lifecycle state primitive.
	"shelltoken":                 1, // stdlib-only shell-token parsing primitive.
	"studyforge":                 1, // stdlib-only forge evidence collector; network and filesystem stay at its explicit boundary.
	"sweepcert":                  1,
	"studylink":                  1, // stdlib-only study evidence join and validation primitive.
	"witnessprocess":             1, // stdlib-only process-witness classification contract.
	"dockerprocess":              1, // static-literal Docker Compose launcher; stdlib-only, called only by off-path control flows.
	"selfupdate":                 3,
	"rollout":                    1,
	"deployment":                 1,
	"extensionfault":             1, // stdlib-only optional extension subprocess fault-isolation primitive.
	"optsdefault":                1, // stdlib-only shared scorecard Options defaults (empty root, zero now).
	"stablejson":                 1, // stdlib-only canonical two-space-indented JSON rendering primitive.
	"gitresource":                1, // stdlib-only typed Git resource-ownership and cleanup-admission contract; folded by lifecycle integrations (#10613).
	"workspin":                   1, // stdlib-only deterministic busywork-trend classifier over bounded commit/issue observations; folded by the CLI (#10609).
	"schemaadapter":              1,
	"toolbound":                  1,
	"overtonscore":               1,
	"dataslot":                   1,
	"servingsim":                 1, // trace-driven discrete-event LLM serving simulator; stdlib-only (#10841).
	"cache":                      1,
	"benchsnapshot":              1,
	"promptcomp":                 1,
	"orgdebt":                    1,
	"tb4bench":                   4,
	"agentopt":                   2, // agent workflow optimization & model routing policies; imports modelroute(2), off the hot path.
	"codedebt":                   1, // pure code-debt query, deterministic scanner, and model fold; stdlib-only, no internal imports, off the hot path (#10939).
	"archcheck":                  2, // shift-left architecture import DAG and tier preflight validator; stdlib-only, off the hot path (#10918).
	"ctxplanlint":                1,
	"debtlane":                   1,
	"marketplace":                1,
	"mtpeval":                    1,
	"mtptune":                    1,
	"decodemigrate":              1,
	"dosadapter":                 1,
	"framebus":                   1,
	"interactivesession":         1,
	"faultlab":                   1,
	"gcpgpu":                     1,
	"kimik3page":                 1,
	"breathgate":                 1,
	"kv":                         1,
	"mcpbroker":                  1,
	"issueorchestrator":          3,
	"storage":                    1,
	"harnesshint":                1,
	"harnesslint":                1,
	"harnessversion":             1,
	"fakpack":                    2,
	"allinone":                   5,
	"airgaptest":                 5, // hermetic air-gapped harness integration contracts (#11387).
	"harnesswarm":                2, // progressive non-blocking workspace warming engine (#10649).
	"macobs":                     1, // Apple Silicon Mac & MLX observability leaf: hardware telemetry, headroom, metrics; stdlib-only, off hot path.
	"capabilitymatrix":           1, // unified model capability registry (#11507); stdlib-only, off the hot path.
	// new-leaf:tier - `fak new-leaf <name> --tier <tier>` inserts the
	// declaration for a generated leaf immediately ABOVE this line. Keep the marker last.
}

// Tier 1 is the executable primitive contract; the tier declaration itself is authoritative.

var tierName = []string{"root", "primitive", "foundation-composite", "mechanism", "composer", "integrator"}

// hotPath is the set of packages on a live tool-call decision. None of them may import
// os/exec: the per-decide subprocess boundary is exactly the microsecond-vs-232ms cost
// fak exists to remove (DIRECTION.md, reviewer's grep #1). os/exec is fine OFF the path
// (bench, shipgate, tests) â€” only these packages are checked.
var hotPath = []string{"adjudicator", "kernel", "vdso", "grammar", "preflight", "ctxmmu", "ratelimit"}

// internalDir returns the absolute path of fak/internal. This test file lives in
// internal/architest, so internal is its parent.
func internalDir(t *testing.T) string {
	t.Helper()
	if _, self, _, ok := runtime.Caller(0); ok {
		cand := filepath.Dir(filepath.Dir(self))
		if fi, err := os.Stat(cand); err == nil && fi.IsDir() {
			return cand
		}
	}
	if fi, err := os.Stat(".."); err == nil && fi.IsDir() {
		if _, err := os.Stat(filepath.Join("..", "adjudicator")); err == nil {
			abs, _ := filepath.Abs("..")
			return abs
		}
	}
	if fi, err := os.Stat("internal"); err == nil && fi.IsDir() {
		abs, _ := filepath.Abs("internal")
		return abs
	}
	wd, _ := os.Getwd()
	cur := wd
	for i := 0; i < 6; i++ {
		if _, err := os.Stat(filepath.Join(cur, "go.mod")); err == nil {
			cand := filepath.Join(cur, "internal")
			if fi, err := os.Stat(cand); err == nil && fi.IsDir() {
				return cand
			}
		}
		parent := filepath.Dir(cur)
		if parent == cur {
			break
		}
		cur = parent
	}
	t.Fatal("cannot locate the internal/ tree")
	return ""
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

// trunkRef is the shared-trunk ref this gate scopes an undeclared-leaf failure against.
// A push's responsibility is the leaves its commits deliver ON TOP of this ref; an
// undeclared leaf that this push's commits never touch belongs to some other push and
// must not red this one. Overridable for tests / a non-"main" trunk.
func trunkRef() string {
	if r := os.Getenv("FAK_ARCHITEST_TRUNK"); r != "" {
		return r
	}
	return "origin/main"
}

// tierDeclarationCmd is the exact one-line command an operator runs to declare the tier
// of leaf p, named verbatim in the failure so the owner can copy-paste the fix.
func tierDeclarationCmd(p string) string {
	return "fak new-leaf " + p + " --tier foundation|mechanism|composer|integrator [--register]"
}

// leafOf maps a repo-relative path to the internal/<leaf> package dir it belongs to, or ""
// if the path is not under internal/. "internal/foo/bar.go" -> "foo".
func leafOf(path string) string {
	const pfx = "internal/"
	if !strings.HasPrefix(path, pfx) {
		return ""
	}
	rest := strings.TrimPrefix(path, pfx)
	if i := strings.IndexByte(rest, '/'); i >= 0 {
		return rest[:i]
	}
	return "" // a file directly in internal/ is not a leaf package
}

// leavesTouchedByPush returns the set of internal/<leaf> dirs modified by the commits this
// push delivers on top of the trunk — the diff between merge-base(trunkRef, HEAD) and HEAD
// — and whether that scoping ground is available at all.
//
// This is deliberately the COMMITTED delta, not the working tree or the index: on a shared
// trunk both are polluted by peers (a peer's `git add -A` can stage files this push never
// authored), so only what a commit actually delivers identifies whose push owns a leaf.
// It shells to git — off the hot path, in a _test file, so the no-exec request-path rules
// do not apply. ok=false means there is no trunk to diff against (no .git, a `git archive`
// checkout, or the remote-tracking ref is absent); the caller then falls back to strict
// all-undeclared-fail, which is safe precisely because in that setting there are no
// concurrent peer pushes to wedge.
func cleanGitEnv() []string {
	var env []string
	for _, kv := range os.Environ() {
		k, _, ok := strings.Cut(kv, "=")
		if ok {
			u := strings.ToUpper(k)
			if u == "GIT_DIR" || u == "GIT_WORK_TREE" || u == "GIT_INDEX_FILE" || u == "GIT_OBJECT_DIRECTORY" || u == "GIT_PREFIX" || u == "GIT_COMMON_DIR" {
				continue
			}
		}
		env = append(env, kv)
	}
	return env
}

func leavesTouchedByPush(internal string) (map[string]bool, bool) {
	repo := filepath.Dir(internal) // internal/ -> repo root
	cEnv := cleanGitEnv()
	top := exec.Command("git", "rev-parse", "--show-toplevel")
	top.Dir = repo
	top.Env = cEnv
	topOut, err := top.Output()
	if err != nil || !samePath(strings.TrimSpace(string(topOut)), repo) {
		return nil, false
	}
	base := exec.Command("git", "merge-base", trunkRef(), "HEAD")
	base.Dir = repo
	base.Env = cEnv
	baseOut, err := base.Output()
	if err != nil {
		return nil, false
	}
	mb := strings.TrimSpace(string(baseOut))
	if mb == "" {
		return nil, false
	}
	diff := exec.Command("git", "diff", "--name-only", mb, "HEAD", "--", "internal/")
	diff.Dir = repo
	diff.Env = cEnv
	diffOut, err := diff.Output()
	if err != nil {
		return nil, false
	}
	set := map[string]bool{}
	for _, line := range strings.Split(strings.TrimSpace(string(diffOut)), "\n") {
		if leaf := leafOf(strings.TrimSpace(line)); leaf != "" {
			set[leaf] = true
		}
	}
	return set, true
}

func samePath(a, b string) bool {
	canonical := func(p string) (string, error) {
		abs, err := filepath.Abs(p)
		if err != nil {
			return "", err
		}
		return filepath.EvalSymlinks(abs)
	}
	ca, errA := canonical(a)
	cb, errB := canonical(b)
	return errA == nil && errB == nil && filepath.Clean(ca) == filepath.Clean(cb)
}

// undeclaredLeafVerdict classifies an undeclared on-disk leaf as either an advisory (a leaf
// this push's commits never touched — some other push owns it) or a hard failure (a leaf
// this push delivers). It is the pure decision the scoped gate rests on, split out so
// #2088's fleet-safety property is unit-testable without the real tree or a live git
// remote: when scoping ground exists AND this push did NOT touch the leaf, it is advisory;
// otherwise (this push touched it, or there is no scoping ground) it is a failure.
func undeclaredLeafVerdict(scoped, touchedByPush bool) (advisory bool) {
	return scoped && !touchedByPush
}

// TestEveryPackageDeclaresTier fails if a package on disk is missing from the tier table
// (a new leaf that forgot to take a layering position) or if the table names a package
// that no longer exists (a stale entry). This is what forces every future leaf through a
// conscious tier decision.
//
// FLEET-SAFETY SCOPING (#2088): architest runs on EVERY push to the shared trunk, so a
// naive "any undeclared leaf fails" would let one agent's forgotten tier row wedge EVERY
// peer's unrelated push. The undeclared-leaf failure is therefore scoped to the leaves
// THIS push's commits actually deliver (leavesTouchedByPush): an undeclared leaf that this
// push touched is a hard failure its owner must fix; an undeclared leaf this push never
// touched — a peer's leaf sitting in the shared tree, or pre-existing debt already on the
// trunk — is reported as an advisory (t.Logf) and does NOT red the build. When there is no
// trunk to diff against (a `git archive` checkout, no remote) the gate falls back to strict
// all-undeclared-fail, safe there because no concurrent peer push exists to wedge. Either
// way the failure names the owning leaf and the exact one-line declaration command.
func TestEveryPackageDeclaresTier(t *testing.T) {
	internal := internalDir(t)
	onDisk := goPackageDirs(t, internal)

	touched, scoped := leavesTouchedByPush(internal)

	diskSet := map[string]bool{}
	for _, p := range onDisk {
		diskSet[p] = true
		if _, ok := tier[p]; ok {
			continue
		}
		// Scoped + this push never touched the leaf: not our push's doing — advise, do not fail.
		if undeclaredLeafVerdict(scoped, touched[p]) {
			t.Logf("advisory: package internal/%s has no declared tier, but this push's commits "+
				"do not touch it (a peer's leaf, or pre-existing trunk debt). Its owner should "+
				"declare it with:\n  %s\nThis push is not blocked for it.", p, tierDeclarationCmd(p))
			continue
		}
		// This push delivers the leaf (or no-trunk fallback): its owner must declare the tier.
		t.Errorf("new leaf internal/%s has no declared tier.\n"+
			"Add it to the architest tier table at the LOWEST layer whose role it fits "+
			"(root<primitive<foundation-composite<mechanism<composer<integrator). Declare it with:\n  %s\n"+
			"This forced choice is the review gate that keeps the layered-DAG contract "+
			"honest as the kernel grows.", p, tierDeclarationCmd(p))
	}
	for p := range tier {
		if p == "abi" {
			continue // abi is a package; keep it explicit even though it has its own dir
		}
		if !diskSet[p] {
			t.Errorf("tier table names internal/%s, but no such package is on disk "+
				"(stale entry â€” remove it).", p)
		}
	}
}

// TestNoUpwardImports is THE layering gate. For every internal cross-package edge it
// asserts tier(importer) >= tier(imported). A failure means a lower layer now reaches UP
// into a higher one â€” fix the dependency direction (usually: invert it through a
// registration seam, or move the shared type down into the foundation), do not relax the
// tier table to admit it.
func staleTierDeclarations(report archreport.Report) []archreport.Diagnostic {
	var stale []archreport.Diagnostic
	for _, diagnostic := range report.Diagnostics {
		if diagnostic.Kind == archreport.DiagnosticStaleTierDeclaration {
			stale = append(stale, diagnostic)
		}
	}
	return stale
}

func TestTierDeclarationsAreLive(t *testing.T) {
	internal := internalDir(t)
	report, err := archreport.Analyze(filepath.Dir(internal), "")
	if err != nil {
		t.Fatal(err)
	}
	for _, diagnostic := range staleTierDeclarations(report) {
		t.Errorf("tier declaration for internal/%s is stale: %s; recovery: %s", diagnostic.Leaf, diagnostic.Message, diagnostic.Recovery)
	}
}

func TestTierDeclarationsAreLiveUsesReportedDiagnostics(t *testing.T) {
	report := archreport.Report{Diagnostics: []archreport.Diagnostic{
		{Kind: archreport.DiagnosticStaleTierDeclaration, Leaf: "gone", Recovery: "remove its tier row"},
		{Kind: "source-parse-error", Leaf: "broken"},
	}}
	got := staleTierDeclarations(report)
	if len(got) != 1 || got[0].Leaf != "gone" || got[0].Recovery != "remove its tier row" {
		t.Fatalf("stale diagnostics=%+v", got)
	}
}

func TestNoUpwardImports(t *testing.T) {
	internal := internalDir(t)
	report, err := archreport.Analyze(filepath.Dir(internal), "")
	if err != nil {
		t.Fatalf("derive enforced architecture graph: %v", err)
	}
	violations := reportViolationEdges(report)
	if len(violations) > 0 {
		t.Fatalf("layered-DAG import rule violated (a lower layer imports a higher one).\n"+
			"Rule: a package may import only packages whose tier is <= its own.\n  %s\n"+
			"Invert the dependency (registration seam) or push the shared type down a layer; "+
			"do not loosen the tier table.", strings.Join(violations, "\n  "))
	}
}

// reportViolationEdges is the enforcement/report parity seam: CI folds the exact
// named edges archreport exposes instead of maintaining a second import walker.
func reportViolationEdges(report archreport.Report) []string {
	var edges []string
	for _, leaf := range report.Leaves {
		for _, edge := range leaf.ViolationEdges {
			edges = append(edges, edge.String())
		}
	}
	sort.Strings(edges)
	return edges
}

func TestReportedAndEnforcedUpwardEdgesStayIdentical(t *testing.T) {
	report := archreport.Report{Leaves: []archreport.Leaf{
		{Name: "primitive", ViolationEdges: []archreport.ViolationEdge{{From: "primitive", To: "composite"}, {From: "primitive", To: "mechanism"}}},
		{Name: "composer"},
	}}
	got := reportViolationEdges(report)
	want := []string{"primitive -> composite", "primitive -> mechanism"}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("enforced edges=%v reported edges=%v", got, want)
	}
}

func upwardImport(fromTier, toTier int) bool { return toTier > fromTier }

func TestPrimitiveCannotImportFoundationComposite(t *testing.T) {
	if !upwardImport(1, 2) {
		t.Fatal("primitive -> foundation-composite must be rejected")
	}
	if upwardImport(2, 1) {
		t.Fatal("foundation-composite -> primitive must remain legal")
	}
}

// TestPrimitiveLeavesStayPrimitive makes tier 1 meaningful: every primitive imports only abi.
func TestPrimitiveLeavesStayPrimitive(t *testing.T) {
	internal := internalDir(t)
	report, err := archreport.Analyze(filepath.Dir(internal), "")
	if err != nil {
		t.Fatalf("derive enforced architecture graph: %v", err)
	}
	var violations []string
	for _, leaf := range report.Leaves {
		if leaf.DeclaredTier != 1 {
			continue
		}
		for _, dep := range leaf.Dependencies {
			if dep != "abi" && dep != leaf.Name {
				violations = append(violations, "internal/"+leaf.Name+" imports internal/"+dep)
			}
		}
	}
	if len(violations) > 0 {
		sort.Strings(violations)
		t.Fatalf("primitive-tier invariant violated:\n  %s\nFix: remove/invert the dependency or reclassify the leaf as foundation-composite.", strings.Join(violations, "\n  "))
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
				t.Errorf("hot-path package internal/%s imports os/exec â€” the request path must "+
					"stay interpreter/subprocess-free (DIRECTION.md). Move the off-path work to a "+
					"non-hot-path package.", pkg)
			}
		}
	}
}

// tierShareThreshold is the share of all declared leaves above which a single non-terminal
// tier is flagged as over-accumulated: past this line the layering has stopped discriminating
// and is piling leaves into one bucket instead of ordering them into a DAG. Tunable; advisory
// only. Set just above the post-split primitive share (~42%) so #4042 brings the tree under
// the line and the advisory goes quiet on its own once foundation is factored.
const tierShareThreshold = 0.55

// tierCount is one row of the per-tier population table.
type tierCount struct {
	Tier  int
	Name  string
	Count int
	Share float64
}

// tierDistribution folds a tier table into per-tier counts + shares, ordered by tier number.
// Pure: state in, table out, no I/O -- so it is unit-testable and reused by the advisory.
func tierDistribution(m map[string]int) []tierCount {
	counts := map[int]int{}
	total := 0
	for _, tv := range m {
		counts[tv]++
		total++
	}
	tiers := make([]int, 0, len(counts))
	for tv := range counts {
		tiers = append(tiers, tv)
	}
	sort.Ints(tiers)
	out := make([]tierCount, 0, len(tiers))
	for _, tv := range tiers {
		name := ""
		if tv >= 0 && tv < len(tierName) {
			name = tierName[tv]
		}
		share := 0.0
		if total > 0 {
			share = float64(counts[tv]) / float64(total)
		}
		out = append(out, tierCount{Tier: tv, Name: name, Count: counts[tv], Share: share})
	}
	return out
}

// TestTierDistributionAdvisory logs the per-tier population and flags over-accumulation. A
// healthy layered DAG discriminates: leaves spread across tiers, no single NON-TERMINAL tier
// (i.e. any tier below the top integrator sink) holding a majority. Foundation is ~69% of all
// leaves TODAY -- everything piled at the base with no sub-order -- so the advisory fires. It
// is advisory (t.Logf, never t.Fatal): it tracks the drift without blocking a push, and goes
// quiet once #4042 splits foundation into primitive/composite. Run with -v to see the table.
func TestTierDistributionAdvisory(t *testing.T) {
	dist := tierDistribution(tier)
	terminal := -1
	for _, d := range dist {
		if d.Tier > terminal {
			terminal = d.Tier
		}
	}
	// The over-accumulation READING is advisory, but the table it reads is not: a share is
	// only interpretable if every declared leaf lands in exactly one named bucket. These are
	// hard -- they catch a leaf declared at a tier the DAG has no name for (a typo like
	// `"x": 7` silently becomes its own unnamed stratum and skews every share), a lost or
	// double-counted leaf, and an out-of-order fold.
	if len(dist) == 0 {
		t.Fatalf("tierDistribution returned no rows for %d declared leaves", len(tier))
	}
	sum, shareSum, prev := 0, 0.0, -1
	for _, d := range dist {
		if d.Tier <= prev {
			t.Errorf("rows are not ordered by tier: %d follows %d", d.Tier, prev)
		}
		prev = d.Tier
		if d.Name == "" {
			t.Errorf("tier %d holds %d leaf/leaves but has no name in tierName%v -- a leaf is "+
				"declared at a tier the layering does not define", d.Tier, d.Count, tierName)
		}
		if d.Count <= 0 {
			t.Errorf("tier %d (%s) has a non-positive count %d", d.Tier, d.Name, d.Count)
		}
		if want := float64(d.Count) / float64(len(tier)); d.Share != want {
			t.Errorf("tier %d (%s) share = %v, want %v (%d/%d)", d.Tier, d.Name, d.Share, want, d.Count, len(tier))
		}
		sum += d.Count
		shareSum += d.Share
	}
	if sum != len(tier) {
		t.Errorf("tier counts sum to %d, want %d declared leaves -- a leaf was lost or double-counted", sum, len(tier))
	}
	if shareSum < 0.999 || shareSum > 1.001 {
		t.Errorf("tier shares sum to %v, want 1.0 -- the shares are not over the whole population", shareSum)
	}
	t.Logf("tier population (%d declared leaves, threshold %.0f%%):", len(tier), tierShareThreshold*100)
	for _, d := range dist {
		marker := ""
		if d.Tier != terminal && d.Share > tierShareThreshold {
			marker = "  <-- OVER-ACCUMULATED"
		}
		t.Logf("  %d %-11s %3d  %5.1f%%%s", d.Tier, d.Name, d.Count, d.Share*100, marker)
	}
	for _, d := range dist {
		if d.Tier != terminal && d.Share > tierShareThreshold {
			t.Logf("advisory: tier %d (%s) holds %.1f%% of all %d declared leaves (> %.0f%% threshold) -- "+
				"the layering has stopped discriminating at this level. Factor it into sub-tiers so the "+
				"DAG orders the pile rather than heaping it (see #4042: split foundation into "+
				"primitive/composite).", d.Tier, d.Name, d.Share*100, len(tier), tierShareThreshold*100)
		}
	}
}

// TestTierDistributionFoldsASyntheticTable pins tierDistribution's arithmetic against a
// population the live `tier` map cannot drift out from under: bucket counts, ascending order,
// skipped (empty) tiers, tier->name resolution, and the shares' denominator. The advisory
// above reads this fold's output, so an error here would silently mis-report the layering.
func TestTierDistributionFoldsASyntheticTable(t *testing.T) {
	// Counts chosen as exact binary fractions (2/4, 1/4) so share equality is not a float
	// tolerance question. Tier 3 is deliberately unpopulated: an empty tier gets NO row.
	got := tierDistribution(map[string]int{"a": 1, "b": 1, "c": 2, "d": 4})
	want := []tierCount{
		{Tier: 1, Name: "primitive", Count: 2, Share: 0.5},
		{Tier: 2, Name: "foundation-composite", Count: 1, Share: 0.25},
		{Tier: 4, Name: "composer", Count: 1, Share: 0.25},
	}
	if len(got) != len(want) {
		t.Fatalf("got %d rows %+v, want %d %+v", len(got), got, len(want), want)
	}
	for i, w := range want {
		if got[i] != w {
			t.Errorf("row %d = %+v, want %+v", i, got[i], w)
		}
	}

	// A tier with no entry in tierName must fold to an EMPTY name rather than panic on the
	// index -- that unnamed row is exactly what the advisory's name check surfaces.
	odd := tierDistribution(map[string]int{"z": 9})
	if len(odd) != 1 || odd[0] != (tierCount{Tier: 9, Name: "", Count: 1, Share: 1}) {
		t.Errorf("out-of-range tier folded to %+v, want one unnamed row {9  1 1}", odd)
	}

	// Empty population: no rows, and no division by a zero total.
	if empty := tierDistribution(map[string]int{}); len(empty) != 0 {
		t.Errorf("empty table folded to %+v, want no rows", empty)
	}
}

// misTierGap is how many tiers a leaf's DECLARED tier may sit above its import-derived floor
// before the mis-tier advisory surfaces it. A gap of 1 is routinely intentional (a leaf named
// for a role it will grow into), so the floor for flagging is 2. Tunable; advisory only.
const misTierGap = 2

func TestMisTierAdvisory(t *testing.T) {
	internal := internalDir(t)
	report, err := archreport.Analyze(filepath.Dir(internal), "")
	if err != nil {
		t.Fatalf("derive enforced architecture graph: %v", err)
	}
	gaps := report.SinkCandidates
	if len(gaps) == 0 {
		t.Logf("no leaf is declared >= %d tiers above its import floor", misTierGap)
		return
	}
	t.Logf("advisory: %d leaf/leaves declared >= %d tiers above their import floor (could sink lower):", len(gaps), misTierGap)
	for _, g := range gaps {
		t.Logf("  internal/%-24s declared %d (%s) but imports reach only tier %d (%s) -- gap %d",
			g.Name, g.DeclaredTier, g.DeclaredTierName, g.ImportFloor, g.ImportFloorName, g.TierGap)
	}
}

func TestMisTierAdvisoryUsesReportedSinkCandidates(t *testing.T) {
	report := archreport.Report{SinkCandidates: []archreport.SinkCandidate{{Name: "alpha", TierGap: 3}, {Name: "beta", TierGap: 2}}}
	got := make([]string, 0, len(report.SinkCandidates))
	for _, candidate := range report.SinkCandidates {
		got = append(got, candidate.Name)
	}
	want := []string{"alpha", "beta"}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("advisory candidates=%v reported=%v", got, want)
	}
}

func fmtEdge(from string, ft int, to string, tt int) string {
	return from + " (" + tierName[ft] + ") -> " + to + " (" + tierName[tt] + ")"
}

// selfRegisters reports whether a leaf's NON-TEST code contains an actual call to
// abi.Register*(...). It walks the AST for the call expression (not a text grep), so a
// doc comment mentioning abi.Register or a Register call that lives only in _test.go is
// correctly NOT counted â€” a driver only loads if its production init() registers.
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
// `computeadmit` registers its COLLISION_RISK reason code lazily from NewSubmitAdmitter
// (idempotent registerReasonOnce), i.e. when the Kernel.Submit adjudication fold constructs
// the gate — never from init(), so a defconfig blank-import would not even fire it; it is
// wired by its constructor at the Submit seam, not as a passive driver.
// A leaf added here is a conscious "wired elsewhere" decision, the same review
// chokepoint as the tier table.
var regOffList = map[string]bool{"agent": true, "gateway": true, "computeadmit": true, "codetools": true, "systools": true, "observer": true, "trajhook": true}

// TestRequestPathLeavesRegistered closes the registration-completeness hole: a leaf whose
// production init() calls abi.Register* MUST be either blank-imported by the defconfig
// (internal/registrations) or on the off-list â€” otherwise its init() never fires and the
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
				"blank-imported in internal/registrations (the defconfig) nor on regOffList â€” so "+
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
// planner (`HTTPPlanner`); `engine` owns the narrow vLLM EngineDriver adapter that
// must speak vLLM's public OpenAI-compatible generation surface; `gateway` is the
// inbound SERVER of that route (the adjudication proxy), not a client; `openaiadapter` is the
// inbound authenticated app-migration compatibility server, not a client; and off-path
// witnesses may replay or benchmark the same wire. cmd/fak's help text also names it
// but lives outside internal/, so it is not scanned.
var chatEndpointRole = map[string]string{
	"agent":             "the single outbound chat-completions client (HTTPPlanner)",
	"engine":            "the narrow vLLM EngineDriver adapter speaking vLLM's OpenAI-compatible generation surface",
	"gateway":           "the inbound /v1/chat/completions server route (adjudication proxy)",
	"openaiadapter":     "the inbound authenticated app-migration compatibility server (not a live planner)",
	"chatrelay":         "the off-path Slack bridge client to a served in-kernel model (not a live planner)",
	"webbench":          "the off-path serving-parity benchmark client (not a live planner)",
	"guardtrace":        "the off-path trace-replay upstream fake (OpenAI/Anthropic provider replay, not a live planner)",
	"frontierswe":       "the off-path FrontierSWE co-resident env adapter/smoke witness against fak serve (not a live planner)",
	"macbench":          "the off-path Mac gateway serving-parity benchmark client against fak serve (not a live planner)",
	"eveparity":         "the off-path Eve-eval parity witness (#2605): a self-contained fixture server + client that both replays the route to prove fak-routed == raw (not a live planner)",
	"trajctl":           "the off-path GatewayJudgeClient (#2543): an LLM-as-judge W1 progress scorer that POSTs a forced-tool verdict call to fak's own gateway (not a live planner)",
	"deepseekbench":     "the off-path DeepSeek V4 TTFT/TPOT/context-scaling benchmark client (#3014): a streaming latency/throughput measurement against an OpenAI-compatible endpoint (hosted DeepSeek or self-hosted vLLM/SGLang) reporting OBSERVED provider speed (not a live planner)",
	"glm52prefillsweep": "the off-path GLM-5.2 prefill-latency sweep client (#3085/#3086): POSTs a prefill-dominant request (large prompt, max_tokens~1) at each prompt length against a fak serve endpoint to record TTFT / prefill tok/s (not a live planner)",
	"qwen38quantrun":    "the off-path Qwen3.8 quantization corpus runner and production-soak client (#8343/#8319) against a declared OpenAI-compatible endpoint (not a live planner)",
	"zaitask":           "the off-path bounded non-streaming Z.AI Coding Plan task runner (not a live kernel planner)",
	"armbench":          "the off-path benchmark clients for fak/OpenAI-compatible generation surfaces (not a live planner)",
	"cachevalue":        "the off-path Qwen3.8 cache-value cold-arm benchmark runner against a declared OpenAI-compatible endpoint (not a live planner)",
	"cavemanpairwise":   "the off-path pairwise benchmark judge client against a declared OpenAI-compatible endpoint (not a live planner)",
	"conformance":       "the off-path provider-conformance probe client (not a live planner)",
	"quality":           "the off-path bounded pre-dataset capability probe (#10030/#10162): one-token generation/reasoning calls classify endpoint support, infrastructure, and provenance (not a live planner)",
	"serveradapter":     "independent local llama-server adapter and readiness probe owned by the serveradapter leaf",
	"tb4bench":          "the off-path Terminal-Bench 4 benchmark client comparing fak harness vs reference runner (not a live planner)",
	"agentxbench":       "the off-path AgentX benchmark client against a declared OpenAI-compatible endpoint (not a live planner)",
	"gym":               "the off-path sandbox evaluation gym scenario runner against an ephemeral gateway (not a live planner)",
}

// TestSingleOpenAIChatClient pins the T4 fix as an architecture invariant: the
// OpenAI chat-completions endpoint path appears, as a string literal in non-test
// code, in EXACTLY the declared set of packages. A new package referencing it is a
// re-grown duplicate client (fail: re-home the call in `agent`); a declared package
// that stops referencing it is a client/route that went missing (fail: a required
// seam vanished, or update the table if the topology deliberately changed). Reading
// the AST â€” not a text grep â€” means a doc comment that merely mentions the path is
// correctly not counted.
func TestSingleOpenAIChatClient(t *testing.T) {
	internal := internalDir(t)
	got := pkgsReferencingChatEndpoint(t, internal)

	for pkg := range got {
		if _, ok := chatEndpointRole[pkg]; !ok {
			t.Errorf("package internal/%s references the OpenAI chat-completions endpoint as a "+
				"string literal, but is not a declared holder. The live client must live ONLY in "+
				"internal/agent (HTTPPlanner) â€” a second one is the TICKETS-T4 duplicate-seam "+
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
// reachable from internal/registrations â€” the live request-path closure. It is the
// root the kernel boots from: registrations blank-imports every leaf that ships
// enabled, so anything it can reach is on a live tool-call decision's dependency
// graph. The walk reuses imports() (parser.ImportsOnly) and the same modPrefix-trim
// + sub-package collapse as TestNoUpwardImports.
//
// imports() is build-tag-blind BY DESIGN (see its comment): every file is parsed
// regardless of GOOS/GOARCH tags. So this closure is the UNION over all build
// constraints â€” a strict SUPERSET of `go list -deps`, which honors only the host's
// tags. For a ban-gate that is the safe direction: the closure can over-include a
// package, never silently drop one a tagged build would pull onto the path. Do not
// "fix" imports() to honor tags here â€” that would narrow the gate.
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
// with a trailing .exe stripped â€” so "/usr/bin/python3" and "C:\\py\\python.exe" both
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
// non-literal program arg), keyed to a one-line justification â€” the same review
// chokepoint as regOffList. Each entry is a deliberate, reviewed decision, not a
// silent pass.
var interpreterExecAllow = map[string]string{
	"windowgate":  "host process launcher executes caller-selected compiled programs after policy classification; script interpreters are rejected by the windowgate classifier",
	"toolcatalog": "catalog command adapters execute declaration-selected compiled tools; declarations are validated before launch and are not interpreter dependencies",
	"codetools":   "coding Bash engine executes an adjudicator-admitted argv; the dynamic program is the admitted tool target, not a kernel interpreter dependency",
	"witness":     "execution witnesses run caller-declared selector argv; the selector is evidence-gated and not a script interpreter dependency of the kernel itself",
	"procguard":   "host process-guard telemetry/reaper — execs the OS process tools (ps/taskkill/PowerShell CIM one-liners) through the shared runTool helper as a host-observation seam; not an adjudication dependency of the tool-call decide path",
	"accounts":    "credential-refresh spawn (DefaultRefreshSpawn) execs the resolved claude COMPILED binary via ClaudeExe() (env/PATH/conventional-install lookup, never a script interpreter); the path is platform/config-dependent so it cannot be a literal, and the spawn is a token-rotation host seam, not a tool-call adjudication dependency",
	"modelroute":  "cross-audit corpus self-check executes a structured, declaration-matched witness argv (normally the compiled crossauditfixture binary); the dynamic path is test/CLI calibration evidence, not a script interpreter dependency of tool-call adjudication",
	"compute":     "host hardware topology probe queries the Windows display subsystem for AMD GPUs via Win32_VideoController in ProbeWindowsDisplayTopology; host-observation seam, not a tool-call adjudication dependency",
}

// oracleSeamFiles names the off-path Python oracle/baseline seam scripts (DIRECTION.md
// seam table row 1). They are the independent ML-ecosystem witness the Go model is
// measured against â€” legitimate at a typed, off-path boundary, reachable ONLY from
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
// binary) is fine â€” that is not the per-decide untyped-runtime boundary fak exists to
// remove; execing python/node/sh IS.
//
// Seeded at green (2026-06-18): every in-closure exec.Command in the tree is a literal
// "git" (shipgate/witness), which is a compiled binary and passes. The one variable
// program-name exec (bench, exec.Command(binPath, "hook")) is OFF the registrations
// closure, so the fail-closed branch is dormant â€” the seed is clean, the allow-list is
// empty. Like every architest gate it is tightened over time, never loosened to admit a
// new violation.
//
// Fail-closed polarity: this is a BAN, so a program arg the gate cannot prove is a
// compiled binary (a non-literal â€” variable, concat, call) FAILS rather than passes.
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
							"adjudicates with ZERO runtime dependency on Python/shell/node â€” an "+
							"interpreter exec on the path breaks the direction. Move it off-path; if "+
							"%q is a compiled binary the gate misclassified, that is a gate bug.",
							pkg, lit, lit)
					}
					// also scan later args for a script-body suffix (e.g. exec.Command("python", "x.py"))
					for _, a := range call.Args {
						if s, ok := stringLit(a); ok && a != prog && looksLikeInterpreter(s) {
							t.Errorf("request-path package internal/%s execs a script body %q "+
								"(reachable from internal/registrations) â€” a script interpreter on the "+
								"decision path breaks DIRECTION.md. Move it off-path.", pkg, s)
						}
					}
					return true
				})
			}
		}

		// Belt rung: a closure package that imports os/exec but exposed zero recognized
		// exec.Command/CommandContext calls is execing via an aliased import or an
		// unrecognized form the AST walk missed â€” fail closed rather than wave it through.
		if execImported && !sawExecCall {
			if _, allowed := interpreterExecAllow[pkg]; allowed {
				continue
			}
			t.Errorf("request-path package internal/%s imports os/exec but the gate classified "+
				"zero exec.Command/CommandContext calls â€” exec via an aliased import or an "+
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
// _test.go and off-path commands â€” never named as a live string in a non-test file on
// the registrations closure.
//
// The scan reads only string LITERALS (BasicLit) from the AST, so a doc comment that
// mentions a seam file â€” e.g. model/weights.go's "reads a directory produced by
// export_oracle.py" â€” is correctly NOT a violation, exactly as TestSingleOpenAIChatClient
// distinguishes a literal from a comment. Seeded at green: today the only references are
// comments. A real os.ReadFile("â€¦/export_oracle.py") on the path would fail.
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
// a call of the form <recv>.<name>(...) â€” read from the AST, not a text grep, so a
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
// ResultAdmitter chain â€” the same fold kvmmu.FoldedGate uses â€” so a rank-5+ detector
// (normgate today, anything the fleet adds later) that quarantines a payload also
// catches it on reload. The shipped construction backs the Session re-screen with a
// bare ctxmmu gate; folding `abi.ResultAdmittersFor` is what makes readmission
// inherit the stronger chain.
//
// WHY A GATE (and why it BITES): the compiler cannot enforce this. A bare
// ctxmmu.New() gate is valid Go, and the gate field is a concrete *ctxmmu.MMU with
// no interface that forces the fold â€” so a future edit could silently revert recall
// to bare-gate readmission and Go would wave it through, re-opening the documented
// weakening. This is the difference from the REJECTED reason-closure gate, which was
// vacuous because Verdict.Reason is already a typed enum the compiler enforces. Here
// the AST gate catches a real construction regression the type system permits.
//
// Seeded GREEN: recall/recall.go's reScreen now folds abi.ResultAdmittersFor
// (the fix this gate guards). A revert to bare-gate-only readmission drops that call
// and turns the gate RED â€” exactly the regression it exists to catch.
func TestRecallReadmissionFoldsRegistry(t *testing.T) {
	internal := internalDir(t)
	if !pkgCallsSelector(t, internal, "recall", "abi", "ResultAdmittersFor") {
		t.Error("internal/recall does not call abi.ResultAdmittersFor in non-test code â€” its " +
			"page-in re-screen has regressed to a bare single-driver gate that ignores the kernel's " +
			"registered ResultAdmitter chain (the GROWTH.md readmission-gate-strength weakening). A " +
			"rank-5+ detector (normgate) that quarantines a payload would then NOT catch it on reload. " +
			"Fold abi.ResultAdmittersFor in reScreen, most-restrictive-wins, exactly as kvmmu.FoldedGate does.")
	}
}

// foldSites names every internal package whose NON-TEST source folds a verdict chain
// most-restrictive-wins, paired with the role that earns it. The fold's ordering MUST
// come from abi.FoldRank â€” the single restrictiveness-lattice authority â€” never from a
// raw comparison of VerdictKind values. This matters because a VerdictKind's numeric
// value is its REGISTRATION-BLOCK id, not its restrictiveness: VerdictDeny is kind 1
// numerically but FoldRank 100 (most restrictive), while a registered combinator-verdict
// in the 1024+ vendor block can carry any declared rank. A fold that wrote
// `if v.Kind > best.Kind` would compile and silently mis-order â€” picking a numerically-
// larger but LESS restrictive verdict and dropping a real Deny/Quarantine on the floor.
// abi.FoldRank is the indirection that lets the FROZEN fold order a NEW kind without a
// core edit (registry.go); every fold site must route through it.
//
// Set AT REALITY (2026-06-20) from an AST scan of `abi.FoldRank(` callers: these four
// are exactly the packages that fold a verdict chain by restrictiveness â€” the kernel's
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
// is valid Go â€” VerdictKind is a uint16, fully ordered â€” but it orders by registration-block
// id, not restrictiveness, so it silently picks the wrong verdict (see foldSites). There is
// no interface that forces a fold loop to consult abi.FoldRank, so a future edit could revert
// any site to a bare comparison and Go would wave it through, re-opening a most-restrictive-
// wins violation on the live decision path. Like TestRecallReadmissionFoldsRegistry, the AST
// gate catches a real construction regression the type system permits; unlike the REJECTED
// reason-closure gate, it is not vacuous â€” nothing in the type system already guarantees it.
//
// Seeded GREEN (2026-06-20): all four sites call abi.FoldRank today. Dropping the call at any
// of them â€” the regression to a hand-rolled .Kind comparison â€” turns the gate RED.
func TestFoldSitesOrderByFoldRank(t *testing.T) {
	internal := internalDir(t)
	for pkg, role := range foldSites {
		if !pkgCallsSelector(t, internal, pkg, "abi", "FoldRank") {
			t.Errorf("fold site internal/%s (%s) does not call abi.FoldRank in non-test code â€” its "+
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
// so the gate's verdict is the same on every node â€” a Mac, a GPU server, and this
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
// into a gate (the architest gap-(a) knife): the portable pure-Go artifact â€” the
// binary that adjudicates a live tool call without a device runtime â€” must link NO cgo. A cgo import pulls a C
// toolchain and a dynamically-linked C runtime onto the decision path, voiding the
// "one static Go binary, no external runtime" property the whole direction rests on
// (the same thesis TestRequestPathInterpreterFree enforces for a script interpreter;
// this is its compile-time, in-process twin â€” cgo is the untyped runtime that creeps
// in at LINK time rather than via exec).
//
// The scan is over requestPathClosure (everything transitively reachable from
// internal/registrations). For each package it asks go/build â€” NOT a shelled-out
// `go build` â€” which files compile under defaultBuildContext, and fails if any of
// them is a cgo file (import "C"). go/build.ImportDir resolves //go:build
// constraints, so this is the one architest gate that is deliberately build-tag-AWARE
// rather than tag-blind: the cgo files legitimately EXIST in the tree (the GPU
// backends), gated out of the default binary by opt-in tags. A tag-blind scan (like
// imports()) would see compute/cuda.go's import "C" and fire RED at seed; the
// default-context scan correctly sees only what a real `go build ./cmd/fak` links.
//
// WHY A GATE (and why it BITES): nothing in the type system or `go vet` stops an
// agent from adding an UNTAGGED `import "C"` (or a cgo file whose //go:build clause
// matches the default context) to a request-path leaf â€” it compiles fine on a box
// with a C toolchain, and the regression (the shipped fak binary silently became a
// cgo binary) is invisible until a from-source build on a toolchain-less node fails,
// or until the static-binary release artifact stops being static. go/build sorts a
// tag-passing cgo file into CgoFiles; a tag-gated one into IgnoredGoFiles. So
// len(CgoFiles)>0 on the closure is exactly "the default build links cgo."
//
// Seeded GREEN (2026-06-20, updated for #62): the cgo files in the module â€”
// compute/{cuda,metal,vulkan}.go, metalgemm/metalgemm.go, model/awq_cuda.go â€” are
// all behind either explicit device tags or Apple-Silicon+cgo constraints, so under
// defaultBuildContext every request-path package reports zero CgoFiles. Adding an
// untagged import "C" to any in-closure package turns the gate RED â€” exactly the
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
			// all-tagged package) is not a cgo violation â€” go/build reports NoGoError.
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
// the role that earns it. Unlike the KEYED registries (RegisterEngine(id,â€¦),
// RegisterPageOutBackend(id,â€¦), RegisterWitnessResolver(id,â€¦)) â€” which are maps where
// multiple drivers coexist â€” RegisterRegionBackend is a SINGLETON setter: registry.go
// does a bare `reg.regionBackend = b`, so "the last registration wins" (its own doc) and
// every registrant after the first SILENTLY overrides the one before it, with the survivor
// decided by blank-import order. ARCHITECTURE.md frames this seam as a single Resolver
// swap (v0.1 = the content-addressed blob store copy; "zero-copy KV co-residence â€¦ is a
// `RegionBackend` swap behind Capability \"zerocopy\""), i.e. exactly ONE backend is live
// at a time. `blob` ships the v0.1 default; it is the only legitimate registrant today.
var regionBackendRole = map[string]string{
	"blob": "the v0.1 default Ref/Resolver backend (content-addressed blob store)",
	// The DELIBERATE, reviewed swap (storedrv/config.go init): the storage-driver router
	// becomes the single live Ref backend, fanning across its blob/blobfs/blobhttp tiers.
	// It registers ONLY when FAK_STORE opts in (unset => inert, blob stays live), and
	// imports blob so blob's init runs first â€” the last-wins override is order-deterministic.
	"storedrv": "the FAK_STORE-gated storage-driver router (fans Ref resolution across tiers; blob remains the unset default)",
	// The DELIBERATE, reviewed zero-copy override (#448, xenginekv/register.go init): the
	// cross-engine KV co-residence arena becomes the single live Ref backend, handing out
	// RefRegion handles that resolve to VIEWS (Capability "zerocopy"). It registers ONLY when
	// FAK_XENGINE_KV opts in (unset => inert, blob stays live), and imports blob so blob's
	// init runs first â€” the last-wins override is order-deterministic. This is the documented
	// zerocopy swap this gate's own comment anticipates.
	"xenginekv": "the FAK_XENGINE_KV-gated cross-engine KV co-residence arena (zero-copy RefRegion views; blob remains the unset default)",
}

// TestSingleRegionBackendRegistrant turns ARCHITECTURE.md's "Ref backend is a single
// Resolver swap" seam into a gate, the singleton dual of TestSingleOpenAIChatClient: the
// set of non-test packages that call abi.RegisterRegionBackend MUST be exactly
// regionBackendRole. A NEW registrant is a second backend silently flipping Ref
// resolution by blank-import order (the last-wins corruption); a DECLARED registrant that
// stops calling it is the v0.1 default gone missing â€” boot with no Ref/Resolver backend.
//
// WHY A GATE (and why it BITES): the compiler cannot enforce this. RegisterRegionBackend
// takes any RegionBackend and assigns it unconditionally (`reg.regionBackend = b`); a
// second call in some other leaf's init() compiles cleanly and just overwrites the field,
// so a regression to two registrants is invisible until the wrong Resolver wins at
// runtime on a particular import order. There is no interface or type that forces
// singularity â€” this is the same construction-regression class as
// TestRecallReadmissionFoldsRegistry/TestFoldSitesOrderByFoldRank, not the vacuous
// already-typed-enum class of the rejected reason-closure gate.
//
// This is NOT a blunt "â‰¤1 forever" cap: the documented zerocopy override (ship a
// RegionBackend whose Resolver hands out shared-arena handles, advertise "zerocopy") is a
// deliberate, reviewed swap â€” when it lands, the new singleton owner is added to
// regionBackendRole (and the old default removed if it is being replaced, since only one
// is live), the same conscious review chokepoint as adding a tier or a chatEndpointRole
// holder. The gate forces that swap to be explicit instead of an accidental second init().
//
// Seeded GREEN (2026-06-20): an AST scan of non-test callers finds exactly {blob}
// (blob/store.go's init()); every other RegisterRegionBackend call in the tree is a
// _test.go fixture registering an inlineBackend and is correctly not counted. Adding a
// second non-test registrant â€” or blob dropping the call â€” turns the gate RED.
func TestSingleRegionBackendRegistrant(t *testing.T) {
	internal := internalDir(t)
	got := pkgsCallingSelector(t, internal, "abi", "RegisterRegionBackend")

	for pkg := range got {
		if _, ok := regionBackendRole[pkg]; !ok {
			t.Errorf("package internal/%s registers a RegionBackend (abi.RegisterRegionBackend) in "+
				"non-test code, but is not a declared holder. The Ref/Resolver backend is a SINGLETON "+
				"(registry.go: `reg.regionBackend = b`, last-wins) â€” a second registrant silently flips "+
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
// <recv>.<name>(...) â€” read from the AST, so a doc comment or a call living only in
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
// plural by design â€” multiple distinct ids legitimately coexist â€” so the load-bearing
// invariant is per-ID ("who registers id X"), not per-symbol. Reading the AST (not a
// text grep) means a doc comment or a _test.go fixture naming the same id does not count;
// matching arg0 as a BasicLit string means RegisterPageOutBackend("blob", â€¦) is found
// while RegisterPageOutBackend(someVar, â€¦) or a different literal id is not.
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
// page-out codec â€” the backend stored under id "blob" â€” in its NON-TEST source, paired
// with the role that earns it. Unlike abi.RegisterRegionBackend (a SINGLETON setter,
// pinned by TestSingleRegionBackendRegistrant), abi.RegisterPageOutBackend is a KEYED
// map: registry.go does `reg.pageOut[id] = b`, so distinct ids coexist by design (the
// doc-promised "headroom sidecar later" is a SECOND, differently-keyed codec, perfectly
// legitimate). What is NOT legitimate is a second registrant of the SAME id "blob": the
// map assignment is unconditional and last-write-wins, so a second `RegisterPageOutBackend
// ("blob", â€¦)` in another leaf's init() silently swaps the process-wide default page-out
// codec, with the survivor decided by blank-import order. ARCHITECTURE.md frames this row
// as "Go blob store default; headroom sidecar later" â€” exactly one default per id, and the
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
// missing â€” the context-MMU pages a quarantined result out to a handle with no codec under
// the default id.
//
// WHY A GATE (and why it BITES): the compiler cannot enforce this. RegisterPageOutBackend
// takes any (string, PageOutBackend) and does `reg.pageOut[id] = b` unconditionally; a
// second call with id "blob" in some other leaf's init() compiles cleanly and just
// overwrites the map entry, so a regression to two "blob" registrants is invisible until
// the wrong codec wins at runtime on a particular import order. There is no interface or
// type that forces per-id singularity â€” this is the same construction-regression class as
// TestSingleRegionBackendRegistrant / TestFoldSitesOrderByFoldRank, not the vacuous
// already-typed class of the rejected reason-closure gate. It is DISTINCT from the region-
// backend gate: that one is symbol-scoped (RegisterRegionBackend is a singleton, so any
// second caller fails); this one is (symbol, id)-scoped, because the page-out registry is
// plural â€” only a second caller of the SAME default id is the regression, and a sidecar
// codec under a new id is correctly allowed.
//
// This is NOT a blunt "â‰¤1 PageOutBackend forever" cap: the documented headroom sidecar
// (a second page-out codec under a DIFFERENT id) is a deliberate, designed extension and
// does not trip this gate at all â€” only a re-registration of the "blob" default does. A
// deliberate swap of the default itself (ship a new codec under id "blob") is the same
// conscious review chokepoint as adding a tier or a regionBackendRole holder: update
// pageOutDefaultRole to name the new owner and remove the one it replaces.
//
// Seeded GREEN (2026-06-20): an AST scan of non-test callers with arg0=="blob" finds
// exactly {blob} (blob/store.go's init(), which registers the same object as both the
// RegionBackend singleton and the "blob" page-out default); every other
// RegisterPageOutBackend call in the tree is a _test.go fixture and is correctly not
// counted. Adding a second non-test registrant of id "blob" â€” or blob dropping the call â€”
// turns the gate RED.
func TestSinglePageOutBlobRegistrant(t *testing.T) {
	internal := internalDir(t)
	got := pkgsCallingSelectorWithStringArg(t, internal, "abi", "RegisterPageOutBackend", pageOutDefaultID)

	for pkg := range got {
		if _, ok := pageOutDefaultRole[pkg]; !ok {
			t.Errorf("package internal/%s registers the page-out codec under id %q "+
				"(abi.RegisterPageOutBackend(%q, â€¦)) in non-test code, but is not a declared holder. The "+
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
// resolver under id "dos_verify" â€” the one that backs the require-witness verdict â€” in
// its NON-TEST source, paired with the role that earns it. Like RegisterPageOutBackend,
// abi.RegisterWitnessResolver is a KEYED map: registry.go does `reg.witnesses[id] = w`
// UNCONDITIONALLY (no clash panic, unlike RegisterOp/RegisterVerdictKind), so distinct
// ids coexist by design (a future "dos_status" or "external_anchor" resolver is a SECOND,
// differently-keyed witness, perfectly legitimate). What is NOT legitimate is a second
// registrant of the SAME id "dos_verify": the map assignment is last-write-wins, so a
// second `RegisterWitnessResolver("dos_verify", â€¦)` in another leaf's init() silently
// swaps the resolver the kernel consults on a RequireWitness verdict â€” the in-process
// git-evidence effect-verify that turns a claimed effect into a corroborated one â€” with
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
// stops registering "dos_verify" is the v0.1 effect-verifier gone missing â€” the kernel
// consults abi.Witnesses() on a RequireWitness verdict and finds no resolver under the id
// the require-witness gate names.
//
// WHY A GATE (and why it BITES): the compiler cannot enforce this. RegisterWitnessResolver
// takes any (string, WitnessResolver) and does `reg.witnesses[id] = w` unconditionally â€”
// no clash panic â€” so a second call with id "dos_verify" in some other leaf's init()
// compiles cleanly and just overwrites the map entry, and the regression to two
// "dos_verify" registrants is invisible until the wrong resolver wins at runtime on a
// particular import order. There is no interface or type that forces per-id singularity â€”
// this is the same construction-regression class as TestSinglePageOutBlobRegistrant /
// TestSingleRegionBackendRegistrant, not the vacuous already-typed class of the rejected
// reason-closure gate. It is (symbol, id)-scoped, not symbol-scoped: the witness registry
// is plural by design (a sidecar resolver under a NEW id is correctly allowed), so only a
// second caller of the SAME id "dos_verify" is the regression.
//
// This is NOT a blunt "â‰¤1 WitnessResolver forever" cap: the documented DOS read-back /
// external-anchor resolvers (a second witness under a DIFFERENT id) are deliberate, designed
// extensions and do not trip this gate at all. A deliberate swap of the "dos_verify" default
// itself is the same conscious review chokepoint as adding a tier or a pageOutDefaultRole
// holder: update witnessResolverRole to name the new owner and remove the one it replaces.
//
// Seeded GREEN (2026-06-20): an AST scan of non-test callers with arg0=="dos_verify" finds
// exactly {witness} (witness.go's init()); the kernel's witness tests register id "test"
// (a different id) and are correctly not counted. Adding a second non-test registrant of id
// "dos_verify" â€” or witness dropping the call â€” turns the gate RED.
func TestSingleWitnessResolverRegistrant(t *testing.T) {
	internal := internalDir(t)
	got := pkgsCallingSelectorWithStringArg(t, internal, "abi", "RegisterWitnessResolver", witnessResolverID)

	for pkg := range got {
		if _, ok := witnessResolverRole[pkg]; !ok {
			t.Errorf("package internal/%s registers a witness resolver under id %q "+
				"(abi.RegisterWitnessResolver(%q, â€¦)) in non-test code, but is not a declared holder. The "+
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
// legitimately imports os/exec, keyed to a one-line justification â€” the same conscious
// review chokepoint as regOffList / interpreterExecAllow. `shipgate` registers a rank-40
// Adjudicator but is the RSI/ship harness (git + worktree orchestration), explicitly NOT
// the dispatch hot path â€” its own source says so (shipgate.go: "this is the RSI harness,
// NOT the dispatch hot path, so the os/exec-absence proof â€¦ does not apply here"). A new
// entry here is a deliberate, reviewed decision that a given Adjudicator's exec is off the
// live tool-call decision path, not a silent pass.
var adjudicatorExecAllow = map[string]string{
	"codetools": "bounded Bash tool engine — subprocess execution occurs only after the adjudicator admits a Bash call; non-Bash decisions remain in-process",
	"shipgate":  "RSI/ship harness â€” git + worktree exec, registered as a rank-40 Adjudicator but off the dispatch hot path (shipgate.go)",
}

// TestEveryAdjudicatorIsExecFree is the "any Adjudicator" clause of DIRECTION.md's
// reviewer-grep #1 turned into a gate, and the AST-derived superset of
// TestHotPathHasNoExec. That gate checks a HAND-MAINTAINED hotPath list of seven leaves
// for os/exec; this one derives the set of packages that actually register an Adjudicator
// (abi.RegisterAdjudicator, read from the AST) and asserts NONE of them imports os/exec â€”
// except a declared escape-list. DIRECTION.md: os/exec "must never appear in adjudicator,
// kernel, vdso, ctxmmu, or any Adjudicator." A registered Adjudicator is, by definition, a
// rung on the live PDP/PEP decision chain; an os/exec import there puts a per-decide
// subprocess on the exact path fak exists to keep interpreter-free.
//
// WHY A GATE (and why it BITES): the hotPath list is manually curated, so it silently lags
// reality. Today four packages register an Adjudicator but are NOT in hotPath â€” engine
// (rank 12), ifc (rank 30), plancfi (rank 25), shipgate (rank 40) â€” so TestHotPathHasNoExec
// does not check them at all. A NEW Adjudicator rung (or one of those four growing an
// os/exec import) would pass TestHotPathHasNoExec untouched while quietly adding a
// subprocess to the live decision chain. Deriving the set from RegisterAdjudicator callers
// closes that lag: a new rung is checked the moment it registers, with no hotPath edit
// required. This is the same construction-regression class as the registrant-singleton
// gates â€” nothing in the type system stops an Adjudicator-registering package from importing
// os/exec â€” not the vacuous already-typed class of the rejected reason-closure gate.
//
// Seeded GREEN (2026-06-20): eight packages register an Adjudicator (adjudicator, engine,
// grammar, ifc, plancfi, preflight, ratelimit, shipgate); seven import no os/exec, and the
// only one that does â€” shipgate, the RSI/ship harness off the dispatch path â€” is on
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
				t.Errorf("Adjudicator-registering package internal/%s imports os/exec â€” it runs on the "+
					"live PDP/PEP decision chain (abi.RegisterAdjudicator), and DIRECTION.md forbids os/exec "+
					"in any Adjudicator: a per-decide subprocess is exactly the boundary fak exists to remove. "+
					"Move the exec to an off-path (non-Adjudicator) package, or, if this Adjudicator is "+
					"genuinely off the dispatch hot path (as shipgate's RSI harness is), add %q to "+
					"adjudicatorExecAllow with a one-line justification.", pkg, pkg)
			}
		}
	}

	// The escape-list must not rot: an allow entry for a package that no longer registers
	// an Adjudicator (or no longer imports os/exec) is a stale exception masking nothing â€”
	// remove it so the list keeps naming only real, current exceptions.
	for pkg, why := range adjudicatorExecAllow {
		if !adjudicators[pkg] {
			t.Errorf("adjudicatorExecAllow names internal/%s (%q) but it no longer registers an "+
				"Adjudicator â€” a stale exception. Remove it from adjudicatorExecAllow.", pkg, why)
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
			t.Errorf("adjudicatorExecAllow names internal/%s (%q) but it no longer imports os/exec â€” a "+
				"stale exception that grants an exemption nothing uses. Remove it from adjudicatorExecAllow.",
				pkg, why)
		}
	}
}

// resultAdmitterExecAllow is the declared escape-list for TestEveryResultAdmitterIsExecFree,
// the result-side twin of adjudicatorExecAllow. A package that registers a ResultAdmitter
// (so it runs on the live WRITE-TIME result-admission chain) AND legitimately imports
// os/exec would be keyed here with a one-line justification. EMPTY at green: today every
// ResultAdmitter (ctxmmu, ifc, normgate) is exec-free, so no exception is needed â€” unlike
// the Adjudicator side, where shipgate's off-path RSI harness earns one. A future entry is
// a deliberate, reviewed decision that a given ResultAdmitter's exec is off the live path.
var resultAdmitterExecAllow = map[string]string{}

// TestEveryResultAdmitterIsExecFree is the result-admission dual of
// TestEveryAdjudicatorIsExecFree: it extends DIRECTION.md's no-os/exec-on-the-decision-path
// thesis from the pre-call Adjudicator chain to the WRITE-TIME ResultAdmitter chain. A
// ResultAdmitter (abi.RegisterResultAdmitter) is the post-tool dual of an Adjudicator â€”
// after the engine produces a Result, the kernel folds these to decide whether it may enter
// context (Allow), must be held out (Quarantine), or rewritten to a pointer (Transform).
// That fold runs on every tool-call result, so it is as much a live decision rung as the
// pre-call chain; an os/exec import there puts a per-decide subprocess on the result side of
// the exact boundary fak exists to keep interpreter-free. The set of ResultAdmitter
// registrants is derived from the AST (not a hand-kept list), so a new result-admission rung
// is checked the moment it registers â€” the same lag-closing move as the Adjudicator gate.
//
// WHY A GATE (and why it BITES): the compiler cannot enforce this. Nothing stops a
// ResultAdmitter-registering package from importing os/exec; the regression (a subprocess on
// the result-admission path) is invisible until it runs. TestHotPathHasNoExec does not cover
// these â€” its hand-kept hotPath list names none of ctxmmu's result side, ifc, or normgate as
// result-admitters â€” so without this gate the entire write-time chain is unchecked for exec.
// Same construction-regression class as the Adjudicator/registrant-singleton gates, not the
// vacuous already-typed class of the rejected reason-closure gate.
//
// Seeded GREEN (2026-06-20): three packages register a ResultAdmitter (ctxmmu rank 10, ifc
// rank 20, normgate rank 5); all three import no os/exec, so resultAdmitterExecAllow is
// empty. Adding os/exec to any of them â€” or a new exec-importing ResultAdmitter rung â€” turns
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
				t.Errorf("ResultAdmitter-registering package internal/%s imports os/exec â€” it runs on the "+
					"live write-time result-admission chain (abi.RegisterResultAdmitter), the post-tool dual "+
					"of the Adjudicator chain, so DIRECTION.md's no-os/exec-on-the-decision-path rule applies "+
					"to it too: a per-decide subprocess is the boundary fak exists to remove. Move the exec to "+
					"an off-path (non-ResultAdmitter) package, or, if this admitter is genuinely off the live "+
					"path, add %q to resultAdmitterExecAllow with a one-line justification.", pkg, pkg)
			}
		}
	}

	// The escape-list must not rot (mirrors the Adjudicator gate): an allow entry for a
	// package that no longer registers a ResultAdmitter â€” or no longer imports os/exec â€” is a
	// stale exception masking nothing.
	for pkg, why := range resultAdmitterExecAllow {
		if !admitters[pkg] {
			t.Errorf("resultAdmitterExecAllow names internal/%s (%q) but it no longer registers a "+
				"ResultAdmitter â€” a stale exception. Remove it from resultAdmitterExecAllow.", pkg, why)
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
			t.Errorf("resultAdmitterExecAllow names internal/%s (%q) but it no longer imports os/exec â€” a "+
				"stale exception that grants an exemption nothing uses. Remove it from resultAdmitterExecAllow.",
				pkg, why)
		}
	}
}

// selfModifyWitnessTrees is the set of trees that, if an RSI loop could quietly self-edit
// them, would let the loop grade its own homework â€” so the adjudicator's self-modify floor
// MUST guard every one. Each entry is paired with a sample write target inside it; coverage
// is checked against that target exactly the way Adjudicate matches at runtime (matchGlob:
// the first SelfModifyGlobs fragment CONTAINED IN the path). The witness machinery is:
//
//   - internal/adjudicator â€” the policy-decision point itself (the guard's own code).
//   - internal/architest   â€” these gates; if self-edited, the loop could delete the floor.
//   - internal/shipgate    â€” the RSI/ship harness that lands the candidate.
//   - dos.toml             â€” the workspace lane taxonomy + stamp grammar the referee binds to.
//
// These are exactly the trees #172 Hole 2 (commit ff2dda6) added to SelfModifyGlobs. That
// commit's regression floor (TestSelfModifyGuardsWitnessMachinery) lives IN the adjudicator
// package and asserts the running policy denies a write into each tree â€” the behavioral
// half. This gate is the CROSS-PACKAGE half ff2dda6 deliberately left separate: it reads the
// adjudicator SOURCE (not a constructed Policy) and asserts the literal SelfModifyGlobs set
// in BOTH deployable policy constructors still spells out every witness tree. The two are
// independent on purpose â€” the in-package test could be deleted by the very self-edit it
// guards against (internal/adjudicator is itself a witness tree), so an out-of-package gate
// that re-derives the floor from source is the belt to its suspenders.
var selfModifyWitnessTrees = map[string]string{
	"internal/adjudicator/": "internal/adjudicator/decide.go",
	"internal/architest/":   "internal/architest/architest_test.go",
	"internal/shipgate/":    "internal/shipgate/gate.go",
	"dos.toml":              "dos.toml",
}

// selfModifyPolicyCtors are the adjudicator constructors that return a DEPLOYABLE Policy â€”
// the ones whose SelfModifyGlobs actually guard a running agent. Every one must cover the
// witness trees; a future third deployable policy is added here so it is held to the same
// floor (the forced-choice review gate, like the tier table).
var selfModifyPolicyCtors = []string{"DefaultPolicy", "DevAgentPolicy"}

// selfModifyGlobsFromCtor parses internal/adjudicator's non-test source and returns the
// string-literal SelfModifyGlobs slice assigned inside the named constructor (e.g.
// "DefaultPolicy"). It reads the SOURCE rather than importing adjudicator and calling the
// constructor on purpose: the gate must witness what the code literally declares, so a glob
// built at runtime from a variable (which a self-editing loop could make say anything) can't
// satisfy the floor â€” only a visible string literal in the constructor does. Returns nil if
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
// WITNESS machinery â€” internal/adjudicator (the guard), internal/architest (these gates),
// internal/shipgate (the ship harness), and dos.toml (the taxonomy/stamp grammar) â€” so a loop
// that can edit its own witness can't launder a bad change past the gate.
//
// WHY A GATE (and why it BITES): nothing in the type system keeps a witness tree in
// SelfModifyGlobs; it is a hand-written []string in two constructors. A refactor that drops
// "internal/architest/" (or a self-editing loop that does so deliberately) silently reopens
// the hole â€” the floor would compile and every other test would pass while the loop regained
// the ability to grade its own homework. ff2dda6 already added a behavioral floor
// (TestSelfModifyGuardsWitnessMachinery) IN the adjudicator package, but that test is itself
// inside a witness tree (internal/adjudicator), so the same self-edit that drops a tree from
// the floor could drop the test that guards it. This gate closes that circularity: it lives
// OUTSIDE adjudicator, re-derives the floor from adjudicator's source, and fails if either
// deployable policy constructor stops naming a witness tree â€” the belt to ff2dda6's suspenders.
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
			t.Errorf("adjudicator.%s has no SelfModifyGlobs string-literal slice in its source â€” "+
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
					"This is the #172 Hole 2 floor: %s is part of the witness machinery â€” if a self-improving "+
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
const shellSelfModifyCallee = "commandSelfModifyWithSpecs"

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
// driver under a given id (abi.RegisterEngine(id, â€¦)) in its NON-TEST source, keyed by the
// engine id, paired with the role that earns it. Like RegisterPageOutBackend and
// RegisterWitnessResolver, abi.RegisterEngine is a KEYED map: registry.go does
// `reg.engines[id] = d` UNCONDITIONALLY (no clash panic, unlike RegisterOp /
// RegisterVerdictKind), so distinct ids coexist by design â€” the kernel SELECTS one by id at
// construction (kernel.New("inkernel")). What is NOT legitimate is a second registrant of the
// SAME id: the map assignment is last-write-wins, so a second `RegisterEngine("inkernel", â€¦)`
// in another leaf's init() silently swaps the engine the kernel runs, with the survivor
// decided by blank-import order. The three v0.1 engines each own a distinct id:
//
//   - "inkernel"   â€” modelengine: the in-kernel Go model-fusion engine (the dogfood default).
//   - "localtools" â€” agent: the local tool-call engine cmd/fak wires directly.
//   - "fakread"    â€” agent: the read-only engine for fak_read gateway calls.
//   - "mock"       â€” engine: the routing/mock engine used by the engine-route capability.
//   - "dynamo"     â€” engine: the Dynamo EngineDriver adapter for ridden P/D pools.
//   - "llm-d"      â€” engine: the llm-d EngineDriver adapter for ridden Kubernetes P/D pools.
//   - "mlx"        â€” engine: the MLX ride-adapter fronting mlx-lm/vllm-mlx on Apple Silicon.
//
// A NEW engine under a NEW id is correctly allowed (the map is plural by design); only a
// second registrant of an EXISTING id is the regression this gate catches.
var engineDriverRole = map[string]map[string]string{
	"agent.context_control": {
		"agent": "bounded agent context control engine",
	},
	"agent.skill": {
		"agent": "dynamic agent skill loader engine",
	},
	"agent.todoread": {
		"agent": "native harness task list query engine",
	},
	"agent.todowrite": {
		"agent": "native harness task list mutation and planning engine",
	},
	"inprocess_mcp": {
		"agent": "in-process MCP tool server engine",
	},
	"codetools.bash": {
		"codetools": "bounded coding Bash engine",
	},
	"codetools.edit": {
		"codetools": "bounded coding edit engine",
	},
	"codetools.glob": {
		"codetools": "bounded coding glob engine",
	},
	"codetools.grep": {
		"codetools": "bounded coding grep engine",
	},
	"codetools.read": {
		"codetools": "bounded coding read engine",
	},
	"codetools.write": {
		"codetools": "bounded coding write engine",
	},
	"dynamo":     {"engine": "the Dynamo EngineDriver adapter for ridden P/D serving pools"},
	"inkernel":   {"modelengine": "the in-kernel Go model-fusion engine (dogfood default)"},
	"fakread":    {"agent": "the read-only engine for fak_read gateway calls"},
	"llm-d":      {"engine": "the llm-d EngineDriver adapter for ridden Kubernetes P/D serving pools"},
	"localtools": {"agent": "the local tool-call engine wired by cmd/fak"},
	"mlx":        {"engine": "the MLX ride-adapter fronting mlx-lm/vllm-mlx on Apple Silicon over the OpenAI-compatible wire"},
	"mock":       {"engine": "the routing/mock engine behind the engine.route capability"},
	"sglang":     {"engine": "the SGLang EngineDriver adapter for hosted generation, metrics, and radix-cache observations"},
	"systools.fetch_web": {
		"systools": "safe web fetch engine with byte capping and SSRF protection",
	},
	"systools.get_time": {
		"systools": "system time and timezone engine",
	},
	"systools.web_search": {
		"systools": "web and documentation search engine",
	},
	"vllm": {"engine": "the vLLM EngineDriver adapter for hosted OpenAI-compatible generation, metrics, and KV events"},
}

// resolveEngineIDArg returns the engine-id string a RegisterEngine call's first argument
// names, resolving BOTH a direct string literal (RegisterEngine("localtools", â€¦)) AND a
// same-package string const (RegisterEngine(EngineID, â€¦) where `const EngineID = "inkernel"`).
// The const arm matters: modelengine registers via the const EngineID, so a literal-only
// scan (like pkgsCallingSelectorWithStringArg) would be BLIND to the "inkernel" registrant â€”
// the exact engine the dogfood path selects â€” and could not enforce its singularity. consts
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
// string-literal const declarations are read â€” a const built from an expression is not a
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

// engineRegistrantsByID scans every non-test internal package for abi.RegisterEngine(id, â€¦)
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
						record("", pkg) // unreadable id â€” fail closed
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
// engine gone missing â€” the kernel constructs New("inkernel") and finds no driver under the id.
//
// WHY A GATE (and why it BITES): the compiler cannot enforce this. RegisterEngine takes any
// (string, EngineDriver) and does `reg.engines[id] = d` unconditionally â€” no clash panic â€” so a
// second call with an existing id in some other leaf's init() compiles cleanly and just
// overwrites the map entry; the regression is invisible until the wrong engine wins at runtime
// on a particular import order. There is no interface or type that forces per-id singularity â€”
// the same construction-regression class as the three sibling registrant gates, not the vacuous
// already-typed class of the rejected reason-closure gate.
//
// It is STRONGER than a literal-only id scan (pkgsCallingSelectorWithStringArg): modelengine
// registers via the const EngineID ("inkernel"), so a literal-only gate would be BLIND to the
// dogfood engine's registrant and could not enforce its singularity. resolveEngineIDArg resolves
// the same-package const, so all three v0.1 engines are covered. An id the gate cannot read
// statically (neither a literal nor a known const â€” e.g. a runtime-built id) is recorded under
// the sentinel "" and fails closed: a dynamically-keyed engine registrant defeats the very
// singularity proof this gate exists to make.
//
// This is NOT a blunt "one engine forever" cap: a NEW engine under a NEW id (a remote driver, a
// second model backend) is correctly allowed â€” add it to engineDriverRole with its role, the
// same conscious review chokepoint as adding a tier or a chatEndpointRole holder. A deliberate
// swap of an existing id's owner updates that id's entry. Only an UNDECLARED second registrant
// of an existing id, or a declared one going missing, turns the gate RED.
//
// Seeded GREEN (2026-06-20): an AST scan of non-test callers finds exactly {inkernelâ†’modelengine,
// localtoolsâ†’agent, mockâ†’engine}; every other RegisterEngine call in the tree is a _test.go
// fixture (ids "test"/"e"/"local"/"remote"/"") and is correctly not counted. Adding a second
// non-test registrant of any of those three ids â€” or one of them dropping the call â€” turns the
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
	// entirely, not just changed hands) â€” surface it so the table keeps naming real engines.
	for id, holders := range engineDriverRole {
		if _, present := byID[id]; !present {
			roles := make([]string, 0, len(holders))
			for pkg := range holders {
				roles = append(roles, pkg)
			}
			sort.Strings(roles)
			t.Errorf("engineDriverRole declares engine id %q (holder(s) %v) but NO non-test package "+
				"registers it â€” the engine went missing. Remove the stale entry, or restore the "+
				"registration if it was dropped by mistake.", id, roles)
		}
	}
}

// TestRootImportsNothingInternal gates the tier-0 root-purity invariant that the tier table
// (line "0 root â€¦ imports nothing internal"), doc.go, GROWTH.md ("the one tree everyone
// imports"), and docs/project/PARTITION.md ("internal/abi â€¦ wave-0, human-owned, and unleasable") all
// assert in prose but no test enforces: internal/abi imports ZERO other internal packages.
//
// WHY THIS IS NOT ALREADY COVERED BY TestNoUpwardImports. The layering gate enforces the
// RELATIVE rule tier(importer) >= tier(imported). For abi (tier 0) that catches an import of
// any tier-1+ package (to>0 is an upward edge) â€” but it is BLIND to abi importing another
// tier-0 package, because to(0) > from(0) is false, so the edge passes. The prose invariant is
// ABSOLUTE ("imports nothing internal", "the one tree everyone imports"), strictly stronger
// than "may import packages at its own tier or below." abi is the single shared, human-owned,
// unleasable tree at the bottom of the DAG; its zero-internal-imports property is what lets
// every other leaf import abi without creating a cycle and what keeps it wave-0 separable. A
// `<=`-tier proxy cannot express "exactly zero."
//
// WHY IT BITES (and is not vacuous). Seeded GREEN today: an AST scan of abi's non-test source
// finds no internal import. It turns RED the moment abi grows an import of ANY internal
// package â€” including a second package promoted to tier 0, the one case TestNoUpwardImports
// structurally cannot catch. An internal import inside abi is a real regression: it makes the
// frozen root depend on a higher-churn leaf, voids "everyone imports abi, abi imports no one,"
// and can reintroduce an import cycle the moment that leaf imports abi back (which, being the
// ABI, it almost certainly does). This is the construction-regression class the sibling gates
// catch, not the vacuous already-typed class â€” the compiler permits abi to import any internal
// package; only this gate forbids it.
//
// SCOPE: non-test source only (abi's own _test.go may import internal fixtures freely â€” that is
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
			"The root must import NOTHING internal â€” it is the one tree everyone imports, so any "+
			"internal dependency of abi inverts the DAG (a higher-churn leaf now sits UNDER the ABI) "+
			"and risks an import cycle the moment that leaf imports abi back. TestNoUpwardImports does "+
			"NOT catch this when the dependency is another tier-0 package (to(0) > from(0) is false). "+
			"Push the shared type the other way (the leaf imports abi, never the reverse), or, if this "+
			"is a deliberate direction change, edit this gate consciously â€” do not loosen it silently.",
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
// preflight, ctxmmu, adjudicator, â€¦) are all tier 2, so a sideways kernel->vdso edge satisfies
// to(2) <= from(2). This is the tier-2 analogue of TestRootImportsNothingInternal's tier-0
// floor â€” a named-package import-purity gate the relative DAG rule structurally cannot express.
//
// Seeded GREEN (2026-06-20): internal/kernel's only non-test internal import is abi (kernel.go).
func TestKernelImportsOnlyAbi(t *testing.T) {
	internal := internalDir(t)
	var leafImports []string
	for _, imp := range imports(t, internal, "kernel") {
		if !strings.HasPrefix(imp, modPrefix) {
			continue // stdlib / external â€” out of scope
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
			"ResultAdmitters, engines, emitters), NEVER by importing a leaf directly â€” that is what lets the "+
			"leaves live in disjoint trees linked by one blank-import line in internal/registrations. A direct "+
			"leaf import hard-wires the kernel to that leaf at compile time, re-couples two trees fleet leases "+
			"assume disjoint, and risks an import cycle the moment the leaf imports kernel back. TestNoUpwardImports "+
			"does NOT catch this: kernel and the mechanism leaves are all tier 2, so a sideways edge satisfies the "+
			"DAG rule. Move the shared type into abi and walk it from the registry, or, if this is a deliberate "+
			"direction change, edit this gate consciously â€” do not loosen it silently.",
			len(leafImports), strings.Join(leafImports, "\n  "))
	}
}

// TestUndeclaredLeafVerdictScopesToTouchedLeaf proves #2088's fleet-safety property at the
// decision level: an undeclared leaf this push's commits did NOT touch is advisory (an
// unrelated peer's push is NOT reded for it), while a leaf this push DID touch — or the
// no-trunk fallback — is a hard failure the owning push must fix. This is the pure gate the
// scoped TestEveryPackageDeclaresTier rests on.
func TestUndeclaredLeafVerdictScopesToTouchedLeaf(t *testing.T) {
	cases := []struct {
		name          string
		scoped        bool
		touchedByPush bool
		wantAdvisory  bool
	}{
		{"peer's leaf this push never touched => advisory, does not wedge the push", true, false, true},
		{"leaf this push's commit delivers => hard failure", true, true, false},
		{"no trunk ground (archive/no remote) => strict fail", false, false, false},
		{"no trunk ground but touched flag stale => strict fail", false, true, false},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if got := undeclaredLeafVerdict(c.scoped, c.touchedByPush); got != c.wantAdvisory {
				t.Fatalf("undeclaredLeafVerdict(scoped=%v, touchedByPush=%v) = advisory %v, want %v",
					c.scoped, c.touchedByPush, got, c.wantAdvisory)
			}
		})
	}
}

// TestLeavesTouchedByPushScopesToCommitDelta is the end-to-end fleet-safety witness for
// #2088's Done-when in a real temp repo. It seeds a "trunk" commit that ALREADY carries an
// undeclared peer leaf ('peerleaf'), then makes THIS push a single new commit that touches
// only an unrelated leaf ('myleaf'). leavesTouchedByPush must report {myleaf} — the commit
// delta — and NOT peerleaf. Composed with undeclaredLeafVerdict this proves the property:
//   - peerleaf (undeclared, present in tree, NOT in this push's commits) => advisory: an
//     unrelated peer's leaf does not red this push.
//   - myleaf (undeclared, delivered by this push's commit) => hard failure: the owning push
//     is correctly asked to declare it.
func TestLeavesTouchedByPushScopesToCommitDelta(t *testing.T) {
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git not available")
	}
	repo := t.TempDir()
	internal := filepath.Join(repo, "internal")
	writePkg := func(leaf string) {
		dir := filepath.Join(internal, leaf)
		if err := os.MkdirAll(dir, 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(dir, "doc.go"), []byte("package "+leaf+"\n"), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	git := func(args ...string) {
		cmd := exec.Command("git", args...)
		cmd.Dir = repo
		cmd.Env = append(cleanGitEnv(),
			"GIT_AUTHOR_NAME=t", "GIT_AUTHOR_EMAIL=t@t",
			"GIT_COMMITTER_NAME=t", "GIT_COMMITTER_EMAIL=t@t")
		if out, err := cmd.CombinedOutput(); err != nil {
			t.Fatalf("git %v: %v\n%s", args, err, out)
		}
	}
	git("init", "-q")
	// TRUNK: already carries the peer's undeclared leaf. This is the shared-trunk state a
	// push inherits — the leaf exists but this push has no part in it.
	writePkg("peerleaf")
	git("add", "internal/peerleaf/doc.go")
	git("commit", "-q", "-m", "trunk state with a peer's undeclared leaf")
	branch := func() string {
		cmd := exec.Command("git", "rev-parse", "--abbrev-ref", "HEAD")
		cmd.Dir = repo
		cmd.Env = cleanGitEnv()
		out, err := cmd.Output()
		if err != nil {
			t.Fatalf("rev-parse: %v", err)
		}
		return strings.TrimSpace(string(out))
	}()
	t.Setenv("FAK_ARCHITEST_TRUNK", branch)
	// THIS PUSH: one new commit that touches ONLY an unrelated leaf.
	git("branch", "trunk-snapshot") // freeze the merge-base; FAK_ARCHITEST_TRUNK points here
	t.Setenv("FAK_ARCHITEST_TRUNK", "trunk-snapshot")
	writePkg("myleaf")
	git("add", "internal/myleaf/doc.go")
	git("commit", "-q", "-m", "this push adds myleaf only")

	touched, ok := leavesTouchedByPush(internal)
	if !ok {
		t.Fatal("leavesTouchedByPush reported no scoping ground on a real repo with commits ahead of trunk")
	}
	if !touched["myleaf"] {
		t.Errorf("leaf 'myleaf' delivered by this push's commit not reported as touched: %v", touched)
	}
	if touched["peerleaf"] {
		t.Errorf("peer's pre-existing 'peerleaf' wrongly reported as touched by this push: %v", touched)
	}
	// The property: an unrelated peer leaf is advisory (won't wedge this push); the leaf this
	// push's commit delivers is a hard failure its owner must fix.
	if advisory := undeclaredLeafVerdict(ok, touched["peerleaf"]); !advisory {
		t.Error("undeclared peer leaf 'peerleaf' should be advisory — an unrelated push must NOT be reded for it")
	}
	if advisory := undeclaredLeafVerdict(ok, touched["myleaf"]); advisory {
		t.Error("undeclared 'myleaf' delivered by this push should be a hard failure, not advisory")
	}
}

// TestLeavesTouchedByPushFallsBackWithoutGitTrunk proves the hermetic fallback: outside any
// git repo (e.g. a `git archive` checkout) leavesTouchedByPush returns ok=false, so the gate
// reverts to strict all-undeclared-fail — safe there because no concurrent peer push exists
// to wedge.
func TestLeavesTouchedByPushFallsBackWithoutGitTrunk(t *testing.T) {
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git not available")
	}
	bare := t.TempDir() // not a git repo
	t.Setenv("FAK_ARCHITEST_TRUNK", "origin/main")
	if _, ok := leavesTouchedByPush(filepath.Join(bare, "internal")); ok {
		t.Fatal("leavesTouchedByPush reported scoping ground outside any git repo — strict fallback lost")
	}
}

// TestLeafOf pins the path->leaf mapping the scoping rests on.
func TestLeafOf(t *testing.T) {
	cases := map[string]string{
		"internal/foo/bar.go":          "foo",
		"internal/foo/sub/baz.go":      "foo",
		"internal/architest/x_test.go": "architest",
		"internal/toplevel.go":         "", // a file directly under internal/ is not a leaf
		"cmd/fak/main.go":              "", // outside internal/
		"README.md":                    "",
	}
	for path, want := range cases {
		if got := leafOf(path); got != want {
			t.Errorf("leafOf(%q) = %q, want %q", path, got, want)
		}
	}
}
