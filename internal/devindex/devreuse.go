package devindex

import (
	"fmt"
	"sort"
)

// DevReuse is independent of binary ownership. It distinguishes reusable
// development patterns from machinery specific to fak or its private lab.
type DevReuse string

const (
	DevReuseNA           DevReuse = "not-applicable"
	DevReusePortable     DevReuse = "portable-pattern"
	DevReuseMaintainer   DevReuse = "fak-maintainer"
	DevReuseLab          DevReuse = "lab-operations"
	DevReuseUnclassified DevReuse = "unclassified"
)

func (r DevReuse) valid() bool {
	return r == DevReuseNA || r == DevReusePortable || r == DevReuseMaintainer || r == DevReuseLab
}

var portableDevPatterns = map[string]string{
	"affected":     "dependency-aware affected-test selection is reusable across repositories",
	"blockers":     "typed blocker discovery is a reusable multi-agent coordination pattern",
	"build":        "durable phase-timed build receipts are reusable across repositories",
	"buildcheck":   "peer-dirty-safe compile checking is a reusable shared-checkout pattern",
	"ci-preflight": "clean-tip verification separates committed evidence from working-tree state",
	"commit":       "path-scoped serialized commits are a reusable shared-checkout pattern",
	"dispatch":     "contract-based work dispatch is a reusable multi-agent pattern",
	"done":         "witnessed completion is a reusable agent-workflow pattern",
	"hooks":        "tool-call and git-hook enforcement is a reusable policy pattern",
	"issue":        "contract-shaped issue creation and fan-out are reusable planning patterns",
	"project":      "typed project decomposition is reusable beyond this repository",
	"sweep":        "ownership-aware dirty-tree grouping is a reusable shared-checkout pattern",
	"task":         "typed task contracts are reusable beyond this repository",
	"tasks":        "typed task portfolio views are reusable beyond this repository",
	"validate":     "explicit-path overlay validation is a reusable shared-checkout pattern",
	"workspace":    "workspace state inspection is a reusable agent orientation pattern",
	"worktree":     "leased detached worker isolation is a reusable concurrency pattern",
}

var maintainerDevCommands = []string{
	"accounts",
	"agent-queue",
	// `agentdemo` is the offline mock demo moved off the front-door `fak agent`
	// (the real agent loop). It demonstrates fak's own turn-count A/B, so it is
	// fak-maintainer machinery, not a portable adopter pattern.
	"agentdemo",
	"agents",
	"ailuminate",
	// Host-only GGUF-vs-RCO-sidecar tensor audit (#14091): fak's own weight
	// machinery, not a portable adopter pattern.
	"allocation-check",
	"answer-shape",
	"antipattern-scorecard",
	"api-host",
	"arch",
	"architecture",
	"armbench",
	"arms",
	"assume",
	"audit-leak",
	"auditleak",
	"backends",
	"backend",
	"balance",
	"baseline-arms",
	"bench",
	"bench-effort",
	"bench-ingest",
	"bench-loop",
	"bench-runs",
	"bench-subagent-fanout",
	"benchmarks",
	"bgloop",
	"bitnetmeta",
	"bitnetruntime",
	"blast",
	"borrow-provenance",
	"boundary",
	"breath",
	"budget",
	"build-matrix",
	"c",
	"cache",
	"cachedoc-numbers-audit",
	"cachesweep",
	"cachevalue",
	"cadence",
	"callavoid",
	"causal-receipt",
	"causalreceipt",
	"capabilities",
	"catchup",
	"chat",
	"chatops",
	"chatrelay",
	"check-cache-headlines",
	"check-tool-failure",
	"checkpoint",
	"checkpoint-debt-dispatch",
	"checkpoint-scorecard",
	"citeverify",
	"claim-check",
	"claude",
	"clean-bins",
	"cloud-handoff",
	"cloudhandoff",
	"codelint",
	"code-debt",
	"codebookmeta",
	"codex-hook-census",
	"codex-hook-gate",
	"codex-hook-profile",
	"codex-mcp-health",
	"codex-memory",
	"codex-plugin-sync",
	"codex-resume",
	"codex-stop-acceptance",
	"codex-tool-errors",
	"commit-subject-coverage",
	"complain",
	"completion",
	"compute",
	"component",
	// Storage-compute arbitration readout (internal/computetune Decide):
	// fak's own storage/decode-regime diagnostics, not a portable adopter pattern.
	"computetune",
	"concept",
	"concept-usage-score",
	"config",
	"conflation-scorecard",
	"conformance",
	"conpty",
	"contract",
	"console",
	"coverage-matrix",
	"cron",
	"ctxplans",
	"cubicquanteval",
	"customization-index",
	"debug",
	"debt-lanes",
	"decode-anatomy",
	"deepseekbench",
	"demo",
	"disambiguation",
	"dispatch-aging",
	"dispatch-conservation",
	"dispatchlat",
	"doc-freshness",
	"docs-freshness",
	"dogfood-coverage",
	"dogfood-issues",
	"dogfood-score",
	"dojo",
	"dojo-rsi",
	"doomloop",
	"dormancy",
	"dream",
	"dup",
	"edit-tx",
	"egresslist",
	"enroll",
	"eve",
	"execution-route",
	"experiments",
	"fanout",
	"feature",
	"footprint",
	"fresh-status",
	"frontierswe",
	"fused",
	"garden",
	"gate",
	"geminicache",
	"gh-spam-comments",
	"git-daily",
	"git-maint",
	"gitd",
	"glm52-prefill-sweep",
	// Verified sync host<->device round-trips/token census for the GLM-DSA
	// decode path: recomputable metric + source-drift gate over fak's own
	// kernel seams, not a portable adopter pattern.
	"glue-census",
	"go",
	"goal",
	"goal-park",
	// Ratcheting god-file LOC ceiling gate over fak's own tree (debtwave2 lane):
	// implementation maintains/operates fak itself, not a portable adopter pattern.
	"godfile-ceiling",
	"godsplit-plan",
	"grafana",
	// Presence-vs-invokability audit of CUDA/Vulkan graph capture/replay on the
	// small-batch decode paths: recomputable coverage table + source-drift gate
	// over fak's own engine seams, not a portable adopter pattern.
	"graph-audit",
	"growthgate",
	"guard-audit",
	"guard-goal-question",
	"guard-rsi-scorecard",
	"guard-stops",
	"guard-stops-slack",
	"guard-verdict-rsi",
	// Per-token host->device copy inventory on the decode state paths
	// (sampler/mask/KV-append metadata): recomputable copies/token + bytes/token
	// + source-drift gate over fak's own engine seams, not a portable adopter
	// pattern.
	"h2d-inventory",
	"harness",
	"harness-debt-dispatch",
	"headless-lint",
	"headroom",
	"hooklat",
	"horizon-recovery",
	"host-crash",
	"host-relaunch-broker",
	// Cross-receipt host-overhead regression trend gate: deterministic latest-vs-
	// baseline delta + regime-flip verdict over the same committed receipts,
	// fak's own decode-regime diagnostics, not a portable adopter pattern.
	"hostregress",
	// Systematic host-share reader + launch-bound gate over perf receipts: fak's own
	// decode-regime diagnostics, not a portable adopter pattern.
	"hostshare",
	// Per-layer host-setup vs kernel microbench at batch 1 on the compute Planner
	// seam (PrepareMatMul vs Plan.Run): direct setup/kernel ratio + regression gate,
	// fak's own decode-regime diagnostics, not a portable adopter pattern.
	"hostsetup-bench",
	"hwgate-lint",
	// Cited hardware-spec inventory over fak's own accelerator corpus: fak-maintainer
	// fleet diagnostics (the nativeperf/optimize evidence base), not a portable adopter pattern.
	"hwspecs",
	"hygiene",
	"idea-scout",
	"idempotency",
	"index",
	"init",
	"intent",
	"issue-contract-repair",
	"issue-lanes",
	"issue-orchestrator",
	"issue-queue",
	"issue-stats",
	"issues-recent",
	"issues-solved",
	"kernel-oracle",
	"knownbad",
	"kvbm",
	"l3-serve",
	"l3serve",
	"launch",
	"launch-census",
	// Read-only tensor-layout contract registry CLI over internal/compute/layoutcontract:
	// port workers and CI verify a kernel's expected packed/tile layout before a port
	// silently pays a runtime repack; fak-maintainer machinery, not a portable adopter pattern.
	"layoutcontract",
	// The concept-coverage drift watcher (Go port of the retired tools/concept_drift_watcher.py):
	// checks whether the concept-disambiguation scorecard's coverage debt re-opened since the
	// recorded baseline; fak-maintainer corpus machinery, not a portable adopter pattern.
	"conceptdrift",
	"leak-audit",
	"learning-debt-dispatch",
	"learning-observation",
	"lifecycle",
	"leaseref",
	"lint",
	"llm-d-smoke",
	"llmd-smoke",
	"llms-full",
	"logvault",
	"loop",
	"loop-index-scorecard",
	"loop-map",
	"loop-score",
	"lsp",
	"macfit",
	"macobs",
	"marketing",
	"maturity",
	"mcp-filter-proof",
	"memgate",
	"memory",
	"memory-read",
	"memory-stability-governor",
	"merge",
	"micro",
	"microbench",
	"milestone",
	"milestone-scorecard",
	"mlp-score",
	"mode-debt-dispatch",
	"model-default",
	"model-observe",
	"multisubmit",
	"native-benchmarks",
	"native-first-lint",
	"native-performance",
	"negate",
	"new-leaf",
	"new-model",
	"news",
	"operator",
	"opencode",
	"ops",
	"opt",
	// The dedicated inference-optimization driver (plan/measure/verify/loop over the
	// committed nativeperf hill-climb graph): fak's own optimization loop, not a
	// portable adopter pattern.
	"optimize",
	"org",
	"orient",
	"perf",
	"perfscout",
	"pack",
	"papers",
	"pi",
	"plan-audit",
	"popularization-tickets",
	"process-guard",
	"product",
	"product-scorecard",
	"profile",
	"program",
	"progress",
	"provenance",
	"propagation-debt-dispatch",
	"propagation-scorecard",
	"provider-cost",
	"public-scrub",
	"qa-process-debt-dispatch",
	"queue",
	"quality",
	"quantbench",
	"quantwatch",
	"question-ledger",
	"qwen36-node-reports",
	"qwen36-parity-witness-gate",
	"readme-visual-audit",
	"recall",
	"refactor-verify",
	// Multi-signal decode-step overhead-regime classifier over committed perf
	// receipts: fak's own decode-regime diagnostics, not a portable adopter pattern.
	"regime",
	"relay",
	"release",
	"release-lock",
	"release-staleness",
	"rename-concept",
	"repo-hygiene-scorecard",
	"resume",
	"rollup",
	"route",
	"routebench",
	"router",
	// Bounded reader for a serve's go_runtime receipt: fak's own runtime diagnostics.
	"runtime-mem",
	"rungstats",
	"savings",
	"savings-vector",
	"schedscan",
	"schedule-held",
	"search",
	"score",
	"scoreboard",
	"scorecard",
	"scratch",
	"scratch-janitor",
	"serve-wiring",
	"service",
	"session-audit",
	"sessiondiag",
	"sessionjournal",
	"sessions",
	"shadowgit",
	"sidecar",
	"signals",
	"skill",
	"skill-effectiveness-scorecard",
	"slack",
	"snapshot",
	"sota",
	"sota-coverage-scorecard",
	"speed-ab",
	"spend",
	"stale-work",
	"stallscan",
	"steer",
	"steering",
	"stopfailure",
	"storage-pressure",
	"study-forge",
	"study-adjacency",
	"inference-topology",
	"study-classify",
	"study-link",
	"study-priority",
	"study-tickets",
	"study-inventory",
	"study-monitor",
	"subagent-fanout-bench",
	"superloop",
	"superstream",
	// The deterministic supervisor seat's dispatch-side wake (#13464 S3): one
	// dry-run assemble-decide-gate-journal cycle over payload-free typed
	// witnesses. It operates fak's own fleet/dispatch machinery, so it is
	// maintainer surface, not a portable adopter pattern.
	"supervisor-seat",
	"support",
	"support-maturity-scorecard",
	"swebench",
	"sync",
	"temp-artifacts",
	"terminal-relief",
	"test",
	"ticket",
	"tickets",
	"test-quality",
	"tier-calibrate",
	"token-defaults-scorecard",
	"token-profile",
	"tool-width",
	"toolproc",
	"traj",
	"trajctl",
	"trajectory",
	"trajquery",
	"tree-doctor",
	"trunk-build-probe",
	"trunk-red",
	"turnavoid",
	"turntax",
	"ui-quality-scorecard",
	"unwired-debt-dispatch",
	"unwired-scorecard",
	"usage",
	"vcache",
	"waiting",
	"watchdog",
	"webbench",
	"whats-changed",
	"wiki",
	"windowgate",
	"windows-setup",
	"wip",
	"work-delivery",
	"workflow",
	"workflow-audit",
	"workpattern",
	"workspin",
	"worktype",
	"agentic",
	"compute-trace",
	"expert-trace",
	"coordinate",
	"hostdiag",
	"learning-mesh",
	"launchguard",
	"performance-rsi-scorecard",
	"runtime-capabilities",
	"shellprov",
	"up",
	"watchdog-audit-health",
	"watchdog-audit-run",
	"agentx",
	"client",
	"conversation-profile",
	"debt-orchestrator",
	"factory-migrate",
	"factorymigrate",
	"goal-fleet",
	"goal-launch",
	"harness-hint",
	"harnesshint",
	"managedocs",
	"observer",
	"readme",
	"readmenext",
	"redteam",
	"results",
	"sensecheck",
	"sessionsearch",
	"sweep-speculative",
}

var labDevCommands = map[string]string{
	"amd-gpu-facts":      "hardware-lab inventory command",
	"amd-gotchas":        "hardware-lab audit command",
	"amd-gpudirect":      "hardware-lab transport command",
	"amd-setup":          "hardware-lab setup command",
	"amd-sim":            "hardware-lab simulation command",
	"amd-strix-package":  "hardware-lab appliance packaging command",
	"amd-strix-probe":    "hardware-lab appliance probe command",
	"amd-strix-validate": "hardware-lab appliance validation command",
	"amdgpu":             "hardware-lab AMD GPU operations and audit command",
	"build-cuda":         "hardware-lab CUDA compute build orchestration",
	"build-vulkan":       "hardware-lab Vulkan compute build orchestration",
	"claude-mac-fak":     "fak lab-host control command",
	"cluster":            "fak compute-fleet operation",
	"containment":        "hardware-lab APU containment and resource arbitration",
	"cuda-build":         "hardware-lab CUDA compute build orchestration",
	"cuda-gpudirect":     "hardware-lab transport command",
	"fleet":              "fak compute-fleet operation",
	"fleet-accounts":     "fak compute-fleet account operation",
	"fleet-trend":        "fak compute-fleet telemetry",
	"fleetcap":           "fak compute-fleet capacity operation",
	"hil":                "hardware-in-the-loop micro-dose execution and comparison audit",
	"lab":                "fak private-lab operation",
	"macbench":           "fak lab-host benchmark operation",
	"node":               "fak compute-node operation",
	"node-compare":       "fak compute-node comparison",
	"nodeusage":          "fak compute-node telemetry",
	"nightrun":           "fak lab/fleet scheduled operation",
	"roofline":           "hardware-lab RDNA roofline and WMMA microprobe",
	"strix":              "hardware-lab appliance operation (strix-bench/status trigger)",
	"strix-probe":        "hardware-lab appliance probe command",
	"strix-validate":     "hardware-lab appliance validation command",
	"vulkan-build":       "hardware-lab Vulkan compute build orchestration",
}

// ClassifyDevReuse returns a total reuse classification. Portable means that
// the concept is suitable for examples and dogfood; it does not make the
// current repository-bound command a stable adopter API.
func ClassifyDevReuse(name string, owner CommandOwner) (DevReuse, string) {
	if owner != OwnerDev {
		return DevReuseNA, "runtime ownership is outside the development-reuse axis"
	}
	if rationale, ok := portableDevPatterns[name]; ok {
		return DevReusePortable, rationale
	}
	if rationale, ok := labDevCommands[name]; ok {
		return DevReuseLab, rationale
	}
	for _, candidate := range maintainerDevCommands {
		if name == candidate {
			return DevReuseMaintainer, "implementation maintains, measures, releases, or operates fak itself"
		}
	}
	return DevReuseUnclassified, "new dev command must be added to exactly one reuse registry in internal/devindex/devreuse.go"
}

// ValidateDevReuseRegistry proves that a spelling is declared exactly once.
// It runs as part of ownership validation so adding a TierDev command without
// making the reuse decision reds the same exhaustive gate.
func ValidateDevReuseRegistry() []string {
	return validateDevReuseRegistry(portableDevPatterns, maintainerDevCommands, labDevCommands)
}

func validateDevReuseRegistry(portable map[string]string, maintainer []string, lab map[string]string) []string {
	seen := make(map[string]DevReuse, len(portable)+len(maintainer)+len(lab))
	var problems []string
	add := func(name string, class DevReuse, rationale string) {
		if name == "" {
			problems = append(problems, "dev reuse registry contains an empty command name")
			return
		}
		if rationale == "" {
			problems = append(problems, "dev reuse registry has empty rationale: "+name)
		}
		if prior, ok := seen[name]; ok {
			problems = append(problems, fmt.Sprintf("dev reuse registry classifies %q more than once (%s and %s)", name, prior, class))
			return
		}
		seen[name] = class
	}
	for name, rationale := range portable {
		add(name, DevReusePortable, rationale)
	}
	for _, name := range maintainer {
		add(name, DevReuseMaintainer, "explicit fak-maintainer classification")
	}
	for name, rationale := range lab {
		add(name, DevReuseLab, rationale)
	}
	sort.Strings(problems)
	return problems
}
