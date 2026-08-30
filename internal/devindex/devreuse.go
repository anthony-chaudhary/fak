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
	"agents",
	"ailuminate",
	"answer-shape",
	"antipattern-scorecard",
	"api-host",
	"architecture",
	"armbench",
	"assume",
	"backend",
	"balance",
	"bench",
	"bench-ingest",
	"bench-loop",
	"bench-runs",
	"benchmarks",
	"bgloop",
	"blast",
	"borrow-provenance",
	"boundary",
	"breath",
	"budget",
	"c",
	"cache",
	"cachesweep",
	"cachevalue",
	"cadence",
	"callavoid",
	"capabilities",
	"catchup",
	"chat",
	"chatops",
	"chatrelay",
	"check-tool-failure",
	"checkpoint",
	"checkpoint-debt-dispatch",
	"checkpoint-scorecard",
	"claim-check",
	"clean-bins",
	"codelint",
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
	"component",
	"concept",
	"concept-usage-score",
	"config",
	"conflation-scorecard",
	"conformance",
	"conpty",
	"console",
	"coverage-matrix",
	"cron",
	"customization-index",
	"debug",
	"deepseekbench",
	"demo",
	"disambiguation",
	"dispatch-aging",
	"dispatch-conservation",
	"dispatchlat",
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
	"frontierswe",
	"fused",
	"garden",
	"gh-spam-comments",
	"git-daily",
	"git-maint",
	"gitd",
	"glm52-prefill-sweep",
	"go",
	"goal",
	"goal-park",
	"godsplit-plan",
	"grafana",
	"growthgate",
	"guard-audit",
	"guard-goal-question",
	"guard-rsi-scorecard",
	"guard-stops",
	"guard-stops-slack",
	"guard-verdict-rsi",
	"harness",
	"harness-debt-dispatch",
	"headless-lint",
	"headroom",
	"hooklat",
	"horizon-recovery",
	"host-crash",
	"host-relaunch-broker",
	"hwgate-lint",
	"hygiene",
	"idea-scout",
	"idempotency",
	"index",
	"init",
	"intent",
	"issue-contract-repair",
	"knownbad",
	"kvbm",
	"launch",
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
	"macfit",
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
	"opt",
	"org",
	"orient",
	"plan-audit",
	"popularization-tickets",
	"process-guard",
	"product",
	"product-scorecard",
	"profile",
	"program",
	"progress",
	"propagation-debt-dispatch",
	"propagation-scorecard",
	"provider-cost",
	"public-scrub",
	"qa-process-debt-dispatch",
	"quality",
	"quantbench",
	"quantwatch",
	"question-ledger",
	"qwen36-node-reports",
	"qwen36-parity-witness-gate",
	"readme-visual-audit",
	"recall",
	"refactor-verify",
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
	"rungstats",
	"savings",
	"savings-vector",
	"schedscan",
	"schedule-held",
	"search",
	"score",
	"scoreboard",
	"scorecard",
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
	"study-forge",
	"study-adjacency",
	"study-classify",
	"study-link",
	"study-priority",
	"study-tickets",
	"study-inventory",
	"study-monitor",
	"superloop",
	"support",
	"support-maturity-scorecard",
	"swebench",
	"sync",
	"temp-artifacts",
	"terminal-relief",
	"test",
	"test-quality",
	"tier-calibrate",
	"token-defaults-scorecard",
	"token-profile",
	"tool-coverage-audit",
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
	"worktype",
	"agentic",
	"compute-trace",
	"coordinate",
	"hostdiag",
	"learning-mesh",
	"performance-rsi-scorecard",
	"runtime-capabilities",
	"shellprov",
	"up",
	"watchdog-audit-health",
	"watchdog-audit-run",
}

var labDevCommands = map[string]string{
	"amd-gpu-facts":  "hardware-lab inventory command",
	"claude-mac-fak": "fak lab-host control command",
	"cluster":        "fak compute-fleet operation",
	"fleet":          "fak compute-fleet operation",
	"fleet-accounts": "fak compute-fleet account operation",
	"fleet-trend":    "fak compute-fleet telemetry",
	"fleetcap":       "fak compute-fleet capacity operation",
	"lab":            "fak private-lab operation",
	"macbench":       "fak lab-host benchmark operation",
	"node":           "fak compute-node operation",
	"node-compare":   "fak compute-node comparison",
	"nodeusage":      "fak compute-node telemetry",
	"nightrun":       "fak lab/fleet scheduled operation",
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
