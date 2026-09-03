package devindex

// C1 of epic #2228 (#2230): the VERB-TIER classification as data. The `fak` CLI
// grew ~170 canonical top-level verbs, of which only ~two dozen are the PRODUCT
// (what an adopter/operator of the kernel touches); the rest are internal dev/
// fleet tooling. Until now that split lived only in help-text taste
// (cmd/fak/help.go overviewGroups) — presentation, not a fact. This table makes
// the tier a queryable fact with ONE home, so the `fak dev` namespace (C2), the
// tiered help (C3), and the eventual bare-spelling gate (C5) all read the same
// answer, and `fak index verbs` / the MCP mirror can report it.
//
// Design constraints, in order:
//   - EXHAUSTIVE-EXPLICIT, not default-tiered: every canonical dispatch verb is
//     named here, and the coverage witness (tiers_test.go) reds when a new
//     dispatch case lands unclassified. A new verb's tier is a conscious
//     decision at authoring time, never silent accretion — the exact ambiguity
//     epic #2228 exists to kill.
//   - COMPILED-IN: an installed binary outside a repo must still answer TierOf
//     (the C2 dispatcher and C5 gate run everywhere `fak` runs), so the table is
//     a Go literal, not a file the catalog loads. The map literal also makes a
//     duplicate key a compile error — "no verb in two tiers" by construction.
//   - Keyed by CANONICAL name (the manifest Name where curated); alias spellings
//     resolve through the manifest in TierOf. The multi-spelling verb llmd-smoke /
//     llm-d-smoke is now cataloged (verbs.go carries llm-d-smoke as an alias), so
//     TierOf canonicalizes either spelling; both keys below are kept for a direct,
//     manifest-free lookup and stay in agreement.
//
// The Landlock trampoline verb (`case guard.TrampolineVerb:`) is a non-literal
// case the mainDispatchVerbs scan never emits; it stays invisible here exactly as
// it is invisible to the freshness detector — an internal re-exec seam, not a verb.

import "strings"

// VerbTier is one of the three CLI concept tiers of epic #2228.
type VerbTier string

const (
	// TierFrontdoor — the product surface: what an adopter or operator of the
	// kernel touches. Stays small (the gate test holds the ceiling); listed by
	// the compact `fak help`.
	TierFrontdoor VerbTier = "frontdoor"
	// TierDev — internal dev/fleet tooling: repo-workflow verbs, scorecards,
	// Slack surfaces, benches, loop/dispatch plumbing. The `fak dev <verb>`
	// namespace (C2) is its canonical spelling.
	TierDev VerbTier = "dev"
	// TierHidden — internal re-exec/hook seams spawned by fak itself, never
	// typed by a person and never listed.
	TierHidden VerbTier = "hidden"
)

// verbTiers is the exhaustive tier classification, keyed by canonical verb name.
// Grouped frontdoor -> hidden -> dev; alphabetical inside each group for stable
// diffs. Ratified on #2230 from the epic #2228 proposal; anything debatable is
// DEV (the epic's premise: most verbs are). Companion spellings that are their
// own dispatch cases (replay/top/pull/ls beside run/ps/model) tier with their
// concept.
var verbTiers = map[string]VerbTier{
	// ---- frontdoor: the product (ceiling gated by TestFrontdoorTierStaysSmall) ----
	"ablate": TierFrontdoor,
	// `agent` is the README's headline proof (`fak agent --offline`) and the first
	// command a new evaluator is told to run, so it belongs in the visible front
	// door rather than under `fak dev` (#5464).
	"agent":        TierFrontdoor,
	"capabilities": TierFrontdoor,
	"attest":       TierFrontdoor,
	"audit":        TierFrontdoor,
	"value-chain":  TierFrontdoor,
	"codex":        TierFrontdoor,
	"doctor":       TierFrontdoor,
	"egress":       TierFrontdoor,
	"manage":       TierFrontdoor,
	"help":         TierFrontdoor,
	"info":         TierFrontdoor,
	"ls":           TierFrontdoor,
	"model":        TierFrontdoor,
	"policy":       TierFrontdoor,
	"preflight":    TierFrontdoor,
	"ps":           TierFrontdoor,
	"pull":         TierFrontdoor,
	"recover":      TierFrontdoor,
	"replay":       TierFrontdoor,
	"resume":       TierDev,
	"run":          TierFrontdoor,
	"self-update":  TierFrontdoor,
	"serve":        TierFrontdoor,
	"session":      TierFrontdoor,
	"signal":       TierFrontdoor,
	"study":        TierFrontdoor,
	"top":          TierFrontdoor,
	"version":      TierFrontdoor,

	// ---- hidden: internal re-exec/hook seams, never listed ----
	"ablate-arm":         TierHidden,
	"guard-commit-gate":  TierHidden,
	"guard-precompact":   TierHidden,
	"guard-sessionstart": TierHidden,
	"guard-stophook":     TierHidden,
	"hook":               TierHidden,

	// ---- dev: everything else — spelled `fak dev <verb>` once C2 lands ----
	"agents":                TierDev,
	"architecture":          TierDev,
	"borrow-provenance":     TierDev,
	"catchup":               TierDev,
	"checkpoint":            TierDev,
	"codex-hook-census":     TierDev,
	"codex-hook-gate":       TierDev,
	"codex-hook-profile":    TierDev,
	"codex-plugin-sync":     TierDev,
	"codex-resume":          TierDev,
	"codex-stop-acceptance": TierDev,
	"codex-tool-errors":     TierDev,
	"component":             TierDev,
	"config":                TierDev,
	"customization-index":   TierDev,
	"disambiguation":        TierDev,
	"dormancy":              TierDev,
	"fanout":                TierDev,
	"goal":                  TierDev,
	"guard-goal-question":   TierDev,
	"harness":               TierDev,
	"learning-observation":  TierDev,
	"lifecycle":             TierDev,
	"mcp-filter-proof":      TierDev,
	"model-default":         TierDev,
	"model-observe":         TierDev,
	"native-benchmarks":     TierDev,
	"native-first-lint":     TierDev,
	"native-performance":    TierDev,
	"org":                   TierDev,
	"progress":              TierDev,
	"provider-cost":         TierDev,
	"quantbench":            TierDev,
	"quantwatch":            TierDev,
	"schedule-held":         TierDev,
	"search":                TierDev,
	"scratch-janitor":       TierDev,
	"speed-ab":              TierDev,
	"stale-work":            TierDev,
	"study-inventory":       TierDev,
	"study-forge":           TierDev,
	"study-adjacency":       TierDev,
	"study-classify":        TierDev,
	"study-link":            TierDev,
	"study-monitor":         TierDev,
	"study-priority":        TierDev,
	"study-tickets":         TierDev,
	"temp-artifacts":        TierDev,
	"terminal-relief":       TierDev,
	"test-quality":          TierDev,
	"tool-width":            TierDev,
	"trajectory":            TierDev,
	"windows-setup":         TierDev,
	"work-delivery":         TierDev,
	"workpattern":           TierDev,
	"worktype":              TierDev,

	"accounts":                      TierDev,
	"affected":                      TierDev,
	"agent-queue":                   TierDev,
	"ailuminate":                    TierDev,
	"amd-gpu-facts":                 TierDev,
	"amd-setup":                     TierDev,
	"answer-shape":                  TierDev,
	"antipattern-scorecard":         TierDev,
	"api-host":                      TierDev,
	"armbench":                      TierDev,
	"assume":                        TierDev,
	"backend":                       TierDev,
	"balance":                       TierDev,
	"bench":                         TierDev,
	"bench-ingest":                  TierDev,
	"bench-loop":                    TierDev,
	"bench-runs":                    TierDev,
	"benchmarks":                    TierDev,
	"bgloop":                        TierDev,
	"blast":                         TierDev,
	"blockers":                      TierDev,
	"boundary":                      TierDev,
	"breath":                        TierDev,
	"budget":                        TierDev,
	"build":                         TierDev,
	"buildcheck":                    TierDev,
	"c":                             TierDev,
	"cachesweep":                    TierDev,
	"cachevalue":                    TierDev,
	"cadence":                       TierDev,
	"callavoid":                     TierDev,
	"chat":                          TierDev,
	"chatops":                       TierDev,
	"chatrelay":                     TierDev,
	"check-tool-failure":            TierDev,
	"checkpoint-debt-dispatch":      TierDev,
	"checkpoint-scorecard":          TierDev,
	"ci-preflight":                  TierDev,
	"validate":                      TierDev,
	"llms-full":                     TierDev,
	"claim-check":                   TierDev,
	"claude-mac-fak":                TierDev,
	"clean-bins":                    TierDev,
	"cluster":                       TierDev,
	"codelint":                      TierDev,
	"codex-mcp-health":              TierDev,
	"codex-memory":                  TierDev,
	"sessiondiag":                   TierDev,
	"commit":                        TierDev,
	"commit-subject-coverage":       TierDev,
	"complain":                      TierDev,
	"concept":                       TierDev,
	"concept-usage-score":           TierDev,
	"conflation-scorecard":          TierDev,
	"conformance":                   TierDev,
	"conpty":                        TierDev,
	"console":                       TierDev,
	"coverage-matrix":               TierDev,
	"cron":                          TierDev,
	"debug":                         TierDev,
	"deepseekbench":                 TierDev,
	"demo":                          TierDev,
	"dispatch":                      TierDev,
	"dispatch-aging":                TierDev,
	"dispatch-conservation":         TierDev,
	"dispatchlat":                   TierDev,
	"execution-route":               TierDev,
	"dogfood-issues":                TierDev,
	"dogfood-score":                 TierDev,
	"dojo":                          TierDev,
	"dojo-rsi":                      TierDev,
	"done":                          TierDev,
	"doomloop":                      TierDev,
	"dream":                         TierDev,
	"dup":                           TierDev,
	"edit-tx":                       TierDev,
	"egresslist":                    TierDev,
	"enroll":                        TierDev,
	"eve":                           TierDev,
	"experiments":                   TierDev,
	"feature":                       TierDev,
	"fleet":                         TierDev,
	"fleet-accounts":                TierDev,
	"fleet-trend":                   TierDev,
	"fleetcap":                      TierDev,
	"footprint":                     TierDev,
	"frontierswe":                   TierDev,
	"fused":                         TierDev,
	"garden":                        TierDev,
	"git-daily":                     TierDev,
	"git-maint":                     TierDev,
	"gitd":                          TierDev,
	"glm52-prefill-sweep":           TierDev,
	"go":                            TierDev,
	"goal-park":                     TierDev,
	"godsplit-plan":                 TierDev,
	"grafana":                       TierDev,
	"growthgate":                    TierDev,
	"guard-audit":                   TierDev,
	"guard-rsi-scorecard":           TierDev,
	"guard-stops":                   TierDev,
	"guard-stops-slack":             TierDev,
	"guard-verdict-rsi":             TierDev,
	"harness-debt-dispatch":         TierDev,
	"headless-lint":                 TierDev,
	"headroom":                      TierDev,
	"hooklat":                       TierDev,
	"hooks":                         TierDev,
	"horizon-recovery":              TierDev,
	"host-crash":                    TierDev,
	"host-relaunch-broker":          TierDev,
	"hwgate-lint":                   TierDev,
	"hygiene":                       TierDev,
	"idea-scout":                    TierDev,
	"idempotency":                   TierDev,
	"index":                         TierDev,
	"init":                          TierDev,
	"intent":                        TierDev,
	"issue":                         TierDev,
	"issue-contract-repair":         TierDev,
	"knownbad":                      TierDev,
	"kvbm":                          TierDev,
	"lab":                           TierDev,
	"launch":                        TierDev,
	"launchguard":                   TierDev,
	"learning-debt-dispatch":        TierDev,
	"leaseref":                      TierDev,
	"lint":                          TierDev,
	"llm-d-smoke":                   TierDev,
	"llmd-smoke":                    TierDev,
	"logvault":                      TierDev,
	"loop":                          TierDev,
	"loop-index-scorecard":          TierDev,
	"loop-map":                      TierDev,
	"loop-score":                    TierDev,
	"lsp":                           TierDev,
	"macbench":                      TierDev,
	"macfit":                        TierDev,
	"marketing":                     TierDev,
	"maturity":                      TierDev,
	"memgate":                       TierDev,
	"memory":                        TierDev,
	"memory-read":                   TierDev,
	"memory-stability-governor":     TierDev,
	"merge":                         TierDev,
	"micro":                         TierDev,
	"microbench":                    TierDev,
	"milestone":                     TierDev,
	"milestone-scorecard":           TierDev,
	"mlp-score":                     TierDev,
	"mode-debt-dispatch":            TierDev,
	"multisubmit":                   TierDev,
	"negate":                        TierDev,
	"new-leaf":                      TierDev,
	"new-model":                     TierDev,
	"news":                          TierDev,
	"nightrun":                      TierDev,
	"node":                          TierDev,
	"node-compare":                  TierDev,
	"nodeusage":                     TierDev,
	"opencode":                      TierDev,
	"operator":                      TierDev,
	"opt":                           TierDev,
	"orient":                        TierDev,
	"perfscout":                     TierDev,
	"plan-audit":                    TierDev,
	"popularization-tickets":        TierDev,
	"process-guard":                 TierDev,
	"product":                       TierDev,
	"product-scorecard":             TierDev,
	"profile":                       TierDev,
	"program":                       TierDev,
	"project":                       TierDev,
	"propagation-debt-dispatch":     TierDev,
	"propagation-scorecard":         TierDev,
	"public-scrub":                  TierDev,
	"qa-process-debt-dispatch":      TierDev,
	"quality":                       TierDev,
	"question-ledger":               TierDev,
	"qwen36-node-reports":           TierDev,
	"qwen36-parity-witness-gate":    TierDev,
	"readme-visual-audit":           TierDev,
	"recall":                        TierDev,
	"refactor-verify":               TierDev,
	"relay":                         TierDev,
	"release":                       TierDev,
	"release-lock":                  TierDev,
	"release-staleness":             TierDev,
	"rename-concept":                TierDev,
	"repo-hygiene-scorecard":        TierDev,
	"rollup":                        TierDev,
	"route":                         TierDev,
	"routebench":                    TierDev,
	"rungstats":                     TierDev,
	"savings":                       TierDev,
	"savings-vector":                TierDev,
	"schedscan":                     TierDev,
	"score":                         TierDev,
	"scoreboard":                    TierDev,
	"scorecard":                     TierDev,
	"serve-wiring":                  TierDev,
	"service":                       TierDev,
	"session-audit":                 TierDev,
	"sessionjournal":                TierDev,
	"sessions":                      TierDev,
	"shadowgit":                     TierDev,
	"sidecar":                       TierDev,
	"signals":                       TierDev,
	"skill":                         TierDev,
	"skill-effectiveness-scorecard": TierDev,
	"slack":                         TierDev,
	"snapshot":                      TierDev,
	"sota":                          TierDev,
	"sota-coverage-scorecard":       TierDev,
	"spend":                         TierDev,
	"stallscan":                     TierDev,
	"steer":                         TierDev,
	"steering":                      TierDev,
	"stopfailure":                   TierDev,
	"storage-pressure":              TierDev,
	"superloop":                     TierDev,
	"support":                       TierDev,
	"support-maturity-scorecard":    TierDev,
	"sweep":                         TierDev,
	"swebench":                      TierDev,
	"sync":                          TierDev,
	"task":                          TierDev,
	"tasks":                         TierDev,
	"test":                          TierDev,
	"tier-calibrate":                TierDev,
	"token-defaults-scorecard":      TierDev,
	"token-profile":                 TierDev,
	"tool-coverage-audit":           TierDev,
	"toolproc":                      TierDev,
	"traj":                          TierDev,
	"trajctl":                       TierDev,
	"trajquery":                     TierDev,
	"tree-doctor":                   TierDev,
	"trunk-build-probe":             TierDev,
	"trunk-red":                     TierDev,
	"turnavoid":                     TierDev,
	"turntax":                       TierDev,
	"ui-quality-scorecard":          TierDev,
	"unwired-debt-dispatch":         TierDev,
	"unwired-scorecard":             TierDev,
	"usage":                         TierDev,
	"vcache":                        TierDev,
	"waiting":                       TierDev,
	"watchdog":                      TierDev,
	"webbench":                      TierDev,
	"whats-changed":                 TierDev,
	"wiki":                          TierDev,
	"windowgate":                    TierDev,
	"wip":                           TierDev,
	"workflow":                      TierDev,
	"workflow-audit":                TierDev,
	"workspin":                      TierDev,
	"worktree":                      TierDev,
	"agentic":                       TierDev,
	"compute-trace":                 TierDev,
	"coordinate":                    TierDev,
	"hostdiag":                      TierDev,
	"learning-mesh":                 TierDev,
	"performance-rsi-scorecard":     TierDev,
	"runtime-capabilities":          TierDev,
	"shellprov":                     TierDev,
	"up":                            TierDev,
	"watchdog-audit-health":         TierDev,
	"watchdog-audit-run":            TierDev,
}

// TierOf resolves a verb token (canonical name OR any alias spelling, any case)
// to its tier. Alias spellings canonicalize through the curated verb manifest
// first (`-h` -> help, `benchloop` -> bench-loop); an uncataloged token falls
// back to a direct table lookup under its own spelling. ok=false means the token
// is not a classified verb — for a LIVE dispatch token that is exactly the drift
// tiers_test.go reds on, so callers may treat it as "unknown verb", not "dev by
// default". Package-level (no Catalog) because the answer must not require a
// readable repo.
func TierOf(name string) (VerbTier, bool) {
	n := strings.ToLower(strings.TrimSpace(name))
	if n == "" {
		return "", false
	}
	if v, ok := manifestVerbByName(n); ok {
		if t, ok := verbTiers[strings.ToLower(v.Name)]; ok {
			return t, true
		}
	}
	t, ok := verbTiers[n]
	return t, ok
}

// tierFor is the omitempty-friendly form Verbs() uses to stamp the field: the
// tier when classified, else the empty VerbTier (dropped from JSON) — e.g. a
// curated manifest entry whose verb is not yet dispatched.
func tierFor(name string) VerbTier {
	t, _ := TierOf(name)
	return t
}
