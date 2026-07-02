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
//     resolve through the manifest in TierOf. The one uncataloged multi-spelling
//     verb (llmd-smoke / llm-d-smoke) carries both spellings explicitly.
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
	"attest":      TierFrontdoor,
	"audit":       TierFrontdoor,
	"codex":       TierFrontdoor,
	"doctor":      TierFrontdoor,
	"egress":      TierFrontdoor,
	"guard":       TierFrontdoor,
	"help":        TierFrontdoor,
	"info":        TierFrontdoor,
	"ls":          TierFrontdoor,
	"model":       TierFrontdoor,
	"policy":      TierFrontdoor,
	"preflight":   TierFrontdoor,
	"ps":          TierFrontdoor,
	"pull":        TierFrontdoor,
	"recover":     TierFrontdoor,
	"replay":      TierFrontdoor,
	"resume":      TierFrontdoor,
	"run":         TierFrontdoor,
	"self-update": TierFrontdoor,
	"serve":       TierFrontdoor,
	"session":     TierFrontdoor,
	"signal":      TierFrontdoor,
	"top":         TierFrontdoor,
	"version":     TierFrontdoor,

	// ---- hidden: internal re-exec/hook seams, never listed ----
	"ablate-arm":       TierHidden,
	"guard-precompact": TierHidden,
	"guard-stophook":   TierHidden,
	"hook":             TierHidden,

	// ---- dev: everything else — spelled `fak dev <verb>` once C2 lands ----
	"ablate":                        TierDev,
	"accounts":                      TierDev,
	"affected":                      TierDev,
	"agent":                         TierDev,
	"ailuminate":                    TierDev,
	"amd-gpu-facts":                 TierDev,
	"answer-shape":                  TierDev,
	"api-host":                      TierDev,
	"bench":                         TierDev,
	"bench-loop":                    TierDev,
	"bench-runs":                    TierDev,
	"benchmarks":                    TierDev,
	"bgloop":                        TierDev,
	"blockers":                      TierDev,
	"c":                             TierDev,
	"cachevalue":                    TierDev,
	"cadence":                       TierDev,
	"callavoid":                     TierDev,
	"chat":                          TierDev,
	"chatrelay":                     TierDev,
	"check-tool-failure":            TierDev,
	"claim-check":                   TierDev,
	"claude-mac-fak":                TierDev,
	"cluster":                       TierDev,
	"codelint":                      TierDev,
	"codex-mcp-health":              TierDev,
	"codex-memory":                  TierDev,
	"commit":                        TierDev,
	"commit-subject-coverage":       TierDev,
	"complain":                      TierDev,
	"concept-usage-score":           TierDev,
	"conflation-scorecard":          TierDev,
	"console":                       TierDev,
	"coverage-matrix":               TierDev,
	"cron":                          TierDev,
	"debug":                         TierDev,
	"dispatch":                      TierDev,
	"dogfood-issues":                TierDev,
	"dogfood-score":                 TierDev,
	"dojo":                          TierDev,
	"dojo-rsi":                      TierDev,
	"done":                          TierDev,
	"dream":                         TierDev,
	"edit-tx":                       TierDev,
	"experiments":                   TierDev,
	"feature":                       TierDev,
	"fleet":                         TierDev,
	"fleet-accounts":                TierDev,
	"fleet-trend":                   TierDev,
	"fleetcap":                      TierDev,
	"frontierswe":                   TierDev,
	"fused":                         TierDev,
	"garden":                        TierDev,
	"grafana":                       TierDev,
	"guard-rsi-scorecard":           TierDev,
	"guard-verdict-rsi":             TierDev,
	"headroom":                      TierDev,
	"hooklat":                       TierDev,
	"hooks":                         TierDev,
	"horizon-recovery":              TierDev,
	"hygiene":                       TierDev,
	"index":                         TierDev,
	"intent":                        TierDev,
	"issue":                         TierDev,
	"issue-contract-repair":         TierDev,
	"lab":                           TierDev,
	"learning-debt-dispatch":        TierDev,
	"leaseref":                      TierDev,
	"lint":                          TierDev,
	"llm-d-smoke":                   TierDev,
	"llmd-smoke":                    TierDev,
	"loop":                          TierDev,
	"loop-index-scorecard":          TierDev,
	"loop-map":                      TierDev,
	"loop-score":                    TierDev,
	"marketing":                     TierDev,
	"maturity":                      TierDev,
	"memgate":                       TierDev,
	"memory":                        TierDev,
	"memory-read":                   TierDev,
	"memory-stability-governor":     TierDev,
	"merge":                         TierDev,
	"milestone":                     TierDev,
	"milestone-scorecard":           TierDev,
	"new-leaf":                      TierDev,
	"new-model":                     TierDev,
	"news":                          TierDev,
	"nightrun":                      TierDev,
	"node":                          TierDev,
	"node-compare":                  TierDev,
	"nodeusage":                     TierDev,
	"operator":                      TierDev,
	"opt":                           TierDev,
	"orient":                        TierDev,
	"plan-audit":                    TierDev,
	"popularization-tickets":        TierDev,
	"process-guard":                 TierDev,
	"product":                       TierDev,
	"product-scorecard":             TierDev,
	"profile":                       TierDev,
	"program":                       TierDev,
	"propagation-debt-dispatch":     TierDev,
	"propagation-scorecard":         TierDev,
	"public-scrub":                  TierDev,
	"qwen36-node-reports":           TierDev,
	"qwen36-parity-witness-gate":    TierDev,
	"readme-visual-audit":           TierDev,
	"recall":                        TierDev,
	"relay":                         TierDev,
	"release":                       TierDev,
	"release-lock":                  TierDev,
	"release-staleness":             TierDev,
	"repo-hygiene-scorecard":        TierDev,
	"rollup":                        TierDev,
	"route":                         TierDev,
	"routebench":                    TierDev,
	"rungstats":                     TierDev,
	"savings-vector":                TierDev,
	"scoreboard":                    TierDev,
	"scorecard":                     TierDev,
	"serve-wiring":                  TierDev,
	"session-audit":                 TierDev,
	"sessions":                      TierDev,
	"skill-effectiveness-scorecard": TierDev,
	"slack":                         TierDev,
	"snapshot":                      TierDev,
	"sota":                          TierDev,
	"sota-coverage-scorecard":       TierDev,
	"steering":                      TierDev,
	"stopfailure":                   TierDev,
	"superloop":                     TierDev,
	"support":                       TierDev,
	"support-maturity-scorecard":    TierDev,
	"sweep":                         TierDev,
	"swebench":                      TierDev,
	"sync":                          TierDev,
	"task":                          TierDev,
	"test":                          TierDev,
	"token-defaults-scorecard":      TierDev,
	"tool-coverage-audit":           TierDev,
	"toolproc":                      TierDev,
	"traj":                          TierDev,
	"tree-doctor":                   TierDev,
	"turntax":                       TierDev,
	"ui-quality-scorecard":          TierDev,
	"usage":                         TierDev,
	"vcache":                        TierDev,
	"webbench":                      TierDev,
	"whats-changed":                 TierDev,
	"windowgate":                    TierDev,
	"workflow":                      TierDev,
	"workflow-audit":                TierDev,
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
