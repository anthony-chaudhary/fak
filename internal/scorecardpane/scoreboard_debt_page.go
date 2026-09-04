package scorecardpane

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

// ScoreboardDebtDocRel is the repository-relative path to the scoreboard debt summary document.
const ScoreboardDebtDocRel = "docs/scoreboard-debt.md"

// DebtCategory represents one scored debt dimension in the portfolio.
type DebtCategory struct {
	Key         string
	Label       string
	DebtKey     string
	Unit        string
	DefaultPin  int
	DefaultGrad int
	Tool        string
	DocLink     string
	Cluster     string
	Description string
}

// DebtCategories defines the full catalog of portfolio debt categories.
var DebtCategories = []DebtCategory{
	{
		Key:         "doc",
		Label:       "docs",
		DebtKey:     "doc_debt",
		Unit:        "Dead links, missing H1, structure/readability defects",
		DefaultPin:  3,
		DefaultGrad: 0,
		Tool:        "tools/docs_scorecard.py",
		DocLink:     "`tools/docs_scorecard.py`",
		Cluster:     "Documentation, freshness, and discoverability",
		Description: "Audits documentation for dead markdown links, stale version pins, missing H1 titles, and navigational dead ends.",
	},
	{
		Key:         "readme",
		Label:       "readme-freshness",
		DebtKey:     "readme_debt",
		Unit:        "Stale pins, broken quickstarts, front-page claims",
		DefaultPin:  0,
		DefaultGrad: 0,
		Tool:        "tools/readme_freshness_audit.py",
		DocLink:     "[README.md](../README.md)",
		Cluster:     "Documentation, freshness, and discoverability",
		Description: "Ensures README.md reflects current CLI syntax, working quickstart snippets, and verified benchmark citations.",
	},
	{
		Key:         "code",
		Label:       "code",
		DebtKey:     "code_debt",
		Unit:        "God-files (>1500 lines), god-functions (>200 lines), cycles",
		DefaultPin:  38,
		DefaultGrad: 2,
		Tool:        "tools/code_quality_scorecard.py",
		DocLink:     "[docs/CODE-QUALITY-SCORECARD.md](CODE-QUALITY-SCORECARD.md)",
		Cluster:     "Code quality and maintenance",
		Description: "Measured statically across the Go module. Flags architectural god-files (FILE_HARD_MAX=1500), god-functions (FUNC_HARD_MAX=200), cyclomatic complexity traps, and package circularity.",
	},
	{
		Key:         "appeal",
		Label:       "doc-appeal",
		DebtKey:     "appeal_debt",
		Unit:        "AI tells, em-dash floods, passive framing, buzzwords",
		DefaultPin:  0,
		DefaultGrad: 0,
		Tool:        "tools/doc_appeal_scorecard.py",
		DocLink:     "`tools/doc_appeal_scorecard.py`",
		Cluster:     "Documentation, freshness, and discoverability",
		Description: "Detects AI voice tells, em-dash floods exceeding line budgets, and technical jargon lacking plain-language glosses.",
	},
	{
		Key:         "seo",
		Label:       "seo",
		DebtKey:     "seo_debt",
		Unit:        "Missing title/desc front-matter, uncrawlable links",
		DefaultPin:  8,
		DefaultGrad: 1,
		Tool:        "fak score seo",
		DocLink:     "[docs/index.md](index.md)",
		Cluster:     "Documentation, freshness, and discoverability",
		Description: "Audits published pages for front-matter titles, descriptions, crawlable link paths, and structured JSON-LD schemas.",
	},
	{
		Key:         "demo",
		Label:       "demo-quality",
		DebtKey:     "demo_debt",
		Unit:        "Broken demo runs, missing prerequisites, failed output",
		DefaultPin:  0,
		DefaultGrad: 0,
		Tool:        "tools/demo_quality_scorecard.py",
		DocLink:     "[docs/DEMO-QUALITY-SCORECARD.md](DEMO-QUALITY-SCORECARD.md)",
		Cluster:     "Delivery flow, commits, and operations",
		Description: "Validates that runnable demonstration scripts execute cleanly without network access or unstated assumptions.",
	},
	{
		Key:         "robustness",
		Label:       "demo-robustness",
		DebtKey:     "robustness_debt",
		Unit:        "Demos failing under boundary inputs or odd environments",
		DefaultPin:  0,
		DefaultGrad: 0,
		Tool:        "tools/demo_robustness_scorecard.py",
		DocLink:     "[docs/DEMO-ROBUSTNESS-SCORECARD.md](DEMO-ROBUSTNESS-SCORECARD.md)",
		Cluster:     "Delivery flow, commits, and operations",
		Description: "Tests that sample apps survive malformed user input, abnormal flags, and missing environment variables.",
	},
	{
		Key:         "hygiene",
		Label:       "repo-hygiene",
		DebtKey:     "hygiene_debt",
		Unit:        "Duplicate dirs, misplaced notes, orphan pages",
		DefaultPin:  22,
		DefaultGrad: 2,
		Tool:        "tools/repo_hygiene_scorecard.py",
		DocLink:     "[docs/REPO-HYGIENE-SCORECARD.md](REPO-HYGIENE-SCORECARD.md)",
		Cluster:     "Documentation, freshness, and discoverability",
		Description: "Binds tree structure rules — flags duplicate directory names, misplaced dated notes, oversized markdown documents, and unindexed files.",
	},
	{
		Key:         "parity",
		Label:       "industry-parity",
		DebtKey:     "parity_debt",
		Unit:        "LLM serving feature parity gaps vs SOTA runtimes",
		DefaultPin:  0,
		DefaultGrad: 0,
		Tool:        "tools/industry_scorecard.py",
		DocLink:     "[docs/industry-scorecard/](industry-scorecard/README.md)",
		Cluster:     "Kernel architecture and compute SOTA",
		Description: "Tracks feature parity gaps against production engines (vLLM, SGLang, TensorRT-LLM, llama.cpp).",
	},
	{
		Key:         "sota",
		Label:       "sota-coverage",
		DebtKey:     "sota_debt",
		Unit:        "Compute operations lacking SOTA prior-art reference",
		DefaultPin:  0,
		DefaultGrad: 0,
		Tool:        "tools/sota_coverage_scorecard.py",
		DocLink:     "[docs/sota/](sota/README.md)",
		Cluster:     "Kernel architecture and compute SOTA",
		Description: "Ensures every compute kernel operation maps to a documented SOTA reference and comparative benchmark baseline.",
	},
	{
		Key:         "agent",
		Label:       "agent-readiness",
		DebtKey:     "friction_debt",
		Unit:        "Friction preventing agents discovering/adopting fak",
		DefaultPin:  12,
		DefaultGrad: 0,
		Tool:        "fak score agent-readiness",
		DocLink:     "[docs/AGENT-READINESS-SCORECARD.md](AGENT-READINESS-SCORECARD.md)",
		Cluster:     "Product, persona, and adoption funnel",
		Description: "Measures mechanical barriers preventing autonomous coding agents from discovering and driving fak.",
	},
	{
		Key:         "brittleness",
		Label:       "brittleness",
		DebtKey:     "brittleness_debt",
		Unit:        "Flaky tests, timing hazards, unpinned dependencies",
		DefaultPin:  0,
		DefaultGrad: 0,
		Tool:        "fak score brittleness",
		DocLink:     "`cmd/fak/brittleness.go`",
		Cluster:     "Code quality and maintenance",
		Description: "Identifies fragile test configurations, unpinned timeouts, and race hazards.",
	},
	{
		Key:         "product",
		Label:       "product",
		DebtKey:     "product_debt",
		Unit:        "Concepts lacking runnable end-to-end examples",
		DefaultPin:  2,
		DefaultGrad: 0,
		Tool:        "tools/product_scorecard.py",
		DocLink:     "[docs/product-scorecard/](product-scorecard/README.md)",
		Cluster:     "Product, persona, and adoption funnel",
		Description: "Validates that every named product concept is backed by working code, examples, and user documentation.",
	},
	{
		Key:         "persona",
		Label:       "persona",
		DebtKey:     "persona_debt",
		Unit:        "Unmet affordances across top-10 persona segments",
		DefaultPin:  0,
		DefaultGrad: 0,
		Tool:        "tools/persona_readiness_scorecard.py",
		DocLink:     "[docs/adoption/personas.md](adoption/personas.md)",
		Cluster:     "Product, persona, and adoption funnel",
		Description: "Audits whether landing affordances exist for all 10 key personas (from open-source developers to regulated enterprise operators).",
	},
	{
		Key:         "popularization",
		Label:       "popularization",
		DebtKey:     "popularization_debt",
		Unit:        "Leaks across 5-stage conversion funnel",
		DefaultPin:  0,
		DefaultGrad: 0,
		Tool:        "tools/popularization_readiness_scorecard.py",
		DocLink:     "[docs/popularization-scorecard/](popularization-scorecard/README.md)",
		Cluster:     "Product, persona, and adoption funnel",
		Description: "Tracks visitor conversion friction across the land, orient, trust, install, and act funnels.",
	},
	{
		Key:         "stability",
		Label:       "stability",
		DebtKey:     "stability_debt",
		Unit:        "Regressions, tail-wags, broken rollback/bisect paths",
		DefaultPin:  0,
		DefaultGrad: 0,
		Tool:        "tools/stability_scorecard.py",
		DocLink:     "[docs/STABILITY-SCORECARD.md](STABILITY-SCORECARD.md)",
		Cluster:     "Governance, safety, and guard systems",
		Description: "Monitors regression vectors, tail-wagging dependencies, and verified rollback procedures.",
	},
	{
		Key:         "slop",
		Label:       "code-slop",
		DebtKey:     "slop_debt",
		Unit:        "Clones, vacuous tests, dead code, comment cruft",
		DefaultPin:  358,
		DefaultGrad: 8,
		Tool:        "tools/code_slop_scorecard.py",
		DocLink:     "`tools/code_slop_scorecard.py`",
		Cluster:     "Code quality and maintenance",
		Description: "Detects code the compiler cannot see — identical copy-paste clones, dead unexported functions, vacuous tests that assert no invariants, and tautological comments.",
	},
	{
		Key:         "steer",
		Label:       "steerability",
		DebtKey:     "steerability_debt",
		Unit:        "Blast radius, coupling entropy, circular dependencies",
		DefaultPin:  0,
		DefaultGrad: 0,
		Tool:        "tools/steerability_scorecard.py",
		DocLink:     "[docs/STEERABILITY-SCORECARD.md](STEERABILITY-SCORECARD.md)",
		Cluster:     "Code quality and maintenance",
		Description: "Measures package blast radius, dependency coupling, and navigation drag to ensure codebase remains agile as it grows.",
	},
	{
		Key:         "conflation",
		Label:       "conflation",
		DebtKey:     "conflation_debt",
		Unit:        "Reporting provider-observed values as fak-witnessed",
		DefaultPin:  0,
		DefaultGrad: 0,
		Tool:        "fak score conflation",
		DocLink:     "[docs/CONFLATION-SCORECARD.md](CONFLATION-SCORECARD.md)",
		Cluster:     "Concept clarity and truth maintenance",
		Description: "Enforces truth maintenance by strictly distinguishing provider-observed metrics (OBSERVED) from fak-authored invariants (WITNESSED).",
	},
	{
		Key:         "ui_quality",
		Label:       "ui-quality",
		DebtKey:     "ui_quality_debt",
		Unit:        "Multibyte rune slicing, column shear, missing help",
		DefaultPin:  0,
		DefaultGrad: 0,
		Tool:        "fak score ui-quality",
		DocLink:     "[docs/UI-QUALITY-SCORECARD.md](UI-QUALITY-SCORECARD.md)",
		Cluster:     "Delivery flow, commits, and operations",
		Description: "Inspects TUI terminal surfaces for rune-safe string truncation, column padding alignment, and help overlays.",
	},
	{
		Key:         "disambiguation",
		Label:       "concept-disambiguation",
		DebtKey:     "disambiguation_debt",
		Unit:        "Overloaded symbol roots, terminology collisions",
		DefaultPin:  5,
		DefaultGrad: 0,
		Tool:        "tools/concept_disambiguation_scorecard.py",
		DocLink:     "[docs/concept-disambiguation-scorecard/](concept-disambiguation-scorecard/README.md)",
		Cluster:     "Concept clarity and truth maintenance",
		Description: "Prevents terminology collision across overloaded roots (such as attention cache vs prompt cache, or kernel reference monitor vs CUDA compute kernel).",
	},
	{
		Key:         "intent_literal",
		Label:       "intent-literal",
		DebtKey:     "intent_literal_debt",
		Unit:        "Divergence between test intent assertions and reality",
		DefaultPin:  6,
		DefaultGrad: 2,
		Tool:        "tools/intent_literal_scorecard.py",
		DocLink:     "[docs/intent-literal-scorecard/](intent-literal-scorecard/README.md)",
		Cluster:     "Concept clarity and truth maintenance",
		Description: "Flags divergences between human-stated intent in test names and the literal assertions evaluated.",
	},
	{
		Key:         "tokendefaults",
		Label:       "token-defaults",
		DebtKey:     "token_defaults_debt",
		Unit:        "High-value token savers disabled by default",
		DefaultPin:  0,
		DefaultGrad: 0,
		Tool:        "fak score token-defaults",
		DocLink:     "[docs/serving/token-defaults-scorecard.md](serving/token-defaults-scorecard.md)",
		Cluster:     "Governance, safety, and guard systems",
		Description: "Prevents high-value token-saving optimizers from remaining disabled or gated behind hidden flags.",
	},
	{
		Key:         "guard_rsi",
		Label:       "guard-rsi",
		DebtKey:     "guard_rsi_debt",
		Unit:        "Guard decisions lacking hash-chained audit trails",
		DefaultPin:  1,
		DefaultGrad: 1,
		Tool:        "fak score guard-rsi",
		DocLink:     "`cmd/fak/guardrsi.go`",
		Cluster:     "Governance, safety, and guard systems",
		Description: "Ensures every kernel guard policy decision is accompanied by structured, replayable explanations.",
	},
	{
		Key:         "guard_accuracy",
		Label:       "guard-accuracy",
		DebtKey:     "guard_accuracy_debt",
		Unit:        "False-positives and false-negatives in command classifier",
		DefaultPin:  0,
		DefaultGrad: 0,
		Tool:        "fak score guard-accuracy",
		DocLink:     "`cmd/fak/guard_accuracy.go`",
		Cluster:     "Governance, safety, and guard systems",
		Description: "Measures classifier accuracy on command escalation to prevent false blocks and safety bypasses.",
	},
	{
		Key:         "dogfood",
		Label:       "dogfood-loop",
		DebtKey:     "dogfood_debt",
		Unit:        "Omission of real binary/model execution in loop tests",
		DefaultPin:  1,
		DefaultGrad: 1,
		Tool:        "fak score dogfood",
		DocLink:     "[docs/fak/dogfood-loop-scorecard.md](fak/dogfood-loop-scorecard.md)",
		Cluster:     "Delivery flow, commits, and operations",
		Description: "Enforces live binary and model dogfooding in tests rather than mock-only verification.",
	},
	{
		Key:         "conceptusage",
		Label:       "concept-usage",
		DebtKey:     "conceptusage_debt",
		Unit:        "Internal development bypassing fak's own concepts",
		DefaultPin:  3,
		DefaultGrad: 8,
		Tool:        "fak score concept-usage",
		DocLink:     "[docs/fak/concept-usage-scorecard.md](fak/concept-usage-scorecard.md)",
		Cluster:     "Concept clarity and truth maintenance",
		Description: "Catches development that bypasses fak's core concepts (such as using raw shell scripts instead of guarded syscalls and leases).",
	},
	{
		Key:         "lightgap",
		Label:       "lightgap",
		DebtKey:     "lightgap_debt",
		Unit:        "Usability gaps vs next-best operator alternatives",
		DefaultPin:  0,
		DefaultGrad: 0,
		Tool:        "fak score lightgap",
		DocLink:     "[docs/lightgap-scorecard/](lightgap-scorecard/README.md)",
		Cluster:     "Product, persona, and adoption funnel",
		Description: "Measures competitive lightgap deficits and missing affordances compared to alternative stacks.",
	},
	{
		Key:         "maturity",
		Label:       "maturity",
		DebtKey:     "maturity_debt",
		Unit:        "Capability ladder skips (untested/unbenchmarked leaves)",
		DefaultPin:  1,
		DefaultGrad: 2,
		Tool:        "fak maturity --json",
		DocLink:     "[docs/MATURITY-SCORECARD.md](MATURITY-SCORECARD.md)",
		Cluster:     "Lifecycle, milestones, and release velocity",
		Description: "Enforces the capability lifecycle ladder (proposed -> prototyped -> tested -> dogfooded -> default), preventing untested production code.",
	},
	{
		Key:         "growth",
		Label:       "growth-debt",
		DebtKey:     "growth_debt",
		Unit:        "Unimplemented cells in model x platform matrix",
		DefaultPin:  0,
		DefaultGrad: 2,
		Tool:        "fak coverage-matrix",
		DocLink:     "[docs/coverage-matrix.md](coverage-matrix.md)",
		Cluster:     "Kernel architecture and compute SOTA",
		Description: "Identifies empty cells in the combinatorial capability x execution backend matrix.",
	},
	{
		Key:         "support_maturity",
		Label:       "support-maturity",
		DebtKey:     "support_maturity_debt",
		Unit:        "Model architecture x accelerator backend gaps",
		DefaultPin:  0,
		DefaultGrad: 0,
		Tool:        "fak score support-maturity",
		DocLink:     "[docs/HARDWARE-MATRIX.md](HARDWARE-MATRIX.md)",
		Cluster:     "Kernel architecture and compute SOTA",
		Description: "Evaluates hardware backend coverage across AMD, Apple Silicon, and NVIDIA accelerators.",
	},
	{
		Key:         "milestone",
		Label:       "milestone",
		DebtKey:     "milestone_debt",
		Unit:        "Distance to MATURED rungs across support grid",
		DefaultPin:  72,
		DefaultGrad: 2,
		Tool:        "fak score milestone",
		DocLink:     "[docs/milestones/STATUS.md](milestones/STATUS.md)",
		Cluster:     "Lifecycle, milestones, and release velocity",
		Description: "Measures distance-to-matured across the backend support grid and open milestone epics.",
	},
	{
		Key:         "milestone_climb",
		Label:       "milestone-climb",
		DebtKey:     "climb_ratchet_debt",
		Unit:        "Regressions in matured cells vs baseline pin",
		DefaultPin:  0,
		DefaultGrad: 0,
		Tool:        "fak score milestone --ratchet",
		DocLink:     "[docs/milestones/STATUS.md](milestones/STATUS.md)",
		Cluster:     "Lifecycle, milestones, and release velocity",
		Description: "Hard ratchet preventing any loss of matured cells across releases.",
	},
	{
		Key:         "loopindex",
		Label:       "loop-index",
		DebtKey:     "loopindex_debt",
		Unit:        "Broken stages in 6-stage autonomous coding loop",
		DefaultPin:  0,
		DefaultGrad: 0,
		Tool:        "fak score loop-index",
		DocLink:     "[docs/fak/loop-scorecard.md](fak/loop-scorecard.md)",
		Cluster:     "Lifecycle, milestones, and release velocity",
		Description: "Audits connectivity of the 6-stage autonomous coding loop (orient, plan, act, verify, ship, learn).",
	},
	{
		Key:         "heaviness",
		Label:       "operator-heaviness",
		DebtKey:     "heaviness_debt",
		Unit:        "Cognitive and operational burden on operators",
		DefaultPin:  0,
		DefaultGrad: 4,
		Tool:        "fak operator heaviness",
		DocLink:     "`internal/operatorheaviness`",
		Cluster:     "Delivery flow, commits, and operations",
		Description: "Quantifies operational complexity and manual flags required to drive the system.",
	},
	{
		Key:         "propagation",
		Label:       "propagation",
		DebtKey:     "propagation_debt",
		Unit:        "Cross-subsystem protocol lag and mirrored model drift",
		DefaultPin:  14,
		DefaultGrad: 0,
		Tool:        "fak propagation-scorecard",
		DocLink:     "`cmd/fak/propagation_scorecard.go`",
		Cluster:     "Delivery flow, commits, and operations",
		Description: "Identifies protocol drift and delayed propagation across mirrored interfaces.",
	},
	{
		Key:         "claim_repro",
		Label:       "claim-repro",
		DebtKey:     "claim_repro_debt",
		Unit:        "Unresolvable/unfalsifiable witness claims",
		DefaultPin:  0,
		DefaultGrad: 0,
		Tool:        "tools/claim_repro_scorecard.py",
		DocLink:     "[docs/CLAIM-REPRO-SCORECARD.md](CLAIM-REPRO-SCORECARD.md)",
		Cluster:     "Delivery flow, commits, and operations",
		Description: "Eliminates unfalsifiable witness claims from CLAIMS.md.",
	},
	{
		Key:         "release",
		Label:       "release-readiness",
		DebtKey:     "release_debt",
		Unit:        "Manual friction in cutting, signing, or rolling back",
		DefaultPin:  2,
		DefaultGrad: 1,
		Tool:        "tools/release_readiness_scorecard.py",
		DocLink:     "[docs/RELEASE-READINESS-SCORECARD.md](RELEASE-READINESS-SCORECARD.md)",
		Cluster:     "Lifecycle, milestones, and release velocity",
		Description: "Flags manual steps, unverified binaries, and missing rollback metadata in the release workflow.",
	},
	{
		Key:         "observability",
		Label:       "observability",
		DebtKey:     "observability_debt",
		Unit:        "Dashboards/alerts referencing non-existent metrics",
		DefaultPin:  0,
		DefaultGrad: 0,
		Tool:        "tools/observability_scorecard.py",
		DocLink:     "[docs/OBSERVABILITY-SCORECARD.md](OBSERVABILITY-SCORECARD.md)",
		Cluster:     "Delivery flow, commits, and operations",
		Description: "Ensures all metrics referenced in dashboards and alerts are actively exported by the binary.",
	},
	{
		Key:         "learning",
		Label:       "learning",
		DebtKey:     "learning_debt",
		Unit:        "Tutorials without worked output, orphan guides",
		DefaultPin:  3,
		DefaultGrad: 0,
		Tool:        "tools/learning_scorecard.py",
		DocLink:     "[docs/LEARNING-SCORECARD.md](LEARNING-SCORECARD.md)",
		Cluster:     "Documentation, freshness, and discoverability",
		Description: "Ensures guides and tutorials provide runnable commands with worked output examples, eliminating orphan lessons.",
	},
	{
		Key:         "rsi_maturity",
		Label:       "rsi-maturity",
		DebtKey:     "rsi_debt",
		Unit:        "Self-improvement loops lacking closed hash chains",
		DefaultPin:  0,
		DefaultGrad: 0,
		Tool:        "tools/rsi_maturity_scorecard.py",
		DocLink:     "`tools/rsi_maturity_scorecard.py`",
		Cluster:     "Delivery flow, commits, and operations",
		Description: "Validates that self-improvement loops close on real telemetry.",
	},
	{
		Key:         "tooling_quality",
		Label:       "tooling-quality",
		DebtKey:     "py_debt",
		Unit:        "Untyped maintenance scripts, unhandled subprocesses",
		DefaultPin:  35,
		DefaultGrad: 4,
		Tool:        "tools/tooling_quality_scorecard.py",
		DocLink:     "`tools/tooling_quality_scorecard.py`",
		Cluster:     "Code quality and maintenance",
		Description: "Measures maintenance scripts for type annotations, unhandled subprocess errors, and unquarantined external dependencies.",
	},
	{
		Key:         "bench_dx",
		Label:       "bench-dx",
		DebtKey:     "bench_dx_debt",
		Unit:        "Benchmark developer experience and fixture defects",
		DefaultPin:  5,
		DefaultGrad: 0,
		Tool:        "tools/bench_dx_scorecard.py",
		DocLink:     "`tools/bench_dx_scorecard.py`",
		Cluster:     "Delivery flow, commits, and operations",
		Description: "Measures benchmark runner developer experience, fixtures, and CLI workflows.",
	},
	{
		Key:         "cuda_dev",
		Label:       "cuda-dev",
		DebtKey:     "process_debt",
		Unit:        "CUDA compilation defects and missing GPU guards",
		DefaultPin:  0,
		DefaultGrad: 0,
		Tool:        "tools/cuda_dev_scorecard.py",
		DocLink:     "[docs/CUDA-DEV-SCORECARD.md](CUDA-DEV-SCORECARD.md)",
		Cluster:     "Kernel architecture and compute SOTA",
		Description: "Measures CUDA build reproducibility, environment isolation, and host fallback safety.",
	},
	{
		Key:         "persona_fit",
		Label:       "persona-fit",
		DebtKey:     "persona_fit_debt",
		Unit:        "Persona matrix grounding gaps for enterprise users",
		DefaultPin:  0,
		DefaultGrad: 0,
		Tool:        "tools/persona_fit_scorecard.py",
		DocLink:     "[docs/persona-fit-scorecard/](persona-fit-scorecard/README.md)",
		Cluster:     "Product, persona, and adoption funnel",
		Description: "Checks matrix-integrity and grounding for developer and enterprise user workflows.",
	},
	{
		Key:         "commit_subject",
		Label:       "commit-subject",
		DebtKey:     "commit_debt",
		Unit:        "Missing DCO, missing trailers, malformed subjects",
		DefaultPin:  13,
		DefaultGrad: 2,
		Tool:        "tools/commit_subject_coverage.py",
		DocLink:     "`tools/commit_subject_coverage.py`",
		Cluster:     "Delivery flow, commits, and operations",
		Description: "Audits git commit messages for Conventional Commits compliance, DCO sign-offs, and (fak <leaf>) trailers.",
	},
	{
		Key:         "flow",
		Label:       "flow-metrics",
		DebtKey:     "flow_debt",
		Unit:        "Tripped Little's Law delivery flow axes",
		DefaultPin:  6,
		DefaultGrad: 4,
		Tool:        "fak score flow",
		DocLink:     "`internal/flowmetrics`",
		Cluster:     "Delivery flow, commits, and operations",
		Description: "Tracks Little's Law flow defects including long queue times, untracked local WIP, and unstarted backlog items.",
	},
	{
		Key:         "osp_residual",
		Label:       "osp-residual",
		DebtKey:     "residual_count",
		Unit:        "Forming PR units awaiting operator review",
		DefaultPin:  0,
		DefaultGrad: 0,
		Tool:        "fak steer prs",
		DocLink:     "[docs/operator-steerability-prs.md](operator-steerability-prs.md)",
		Cluster:     "Delivery flow, commits, and operations",
		Description: "Tracks unreviewed or unwitnessed PR units in flight.",
	},
	{
		Key:         "antipattern",
		Label:       "antipattern",
		DebtKey:     "antipattern_debt",
		Unit:        "Detected anti-patterns across agent sessions",
		DefaultPin:  0,
		DefaultGrad: 0,
		Tool:        "fak antipattern-scorecard",
		DocLink:     "[docs/notes/AGENTIC-DEV-ANTIPATTERNS-2026-07-02.md](notes/AGENTIC-DEV-ANTIPATTERNS-2026-07-02.md)",
		Cluster:     "Code quality and maintenance",
		Description: "Scans active sessions for agentic anti-patterns such as unbounded scratch accumulation, unverified commits, and orphan tool processes.",
	},
	{
		Key:         "negframe",
		Label:       "negframe",
		DebtKey:     "negframe_debt",
		Unit:        "Prohibition-first instructions in agent steering prose",
		DefaultPin:  0,
		DefaultGrad: 0,
		Tool:        "fak score negframe",
		DocLink:     "`internal/negframe`",
		Cluster:     "Concept clarity and truth maintenance",
		Description: "Ensures steering instructions in AGENTS.md and skills lead with positive affordances rather than negative prohibitions.",
	},
}

func weightGrade(w int) string {
	switch w {
	case 0:
		return "A"
	case 1:
		return "B"
	case 2:
		return "C"
	case 4:
		return "D"
	default:
		return "F"
	}
}

// GenerateScoreboardDebtDoc renders the deterministic Markdown document body.
func GenerateScoreboardDebtDoc(b Baseline) string {
	var sb strings.Builder
	sb.WriteString("---\n")
	sb.WriteString("title: \"Scoreboard debt — portfolio debt summary across all categories\"\n")
	sb.WriteString("description: \"Unified unbounded summary of fak's scorecard portfolio debt across all categories, tracking mechanical defects, severity weights, and trends.\"\n")
	sb.WriteString("---\n\n")
	sb.WriteString("# Scoreboard debt — unbounded portfolio summary\n\n")
	sb.WriteString("Scoreboard debt is the portfolio-wide sum of discrete, reproducible defects tracked across fak's scorecards. It powers the central Slack `#scoreboard` feed, the CI ratchet gate, and the developer control pane (`fak scorecard control-pane` / `tools/scorecard_control_pane.py`).\n\n")
	sb.WriteString("Unlike bounded vanity scores (such as a 0–100 score that saturates or hides underlying flaws once thresholds pass), **debt in fak is strictly unbounded and counts concrete mechanical defects**. Each scorecard measures a distinct surface of the repository, calculates an exact `*_debt` integer, and enforces a single target: drive the debt to zero.\n\n")

	sb.WriteString("## The portfolio ratchet architecture\n\n")
	sb.WriteString("The scorecard control pane synthesizes individual scorecards into an enforceable, dual-axis ratchet:\n\n")
	sb.WriteString("1. **Total Debt (`total_debt`) — Raw Defect Census:** An observational heterogeneous sum of every raw debt integer across all registered scorecards. It answers \"how many concrete defect units remain in the repository today?\"\n")
	sb.WriteString("2. **Grade Debt (`grade_debt`) — Scale-Invariant Severity:** Because raw occurrence counters (such as code-slop or concept-disambiguation with hundreds of occurrences) could otherwise drown out regressions in bounded metrics (such as a single god-file in code quality), every metric is graded on a shared A–F ladder (`A=0`, `B=1`, `C=2`, `D=4`, `F=8`). Grade debt sums these severity weights, ensuring that an `A -> B` slip in stability or release readiness weighs just as heavily as an `A -> B` slip in code-slop.\n")
	sb.WriteString("3. **Early-Warning Lens:** If a metric regresses above its pinned baseline, the control pane raises an advisory warning even if total portfolio debt decreased due to improvements elsewhere.\n")
	sb.WriteString("4. **Index Coverage Gate:** Every scorecard implementation in `tools/*_scorecard.py` and Go native scorecards must be registered in the control pane or explicitly excluded with a documented rationale. Unindexed scorecards fail CI.\n")
	sb.WriteString("5. **Deterministic Maintenance:** This document is generated and verified deterministically to prevent documentation rot against the live scorecard suite.\n\n")

	sb.WriteString("## Debt categories overview\n\n")
	sb.WriteString(fmt.Sprintf("The portfolio tracks %d distinct debt categories across code quality, documentation, concepts, adoption, kernel compute, governance, and operational flow:\n\n", len(DebtCategories)))
	sb.WriteString("| Metric | Debt Key | Defect Unit / Unbounded Driver | Baseline Pin | Severity Weight | Owning Command / Tool | Reference Doc |\n")
	sb.WriteString("|---|---|---|---|---|---|---|\n")

	for _, c := range DebtCategories {
		pin := c.DefaultPin
		if b.Metrics != nil {
			if v, ok := b.Metrics[c.Key]; ok {
				pin = v
			}
		}
		grad := c.DefaultGrad
		if b.GradeWeights != nil {
			if w, ok := b.GradeWeights[c.Key]; ok {
				grad = w
			}
		}
		letter := weightGrade(grad)
		sb.WriteString(fmt.Sprintf("| `%s` | `%s` | %s | %d | %d (%s) | `%s` | %s |\n",
			c.Label, c.DebtKey, c.Unit, pin, grad, letter, c.Tool, c.DocLink))
	}
	sb.WriteString("\n")

	sb.WriteString("## Debt category details\n\n")
	sb.WriteString("### 1. Code quality and maintenance\n\n")
	sb.WriteString("- **Code Quality (`code_debt`):** Measured statically across the Go module. Flags architectural god-files (`FILE_HARD_MAX=1500`), god-functions (`FUNC_HARD_MAX=200`), cyclomatic complexity traps, and package circularity.\n")
	sb.WriteString("- **Code Slop (`slop_debt`):** Detects code the compiler cannot see — identical copy-paste clones, dead unexported functions, vacuous tests that assert no invariants, and tautological comments.\n")
	sb.WriteString("- **Tooling Quality (`py_debt`):** Measures maintenance scripts for type annotations, unhandled subprocess errors, and unquarantined external dependencies.\n")
	sb.WriteString("- **Brittleness (`brittleness_debt`):** Identifies fragile test configurations, unpinned timeouts, and race hazards.\n")
	sb.WriteString("- **Anti-patterns (`antipattern_debt`):** Scans active sessions for agentic anti-patterns such as unbounded scratch accumulation, unverified commits, and orphan tool processes.\n\n")

	sb.WriteString("### 2. Documentation, freshness, and discoverability\n\n")
	sb.WriteString("- **Core Documentation (`doc_debt`):** Audits documentation for dead markdown links, stale version pins, missing H1 titles, and navigational dead ends.\n")
	sb.WriteString("- **README Freshness (`readme_debt`):** Ensures `README.md` reflects current CLI syntax, working quickstart snippets, and verified benchmark citations.\n")
	sb.WriteString("- **Doc Appeal (`appeal_debt`):** Detects AI voice tells, em-dash floods exceeding line budgets, and technical jargon lacking plain-language glosses.\n")
	sb.WriteString("- **SEO & AEO Discoverability (`seo_debt`):** Audits published pages for front-matter titles, descriptions, crawlable link paths, and structured JSON-LD schemas.\n")
	sb.WriteString("- **Learning & Pedagogy (`learning_debt`):** Ensures guides and tutorials provide runnable commands with worked output examples, eliminating orphan lessons.\n")
	sb.WriteString("- **Repository Hygiene (`hygiene_debt`):** Binds tree structure rules — flags duplicate directory names, misplaced dated notes, oversized markdown documents, and unindexed files.\n\n")

	sb.WriteString("### 3. Concept clarity and truth maintenance\n\n")
	sb.WriteString("- **Concept Disambiguation (`disambiguation_debt`):** Prevents terminology collision across overloaded roots (such as attention cache vs prompt cache, or kernel reference monitor vs CUDA compute kernel).\n")
	sb.WriteString("- **Conflation (`conflation_debt`):** Enforces truth maintenance by strictly distinguishing provider-observed metrics (`OBSERVED`) from fak-authored invariants (`WITNESSED`).\n")
	sb.WriteString("- **Concept Usage (`conceptusage_debt`):** Catches development that bypasses fak's core concepts (such as using raw shell scripts instead of guarded syscalls and leases).\n")
	sb.WriteString("- **Intent Literal (`intent_literal_debt`):** Flags divergences between human-stated intent in test names and the literal assertions evaluated.\n")
	sb.WriteString("- **Negative Framing (`negframe_debt`):** Ensures steering instructions in `AGENTS.md` and skills lead with positive affordances rather than negative prohibitions.\n\n")

	sb.WriteString("### 4. Product, persona, and adoption funnel\n\n")
	sb.WriteString("- **Product Readiness (`product_debt`):** Validates that every named product concept is backed by working code, examples, and user documentation.\n")
	sb.WriteString("- **Persona Readiness (`persona_debt`):** Audits whether landing affordances exist for all 10 key personas (from open-source developers to regulated enterprise operators).\n")
	sb.WriteString("- **Persona Fit (`persona_fit_debt`):** Checks matrix-integrity and grounding for developer and enterprise user workflows.\n")
	sb.WriteString("- **Popularization (`popularization_debt`):** Tracks visitor conversion friction across the land, orient, trust, install, and act funnels.\n")
	sb.WriteString("- **Lightgap (`lightgap_debt`):** Measures competitive lightgap deficits and missing affordances compared to alternative stacks.\n")
	sb.WriteString("- **Agent Readiness (`friction_debt`):** Measures mechanical barriers preventing autonomous coding agents from discovering and driving fak.\n\n")

	sb.WriteString("### 5. Kernel architecture and compute SOTA\n\n")
	sb.WriteString("- **SOTA Coverage (`sota_debt`):** Ensures every compute kernel operation maps to a documented SOTA reference and comparative benchmark baseline.\n")
	sb.WriteString("- **Industry Parity (`parity_debt`):** Tracks feature parity gaps against production engines (vLLM, SGLang, TensorRT-LLM, llama.cpp).\n")
	sb.WriteString("- **CUDA Development (`process_debt`):** Measures CUDA build reproducibility, environment isolation, and host fallback safety.\n")
	sb.WriteString("- **Support Maturity (`support_maturity_debt`):** Evaluates hardware backend coverage across AMD, Apple Silicon, and NVIDIA accelerators.\n")
	sb.WriteString("- **Growth Debt (`growth_debt`):** Identifies empty cells in the combinatorial capability x execution backend matrix.\n\n")

	sb.WriteString("### 6. Governance, safety, and guard systems\n\n")
	sb.WriteString("- **Token Defaults (`token_defaults_debt`):** Prevents high-value token-saving optimizers from remaining disabled or gated behind hidden flags.\n")
	sb.WriteString("- **Guard RSI (`guard_rsi_debt`):** Ensures every kernel guard policy decision is accompanied by structured, replayable explanations.\n")
	sb.WriteString("- **Guard Accuracy (`guard_accuracy_debt`):** Measures classifier accuracy on command escalation to prevent false blocks and safety bypasses.\n")
	sb.WriteString("- **Stability (`stability_debt`):** Monitors regression vectors, tail-wagging dependencies, and verified rollback procedures.\n\n")

	sb.WriteString("### 7. Lifecycle, milestones, and release velocity\n\n")
	sb.WriteString("- **Maturity (`maturity_debt`):** Enforces the capability lifecycle ladder (`proposed -> prototyped -> tested -> dogfooded -> default`), preventing untested production code.\n")
	sb.WriteString("- **Milestones (`milestone_debt`):** Measures distance-to-matured across the backend support grid and open milestone epics.\n")
	sb.WriteString("- **Milestone Climb (`climb_ratchet_debt`):** Hard ratchet preventing any loss of matured cells across releases.\n")
	sb.WriteString("- **Loop Index (`loopindex_debt`):** Audits connectivity of the 6-stage autonomous coding loop (orient, plan, act, verify, ship, learn).\n")
	sb.WriteString("- **Release Readiness (`release_debt`):** Flags manual steps, unverified binaries, and missing rollback metadata in the release workflow.\n\n")

	sb.WriteString("### 8. Delivery flow, commits, and operations\n\n")
	sb.WriteString("- **Flow Metrics (`flow_debt`):** Tracks Little's Law flow defects including long queue times, untracked local WIP, and unstarted backlog items.\n")
	sb.WriteString("- **Commit Subjects (`commit_debt`):** Audits git commit messages for Conventional Commits compliance, DCO sign-offs, and `(fak <leaf>)` trailers.\n")
	sb.WriteString("- **Operator Heaviness (`heaviness_debt`):** Quantifies operational complexity and manual flags required to drive the system.\n")
	sb.WriteString("- **OSP Residual (`residual_count`):** Tracks unreviewed or unwitnessed PR units in flight.\n")
	sb.WriteString("- **Propagation (`propagation_debt`):** Identifies protocol drift and delayed propagation across mirrored interfaces.\n")
	sb.WriteString("- **Claim Reproducibility (`claim_repro_debt`):** Eliminates unfalsifiable witness claims from `CLAIMS.md`.\n")
	sb.WriteString("- **Observability (`observability_debt`):** Ensures all metrics referenced in dashboards and alerts are actively exported by the binary.\n")
	sb.WriteString("- **RSI Maturity (`rsi_debt`):** Validates that self-improvement loops close on real telemetry.\n")
	sb.WriteString("- **Dogfood Loop (`dogfood_debt`):** Enforces live binary and model dogfooding in tests rather than mock-only verification.\n\n")

	sb.WriteString("## Deterministic maintenance and verification\n\n")
	sb.WriteString("To ensure this document never drifts from the live scorecard registry, it is governed by deterministic tooling:\n\n")
	sb.WriteString("- **Regenerate in place:**\n")
	sb.WriteString("  ```bash\n")
	sb.WriteString("  go run ./cmd/fak scoreboard debt-page --write-doc\n")
	sb.WriteString("  # or\n")
	sb.WriteString("  python tools/scorecard_control_pane.py --write-doc\n")
	sb.WriteString("  ```\n\n")
	sb.WriteString("- **Verify freshness in CI:**\n")
	sb.WriteString("  ```bash\n")
	sb.WriteString("  go run ./cmd/fak scoreboard debt-page --check-doc\n")
	sb.WriteString("  # or\n")
	sb.WriteString("  python tools/scorecard_control_pane.py --check-doc\n")
	sb.WriteString("  ```\n\n")
	sb.WriteString("A non-zero exit from `--check-doc` indicates that new scorecards or baseline adjustments have been made without updating this reference page.\n")

	return sb.String()
}

// NormalizeDocLineEndings ensures uniform LF line endings.
func NormalizeDocLineEndings(s string) string {
	return strings.ReplaceAll(s, "\r\n", "\n")
}

// FreshScoreboardDebtDoc reports whether the doc string matches the expected output.
func FreshScoreboardDebtDoc(doc string, b Baseline) bool {
	expected := GenerateScoreboardDebtDoc(b)
	return NormalizeDocLineEndings(doc) == NormalizeDocLineEndings(expected)
}

func findRepoRoot() string {
	dir, err := os.Getwd()
	if err != nil {
		return "."
	}
	for {
		if _, err := os.Stat(filepath.Join(dir, "go.mod")); err == nil {
			return dir
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			return "."
		}
		dir = parent
	}
}

// WriteScoreboardDebtDoc writes the generated document to docPath.
func WriteScoreboardDebtDoc(workspace string) (bool, error) {
	root := workspace
	if root == "" {
		root = findRepoRoot()
	}
	baselinePath := filepath.Join(root, filepath.FromSlash(BaselineRel))
	baseline := LoadBaseline(baselinePath)
	var bl Baseline
	if baseline != nil {
		bl = *baseline
	}

	docPath := filepath.Join(root, filepath.FromSlash(ScoreboardDebtDocRel))
	content := GenerateScoreboardDebtDoc(bl)

	existing, err := os.ReadFile(docPath)
	if err == nil && NormalizeDocLineEndings(string(existing)) == NormalizeDocLineEndings(content) {
		return false, nil // already fresh
	}

	if err := os.MkdirAll(filepath.Dir(docPath), 0o755); err != nil {
		return false, err
	}
	if err := os.WriteFile(docPath, []byte(NormalizeDocLineEndings(content)), 0o644); err != nil {
		return false, err
	}
	return true, nil
}

// CheckScoreboardDebtDoc verifies freshness of the scoreboard debt page.
func CheckScoreboardDebtDoc(workspace string) (bool, string, error) {
	root := workspace
	if root == "" {
		root = findRepoRoot()
	}
	baselinePath := filepath.Join(root, filepath.FromSlash(BaselineRel))
	baseline := LoadBaseline(baselinePath)
	var bl Baseline
	if baseline != nil {
		bl = *baseline
	}

	docPath := filepath.Join(root, filepath.FromSlash(ScoreboardDebtDocRel))
	raw, err := os.ReadFile(docPath)
	if err != nil {
		return false, fmt.Sprintf("STALE  %s: file not found; run `fak scoreboard debt-page --write-doc`", ScoreboardDebtDocRel), nil
	}
	if !FreshScoreboardDebtDoc(string(raw), bl) {
		return false, fmt.Sprintf("STALE  %s: drifted from scorecard baseline; run `fak scoreboard debt-page --write-doc`", ScoreboardDebtDocRel), nil
	}
	return true, fmt.Sprintf("OK  %s: fresh (%d debt categories verified)", ScoreboardDebtDocRel, len(DebtCategories)), nil
}
