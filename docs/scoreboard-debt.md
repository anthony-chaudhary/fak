---
title: "Scoreboard debt — portfolio debt summary across all categories"
description: "Unified unbounded summary of fak's scorecard portfolio debt across all categories, tracking mechanical defects, severity weights, and trends."
---

# Scoreboard debt — unbounded portfolio summary

Scoreboard debt is the portfolio-wide sum of discrete, reproducible defects tracked across fak's scorecards. It powers the central Slack `#scoreboard` feed, the CI ratchet gate, and the developer control pane (`fak scorecard control-pane` / `tools/scorecard_control_pane.py`).

Unlike bounded vanity scores (such as a 0–100 score that saturates or hides underlying flaws once thresholds pass), **debt in fak is strictly unbounded and counts concrete mechanical defects**. Each scorecard measures a distinct surface of the repository, calculates an exact `*_debt` integer, and enforces a single target: drive the debt to zero.

## The portfolio ratchet architecture

The scorecard control pane synthesizes individual scorecards into an enforceable, dual-axis ratchet:

1. **Total Debt (`total_debt`) — Raw Defect Census:** An observational heterogeneous sum of every raw debt integer across all registered scorecards. It answers "how many concrete defect units remain in the repository today?"
2. **Grade Debt (`grade_debt`) — Scale-Invariant Severity:** Because raw occurrence counters (such as code-slop or concept-disambiguation with hundreds of occurrences) could otherwise drown out regressions in bounded metrics (such as a single god-file in code quality), every metric is graded on a shared A–F ladder (`A=0`, `B=1`, `C=2`, `D=4`, `F=8`). Grade debt sums these severity weights, ensuring that an `A -> B` slip in stability or release readiness weighs just as heavily as an `A -> B` slip in code-slop.
3. **Early-Warning Lens:** If a metric regresses above its pinned baseline, the control pane raises an advisory warning even if total portfolio debt decreased due to improvements elsewhere.
4. **Index Coverage Gate:** Every scorecard implementation in `tools/*_scorecard.py` and Go native scorecards must be registered in the control pane or explicitly excluded with a documented rationale. Unindexed scorecards fail CI.
5. **Deterministic Maintenance:** This document is generated and verified deterministically to prevent documentation rot against the live scorecard suite.

## Debt categories overview

The portfolio tracks 50 distinct debt categories across code quality, documentation, concepts, adoption, kernel compute, governance, and operational flow:

| Metric | Debt Key | Defect Unit / Unbounded Driver | Baseline Pin | Severity Weight | Owning Command / Tool | Reference Doc |
|---|---|---|---|---|---|---|
| `docs` | `doc_debt` | Dead links, missing H1, structure/readability defects | 3 | 0 (A) | `tools/docs_scorecard.py` | `tools/docs_scorecard.py` |
| `readme-freshness` | `readme_debt` | Stale pins, broken quickstarts, front-page claims | 0 | 0 (A) | `tools/readme_freshness_audit.py` | [README.md](../README.md) |
| `code` | `code_debt` | God-files (>1500 lines), god-functions (>200 lines), cycles | 38 | 2 (C) | `tools/code_quality_scorecard.py` | [docs/CODE-QUALITY-SCORECARD.md](CODE-QUALITY-SCORECARD.md) |
| `doc-appeal` | `appeal_debt` | AI tells, em-dash floods, passive framing, buzzwords | 0 | 0 (A) | `tools/doc_appeal_scorecard.py` | `tools/doc_appeal_scorecard.py` |
| `seo` | `seo_debt` | Missing title/desc front-matter, uncrawlable links | 8 | 1 (B) | `fak score seo` | [docs/index.md](index.md) |
| `demo-quality` | `demo_debt` | Broken demo runs, missing prerequisites, failed output | 0 | 0 (A) | `tools/demo_quality_scorecard.py` | [docs/DEMO-QUALITY-SCORECARD.md](DEMO-QUALITY-SCORECARD.md) |
| `demo-robustness` | `robustness_debt` | Demos failing under boundary inputs or odd environments | 0 | 0 (A) | `tools/demo_robustness_scorecard.py` | [docs/DEMO-ROBUSTNESS-SCORECARD.md](DEMO-ROBUSTNESS-SCORECARD.md) |
| `repo-hygiene` | `hygiene_debt` | Duplicate dirs, misplaced notes, orphan pages | 22 | 2 (C) | `tools/repo_hygiene_scorecard.py` | [docs/REPO-HYGIENE-SCORECARD.md](REPO-HYGIENE-SCORECARD.md) |
| `industry-parity` | `parity_debt` | LLM serving feature parity gaps vs SOTA runtimes | 0 | 0 (A) | `tools/industry_scorecard.py` | [docs/industry-scorecard/](industry-scorecard/README.md) |
| `sota-coverage` | `sota_debt` | Compute operations lacking SOTA prior-art reference | 0 | 0 (A) | `tools/sota_coverage_scorecard.py` | [docs/sota/](sota/README.md) |
| `agent-readiness` | `friction_debt` | Friction preventing agents discovering/adopting fak | 12 | 0 (A) | `fak score agent-readiness` | [docs/AGENT-READINESS-SCORECARD.md](AGENT-READINESS-SCORECARD.md) |
| `brittleness` | `brittleness_debt` | Flaky tests, timing hazards, unpinned dependencies | 0 | 0 (A) | `fak score brittleness` | `cmd/fak/brittleness.go` |
| `product` | `product_debt` | Concepts lacking runnable end-to-end examples | 2 | 0 (A) | `tools/product_scorecard.py` | [docs/product-scorecard/](product-scorecard/README.md) |
| `persona` | `persona_debt` | Unmet affordances across top-10 persona segments | 0 | 0 (A) | `tools/persona_readiness_scorecard.py` | [docs/adoption/personas.md](adoption/personas.md) |
| `popularization` | `popularization_debt` | Leaks across 5-stage conversion funnel | 0 | 0 (A) | `tools/popularization_readiness_scorecard.py` | [docs/popularization-scorecard/](popularization-scorecard/README.md) |
| `stability` | `stability_debt` | Regressions, tail-wags, broken rollback/bisect paths | 0 | 0 (A) | `tools/stability_scorecard.py` | [docs/STABILITY-SCORECARD.md](STABILITY-SCORECARD.md) |
| `code-slop` | `slop_debt` | Clones, vacuous tests, dead code, comment cruft | 358 | 8 (F) | `tools/code_slop_scorecard.py` | `tools/code_slop_scorecard.py` |
| `steerability` | `steerability_debt` | Blast radius, coupling entropy, circular dependencies | 0 | 0 (A) | `tools/steerability_scorecard.py` | [docs/STEERABILITY-SCORECARD.md](STEERABILITY-SCORECARD.md) |
| `conflation` | `conflation_debt` | Reporting provider-observed values as fak-witnessed | 0 | 0 (A) | `fak score conflation` | [docs/CONFLATION-SCORECARD.md](CONFLATION-SCORECARD.md) |
| `ui-quality` | `ui_quality_debt` | Multibyte rune slicing, column shear, missing help | 0 | 0 (A) | `fak score ui-quality` | [docs/UI-QUALITY-SCORECARD.md](UI-QUALITY-SCORECARD.md) |
| `concept-disambiguation` | `disambiguation_debt` | Overloaded symbol roots, terminology collisions | 5 | 0 (A) | `tools/concept_disambiguation_scorecard.py` | [docs/concept-disambiguation-scorecard/](concept-disambiguation-scorecard/README.md) |
| `intent-literal` | `intent_literal_debt` | Divergence between test intent assertions and reality | 6 | 2 (C) | `tools/intent_literal_scorecard.py` | [docs/intent-literal-scorecard/](intent-literal-scorecard/README.md) |
| `token-defaults` | `token_defaults_debt` | High-value token savers disabled by default | 0 | 0 (A) | `fak score token-defaults` | [docs/serving/token-defaults-scorecard.md](serving/token-defaults-scorecard.md) |
| `guard-rsi` | `guard_rsi_debt` | Guard decisions lacking hash-chained audit trails | 1 | 1 (B) | `fak score guard-rsi` | `cmd/fak/guardrsi.go` |
| `guard-accuracy` | `guard_accuracy_debt` | False-positives and false-negatives in command classifier | 0 | 0 (A) | `fak score guard-accuracy` | `cmd/fak/guard_accuracy.go` |
| `dogfood-loop` | `dogfood_debt` | Omission of real binary/model execution in loop tests | 1 | 1 (B) | `fak score dogfood` | [docs/fak/dogfood-loop-scorecard.md](fak/dogfood-loop-scorecard.md) |
| `concept-usage` | `conceptusage_debt` | Internal development bypassing fak's own concepts | 3 | 8 (F) | `fak score concept-usage` | [docs/fak/concept-usage-scorecard.md](fak/concept-usage-scorecard.md) |
| `lightgap` | `lightgap_debt` | Usability gaps vs next-best operator alternatives | 0 | 0 (A) | `fak score lightgap` | [docs/lightgap-scorecard/](lightgap-scorecard/README.md) |
| `maturity` | `maturity_debt` | Capability ladder skips (untested/unbenchmarked leaves) | 1 | 2 (C) | `fak maturity --json` | [docs/MATURITY-SCORECARD.md](MATURITY-SCORECARD.md) |
| `growth-debt` | `growth_debt` | Unimplemented cells in model x platform matrix | 0 | 2 (C) | `fak coverage-matrix` | [docs/coverage-matrix.md](coverage-matrix.md) |
| `support-maturity` | `support_maturity_debt` | Model architecture x accelerator backend gaps | 0 | 0 (A) | `fak score support-maturity` | [docs/HARDWARE-MATRIX.md](HARDWARE-MATRIX.md) |
| `milestone` | `milestone_debt` | Distance to MATURED rungs across support grid | 72 | 2 (C) | `fak score milestone` | [docs/milestones/STATUS.md](milestones/STATUS.md) |
| `milestone-climb` | `climb_ratchet_debt` | Regressions in matured cells vs baseline pin | 0 | 0 (A) | `fak score milestone --ratchet` | [docs/milestones/STATUS.md](milestones/STATUS.md) |
| `loop-index` | `loopindex_debt` | Broken stages in 6-stage autonomous coding loop | 0 | 0 (A) | `fak score loop-index` | [docs/fak/loop-scorecard.md](fak/loop-scorecard.md) |
| `operator-heaviness` | `heaviness_debt` | Cognitive and operational burden on operators | 0 | 4 (D) | `fak operator heaviness` | `internal/operatorheaviness` |
| `propagation` | `propagation_debt` | Cross-subsystem protocol lag and mirrored model drift | 14 | 0 (A) | `fak propagation-scorecard` | `cmd/fak/propagation_scorecard.go` |
| `claim-repro` | `claim_repro_debt` | Unresolvable/unfalsifiable witness claims | 0 | 0 (A) | `tools/claim_repro_scorecard.py` | [docs/CLAIM-REPRO-SCORECARD.md](CLAIM-REPRO-SCORECARD.md) |
| `release-readiness` | `release_debt` | Manual friction in cutting, signing, or rolling back | 2 | 1 (B) | `tools/release_readiness_scorecard.py` | [docs/RELEASE-READINESS-SCORECARD.md](RELEASE-READINESS-SCORECARD.md) |
| `observability` | `observability_debt` | Dashboards/alerts referencing non-existent metrics | 0 | 0 (A) | `tools/observability_scorecard.py` | [docs/OBSERVABILITY-SCORECARD.md](OBSERVABILITY-SCORECARD.md) |
| `learning` | `learning_debt` | Tutorials without worked output, orphan guides | 3 | 0 (A) | `tools/learning_scorecard.py` | [docs/LEARNING-SCORECARD.md](LEARNING-SCORECARD.md) |
| `rsi-maturity` | `rsi_debt` | Self-improvement loops lacking closed hash chains | 0 | 0 (A) | `tools/rsi_maturity_scorecard.py` | `tools/rsi_maturity_scorecard.py` |
| `tooling-quality` | `py_debt` | Untyped maintenance scripts, unhandled subprocesses | 35 | 4 (D) | `tools/tooling_quality_scorecard.py` | `tools/tooling_quality_scorecard.py` |
| `bench-dx` | `bench_dx_debt` | Benchmark developer experience and fixture defects | 5 | 0 (A) | `tools/bench_dx_scorecard.py` | `tools/bench_dx_scorecard.py` |
| `cuda-dev` | `process_debt` | CUDA compilation defects and missing GPU guards | 0 | 0 (A) | `tools/cuda_dev_scorecard.py` | [docs/CUDA-DEV-SCORECARD.md](CUDA-DEV-SCORECARD.md) |
| `persona-fit` | `persona_fit_debt` | Persona matrix grounding gaps for enterprise users | 0 | 0 (A) | `tools/persona_fit_scorecard.py` | [docs/persona-fit-scorecard/](persona-fit-scorecard/README.md) |
| `commit-subject` | `commit_debt` | Missing DCO, missing trailers, malformed subjects | 13 | 2 (C) | `tools/commit_subject_coverage.py` | `tools/commit_subject_coverage.py` |
| `flow-metrics` | `flow_debt` | Tripped Little's Law delivery flow axes | 6 | 4 (D) | `fak score flow` | `internal/flowmetrics` |
| `osp-residual` | `residual_count` | Forming PR units awaiting operator review | 0 | 0 (A) | `fak steer prs` | [docs/operator-steerability-prs.md](operator-steerability-prs.md) |
| `antipattern` | `antipattern_debt` | Detected anti-patterns across agent sessions | 0 | 0 (A) | `fak antipattern-scorecard` | [docs/notes/AGENTIC-DEV-ANTIPATTERNS-2026-07-02.md](notes/AGENTIC-DEV-ANTIPATTERNS-2026-07-02.md) |
| `negframe` | `negframe_debt` | Prohibition-first instructions in agent steering prose | 0 | 0 (A) | `fak score negframe` | `internal/negframe` |

## Debt category details

### 1. Code quality and maintenance

- **Code Quality (`code_debt`):** Measured statically across the Go module. Flags architectural god-files (`FILE_HARD_MAX=1500`), god-functions (`FUNC_HARD_MAX=200`), cyclomatic complexity traps, and package circularity.
- **Code Slop (`slop_debt`):** Detects code the compiler cannot see — identical copy-paste clones, dead unexported functions, vacuous tests that assert no invariants, and tautological comments.
- **Tooling Quality (`py_debt`):** Measures maintenance scripts for type annotations, unhandled subprocess errors, and unquarantined external dependencies.
- **Brittleness (`brittleness_debt`):** Identifies fragile test configurations, unpinned timeouts, and race hazards.
- **Anti-patterns (`antipattern_debt`):** Scans active sessions for agentic anti-patterns such as unbounded scratch accumulation, unverified commits, and orphan tool processes.

### 2. Documentation, freshness, and discoverability

- **Core Documentation (`doc_debt`):** Audits documentation for dead markdown links, stale version pins, missing H1 titles, and navigational dead ends.
- **README Freshness (`readme_debt`):** Ensures `README.md` reflects current CLI syntax, working quickstart snippets, and verified benchmark citations.
- **Doc Appeal (`appeal_debt`):** Detects AI voice tells, em-dash floods exceeding line budgets, and technical jargon lacking plain-language glosses.
- **SEO & AEO Discoverability (`seo_debt`):** Audits published pages for front-matter titles, descriptions, crawlable link paths, and structured JSON-LD schemas.
- **Learning & Pedagogy (`learning_debt`):** Ensures guides and tutorials provide runnable commands with worked output examples, eliminating orphan lessons.
- **Repository Hygiene (`hygiene_debt`):** Binds tree structure rules — flags duplicate directory names, misplaced dated notes, oversized markdown documents, and unindexed files.

### 3. Concept clarity and truth maintenance

- **Concept Disambiguation (`disambiguation_debt`):** Prevents terminology collision across overloaded roots (such as attention cache vs prompt cache, or kernel reference monitor vs CUDA compute kernel).
- **Conflation (`conflation_debt`):** Enforces truth maintenance by strictly distinguishing provider-observed metrics (`OBSERVED`) from fak-authored invariants (`WITNESSED`).
- **Concept Usage (`conceptusage_debt`):** Catches development that bypasses fak's core concepts (such as using raw shell scripts instead of guarded syscalls and leases).
- **Intent Literal (`intent_literal_debt`):** Flags divergences between human-stated intent in test names and the literal assertions evaluated.
- **Negative Framing (`negframe_debt`):** Ensures steering instructions in `AGENTS.md` and skills lead with positive affordances rather than negative prohibitions.

### 4. Product, persona, and adoption funnel

- **Product Readiness (`product_debt`):** Validates that every named product concept is backed by working code, examples, and user documentation.
- **Persona Readiness (`persona_debt`):** Audits whether landing affordances exist for all 10 key personas (from open-source developers to regulated enterprise operators).
- **Persona Fit (`persona_fit_debt`):** Checks matrix-integrity and grounding for developer and enterprise user workflows.
- **Popularization (`popularization_debt`):** Tracks visitor conversion friction across the land, orient, trust, install, and act funnels.
- **Lightgap (`lightgap_debt`):** Measures competitive lightgap deficits and missing affordances compared to alternative stacks.
- **Agent Readiness (`friction_debt`):** Measures mechanical barriers preventing autonomous coding agents from discovering and driving fak.

### 5. Kernel architecture and compute SOTA

- **SOTA Coverage (`sota_debt`):** Ensures every compute kernel operation maps to a documented SOTA reference and comparative benchmark baseline.
- **Industry Parity (`parity_debt`):** Tracks feature parity gaps against production engines (vLLM, SGLang, TensorRT-LLM, llama.cpp).
- **CUDA Development (`process_debt`):** Measures CUDA build reproducibility, environment isolation, and host fallback safety.
- **Support Maturity (`support_maturity_debt`):** Evaluates hardware backend coverage across AMD, Apple Silicon, and NVIDIA accelerators.
- **Growth Debt (`growth_debt`):** Identifies empty cells in the combinatorial capability x execution backend matrix.

### 6. Governance, safety, and guard systems

- **Token Defaults (`token_defaults_debt`):** Prevents high-value token-saving optimizers from remaining disabled or gated behind hidden flags.
- **Guard RSI (`guard_rsi_debt`):** Ensures every kernel guard policy decision is accompanied by structured, replayable explanations.
- **Guard Accuracy (`guard_accuracy_debt`):** Measures classifier accuracy on command escalation to prevent false blocks and safety bypasses.
- **Stability (`stability_debt`):** Monitors regression vectors, tail-wagging dependencies, and verified rollback procedures.

### 7. Lifecycle, milestones, and release velocity

- **Maturity (`maturity_debt`):** Enforces the capability lifecycle ladder (`proposed -> prototyped -> tested -> dogfooded -> default`), preventing untested production code.
- **Milestones (`milestone_debt`):** Measures distance-to-matured across the backend support grid and open milestone epics.
- **Milestone Climb (`climb_ratchet_debt`):** Hard ratchet preventing any loss of matured cells across releases.
- **Loop Index (`loopindex_debt`):** Audits connectivity of the 6-stage autonomous coding loop (orient, plan, act, verify, ship, learn).
- **Release Readiness (`release_debt`):** Flags manual steps, unverified binaries, and missing rollback metadata in the release workflow.

### 8. Delivery flow, commits, and operations

- **Flow Metrics (`flow_debt`):** Tracks Little's Law flow defects including long queue times, untracked local WIP, and unstarted backlog items.
- **Commit Subjects (`commit_debt`):** Audits git commit messages for Conventional Commits compliance, DCO sign-offs, and `(fak <leaf>)` trailers.
- **Operator Heaviness (`heaviness_debt`):** Quantifies operational complexity and manual flags required to drive the system.
- **OSP Residual (`residual_count`):** Tracks unreviewed or unwitnessed PR units in flight.
- **Propagation (`propagation_debt`):** Identifies protocol drift and delayed propagation across mirrored interfaces.
- **Claim Reproducibility (`claim_repro_debt`):** Eliminates unfalsifiable witness claims from `CLAIMS.md`.
- **Observability (`observability_debt`):** Ensures all metrics referenced in dashboards and alerts are actively exported by the binary.
- **RSI Maturity (`rsi_debt`):** Validates that self-improvement loops close on real telemetry.
- **Dogfood Loop (`dogfood_debt`):** Enforces live binary and model dogfooding in tests rather than mock-only verification.

## Deterministic maintenance and verification

To ensure this document never drifts from the live scorecard registry, it is governed by deterministic tooling:

- **Regenerate in place:**
  ```bash
  go run ./cmd/fak scoreboard debt-page --write-doc
  # or
  python tools/scorecard_control_pane.py --write-doc
  ```

- **Verify freshness in CI:**
  ```bash
  go run ./cmd/fak scoreboard debt-page --check-doc
  # or
  python tools/scorecard_control_pane.py --check-doc
  ```

A non-zero exit from `--check-doc` indicates that new scorecards or baseline adjustments have been made without updating this reference page.
