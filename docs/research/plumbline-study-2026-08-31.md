# ActaClad/plumbline study — 2026-08-31

**Verdict:** borrow four bounded mechanisms; do not adopt Plumbline as a second analyzer framework.

- **Pinned source:** `ActaClad/plumbline@cae5eeaa244830ef88bd47e18d39b7b7761df41e` (main, committed 2026-08-27).
- **License:** Apache-2.0 (`LICENSE`, `NOTICE`). Direct reuse is legally possible with attribution, but the surviving work adapts ideas to FAK's Go-native types rather than copying code.
- **Durable decision:** `study_b61c432e9a8f2ad5574036e224b8fd957aa482a9c588e0fe295396ed532edad6`.
- **Parent tracker:** #10452.

## What it is

Plumbline is a Python static analyzer for LLM and agent applications. Its pipeline parses Python into an AST layer, resolves a bounded value domain, derives framework-independent semantics, computes source-to-sink evidence, runs declarative rules, then renders CLI, JSON, HTML, or SARIF output. It also supports baselines, inline suppression, framework adapters, readiness scoring, optional AI enrichment, and remediation-skill export.

The architecture is coherent for a young project: the first commits build the model, config, AST, value resolver, adapter contract, rule contract, engine, taint engine, reporters, baselines, and benchmark harness in that order. At the pinned revision the repository has 125 commits, releases `v0.0.1` and `v0.0.2`, 26 stars, 3 forks, one primary human contributor, Dependabot maintenance, and no non-PR GitHub issues. These are adoption/context facts, not quality scores.

## Evidence surface inspected

- Source: all files under `src/plumbline/`, with focused reads of `model.py`, `engine.py`, `core/{ast_layer,values,derive,evidence,taint}.py`, `rules/base.py`, `baseline.py`, `benchmark.py`, `scoring/`, `reporters/`, `skills/export.py`, and all adapters.
- Tests: the full `tests/` tree, rule fixtures, reporter tests, robustness/performance tests, and benchmark tests.
- Design and operations: README, changelog, ADRs/specs, backlog, CI/action metadata, release tags, commit history, GitHub issues/PRs, license, packaging, and contribution/security docs.
- FAK comparison: current typed policy/adjudication, IFC/provenance, scorecard/lint families, study receipts, skill lifecycle, reporting surfaces, GitHub backlog, and existing SARIF references.

## Reconstructed design

1. **Typed finding boundary.** `src/plumbline/model.py:179-268` defines findings, drafts, locations, evidence, and deterministic fingerprints. Rules do not own rendering.
2. **Layered static semantics.** `core/ast_layer.py` wraps Python AST details; `core/values.py` resolves bounded values; `core/derive.py` adds cross-module and framework-neutral facts; `core/taint.py` propagates sources and witnesses. Rules stay smaller than an all-in-one AST visitor.
3. **Declarative rule metadata.** `rules/base.py` couples stable IDs and severity with rationale, remediation, references, standards, scope, and evidence requirements.
4. **Reporter boundary.** The same finding model drives CLI, JSON, HTML, and SARIF. `reporters/sarif.py` maps rule identity, source regions, severity, help, and fingerprints into SARIF 2.1.0.
5. **Ratcheting.** `baseline.py` compares fingerprints so CI can focus on new findings while preserving existing debt; inline suppression is separately modeled and tested.
6. **Measured rules.** `benchmark.py`, `benchmark/`, and `tests/test_benchmark.py` use labeled fixtures to compute TP/FP/FN, precision, and recall rather than treating rule count as evidence.
7. **Framework normalization.** OpenAI SDK, LangChain, CrewAI, LiteLLM, and Gemini adapters translate call shapes into shared semantic queries.
8. **Operator packaging.** Readiness pillars, HTML aggregation, SARIF, GitHub Action integration, and remediation-skill export make analysis actionable without changing the core engine.

## Candidate matrix

| Mechanism | FAK status | Disposition | Evidence and reason |
|---|---|---|---|
| SARIF projection for typed findings | **ABSENT** | **ADAPT — #10460** | Plumbline separates findings from `reporters/sarif.py`; FAK has many typed analyzers but only a documentation example mentioning CodeQL SARIF, not a native projection. Add one narrow reporter, not another analyzer framework. |
| Labeled precision/recall corpus contract | **PARTIAL** | **ADAPT — #10461** | FAK has rich fixtures and several benchmark/scorecard gates, but no common TP/FP/FN contract for deterministic analyzers. Borrow Plumbline's explicit expected-findings corpus and strengthen it with revision and repeatability receipts. |
| Stable finding identity plus baseline delta | **PARTIAL** | **ADAPT — #10462** | FAK has typed IDs, evidence, and historical ledgers in individual leaves, but no shared path-oriented `new/unchanged/resolved` contract. Adapt the seam; reject Plumbline's exact location-sensitive identity because unrelated line movement should not churn findings. |
| Bounded cross-function source/value flow | **ABSENT** | **ADAPT — #10464** | FAK's runtime policy and IFC are stronger for executed calls, while source lints are mostly leaf-specific. Prove one high-value Go rule using an explicit supported envelope and `unknown`, rather than building a general taint engine. |
| Whole Plumbline framework: Python registry, adapters, readiness score, HTML report, remediation-skill export | **PRESENT / superseded** | **EXCLUDE** | FAK already owns Go-native leaves, runtime policy, provenance, scorecards, reports, plugin/skill lifecycle, and issue-backed remediation. Porting the framework would create duplicate authority and a Python dependency. |
| Optional AI enrichment of static findings | **PRESENT / stronger equivalent** | **EXCLUDE** | FAK already has guarded model routing and evidence-gated workflows. Static findings should remain authoritative; optional prose enrichment is not a missing kernel mechanism. |
| Broad rule catalog as-is | **PARTIAL** | **WATCH, not port** | Many rules are Python/framework conventions, deprecated-model lists, or generic configuration checks. Propose individual rules only when a FAK corpus demonstrates a real miss; rule count alone is not value. |

## What FAK should retain from the worldview

- Put evidence paths and stable rule identity below presentation so every output carries the same fact.
- Treat static analysis as a bounded interpretation, with explicit unknown/unsupported states rather than implied soundness.
- Measure rules against negative as well as positive fixtures; a larger rule catalog is not improvement if noise grows.
- Keep baseline debt visible while allowing a new-findings gate.
- Export into operators' existing surfaces rather than requiring one bespoke report.

## What FAK should not copy

- A second Python plugin/adapter ecosystem beside the Go leaf registry.
- A single composite readiness score as primary truth; FAK's provenance-specific receipts and axis-separated status are less conflating.
- Location-sensitive fingerprints without a versioned identity contract and line-movement test.
- Static source analysis as a replacement for runtime adjudication. It complements the kernel seam but cannot observe actual arguments, results, model routing, or context taint.
- Claimed determinism without a cross-process witness.

## Verification and honest limits

Running the pinned checkout with `PYTHONPATH=src python -m pytest -q` reached **481 passed and 2 failed** on this Windows/Python 3.13 host:

- `tests/test_cli.py::test_ai_enabled_without_key_emits_notice` fails because the assertion searches for lowercase `static remediation` while output begins `Static remediation`.
- `tests/test_engine.py::test_cross_process_determinism_across_hashseeds` fails because subprocess result bytes differ across hash seeds.

The failures do not negate the studied mechanisms, but they prevent treating the current checkout's deterministic-output claim as fully witnessed. The follow-on corpus and identity tickets therefore include repeated-process gates.

No performance comparison with FAK is claimed. Plumbline scans Python source; FAK is primarily a runtime agent kernel. Their operating envelopes and work products differ, so rule count, repository size, and scan latency are not a matched comparison.

## Filed work

- #10460 — project selected typed analyzer findings to SARIF 2.1.0.
- #10461 — add a labeled precision/recall corpus contract.
- #10462 — add stable finding identities and baseline deltas.
- #10464 — prove one bounded cross-function value-flow rule.

Each issue carries the pinned revision, narrow spine, non-goals, falsifiable witness, and durable receipt ID `study_b61c432e9a8f2ad5574036e224b8fd957aa482a9c588e0fe295396ed532edad6`.

## Refresh trigger

Recheck when Plumbline ships a materially different analysis engine or reporter contract, fixes the cross-process determinism failure, publishes a release after `v0.0.2`, or adds source-language support relevant to FAK. Otherwise the four child issues are the durable adoption path.
