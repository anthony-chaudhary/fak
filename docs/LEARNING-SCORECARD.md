---
title: "fak learning-docs scorecard — the learning-debt measuring stick"
description: "fak's deterministic learning-docs scorecard: does the teaching set actually teach? Pedagogy KPIs (a how-to with no runnable command, a tutorial with no worked output, an orphan lesson, an uncovered learning topic) folded into a composite score and the headline learning-debt metric, re-derived from disk."
---

# Learning-docs scorecard

<!-- learning-scorecard: 2026-06-29 · process: tools/learning_scorecard.py -->

> Regenerate: `python tools/learning_scorecard.py --markdown --stamp DATE > docs/LEARNING-SCORECARD.md`

This is the measuring stick for the learning-2× program: does the *teaching* set actually teach? Every number below is re-derived from disk by `tools/learning_scorecard.py` — no hand-entry. The headline metric is **learning-debt**: the count of concrete teaching defects (a how-to with no runnable command, a tutorial with no worked output, a lesson with no orientation signpost, an orphan lesson, an uncovered learning topic). Driving learning-debt toward zero — then raising the most-important docs — is what makes "the docs teach 2× better" provable. Pedagogy counterpart of `docs_scorecard` (hygiene) and `doc_appeal_scorecard` (voice).

## Corpus

| Metric | Value |
|---|---|
| Learning docs scored | 76 |
| **Learning-debt (total HARD defects)** | **0** (0 in-doc + 0 coverage) |
| Soft signals (judgment calls) | 144 |
| Mean score | 96.8/100 |
| Median / min / max | 97.0 / 90.5 / 100.0 |
| Grade distribution | A:76 B:0 C:0 D:0 F:0 |
| Coverage (overall) | 100.0% |
| — reachable from a front door | 100.0% |
| — expected learning topics covered | 100.0% |

## Priorities — fix the most-important, most-broken first

Ranked by importance (funnel-centrality: link-hop proximity to a front door + in-degree from other learning docs) × teaching pressure (defects + soft signals + score gap). These are the 2× targets.

| Priority | Importance | Score | Grade | Debt | Soft | Type | Doc |
|---:|---:|---:|:--:|:--:|:--:|:--|---|
| 0.756 | 70.0 | 92.0 | A | 0 | 4 | reference | `docs/fak/server-config.md` |
| 0.436 | 32.4 | 90.5 | A | 0 | 5 | explainer | `docs/explainers/awq-quantization.md` |
| 0.434 | 32.4 | 91.2 | A | 0 | 5 | explainer | `docs/explainers/local-vs-frontier-parity.md` |
| 0.403 | 30.0 | 90.8 | A | 0 | 5 | reference | `docs/explainers/video-content-plan.md` |
| 0.351 | 44.1 | 95.4 | A | 0 | 3 | explainer | `docs/explainers/kv-cache-agentic-context.md` |
| 0.347 | 32.4 | 93.0 | A | 0 | 4 | explainer | `docs/explainers/code-linting-at-the-kernel.md` |
| 0.333 | 62.9 | 97.0 | A | 0 | 2 | howto | `docs/fak/observability.md` |
| 0.314 | 39.4 | 95.4 | A | 0 | 3 | explainer | `docs/explainers/frozen-trajectory-cache-cliff.md` |
| 0.299 | 37.1 | 94.4 | A | 0 | 3 | howto | `docs/fak/session-observability-rsi-loop.md` |
| 0.298 | 37.1 | 94.8 | A | 0 | 3 | explainer | `docs/explainers/hardware-portability.md` |

## Per-doc scores

Seven KPIs, each 0–100 — pedagogy (orientation · runnable · worked · honesty) + mechanical (structure · links · freshness) — weighted into a score and an A–F grade. `def` = units of learning-debt in that doc.

| Score | Grade | Debt | orient | run | work | hon | struct | link | fresh | Imp | Doc |
|---:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|---:|---|
| 90.5 | A | 0 | 80 | 100 | 88 | 85 | 85 | 100 | 100 | 32.4 | `docs/explainers/awq-quantization.md` |
| 90.8 | A | 0 | 78 | 100 | 88 | 100 | 75 | 100 | 100 | 30.0 | `docs/explainers/video-content-plan.md` |
| 91.2 | A | 0 | 80 | 100 | 88 | 100 | 75 | 100 | 100 | 32.4 | `docs/explainers/local-vs-frontier-parity.md` |
| 92.0 | A | 0 | 78 | 100 | 88 | 100 | 85 | 100 | 100 | 70.0 | `docs/fak/server-config.md` |
| 93.0 | A | 0 | 80 | 100 | 100 | 100 | 75 | 100 | 100 | 32.4 | `docs/explainers/code-linting-at-the-kernel.md` |
| 94.1 | A | 0 | 100 | 88 | 88 | 85 | 100 | 100 | 100 | 34.7 | `docs/fak/loop-tool-map.md` |
| 94.2 | A | 0 | 80 | 100 | 100 | 100 | 85 | 100 | 100 | 30.0 | `docs/explainers/context-signal-to-noise.md` |
| 94.2 | A | 0 | 80 | 100 | 100 | 100 | 85 | 100 | 100 | 32.4 | `docs/prefill-elimination-explained.md` |
| 94.4 | A | 0 | 92 | 88 | 88 | 100 | 100 | 100 | 100 | 34.7 | `docs/fak/hosted-control-plane.md` |
| 94.4 | A | 0 | 92 | 88 | 88 | 100 | 100 | 100 | 100 | 37.1 | `docs/fak/session-observability-rsi-loop.md` |
| 94.8 | A | 0 | 92 | 90 | 88 | 100 | 100 | 100 | 100 | 37.1 | `docs/explainers/hardware-portability.md` |
| 94.8 | A | 0 | 92 | 90 | 88 | 100 | 100 | 100 | 100 | 37.1 | `docs/explainers/sota-optimizations.md` |
| 94.8 | A | 0 | 92 | 90 | 88 | 100 | 100 | 100 | 100 | 30.0 | `docs/explainers/ultracode-multi-agent-dogfood.md` |
| 95.0 | A | 0 | 100 | 100 | 88 | 85 | 90 | 100 | 100 | 32.4 | `docs/fak/lab-dev-loop.md` |
| 95.0 | A | 0 | 100 | 100 | 88 | 85 | 90 | 100 | 100 | 37.1 | `docs/fak/multi-language-examples.md` |
| 95.0 | A | 0 | 90 | 100 | 88 | 100 | 90 | 100 | 100 | 32.4 | `docs/fak/related-items.md` |
| 95.4 | A | 0 | 92 | 100 | 88 | 100 | 90 | 100 | 100 | 39.4 | `docs/explainers/frozen-trajectory-cache-cliff.md` |
| 95.4 | A | 0 | 92 | 100 | 88 | 100 | 90 | 100 | 100 | 44.1 | `docs/explainers/kv-cache-agentic-context.md` |
| 95.4 | A | 0 | 92 | 100 | 88 | 100 | 90 | 100 | 100 | 34.7 | `docs/fak/mac-agent-ui.md` |
| 96.0 | A | 0 | 100 | 88 | 88 | 100 | 100 | 100 | 100 | 32.4 | `docs/fak/batching-config.md` |
| 96.0 | A | 0 | 100 | 88 | 88 | 100 | 100 | 100 | 100 | 32.4 | `docs/fak/dogfood-loop-scorecard.md` |
| 96.2 | A | 0 | 90 | 100 | 88 | 100 | 100 | 100 | 100 | 46.5 | `docs/fak/api-reference.md` |
| 96.4 | A | 0 | 100 | 90 | 88 | 100 | 100 | 100 | 100 | 34.7 | `docs/explainers/vdso-revoke-as-comm-revoke.md` |
| 96.6 | A | 0 | 92 | 100 | 88 | 100 | 100 | 100 | 100 | 34.7 | `docs/explainers/o1-context-window-economics.md` |
| 96.6 | A | 0 | 92 | 100 | 88 | 100 | 100 | 100 | 100 | 41.8 | `docs/explainers/one-binary-one-surface.md` |
| 96.6 | A | 0 | 92 | 100 | 88 | 100 | 100 | 100 | 100 | 34.7 | `docs/fak/dojo-rsi-loop.md` |
| 96.6 | A | 0 | 92 | 100 | 88 | 100 | 100 | 100 | 100 | 34.7 | `docs/fak/gcp-tier2-control-vm.md` |
| 96.6 | A | 0 | 92 | 100 | 88 | 100 | 100 | 100 | 100 | 32.4 | `docs/fak/guard-hop-rsi-loop.md` |
| 96.6 | A | 0 | 92 | 100 | 88 | 100 | 100 | 100 | 100 | 32.4 | `docs/fak/mcp-registry.md` |
| 96.6 | A | 0 | 92 | 100 | 88 | 100 | 100 | 100 | 100 | 22.4 | `docs/fak/session-control.md` |
| 96.8 | A | 0 | 90 | 100 | 100 | 100 | 90 | 100 | 100 | 34.7 | `docs/FAQ.md` |
| 96.8 | A | 0 | 90 | 100 | 100 | 100 | 90 | 100 | 100 | 34.7 | `docs/fak/faq.md` |
| 97.0 | A | 0 | 100 | 100 | 88 | 100 | 90 | 100 | 100 | 20.0 | `docs/explainers/compounding-benefits-of-a-saved-call.md` |
| 97.0 | A | 0 | 100 | 100 | 88 | 100 | 90 | 100 | 100 | 22.4 | `docs/explainers/hardware-limits-and-capacity.md` |
| 97.0 | A | 0 | 100 | 100 | 88 | 100 | 90 | 100 | 100 | 41.8 | `docs/fak/advanced-topics.md` |
| 97.0 | A | 0 | 100 | 100 | 88 | 100 | 90 | 100 | 100 | 39.4 | `docs/fak/agent-framework-integration.md` |
| 97.0 | A | 0 | 100 | 100 | 88 | 100 | 90 | 100 | 100 | 41.8 | `docs/fak/agent-integration-architecture.md` |
| 97.0 | A | 0 | 100 | 100 | 88 | 100 | 90 | 100 | 100 | 48.8 | `docs/fak/always-on-dogfood-server.md` |
| 97.0 | A | 0 | 100 | 100 | 88 | 100 | 90 | 100 | 100 | 34.7 | `docs/fak/claude-glm-gcp.md` |
| 97.0 | A | 0 | 100 | 100 | 88 | 100 | 90 | 100 | 100 | 51.2 | `docs/fak/deployment-guide.md` |
| 97.0 | A | 0 | 100 | 100 | 88 | 100 | 90 | 100 | 100 | 34.7 | `docs/fak/dojo.md` |
| 97.0 | A | 0 | 100 | 100 | 88 | 100 | 90 | 100 | 100 | 37.1 | `docs/fak/guard-verdict-rsi-loop.md` |
| 97.0 | A | 0 | 100 | 100 | 88 | 100 | 90 | 100 | 100 | 34.7 | `docs/fak/node-macos-a-activation.md` |
| 97.0 | A | 0 | 100 | 100 | 88 | 100 | 90 | 100 | 100 | 32.4 | `docs/fak/node-setup.md` |
| 97.0 | A | 0 | 100 | 100 | 88 | 100 | 90 | 100 | 100 | 62.9 | `docs/fak/observability.md` |
| 97.0 | A | 0 | 100 | 100 | 88 | 100 | 90 | 100 | 100 | 32.4 | `docs/fak/qwen36-a100-gcp.md` |
| 97.0 | A | 0 | 100 | 100 | 88 | 100 | 90 | 100 | 100 | 39.4 | `docs/fak/server-troubleshooting.md` |
| 97.2 | A | 0 | 92 | 100 | 100 | 100 | 90 | 100 | 100 | 34.7 | `docs/fak/opencode-glm-guard.md` |
| 97.8 | A | 0 | 100 | 88 | 100 | 100 | 100 | 100 | 100 | 20.0 | `docs/fak/concept-glossary.md` |
| 98.2 | A | 0 | 100 | 100 | 88 | 100 | 100 | 100 | 100 | 34.7 | `docs/MEMORY-LAYERS-EXPLAINER.md` |
| 98.2 | A | 0 | 100 | 90 | 100 | 100 | 100 | 100 | 100 | 32.4 | `docs/concepts-and-story.md` |
| 98.2 | A | 0 | 100 | 100 | 88 | 100 | 100 | 100 | 100 | 32.4 | `docs/explainers/context-tape-visuals.md` |
| 98.2 | A | 0 | 100 | 100 | 88 | 100 | 100 | 100 | 100 | 34.7 | `docs/explainers/cross-platform-spine.md` |
| 98.2 | A | 0 | 100 | 100 | 88 | 100 | 100 | 100 | 100 | 48.8 | `docs/explainers/engineering-is-building-loops.md` |
| 98.2 | A | 0 | 100 | 100 | 88 | 100 | 100 | 100 | 100 | 34.7 | `docs/explainers/fleet-benchmarks.md` |
| 98.2 | A | 0 | 100 | 100 | 88 | 100 | 100 | 100 | 100 | 30.0 | `docs/explainers/long-sessions-keep-the-cache-hit.md` |
| 98.2 | A | 0 | 100 | 90 | 100 | 100 | 100 | 100 | 100 | 48.8 | `docs/explainers/policy-in-the-kernel.md` |
| 98.2 | A | 0 | 100 | 90 | 100 | 100 | 100 | 100 | 100 | 62.4 | `docs/fak/README.md` |
| 98.2 | A | 0 | 100 | 100 | 88 | 100 | 100 | 100 | 100 | 32.4 | `docs/fak/green-gate-budget.md` |
| 98.4 | A | 0 | 92 | 100 | 100 | 100 | 100 | 100 | 100 | 22.4 | `docs/explainers/multi-gpu-tensor-parallelism.md` |
| 98.8 | A | 0 | 100 | 100 | 100 | 100 | 90 | 100 | 100 | 34.7 | `GETTING-STARTED.md` |
| 98.8 | A | 0 | 100 | 100 | 100 | 100 | 90 | 100 | 100 | 30.0 | `INSTALL.md` |
| 98.8 | A | 0 | 100 | 100 | 100 | 100 | 90 | 100 | 100 | 34.7 | `docs/fak/edge-quickstart.md` |
| 98.8 | A | 0 | 100 | 100 | 100 | 100 | 90 | 100 | 100 | 41.8 | `docs/fak/migration-guide.md` |
| 98.8 | A | 0 | 100 | 100 | 100 | 100 | 90 | 100 | 100 | 53.5 | `docs/fak/policy-guide.md` |
| 98.8 | A | 0 | 100 | 100 | 100 | 100 | 90 | 100 | 100 | 62.9 | `docs/fak/security.md` |
| 98.8 | A | 0 | 100 | 100 | 100 | 100 | 90 | 100 | 100 | 60.6 | `docs/fak/server-quickstart.md` |
| 98.8 | A | 0 | 100 | 100 | 100 | 100 | 90 | 100 | 100 | 65.3 | `docs/fak/tutorial.md` |
| 98.8 | A | 0 | 100 | 100 | 100 | 100 | 90 | 100 | 100 | 30.0 | `docs/run-the-demos.md` |
| 100.0 | A | 0 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 67.1 | `LEARNING-PATH.md` |
| 100.0 | A | 0 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 62.4 | `START-HERE.md` |
| 100.0 | A | 0 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 32.4 | `docs/CONTEXT-IS-NOT-MEMORY.md` |
| 100.0 | A | 0 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 55.9 | `docs/explainers/addressable-kv-cache.md` |
| 100.0 | A | 0 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 34.7 | `docs/fak/documentation-roadmap.md` |
| 100.0 | A | 0 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 22.4 | `docs/fak/scrubbing-real-values.md` |
| 100.0 | A | 0 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 32.4 | `docs/glossary.md` |

## Learning-debt work-list

No learning-debt: every learning doc clears the HARD bar. 🎉

Remaining work is soft-signal polish on the priority docs above.

