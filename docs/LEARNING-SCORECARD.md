# Learning-docs scorecard

<!-- learning-scorecard: 2026-06-24 · process: tools/learning_scorecard.py -->

> Regenerate: `python tools/learning_scorecard.py --markdown --stamp DATE > docs/LEARNING-SCORECARD.md`

This is the measuring stick for the learning-2× program: does the *teaching* set actually teach? Every number below is re-derived from disk by `tools/learning_scorecard.py` — no hand-entry. The headline metric is **learning-debt**: the count of concrete teaching defects (a how-to with no runnable command, a tutorial with no worked output, a lesson with no orientation signpost, an orphan lesson, an uncovered learning topic). Driving learning-debt toward zero — then raising the most-important docs — is what makes "the docs teach 2× better" provable. Pedagogy counterpart of `docs_scorecard` (hygiene) and `doc_appeal_scorecard` (voice).

## Corpus

| Metric | Value |
|---|---|
| Learning docs scored | 42 |
| **Learning-debt (total HARD defects)** | **0** (0 in-doc + 0 coverage) |
| Soft signals (judgment calls) | 89 |
| Mean score | 96.4/100 |
| Median / min / max | 97.0 / 88.7 / 100.0 |
| Grade distribution | A:41 B:1 C:0 D:0 F:0 |
| Coverage (overall) | 100.0% |
| — reachable from a front door | 100.0% |
| — expected learning topics covered | 100.0% |

## Priorities — fix the most-important, most-broken first

Ranked by importance (funnel-centrality: link-hop proximity to a front door + in-degree from other learning docs) × teaching pressure (defects + soft signals + score gap). These are the 2× targets.

| Priority | Importance | Score | Grade | Debt | Soft | Type | Doc |
|---:|---:|---:|:--:|:--:|:--:|:--|---|
| 0.725 | 67.1 | 92.0 | A | 0 | 4 | reference | `docs/fak/server-config.md` |
| 0.554 | 41.4 | 91.2 | A | 0 | 5 | explainer | `docs/explainers/kv-cache-agentic-context.md` |
| 0.531 | 32.9 | 88.7 | B | 0 | 6 | explainer | `docs/explainers/hardware-portability.md` |
| 0.48 | 35.7 | 90.6 | A | 0 | 5 | explainer | `docs/explainers/sota-optimizations.md` |
| 0.443 | 32.9 | 90.5 | A | 0 | 5 | explainer | `docs/explainers/awq-quantization.md` |
| 0.44 | 32.9 | 91.2 | A | 0 | 5 | explainer | `docs/explainers/local-vs-frontier-parity.md` |
| 0.403 | 30.0 | 90.8 | A | 0 | 5 | reference | `docs/explainers/video-content-plan.md` |
| 0.352 | 32.9 | 93.0 | A | 0 | 4 | explainer | `docs/explainers/code-linting-at-the-kernel.md` |
| 0.325 | 61.4 | 97.0 | A | 0 | 2 | howto | `docs/fak/observability.md` |
| 0.309 | 38.6 | 95.0 | A | 0 | 3 | howto | `docs/fak/multi-language-examples.md` |

## Per-doc scores

Seven KPIs, each 0–100 — pedagogy (orientation · runnable · worked · honesty) + mechanical (structure · links · freshness) — weighted into a score and an A–F grade. `def` = units of learning-debt in that doc.

| Score | Grade | Debt | orient | run | work | hon | struct | link | fresh | Imp | Doc |
|---:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|---:|---|
| 88.7 | B | 0 | 80 | 90 | 88 | 85 | 85 | 100 | 100 | 32.9 | `docs/explainers/hardware-portability.md` |
| 90.5 | A | 0 | 80 | 100 | 88 | 85 | 85 | 100 | 100 | 32.9 | `docs/explainers/awq-quantization.md` |
| 90.6 | A | 0 | 80 | 90 | 88 | 100 | 85 | 100 | 100 | 35.7 | `docs/explainers/sota-optimizations.md` |
| 90.8 | A | 0 | 78 | 100 | 88 | 100 | 75 | 100 | 100 | 30.0 | `docs/explainers/video-content-plan.md` |
| 91.2 | A | 0 | 80 | 100 | 88 | 100 | 75 | 100 | 100 | 41.4 | `docs/explainers/kv-cache-agentic-context.md` |
| 91.2 | A | 0 | 80 | 100 | 88 | 100 | 75 | 100 | 100 | 32.9 | `docs/explainers/local-vs-frontier-parity.md` |
| 92.0 | A | 0 | 78 | 100 | 88 | 100 | 85 | 100 | 100 | 67.1 | `docs/fak/server-config.md` |
| 93.0 | A | 0 | 80 | 100 | 100 | 100 | 75 | 100 | 100 | 32.9 | `docs/explainers/code-linting-at-the-kernel.md` |
| 94.2 | A | 0 | 80 | 100 | 100 | 100 | 85 | 100 | 100 | 32.9 | `docs/prefill-elimination-explained.md` |
| 94.4 | A | 0 | 92 | 88 | 88 | 100 | 100 | 100 | 100 | 35.7 | `docs/fak/hosted-control-plane.md` |
| 95.0 | A | 0 | 100 | 100 | 88 | 85 | 90 | 100 | 100 | 38.6 | `docs/fak/multi-language-examples.md` |
| 95.0 | A | 0 | 90 | 100 | 88 | 100 | 90 | 100 | 100 | 32.9 | `docs/fak/related-items.md` |
| 96.2 | A | 0 | 90 | 100 | 88 | 100 | 100 | 100 | 100 | 47.1 | `docs/fak/api-reference.md` |
| 96.2 | A | 0 | 90 | 100 | 88 | 100 | 100 | 100 | 100 | 35.7 | `docs/fak/documentation-roadmap.md` |
| 96.6 | A | 0 | 92 | 100 | 88 | 100 | 100 | 100 | 100 | 30.0 | `docs/explainers/o1-context-window-economics.md` |
| 96.6 | A | 0 | 92 | 100 | 88 | 100 | 100 | 100 | 100 | 41.4 | `docs/explainers/one-binary-one-surface.md` |
| 96.8 | A | 0 | 90 | 100 | 100 | 100 | 90 | 100 | 100 | 32.9 | `docs/FAQ.md` |
| 96.8 | A | 0 | 90 | 100 | 100 | 100 | 90 | 100 | 100 | 35.7 | `docs/fak/faq.md` |
| 97.0 | A | 0 | 100 | 100 | 88 | 100 | 90 | 100 | 100 | 38.6 | `docs/fak/advanced-topics.md` |
| 97.0 | A | 0 | 100 | 100 | 88 | 100 | 90 | 100 | 100 | 41.4 | `docs/fak/agent-framework-integration.md` |
| 97.0 | A | 0 | 100 | 100 | 88 | 100 | 90 | 100 | 100 | 41.4 | `docs/fak/agent-integration-architecture.md` |
| 97.0 | A | 0 | 100 | 100 | 88 | 100 | 90 | 100 | 100 | 41.4 | `docs/fak/deployment-guide.md` |
| 97.0 | A | 0 | 100 | 100 | 88 | 100 | 90 | 100 | 100 | 61.4 | `docs/fak/observability.md` |
| 97.0 | A | 0 | 100 | 100 | 88 | 100 | 90 | 100 | 100 | 41.4 | `docs/fak/server-troubleshooting.md` |
| 98.2 | A | 0 | 100 | 100 | 88 | 100 | 100 | 100 | 100 | 35.7 | `docs/MEMORY-LAYERS-EXPLAINER.md` |
| 98.2 | A | 0 | 100 | 90 | 100 | 100 | 100 | 100 | 100 | 32.9 | `docs/concepts-and-story.md` |
| 98.2 | A | 0 | 100 | 100 | 88 | 100 | 100 | 100 | 100 | 32.9 | `docs/explainers/fleet-benchmarks.md` |
| 98.2 | A | 0 | 100 | 90 | 100 | 100 | 100 | 100 | 100 | 50.0 | `docs/explainers/policy-in-the-kernel.md` |
| 98.2 | A | 0 | 100 | 90 | 100 | 100 | 100 | 100 | 100 | 62.9 | `docs/fak/README.md` |
| 98.2 | A | 0 | 100 | 90 | 100 | 100 | 100 | 100 | 100 | 32.9 | `docs/glossary.md` |
| 98.8 | A | 0 | 100 | 100 | 100 | 100 | 90 | 100 | 100 | 35.7 | `GETTING-STARTED.md` |
| 98.8 | A | 0 | 100 | 100 | 100 | 100 | 90 | 100 | 100 | 30.0 | `INSTALL.md` |
| 98.8 | A | 0 | 100 | 100 | 100 | 100 | 90 | 100 | 100 | 41.4 | `docs/fak/migration-guide.md` |
| 98.8 | A | 0 | 100 | 100 | 100 | 100 | 90 | 100 | 100 | 52.9 | `docs/fak/policy-guide.md` |
| 98.8 | A | 0 | 100 | 100 | 100 | 100 | 90 | 100 | 100 | 64.3 | `docs/fak/security.md` |
| 98.8 | A | 0 | 100 | 100 | 100 | 100 | 90 | 100 | 100 | 61.4 | `docs/fak/server-quickstart.md` |
| 98.8 | A | 0 | 100 | 100 | 100 | 100 | 90 | 100 | 100 | 70.0 | `docs/fak/tutorial.md` |
| 98.8 | A | 0 | 100 | 100 | 100 | 100 | 90 | 100 | 100 | 30.0 | `docs/run-the-demos.md` |
| 100.0 | A | 0 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 62.9 | `LEARNING-PATH.md` |
| 100.0 | A | 0 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 62.9 | `START-HERE.md` |
| 100.0 | A | 0 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 32.9 | `docs/CONTEXT-IS-NOT-MEMORY.md` |
| 100.0 | A | 0 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 44.3 | `docs/explainers/addressable-kv-cache.md` |

## Learning-debt work-list

No learning-debt: every learning doc clears the HARD bar. 🎉

Remaining work is soft-signal polish on the priority docs above.

