---
title: "fak learning-docs scorecard — the learning-debt measuring stick"
description: "fak's deterministic learning-docs scorecard: does the teaching set actually teach? Pedagogy KPIs (a how-to with no runnable command, a tutorial with no worked output, an orphan lesson, an uncovered learning topic) folded into a composite score and the headline learning-debt metric, re-derived from disk."
---

# Learning-docs scorecard

<!-- learning-scorecard: 2026-11-24 · process: tools/learning_scorecard.py -->

> Regenerate: `python tools/learning_scorecard.py --markdown --stamp DATE > docs/LEARNING-SCORECARD.md`

This is the measuring stick for the learning-2× program: does the *teaching* set actually teach? Every number below is re-derived from disk by `tools/learning_scorecard.py` — no hand-entry. The headline metric is **learning-debt**: the count of concrete teaching defects (a how-to with no runnable command, a tutorial with no worked output, a lesson with no orientation signpost, an orphan lesson, an uncovered learning topic). Driving learning-debt toward zero — then raising the most-important docs — is what makes "the docs teach 2× better" provable. Pedagogy counterpart of `docs_scorecard` (hygiene) and `doc_appeal_scorecard` (voice).

## Corpus

| Metric | Value |
|---|---|
| Learning docs scored | 189 |
| **Learning-debt (total HARD defects)** | **66** (12 in-doc + 53 coverage + 1 stamp) |
| Soft signals (judgment calls) | 288 |
| Mean score | 97.2/100 |
| Median / min / max | 98.2 / 75.4 / 100.0 |
| Grade distribution | A:185 B:3 C:1 D:0 F:0 |
| Coverage (overall) | 59.0% |
| — reachable from a front door | 98.9% |
| — expected learning topics covered | 19.0% |
| Stamp freshness | age 81d / cadence 1d — **stale-stamp** |

## Priorities — fix the most-important, most-broken first

Ranked by importance (funnel-centrality: link-hop proximity to a front door + in-degree from other learning docs) × teaching pressure (defects + soft signals + score gap). These are the 2× targets.

| Priority | Importance | Score | Grade | Debt | Soft | Type | Doc |
|---:|---:|---:|:--:|:--:|:--:|:--|---|
| 0.763 | 21.1 | 88.6 | B | 3 | 2 | howto | `docs/fak/team-node-deployment.md` |
| 0.748 | 31.1 | 84.6 | B | 1 | 5 | explainer | `docs/explainers/tool-feedback-continuation.md` |
| 0.697 | 45.1 | 95.4 | A | 1 | 2 | howto | `docs/fak/deployment-guide.md` |
| 0.513 | 33.2 | 95.6 | A | 1 | 2 | howto | `docs/fak/mac-local-models.md` |
| 0.433 | 32.2 | 90.6 | A | 0 | 5 | explainer | `docs/explainers/vllm-lifecycle-cache-loss-bridge.md` |
| 0.418 | 51.6 | 94.0 | A | 0 | 3 | reference | `docs/fak/server-config.md` |
| 0.418 | 31.1 | 90.6 | A | 0 | 5 | explainer | `docs/explainers/context-control-surfaces-and-the-pin-audit-gap.md` |
| 0.405 | 50.5 | 94.8 | A | 0 | 3 | explainer | `docs/explainers/hardware-portability.md` |
| 0.403 | 30.0 | 90.8 | A | 0 | 5 | reference | `docs/explainers/video-content-plan.md` |
| 0.345 | 32.2 | 93.0 | A | 0 | 4 | explainer | `docs/explainers/code-linting-at-the-kernel.md` |

## Per-doc scores

Seven KPIs, each 0–100 — pedagogy (orientation · runnable · worked · honesty) + mechanical (structure · links · freshness) — weighted into a score and an A–F grade. `def` = units of learning-debt in that doc.

| Score | Grade | Debt | orient | run | work | hon | struct | link | fresh | Imp | Doc |
|---:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|---:|---|
| 75.4 | C | 3 | 37 | 60 | 88 | 100 | 75 | 100 | 100 | 0.0 | `docs/fak/concept-runtime-burndown-20261102.md` |
| 81.8 | B | 3 | 75 | 60 | 88 | 100 | 65 | 100 | 100 | 0.0 | `docs/fak/feature-activation.md` |
| 84.6 | B | 1 | 50 | 90 | 88 | 100 | 85 | 100 | 100 | 31.1 | `docs/explainers/tool-feedback-continuation.md` |
| 88.6 | B | 3 | 100 | 100 | 88 | 100 | 90 | 40 | 100 | 21.1 | `docs/fak/team-node-deployment.md` |
| 90.5 | A | 0 | 80 | 100 | 88 | 85 | 85 | 100 | 100 | 25.4 | `docs/explainers/awq-quantization.md` |
| 90.6 | A | 0 | 80 | 90 | 88 | 100 | 85 | 100 | 100 | 31.1 | `docs/explainers/context-control-surfaces-and-the-pin-audit-gap.md` |
| 90.6 | A | 0 | 80 | 90 | 88 | 100 | 85 | 100 | 100 | 32.2 | `docs/explainers/vllm-lifecycle-cache-loss-bridge.md` |
| 90.8 | A | 0 | 78 | 100 | 88 | 100 | 75 | 100 | 100 | 30.0 | `docs/explainers/video-content-plan.md` |
| 91.2 | A | 0 | 80 | 100 | 88 | 100 | 75 | 100 | 100 | 21.1 | `docs/explainers/local-vs-frontier-parity.md` |
| 92.4 | A | 0 | 80 | 100 | 88 | 100 | 85 | 100 | 100 | 22.2 | `docs/explainers/require-witness-tool-call-continuations.md` |
| 93.0 | A | 0 | 80 | 100 | 100 | 100 | 75 | 100 | 100 | 32.2 | `docs/explainers/code-linting-at-the-kernel.md` |
| 94.0 | A | 0 | 88 | 100 | 88 | 100 | 85 | 100 | 100 | 51.6 | `docs/fak/server-config.md` |
| 94.1 | A | 0 | 100 | 88 | 88 | 85 | 100 | 100 | 100 | 32.2 | `docs/fak/loop-tool-map.md` |
| 94.2 | A | 0 | 80 | 100 | 100 | 100 | 85 | 100 | 100 | 32.2 | `docs/explainers/context-signal-to-noise.md` |
| 94.2 | A | 0 | 80 | 100 | 100 | 100 | 85 | 100 | 100 | 31.1 | `docs/explainers/loader-semantic-transform-contracts.md` |
| 94.2 | A | 0 | 80 | 100 | 100 | 100 | 85 | 100 | 100 | 20.0 | `docs/prefill-elimination-explained.md` |
| 94.4 | A | 0 | 92 | 88 | 88 | 100 | 100 | 100 | 100 | 21.1 | `docs/fak/glossary-families-a.md` |
| 94.4 | A | 0 | 92 | 88 | 88 | 100 | 100 | 100 | 100 | 21.1 | `docs/fak/glossary-positions-3.md` |
| 94.4 | A | 0 | 92 | 88 | 88 | 100 | 100 | 100 | 100 | 33.2 | `docs/fak/hosted-control-plane.md` |
| 94.4 | A | 0 | 92 | 88 | 88 | 100 | 100 | 100 | 100 | 32.2 | `docs/fak/session-observability-rsi-loop.md` |
| 94.8 | A | 0 | 92 | 90 | 88 | 100 | 100 | 100 | 100 | 31.1 | `docs/explainers/data-residency-and-compliance.md` |
| 94.8 | A | 0 | 92 | 90 | 88 | 100 | 100 | 100 | 100 | 50.5 | `docs/explainers/hardware-portability.md` |
| 94.8 | A | 0 | 92 | 90 | 88 | 100 | 100 | 100 | 100 | 35.4 | `docs/explainers/memory-engineering.md` |
| 94.8 | A | 0 | 92 | 90 | 88 | 100 | 100 | 100 | 100 | 28.6 | `docs/explainers/sota-optimizations.md` |
| 94.8 | A | 0 | 92 | 90 | 88 | 100 | 100 | 100 | 100 | 31.1 | `docs/explainers/structured-generation-became-infrastructure.md` |
| 94.8 | A | 0 | 92 | 90 | 88 | 100 | 100 | 100 | 100 | 31.1 | `docs/explainers/ultracode-multi-agent-dogfood.md` |
| 94.8 | A | 0 | 100 | 88 | 88 | 100 | 90 | 100 | 100 | 31.1 | `docs/fak/configuration.md` |
| 95.0 | A | 0 | 100 | 100 | 88 | 85 | 90 | 100 | 100 | 32.2 | `docs/fak/multi-language-examples.md` |
| 95.0 | A | 0 | 90 | 100 | 88 | 100 | 90 | 100 | 100 | 31.1 | `docs/fak/related-items.md` |
| 95.4 | A | 1 | 100 | 100 | 88 | 100 | 90 | 100 | 80 | 45.1 | `docs/fak/deployment-guide.md` |
| 95.4 | A | 0 | 92 | 100 | 88 | 100 | 90 | 100 | 100 | 32.2 | `docs/explainers/cache.md` |
| 95.4 | A | 0 | 92 | 100 | 88 | 100 | 90 | 100 | 100 | 36.5 | `docs/explainers/frozen-trajectory-cache-cliff.md` |
| 95.4 | A | 0 | 92 | 100 | 88 | 100 | 90 | 100 | 100 | 41.9 | `docs/explainers/kv-cache-agentic-context.md` |
| 95.4 | A | 0 | 92 | 100 | 88 | 100 | 90 | 100 | 100 | 31.1 | `docs/explainers/less-context-less-code.md` |
| 95.4 | A | 0 | 92 | 100 | 88 | 100 | 90 | 100 | 100 | 31.1 | `docs/fak/slack-sessions.md` |
| 95.6 | A | 1 | 92 | 100 | 100 | 100 | 90 | 100 | 80 | 33.2 | `docs/fak/mac-local-models.md` |
| 96.0 | A | 0 | 100 | 88 | 88 | 100 | 100 | 100 | 100 | 31.1 | `docs/fak/batching-config.md` |
| 96.0 | A | 0 | 100 | 88 | 88 | 100 | 100 | 100 | 100 | 31.1 | `docs/fak/dogfood-loop-scorecard.md` |
| 96.2 | A | 0 | 90 | 100 | 88 | 100 | 100 | 100 | 100 | 38.6 | `docs/fak/api-reference.md` |
| 96.2 | A | 0 | 92 | 88 | 100 | 100 | 100 | 100 | 100 | 21.1 | `docs/fak/glossary-families-b.md` |
| 96.2 | A | 0 | 92 | 88 | 100 | 100 | 100 | 100 | 100 | 21.1 | `docs/fak/glossary-positions-1.md` |
| 96.2 | A | 0 | 92 | 88 | 100 | 100 | 100 | 100 | 100 | 21.1 | `docs/fak/glossary-positions-2.md` |
| 96.2 | A | 0 | 92 | 88 | 100 | 100 | 100 | 100 | 100 | 21.1 | `docs/fak/glossary-positions-4.md` |
| 96.4 | A | 0 | 100 | 90 | 88 | 100 | 100 | 100 | 100 | 35.4 | `docs/explainers/addressable-kv-cache-in-5-min.md` |
| 96.4 | A | 0 | 100 | 90 | 88 | 100 | 100 | 100 | 100 | 51.6 | `docs/explainers/caching/README.md` |
| 96.4 | A | 0 | 100 | 90 | 88 | 100 | 100 | 100 | 100 | 34.3 | `docs/explainers/change-data-capture-for-agents.md` |
| 96.4 | A | 0 | 100 | 90 | 88 | 100 | 100 | 100 | 100 | 60.0 | `docs/explainers/glossary.md` |
| 96.4 | A | 0 | 100 | 90 | 88 | 100 | 100 | 100 | 100 | 31.1 | `docs/explainers/shared-workspace-and-the-negation-operator.md` |
| 96.4 | A | 0 | 100 | 90 | 88 | 100 | 100 | 100 | 100 | 23.2 | `docs/explainers/status-a-peer-can-trust.md` |
| 96.4 | A | 0 | 100 | 90 | 88 | 100 | 100 | 100 | 100 | 23.2 | `docs/explainers/vdso-revoke-as-comm-revoke.md` |
| 96.4 | A | 0 | 100 | 90 | 88 | 100 | 100 | 100 | 100 | 37.6 | `docs/explainers/what-fak-is-not.md` |
| 96.4 | A | 0 | 100 | 90 | 88 | 100 | 100 | 100 | 100 | 24.3 | `docs/explainers/you-never-manage-the-context-window.md` |
| 96.6 | A | 0 | 92 | 90 | 100 | 100 | 100 | 100 | 100 | 32.2 | `docs/explainers/agent-virtual-filesystem.md` |
| 96.6 | A | 0 | 92 | 100 | 88 | 100 | 100 | 100 | 100 | 34.3 | `docs/explainers/context-as-a-variable.md` |
| 96.6 | A | 0 | 92 | 100 | 88 | 100 | 100 | 100 | 100 | 37.6 | `docs/explainers/context.md` |
| 96.6 | A | 0 | 92 | 100 | 88 | 100 | 100 | 100 | 100 | 36.5 | `docs/explainers/o1-context-window-economics.md` |
| 96.6 | A | 0 | 92 | 100 | 88 | 100 | 100 | 100 | 100 | 40.8 | `docs/explainers/one-binary-one-surface.md` |
| 96.6 | A | 0 | 92 | 100 | 88 | 100 | 100 | 100 | 100 | 31.1 | `docs/explainers/routing.md` |
| 96.6 | A | 0 | 92 | 100 | 88 | 100 | 100 | 100 | 100 | 21.1 | `docs/explainers/tensorrt-llm-vs-sglang-and-fak.md` |
| 96.6 | A | 0 | 92 | 100 | 88 | 100 | 100 | 100 | 100 | 39.7 | `docs/explainers/the-abstraction-map.md` |
| 96.6 | A | 0 | 92 | 100 | 88 | 100 | 100 | 100 | 100 | 45.1 | `docs/explainers/what-is-a-cuda-kernel.md` |
| 96.6 | A | 0 | 92 | 100 | 88 | 100 | 100 | 100 | 100 | 31.1 | `docs/fak/claude-gemini-gcp.md` |
| 96.6 | A | 0 | 92 | 100 | 88 | 100 | 100 | 100 | 100 | 33.2 | `docs/fak/dojo-rsi-loop.md` |
| 96.6 | A | 0 | 92 | 100 | 88 | 100 | 100 | 100 | 100 | 32.2 | `docs/fak/gcp-tier2-control-vm.md` |
| 96.6 | A | 0 | 92 | 100 | 88 | 100 | 100 | 100 | 100 | 31.1 | `docs/fak/guard-hop-rsi-loop.md` |
| 96.6 | A | 0 | 92 | 100 | 88 | 100 | 100 | 100 | 100 | 32.2 | `docs/fak/mcp-registry.md` |
| 96.6 | A | 0 | 92 | 100 | 88 | 100 | 100 | 100 | 100 | 21.1 | `docs/fak/session-control.md` |
| 96.8 | A | 0 | 90 | 100 | 100 | 100 | 90 | 100 | 100 | 35.4 | `docs/FAQ.md` |
| 96.8 | A | 0 | 100 | 100 | 100 | 85 | 90 | 100 | 100 | 24.3 | `docs/explainers/hardware-isa/level-3-the-gpu-isa-ptx-sass-and-tensor-cores.md` |
| 96.8 | A | 0 | 90 | 100 | 100 | 100 | 90 | 100 | 100 | 32.2 | `docs/fak/faq.md` |
| 97.0 | A | 0 | 100 | 100 | 88 | 100 | 90 | 100 | 100 | 21.1 | `docs/explainers/compounding-benefits-of-a-saved-call.md` |
| 97.0 | A | 0 | 100 | 100 | 88 | 100 | 90 | 100 | 100 | 27.6 | `docs/explainers/hardware-limits-and-capacity.md` |
| 97.0 | A | 0 | 100 | 100 | 88 | 100 | 90 | 100 | 100 | 34.3 | `docs/fak/advanced-topics.md` |
| 97.0 | A | 0 | 100 | 100 | 88 | 100 | 90 | 100 | 100 | 33.2 | `docs/fak/agent-framework-integration.md` |
| 97.0 | A | 0 | 100 | 100 | 88 | 100 | 90 | 100 | 100 | 36.5 | `docs/fak/agent-integration-architecture.md` |
| 97.0 | A | 0 | 100 | 100 | 88 | 100 | 90 | 100 | 100 | 40.8 | `docs/fak/always-on-dogfood-server.md` |
| 97.0 | A | 0 | 100 | 100 | 88 | 100 | 90 | 100 | 100 | 33.2 | `docs/fak/claude-glm-gcp.md` |
| 97.0 | A | 0 | 100 | 100 | 88 | 100 | 90 | 100 | 100 | 32.2 | `docs/fak/claude-mac.md` |
| 97.0 | A | 0 | 100 | 100 | 88 | 100 | 90 | 100 | 100 | 32.2 | `docs/fak/dojo.md` |
| 97.0 | A | 0 | 100 | 100 | 88 | 100 | 90 | 100 | 100 | 33.2 | `docs/fak/guard-verdict-rsi-loop.md` |
| 97.0 | A | 0 | 100 | 100 | 88 | 100 | 90 | 100 | 100 | 32.2 | `docs/fak/in-kernel-model.md` |
| 97.0 | A | 0 | 100 | 100 | 88 | 100 | 90 | 100 | 100 | 32.2 | `docs/fak/lab-dev-loop.md` |
| 97.0 | A | 0 | 100 | 100 | 88 | 100 | 90 | 100 | 100 | 30.0 | `docs/fak/neo-cloud-deploy.md` |
| 97.0 | A | 0 | 100 | 100 | 88 | 100 | 90 | 100 | 100 | 32.2 | `docs/fak/node-macos-a-activation.md` |
| 97.0 | A | 0 | 100 | 100 | 88 | 100 | 90 | 100 | 100 | 32.2 | `docs/fak/node-setup.md` |
| 97.0 | A | 0 | 100 | 100 | 88 | 100 | 90 | 100 | 100 | 45.1 | `docs/fak/observability.md` |
| 97.0 | A | 0 | 100 | 100 | 88 | 100 | 90 | 100 | 100 | 31.1 | `docs/fak/qwen36-a100-gcp.md` |
| 97.0 | A | 0 | 100 | 100 | 88 | 100 | 90 | 100 | 100 | 33.2 | `docs/fak/server-troubleshooting.md` |
| 97.2 | A | 0 | 92 | 100 | 100 | 100 | 90 | 100 | 100 | 31.1 | `docs/explainers/god-file-growth-gate.md` |
| 97.2 | A | 0 | 92 | 100 | 100 | 100 | 90 | 100 | 100 | 32.2 | `docs/fak/opencode-glm-guard.md` |
| 97.6 | A | 0 | 100 | 100 | 100 | 100 | 80 | 100 | 100 | 20.0 | `docs/run-the-demos.md` |
| 98.0 | A | 0 | 100 | 100 | 100 | 85 | 100 | 100 | 100 | 23.2 | `docs/explainers/mixed-precision/level-2-number-formats.md` |
| 98.2 | A | 0 | 100 | 100 | 88 | 100 | 100 | 100 | 100 | 34.3 | `docs/MEMORY-LAYERS-EXPLAINER.md` |
| 98.2 | A | 0 | 100 | 90 | 100 | 100 | 100 | 100 | 100 | 31.1 | `docs/concepts-and-story.md` |
| 98.2 | A | 0 | 100 | 90 | 100 | 100 | 100 | 100 | 100 | 32.2 | `docs/explainers/README.md` |
| 98.2 | A | 0 | 100 | 100 | 88 | 100 | 100 | 100 | 100 | 39.7 | `docs/explainers/accelerators/README.md` |
| 98.2 | A | 0 | 100 | 100 | 88 | 100 | 100 | 100 | 100 | 25.4 | `docs/explainers/accelerators/level-2-the-hal-seam.md` |
| 98.2 | A | 0 | 100 | 100 | 88 | 100 | 100 | 100 | 100 | 24.3 | `docs/explainers/accelerators/level-3-vulkan-how-it-works.md` |
| 98.2 | A | 0 | 100 | 100 | 88 | 100 | 100 | 100 | 100 | 23.2 | `docs/explainers/accelerators/level-4-rocm-how-it-works.md` |
| 98.2 | A | 0 | 100 | 100 | 88 | 100 | 100 | 100 | 100 | 22.2 | `docs/explainers/context-tape-visuals.md` |
| 98.2 | A | 0 | 100 | 100 | 88 | 100 | 100 | 100 | 100 | 34.3 | `docs/explainers/cross-platform-spine.md` |
| 98.2 | A | 0 | 100 | 100 | 88 | 100 | 100 | 100 | 100 | 38.6 | `docs/explainers/default-deny-vs-classifier.md` |
| 98.2 | A | 0 | 100 | 100 | 88 | 100 | 100 | 100 | 100 | 39.7 | `docs/explainers/engineering-is-building-loops.md` |
| 98.2 | A | 0 | 100 | 100 | 88 | 100 | 100 | 100 | 100 | 23.2 | `docs/explainers/fleet-benchmarks.md` |
| 98.2 | A | 0 | 100 | 100 | 88 | 100 | 100 | 100 | 100 | 22.2 | `docs/explainers/gateway.md` |
| 98.2 | A | 0 | 100 | 90 | 100 | 100 | 100 | 100 | 100 | 38.6 | `docs/explainers/hardware-isa/README.md` |
| 98.2 | A | 0 | 100 | 100 | 88 | 100 | 100 | 100 | 100 | 40.8 | `docs/explainers/kernel-development/README.md` |
| 98.2 | A | 0 | 100 | 100 | 88 | 100 | 100 | 100 | 100 | 22.2 | `docs/explainers/kernel-development/level-6-the-kernel-development-frontier.md` |
| 98.2 | A | 0 | 100 | 90 | 100 | 100 | 100 | 100 | 100 | 38.6 | `docs/explainers/long-session-economics.md` |
| 98.2 | A | 0 | 100 | 100 | 88 | 100 | 100 | 100 | 100 | 35.4 | `docs/explainers/long-sessions-keep-the-cache-hit.md` |
| 98.2 | A | 0 | 100 | 100 | 88 | 100 | 100 | 100 | 100 | 38.6 | `docs/explainers/memory-hierarchy/README.md` |
| 98.2 | A | 0 | 100 | 100 | 88 | 100 | 100 | 100 | 100 | 40.8 | `docs/explainers/mixed-precision/README.md` |
| 98.2 | A | 0 | 100 | 100 | 88 | 100 | 100 | 100 | 100 | 23.2 | `docs/explainers/mixed-precision/level-1-what-is-a-bit-and-why-fewer.md` |
| 98.2 | A | 0 | 100 | 100 | 88 | 100 | 100 | 100 | 100 | 23.2 | `docs/explainers/mixed-precision/level-4-the-precision-machinery.md` |
| 98.2 | A | 0 | 100 | 100 | 88 | 100 | 100 | 100 | 100 | 22.2 | `docs/explainers/mixed-precision/level-5-the-evidence-boundary.md` |
| 98.2 | A | 0 | 100 | 90 | 100 | 100 | 100 | 100 | 100 | 45.1 | `docs/explainers/policy-in-the-kernel.md` |
| 98.2 | A | 0 | 100 | 100 | 88 | 100 | 100 | 100 | 100 | 26.5 | `docs/explainers/runtime-vs-client.md` |
| 98.2 | A | 0 | 100 | 100 | 88 | 100 | 100 | 100 | 100 | 23.2 | `docs/explainers/tokenization/level-1-what-is-a-tokenizer.md` |
| 98.2 | A | 0 | 100 | 100 | 88 | 100 | 100 | 100 | 100 | 24.3 | `docs/explainers/tokenization/level-3-the-merge-loop-by-hand.md` |
| 98.2 | A | 0 | 100 | 100 | 88 | 100 | 100 | 100 | 100 | 23.2 | `docs/explainers/tokenization/level-4-why-the-merge-loop-is-the-hot-path.md` |
| 98.2 | A | 0 | 100 | 100 | 88 | 100 | 100 | 100 | 100 | 22.2 | `docs/explainers/tokenization/level-5-the-tokenizer-frontier.md` |
| 98.2 | A | 0 | 100 | 100 | 88 | 100 | 100 | 100 | 100 | 38.6 | `docs/explainers/verify-dont-trust.md` |
| 98.2 | A | 0 | 100 | 90 | 100 | 100 | 100 | 100 | 100 | 31.1 | `docs/explainers/why-fak-the-market-moment.md` |
| 98.2 | A | 0 | 100 | 90 | 100 | 100 | 100 | 100 | 100 | 61.1 | `docs/fak/README.md` |
| 98.2 | A | 0 | 100 | 100 | 88 | 100 | 100 | 100 | 100 | 38.6 | `docs/fak/concept-glossary.md` |
| 98.2 | A | 0 | 100 | 100 | 88 | 100 | 100 | 100 | 100 | 31.1 | `docs/fak/green-gate-budget.md` |
| 98.2 | A | 0 | 100 | 100 | 88 | 100 | 100 | 100 | 100 | 34.3 | `docs/fak/org-policy-plane.md` |
| 98.4 | A | 0 | 92 | 100 | 100 | 100 | 100 | 100 | 100 | 69.7 | `LEARNING-PATH.md` |
| 98.4 | A | 0 | 92 | 100 | 100 | 100 | 100 | 100 | 100 | 63.2 | `START-HERE.md` |
| 98.4 | A | 0 | 92 | 100 | 100 | 100 | 100 | 100 | 100 | 31.1 | `docs/explainers/agent-runtime.md` |
| 98.4 | A | 0 | 92 | 100 | 100 | 100 | 100 | 100 | 100 | 34.1 | `docs/explainers/multi-gpu-tensor-parallelism.md` |
| 98.8 | A | 0 | 100 | 100 | 100 | 100 | 90 | 100 | 100 | 34.3 | `GETTING-STARTED.md` |
| 98.8 | A | 0 | 100 | 100 | 100 | 100 | 90 | 100 | 100 | 31.1 | `INSTALL.md` |
| 98.8 | A | 0 | 100 | 100 | 100 | 100 | 90 | 100 | 100 | 25.4 | `docs/explainers/caching/level-4-kernel-kv-cache.md` |
| 98.8 | A | 0 | 100 | 100 | 100 | 100 | 90 | 100 | 100 | 25.4 | `docs/explainers/hardware-isa/level-5-why-the-isa-is-not-the-whole-story.md` |
| 98.8 | A | 0 | 100 | 100 | 100 | 100 | 90 | 100 | 100 | 26.5 | `docs/explainers/inference-parallelism/level-3-pipeline-parallelism.md` |
| 98.8 | A | 0 | 100 | 100 | 100 | 100 | 90 | 100 | 100 | 25.4 | `docs/explainers/inference-parallelism/level-4-tensor-and-expert-parallelism.md` |
| 98.8 | A | 0 | 100 | 100 | 100 | 100 | 90 | 100 | 100 | 23.2 | `docs/explainers/inference-parallelism/level-5-the-parallelism-frontier.md` |
| 98.8 | A | 0 | 100 | 100 | 100 | 100 | 90 | 100 | 100 | 23.2 | `docs/explainers/kernel-development/level-2-the-registration-seam.md` |
| 98.8 | A | 0 | 100 | 100 | 100 | 100 | 90 | 100 | 100 | 23.2 | `docs/explainers/kernel-development/level-4-prove-faster.md` |
| 98.8 | A | 0 | 100 | 100 | 100 | 100 | 90 | 100 | 100 | 23.2 | `docs/explainers/kernel-development/level-5-the-adjudicator-rung.md` |
| 98.8 | A | 0 | 100 | 100 | 100 | 100 | 90 | 100 | 100 | 48.4 | `docs/explainers/tool-call-is-a-syscall.md` |
| 98.8 | A | 0 | 100 | 100 | 100 | 100 | 90 | 100 | 100 | 32.2 | `docs/fak/edge-quickstart.md` |
| 98.8 | A | 0 | 100 | 100 | 100 | 100 | 90 | 100 | 100 | 32.2 | `docs/fak/governed-agent-quickstart.md` |
| 98.8 | A | 0 | 100 | 100 | 100 | 100 | 90 | 100 | 100 | 33.2 | `docs/fak/mac-agent-ui.md` |
| 98.8 | A | 0 | 100 | 100 | 100 | 100 | 90 | 100 | 100 | 34.3 | `docs/fak/migration-guide.md` |
| 98.8 | A | 0 | 100 | 100 | 100 | 100 | 90 | 100 | 100 | 44.1 | `docs/fak/policy-guide.md` |
| 98.8 | A | 0 | 100 | 100 | 100 | 100 | 90 | 100 | 100 | 46.2 | `docs/fak/security.md` |
| 98.8 | A | 0 | 100 | 100 | 100 | 100 | 90 | 100 | 100 | 48.4 | `docs/fak/server-quickstart.md` |
| 98.8 | A | 0 | 100 | 100 | 100 | 100 | 90 | 100 | 100 | 48.4 | `docs/fak/tutorial.md` |
| 100.0 | A | 0 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 34.3 | `docs/CONTEXT-IS-NOT-MEMORY.md` |
| 100.0 | A | 0 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 23.2 | `docs/explainers/accelerators/level-1-why-not-just-cpu.md` |
| 100.0 | A | 0 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 24.3 | `docs/explainers/accelerators/level-5-vulkan-vs-rocm.md` |
| 100.0 | A | 0 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 60.3 | `docs/explainers/addressable-kv-cache.md` |
| 100.0 | A | 0 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 23.2 | `docs/explainers/caching/level-1-what-is-caching.md` |
| 100.0 | A | 0 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 24.3 | `docs/explainers/caching/level-2-managed-cache-in-practice.md` |
| 100.0 | A | 0 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 24.3 | `docs/explainers/caching/level-3-cache-economics-and-the-wire.md` |
| 100.0 | A | 0 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 23.2 | `docs/explainers/caching/level-5-the-caching-frontier.md` |
| 100.0 | A | 0 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 36.5 | `docs/explainers/context-shedding.md` |
| 100.0 | A | 0 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 24.3 | `docs/explainers/hardware-isa/level-1-what-is-an-instruction-set.md` |
| 100.0 | A | 0 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 25.4 | `docs/explainers/hardware-isa/level-2-simd-one-instruction-many-numbers.md` |
| 100.0 | A | 0 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 24.3 | `docs/explainers/hardware-isa/level-4-how-the-isa-reaches-silicon.md` |
| 100.0 | A | 0 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 37.6 | `docs/explainers/inference-parallelism/README.md` |
| 100.0 | A | 0 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 25.4 | `docs/explainers/inference-parallelism/level-1-what-is-inference.md` |
| 100.0 | A | 0 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 24.3 | `docs/explainers/inference-parallelism/level-2-the-layer-loop.md` |
| 100.0 | A | 0 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 48.4 | `docs/explainers/inference/README.md` |
| 100.0 | A | 0 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 23.2 | `docs/explainers/inference/level-1-what-is-a-token.md` |
| 100.0 | A | 0 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 24.3 | `docs/explainers/inference/level-2-the-forward-pass.md` |
| 100.0 | A | 0 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 24.3 | `docs/explainers/inference/level-3-inside-a-block.md` |
| 100.0 | A | 0 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 25.4 | `docs/explainers/inference/level-4-prefill-vs-decode.md` |
| 100.0 | A | 0 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 24.3 | `docs/explainers/inference/level-5-at-scale.md` |
| 100.0 | A | 0 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 23.2 | `docs/explainers/kernel-development/level-1-what-is-kernel-development.md` |
| 100.0 | A | 0 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 24.3 | `docs/explainers/kernel-development/level-3-prove-correct.md` |
| 100.0 | A | 0 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 23.2 | `docs/explainers/memory-hierarchy/level-1-what-is-a-buffer.md` |
| 100.0 | A | 0 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 23.2 | `docs/explainers/memory-hierarchy/level-2-the-forward-pass-as-buffer-moves.md` |
| 100.0 | A | 0 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 23.2 | `docs/explainers/memory-hierarchy/level-3-sram-and-the-kernels-own-scratch.md` |
| 100.0 | A | 0 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 23.2 | `docs/explainers/memory-hierarchy/level-4-hbm-vram-and-the-roofline.md` |
| 100.0 | A | 0 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 22.2 | `docs/explainers/memory-hierarchy/level-5-host-dram-offload-and-the-frontier.md` |
| 100.0 | A | 0 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 24.3 | `docs/explainers/mixed-precision/level-3-what-to-quantize.md` |
| 100.0 | A | 0 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 37.6 | `docs/explainers/tokenization/README.md` |
| 100.0 | A | 0 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 23.2 | `docs/explainers/tokenization/level-2-the-three-stages.md` |
| 100.0 | A | 0 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 33.2 | `docs/explainers/what-a-saved-token-is-worth.md` |
| 100.0 | A | 0 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 24.3 | `docs/explainers/what-is-managed-cache.md` |
| 100.0 | A | 0 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 30.0 | `docs/fak/concept-usage-scorecard.md` |
| 100.0 | A | 0 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 32.2 | `docs/fak/documentation-roadmap.md` |
| 100.0 | A | 0 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 30.0 | `docs/fak/loop-scorecard.md` |
| 100.0 | A | 0 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 31.1 | `docs/fak/mystery-free-adjustment-atlas.md` |
| 100.0 | A | 0 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 22.2 | `docs/fak/scrubbing-real-values.md` |
| 100.0 | A | 0 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 36.5 | `docs/glossary.md` |

## Learning-debt work-list

### `docs/fak/concept-runtime-burndown-20261102.md` — 3 defect(s), score 75.4 (C)
- orientation: no orientation signpost (audience / prereq / TL;DR / 'you'll be able to')
- orientation: no outbound link to another doc (a navigational dead-end in the curriculum)
- runnable: teaches no runnable command or code block (prose-only)

### `docs/fak/feature-activation.md` — 3 defect(s), score 81.8 (B)
- orientation: no outbound link to another doc (a navigational dead-end in the curriculum)
- runnable: teaches no runnable command or code block (prose-only)
- structure: long doc with no section headings (## …)

### `docs/fak/team-node-deployment.md` — 3 defect(s), score 88.6 (B)
- links: dead link: ../deploy/systemd/fak.env.example
- links: dead link: ../deploy/systemd/fak.service
- links: dead link: ../tools/grafana/README.md

### `docs/explainers/tool-feedback-continuation.md` — 1 defect(s), score 84.6 (B)
- orientation: no orientation signpost (audience / prereq / TL;DR / 'you'll be able to')

### `docs/fak/deployment-guide.md` — 1 defect(s), score 95.4 (A)
- freshness: stale install pin 0.47.0 (VERSION is 0.57.0)

### `docs/fak/mac-local-models.md` — 1 defect(s), score 95.6 (A)
- freshness: stale install pin 0.54.0 (VERSION is 0.57.0)

### coverage
- orphan lesson (unreachable from any front door): docs/fak/concept-runtime-burndown-20261102.md
- orphan lesson (unreachable from any front door): docs/fak/feature-activation.md
- shipped-but-untaught surface: fak agentdemo (added since learning-scorecard stamp 2026-09-04)
- shipped-but-untaught surface: fak agentx (added since learning-scorecard stamp 2026-09-04)
- shipped-but-untaught surface: fak amdgpu (added since learning-scorecard stamp 2026-09-04)
- shipped-but-untaught surface: fak arms (added since learning-scorecard stamp 2026-09-04)
- shipped-but-untaught surface: fak baseline-arms (added since learning-scorecard stamp 2026-09-04)
- shipped-but-untaught surface: fak cache (added since learning-scorecard stamp 2026-09-04)
- shipped-but-untaught surface: fak cachedoc-numbers-audit (added since learning-scorecard stamp 2026-09-04)
- shipped-but-untaught surface: fak check-cache-headlines (added since learning-scorecard stamp 2026-09-04)
- shipped-but-untaught surface: fak claude (added since learning-scorecard stamp 2026-09-04)
- shipped-but-untaught surface: fak client (added since learning-scorecard stamp 2026-09-04)
- shipped-but-untaught surface: fak completion (added since learning-scorecard stamp 2026-09-04)
- shipped-but-untaught surface: fak computetune (added since learning-scorecard stamp 2026-09-04)
- shipped-but-untaught surface: fak conceptdrift (added since learning-scorecard stamp 2026-09-04)
- shipped-but-untaught surface: fak containment (added since learning-scorecard stamp 2026-09-04)
- shipped-but-untaught surface: fak conversation-profile (added since learning-scorecard stamp 2026-09-04)
- shipped-but-untaught surface: fak debt-orchestrator (added since learning-scorecard stamp 2026-09-04)
- shipped-but-untaught surface: fak doc-freshness (added since learning-scorecard stamp 2026-09-04)
- shipped-but-untaught surface: fak docs-freshness (added since learning-scorecard stamp 2026-09-04)
- shipped-but-untaught surface: fak dogfood-coverage (added since learning-scorecard stamp 2026-09-04)
- shipped-but-untaught surface: fak expert-trace (added since learning-scorecard stamp 2026-09-04)
- shipped-but-untaught surface: fak fresh-status (added since learning-scorecard stamp 2026-09-04)
- shipped-but-untaught surface: fak godfile-ceiling (added since learning-scorecard stamp 2026-09-04)
- shipped-but-untaught surface: fak harness-hint (added since learning-scorecard stamp 2026-09-04)
- shipped-but-untaught surface: fak harnesshint (added since learning-scorecard stamp 2026-09-04)
- shipped-but-untaught surface: fak hil (added since learning-scorecard stamp 2026-09-04)
- shipped-but-untaught surface: fak hwspecs (added since learning-scorecard stamp 2026-09-04)
- shipped-but-untaught surface: fak issue-lanes (added since learning-scorecard stamp 2026-09-04)
- shipped-but-untaught surface: fak issue-queue (added since learning-scorecard stamp 2026-09-04)
- shipped-but-untaught surface: fak issue-stats (added since learning-scorecard stamp 2026-09-04)
- shipped-but-untaught surface: fak issues-recent (added since learning-scorecard stamp 2026-09-04)
- shipped-but-untaught surface: fak issues-solved (added since learning-scorecard stamp 2026-09-04)
- shipped-but-untaught surface: fak keys (added since learning-scorecard stamp 2026-09-04)
- shipped-but-untaught surface: fak macobs (added since learning-scorecard stamp 2026-09-04)
- shipped-but-untaught surface: fak managedocs (added since learning-scorecard stamp 2026-09-04)
- shipped-but-untaught surface: fak observer (added since learning-scorecard stamp 2026-09-04)
- shipped-but-untaught surface: fak optimize (added since learning-scorecard stamp 2026-09-04)
- shipped-but-untaught surface: fak pack (added since learning-scorecard stamp 2026-09-04)
- shipped-but-untaught surface: fak perf (added since learning-scorecard stamp 2026-09-04)
- shipped-but-untaught surface: fak pi (added since learning-scorecard stamp 2026-09-04)
- shipped-but-untaught surface: fak readme (added since learning-scorecard stamp 2026-09-04)
- shipped-but-untaught surface: fak readmenext (added since learning-scorecard stamp 2026-09-04)
- shipped-but-untaught surface: fak redteam (added since learning-scorecard stamp 2026-09-04)
- shipped-but-untaught surface: fak results (added since learning-scorecard stamp 2026-09-04)
- shipped-but-untaught surface: fak roofline (added since learning-scorecard stamp 2026-09-04)
- shipped-but-untaught surface: fak router (added since learning-scorecard stamp 2026-09-04)
- shipped-but-untaught surface: fak scratch (added since learning-scorecard stamp 2026-09-04)
- shipped-but-untaught surface: fak sensecheck (added since learning-scorecard stamp 2026-09-04)
- shipped-but-untaught surface: fak serve-receipt (added since learning-scorecard stamp 2026-09-04)
- shipped-but-untaught surface: fak sessionsearch (added since learning-scorecard stamp 2026-09-04)
- shipped-but-untaught surface: fak supervisor-seat (added since learning-scorecard stamp 2026-09-04)
- shipped-but-untaught surface: fak sweep-speculative (added since learning-scorecard stamp 2026-09-04)

### stamp freshness
- stale-stamp: docs/LEARNING-SCORECARD.md stamp 2026-09-04 is 81d old (cadence 1d) and the learning corpus changed since

