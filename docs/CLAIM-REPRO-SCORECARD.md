---
title: "fak claim-reproducibility scorecard — are claims falsifiable from a clean clone?"
description: " fak's deterministic claim-reproducibility scorecard: validates that every witness in CLAIMS.md and BENCHMARK-AUTHORITY.md resolves to a real artifact, test, or command path."
---

# Claim-reproducibility scorecard

<!-- claim-repro-scorecard: 2026-06-28 · process: tools/claim_repro_scorecard.py -->

This scorecard validates that every witness handle in ``CLAIMS.md`` (``[SHIPPED]``/``[SIMULATED]``/``[STUB]`` claims) and every artifact path or ``Reproduce:`` command in ``BENCHMARK-AUTHORITY.md`` is **resolvable by an outsider from a clean clone**. An un-falsifiable claim — a ``Witness: TestFooBar`` that names a non-existent test, or a ``Reproduce: go run ./cmd/gone`` pointing at a deleted binary — is the worst failure mode for a skeptical reader, because it looks checkable and isn't.

> Regenerate: ``python tools/claim_repro_scorecard.py --markdown --stamp DATE > docs/CLAIM-REPRO-SCORECARD.md``

## Headline

| Metric | Value |
|---|---|
| **Un-falsifiable claims (total HARD defects)** | **3** |
| Composite score | 58.0/100 (grade F) |
| Advisory (soft) signals | 0 |

## Per-KPI

Two KPIs, each 0–100. ``debt`` = units of HARD un-falsifiable claims in that KPI.

| KPI | Score | Debt | Detail |
|---|---:|:--:|---|
| ``benchmarks`` | 40 | 2 | 151 benchmarks, 2 un-falsifiable |
| ``claims`` | 70 | 1 | 160 claims, 1 un-falsifiable |

## Un-falsifiable claim work-list

### ``benchmarks`` — 2 defect(s), score 40
- un-falsifiable benchmark: | Session value-add (SmolLM2 P=512, re-measured) | **5.3–7.4×** | SmolLM2-135M Q8 | Naive stateless | `885ae8a` | `bench — missing artifact: benchmark-run-opencode-20260619/sessionbench-smollm2-135m-q8-authority.json
- un-falsifiable benchmark: | **Pure-kernel admission latency (M3 Pro)** | **1.8–14 µs** scan · 3.3–15.8 µs Admit · 29–87 µs chain | ctxmmu / normga — missing artifact: MAC-M3PRO-KERNEL-BENCH-2026-06-20.md

### ``claims`` — 1 defect(s), score 70
- un-falsifiable claim: - [SHIPPED] **Serving-latency observability — percentile-capable TTFT / TPOT / end-to-end histograms on `/metrics`.** Th — -run 'TestInferenceLatencyHistograms' matches no test in internal/gateway, test function not found: TestInferenceLatencyHistograms

