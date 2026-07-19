---
title: "fak claim-reproducibility scorecard — are claims falsifiable from a clean clone?"
description: " fak's deterministic claim-reproducibility scorecard: validates that every witness in CLAIMS.md and BENCHMARK-AUTHORITY.md resolves to a real artifact, test, or command path."
---

# Claim-reproducibility scorecard

<!-- claim-repro-scorecard: 2026-07-18 · process: tools/claim_repro_scorecard.py -->

This scorecard validates that every witness handle in ``CLAIMS.md`` (``[SHIPPED]``/``[SIMULATED]``/``[STUB]`` claims) and every artifact path or ``Reproduce:`` command in ``BENCHMARK-AUTHORITY.md`` is **resolvable by an outsider from a clean clone**. An un-falsifiable claim — a ``Witness: TestFooBar`` that names a non-existent test, or a ``Reproduce: go run ./cmd/gone`` pointing at a deleted binary — is the worst failure mode for a skeptical reader, because it looks checkable and isn't.

> Regenerate: ``python tools/claim_repro_scorecard.py --markdown --stamp DATE > docs/CLAIM-REPRO-SCORECARD.md``

## Headline

**Claim-repro-debt: 0** unresolvable witness(es) — the count of witnesses that do NOT resolve from a clean clone (a ``Witness: TestFooBar`` naming a non-existent test, a ``Reproduce: go run ./cmd/gone`` pointing at a deleted binary). This is the **primary metric: unbounded, drive it to 0.**

| Metric (primary first) | Value |
|---|---|
| **Claim-repro-debt — unresolvable witnesses (unbounded; drive to 0)** | **0** |
| Advisory (soft) signals | 0 |

Legacy bounded score (saturates; not the driver): 100.0/100 (grade A).

## Per-KPI

Two KPIs, each 0–100. ``debt`` = units of HARD un-falsifiable claims in that KPI.

| KPI | Score | Debt | Detail |
|---|---:|:--:|---|
| ``claims`` | 100 | 0 | 186 claims, all falsifiable |
| ``benchmarks`` | 100 | 0 | 169 benchmarks, all falsifiable |

## Un-falsifiable claim work-list

No un-falsifiable claims: every witness resolves. 🎉

