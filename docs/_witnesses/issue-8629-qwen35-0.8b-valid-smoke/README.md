---
title: "Issue #8629 — valid paired Qwen proxy climb"
description: "Reference documentation for Issue #8629 — valid paired Qwen proxy climb, preserving the page's implementation details, evidence, and operating context."
---

# Issue #8629 — valid paired Qwen proxy climb

**Verdict: PROMOTE through Qwen3.5-4B; next stage is pinned Qwen3.5-9B.**

The repaired experiment uses one deterministic arithmetic prompt, one semantic correctness rule, and one shared stop condition: stop when the decoded stream first contains the standalone correct answer `4`, with a shared 256-token safety ceiling. This measures time-to-first-correct-answer rather than letting the thinking baseline fail at an arbitrary short output cap.

| Stage | Baseline pass / p95 ms | Candidate pass / p95 ms | Observed reduction | Verdict |
|---|---:|---:|---:|---|
| Qwen3.5-0.8B | 3/3 / 6671.7386 | 3/3 / 183.7214 | 97.25% | PROMOTE |
| Qwen3.5-2B | 3/3 / 9946.7602 | 3/3 / 272.1130 | 97.26% | PROMOTE |
| Qwen3.5-4B | 3/3 / 17905.7870 | 3/3 / 517.8081 | 97.11% | PROMOTE |

The candidate is the committed `enable_thinking=false` campaign change (`b0ce51b599718cb1a08a886ac61af928a5209b78`) against parent baseline `764328d54289e5685ac1ca12878c0c39d00e9c76`. Each stage pins its ladder model revision and records its own environment hash and raw outputs.

Reproduce the latest gate:

```text
fak model qwen38-ladder --evidence docs/_witnesses/issue-8629-qwen35-0.8b-valid-smoke/evidence-through-4b.json
```

This evidence proves the concept survives smoke, behavior, and wider-tensor proxies. It does **not** prove 9B quality, 27B geometry/memory, Qwen3.8 weights, or exact-target performance. Those remain sequential gates.
