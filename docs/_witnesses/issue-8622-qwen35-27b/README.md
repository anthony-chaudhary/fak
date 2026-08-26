---
title: "Issue #8622 — Qwen3.5-27B scale rehearsal"
description: "Reference documentation for Issue #8622 — Qwen3.5-27B scale rehearsal, preserving the page's implementation details, evidence, and operating context."
---

# Issue #8622 — Qwen3.5-27B scale rehearsal

**Verdict: PROMOTE to exact pinned Qwen3.8-27B.**

The unchanged semantic time-to-correct-answer experiment passed at `Qwen/Qwen3.5-27B@fc05daec18b0a78c049392ed2e771dde82bdf654` on two datacenter GPUs with BF16 tensor parallelism.

- Baseline (`enable_thinking=true`): 3/3 correct; nearest-rank p95 6881.5310 ms.
- Candidate (`enable_thinking=false`): 3/3 correct; nearest-rank p95 386.2153 ms.
- Observed p95 reduction: 94.39%.
- Memory: the single-datacenter GPU BF16 attempt failed to fit; TP2 used 35,729 MiB per GPU after service allocation and left 4,712 MiB free per GPU.
- Shared controls: one resident checkpoint, deterministic greedy decode, seed 0, identical prompt, standalone-answer scorer over the decoded SSE stream, client semantic early-stop, and 256-token ceiling.

Reproduce the evaluator decision:

```text
fak model qwen38-ladder --evidence docs/_witnesses/issue-8622-qwen35-27b/evidence-through-27b.json
```

This proves the concept survives 27B Qwen3.5 geometry and the measured BF16 TP2 memory envelope. It does not prove Qwen3.8 weights, exact-target behavior, or exact-target performance.

