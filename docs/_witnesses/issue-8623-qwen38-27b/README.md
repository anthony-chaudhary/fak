---
title: "Issue #8623 — exact Qwen3.8-27B paired confirmation"
description: "Verdict: PASS on the declared exact-target envelope."
---
# Issue #8623 — exact Qwen3.8-27B paired confirmation

**Verdict: PASS on the declared exact-target envelope.**

The full frozen ladder culminated at `Qwen/Qwen3.8-27B@1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`. The exact BF16 checkpoint ran on two datacenter GPUs with tensor parallelism, and both arms met the correctness floor.

- Baseline (`enable_thinking=true`): 3/3 correct; nearest-rank p95 3378.0197 ms.
- Candidate (`enable_thinking=false`): 3/3 correct; nearest-rank p95 376.1818 ms.
- Observed p95 reduction: 88.86%, above the frozen 5% gate.
- Memory: service allocation used 35,725 MiB per GPU and left 4,716 MiB free per GPU.
- Shared controls: one resident checkpoint, deterministic greedy decode, seed 0, identical prompt, standalone-answer scorer over the decoded SSE stream, client semantic early-stop, and 256-token ceiling.

Reproduce the final ladder decision:

```text
fak model qwen38-ladder --evidence docs/_witnesses/issue-8623-qwen38-27b/evidence-complete.json
```

The PASS is scoped to this BF16/TP2/datacenter GPU envelope, arithmetic corpus, and p95 time-to-first-correct-answer metric. It does not establish GGUF, FP8, Metal, tool-call, structured-output, broad quality, or production-readiness claims from parent #8011.

