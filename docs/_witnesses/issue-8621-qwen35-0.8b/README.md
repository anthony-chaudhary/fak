---
title: "Issue #8621 — Qwen3.5-0.8B paired smoke result"
description: "Reference documentation for Issue #8621 — Qwen3.5-0.8B paired smoke result, preserving the page's implementation details, evidence, and operating context."
---

# Issue #8621 — Qwen3.5-0.8B paired smoke result

**Verdict: HOLD.** The candidate passed 3/3 and reduced p95 generation wall time from 2509.8482 ms to 613.7351 ms, but the baseline passed 0/3. The ladder now rejects that invalid attribution because both arms must meet the stage's absolute correctness floor before performance can promote.

- Model: `Qwen/Qwen3.5-0.8B@2fc06364715b967f1860aea9cf38778875588b17`
- Concept: pin campaign requests to `enable_thinking=false`
- Runtime pair: `764328d54289e5685ac1ca12878c0c39d00e9c76` → `b0ce51b599718cb1a08a886ac61af928a5209b78`
- Corpus: one exact-output smoke prompt, SHA-256 in `evidence.json`
- Trials: three deterministic, interleaved trials per arm; seed 0; float32; Transformers 5.10.2; PyTorch 2.12.0 CPU
- Correctness scorer: decoded output stripped equals `Q38`
- Metric: nearest-rank p95 generation wall milliseconds

The baseline's thinking response exhausted the 32-token output budget before producing the answer. That means the experiment cannot separate a candidate improvement from an invalid baseline configuration. `evaluator-output.json` is the reproducible typed HOLD; `raw-run.json` preserves every trial and output.

Reproduce from repository root after building `fak`:

```text
fak model qwen38-ladder --evidence docs/_witnesses/issue-8621-qwen35-0.8b/evidence.json
```

The command exits 1 on HOLD by design. No proxy result here proves Qwen3.8 quality, identity, memory behavior, or performance.
