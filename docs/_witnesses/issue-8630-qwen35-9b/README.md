# Issue #8630 — Qwen3.5-9B quality-proxy result

**Verdict: PROMOTE to pinned Qwen3.5-27B scale rehearsal.**

The frozen semantic time-to-correct-answer experiment from #8629 was run unchanged at `Qwen/Qwen3.5-9B@c202236235762e1c871ad0ccb60c8ee5ba337b9a`.

- Baseline (`enable_thinking=true`): 3/3 correct; nearest-rank p95 45372.2273 ms.
- Candidate (`enable_thinking=false`): 3/3 correct; nearest-rank p95 948.2820 ms.
- Observed p95 reduction: 97.91%.
- Shared controls: float32, resident weights, deterministic greedy decode, seed 0, identical arithmetic prompt, standalone-answer scorer, semantic early-stop, and 256-token safety ceiling.

Reproduce:

```text
fak model qwen38-ladder --evidence docs/_witnesses/issue-8630-qwen35-9b/evidence-through-9b.json
```

The result proves the concept survives the 9B medium-scale/untied-embedding proxy. It does not prove 27B geometry or memory, Qwen3.8 weights, exact target quality, or target performance.
