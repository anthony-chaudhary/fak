# SOTA vs fak — SWE-bench Verified Comparison

**Date:** 2026-06-20
**Status:** ✅ Infrastructure complete, awaiting real agent integration

## Executive Summary

| Aspect | SOTA Leaders | fak (Current) | fak (Potential) |
|--------|--------------|---------------|------------------|
| **Resolve Rate** | 72-77% | Not measured | TBD (needs real agent) |
| **Cost Efficiency** | Prefix prompt-cached per session | **+ cross-worker/session reuse (~1.1–1.2× at 2-4 workers)** | Scales with shared-prefix fraction & workers |
| **Cross-worker Reuse** | None (single-agent / single-instance) | **1.13-1.22x at 2-4 workers** | Scales with workers |
| **Turn Tax** | Already reused via prompt cache | **KV persistence (≈ parity)** | Eliminated |
| **Infrastructure** | mini-SWE-agent (100-line ReAct) | **Production-grade gateway** | Multi-model, multi-worker |

**Bottom line:** the SOTA leaders run on API models that **already prompt-cache the shared prefix across turns**, so the honest infra comparison is *not* a 20×. fak's incremental win is **cross-worker / cross-session prefix sharing — ~1.1–1.2× at 2-4 workers, growing with the shared-prefix fraction** — plus addressable eviction and a default-deny safety floor those stacks don't offer. (A 20–24× number exists only against a *naive* re-prefill-every-turn loop — a worst case no SOTA run ships — and is never the comparison.) With a competitive model (Claude/GPT class), fak could match SOTA resolve rates at a modestly lower and more *scalable* infra cost.

---

## Current SOTA Leaders (SWE-bench Verified)

Data from [swebench.com](https://www.swebench.com/) as of February 2026.

### Top 10 Models (mini-SWE-agent v2 harness)

| Rank | Model | % Resolved | Avg Cost $ | Org | Date |
|------|-------|------------|------------|-----|------|
| 1 | Claude 4.5 Opus (high reasoning) | **76.80%** | $0.75 | Anthropic | 2026-02-17 |
| 2 | Gemini 3 Flash (high reasoning) | **75.80%** | $0.36 | Google | 2026-02-17 |
| 3 | MiniMax M2.5 (high reasoning) | **75.80%** | $0.07 | MiniMax | 2026-02-17 |
| 4 | Claude Opus 4.6 | **75.60%** | $0.55 | Anthropic | 2026-02-17 |
| 5 | GPT-5-2 Codex | **72.80%** | $0.45 | OpenAI | 2026-02-19 |
| 6 | GLM-5 (high reasoning) | **72.80%** | $0.53 | Z-AI | 2026-02-17 |
| 7 | GPT-5-2 (high reasoning) | **72.80%** | $0.47 | OpenAI | 2026-02-17 |
| 8 | Claude 4.5 Sonnet (high reasoning) | **71.40%** | $0.66 | Anthropic | 2026-02-17 |
| 9 | Kimi K2.5 (high reasoning) | **70.80%** | $0.15 | Moonshot | 2026-02-17 |
| 10 | DeepSeek V3.2 (high reasoning) | **70.00%** | $0.45 | DeepSeek | 2026-02-17 |

### Common Infrastructure

**All SOTA methods use the same evaluation harness:**
- **mini-SWE-agent** — 100-line Python ReAct loop
- **Bash-only environment** — No tools, just LM + shell
- **Temperature 0.0** — Deterministic (for release 1.x)
- **KV reuse happens at the provider** — the harness re-sends the transcript each turn, but the API models it calls (Claude / GPT / Gemini) **prompt-cache the shared prefix** (system prompt + tool schemas + earlier turns), read back at ~0.1× cost. So SOTA runs do **not** re-prefill the shared prefix from scratch every turn; the naive "re-prefill everything" loop is a worst case nobody runs, not the SOTA baseline.

**This means the differences in resolve rate come from:**
1. Model capability (not infrastructure)
2. Prompt engineering (within mini-SWE-agent)
3. Reasoning mode ("high reasoning" flag)

---

## fak's Value Proposition

### Infrastructure Advantages (Measured)

From the smoke test on 5 instances (difficulty-derived geometry):

#### Prefill Work-Elimination (A/C Ratio)

| Workers | A (Naive) | B (Per-Agent KV) | C (fak Fused) | **A/C** | **B/C** |
|---------|-----------|-----------------|---------------|---------|---------|
| 1       | 1.04M tokens | 52.9K tokens | 52.9K tokens | **19.7x** | 1.00x |
| 2       | 2.09M tokens | 105.8K tokens | 93.3K tokens | **22.4x** | 1.13x |
| 4       | 4.17M tokens | 211.6K tokens | 174.1K tokens | **24.0x** | **1.22x** |

**Interpretation:**
- **A/C (Net Work-Elimination):** fak reduces 95%+ of prefill work vs naive re-prefetch-every-turn
- **B/C (Cross-Worker Reuse):** Shared prefix gives 1.22x benefit at 4 workers (the value stack)
- **A/B (Turn Tax):** 19.7x — re-prefill vs KV persistence, worker-independent

> **Which of these is the SOTA comparison? Only B/C.** A/C and A/B are against the *naive* re-prefill
> loop — a worst case no SOTA run ships. A real SOTA setup (provider prompt caching / per-instance
> prefix caching) already eliminates the same turn-tax, so fak's incremental infra win over it is the
> **cross-worker / cross-session B/C reuse** (~1.1–1.2× at 4 workers, larger as the shared-prefix
> fraction grows), **not** the 20–24× floor.

#### What This Means in Practice

For a full SWE-bench Verified run (500 instances, ~20 turns median):

| Approach | Prefill Tokens (est.) | vs the realistic SOTA |
|----------|----------------------|------------------------|
| Naive (re-prefill every turn) — *worst-case floor, not SOTA* | ~500M tokens | — |
| Per-Agent KV / provider prompt cache — **the SOTA baseline** | ~25M tokens | baseline |
| **fak (shared prefix across workers/sessions)** | **~21M tokens** | **~1.1–1.2× at small scale, rising with shared-prefix fraction** |

**On SWE-bench's ~500 instances:**
- SOTA methods **already reuse** the shared prefix (provider prompt caching / per-instance prefix caching) — they do *not* pay the full prefill tax every turn.
- fak adds **cross-worker and cross-session** sharing on top, so the same system prompt + tool schemas are shared across instances, not just across turns within one.
- Net vs the realistic SOTA: a **~1.1–1.2× infra reduction at a few workers, growing with the shared-prefix fraction** — *not* the 20–24× that only appears against the naive floor.

---

## Comparative Analysis

### What SOTA Does Better

1. **Model Integration** — Direct access to Claude/GPT/DeepSeek APIs
2. **Proven Resolve Rates** — 70-77% on the official leaderboard
3. **Simplicity** — 100-line Python agent, easy to run
4. **Ecosystem** — Extensive tooling, papers, community

### What fak Does Better

1. **Infrastructure Efficiency** — cross-worker / cross-session prefix sharing (~1.1–1.2× at a few workers over a SOTA stack that already prompt-caches; the 20–24× is only vs a naive re-prefill floor)
2. **Scalability** — Cross-worker reuse (1.22x at 4 workers), rising with the shared-prefix fraction
3. **Production Readiness** — Gateway architecture, multi-model support
4. **Session Persistence** — KV cache reuse across turns
5. **Adjudication** — In-process vs spawn-per-hook (measurable via `fak bench`)

### The Gap

| Aspect | SOTA | fak | Path to Parity |
|--------|------|-----|----------------|
| **Resolve Rate** | 70-77% | Not measured | Integrate real model + agent |
| **Model Access** | Direct API | Placeholder | Add Claude/GPT endpoint |
| **Agent Logic** | ReAct loop | Placeholder | Port mini-SWE-agent logic |

**Key insight:** fak's infrastructure advantages are orthogonal to model capability. With the same model (e.g., Claude 4.5 Opus), fak could achieve ~77% resolve rate at a modestly lower, *fleet-scaling* infrastructure cost — cross-worker/session prefix sharing on top of the prompt caching the SOTA already does (not the 20–24× that only holds vs a naive re-prefill loop).

---

## Projections

### Cost Comparison (Hypothetical Full Run)

Assuming:
- 500 instances (SWE-bench Verified)
- Median 20 turns per instance
- 200 tokens/turn (assistant) + 400 tokens/turn (tool results)
- Claude 4.5 Opus at $3/M input, $15/M output

| Approach | Input Tokens (billable) | Output Tokens | Est Cost |
|----------|-------------------------|---------------|----------|
| Naive, no reuse — *worst-case floor; no SOTA run pays this* | ~2.0B | ~1.0B | **~$21,000** |
| **Realistic SOTA** (provider prompt cache / per-instance prefix cache) | far lower — the shared prefix is cache-read at ~0.1× | ~1.0B | **a small fraction of the floor** |
| **fak** (+ cross-worker / cross-session sharing) | lower still by the shared-prefix fraction | ~1.0B | **~1.1–1.2× below the SOTA at a few workers** |

*Note: the "$21,000 → ~93% savings" figure is against the **naive** floor, which no SOTA run pays. Output tokens are identical (same generated code). Against the **realistic SOTA** — which already prompt-caches the shared prefix — fak's incremental win is sharing the prefix **across** workers and sessions (the B/C delta), not the 20× turn-tax a tuned engine already eliminates.*

### Performance Projection

If fak integrated a SOTA model:

| Metric | SOTA | fak (with SOTA model) |
|--------|------|----------------------|
| **Resolve Rate** | 70-77% | **70-77% (same)** |
| **Infrastructure Cost** | Already prompt-cached per session | **~1.1–1.2× lower at a few workers (cross-worker/session sharing); grows with shared-prefix fraction** |
| **Latency** | Baseline | **Lower (shared prefix prefilled once across workers)** |

**The win:** same quality; a cross-worker/session infra reduction that *scales with fleet size* — not the 20× that only holds against a naive re-prefill loop.

---

## Next Steps

### Immediate (Ship Real Agent)

1. **Fleet Gateway Integration**
   - Connect `fak swebench run --agent fleet` to real gateway
   - Implement actual agent loop (ReAct or similar)
   - Test with Claude 4.5 Opus endpoint

2. **Model Access**
   - Add Anthropic API key configuration
   - Implement proper tool calling (not placeholder)
   - Add trajectory capture for analysis

3. **Resolve Rate Measurement**
   - Run full smoke test with real model
   - Compare predictions to gold patches
   - Publish first fak resolve rate

### Medium (Competitive Evaluation)

1. **Head-to-Head with SOTA**
   - Run Claude 4.5 Opus via fak infrastructure
   - Compare resolve rate to mini-SWE-agent baseline
   - Quantify cost savings

2. **Full Dataset Run**
   - Scale from 5 instances to 500
   - Measure infrastructure cost at scale
   - Publish fak leaderboard entry

3. **Advanced Features**
   - Multi-worker comparison (1, 2, 4, 8 workers)
   - Real-time adjudication measurement
   - Prometheus metrics integration

### Long Term (Research Directions)

1. **Model-Specific Optimization**
   - Fine-tune prompts for each model class
   - Implement "high reasoning" mode equivalents
   - Ablate value stack benefits per model

2. **Beyond SWE-bench**
   - WebBench (frontier web tasks)
   - Custom enterprise benchmarks
   - Multi-modal code tasks

---

## Conclusion

**fak's infrastructure is ready for SOTA competition.** Its incremental infra win over the realistic SOTA — API models and engines that *already* prompt-cache the shared prefix — is cross-worker / cross-session prefix sharing (~1.1–1.2× at a few workers, growing with the shared-prefix fraction) plus addressable eviction and a safety floor; the 20–24× figure holds only against a naive re-prefill loop nobody ships. With proper model integration, fak could match the 70-77% resolve rate of current leaders at a modestly lower and more *scalable* infrastructure cost.

**The blocking factor is not infrastructure — it's agent implementation.** The placeholder runners need real model access and proper ReAct loop logic. Once that lands, fak can publish competitive resolve rates while demonstrating superior cost efficiency.

**Sources:**
- [SWE-bench Leaderboards](https://www.swebench.com/)
- [SWE-bench Verified](https://www.swebench.com/verified.html)
- [mini-SWE-agent Repository](https://github.com/princeton-nlp/SWE-agent)
