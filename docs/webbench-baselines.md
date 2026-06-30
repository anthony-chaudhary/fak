---
title: "fak WebBench Baselines: 8.8x Prefill Cut on WebVoyager (modeled)"
description: "fak's modeled WebVoyager prefill geometry: over the real 643-task set, a closed-form model puts the work-elimination at 8.8–9.7× vs the naive re-prefill floor (1.0–1.1× vs a tuned per-agent-KV stack). Not a wall-clock measurement."
---

# Frontier WebBench Baselines & SOTA Comparison

This page is fak's WebBench baseline comparison: a deterministic geometry model of the prefill-token work that a fused resident KV eliminates versus a naive per-turn re-prefill harness, computed over the real 643-task WebVoyager set. The headline 8.8x-9.7x is a MODELED A/C ratio against the naive re-prefill floor — a closed-form integer formula, not a wall-clock measurement. The honest cross-worker reuse number, versus a tuned per-agent-KV stack, is B/C = 1.00x-1.10x. fak is not a web agent; this page is the model-only floor for the live cost benchmark still to be run.

**Last Updated:** 2026-06-20

---

## Measurement Status

- Dataset: `testdata/webbench/webvoyager-converted.jsonl`, converted from the official WebVoyager export; 643 tasks in this repo's converted artifact.
- Model: none for the numbers on this page; no live agent/model execution is included.
- Runs: n=0 live model runs; the tables are deterministic `fak webbench describe` geometry recomputations.
- Artifacts: `experiments/webbench/webvoyager-geometry-20260625.json`, `experiments/webbench/webvoyager-fleet-scale-20260626.json`.
- Status: THEORETICAL (MODELED). The page is not a MEASURED or VERIFIED end-to-end WebVoyager cost/latency benchmark.

---

## Provenance: MODELED geometry over the real 643-task set

These numbers are a **deterministic geometry model** computed over the real
WebVoyager task set — **not a wall-clock measurement**. The task *set* is real
(643 official tasks); the per-turn token geometry is derived from each task's
difficulty, and the prefill cost is a closed-form integer formula
(`internal/webbench/geometry.go::ComputeArms`). `fak webbench describe` prints the
table under the honest header *"prefill-token work-elimination (deterministic
floor, no model)."* Reproduce it yourself:
`fak webbench describe --dataset testdata/webbench/webvoyager-converted.jsonl --workers 1,2,4,8`.

| Component | Source | Status |
|-----------|--------|--------|
| Cost arm formulas (A/B/C) | Closed-form integer geometry | ✅ Correct |
| CLI implementation | Code execution | ✅ Shipped |
| WebVoyager task set | **643 tasks from official source** | ✅ Real dataset |
| Prefill numbers | **8.8x – 9.7x vs the naive floor** | ⚙️ Modeled (no wall-clock) |
| Mock-geometry legacy | 5 tasks, example.com | Legacy reference |

**What this shows:** the CLI works and the prefill-token *work-elimination* a
fused resident KV buys over a naive re-prefill harness, computed over the real
task set. The headline 8.8x–9.7x is the **A/C ratio vs the naive re-prefill
floor**; the honest cross-worker reuse number (vs a tuned per-agent-KV stack) is
**B/C = 1.00x–1.10x** (see the table below). It is **not** a measured throughput
or wall-clock gain.

**Modeled vs legacy mock:**
- Real 643-task set: **8.8x – 9.7x** vs naive floor (modeled geometry)
- Legacy mock: 15.6x – 16.6x (5 mock tasks, more conservative assumptions)

The real-set number is lower because actual WebVoyager tasks have fewer turns
(median 12) than the assumptions used for the legacy mock geometry.

![WebBench prefill elimination — modeled 8.8×–9.7× prefill-token elimination vs the naive floor over the real 643-task WebVoyager set, fak's fused resident KV vs naive per-turn re-prefill](https://raw.githubusercontent.com/anthony-chaudhary/fak/main/visuals/51-webbench-prefill-elimination.svg)

---

## Executive Summary

**fak is not a web agent.** This page documents the current WebBench efficiency
floor: a modeled 8.8x-9.7x prefill-work reduction versus the naive re-prefill
floor over the converted WebVoyager task set. That is not a measured cost or
latency result.

The current web agent benchmark leaderboard measures **capability** (success
rate). That's the model's job. fak's current WebBench surface measures a
model-only **efficiency floor** (prefill work-elimination). A live A/B run is
still required before claiming a measured cost reduction for any SOTA web agent.

## The Position: Capability vs. Efficiency

| What | Who | Metric | fak's Role |
|------|-----|--------|------------|
| **Can the agent complete the task?** | Model (Claude, GPT-4, etc.) | Success Rate | ✗ None - model's capability |
| **How much compute does it cost?** | Infrastructure (orchestrator, serving) | $ per task | Current page: **modeled prefill-work floor only** |

**The point:** Every web agent system today is paying the **turn-tax** — re-prefill megabytes of browser state on every navigation action. That wasted work doesn't exist in fak.

## SOTA Web Agent Benchmarks (2026)

### WebVoyager (586 diverse web tasks)

| Agent | Success Rate | Notes |
|-------|-------------|-------|
| **Alumnium MCP with Claude Code** | **98.5%** | Current SOTA; ~$5 total API cost |
| Magnitude | 93.9% | Claims to beat all other browser agents |
| Browser Use | 89.1% | Previous SOTA; widely cited |
| Agent-E | 73.1% | |
| WebVoyager baseline | 57.1% | Original benchmark baseline |

**Sources:** [Alumnium WebVoyager Report](https://alumnium.ai/blog/webvoyager-benchmark/) · [Browser Use SOTA Technical Report](https://browser-use.com/posts/sota-technical-report) · [Magnitude GitHub](https://github.com/magnitudedev/webvoyager)

### Other Notable Benchmarks

| Benchmark | SOTA Performance | Notes |
|----------|------------------|-------|
| **BrowseComp** (OpenAI) | not yet run | New benchmark for hard-to-find information location |
| **WebArena** | OpenAI Operator: 58.1% | Multi-website task completion |
| **Halluminate Web Bench** | rtrvr.ai: 81.4% | 7-23x faster than competitors |
| **Skyvern 2.0** | 85.85% | Maintains 76.8% at 250 concurrent agents |

## Modeled Prefill Floor

The legacy mock geometry showed what the arithmetic could look like before the
real WebVoyager task set was converted. It is retained here only as historical
theory, not as a result claim.

### Deterministic Prefill Work-Elimination

| Workers | Naive Re-Prefill | Per-Agent KV | **fak Fused** | Net Elimination |
|---------|-----------------|--------------|--------------|-----------------|
| 1 | 3.4 M tokens | 217K tokens | 217K tokens | **15.6x** |
| 2 | 6.8 M tokens | 435K tokens | 419K tokens | **16.1x** |
| 4 | 13.5 M tokens | 870K tokens | 824K tokens | **16.4x** |
| 8 | 27.1 M tokens | 1.7 M tokens | 1.6 M tokens | **16.6x** |

**Methodology:** 5-task sample MOCK dataset (example.com domains); ASSUMED WebVoyager-style geometry (P=3.4K, Action=150, DOMState=2K). These are THEORETICAL calculations demonstrating the framework. Live measurements on actual WebVoyager runs are pending.

### The Breakdown

| Metric | Meaning | Value |
|--------|---------|-------|
| **A/C (Net Elimination)** | Re-prefill every turn vs. shared cross-worker prefix | **15.6x - 16.6x** |
| **B/C (Cross-Worker Reuse)** | Isolated agents vs. shared session value stack | **1.00x - 1.07x** |
| **A/B (Turn-Tax)** | Re-prefill vs. per-agent KV persistence | **15.6x** (worker-independent) |

**Historical note:** The turn-tax (A/B = 15.6x) in this table came from a 5-task
mock dataset and assumed geometry. It is not a benchmark result and must not be
used as the WebBench headline.

---

### ⚙️ MODELED over the official WebVoyager set (643 tasks)

**Modeled geometry over the official WebVoyager dataset** (downloaded 2026-06-20 from [MinorJerry/WebVoyager](https://github.com/MinorJerry/WebVoyager)) — closed-form prefill-token arithmetic, no wall-clock:

| Workers | A naive | B per-agent KV | **C fak fused** | A/C (net) | B/C (cross-worker) | A/B (turn-tax) |
|---------|---------|----------------|-------------|-----------|---------------------|----------------|
| 1 | 170.9 M | 19.4 M | 19.4 M | **8.8x** | 1.00x | **8.8x** |
| 2 | 341.9 M | 38.8 M | 36.8 M | **9.3x** | 1.05x | **8.8x** |
| 4 | 683.7 M | 77.5 M | 71.6 M | **9.5x** | 1.08x | **8.8x** |
| 8 | 1.37 G | 155.1 M | 141.3 M | **9.7x** | 1.10x | **8.8x** |

**Dataset Statistics:**
- 643 real WebVoyager tasks
- 8,745 total navigation turns (median: 12 per task)
- Difficulty: easy (87), medium (430), hard (126)
- Categories: shopping (86), information (85), general (343), media (44), travel (42), search (43)

**Methodology:** Real WebVoyager tasks processed through `fak webbench describe`. Geometry derived from each task's difficulty using standard WebVoyager-style turn estimates; the prefill cost is then a closed-form integer formula (`internal/webbench/geometry.go::ComputeArms`). This is a **MODELED** prefill-token work floor over the **real** task set — **not** a wall-clock measurement. The 8.8x–9.7x is the A/C ratio vs the naive re-prefill floor; the cross-worker reuse number vs a tuned per-agent-KV stack is B/C = 1.00x–1.10x.

### Real Breakdown

| Metric | Meaning | Real Value |
|--------|---------|------------|
| **A/C (vs naive floor)** | Modeled over WebVoyager set | **8.8x - 9.7x** |
| **B/C (Cross-Worker Reuse)** | Cross-worker prefix reuse | **1.00x - 1.10x** |
| **A/B (Turn-Tax)** | Re-prefill vs KV persistence | **8.8x** (worker-independent) |

**Key finding:** The modeled turn-tax is **structural** — every agent pays it,
every turn in the geometry model. On the real WebVoyager task set, this computes
to an **8.8x** prefill-work floor.

---

## Why This Matters: The Cost of SOTA

Take Alumnium's 98.5% SOTA run: ~$5 in API costs for 586 tasks. That's **capability pricing** — paying the model for inference. What's missing is the **infrastructure tax**:

- **Without fak:** Every navigation action re-prefills the entire browser context (DOM state, tool schemas, task history) — that's 2K+ tokens per turn, times ~12 turns per task, times 586 tasks.
- **With fak:** The shared prefix is prefilled once; all workers reuse it. Turn-by-turn, only the new DOM state is processed.

The modeled 8.8x-9.7x prefill-work floor suggests where a live cost run should
look for savings. It does not prove that the same 98.5% SOTA agent costs less
through fak; that claim requires the pending live harness run with the same
agent, same task set, and logged token/cost artifacts.

## Proper Comparison: Whatfak Actually Competes With

**fak does NOT compete with:**
- Model capability (success rate) — that's Claude, GPT-4, etc.
- Browser automation frameworks — that's Playwright, Selenium, etc.
- Agent orchestration logic — that's LangChain, custom controllers

**fak DOES compete with:**
- Naive agent serving (re-send full context every turn) — **modeled 8.8x-9.7x less prefill work vs the naive floor**
- Per-agent KV isolation (vLLM prefix caching per worker) — **modeled 1.00x-1.10x cross-worker gain at 1-8 workers**
- Frontier prompt caches (append-only, no eviction) — **addressable eviction advantage**

The only thing that matters for the comparison is: **how much prefill work does your infrastructure do per turn?**

| System | Prefill Strategy | Work Relative to fak |
|--------|------------------|----------------------|
| Naive re-send | Full context every turn | **modeled 8.8x-9.7x more prefill work** |
| Per-agent KV | Prefix cached per worker | **modeled 1.00x-1.10x more prefill work** (at 1-8 workers) |
| vLLM prefix cache | Shared prefix per serving instance | Similar (if single-tenant) |
| Frontier prompt cache | Append-only reuse | Similar (can't evict) |
| **fak fused** | Shared prefix + cross-worker reuse + addressable eviction | **1x (baseline)** |

## Next Steps: Full Harness Evaluation

Current status: **Deterministic floor proven, live eval pending**

### ✅ Complete (No Model/GPU Required)
- [x] Geometry modeling for web tasks (P, T, A, DOMState)
- [x] Cost arm computation (A/B/C ratios)
- [x] Worker sweep analysis (1, 2, 4, 8 workers)
- [x] Sample dataset with real WebVoyager-style structure
- [x] CLI: `fak webbench describe` + `compare` + `eval`

### 🔄 Pending (Requires Model + Browser Harness)
- [ ] Real WebVoyager dataset ingestion (586 tasks)
- [ ] Live agent runs with SOTA models (Claude, GPT-4)
- [ ] Side-by-side comparison: fak vs. baseline infrastructure
- [ ] Success rate parity proof (same agent, same task, different infra)
- [ ] End-to-end cost measurement (API spend + compute)
- [ ] GPU server-scale fleet runs (100+ concurrent agents)

### 📊 Metrics to Capture (Full Run)
| Metric | Kind | Provenance | Status |
|--------|------|------------|--------|
| Prefill/KV work-elimination | fak-native | Computed | ✅ Shipped |
| Navigation turns + tokens | Comparable | Computed | ✅ Shipped |
| In-process adjudication cost | fak-native | Gated (trace data) | 🔄 Pending |
| **Task success rate** | **Comparable** | **Gated (harness)** | **🔄 Pending** |
| End-to-end $ per task | Comparable | Measured | 🔄 Pending |

## How to Reproduce

```bash
# Describe the deterministic floor for any web agent dataset
go run ./cmd/fak webbench describe --dataset testdata/webbench/sample-tasks.jsonl

# Generate full comparison with markdown report
fak webbench compare --dataset <tasks.jsonl> --md report.md

# Grade predictions (when browser harness available)
fak webbench eval --predictions preds.json
```

## Datasets Supported

- **Browser Agent Benchmark** (browser-use.com) — 100 hard browser tasks
- **WebVoyager** — 586 diverse web interaction tasks
- **BrowseComp** (OpenAI) — Hard-to-find information location
- **Custom datasets** — JSONL/JSON with `{task_id, description, instructions, difficulty, category, actions}` fields

## Sources & References

- SOTA performance data: [Alumnium WebVoyager Benchmark Report](https://alumnium.ai/blog/webvoyager-benchmark/)
- Browser Use SOTA: [Browser Use Technical Report](https://browser-use.com/posts/sota-technical-report)
- Magnitude claims: [Magnitude WebVoyager GitHub](https://github.com/magnitudedev/webvoyager)
- OpenAI BrowseComp: [OpenAI BrowseComp Announcement](https://openai.com/index/browsecomp/)
- WebArena methodology: [WebArena Paper](https://arxiv.org/abs/2307.13857)

---

*Last benchmark update: 2026-06-20*  
*Next full harness eval: pending GPU node access*
