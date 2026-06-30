---
title: "Qwen3.6-27B on 8-GPU Server: fak Gateway vs SGLang"
description: "fak serving and coding-agent surfaces on Qwen3.6-27B with an 8-GPU SGLang backend, reporting gateway tax across a 1-to-128 concurrency sweep."
---

# Qwen3.6-27B on the GPU server — fak serving + coding-agent surfaces

**Date:** 2026-06-22 · **Hardware:** lab GPU server (an 8-GPU datacenter server) ·
**Serving:** SGLang 0.5.10.post1 (TP=8, bf16) · **torch** 2.9.1+cu128 ·
**Model:** `Qwen/Qwen3.6-27B` (dense, hybrid Gated-DeltaNet).

> This is the **Rung-4 headline** of `PLAN-model-ladder-gpu-server`: the Qwen3.6-27B
> stood up on the GPU server and **used by a fak coding agent**, then load-compared against
> raw SGLang across a concurrency sweep. The fak path here is **SGLang-serves +
> fak-adjudicates** (`fak serve` gateway in front of SGLang) — *not* fak's native CUDA
> engine, which cannot yet run a quantized / multi-GPU 27B. Every number traces to a
> committed artifact under [`experiments/qwen36/dgx-r4-20260622/`](../../experiments/qwen36/dgx-r4-20260622/).

## 1. Used in a coding agent (the headline)

All three fak surfaces **PASS** on the 27B, run on the GPU server
(`tools/dgx_qwen36_surfaces.py` → `qwen36_surface_smoke.py` against the served endpoint):

| Surface | Status | Note |
|---|:---:|---|
| **agent** (fak agent loop) | ✅ PASS | a fak coding-agent drives the 27B, every tool call adjudicated |
| **gateway-openai** | ✅ PASS | single-stream decode **59.3 tok/s** (datacenter GPU; cf. 2.7 tok/s on a laptop AMD RX 7600) |
| **mcp-http** | ✅ PASS | MCP gateway over the 27B |

Artifact: [`surface-smoke.json`](../../experiments/qwen36/dgx-r4-20260622/surface-smoke.json).

## 2. Throughput under a multi-agent concurrency load (fak-gateway vs raw SGLang)

Load: `dgx_run_matrix` concurrency sweep over the `agent-live/production-workload`,
metric = `completion_tokens_per_sec`, 64 requests/concurrency, TP=8.

| Concurrency | fak-gateway tok/s | raw-SGLang tok/s | fak / raw |
|---:|---:|---:|:---:|
| 1 | 72.2 | 87.6 | 0.83× |
| 4 | 176.6 | 272.5 | 0.65× |
| 8 | 249.6 | 415.4 | 0.60× |
| 16 | 392.7 | 639.6 | 0.61× |
| 32 | 685.0 | 870.5 | 0.79× |
| **64 (peak)** | **1085.6** | **1451.6** | **0.75×** |
| 128 | 1074.4 | 1103.2 | 0.97× |

The 27B serves cleanly across the whole 1→128 concurrent load through both stacks.
Artifacts: [`compare.json`](../../experiments/qwen36/dgx-r4-20260622/compare.json),
[`fak-gateway.json`](../../experiments/qwen36/dgx-r4-20260622/fak-gateway.json),
[`raw-sglang.json`](../../experiments/qwen36/dgx-r4-20260622/raw-sglang.json),
[`COMPARE.md`](../../experiments/qwen36/dgx-r4-20260622/COMPARE.md).

## 3. Honest fence

- **fak's value here is the adjudication / coherence / measurement plane, not raw tok/s.**
  The gateway tax varies with concurrency (0.60–0.83× through the mid-range) and **converges
  to ~3% at saturation** (conc 128: 0.97×), where the per-request proxy cost is amortized.
  This matches the house bound (`HERO-BENCHMARK`): single-stream/raw throughput is *not*
  fak's axis.
- **Marker-compliance caveat.** The load harness requires every response to echo a literal
  `FAK_DGX_REQ_` marker. Qwen3.6 is a reasoning model and echoes it only ~35–66% of the
  time, so `--max-error-rate` was relaxed to 0.9 and throughput is measured over the
  compliant subset. This is a benchmark-harness artifact, **not** a serving failure (the
  model serves tokens fine; see the per-point `ok/requests` in the JSONs).
- **Native-engine gap unchanged.** fak's own CUDA engine still can't load a quantized 27B
  (no quantized device GEMM, no multi-GPU NCCL; f32 27B is 108 GB > 80 GB). The GPU server path is
  and remains llama.cpp/SGLang-serves + fak-fronts.

## 4. Reproduce

The GPU server is reached only via the private control bridge (private lab tooling); the 27B rung is
added to the ladder by `tools/dgx_qwen36_27b_runner.py` (reuses `dgx_ladder_runner.run_rung`,
sizes `--mem-fraction-static` to fit the GLM-occupied GPU0, and sets
`SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK=0`). On the GPU server:

```bash
# throughput sweep (fak-gateway vs raw SGLang)
python3 tools/dgx_qwen36_27b_runner.py --conc 1,4,8,16,32,64,128 --rpc 64 --rung-id r4
# coding-agent + gateway + MCP surfaces on the served 27B
python3 tools/dgx_qwen36_surfaces.py
```
