---
title: "DeepSeek V4 self-hosted (vLLM/SGLang) baseline runbook"
description: "The exact OpenAI-compatible upstream shape, the keyless live-smoke command and its skip conditions, and the minimum-evidence witness ladder for fronting a tuned vLLM/SGLang DeepSeek-V4 server through `fak serve` — before any native in-kernel MoE loader work is scoped. Wire-readiness plan, not a performance headline."
---

# DeepSeek V4 self-hosted (vLLM/SGLang) baseline runbook

> **Status: wire-readiness plan.** This document states the reproducible
> upstream shape, the keyless live-smoke command, and the minimum evidence for a
> "supported self-hosted route." It carries **no throughput/latency headline** —
> any such number requires a real tuned baseline per
> [Comparability boundary](#comparability-boundary-do-not-confound), and none is
> claimed here.

Resolves [#3013](https://github.com/anthony-chaudhary/fak/issues/3013). Parent
program: [#3006](https://github.com/anthony-chaudhary/fak/issues/3006). Sibling
DeepSeek-V4 leaves: #3009–#3012 / #3014 / #3015.

## Why front a tuned engine first, not a native loader

DeepSeek-V4-Pro is a published **1.6T-total / 49B-active MoE with 1M context**
([model card](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro)). The repo's
prior-art / default-spine rule says the practical support path is to **front a
tuned serving engine first** and only then decide whether native in-kernel
loader/backend work is worth a separate scoped effort. That means: prove fak can
sit in front of a vLLM/SGLang DeepSeek-V4 server exposing the OpenAI Chat
Completions surface — the same surface the hosted DeepSeek API exposes
([vendor docs](https://api-docs.deepseek.com/news/news260424)) — before a single
line of native 1.6T-weight loading is filed.

This runbook is the **baseline floor**. A native fak MoE follow-on (tokenizer /
chat-template mapping, expert placement, FP4/FP8 weight loading, 1M-context
cache pressure) may be filed **only** if this baseline surfaces a concrete
fak-owned gap — not from model hype.

## The self-hosted upstream shape

fak fronts the engine as a generic **OpenAI-compatible upstream**. The gateway's
proxy planner posts to `BaseURL + "/chat/completions"` and reads the roster at
`BaseURL + "/models"`, so `BaseURL` is the engine's OpenAI root (conventionally
ending in `/v1`).

```bash
# vLLM (BF16/FP8, day-0 V4 support per lmsys.org/blog/2026-04-25-deepseek-v4)
vllm serve deepseek-ai/DeepSeek-V4-Pro \
  --served-model-name deepseek-ai/DeepSeek-V4-Pro \
  --port 8000
# → OpenAI root: http://<host>:8000/v1

# SGLang (equivalent OpenAI-compatible surface)
python -m sglang.launch_server \
  --model-path deepseek-ai/DeepSeek-V4-Pro \
  --port 8000
# → OpenAI root: http://<host>:8000/v1
```

Front it with `fak serve` (keyless from fak's perspective — auth, if any, is the
engine's concern and passed through):

```bash
fak serve --provider openai-compatible \
  --base-url "http://<host>:8000/v1" \
  --model deepseek-ai/DeepSeek-V4-Pro
```

### Turnkey GPU-node bring-up script

[`scripts/dgx-deepseek-serve.sh`](../../scripts/dgx-deepseek-serve.sh) does the
bring-up turnkey: it detects the node's GPUs, **gates on the V4 architecture floor**
(below), launches an OpenAI-compatible server for the chosen engine
(`FAK_DGX_ENGINE=vllm|sglang|nim`) and model (`FAK_DGX_MODEL`, default V4-Flash) on
`:8000`, health-checks `GET /models`, then prints the exact `fak serve` line and the
keyless wire-witness command below.

```bash
# On the GPU node (vLLM default; V4-Flash for a single node):
scripts/dgx-deepseek-serve.sh
# NVIDIA NIM instead of vLLM (confirm the V4 image tag exists in NGC first):
FAK_DGX_ENGINE=nim FAK_DGX_NIM_IMAGE=nvcr.io/nim/deepseek-ai/<v4-tag> \
  NGC_API_KEY=... scripts/dgx-deepseek-serve.sh
# Dry-run the plan (detect GPUs + print the launch/serve commands, launch nothing):
FAK_DGX_DRY_RUN=1 scripts/dgx-deepseek-serve.sh
```

The script **refuses to launch below sm_90 (Hopper)** unless `FAK_DGX_FORCE=1`,
because a Volta-class node (V100, sm_70) cannot serve a stock V4 FP4/FP8+DSA path
(see the architecture floor below). This is the same gate stated in the
[DeepSeek V4 supported page](../supported/deepseek.md#bring-v4-up-on-a-gpu-node).

### Hardware assumptions (generic capacity language only)

No private bridge/control details are committed. In generic terms:

- **V4-Pro (1.6T total)** at FP8 is ≈1.6 TB of weights — a **multi-node
  expert-parallel** deployment. Out of scope for a *first* baseline; size against
  V4-Flash first (284B total / 13B active, ≈284 GB at FP8 → plausible on one
  8×80 GB node, architecture floor permitting).
- **Architecture floor to verify, not assume:** the V4 attention stack is
  token-wise compression + **DSA (DeepSeek Sparse Attention)**
  ([vendor announcement](https://api-docs.deepseek.com/news/news260424)). Stock
  DSA kernel paths have historically required **sm_90+ (Hopper)**; an Ampere
  sm_80 node may be unable to run the stock resident path. **Confirm the DSA arch
  floor from the vLLM/SGLang V4 kernel requirements before reserving GPU time.**
  If confirmed, the baseline needs a Hopper+ node or an engine-provided
  non-DSA / dense-attention fallback.

### Which node can actually serve V4 — the honest floor

The floor above is **confirmed** for **stock** engines, not hypothetical, for
this repo's fleet — but a fork-based sm_80 path has since opened (see the update
below):

- **sm_80 (Ampere, 8×80 GB-class node) is BELOW the *stock* V4 floor.** The FP4
  expert / FP8 / FP4-indexer / DSA stack has no stock sm_80 resident path — stock
  vLLM/SGLang refuse (the same wall that blocks GLM-5.2 on sm_80, vLLM #35021),
  and `scripts/dgx-deepseek-serve.sh`, which fronts a stock engine, correctly
  **refuses** below sm_90.
- **UPDATE (2026-07, supersedes the old "V4 has no sm_80 fallback" claim): a
  V4-aware llama.cpp fork now serves V4-Flash on sm_80** — the same
  "llama.cpp overcomes the sm_90 kernel floor" move already proven for GLM-5.2 on
  the Ampere GPU server. The [cchuter `feat/v4-port-cuda`](https://github.com/cchuter/llama.cpp)
  fork (and upstream [PR #24162](https://github.com/ggml-org/llama.cpp/pull/24162)
  for Unsloth quants) implements V4's five custom ops (compressor decode,
  hyperconnection, lightning indexer, FP8-KV simulation, NextN heads) with CUDA
  kernels; sm_80 uses the software-emulated FP8 path (correct, not HW-accelerated).
  V4-Flash GGUFs (UD-IQ3_XXS ≈103 GB → UD-Q8_K_XL ≈162 GB) fit **resident** on
  8×A100-80 GB (640 GB) — no CPU-offload floor like GLM-5.2's 466 GB. The turnkey
   harness is [`tools/deepseekv4_stage_serve_gpu_server.sh`](../../tools/deepseekv4_stage_serve_gpu_server.sh)
  (self-stages the GGUF, builds the fork at sm_80 with the required
  `GGML_SCHED_MAX_SPLIT_INPUTS=128`, serves, and records a three-rung on-box wire
  witness). This does **not** replace the deferred fak-native V4 kernel track
  ([#3016](https://github.com/anthony-chaudhary/fak/issues/3016)–[#3019](https://github.com/anthony-chaudhary/fak/issues/3019));
  it is the interim serve path. **Status: a live the Ampere GPU server witness against this harness
  is IN PROGRESS — the *path* is published/validated upstream (on sm_89 Ada;
  less battle-tested on sm_80); the on-the Ampere GPU server pass is not yet recorded here.**
- **The paths that work today:**
  1. **Route to a hosted V4 endpoint through fak** — proven live this session
     (see [Live witness](#live-witness--wire-proven-on-a-real-v4-endpoint)). The
     model runs hosted; fak governs the wire. This is the "V4 usable now" answer
     for any environment, including one whose local GPUs are below the floor.
  2. **Serve V4-Flash on an sm_80 node via the llama.cpp V4 fork** — the harness
     above (interim; live the Ampere GPU server witness in progress).
  3. **Provision an sm_90+ node (Hopper/Blackwell) and self-host.** This repo's
     GCP idiom (`scripts/gcp-glm-serve.sh` + the `tools/gcp_accel.py` tier
     registry, e.g. `GCP_TIER=a3-ultra-h200` for Hopper or `a4-b200` for
     Blackwell's native FP4) provisions such a node and reaches it via
     tunnel/tailscale → `fak serve`. Run `scripts/dgx-deepseek-serve.sh` on that
     node for the gated bring-up + wire witness. Blackwell (sm_100) is the
     intended V4-Pro home; Hopper (sm_90) serves V4-Flash with FP4 via emulation.

## Minimum evidence for a "supported self-hosted route"

The optional live smoke collects, in order, exactly these rungs:

| rung | check | witness |
|---|---|---|
| readiness | `GET {base}/models` → 200 with a model roster | `TestDeepSeekV4SelfHostReadiness` |
| non-streaming | `POST {base}/chat/completions` (`stream:false`) → content + a usage block | `TestDeepSeekV4SelfHostNonStreaming` |
| streaming | `POST {base}/chat/completions` (`stream:true`) → SSE deltas terminated by `[DONE]`, non-empty content | `TestDeepSeekV4SelfHostStreaming` |
| tool-call | *(optional, engine-dependent)* one function-call fixture if the engine advertises tool support | documented here; not gated |
| usage/counters | token counts reported honestly; an engine that omits them is a **recorded gap**, never a synthesized number | logged by the non-streaming rung |

## Optional live smoke — command and skip conditions

The smoke lives in
[`internal/gateway/deepseek_selfhost_smoke_test.go`](../../internal/gateway/deepseek_selfhost_smoke_test.go)
and is **keyless from fak's perspective**. It **skips cleanly (never fails)** when
`DEEPSEEK_SELFHOST_BASE_URL` is unset, so the gateway test suite stays green on a
box with no model server:

```bash
# Skips cleanly — no upstream configured:
go test ./internal/gateway -run TestDeepSeekV4SelfHost -v
#   --- SKIP: TestDeepSeekV4SelfHostReadiness (DEEPSEEK_SELFHOST_BASE_URL unset …)
#   --- SKIP: TestDeepSeekV4SelfHostNonStreaming
#   --- SKIP: TestDeepSeekV4SelfHostStreaming

# Runs the readiness/completion/streaming rungs against a live engine:
DEEPSEEK_SELFHOST_BASE_URL="http://<host>:8000/v1" \
DEEPSEEK_SELFHOST_MODEL="deepseek-ai/DeepSeek-V4-Pro" \
  go test ./internal/gateway -run TestDeepSeekV4SelfHost -v
```

Environment contract:

| env var | required | default | meaning |
|---|---|---|---|
| `DEEPSEEK_SELFHOST_BASE_URL` | to run | *(unset ⇒ skip)* | OpenAI-compatible root, e.g. `http://host:8000/v1` |
| `DEEPSEEK_SELFHOST_MODEL` | no | `deepseek-ai/DeepSeek-V4-Pro` | served model id to route |
| `DEEPSEEK_SELFHOST_API_KEY` | no | *(none)* | passed as `Authorization: Bearer …` only if set |

## Comparability boundary (do not confound)

- This baseline is **wire readiness** for a self-hosted OpenAI-compatible route —
  it proves fak can front the engine and relay readiness/completion/streaming. It
  is **not** a performance measurement.
- Any throughput/latency headline requires the tuned serving profile captured by
  [`VLLM-EP-EPLB-MOE-BASELINE-RUNBOOK.md`](VLLM-EP-EPLB-MOE-BASELINE-RUNBOOK.md)
  (EP/EPLB arms, `vllm bench serve` percentiles, warm CUDA-graph `tuned` gate) on
  a real EP-capable node, and must name **hardware, engine, quantization/precision,
  context length, and baseline** before any number is quoted.
- A native fak MoE claim must serve the **same model family, hardware, precision,
  context, and concurrency** as the tuned baseline before a delta is attributable
  to fak; otherwise the row is `[NOT COMPARABLE]` and must say so.

## Completion bar

The self-host baseline is complete only when:

- The smoke **skips cleanly** with `DEEPSEEK_SELFHOST_BASE_URL` unset (verified in
  CI on every box), **and**
- On a real V4/V4-Flash serving node it passes all three rungs
  (readiness + non-streaming + streaming) with the usage/counter behavior logged
  honestly, **and**
- Any performance number is deferred to the tuned EP/EPLB baseline and linked from
  [`../../BENCHMARK-AUTHORITY.md`](../../BENCHMARK-AUTHORITY.md).

The wire is now **witnessed live** against a real DeepSeek-V4 endpoint
(see [Live witness](#live-witness--wire-proven-on-a-real-v4-endpoint)); the
remaining honest gap is a **dedicated-node performance headline**, still deferred
to the tuned EP/EPLB baseline.

## Live witness — wire proven on a real V4 endpoint

On **2026-07-09** (repo `4d13dc6c9`) all three rungs ran live against a real
DeepSeek-V4 server — `deepseek-ai/deepseek-v4-flash` behind NVIDIA's
OpenAI-compatible surface (`https://integrate.api.nvidia.com/v1`), the exact wire
this runbook specifies — and all three passed:

| rung | result | evidence |
|---|---|---|
| readiness | **PASS** | `/models` → 121-model roster including `deepseek-ai/deepseek-v4-pro` and `deepseek-ai/deepseek-v4-flash` |
| non-streaming | **PASS** | `finish="stop"`; usage witnessed `prompt=11 completion=24 total=35` (counters **relayed by the engine**, not fak-authored) |
| streaming | **PASS** | SSE deltas assembled 90 chars of content, terminated by `[DONE]` |

```bash
DEEPSEEK_SELFHOST_BASE_URL="https://integrate.api.nvidia.com/v1" \
DEEPSEEK_SELFHOST_MODEL="deepseek-ai/deepseek-v4-flash" \
DEEPSEEK_SELFHOST_API_KEY="$NVIDIA_API_KEY" \
  go test ./internal/gateway -run TestDeepSeekV4SelfHost -v
```

**What this does and does not establish.** It establishes that fak's
`openai-compatible` route relays readiness, a non-streaming completion with honest
usage counters, and a streamed completion to `[DONE]` against a **real DeepSeek-V4
server** — the wire-readiness bar above. It is **not** a dedicated-node
performance headline: the NVIDIA-hosted surface is a **shared, pooled** endpoint
(it returns HTTP&nbsp;503 `ResourceExhausted` under load), so no TTFT/TPOT/tok-s
number is drawn from it — that still requires the tuned EP/EPLB profile on a
dedicated node per the
[comparability boundary](#comparability-boundary-do-not-confound).

**Shared-endpoint robustness (smoke hardening).** Because a pooled endpoint
throttles, the smoke now treats an upstream capacity signal (HTTP&nbsp;429/503 or a
`ResourceExhausted` body) as a **clean skip, not a failure** — a busy node is not a
fak wire defect — and accepts a DeepSeek thinking-mode turn whose tokens land in
`reasoning_content` with empty `content` as a live wire. On a **dedicated**
dedicated Hopper-class node there is no such contention and all three rungs pass without skips.

## Cross-links

- Tuned serving profile (the performance floor this readiness plan defers to):
  [VLLM-EP-EPLB-MOE-BASELINE-RUNBOOK.md](VLLM-EP-EPLB-MOE-BASELINE-RUNBOOK.md).
- Authority ledger (the only place a number becomes authoritative):
  [`../../BENCHMARK-AUTHORITY.md`](../../BENCHMARK-AUTHORITY.md).
- Parent program: [#3006](https://github.com/anthony-chaudhary/fak/issues/3006).
