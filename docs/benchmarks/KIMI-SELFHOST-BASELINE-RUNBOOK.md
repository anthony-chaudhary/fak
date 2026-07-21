---
title: "Kimi (Moonshot) self-hosted baseline runbook — best use of the GPU server for fak Kimi testing"
description: "Which Kimi route belongs on the private GPU server (K2 open-weights self-host) vs. the Moonshot cloud (K3), the turnkey vLLM/SGLang/NIM bring-up script and its arch+VRAM-fit gates, the keyless live-smoke witness command and its skip conditions, and how the private control bridge fits without crossing the public/private boundary. Wire-readiness plan, not a performance headline."
---

# Kimi (Moonshot) self-hosted baseline runbook

> **Status: wire-readiness plan.** This document states the reproducible upstream
> shape, the turnkey bring-up gates, and the keyless live-smoke command for
> fronting a self-hosted Kimi K2 server through `fak serve`. It carries **no
> throughput/latency headline** — any such number requires a real tuned EP/EPLB
> baseline, and none is claimed here.

Sibling of the [DeepSeek V4 self-host runbook](DEEPSEEK-V4-SELFHOST-BASELINE-RUNBOOK.md).
The Kimi K3 **cloud** wire contract (Moonshot API) is a separate, already-witnessed
route — see [`DOGFOOD-CLAUDE.md`](../../DOGFOOD-CLAUDE.md#moonshot-kimi-k3-opt-in) and
`internal/agent/kimi_k3_test.go`.

## The one decision: which Kimi route belongs on the GPU server?

The GPU server (reached, per operator, via the **private** control bridge — see the
boundary section below) is the **compute boundary**. Your workstation is the
**control point**. Not every Kimi route is a GPU-server workload:

| Route | What it is | Where it runs | How fak tests it | GPU server? |
|---|---|---|---|---|
| **Kimi K3** | Moonshot's hosted, OpenAI-compatible cloud API (`api.moonshot.ai`). fak applies K3's native wire contract automatically (`reasoning_effort: "max"`, drops generic `temperature`/`top_p`, preserves `reasoning_content`). | Moonshot cloud | `scripts/claude-kimi-k3.{sh,ps1}` (opt-in); wire pinned by `internal/agent/kimi_k3_test.go`. | **No** — cloud only, not self-hostable. |
| **Kimi K2** | Open weights (`moonshotai/Kimi-K2-Instruct`), native block-FP8, ~1T total / ~32B active MoE, MLA attention. | Your GPU server (multi-node) | `scripts/dgx-kimi-serve.sh` → `fak serve --provider openai-compatible` → the keyless `TestKimiSelfHost` witness. | **Yes** — this is the GPU-server self-host target. |
| **Kimi via NIM** | NVIDIA-hosted Kimi through NIM (needs `NVIDIA_API_KEY`). | NVIDIA cloud (or a local NIM container on your node) | `scripts/claude-nim-kimi.{sh,ps1}` (opt-in). | Optional — local NIM is a GPU-server workload; the hosted endpoint is not. |

**Best use of the GPU server for Kimi testing:** self-host **Kimi K2** with a tuned
engine, front it with `fak serve`, and run the keyless wire witness. K3 quality/
behavior testing stays on the cloud route (no GPU needed); K2 is where a device is
genuinely required.

## Why front a tuned engine first, not a native loader

Kimi K2 is a published **~1T-total / ~32B-active MoE** with native block-FP8 weights
([model card](https://huggingface.co/moonshotai/Kimi-K2-Instruct)). The repo's
prior-art / default-spine rule says the practical support path is to **front a tuned
serving engine first** (vLLM/SGLang/NIM), prove fak sits in front of its
OpenAI-compatible surface, and only then decide whether native in-kernel loader work
is worth a separate scoped effort. No native 1T-weight loading is in scope here.

## Turnkey bring-up (on the GPU node/pod)

[`scripts/dgx-kimi-serve.sh`](../../scripts/dgx-kimi-serve.sh) does the boring,
error-prone parts turnkey: detect GPUs, **gate on the arch floor and VRAM fit**,
launch the engine on `:8000`, health-check `GET /models`, and print the exact
`fak serve` line plus the keyless witness command.

```bash
# vLLM (default), auto tensor-parallel across visible GPUs:
scripts/dgx-kimi-serve.sh

# SGLang instead:
FAK_KIMI_ENGINE=sglang scripts/dgx-kimi-serve.sh

# NIM (confirm the image tag exists in NGC first):
FAK_KIMI_ENGINE=nim FAK_KIMI_NIM_IMAGE=nvcr.io/nim/moonshotai/<kimi-tag> \
  NGC_API_KEY=... scripts/dgx-kimi-serve.sh

# Plan only, no launch (prints the gate results + commands):
FAK_KIMI_DRY_RUN=1 scripts/dgx-kimi-serve.sh
```

### The gates (why the script refuses)

Kimi K2 is a **multi-node expert-parallel** deployment, not a single-node resident
model. The script refuses to pretend otherwise:

- **Arch floor.** Native block-FP8 (`fp8_e4m3`) generally requires **sm_90 (Hopper,
  H100/H200)** or **sm_100 (Blackwell)**. On Ampere (A100, sm_80) there is no native
  FP8; a BF16 upcast roughly doubles the footprint. The script **warns** below sm_90
  and **refuses to launch** unless `FAK_KIMI_FORCE=1`.
- **VRAM fit.** ~1T params in FP8 ≈ ~1000 GiB of weights. A single 8-GPU server node
  (8×80 GB = 640 GiB) **does not hold it**. The script sums visible VRAM, compares it
  to `FAK_KIMI_FOOTPRINT_GIB` (default 1000), and **refuses** when the model cannot
  fit — add nodes (expert/pipeline parallel across a pod), pick a smaller checkpoint,
  or set `FAK_KIMI_FOOTPRINT_GIB=0` to skip the check. `FAK_KIMI_FORCE=1` overrides
  (expect OOM at load).

All knobs are `FAK_KIMI_*` env vars documented in the script header (`ENGINE`,
`MODEL`, `PORT`, `TP`, `FOOTPRINT_GIB`, `HEALTH_TIMEOUT`, `DRY_RUN`, `FORCE`, …).

## Front it with fak

```bash
fak serve --provider openai-compatible \
  --base-url "http://<gpu-host>:8000/v1" \
  --model   "moonshotai/Kimi-K2-Instruct"
```

The gateway proxy planner posts to `BaseURL + "/chat/completions"`, so `fak serve`
forwards the same OpenAI-compatible wire the engine exposes.

## The keyless witness ladder

[`internal/gateway/kimi_selfhost_smoke_test.go`](../../internal/gateway/kimi_selfhost_smoke_test.go)
is **keyless from fak's perspective**: with `KIMI_SELFHOST_BASE_URL` unset it
**skips cleanly** (never fails), so `go test ./internal/gateway` stays green on a box
with no model server. Set it against a real Kimi server to collect the minimum
evidence for a "supported self-hosted route":

```bash
KIMI_SELFHOST_BASE_URL="http://<gpu-host>:8000/v1" \
KIMI_SELFHOST_MODEL="moonshotai/Kimi-K2-Instruct" \
  go test ./internal/gateway -run TestKimiSelfHost -v
# (export KIMI_SELFHOST_API_KEY too if your engine requires a token)
```

1. **Readiness** — `GET {base}/models` returns 200 with a non-empty roster.
2. **Non-streaming** — `POST {base}/chat/completions` (`stream:false`) returns content
   (or `reasoning_content` for a thinking variant) and, when the engine reports it, a
   usage block.
3. **Streaming** — `stream:true` yields SSE deltas terminated by `[DONE]`, assembling
   non-empty content.

A capacity throttle (HTTP 429/503, or a `ResourceExhausted` body) is treated as a
clean **skip**, not a failure — the wire can only be judged when the upstream actually
served the request.

## Comparability boundary — do not confound

This runbook proves **wire readiness only**. A throughput/latency number is valid only
from a tuned EP/EPLB baseline on a fixed pod shape, reported with its machine class and
accelerator arch. Do not compare a wire-readiness smoke against a tuned headline, and
do not quote a K2 self-host number next to a K3 cloud number — different targets,
different wire contracts, different cost models.

## Private control-bridge boundary

The GPU server is, per operator, reached via a **private** control bridge that lives
only in `fak-private` (the public tree **refuses** any `cmd/*dgx*/` or `internal/*dgx*/`
path at commit; see [`docs/gpu-server-private-boundary.md`](../gpu-server-private-boundary.md)
and [`docs/fleet-compute-nodes.md`](../fleet-compute-nodes.md)). This runbook, the serve
script, and the witness test are the **public** half: they name the workload class,
the reproducible upstream shape, and the evidence that must come back. They carry **no**
node identity, bridge commands, credentials, or channel identifiers.

A missing credential or bridge session changes the **next actor**, not the compute
requirement: hand the authorized operator this runbook plus the `dgx-kimi-serve.sh`
invocation, and return only the **scrubbed** witness (tested commit, workload, machine
class + accelerator arch, command/step, result, artifact location) through the
[lab development loop](../fak/lab-dev-loop.md). Never report "no local GPU" as a
terminal result for a K2 self-host claim — route it to the pod instead.
