---
title: "Qwen3.6-27B on one GCP datacenter GPU via the pure fak kernel — a coding fallback for Claude Code"
description: "One-command runbook to serve Qwen3.6-27B through fak's OWN in-kernel CUDA forward on a single GCP datacenter GPU, and route Claude Code on your machine to it as a direct coding fallback when a subscription account is unavailable."
---

# Qwen3.6-27B on one GCP datacenter GPU → Claude Code, via the pure fak kernel

> **Audience.** Operators who need a coding fallback for Claude Code on a single GCP GPU. By the end you can serve a model through fak's own in-kernel CUDA forward and route Claude Code on your machine to it, and you'll know which checkpoint actually runs on that forward today.

When your subscription account is unavailable, this stands up **Qwen3.6-27B on a single
GCP A100** through fak's **own** in-kernel CUDA forward (the pure fak kernel — not vLLM,
SGLang, or llama.cpp) and points **Claude Code on this machine** at it. Every turn decodes
on fak's forward on the datacenter GPU, and `fak guard` adjudicates every tool call on the way.

```text
┌──────────────┐ /v1/messages ┌───────────────────────────┐ in-kernel ┌──────────────────┐
│ this machine │ ───────────▶ │ fak serve (PURE KERNEL)    │ ────────▶ │ Qwen3.6-27B q4_k │
│ claude (Code)│ ◀─────────── │ --backend cuda, own forward│ ◀──────── │ resident on ONE  │
│              │              │ adjudicates every tool call│           │ A100-40GB (sm_80)│
└──────────────┘              └───────────────────────────┘           └──────────────────┘
```

## Status — witnessed on a live datacenter GPU (2026-06-27)

The whole **infra** is proven end-to-end on a single GCP datacenter GPU: `--apply` → `-tags cuda`
build (sm_80) → GGUF load → gateway → connect from the laptop → a real coding turn comes back.
But two honest constraints shape what to serve:

- **The literal `Qwen3.6-27B` does NOT run on fak's forward yet.** That checkpoint
  (`lmstudio-community/Qwen3.6-27B-GGUF`) is a **Gated-DeltaNet/SSM hybrid** (every layer is a
  fused `attn_qkv` + `attn_gate` + a full `ssm_*` recurrence — no `self_attn.q_proj`). fak's
  in-kernel forward implements the standard separate-projection attention path, so it loads and
  binds but **panics on the first decode**. Tracked as **#934** (the real blocker; #65/#67 family).
- **The working default is `Qwen2.5-Coder-14B`** — a standard dense arch fak's forward *does*
  serve. Witnessed: correct code out, **~6.7 tok/s** (the decode is launch/op bound — the perf
  levers are #483/#279/#401), ~16 GiB Q8-resident + a 32K context fits one datacenter GPU. A standard
  **32B** is ~32 GiB Q8-resident — no KV room on 40GB (use the 80GB tier for 32B).
- **Keep `--cuda-graph` OFF** (the default). At 14B/datacenter GPU it was witnessed to *crash* the serve
  (lazy KV `cudaMalloc` during graph capture, **#932**), not speed it up — until KV is
  pre-allocated before capture.

So `scripts/gcp-qwen-serve.sh` and `tools/qwen36_a100_fak_serve.sh` default to the **supported
coder**; point them at Qwen3.6-27B (via `QWEN_REPO`/`MODEL_ID`) once #934 lands.

## Why a single datacenter GPU is enough

A ~14–16B coder `q4_k_m` is **~14–16 GB Q8-resident**, so it fits **one datacenter GPU whole** with
room for a 32K KV cache (which holds a full Claude Code agent prompt). No `--cpu-offload-experts`,
no multi-GPU, no 466 GB MoE staging (that is the GLM-5.2 frontier path —
[`claude-glm-gcp.md`](claude-glm-gcp.md) / `scripts/gcp-glm-serve.sh`). This is the cheap,
"actually available" single-GPU tier: `a2-highgpu-1g`, ~\$3.67/hr.

## One command (plan first, then apply)

```bash
# PLAN — prints the exact gcloud create command, the VM startup script, and the
# reach-from-here steps. No creds needed; nothing is created.
./scripts/gcp-qwen-serve.sh

# APPLY — creates the single-A100 VM and serves (needs an authenticated gcloud + GPU quota):
GCP_PROJECT=<your-project> ./scripts/gcp-qwen-serve.sh --apply
```

Knobs (env): `GCP_TIER` (default `a2-high-a100-40gb-1g`; also `a2-ultra-a100-80gb-1g`),
`CUDA_GRAPH=1` (the #483 decode lever, below), `FAK_GATEWAY_KEY` (inbound bearer key — auto
-generated on `--apply` if unset), `QWEN_PORT` (default 8080). See `--help` for the full set.

The VM runs `tools/qwen36_a100_fak_serve.sh` as a durable `systemd` unit: it builds the
`-tags cuda` fak binary, fetches the GGUF, and serves

```bash
FAK_Q4K=1 fak serve --backend cuda --gguf <Qwen3.6-27B.q4_k_m.gguf> --addr 0.0.0.0:8080 \
  --require-key-env FAK_GATEWAY_KEY
```

`FAK_Q4K=1` selects the **direct-resident-Q4_K** decode path (the q4_k matmul tensors stay
raw on device — no Q4_K→f32→Q8 round-trip), which is the Qwen3.6-27B decode lever.

## Connect Claude Code on this machine

After the unit reaches `QWEN36_A100_FAK_SERVE_READY` (`cat /opt/qwen36-q4k/PHASE` on the VM),
point this machine at the gateway. Get the VM's private IP from Tailscale (recommended) or
open an IAP tunnel — both are printed by the plan.

```powershell
# Windows (PowerShell):
. scripts/connect-fak-node.ps1 -GatewayHost <ip> -GatewayKey <key> -GatewayPort 8080 -Probe
claude          # every turn now decodes on fak's own forward on the A100
```

```bash
# macOS / Linux:
source scripts/connect-fak-node.sh <ip> <key> 8080
claude
```

`connect-fak-node` exports `ANTHROPIC_BASE_URL` / `ANTHROPIC_API_KEY` into the shell, so
`claude` (and any tool that reads those) routes through the kernel. Disconnect with the
`-Disconnect` / `--disconnect` form.

## Performance: the levers and how to witness them

A single-stream 27B decode is **memory-bandwidth bound** (it reads ~16 GB of weights per
token). At the datacenter GPU's ~1.5 TB/s that is a low-hundreds-of-tok/s ceiling; reaching it is
about keeping the memory system saturated and not paying per-kernel launch gaps.

| lever | flag / env | what it does | status |
|---|---|---|---|
| direct-resident Q4_K | `FAK_Q4K=1` | native q4_k device GEMV, no dequant-to-f32 (#484/#485) | wired; default in the serve script |
| CUDA-graph decode replay | `--cuda-graph` / `FAK_CUDA_GRAPH=1` | capture each token's op stream, replay as ONE launch instead of N (#483) | wired, but **witnessed to CRASH the serve** at 14B/datacenter GPU — lazy KV `cudaMalloc` during capture (#932). KEEP OFF until KV is pre-allocated before capture. |
| fused flash attention | (automatic) | one online-softmax kernel, no `scores[nPos]` row materialized (#486) | wired |

**Witnessed (14B coder, datacenter GPU, 2026-06-27): ~6.7 tok/s** — only ~6% of the bandwidth
ceiling, so the decode is **launch/op bound**, not BW bound. That is exactly the regime where
graph replay (#483) should win (unlike the 0.5B/L4, where it was a measured no-win). But
`--cuda-graph` currently **crashes** the in-kernel serve (the KV cache is allocated lazily
during the first captured token, violating the "no cudaMalloc during capture" precondition,
#932). So the headline perf work is: pre-allocate KV before capture (#932), then the launch-gap
lever (#483) + fused kernels (#279) + continuous batching (#401).

## Honest scope

- **The literal Qwen3.6-27B does not run on fak's forward yet** — it's a GDN/SSM hybrid; fak
  panics on the first decode (#934). See the Status section at the top. The runbook serves a
  **supported** standard-arch coder (Qwen2.5-Coder-14B) meanwhile.
- **Proven end-to-end on a live datacenter GPU (2026-06-27)**: provision → `-tags cuda` build (sm_80) →
  GGUF load → gateway (`/healthz`, `/v1/models`) → **a real coding turn from the laptop through
  an SSH-forward tunnel** (`is_prime`, a reverse-string lambda — correct code from fak's own
  forward, weights resident on-GPU at ~16 GiB). fak's prior live CUDA-serve witness was 0.5B on
  an L4 (466 tok/s — [`L4-INKERNEL...`](../notes/L4-INKERNEL-SERVE-AND-CONCURRENCY-FIX-2026-06-25.md)); this is the first 14B/datacenter GPU one.
- **Perf is the open item**: ~6.7 tok/s is usable-but-slow for interactive coding; the levers
  above (#932 → #483 → #279 → #401) are the path to the bandwidth ceiling.
- The device serve is **single-stream**; drive it one request at a time until batched device
  decode lands (the concurrency-safety fix is shipped; throughput batching is `#401`).

## See also

- `scripts/gcp-qwen-serve.sh` — the provisioner (this runbook's command).
- `tools/qwen36_a100_fak_serve.sh` — the durable build+serve unit it runs on the VM.
- [`qwen36-claude-dogfood-playbook.md`](../qwen36-claude-dogfood-playbook.md) — the
  local/Mac dogfood path and the end-to-end "what this proves" map.
- [`claude-glm-gcp.md`](claude-glm-gcp.md) — the GLM-5.2 frontier-MoE sibling (8-GPU datacenter server).
