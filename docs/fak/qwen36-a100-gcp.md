---
title: "Qwen3.6-27B on one GCP A100 via the pure fak kernel — a coding fallback for Claude Code"
description: "One-command runbook to serve Qwen3.6-27B through fak's OWN in-kernel CUDA forward on a single GCP A100, and route Claude Code on your machine to it as a direct coding fallback when a subscription account is unavailable."
---

# Qwen3.6-27B on one GCP A100 → Claude Code, via the pure fak kernel

When your subscription account is unavailable, this stands up **Qwen3.6-27B on a single
GCP A100** through fak's **own** in-kernel CUDA forward (the pure fak kernel — not vLLM,
SGLang, or llama.cpp) and points **Claude Code on this machine** at it. Every turn decodes
on fak's forward on the A100, and `fak guard` adjudicates every tool call on the way.

```text
┌──────────────┐ /v1/messages ┌───────────────────────────┐ in-kernel ┌──────────────────┐
│ this machine │ ───────────▶ │ fak serve (PURE KERNEL)    │ ────────▶ │ Qwen3.6-27B q4_k │
│ claude (Code)│ ◀─────────── │ --backend cuda, own forward│ ◀──────── │ resident on ONE  │
│              │              │ adjudicates every tool call│           │ A100-40GB (sm_80)│
└──────────────┘              └───────────────────────────┘           └──────────────────┘
```

## Why a single A100 is enough

Qwen3.6-27B `q4_k_m` is **~16–17 GB resident**, so it fits **one A100-40GB whole** with
~23 GB left for the KV cache and activations. No `--cpu-offload-experts`, no multi-GPU, no
466 GB MoE staging (that is the GLM-5.2 frontier path —
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
token). At the A100's ~1.5 TB/s that is a low-hundreds-of-tok/s ceiling; reaching it is
about keeping the memory system saturated and not paying per-kernel launch gaps.

| lever | flag / env | what it does | status |
|---|---|---|---|
| direct-resident Q4_K | `FAK_Q4K=1` | native q4_k device GEMV, no dequant-to-f32 (#484/#485) | wired; default in the serve script |
| CUDA-graph decode replay | `--cuda-graph` / `FAK_CUDA_GRAPH=1` | capture each token's op stream, replay as ONE launch instead of N (#483) | wired through the HAL; **OFF by default — witness tok/s before trusting** |
| fused flash attention | (automatic) | one online-softmax kernel, no `scores[nPos]` row materialized (#486) | wired |

The `--cuda-graph` lever was a **measured no-win on a tiny 0.5B model on an L4** (launch
overhead is already small there), which is why it is off by default. The 27B-on-A100
calculus is different — far more, larger kernels per token — so it is worth witnessing.
Capture the before/after on **your** node and only then rely on it:

```bash
# A: graph off (default). B: graph on.
CUDA_GRAPH=0 ... ; CUDA_GRAPH=1 ...   # via the serve script, or pass --cuda-graph directly
# then compare warm steady-state decode tok/s (discard the first, cold, token).
```

## Honest scope

- **Correctness is witnessed on the CPU reference**: fak's own `qwen35` forward is cosine
  ≥ 0.9999 vs HF and argmax-exact (the `#442` oracle gates — see
  [`qwen36-claude-dogfood-playbook.md`](../qwen36-claude-dogfood-playbook.md), "Native parity
  witnesses").
- **The CUDA resident-Q4_K decode at 27B is hardware-gated**: this runbook stands the
  endpoint up and smoke-tests a real chat answer, but a live 27B serve turn and its **tok/s**
  must be measured on a real A100. fak's live in-kernel CUDA serve is witnessed today at the
  0.5B scale (466 tok/s warm on an L4 —
  [`docs/notes/L4-INKERNEL-SERVE-AND-CONCURRENCY-FIX-2026-06-25.md`](../notes/L4-INKERNEL-SERVE-AND-CONCURRENCY-FIX-2026-06-25.md));
  the 27B/A100 number is the open perf item this path exists to measure.
- The device serve is **single-stream**; drive it one request at a time until batched device
  decode lands (the concurrency-safety fix is shipped; throughput batching is `#401`).

## See also

- `scripts/gcp-qwen-serve.sh` — the provisioner (this runbook's command).
- `tools/qwen36_a100_fak_serve.sh` — the durable build+serve unit it runs on the VM.
- [`qwen36-claude-dogfood-playbook.md`](../qwen36-claude-dogfood-playbook.md) — the
  local/Mac dogfood path and the end-to-end "what this proves" map.
- [`claude-glm-gcp.md`](claude-glm-gcp.md) — the GLM-5.2 frontier-MoE sibling (8×A100).
