---
title: "claude-glm-gcp: use GLM-5.2 from the GCP kernel setup"
description: "One preset command points the real Claude Code CLI at GLM-5.2 served on a GCP GPU node, fronted by the fak kernel. How to stand the node up, wire the preset, and what is proven here vs hardware-gated."
---

# `claude-glm-gcp` — GLM-5.2 from the GCP kernel setup

One preset command runs the real Claude Code CLI on **GLM-5.2 served on GCP**, with the
fak kernel in front adjudicating every tool call. It is the `dogfood-claude` openai
backend pointed at a GLM `/v1` on a GCP GPU node — the same wire as
[`qwen36-claude-dogfood-playbook.md`](../qwen36-claude-dogfood-playbook.md), preset for
the GCP setup.

```
 ┌────────────────┐   /v1/messages   ┌────────────────────────┐  /v1/chat/...  ┌──────────────────┐
 │ claude-glm-gcp │ ───────────────▶ │ fak serve (the kernel) │ ─────────────▶ │ GLM-5.2 on a GCP  │
 │  (Claude Code) │ ◀──── SSE ─────── │ openai backend, adjud. │ ◀──────────── │ sm_90+ GPU node   │
 └────────────────┘                   └────────────────────────┘                │ (SGLang / vLLM)   │
        ▲ ANTHROPIC_BASE_URL=loopback fak           every tool call crosses the  └──────────────────┘
        │ FAK_GLM_GCP_BASE_URL=http://<tunnel-or-tailscale>:PORT/v1     kernel floor first
```

There are two halves: **stand the node up** (once, on GCP) and **use it here** (the
preset). The preset, the wire, and the bring-up plan are proven on any host; the live
model turn needs the GCP node up — see [What is proven here](#what-is-proven-here).

## Half A — stand up GLM-5.2 on GCP

`scripts/gcp-glm-serve.sh` provisions an sm_90+ GPU VM and runs the preflight-gated
SGLang/vLLM serve (`tools/glm52_sglang_vllm_serve.sh`). It is **plan-by-default**: with no
creds it prints the exact `gcloud` command, the VM startup script, and the
reach-from-laptop steps, so the whole deploy is reviewable first.

```bash
./scripts/gcp-glm-serve.sh                       # PLAN: gcloud + startup + reach steps
GCP_PROJECT=<id> ./scripts/gcp-glm-serve.sh --apply   # create the GPU VM
```

The default tier is `a3-ultra-h200` (8× H200, sm_90, ~$60/hr while up) from the
`tools/gcp_accel.py` registry — the most-provisionable tier that clears the
DeepSeek-Sparse-Attention floor. Knobs: `GCP_TIER` (e.g. `a4-b200` for Blackwell),
`GCP_ZONE`, `ENGINE` (sglang|vllm), `QUANT` (fp8|w4afp8|nvfp4|bf16), `GLM_PORT`,
`HF_TOKEN` (the GLM checkpoint is HF-gated), `TAILSCALE_AUTHKEY`.

> **Why sm_90+.** GLM-5.2's DSA kernels in stock SGLang/vLLM are gated to Hopper
> (sm_90) / Blackwell (sm_100). The default tier clears that, and the on-node preflight
> (`tools/glm52_serve_preflight.py`) **fails closed** if a tier doesn't. On Ampere (A100,
> sm_80) use the llama.cpp MLA path on a GPU server instead: `tools/glm52_serve.sh`.

## Half B — use it here

Install the launchers once, then point the preset at the node's `/v1`:

```bash
./scripts/dogfood-claude.sh --install            # installs `claude-glm-gcp` (+ fak, fak-dogfood, …)
```
```powershell
.\scripts\dogfood-claude.ps1 --install           # Windows: claude-glm-gcp.cmd + fak.exe
```

The node's `/v1` is reached over **Tailscale** or a **localhost SSH/IAP tunnel** (the
serve VM has no public ingress). The bring-up script prints the exact tunnel command;
the preset defaults to the tunnel port `http://127.0.0.1:8200/v1`:

```bash
# SSH/IAP tunnel — local :8200 -> VM :8000 (left running in another terminal)
gcloud compute ssh fak-glm-serve --zone us-central1-a --tunnel-through-iap \
  -- -N -L 8200:localhost:8000

# then, from here — one preset command:
claude-glm-gcp --probe "say pong"                # one witnessable headless turn
claude-glm-gcp                                    # interactive Claude Code on GLM-5.2
```

Or, if the VM joined Tailscale, skip the tunnel and dial it directly:

```bash
FAK_GLM_GCP_BASE_URL=http://fak-glm-serve:8000/v1 claude-glm-gcp
```

### What the preset is

`claude-glm-gcp` is the same launcher as `fak-dogfood`; its name selects
`FAK_DOGFOOD_PRESET=glm-gcp`. The preset defaults to:

| setting | value |
|---|---|
| backend | `openai` (fak proxies straight to the GLM `/v1`) |
| model-server URL | `FAK_GLM_GCP_BASE_URL` (default `http://127.0.0.1:8200/v1`) |
| model id | `glm-5.2` (the SGLang/vLLM `--served-model-name`) — every Claude tier maps onto it |
| timeout | `900s` (the openai-backend floor, for GLM's big prefill) |

Override any of those with the normal `FAK_DOGFOOD_*` env vars (`FAK_DOGFOOD_BASE_URL`
overrides the preset URL; `FAK_DOGFOOD_MODEL` overrides the id; `FAK_DOGFOOD_PORT` the
local kernel port).

## What is proven here

This follows the repo's serving-honesty boundary: the **mechanism** lands and is
witnessed; the **live model turn** is hardware-gated (a GCP GPU node with quota + creds,
which is not stood up from the implementing host — same gate as
[`gcp-tier2-control-vm.md`](gcp-tier2-control-vm.md) and
[`opencode-glm-guard.md`](opencode-glm-guard.md)).

| Item | Witness | Status |
|---|---|---|
| The `glm-gcp` preset resolves to fak's openai backend at the GLM `/v1` with model `glm-5.2` | `go test ./cmd/fak -run TestClaudeGLMGCP` (bash + PowerShell launchers) | ✅ proven on any host |
| The bring-up plan renders the gcloud create + preflight-gated serve + the `claude-glm-gcp` hand-off, with no creds | `go test ./cmd/fak -run TestClaudeGLMGCPBringupPlanRendersWithoutCreds` | ✅ proven on any host |
| The provisioner reads machine/accelerator/image from the single registry | `tools/gcp_accel.py --emit-shell` + `tools/gcp_accel_test.py` | ✅ proven on any host |
| The wire end-to-end (Anthropic `/v1/messages` → kernel) | `claude-glm-gcp --smoke` (offline mock planner; no model needed) | ✅ runnable here |
| A **live GLM-5.2 turn** through the preset | needs the GCP node up (Half A `--apply`) → `claude-glm-gcp --probe` | ⏳ hardware-gated (no GCP creds/GPU on the implementing host) |

The remaining step is operational: run Half A `--apply` on an authenticated host with
H200/B200 quota, open the tunnel, and run `claude-glm-gcp --probe "say pong"`.

## Troubleshooting

- **`OpenAI-compatible endpoint not reachable`** — the node isn't up or the tunnel/Tailscale
  path is down. Confirm `curl $FAK_GLM_GCP_BASE_URL/models` returns the `glm-5.2` id from
  here, then re-run. The launcher refuses to wire Claude Code to a dead upstream.
- **`PREFLIGHT BLOCKED` on the node** — the tier is below sm_90. Use `GCP_TIER=a3-ultra-h200`
  or `a4-b200`; see `glm52-<engine>-preflight.md` on the node.
- **First turn is slow** — GLM-5.2 is a large model; the preset's 900s floor and the
  gateway's SSE `ping` events keep a full Claude Code prompt alive during the prefill.

## Refs

- `scripts/gcp-glm-serve.sh` — the GCP bring-up (plan/apply)
- `scripts/dogfood-claude.sh` / `.ps1` — the launcher + the `glm-gcp` preset
- `tools/gcp_accel.py` — the GCP accelerator registry (`--emit-shell` feeds the bring-up)
- `tools/glm52_sglang_vllm_serve.sh` / `tools/glm52_serve_preflight.py` — the on-node serve + arch gate
- [`always-on-dogfood-server.md`](always-on-dogfood-server.md) — the always-on tiers + GPU-burst lane
- [`DOGFOOD-CLAUDE.md`](../../DOGFOOD-CLAUDE.md) — the general one-command dogfood launcher
