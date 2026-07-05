---
title: "claude-glm-gcp: use GLM-5.2 from the GCP kernel setup"
description: "One preset command points the real Claude Code CLI at GLM-5.2 served on a GCP GPU node, fronted by the fak kernel. How to stand the node up, wire the preset, and what is proven here vs hardware-gated."
---

# `claude-glm-gcp` — GLM-5.2 from the GCP kernel setup

> **Audience.** Operators standing up GLM-5.2 on a GCP GPU node and pointing Claude Code at it through the fak kernel. By the end you can run the `claude-glm-gcp` preset and know what is proven on any host versus what needs the live node.

One preset command runs the real Claude Code CLI on **GLM-5.2 served on GCP**, with the
fak kernel in front adjudicating every tool call. It is the `dogfood-claude` openai
backend pointed at a GLM `/v1` on a GCP GPU node — the same wire as
[`qwen36-claude-dogfood-playbook.md`](../qwen36-claude-dogfood-playbook.md), preset for
the GCP setup.

```
 ┌────────────────┐   /v1/messages   ┌────────────────────────┐  /v1/chat/...  ┌──────────────────┐
 │ claude-glm-gcp │ ───────────────▶ │ fak serve (the kernel) │ ─────────────▶ │ GLM-5.2 on a GCP  │
 │  (Claude Code) │ ◀──── SSE ─────── │ openai backend, adjud. │ ◀──────────── │ GPU node (A100+)   │
 └────────────────┘                   └────────────────────────┘                │ fak kernel/llama  │
        ▲ ANTHROPIC_BASE_URL=loopback fak           every tool call crosses the  └──────────────────┘
        │ FAK_GLM_GCP_BASE_URL=http://<tunnel-or-tailscale>:PORT/v1     kernel floor first
```

There are two halves: **stand the node up** (once, on GCP) and **use it here** (the
preset). The preset, the wire, and the bring-up plan are proven on any host; the live
model turn needs the GCP node up — see [What is proven here](#what-is-proven-here).

## The one command: `gcp-glm-demo.sh`

If you just want the whole demo as a single reviewable command, `scripts/gcp-glm-demo.sh`
chains the four steps end to end, **plan-by-default**, defaulting to the **8× H100** tier
(`a3-high-h100`, 640 GB — GLM-5.2 at `UD-Q4_K_M` is ~466 GB, so it needs the 8-GPU shape):

1. **provision + serve** GLM-5.2 through the **pure fak kernel** (it forces `SERVE=fak`
   even on sm_90, where the bring-up would otherwise pick stock SGLang — this is the whole
   point of "fak kernel", and the precondition for step 3's metric to exist at all). It
   *composes* `gcp-glm-serve.sh` rather than re-implementing the gcloud/serve rendering.
2. **probe** — two `claude-glm-gcp --probe` turns that share a system+tools prefix, so
   turn 2 reuses the KV prefix fak already holds resident.
3. **cache value** — scrape fak's own realized KV-prefix reuse off the serve `/metrics`
   surface (`fak_gateway_kv_prefix_reused_tokens_total` > 0 = the witnessed demo datum).
4. **teardown** — delete the VM on an EXIT trap so the demo leaves zero residual cost.

```bash
./scripts/gcp-glm-demo.sh                                # PLAN: the whole demo, no creds
GCP_PROJECT=<id> HF_TOKEN=<hf> ./scripts/gcp-glm-demo.sh --apply   # run it on 8× H100, then tear down
KEEP=1 GCP_PROJECT=<id> ./scripts/gcp-glm-demo.sh --apply # skip teardown to debug the node
```

For a real dev day, prefer `scripts/gcp-glm-serve.sh` directly with the budget guard below.
`gcp-glm-demo.sh` is for a short witnessed demo and deliberately tears down.

The plan render is proven on any host (`go test ./cmd/fak -run TestClaudeGLMGCPDemoPlan`);
the live cache-value turn is hardware-gated, the same gate as the rest of this page.

## Half A — stand up GLM-5.2 on GCP

`scripts/gcp-glm-serve.sh` provisions a GPU VM and stands the GLM-5.2 serve up on it. It is
**plan-by-default**: with no creds it prints the exact `gcloud` command, the VM startup
script, and the reach-from-laptop steps, so the whole deploy is reviewable first.

```bash
./scripts/gcp-glm-serve.sh                            # PLAN: gcloud + startup + reach steps
GCP_PROJECT=<id> ./scripts/gcp-glm-serve.sh --apply   # create the GPU VM
```

### $500/day dev envelope

A **usable dev setup under $500/day is a bounded burst node, not a 24/7 GLM-5.2
server**. GLM-5.2 needs an 8-GPU shape for the current GGUF path; at current
on-demand order-of-magnitude pricing, every practical 8-GPU tier exceeds $500 if it is
left up all day. The bring-up script now defaults to `MAX_DAILY_USD=500` and renders a
one-shot **budget reaper** on the VM: after the budget-derived max lifetime, the VM
self-`delete`s by default. The idle reaper remains separate and deletes earlier when no
model turns arrive.

| tier | role | approx $/hr in `tools/gcp_accel.py` | max runtime under $500 |
|---|---|---:|---:|
| `a2-high-a100-40gb` | cheapest GLM-sized A100 path; tighter host/VRAM headroom | 29.39 | 17h 0m |
| `a2-ultra-a100-80gb` | recommended budget dev tier when A100 quota is available | 40.55 | 12h 19m |
| `a3-ultra-h200` | Hopper/H200 fallback with much shorter budget window | 84.81 | 5h 53m |
| `a3-mega-h100` | H100 Mega fallback when standard A3 High is stocked out but Mega quota/capacity exists | 90.00 | 5h 33m |
| `a3-high-h100` | demo/Hopper tier; useful for short witnessed runs | 88.49 | 5h 39m |
| `a4-b200` | Blackwell target; quota/reservation gated | 90.00 | 5h 33m |

Recommended dev command:

```bash
MAX_DAILY_USD=500 \
MAX_RUNTIME_ACTION=delete \
ON_IDLE=delete IDLE_MINUTES=45 GRACE_MINUTES=90 \
GCP_TIER=a2-ultra-a100-80gb SERVE=fak \
GCP_PROJECT=<id> HF_TOKEN=<hf> ./scripts/gcp-glm-serve.sh --apply
```

Use `MAX_RUNTIME_ACTION=stop` only when you deliberately want to preserve the boot disk
for faster next-day bring-up; `delete` is the zero-residual-cost default. Use
`MAX_RUNTIME_MINUTES=<n>` to make the cap stricter than the computed `$500 / hourly`
window, for example `MAX_RUNTIME_MINUTES=480` for an eight-hour workday.

If an on-demand full-size H100 create fails with
`ZONE_RESOURCE_POOL_EXHAUSTED_WITH_DETAILS`, first try any `zonesAvailable` hint GCP
returns. If the stockout is broad, `PROVISIONING_MODEL=FLEX_START
REQUEST_VALID_FOR_DURATION=2h` queues the VM request instead of failing fast, but it
uses preemptible GPU quota; verify that quota is at least the tier's GPU count before
expecting Flex-start to work. `a3-mega-h100` uses the separate `NVIDIA_H100_MEGA`
quota family and is the practical full-size fallback when `a3-high-h100` is stocked
out but Mega capacity exists.

Before apply, create a Cloud Billing budget scoped to the project at **$500** with
50% / 80% / 100% actual-spend alerts. Treat that as an alerting plane, not the cap:
Google budgets do not automatically stop resources. The VM reapers are the compute-side
cap; billing alerts are the independent spend witness.

Operator checklist for a live dev setup:

1. Pick the tier: start with `a2-ultra-a100-80gb` for the best budget/headroom balance;
   use `a2-high-a100-40gb` only when runtime is more valuable than headroom.
2. Confirm quota in the chosen region for both the GPU family and the all-regions GPU
   ceiling; if quota is missing, request it before running `--apply`.
3. Create the project-scoped $500 Cloud Billing budget and alerts, then rely on
   `MAX_DAILY_USD`, `MAX_RUNTIME_MINUTES`, and the idle reaper for compute-side stopping.
4. Ensure the VM's attached service account can stop/delete its own instance
   (`roles/compute.instanceAdmin.v1` or a narrower equivalent); otherwise the reapers log a
   refusal and the node stays up.
5. Run the plan command with the exact env you intend to apply; read the rendered
   `gcloud` command, startup script, idle reaper, budget reaper, and tunnel steps.
6. Run `--apply` from an authenticated host with `GCP_PROJECT` and `HF_TOKEN`.
7. Watch readiness on the VM with `journalctl -u glm52serve -f` and `cat /opt/glm52-q4/PHASE`;
   only move to the client when the endpoint answers `/v1/models` and a chat smoke.
8. Open the IAP tunnel or use Tailscale, then run `claude-glm-gcp --probe "say pong"`.
9. Use `claude-glm-gcp` for the dev session. During the run, `curl /metrics` should expose
   fak's gateway counters; after the day, verify the VM stopped/deleted or delete it manually.

It picks the serve path from the tier's GPU arch (override with `SERVE=`):

| `SERVE` | what runs | default on |
|---|---|---|
| `fak` | the **pure fak kernel** — fak serves GLM-5.2 (`glm_moe_dsa`) through its *own* CUDA kernels (`tools/glm52_fak_native_serve.sh`: `fak serve --gguf <shard1> --backend cuda --cpu-offload-experts --context-budget-tokens 8192`). **Preferred.** | Ampere (datacenter GPU, sm_80) |
| `llamacpp` | the **benchmark baseline** — the *same* checkpoint under llama.cpp MLA + CPU expert-offload (`tools/private GLM-5.2 stage runner`, the GPU server example brought to GCP). Stand it up to compare fak apples-to-apples. | — (opt-in, any tier) |
| `sglang` / `vllm` | stock DSA engines (`tools/glm52_sglang_vllm_serve.sh`), gated by `tools/glm52_serve_preflight.py` (fails closed below sm_90). | Hopper / Blackwell (sm_90+) |

So **"whatever is available" includes datacenter GPU** — `GCP_TIER=a2-ultra-a100-80gb` serves GLM-5.2
via the pure fak kernel by default:

```bash
GCP_TIER=a2-ultra-a100-80gb ./scripts/gcp-glm-serve.sh                 # A100, pure fak kernel (default)
GCP_TIER=a2-ultra-a100-80gb SERVE=llamacpp ./scripts/gcp-glm-serve.sh  # A100, llama.cpp benchmark node
GCP_TIER=a3-mega-h100 SERVE=fak ./scripts/gcp-glm-serve.sh             # H100 Mega, pure fak kernel
GCP_TIER=a4-b200 ./scripts/gcp-glm-serve.sh                            # Blackwell, stock SGLang/vLLM
```

The default tier is `a3-ultra-h200` (8× H200, sm_90, ~$84.81/hr) from the `tools/gcp_accel.py`
registry — the most-provisionable tier that clears the DSA floor. The datacenter GPU tiers are
`a2-ultra-a100-80gb` (8-GPU datacenter server 80GB, 640 GB VRAM, ~$40.55/hr — the same shape as the GPU server example)
and `a2-high-a100-40gb` (8-GPU datacenter server 40GB, ~$29.39/hr). Knobs: `GCP_TIER`, `SERVE`, `GCP_ZONE`,
`ENGINE`/`QUANT` (the sm_90 stock path), `GLM_GGUF_REPO`/`GLM_GGUF_SUBDIR` (the fak/llama.cpp
GGUF, default `unsloth/GLM-5.2-GGUF` `UD-Q4_K_M`), `NCPU_MOE`, `GLM_PORT`, `HF_TOKEN`,
`CTX` (stock SGLang/vLLM context, default 65536), `EP_RANKS` (pure-fak resident
expert-parallel ranks; `1` keeps the cpu-offload smoke path, `8` launches one rank per
full-size GPU for the no-cpu-offload path), `TAILSCALE_AUTHKEY`, `MAX_DAILY_USD`,
`MAX_RUNTIME_MINUTES`, and `MAX_RUNTIME_ACTION`.

For a **pure fak performance attempt** on a full-HBM 8-GPU tier, set `SERVE=fak
EP_RANKS=8`. That runs the resident expert-parallel topology: one fak rank per GPU,
`FAK_Q4K=1`, `-tags cuda,nccl`, and `--expert-parallel 8`, instead of the
single-process `--cpu-offload-experts` smoke path.

The stock SGLang/vLLM path defaults `CTX=65536` because Claude Code's GLM dogfood probe
asks for a large Anthropic turn: the witnessed run used 32,641 input tokens plus a
32,000-token output ceiling. A 32,768-token serve window fails that request before the
model starts.

> **Why A100 needs a different serve.** GLM-5.2's DSA kernels in stock SGLang/vLLM are gated
> to Hopper (sm_90) / Blackwell (sm_100); on Ampere (datacenter GPU, sm_80) the preflight
> (`tools/glm52_serve_preflight.py`) **fails closed** (vLLM #35021). Two paths clear it on
> datacenter GPU: fak's **own** kernel runs the `glm_moe_dsa` forward as full MLA (no sm_90 kernel
> needed) — **bit-exact vs the CPU reference at q8 (cosine 1.0, argmax-exact), witnessed on
> sm_80** (`experiments/glm-gpu-witness/a100-glm52-*.json`, incl. the cpu-offload hybrid). The
> `--cpu-offload-experts` serve runs the resident-Q4_K path, which is not yet covered by a
> full-forward cosine witness (the q8 forward is). llama.cpp serves the same GGUF as the honest
> throughput baseline. **Prefer the pure fak kernel; keep llama.cpp for the comparison.** (See the benchmarking framework in
> `docs/notes/GLM52-NATIVE-THROUGHPUT-AND-BENCHMARK-PLAN-2026-06-25.md` — never put the two
> numbers side by side without holding {weights, hardware, precision, context} equal.)

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
| provider extra body | `{"chat_template_kwargs":{"enable_thinking":false}}` (keeps probe output in `content`) |

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
| The bring-up plan renders the gcloud create + serve + the `claude-glm-gcp` hand-off, with no creds | `go test ./cmd/fak -run TestClaudeGLMGCPBringupPlanRendersWithoutCreds` | ✅ proven on any host |
| The bring-up plan renders a `MAX_DAILY_USD` max-lifetime budget reaper in addition to the idle reaper | `go test ./cmd/fak -run TestClaudeGLMGCPBringupPlanWiring`; plan output contains `fak-budget-reaper` | ✅ proven on any host |
| The **one-command demo** (`gcp-glm-demo.sh`) renders provision → pure-fak serve → probe → cache-value scrape → teardown on the **8× H100** tier, no creds | `go test ./cmd/fak -run TestClaudeGLMGCPDemoPlan` (wiring on any OS; bash render on Unix/CI) | ✅ proven on any host |
| A **live cache-value turn** (`fak_gateway_kv_prefix_reused_tokens_total` > 0 on turns 2..N) through the demo | needs the 8× H100 node up (`gcp-glm-demo.sh --apply`) | ⏳ hardware-gated |
| The **datacenter GPU tiers** are in the single registry (`a2-ultra-a100-80gb`, `a2-high-a100-40gb`) | `tools/gcp_accel_test.py` + `go test ./cmd/fak -run TestClaudeGLMGCPA100TiersInRegistry` | ✅ proven on any host |
| The datacenter GPU plan **wires the pure fak kernel** by default; `SERVE=llamacpp` wires the llama.cpp benchmark | `go test ./cmd/fak -run 'TestClaudeGLMGCPA100Plan'` (WSL/Unix CI) + `TestClaudeGLMGCPFakNativeServeWiring` | ✅ proven on any host |
| The fak-native **`glm_moe_dsa` forward is bit-exact** on datacenter GPU (sm_80) **at q8**: cosine 1.0, argmax-exact (incl. the cpu-offload hybrid) | `experiments/glm-gpu-witness/a100-glm52-*.json` (`TestCUDAGLMMoeDsaBackendForward` …) | ✅ witnessed (q8) on sm_80 |
| The resident-**Q4_K** serve path (`--cpu-offload-experts`) is cosine-witnessed end to end | — (the q8 forward is; the served Q4_K path is not) | ⏳ not yet |
| The wire end-to-end (Anthropic `/v1/messages` → kernel) | `claude-glm-gcp --smoke` (offline mock planner; no model needed) | ✅ runnable here |
| A **live GLM-5.2 turn** through the preset against SGLang W4AFP8 on 8x H100 Mega | `experiments/agent-live/dogfood-claude-glm52-gcp-20260705.json`; gateway `/v1/messages` 200 in 11.8s | ✅ witnessed |
| A **live GLM-5.2 turn** through the preset (pure fak kernel **or** llama.cpp) | needs the GCP node up (Half A `--apply`) → `claude-glm-gcp --probe` | ⏳ hardware-gated |

The pure-fak-kernel serve command is **wired** and the `glm_moe_dsa` forward is **witnessed at
q8** (cosine 1.0, sm_80); a live serve turn stays **hardware+load-gated** — the resident-Q4_K
serve path is not yet cosine-witnessed, and the **load** on the dynamic-mixed `UD-Q4_K_M` is
the open perf item (the resident-Q4_K path fully fires only on pure-Q4_K tensors; see
`docs/notes/GLM52-FAK-NATIVE-SERVE-LOAD-SPEED-2026-06-25.md`). The remaining step is
operational: run Half A `--apply` on an authenticated host with datacenter GPU (or H200/B200) quota, open
the tunnel, and run `claude-glm-gcp --probe "say pong"`.

## Troubleshooting

- **`OpenAI-compatible endpoint not reachable`** — the node isn't up or the tunnel/Tailscale
  path is down. Confirm `curl $FAK_GLM_GCP_BASE_URL/models` returns the `glm-5.2` id from
  here, then re-run. The launcher refuses to wire Claude Code to a dead upstream.
- **`PREFLIGHT BLOCKED` on the node** — the tier is below sm_90. Use `GCP_TIER=a3-ultra-h200`
  or `a4-b200`; see `glm52-<engine>-preflight.md` on the node.
- **First turn is slow** — GLM-5.2 is a large model; the preset's 900s floor and the
  gateway's SSE `ping` events keep a full Claude Code prompt alive during the prefill.

## Refs

- `scripts/gcp-glm-demo.sh` — the **one-command demo** (plan/apply): provision → pure-fak serve → probe → cache-value → teardown, default tier `a3-high-h100` (8× H100)
- `scripts/gcp-glm-serve.sh` — the GCP bring-up (plan/apply), `SERVE=fak|llamacpp|sglang|vllm`
- `scripts/dogfood-claude.sh` / `.ps1` — the launcher + the `glm-gcp` preset
- `tools/gcp_accel.py` — the GCP accelerator registry (datacenter GPU tiers + `--emit-shell` feeds the bring-up)
- `tools/glm52_fak_native_serve.sh` — the **pure fak kernel** on-node serve (datacenter GPU default)
- `tools/private GLM-5.2 stage runner` — the **llama.cpp benchmark** on-node serve (the GPU server example)
- `tools/glm52_sglang_vllm_serve.sh` / `tools/glm52_serve_preflight.py` — the sm_90 stock serve + arch gate
- `internal/compute/build_cuda.sh binary <pkg> <out>` — the DRY `-tags cuda` binary build the fak-native serve uses
- [`GLM52-NATIVE-THROUGHPUT-AND-BENCHMARK-PLAN-2026-06-25.md`](../notes/GLM52-NATIVE-THROUGHPUT-AND-BENCHMARK-PLAN-2026-06-25.md) — the honest fak-vs-llama.cpp comparison framework
- [`always-on-dogfood-server.md`](always-on-dogfood-server.md) — the always-on tiers + GPU-burst lane
- [`DOGFOOD-CLAUDE.md`](https://github.com/anthony-chaudhary/fak/blob/main/DOGFOOD-CLAUDE.md) — the general one-command dogfood launcher
