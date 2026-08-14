---
title: "DeepSeek V4 through fak — hosted, Anthropic-compatible, and self-hosted routes"
description: "The exact copy/paste commands for the three DeepSeek V4 routes fak fronts (hosted OpenAI-compatible, Anthropic-compatible for Claude Code-style clients, and self-hosted vLLM/SGLang/NIM), the thinking-mode caveats, the provider-relayed cache-counter stance, the self-hosted GPU-node bring-up path, and the prominent warning that the deepseek-chat / deepseek-reasoner aliases retire 2026-07-24 15:59 UTC."
---

# DeepSeek V4 through fak

fak fronts DeepSeek as a wire, the same way it fronts any provider: it **governs
and relays; it does not generate**. This page is the operator's exact-command
reference for the three DeepSeek V4 routes, the thinking-mode caveats that bite in
practice, and the honest limits. It is grounded in the official DeepSeek docs
(researched **July 2026**, linked in [Sources](#sources-researched-july-2026)).

> ## ⚠️ Alias retirement — `deepseek-chat` / `deepseek-reasoner` retire 2026-07-24 15:59 UTC
>
> The `deepseek-chat` and `deepseek-reasoner` model ids are **compatibility
> aliases only** and retire on **2026-07-24 15:59 UTC**
> ([vendor announcement](https://api-docs.deepseek.com/news/news260424)). Pin
> **`deepseek-v4-pro`** (or `deepseek-v4-flash`) explicitly. Any script, account
> profile, or harness config still naming an alias after the cutoff will break —
> migrate now.

## The three routes at a glance

| Route | When to use it | Model id | Base URL |
|---|---|---|---|
| **Hosted OpenAI-compatible** | Any OpenAI-Chat-Completions client through `fak serve`/`fak manage` | `deepseek-v4-pro` | `https://api.deepseek.com` |
| **Anthropic-compatible** | Claude Code-style clients that speak Anthropic Messages | `deepseek-v4-pro` | `https://api.deepseek.com/anthropic` |
| **Self-hosted (vLLM/SGLang/NIM)** | Your own GPUs (Hopper/Blackwell-class node) | `deepseek-ai/DeepSeek-V4-Pro` / `-Flash` | `http://<host>:8000/v1` |

All examples use the `DEEPSEEK_API_KEY` environment variable and **never a literal
key**.

### 1. Hosted OpenAI-compatible route

The hosted DeepSeek API exposes the OpenAI Chat Completions surface. Front it with
fak as a generic OpenAI-compatible upstream:

```bash
export DEEPSEEK_API_KEY=...   # your key; never commit it

fak serve --provider openai-compatible \
  --base-url "https://api.deepseek.com" \
  --model "deepseek-v4-pro" \
  --api-key-env DEEPSEEK_API_KEY
```

The first request uses model `deepseek-v4-pro`, an OpenAI-compatible base URL, and
supports `thinking`, `reasoning_effort`, and the streaming toggle
([vendor quick start](https://api-docs.deepseek.com/)).

### 2. Anthropic-compatible route (Claude Code-style clients)

DeepSeek also exposes an **Anthropic Messages**-compatible surface at
`https://api.deepseek.com/anthropic`
([coding-agents guide](https://api-docs.deepseek.com/guides/coding_agents)). A
Claude Code-style client can point at fak, which fronts that surface:

```bash
export DEEPSEEK_API_KEY=...

fak serve --provider anthropic \
  --base-url "https://api.deepseek.com/anthropic" \
  --model "deepseek-v4-pro" \
  --api-key-env DEEPSEEK_API_KEY
```

### 3. Self-hosted route (your own GPUs)

Run the model on a tuned engine exposing the OpenAI surface, then front it. This
is the path for a self-hosted bring-up — see [Bring V4 up on a GPU node](#bring-v4-up-on-a-gpu-node)
for the one-command script and the **architecture floor** you must confirm first.

```bash
# On the serving node (see the self-host runbook for engine flags):
vllm serve deepseek-ai/DeepSeek-V4-Pro --served-model-name deepseek-ai/DeepSeek-V4-Pro --port 8000
# → OpenAI root: http://<host>:8000/v1

# Then, from anywhere that can reach it:
fak serve --provider openai-compatible \
  --base-url "http://<host>:8000/v1" \
  --model "deepseek-ai/DeepSeek-V4-Pro"
```

fak makes **no claim to natively run** the 1.6T-total V4-Pro weights; the model is
served by vLLM/SGLang/NIM and fak sits in front. Native in-kernel loader work is a
separately scoped effort (parent program
[#3006](https://github.com/anthony-chaudhary/fak/issues/3006)).

## Thinking-mode caveats (these bite in practice)

DeepSeek V4 is a reasoning model. The behavior differs from a plain chat model in
ways that affect harness integration:

- **`thinking` defaults to ENABLED.** A turn may spend its whole token budget on
  reasoning and return **empty `content` with populated `reasoning_content`** —
  that is a live, correct response, not a failure. (fak's self-host smoke treats a
  reasoning-only turn as a live wire for exactly this reason.)
- **`reasoning_effort` accepts only `high` / `max`.** There is no low/medium tier.
- **Sampling controls do not affect thinking mode.** `temperature`, `top_p`,
  `presence_penalty`, and `frequency_penalty` have **no effect** on the thinking
  path — do not expect them to steer reasoning.
- **Tool-call turns require reasoning-content transcript preservation.** When a
  turn makes a tool call, the `reasoning_content` from that turn must be carried
  forward in the transcript for the follow-up turn to be correct. fak preserves
  DeepSeek reasoning content across tool-call turns (see the gateway/agent
  reasoning-preservation coverage); a harness that strips it will degrade the
  model.

## Cache counters are provider-relayed, not fak-authored savings

DeepSeek reports prompt-cache hit/miss token counts in its usage block. fak
**relays and prices these as the provider reports them** — they are
**provider-relayed, provider-priced** counters. They are **not** a fak-authored
saving and must never be presented as fak cache savings. fak's own cache economics
(the addressable in-kernel cache) are a distinct, separately-witnessed subsystem.

## Bring V4 up on a GPU node

The one-command path is [`scripts/dgx-deepseek-serve.sh`](../../scripts/dgx-deepseek-serve.sh):
it detects your GPUs, **gates on the DeepSeek-V4 architecture floor**, launches the
OpenAI server on `:8000`, health-checks `/models`, and prints the exact `fak serve`
+ wire-witness commands.

> **Architecture floor — confirm before reserving GPU time.** V4 checkpoints are
> **FP4** (MoE experts) + **FP8** (most params) with an **FP4 attention indexer**,
> and the attention stack is **DSA (DeepSeek Sparse Attention)**. **Stock** kernels
> (vLLM/SGLang/NIM) generally need **sm_90+ (Hopper)**, with **sm_100 (Blackwell)**
> for native NVFP4. `scripts/dgx-deepseek-serve.sh` fronts a stock engine and warns
> and refuses below the floor unless `FAK_DGX_FORCE=1`. V4-**Pro** (1.6T total) is a
> **multi-node** expert-parallel deployment; size a single node against V4-**Flash**
> first.
>
> **sm_80 (Ampere/A100) is no longer dead-ended.** A V4-aware **llama.cpp fork**
> ([cchuter `feat/v4-port-cuda`](https://github.com/cchuter/llama.cpp); upstream
> [PR #24162](https://github.com/ggml-org/llama.cpp/pull/24162) for Unsloth quants)
> serves **V4-Flash** on sm_80 via CUDA V4-op kernels + a software-emulated FP8 path
> — the same route that overcame the sm_90 wall for GLM-5.2 on A100. V4-Flash GGUFs
> (≈103–162 GB) fit **resident** on 8×A100-80 GB. The one-command harness is
> [`tools/deepseekv4_stage_serve_gpu_server.sh`](../../tools/deepseekv4_stage_serve_gpu_server.sh).
> This is the interim serve path; the fak-native V4 kernel is a separate track
> ([#3016](https://github.com/anthony-chaudhary/fak/issues/3016)–[#3019](https://github.com/anthony-chaudhary/fak/issues/3019)).

The full procedure, the minimum-evidence witness ladder, and the comparability
boundary are in the
[self-host baseline runbook](../benchmarks/DEEPSEEK-V4-SELFHOST-BASELINE-RUNBOOK.md).

### Wire witness (what fak has proven)

fak's `openai-compatible` route has been **witnessed live against a real DeepSeek-V4
endpoint** (`deepseek-ai/deepseek-v4-flash`), passing all three rungs — readiness,
non-streaming (with honest usage counters), and streaming to `[DONE]`. See the
runbook's [Live witness](../benchmarks/DEEPSEEK-V4-SELFHOST-BASELINE-RUNBOOK.md#live-witness--wire-proven-on-a-real-v4-endpoint)
section. This proves the **wire**; a **performance headline** still requires the
tuned EP/EPLB baseline on a dedicated node and is not claimed here.

## Budget before you claim 1M context

Before quoting 1M-context support, model the memory/KV economics with the
checked-in `fak`-side budget calculator (issue
[#3015](https://github.com/anthony-chaudhary/fak/issues/3015),
`internal/gateway/deepseek_budget.go`): it estimates FP4+FP8 weight storage,
activated-parameter compute proxy, and compressed-KV vs SWA/state-cache tail
pressure per model and context length, labelling every row **MODELED** until fed by
witnessed serving telemetry.

## Acceptance mapping (issue #3012)

- **Supported-provider page added** — this page.
- **Copy/paste commands for all three routes** — [above](#the-three-routes-at-a-glance),
  each using `DEEPSEEK_API_KEY`, never a literal key.
- **Thinking-mode caveats documented** — [Thinking-mode caveats](#thinking-mode-caveats-these-bite-in-practice):
  `thinking` default-on, `reasoning_effort` high/max only, sampling controls inert,
  tool-call reasoning-content preservation.
- **Provider cache counters framed as provider-relayed/provider-priced** —
  [Cache counters](#cache-counters-are-provider-relayed-not-fak-authored-savings).
- **Alias-retirement warning prominent** — the top-of-page callout (2026-07-24
  15:59 UTC).
- **Links to official sources with the research date** —
  [Sources](#sources-researched-july-2026).
- **`rg "deepseek-chat|deepseek-reasoner" docs examples`** finds only this
  migration/compatibility context.

## Sources (researched July 2026)

- DeepSeek API quick start (OpenAI-compatible base URL, `thinking`,
  `reasoning_effort`, streaming): <https://api-docs.deepseek.com/>
- Coding-agents guide (Anthropic-compatible route for Claude Code/OpenCode):
  <https://api-docs.deepseek.com/guides/coding_agents>
- Pricing / model page (capabilities, context, max output, alias retirement):
  <https://api-docs.deepseek.com/quick_start/pricing>
- V4 preview / alias-retirement announcement (`deepseek-chat` / `deepseek-reasoner`
  retire 2026-07-24 15:59 UTC): <https://api-docs.deepseek.com/news/news260424>
- V4-Pro model card (FP4+FP8 mixed precision, MoE experts FP4):
  <https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro>

## Cross-links

- [Clouds & hosted providers](clouds.md) — the full hosted-provider list.
- [Self-host baseline runbook](../benchmarks/DEEPSEEK-V4-SELFHOST-BASELINE-RUNBOOK.md)
  · [Perf scorecard](../benchmarks/DEEPSEEK-V4-PERF-SCORECARD.md)
- [Model accounts](../model-accounts.md) · Parent program
  [#3006](https://github.com/anthony-chaudhary/fak/issues/3006).
