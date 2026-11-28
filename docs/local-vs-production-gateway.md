---
title: "Local runtimes vs production LLM gateways: where fak fits"
description: "How local runtimes (Ollama, LM Studio, llama-server, llama-swap) and production stacks (vLLM, SGLang, llm-d, LiteLLM, Bifrost, Portkey, Kong) split on engine precision, admission, routing, governance, and footprint — and how fak serve unifies gateway, router, local engine, and capability floor in one zero-Python Go binary."
---

# Local runtimes vs production LLM gateways

`fak serve` puts a local engine, an OpenAI-compatible router, and a tool-call guard in one Go binary.

> **TL;DR:** Scaffold a hybrid local-plus-fallback roster with `fak route init --out-dir ./local-router`. Keep private data on-box with `residency_floor: "local_only"` and govern every tool call in the same process.

## Quickstart: local primary, cloud fallback, and `local_only`

Generate a starter roster and manifest, or inspect [`examples/local-router/`](../examples/local-router/):

```sh
fak route init --out-dir ./local-router
```

In `accounts.json`, two bindings show the split:

- Default coding lane (`local` primary with cloud fallback): tries the local `fak` engine first and walks ordered `fallbacks` if its circuit breaker trips.
- Private lane (`residency_floor: "local_only"`): binds only to local accounts. Roster validation rejects remote fallbacks, and runtime failures fail closed instead of spilling off-box.

Start the gateway with the starter roster, manifest, and capability floor armed:

```sh
fak serve \
  --addr 127.0.0.1:8080 \
  --guard \
  --route-manifest examples/local-router/manifest.json \
  --route-accounts examples/local-router/accounts.json \
  --policy examples/customer-support-readonly-policy.json
```

Credentials in `accounts.json` store environment-variable names rather than secret values. `fak` resolves them only at outbound dispatch.

## The market split

Local workstation runtimes and datacenter serving stacks split sharply across engine precision, routing, and governance:

| Axis | Local runtimes (`Ollama`, `LM Studio`, `llama-server`, `llama-swap`) | Production stacks (`vLLM`, `SGLang`, `llm-d`, `LiteLLM`, `Bifrost`, `Portkey`, `Kong`) | `fak serve` |
|---|---|---|---|
| Inference engine and KV precision | Local GGUF/MLX execution (`llama.cpp` and `Ollama` use `q4_0`/`iq4_nl` KV quantization) | Python/CUDA datacenter engines with FP4/FP8 KV cache and multi-node parallelism; proxies bring no engine | In-kernel Go + CUDA/Metal/Vulkan engine (`--engine inkernel`) with native NVFP4 (`sm_120`) and proxy modes |
| Gateway and admission | Single-user local HTTP port without tenant admission or budget isolation | Multi-tenant admission, token/rate limits, continuous batching, disaggregated prefill/decode | `/v1/chat/completions` and `/v1/messages` gateway with request admission (25 ms gap / 75 ms wait on eligible CUDA), context checks, and `/metrics` |
| Routing and failover | Single local model or manual model swaps (`llama-swap`, `llama-server`) | Embedding/cache-aware routing (`vLLM`, `sgl-router`, `llm-d`) and multi-provider failover (`LiteLLM`, `Portkey`, `Kong`, `Bifrost`) | `--route-manifest` + `--route-accounts` with ordered `fallbacks`, per-account circuit breaker, and fail-closed `residency_floor: "local_only"` |
| Tool-call governance | Pass-through to client harness | Text guardrails (regex, moderation, PII webhooks), not per-tool capability locks | Default-deny `--guard` / `--policy` floor on every tool call (`/v1/fak/adjudicate`) plus result quarantine (`/v1/fak/admit`) |
| Operational footprint | Desktop app or C++ binary (`llama-server` + `llama-swap`) | Python runtimes, PyTorch containers, Redis/Postgres, or Kubernetes operators | One static Go binary (`fak`) |

## What `fak serve` unifies

Four integrated layers replace separate local runners, Python proxies, and policy sidecars:

1. **Local engine (`--engine inkernel`):** runs GGUF models directly on Metal, CUDA, Vulkan, or CPU. On a 16 GB RTX 5080 (`sm_120`), opt-in `FAK_INKLING2_KV=nvfp4` keeps all 24/24 expert layers of Inkling-2-Mini 26B (`UD-Q3_K_XL`, `ctx=131,072`) in VRAM with 0 GiB spill vs 12/24 on FP16. That raises 128k prefill 4.21× (283 to 1,190 tok/s) and cuts wall time 4.10× (451s to 110s) with 4/4 `llama.cpp` greedy oracle pass ([README.md](../README.md)).
2. **Gateway and router (`--route-manifest`, `--route-accounts`):** serves OpenAI/Anthropic chat wires and resolves model IDs to concrete accounts. Ordered `fallbacks` take over automatically when a primary cools down.
3. **Fail-closed residency floor (`residency_floor: "local_only"`):** enforces locality both when loading the roster and before dispatch. If a local account is unavailable, the call fails closed rather than leaking to cloud.
4. **Capability floor (`--guard` / `--policy`):** checks every proposed tool call against policy and quarantines tainted tool output before re-entering model context.

## Live inspection

Two HTTP endpoints inspect live models and router health on a running gateway:

```sh
curl -s http://127.0.0.1:8080/v1/models
curl -s http://127.0.0.1:8080/v1/router/status
```

- `GET /v1/models`: returns the in-kernel model plus all bound IDs from `--route-accounts` (`owned_by: "fak-route-accounts"`).
- `GET /v1/router/status`: reports per-account health, circuit cooldowns, fallback chains, and `residency_floor` posture.

## Honest gaps

- Provider catalog breadth: Dedicated gateways such as LiteLLM and Portkey focus on SaaS catalogs. `fak` covers OpenAI, Anthropic, and local wires alongside them (see [fak vs routers](adoption/compare/vs-routers.md)).
- Multi-node distributed serving: `fak` is a single-node binary. Use `llm-d`, `vLLM`, or `sgl-router` for multi-node tensor/pipeline parallelism or disaggregated prefill/decode clusters.
- NVFP4 / FP8 KV precision gating and decode tail: On RTX 5080, `nvfp4` and `fp8` KV stay opt-in (`FAK_INKLING2_KV`) while stricter needle-retrieval KL gating continues. At 1,946 context tokens, `fak` still trails the fastest `llama.cpp` fork on decode (89.0 vs 113.8 tok/s), and first visible token arrives later.
- Local-only KV eviction: Bit-exact mid-run KV eviction (`max|Δ| = 0`) works only on the in-kernel local engine. Remote cloud fallbacks carry no local KV state to evict.
- Learned routing and micro-scout telemetry loop: `/metrics` exports Prometheus latency histograms, and the gateway exports async OTLP `/v1/traces` HTTP spans with `gen_ai.*` and OpenInference attributes. Automated telemetry-to-learned-routing and native micro-scout classification remain follow-ons (see [`CLAIMS.md`](../CLAIMS.md)).

## Cross-references

- [fak vs an API gateway or LLM router](adoption/compare/vs-routers.md): how `fak` composes with OpenRouter, Portkey, LiteLLM, and Kong.
- [Routers & gateways](integrations/routers.md): the three composition topologies and surveyed-router table.
- [Claims ledger](../CLAIMS.md): shipped vs stub status across the kernel, gateway, and engine.
