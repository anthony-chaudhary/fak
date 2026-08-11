---
title: "The serving side of fak — run the gateway, ride an engine, scale out KV"
description: "Home for fak's serving side: run a model behind the fak serve gateway, ride vLLM/SGLang/llm-d/Dynamo, and the scale-out plans for KV cache, poly-model, and dual-track serving."
---

# The serving side of fak

The [front page](../../README.md) leads with **`fak guard`** — wrapping the agent you
already run. This page is the other pillar: **`fak serve`**, the side of `fak` that puts a
model *behind* the kernel. Same substrate, different entry point — the gateway that
adjudicates every tool call is also the seam where reused KV cache lives, is shared, and is
scaled out across a fleet.

`fak` is **not** a faster model server — vLLM, SGLang, llm-d, and llama.cpp win raw
throughput, and you run `fak serve` *in front of* any of them. The serving side owns the
orthogonal questions a fleet hits at scale: where reused KV lives, how it is shared and
rebuilt, and which reuse is still *legal* across tenants. The reuse kernel (addressable,
bit-exact KV cache + the default-deny capability floor) is the shared substrate every track
below builds on.

## Run it — the fast path

Put `fak serve` in front of a model endpoint and point your client at it:

```bash
fak serve --addr 127.0.0.1:8080 \
  --base-url http://localhost:11434/v1 --model qwen2.5:1.5b \
  --policy examples/dev-agent-policy.json
```

OpenAI traffic goes to `http://127.0.0.1:8080/v1`, Anthropic Messages to the bare host.
Harden with `--require-key-env FAK_TOKEN`, scrape `/metrics` for live prefill vs decode
tok/s, and expose the five kernel tools to MCP hosts with
`fak serve --stdio --policy examples/dev-agent-policy.json`.

| To… | Read |
|---|---|
| Stand up a gateway step by step (Ollama, vLLM, in-kernel GGUF) | [../fak/server-quickstart.md](../fak/server-quickstart.md) |
| Every flag and env var | [../fak/server-config.md](../fak/server-config.md) |
| Every endpoint, request, and response | [../fak/api-reference.md](../fak/api-reference.md) |
| Batch multi-request inference (dynamic batch size, padding) | [../fak/batching-config.md](../fak/batching-config.md) |
| Metrics, logs, and traces | [../fak/observability.md](../fak/observability.md) |
| Production deployment (scaling, multi-region, HA) | [../fak/deployment-guide.md](../fak/deployment-guide.md) |
| When something breaks | [../fak/server-troubleshooting.md](../fak/server-troubleshooting.md) |
| The full operator & integrator index | [../fak/README.md](../fak/README.md) |

## The scale-out plans

Below are `fak`'s design and decision docs for serving at fleet scale. They are
**plans and architecture briefs**, not shipped throughput claims — each one says
explicitly what is built, what is a seam, and what is still a gap.

```text
   The 5 serving plans (design / architecture briefs)
 ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌──────────┐ ┌──────────┐
 │ Dual-track │ │ Poly-model │ │  Hardware- │ │Regener-  │ │Multi-    │
 │  serving   │ │  prefill   │ │  aware KV  │ │able KV   │ │tenant CXL│
 │ ride+native│ │  sharing   │ │   cache    │ │cache     │ │mem pool  │
 └─────┬──────┘ └─────┬──────┘ └─────┬──────┘ └────┬─────┘ └────┬─────┘
       │              │              │             │            │
       └──────┬───────┴──────┬───────┴──────┬──────┴─────┬──────┘
              ▼              ▼              ▼            ▼
   ┌──────────────────────────────────────────────────────────┐
   │              the shared reuse kernel substrate             │
   │   addressable, bit-exact KV cache  +  default-deny floor   │
   └──────────────────────────────────────────────────────────┘
```
*Five plans, one shared substrate: each track builds on the addressable,
bit-exact KV cache and the default-deny capability floor.*

| Plan | What it covers |
|---|---|
| [Dual-track serving](dual-track-serving-plan.md) | The authoritative sequencing contract: **ride** best-in-class engines (vLLM, SGLang, llm-d, Dynamo) *and* grow a **native** in-kernel engine, over one shared, track-neutral spine. |
| [llm-d integration](../integrations/llm-d.md) | First-class ride-mode support for the llm-d Kubernetes serving stack: Gateway API OpenAI route, registered `llm-d` engine id, and honest vLLM-worker metrics/KV boundaries. |
| [Dynamo interop](dynamo-interop.md) | Issue #38 decision: fak governs in front of Dynamo's public frontend, Dynamo keeps ownership of P/D routing, and fak normalizes Dynamo role/load/KV signals into `fak_serving_*`. |
| [vLLM V1 adapter](vllm-v1-adapter.md) | Issue #40 driver doc: the registered `vllm` lifecycle adapter rides a vLLM V1 worker's OpenAI HTTP, KV-cache-events, and Prometheus surfaces, and states the honesty boundary where exact-span KV governance degrades to whole-prefix reset. |
| [vLLM internals study](vllm-internals-study.md) | Source-verified background for the adapter: an exhaustive walk through vLLM's V1 internals — PagedAttention & the block manager, the unified token-budget scheduler, automatic prefix caching, attention backends (FA3/FlashInfer/Triton/MLA), TP/PP/DP/EP/EPLB, P/D disaggregation & KV transport, speculative decoding, and quantization — with a section-by-section map of where fak's value-aware KV kernel diverges from vLLM's LRU block pool. |
| [Poly-model prefill sharing](polymodel-prefill-share-plan.md) | Host tens of models in one kernel, share prefill across them, decode one at a time, and put idle models to work on speculative decoding. |
| [Hardware-aware KV cache](hardware-aware-cache.md) | Plan where a KV span lives across HBM, DRAM, NUMA-far, CXL, disk, and remote tiers — per-tier TTL and demote-not-evict placement. |
| [Regenerable KV cache](regenerable-kv-plan.md) | Treat the KV cache as a build artifact rebuilt from durable transcript text, so a model rollout becomes a backfill instead of a cold start. |
| [Multi-tenant CXL memory pool](cxl-memory-pool.md) | Price CXL.mem pooled KV reuse across a fleet and gate cross-tenant cell reuse, failing closed on poisoned or wrong-model cells. |
| [vCache scorecard playbook](vcache-scorecard-playbook.md) | Run `fak vcache score` to read the 2x agent-dev gate, build the hot-anchor index it plans, and move a workload from planned savings to provider-telemetry-proven savings. |
| [P/D + KV-routing SOTA matrix](pd-disaggregation-kv-routing-sota.md) | The ride-vs-own decision matrix comparing vLLM, SGLang, LMCache, Dynamo, Mooncake, and current fak across prefix cache, P/D split, KV transfer, routing, autoscaling, metrics, and invalidation — plus the source-tagged `CacheEvent`/`ServingEvent` vocabulary (#903). |
| [Multi-node compute](multi-node-compute.md) | The runnable witness: `fak cluster` runs a real cross-node collective over the `DistComm` process group on any two CPU hosts today, plus the rung ladder from that host-layer floor to GPU-speed multi-node serving (#652, #639, #85, #30, #29, #25). |
| [Native device mesh and collectives](native-device-mesh-collectives.md) | The R3+ design gate for native DP x TP x PP x EP: process groups, rank/world-size, `compute.CollectiveBackend` primitives, CPU-ref single-rank behavior, and the TP -> EP-for-MoE dependency chain (#25). |
| [Heterogeneous silicon fleets](heterogeneous-silicon-fleet.md) | Public reference architecture for one agent-kernel control plane over `cpu-ref`, CUDA, Vulkan, and vendor backend groups, with route evidence and honest gaps. |

## How these relate

The plans above cover the reuse questions, not the throughput race. You can run `fak serve`
in front of vLLM, SGLang, llm-d, or llama.cpp and keep their raw speed while the kernel
governs reuse, routing, and the capability floor. See the
[addressable KV cache](../explainers/addressable-kv-cache.md) explainer for the core idea,
and [what fak is not](../explainers/what-fak-is-not.md) for the honest fence.

## Epics in flight

- [Multilevel default cache epic](multilevel-default-cache-epic.md) — the progress spine
  that finishes the hardware-capacity bridge (#706): wire the demote-not-evict executor into
  a live loop, derive real pressure for every local tier (HBM/DRAM/disk), and make
  hardware-aware placement the kernel's default. Each rung is a `dos`-verifiable
  prove-or-refute step; builds on the [hardware-aware KV cache](hardware-aware-cache.md) plane.

## Witnesses

- [Token-saving observability](token-savings-observability.md) — live active/ready/bypassed receipts, measured effects, and rollback controls for the default-on saving stack.

- [GLM-5.2 full-size serving witness](glm52-full-size-serving-witness.md) — the
  reproducible runbook behind issue #413: standing up a full-size GLM-5.2 serve and
  capturing the evidence that it ran.
</content>
</invoke>
