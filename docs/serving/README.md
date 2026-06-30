---
title: "fak serving plans: scale-out KV cache, poly-model, dual-track"
description: "Index of fak's large-scale serving design docs: dual-track serving, poly-model prefill sharing, hardware-aware and regenerable KV cache, and CXL memory pooling."
---

# fak serving plans — the scale-out roadmap

These are fak's design and decision docs for serving at fleet scale. They are
**plans and architecture briefs**, not shipped throughput claims — each one says
explicitly what is built, what is a seam, and what is still a gap. The reuse kernel
(addressable, bit-exact KV cache + the default-deny capability floor) is the shared
substrate every track below builds on.

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

`fak` is **not** a faster model server — vLLM, SGLang, llm-d, and llama.cpp win raw
throughput, and you can run `fak serve` in front of any of them. These plans cover the
orthogonal questions a fleet hits at scale: where reused KV lives, how it is shared and
rebuilt, and which reuse is still *legal* across tenants. See the
[addressable KV cache](../explainers/addressable-kv-cache.md) explainer for the core idea.

## Epics in flight

- [Multilevel default cache epic](multilevel-default-cache-epic.md) — the progress spine
  that finishes the hardware-capacity bridge (#706): wire the demote-not-evict executor into
  a live loop, derive real pressure for every local tier (HBM/DRAM/disk), and make
  hardware-aware placement the kernel's default. Each rung is a `dos`-verifiable
  prove-or-refute step; builds on the [hardware-aware KV cache](hardware-aware-cache.md) plane.

## Witnesses

- [GLM-5.2 full-size serving witness](glm52-full-size-serving-witness.md) — the
  reproducible runbook behind issue #413: standing up a full-size GLM-5.2 serve and
  capturing the evidence that it ran.
