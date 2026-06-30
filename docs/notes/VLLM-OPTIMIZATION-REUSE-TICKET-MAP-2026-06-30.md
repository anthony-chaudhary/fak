---
title: "Local vLLM optimization reuse map: source-backed tickets"
description: "A 2026-06-30 map from the local vLLM checkout to fak tickets, with the rule that vLLM/SOTA-native mechanisms are driven, measured, or governed before fak rebuilds them."
---

# Local vLLM optimization reuse map

Date: 2026-06-30.

Source snapshot: sibling local checkout `../vllm`, `main` at `28242824e`
(`[Bugfix][Frontend] Normalize constrained Harmony recipients (#45657)`).
This note uses the local checkout as evidence. It does not add new external
SOTA numbers; those stay in `docs/industry-scorecard/taxonomy.md` and
`docs/explainers/sota-optimizations.md`.

The rule for this backlog is simple: when vLLM already has the SOTA mechanism,
fak should first **drive it, observe it, or govern it**. Native fak versions are
only justified after the adapter/baseline ticket proves the engine path is not
enough for the agent contract.

## Ticket map

| # | vLLM pattern in the local checkout | Use in fak | Ticket |
|---|------------------------------------|------------|--------|
| 1 | Automatic prefix caching: hash-derived KV blocks, parent-prefix identity, multimodal/LoRA/cache-salt axes (`docs/design/prefix_caching.md`, `vllm/distributed/kv_events.py`) | Treat vLLM's prefix cache as an external engine cache with explicit residency, owner attribution, and no correctness trust. | #40, #1498, #1551 |
| 2 | `cache_salt` request isolation plus request `priority` (`docs/design/prefix_caching.md`, `vllm/v1/engine/input_processor.py`, `vllm/config/scheduler.py`) | Pass fak tenant/authority/cache-family into `cache_salt`, and pass advisory turn intent into vLLM priority scheduling when supported. | #1729 |
| 3 | V1 unified scheduler, chunked prefill, FCFS/priority policies (`docs/usage/v1_guide.md`, `vllm/config/scheduler.py`, `vllm/v1/core/sched/scheduler.py`) | Do not rebuild continuous batching first; feed fak intent into the scheduler seam and measure the tuned engine behavior. | #28, #805, #1729 |
| 4 | Speculative decoding families: EAGLE, MTP, n-gram, dynamic schedules (`docs/features/speculative_decoding/`, `vllm/config/speculative.py`) | Ride vLLM speculative decode for serving baselines; native fak verify/accept remains a later, witnessed path. | #23 |
| 5 | Structured outputs through xgrammar/outlines/guidance and GPU bitmasks (`docs/features/structured_outputs.md`, `vllm/v1/structured_output/backend_xgrammar.py`, `vllm/v1/worker/gpu/structured_outputs.py`) | Use vLLM for grammar-constrained bytes, but keep fak's syscall adjudication and result admission above it. | #26 |
| 6 | KV cache event stream: `BlockStored`, `BlockRemoved`, `AllBlocksCleared` (`vllm/config/kv_events.py`, `vllm/distributed/kv_events.py`) | Maintain a vLLM warm-set and route on it only after cachemeta residency and scope checks. | #40, #1498, #1551 |
| 7 | Disaggregated prefill and KV connectors: NIXL, LMCache, MultiConnector, OffloadingConnector, FlexKV (`docs/features/disagg_prefill.md`, `docs/features/nixl_connector_usage.md`) | Treat remote KV as an external lease/residency problem, not an opaque hit bit. | #29, #50, #53, #1732 |
| 8 | NIXL KV lease renewal and push-mode transfer (`docs/design/nixl_kv_cache_lease.md`, `docs/design/nixl_kv_push_connector.md`) | Fold lease create/heartbeat/expiry/completion into cachemeta, demotion, and deletion-proof surfaces. | #1732 |
| 9 | Prometheus metrics: TTFT, TPOT/ITL, queue, KV utilization, speculative/KV connector metrics (`docs/design/metrics.md`, `docs/usage/metrics.md`, `vllm/v1/metrics/`) | Normalize raw vLLM and fak-fronted-vLLM metrics into one serving schema before claiming a win. | #43, #1480 |
| 10 | Sleep/pause/wake controls and cache reset semantics (`docs/features/sleep_mode.md`, `vllm/v1/engine/core.py`, `vllm/v1/engine/llm_engine.py`) | Drive vLLM lifecycle from fak session state, and mark KV warmth cold when sleep/reset forgets KV. | #1730 |
| 11 | `torch.compile` artifact cache and CUDA graph dispatch (`docs/design/torch_compile.md`, `docs/design/debug_vllm_compile.md`, `vllm/v1/cudagraph_dispatcher.py`) | Record warmup/compile/CUDA-graph state in tuned baselines so fak is never compared against a cold vLLM. | #1731 |
| 12 | Expert Parallelism and EPLB for MoE (`docs/serving/expert_parallel_deployment.md`, `vllm/distributed/eplb/policy/default.py`) | Run vLLM EP/EPLB as the MoE serving baseline before native fak expert placement work claims parity. | #1733, #1728 |
| 13 | Batch-invariance and reproducibility modes (`docs/usage/reproducibility.md`, `docs/usage/faq.md`) | Expose determinism as an engine capability; never treat temperature 0 alone as replay proof. | #1734 |
| 14 | Cache/default-on activation and engine cache adapters | Keep the vCache item-level tickets as the implementation backlog for cache usefulness by default. | #1490, #1519-#1568 |

## New tickets filed from this pass

- #1729 -- `feat(engine/vllm): drive cache_salt and priority from fak identity and intent`
- #1730 -- `feat(lifecycle/vllm): map sleep/pause/wake into fak session state with cache-loss witnesses`
- #1731 -- `bench(vllm): record torch.compile cache and CUDA-graph warmup in tuned baselines`
- #1732 -- `feat(disagg/vllm): witness NIXL KV leases and expirations in cachemeta`
- #1733 -- `bench(vllm): run EP/EPLB MoE serving as the baseline before native expert placement`
- #1734 -- `feat(engine/vllm): surface batch-invariance and determinism as served-engine capabilities`

## Existing tickets this pass deliberately reuses

- #23 -- speculative decoding.
- #26 -- structured/guided decoding feeding tool-call gating.
- #28 -- native prefill/decode role split over the continuous-batching scheduler.
- #29 -- native KVCache-to-bytes transport.
- #40 -- vLLM V1 adapter behind `EngineDriver`.
- #43 -- TTFT/TPOT/ITL/goodput/queue/KV-util metrics surface.
- #50 -- large-scale disaggregated serving.
- #53 -- external L3 disaggregated cache as a fak-governed tier.
- #805 -- intent conduit into scheduler and cache placement.
- #1050 -- standing vLLM/SOTA scorecard freshness.
- #1490 -- vCache gates on by default plus honest per-mechanism attribution.
- #1498 -- cross-engine vBlock/anchor abstraction.
- #1551 -- vLLM prefix-cache observation adapter.

## Review rule

Reject a future ticket or PR if it starts by rebuilding a vLLM-native mechanism
without first linking the adapter/baseline ticket above. The acceptable reasons
to go native are: a witnessed correctness gap, a witnessed agent-contract gap
(scope, taint, deletion, proof), or a measured performance gap after the tuned
engine path is actually enabled.
