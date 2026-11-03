# Runtime concepts: pools, sessions, and cache

These distinctions describe existing source behavior; they do not assert that a particular entrypoint invokes it.

## KV pool and capacity

`KVPoolConfig` specifies paged KV geometry, `NewKVPool` constructs a pool, and `KVPool` owns sequence mappings and physical-block allocation. `SharedTokenPool` accounts reserved and committed token capacity across streams; it is not the block mapping. `EvictLRU` reclaims physical KVPool blocks, while `trimBackendIdlePools` asks the planner backend to release idle allocations before retry.

Sources: [`kv_pool.go`](../../internal/ctxmmu/kv_pool.go), [`allocator.go`](../../internal/ctxmmu/allocator.go), [`inkernel_planner.go`](../../internal/agent/inkernel_planner.go), [`inkernel_oom_retention.go`](../../internal/agent/inkernel_oom_retention.go).

## Session branches and soft slots

`ForkSession` creates a copy-on-write `SessionBranch`, identified by a new `childSessionID`; `ErrSessionReleased` signals a released branch. `LoadSessionCheckpoint` reads persisted agent checkpoint data instead of creating an in-memory branch.

The HTTP planner's `LlamaSoftSlot` enables optional per-conversation slot selection, while `LlamaSoftSlotPolicy` chooses the policy. `SoftSlotPolicyLRU` moves a conversation from a busy slot to an idle least-recently-used slot; `SoftSlotPolicyWait` can retain the known busy slot. `withLlamaSoftSlot` performs selection and may add `id_slot` and return a lease.

Sources: [`cow_block.go`](../../internal/ctxmmu/cow_block.go), [`cow_table.go`](../../internal/ctxmmu/cow_table.go), [`session_checkpoint.go`](../../internal/agent/session_checkpoint.go), [`chat.go`](../../internal/agent/chat.go), [`llama_soft_slot.go`](../../internal/agent/llama_soft_slot.go).

## Cache and eviction

The in-kernel planner's `cachePopulate` marks a cache-population execution, and `cachedPrefixLen` reads the number of resident leading tokens. Agent completion `CacheHint` records provider-cache hint negotiation. Elastic allocator `cachedBlocks` holds FIFO reuse order for released blocks whose contents remain cached. The expert-cache `cachePolicyReplayArm` simulates one policy over an access stream.

The radix KV tree's `evictLedger` stores pending eviction records, bounded by `evictionLedgerCap`, for later confirmation.

Sources: [`inkernel_planner.go`](../../internal/agent/inkernel_planner.go), [`inkernel_eviction.go`](../../internal/agent/inkernel_eviction.go), [`chat.go`](../../internal/agent/chat.go), [`elastic_block_allocator.go`](../../internal/modelengine/elastic_block_allocator.go), [`cache_policy_bench.go`](../../internal/deepseekv4moe/cache_policy_bench.go), [`radixkv/plan.go`](../../internal/radixkv/plan.go), [`radixkv.go`](../../internal/radixkv/radixkv.go).

## Gateway and prefix session helpers

`sessionFromPrefixClone` creates a model session and assigns the supplied KV cache. `inKernelRefeedLastTokenForExactHit` removes the final cached prompt token when eligible for refeeding. Gateway `sessionView` snapshots the terminal transcript, sorted effects, and recovery dependency, while `newEngineCacheClient` configures an optional remote reset client.

Sources: [`inkernel_decode_snapshot.go`](../../internal/agent/inkernel_decode_snapshot.go), [`session_client.go`](../../internal/gateway/session_client.go), [`gateway_planners.go`](../../internal/gateway/gateway_planners.go).

## Context controls

Agent run `contextControl` requests a tool-catalog addition; `ContextControlCatalog` returns definitions only while armed. Superstream `EvaluateContextSafety` assesses usage and returns a `ContextSafetyVerdict` with status, remaining budgets, and reset recommendation. MCP broker `WithCompressionContext` supplies the context for compression caller-policy resolution.

Sources: [`loop_wire.go`](../../internal/agent/loop_wire.go), [`context_control.go`](../../internal/agent/context_control.go), [`context_safety.go`](../../internal/superstream/context_safety.go), [`structured_compression.go`](../../internal/mcpbroker/structured_compression.go).

# Runtime next cohort: source map

- COW branch and fork: [internal/ctxmmu/cow_block.go](../../internal/ctxmmu/cow_block.go), [internal/ctxmmu/cow_table.go](../../internal/ctxmmu/cow_table.go); contrast execution-route `SessionFork` in [internal/executionroute/executionroute.go](../../internal/executionroute/executionroute.go).
- Provider hint and telemetry: [internal/agent/chat.go](../../internal/agent/chat.go), [internal/agent/chat_provider_cache.go](../../internal/agent/chat_provider_cache.go); contrast vDSO `CacheHit` in its existing scorecard entry. Prefix owner binding: [internal/agent/prefix_cache_identity.go](../../internal/agent/prefix_cache_identity.go).
- Expert slot acquisition and stream: [internal/compute/slotstream/slotstream.go](../../internal/compute/slotstream/slotstream.go); upstream soft-slot release: [internal/agent/stream.go](../../internal/agent/stream.go); active subagent slots: [internal/modelengine/subagent_sched.go](../../internal/modelengine/subagent_sched.go).
- Staged radix eviction: [internal/radixkv/plan.go](../../internal/radixkv/plan.go); WTinyLFU callback: [internal/l3server/eviction/wtinylfu.go](../../internal/l3server/eviction/wtinylfu.go).
- Gateway reset transaction, budget and omitted span: [internal/gateway/server_state.go](../../internal/gateway/server_state.go). Launch ID: [cmd/fak/launch_session.go](../../cmd/fak/launch_session.go); host label: [cmd/fak/session_durable.go](../../cmd/fak/session_durable.go); default checkpoint directory: [internal/agent/session_checkpoint.go](../../internal/agent/session_checkpoint.go); checkpoint versus COW errors: [internal/ctxmmu/checkpoint.go](../../internal/ctxmmu/checkpoint.go), [internal/ctxmmu/fork.go](../../internal/ctxmmu/fork.go).
- Radix CPU retention accounting and admission: [internal/radixkv/host_cache_budget.go](../../internal/radixkv/host_cache_budget.go).
- Context control tool name: [internal/agent/context_control.go](../../internal/agent/context_control.go); allow expiry/subtraction: [cmd/fak/guard_allow.go](../../cmd/fak/guard_allow.go); guard router target: [cmd/fak/guard_router.go](../../cmd/fak/guard_router.go); API readiness gate: [internal/apihostprobe/apihostprobe.go](../../internal/apihostprobe/apihostprobe.go); raw launcher flag: [cmd/fak/codex_launcher.go](../../cmd/fak/codex_launcher.go); native enforcement posture: [cmd/fak/chat.go](../../cmd/fak/chat.go), [cmd/fak/agent_native.go](../../cmd/fak/agent_native.go).

# Third runtime cohort source map

- Generation result and device headroom: [internal/agent/inkernel_planner.go](../../internal/agent/inkernel_planner.go); tool choice: [internal/agent/inkernel_render.go](../../internal/agent/inkernel_render.go).
- Radix startup claims: [internal/radixkv/startup_claim.go](../../internal/radixkv/startup_claim.go); model precision constructor: [internal/model/kvcache.go](../../internal/model/kvcache.go); eviction hints: [internal/compute/kvcost.go](../../internal/compute/kvcost.go); expert source enablement: [internal/model/expert_pagecache.go](../../internal/model/expert_pagecache.go).
- Paged prefix opt-out: [internal/agent/inkernel_paged_prefix.go](../../internal/agent/inkernel_paged_prefix.go); radix budget and sampling seed: [internal/agent/inkernel_planner_config.go](../../internal/agent/inkernel_planner_config.go); expert page-cache opt-in: [internal/ggufload/expert_pagecache.go](../../internal/ggufload/expert_pagecache.go).
- Context control arming: [internal/agent/context_control.go](../../internal/agent/context_control.go); warmup gate: [internal/gateway/readiness_warmup.go](../../internal/gateway/readiness_warmup.go); COW branch lookup: [internal/ctxmmu/cow_table.go](../../internal/ctxmmu/cow_table.go); request credentials: [internal/gateway/http.go](../../internal/gateway/http.go).
- COW branch presence and registration: [internal/ctxmmu/cow_table.go](../../internal/ctxmmu/cow_table.go); stream lift guard: [internal/gateway/stream_lift_guard.go](../../internal/gateway/stream_lift_guard.go); latest session context: [internal/session/costring.go](../../internal/session/costring.go).
- Gemma cached continuation and refusal: [internal/model/gemma4_session.go](../../internal/model/gemma4_session.go); session resident kernel: [internal/model/kernel.go](../../internal/model/kernel.go); native forward trace kind: [internal/agent/chat.go](../../internal/agent/chat.go); MTP target update: [internal/model/metal_mtp.go](../../internal/model/metal_mtp.go); cancellable Q4K loader: [internal/ggufload/quant_q4k_loader.go](../../internal/ggufload/quant_q4k_loader.go).
- V4.1 Engram options, replacement policy, and constructor: [internal/model/v41_cache.go](../../internal/model/v41_cache.go); layer cache lookup/reset: [internal/model/v41_forward_layer_cache.go](../../internal/model/v41_forward_layer_cache.go); layer expert cache budget: [internal/model/v41_forward_weights.go](../../internal/model/v41_forward_weights.go).
- Warmup readiness state: [internal/gateway/readiness_warmup.go](../../internal/gateway/readiness_warmup.go); checkpoint run option: [internal/agent/loop_session.go](../../internal/agent/loop_session.go); turnkey prefix owner: [cmd/fak/up_agent_warm.go](../../cmd/fak/up_agent_warm.go); llama observer slots override: [internal/engine/llama.go](../../internal/engine/llama.go).
- Backend teardown status: [internal/model/qwen35_hal.go](../../internal/model/qwen35_hal.go); speculative verification: [internal/model/speculative.go](../../internal/model/speculative.go); engine topology label: [internal/allinone/types.go](../../internal/allinone/types.go); scheduler lane cleanup: [internal/modelengine/nativesched_prefill.go](../../internal/modelengine/nativesched_prefill.go).
- V4.1 Engram gate function: [internal/model/v41_engram_gate_device.go](../../internal/model/v41_engram_gate_device.go); gate stream count: [internal/compute/v41_engram_gate.go](../../internal/compute/v41_engram_gate.go); Qwen query-gate split interface: [internal/model/qwen35_hal.go](../../internal/model/qwen35_hal.go); guard posture environment: [cmd/fak/guard.go](../../cmd/fak/guard.go); MCP tool prefix: [internal/agent/mcptools.go](../../internal/agent/mcptools.go); capabilities recognition: [internal/gateway/tool_routing.go](../../internal/gateway/tool_routing.go).

# Fourth runtime cohort source map

- Batched causal attention contraction: [internal/compute/prefill_batch.go](../../internal/compute/prefill_batch.go); CUDA panel validation: [internal/compute/cuda_kernels.go](../../internal/compute/cuda_kernels.go).
- V4.1 compressed attention selection: [internal/model/v41_attention.go](../../internal/model/v41_attention.go); selected shared invocation: [internal/model/v41_shared_attention_device.go](../../internal/model/v41_shared_attention_device.go); shared input validation: [internal/compute/v41_shared_attention.go](../../internal/compute/v41_shared_attention.go); CUDA candidate build tag: [internal/compute/cuda_v41_shared_attention.go](../../internal/compute/cuda_v41_shared_attention.go).
- Native linear-attention load option: [internal/ggufload/quant_q4k_loader.go](../../internal/ggufload/quant_q4k_loader.go); V4.1 state construction: [internal/model/v41_attention_state.go](../../internal/model/v41_attention_state.go); checkpoint descriptor diagnostics: [internal/ggufload/deepseek41_attention_preflight.go](../../internal/ggufload/deepseek41_attention_preflight.go).
- Metal graph result and four attention entrypoints: [internal/metalgemm/graph.go](../../internal/metalgemm/graph.go), [internal/metalgemm/qwen35_graph.m](../../internal/metalgemm/qwen35_graph.m); ratio-zero sink weights: [internal/model/v41/v4_flash_attn_ratio0.go](../../internal/model/v41/v4_flash_attn_ratio0.go); tree-mask attention: [internal/compute/tree_attention.go](../../internal/compute/tree_attention.go).
- Execution policy context and ABI posture: [internal/abi/policy_context.go](../../internal/abi/policy_context.go); compression policy binding: [internal/mcpbroker/structured_compression.go](../../internal/mcpbroker/structured_compression.go); sequenced gateway adjudication: [internal/gateway/tool_routing.go](../../internal/gateway/tool_routing.go); agent posture setter: [internal/agent/tools.go](../../internal/agent/tools.go).
- Gateway soft-slot policy: [internal/gateway/server_state.go](../../internal/gateway/server_state.go), [internal/gateway/chat_route.go](../../internal/gateway/chat_route.go); feature proof verdict: [internal/gateway/feature_proof.go](../../internal/gateway/feature_proof.go); operator approval marker: [internal/adjudicator/operator_approval.go](../../internal/adjudicator/operator_approval.go); routing refusal inputs: [internal/agentopt/intra_model_router.go](../../internal/agentopt/intra_model_router.go).
- GGUF projection and key-normalization tensor names: [internal/ggufload/deepseek41.go](../../internal/ggufload/deepseek41.go), [internal/ggufload/gguf_glm_tensors.go](../../internal/ggufload/gguf_glm_tensors.go), [internal/ggufload/gguf_transform_contract.go](../../internal/ggufload/gguf_transform_contract.go); unsupported Q4K estimate sentinel: [internal/ggufload/estimate.go](../../internal/ggufload/estimate.go).
- Metal serve availability decision and resolver: [cmd/fak/serve_model_load_metaldecision.go](../../cmd/fak/serve_model_load_metaldecision.go); streamed CPU-offload choice: [cmd/fak/serve_model_fit_streamed.go](../../cmd/fak/serve_model_fit_streamed.go).
- In-kernel V4.1 chat segment rendering: [internal/agent/inkernel_render_deepseek41.go](../../internal/agent/inkernel_render_deepseek41.go); upstream template-render failure code/message: [internal/gateway/upstream_template_render.go](../../internal/gateway/upstream_template_render.go); Qwen MTP retained tensors validation: [internal/ggufload/gguf_weightsource.go](../../internal/ggufload/gguf_weightsource.go).
- Session lifecycle view: [internal/sessionview/view.go](../../internal/sessionview/view.go); surrender detection: [internal/stopgate/boundary.go](../../internal/stopgate/boundary.go).
- Scoped tool search cache: [internal/vdso/search_cache.go](../../internal/vdso/search_cache.go).
- Agent final completion gate: [internal/agent/loop_session.go](../../internal/agent/loop_session.go); session process snapshot: [cmd/fak/session_query.go](../../cmd/fak/session_query.go).
- Harness worker pool interface: [pkg/harnesskit/coordination.go](../../pkg/harnesskit/coordination.go); dispatcher proxy clients: [internal/l3server/transport/dispatch/dispatch.go](../../internal/l3server/transport/dispatch/dispatch.go), [internal/l3server/transport/dispatch/cluster.go](../../internal/l3server/transport/dispatch/cluster.go); message buffer pool index: [internal/l3server/transport/protocol/protocol.go](../../internal/l3server/transport/protocol/protocol.go); observer stopped sentinel: [internal/observer/verdict.go](../../internal/observer/verdict.go).
- Info gateway URL default: [cmd/fak/info.go](../../cmd/fak/info.go); Vulkan staging pool geometry: [internal/compute/vulkan_graph.go](../../internal/compute/vulkan_graph.go).
