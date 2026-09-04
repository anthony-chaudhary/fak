---
title: "fak dual-track serving: RIDE + NATIVE over one shared spine"
description: "fak's authoritative dual-track serving plan: ride best-in-class engines (vLLM, SGLang, llm-d) plus a native in-kernel engine, over one shared track-neutral spine."
---

# Dual-track serving plan — RIDE + NATIVE over one shared spine

Dual-track serving is fak's plan to reach large-scale disaggregated inference along two first-class tracks over one shared, track-neutral spine: Track A *rides* best-in-class engines (vLLM, SGLang, llm-d, Dynamo, NIXL) as a governance and trust control plane, while Track B grows fak's own in-kernel engine for the single-node to small-cluster path and its bit-exact KV value-add. This page is the authoritative decision and sequencing contract for that epic — design and scope only; no scheduler, router, transport, or kernel code lands here. It commits fak to both tracks, says exactly where each stands against the live tree today, and asserts no benchmark number — every parity claim is gated on a measured run by the bench-harness sibling.

> **Authoritative decision doc** for the large-scale disaggregated-serving epic
> (#50). This is **not** a
> ride-vs-native decision — the operator has confirmed fak commits to **BOTH** tracks as
> first-class deliverables. This doc is the honest *sequencing + scope + de-dup* contract
> both tracks share, so no downstream issue silently overclaims.
>
> **Scope:** design/decision only. No scheduler, router, transport, or kernel code lands
> here — only the plan, the capability-honesty table, and the de-dup map.
>
> **Provenance:** every `[SHIPPED]/[PARTIAL]/[SEAM-ONLY]/[GAP]` mark below carries a
> `file:line` pointer re-verified against the working tree at commit `32586e6bc` (2026-07-20)
> by re-running the [§11](#11-how-to-re-verify) anchor sweep (#2728). The prior stamp was
> `89abc5d` (2026-06-22); that sweep had since **broken** — `KVCache.Evict` and `StepBatch`
> had both moved file, and the "expect: no real substrate" collective anchor had started
> matching a real NCCL process group. Those are corrected below.
> Line numbers drift; re-verify with the `rg` anchors in [§11](#11-how-to-re-verify). **No
> benchmark number is asserted in this doc** — every parity claim is gated on a *measured*
> run by the bench-harness sibling (#44),
> and where a comparison is unmeasured this doc says so explicitly.

## 1. TL;DR — the decision

1. **Two tracks, both first-class, one shared spine.** Track A *rides* best-in-class engines
   (vLLM V1 / SGLang / Dynamo / NIXL); Track B grows fak's *own* in-kernel disaggregated
   engine. They reuse one spine: streaming/detokenizer, the `EngineDriver` admit/step/stream
   seam, the fleet router + residency index, the membership/health loop, the serving-metrics
   surface, and the parity bench. **The spine is built once, track-neutral.**
2. **Ride-first sequencing.** Track A is the resourced near-term path to many-node *parity*
   because Track B's fleet-scale engine sits on a **greenfield collective-comms substrate**
   (no NCCL communicator, no world-size, no device-mesh — [§7](#7-track-b-fleet-scale-is-not-co-equal-near-term)).
   Building Track A *forces* the shared spine that Track B then reuses.
3. **fak's earned lead is the reason Track B exists:** bit-exact middle-span KV `Evict`
   (`internal/model/kvcache.go:94`) — quarantine eviction the SOTA engines structurally **cannot**
   do. That value-add **rides on top of** the base items; it is out of scope for base-item
   parity ([§9](#9-non-goals)).
4. **Track B fleet-scale is NOT co-equal near-term.** Native is co-equal only at the
   single-node → small-cluster scope; fleet-scale native is sequenced behind the spine and
   the collective-comms layer, and scope-gated.

## 2. The two tracks

- **Track A — RIDE / integrate.** fak as the disaggregation-aware control plane + trust /
  reuse / governance layer **on top of** vLLM V1 / SGLang / NVIDIA Dynamo / NIXL workers
  across many GPU nodes. Reaches parity by *orchestrating* engines that already ship P/D
  disaggregation, KV transport, and TP/PP — and adding fak's value (taint, lease, admission
  verdict, attestable KV movement). Fastest path; inherits their compute parity.
- **Track B — NATIVE / "our own ideal".** fak's own in-kernel disaggregated engine grows the
  base items natively, pursued **only where bit-exact kernel-owned KV + the trust layer make
  native strictly better** — the single-node → small-cluster path and the value-add unlock.
  **Explicitly not** to beat vLLM/SGLang/llama.cpp on raw single-GPU tokens/sec.

The real design question is **sequencing and de-dup, not track selection.**

## 3. Ride-first sequencing — and why

Native multi-node parallelism is **no longer wholly greenfield, but it is build-gated.** The
2026-06-22 stamp of this section claimed an `rg` sweep found **no** `ncclCommInitRank` /
`world_size` / `DeviceMesh` anywhere in the tree. Re-running that sweep at `32586e6bc`
**falsifies** it: `internal/compute/cuda_nccl_pg.cu` is a real multi-process NCCL
process-group bootstrap (`ncclGetUniqueId` → `ncclCommInitRank` → `ncclAllReduce`), bridged
to the `model.Collective` seam by `internal/compute/cuda_collective_pg.go`, and the CUDA
backend now exposes per-rank device binding (`fcuda_set_device(int device)`
`internal/compute/cuda_kernels.cu:233`, `fcuda_malloc_on` `:242`) rather than only the
hardcoded `cudaSetDevice(0)` probe at `:63`.

What that does **not** yet earn is a parity claim. Both files are compiled only under
`-tags cuda,nccl` with `FAK_CUDA_NCCL=1`; the default `go build ./cmd/fak` and even a plain
`-tags cuda` build link none of it, and both carry an explicit self-declared
`STATUS: unverified on a GPU-free host`. So the substrate is **[PARTIAL]** — real code behind
an opt-in build gate, never yet witnessed on this host — not the **[GAP]** this section used
to assert, and not **[SHIPPED]**. The single-box **simulation** + swap-in seam remain what the
default binary actually runs (`model.Collective` / `LocalCollective`
`internal/model/tensor_parallel.go:140`, `compute.CollectiveBackend`
`internal/compute/compute.go:347` with a CPU-reference impl). Expert
parallelism still fails closed (`ForwardTP` errors on MoE, `internal/model/tensor_parallel_forward.go:82`) —
that row re-verified exact and unchanged.

So Track B's fleet-scale engine **cannot exist** until that communicator/world-size/device-mesh
layer is built. Track A reaches many-node *parity* first by orchestrating engines that already
have it — and the act of building Track A *forces* the shared spine (router, residency index,
metrics, `EngineDriver`, bench harness) that Track B then reuses. **Ride-first, but the spine
is built track-neutral from day one.**

## 4. The shared spine (built once, both tracks plug in)

| Layer | Spine deliverable | Issue | Track |
|---|---|---|---|
| L0 | Incremental detokenizer (streaming/token-timing prerequisite) | #48 | shared |
| L0 | Real token-by-token streaming reconciled with whole-turn tool-call gating | #47 | shared |
| L0 | `EngineDriver` extended from one-shot `Complete` → admit/step/stream/cancel — designed against BOTH consumers | #46 | shared |
| L1 | Fleet router skeleton — N-upstream dispatch + static replica registry | #45 | shared |
| L1 | Per-worker prefix-residency index + cache-aware power-of-two routing | #41 | shared |
| L1 | Node membership / health / drain / failover loop feeding the router | #42 | shared |
| L2 | TTFT/TPOT/ITL/goodput/queue/KV-util metrics surface — two emitters, one schema | #43 | shared |
| L4 | Parity bench harness (vLLM/SGLang/native) — gates every parity claim with a MEASURED number | #44 | shared |

The `EngineDriver` seam (#46) is the
spine's keystone: today `abi.EngineDriver` (`internal/abi/registry.go:568`) is exactly
`Complete(ctx, *ToolCall) (*Result, error)` + `Caps()` — one-shot, no admit/step/stream/cancel.
Getting its shape right *before* either consumer lands is the #1 risk; it must be reviewed
against the native-scheduler shape, not just the adapter shape.

## 5. Capability honesty table

Every serving base item, marked against the live tree with the real vLLM-V1 / SGLang
equivalent (never a strawman). Legend: **[SHIPPED]** real & proven · **[PARTIAL]** real but
incomplete · **[SEAM-ONLY]** the interface/seam exists, no production impl behind it ·
**[GAP]** absent.

| Base item | fak today | Anchor (`@89abc5d`) | SOTA equivalent (the parity target) |
|---|---|---|---|
| **Gateway topology** | **[PARTIAL]** static multi-upstream proxy: `ReplicaRouter` can round-robin a configured `BaseURL` + `ReplicaBaseURLs` set through the live `agent.Planner` seam, but there is no residency index, health/drain loop, queueing, or cache-aware placement yet | `internal/gateway/replica_router.go`; `internal/gateway/gateway.go` (`ReplicaBaseURLs`); `cmd/fak/serve.go` (`--replica-base-url`) | SGLang Router / vLLM router / LiteLLM front N replicas with health, load, and KV-locality routing |
| **Engine seam** | **[PARTIAL]** `agent.StreamingPlanner` adds a content callback for streaming-capable HTTP planners, but the native in-kernel engine still exposes a one-shot `Complete`; `ctx` is not yet a per-step cancel/control point inside decode | `internal/agent/stream.go` (`StreamingPlanner`, `CompleteStream`); `internal/modelengine/modelengine.go:139` (`Complete`), `:151` (`sess.Generate` one-shot) | vLLM-V1 `EngineCore` admit→per-step-decode→stream→reclaim lifecycle |
| **Streaming** | **[PARTIAL]** live prose deltas now stream on the OpenAI wire and on Anthropic `/v1/messages` when backed by Anthropic passthrough or a generic streaming planner; tool-call bytes are still held until whole-turn adjudication; non-streaming planners still synthesize SSE post-turn | `internal/gateway/stream_proxy.go`; `internal/gateway/messages_stream_passthrough.go`; `internal/gateway/messages_stream_planner.go` | vLLM-V1 / SGLang flush each decoded token live → real-TTFT SSE, inter-token gaps == TPOT |
| **Incremental detokenizer** | **[GAP]** whole reply detokenized once; no streaming detokenizer | — (prerequisite, #48) | streaming detokenizer feeding per-token SSE |
| **Continuous-batching scheduler** | **[SEAM-ONLY]** `StepBatch` per-step primitive exists; **no** admit/evict loop. `GenerateBatch` runs static fixed-B and re-feeds EOS into finished slots | `internal/model/batch_step.go:7` (`StepBatch`), `:142` (`GenerateBatch`) | vLLM-V1 `EngineCore` / SGLang `Scheduler` iteration loop: admit, retire, rebuild running batch each step |
| **Chunked prefill** | **[PARTIAL]** rectangular equal-length panel ≤512 tokens; ragged batches fall back to serial prefill | `internal/model/batch_prefill.go:50` (`rectangularPrefillLen`); `internal/model/batch.go:46` (`batchRectPrefillMaxTokens=512`) | chunked prefill (vLLM-V1 default) packs different requests' prefill chunks + decode into one ragged varlen batch |
| **Request admission / priority** | **[GAP]** "budget" == matmul worker count (thread-pool width), not request admission | `internal/model/budget.go:50` (`SetWorkerBudget`) | waiting queue + FCFS/priority policy + KV/token budget + preemption + `max-num-seqs` |
| **Paged / block KV** | **[GAP]** contiguous-append flat `[]float32` per layer; `Evict` memmoves to compact | `internal/model/kvcache.go:11` (`KVCache`) | vLLM PagedAttention `BlockManager` / SGLang token-to-KV-pool: fixed blocks + block table, O(1) evict, COW prefix share |
| **Radix prefix cache** | **[SHIPPED]** real RadixAttention trie (longest-prefix walk, edge split, LRU, ref-count leases) — but **single-process, in-memory** | `internal/radixkv/radixkv.go:72` (`Tree`) | SGLang RadixAttention (same algo) + a *distributed* residency index (Mooncake / LMCache) sharded across replicas |
| **Bit-exact middle-span Evict** | **[SHIPPED]** single-rotation re-RoPE from stored pre-RoPE `Kraw` → cache byte-identical to one that never saw the span | `internal/model/kvcache.go:94` (`Evict`) | **None.** vLLM `reset_prefix_cache` / SGLang `flush_cache` drop whole prefixes/LRU leaves only — a genuine structural value-add |
| **Exact-span on a ridden engine** | **[PARTIAL]** degrades to whole-prefix flush — `SupportsExactSpan` is false for SGLang, vLLM **and** MLX (#2724 kept the same fence) | `internal/enginecache/enginecache.go:272` (`SupportsExactSpan`), `:288` (`/flush_cache`), `:296` (`/reset_prefix_cache`); `internal/engine/mlx.go:97` (`engine.cache.whole-prefix`) | engines expose only coarse whole-prefix reset over HTTP; none can evict a named middle span |
| **KV transport / P/D data plane** | **[PARTIAL]** metadata descriptor + governance contract documented; `BytesMoved` is a reported counter, **no path copies KV bytes** | `internal/cachemeta/kvtransfer.go:54` (`FromKVTransfer`), `docs/serving/kv-transport-governance-nixl-mooncake-lmcache.md` (governance contract) | vLLM `KVConnector` (NIXL/LMCache), SGLang + Mooncake: RDMA/NVLink byte movers on the wire |
| **Pipeline parallelism** | codec **[SHIPPED]** / transport **[SEAM-ONLY]**: real bit-exact hidden-state codec + a `TCPTransport`, but its only peer is `EchoFrames` (identity echo), no band-running worker | `internal/model/pipeline.go:159` (`MarshalHidden`); `internal/model/pipeline_transport.go:30` (`TCPTransport`), `:105` (`EchoFrames`) | per-rank worker process that runs its band on GPU and forwards to the next rank over NCCL P2P |
| **Tensor parallelism** | **[PARTIAL]** (was [SEAM-ONLY]/[GAP] @`89abc5d`): single-box host-array **simulation** + swap-in seam is what the default binary runs, but a real multi-process NCCL communicator + per-rank device binding now exist **behind an opt-in build gate** (`-tags cuda,nccl`, `FAK_CUDA_NCCL=1`), self-declared `unverified on a GPU-free host`. Not witnessed here — no parity claim | `internal/model/tensor_parallel.go:140` (`Collective`/`LocalCollective`); `internal/compute/compute.go:431` (`CollectiveBackend`, cpu-ref only); `internal/compute/cuda_nccl_pg.cu` (`ncclCommInitRank`), `internal/compute/cuda_collective_pg.go` (`//go:build cuda && nccl`); `internal/compute/cuda_kernels.cu:233` (`fcuda_set_device`), `:63` (`cudaSetDevice(0)` probe) | vLLM/SGLang: `init_distributed_environment` builds TP/PP/EP groups over a NCCL world (world_size/rank/local_rank) + custom all-reduce |
| **Expert parallelism (MoE)** | **[GAP]** `ForwardTP` fails closed on MoE | `internal/model/tensor_parallel_forward.go:82` | all-to-all expert dispatch (e.g. DeepEP) |
| **Serving metrics** | **[SHIPPED]** normalized `fak_serving_*` Prometheus schema covers TTFT, TPOT, ITL, goodput, running/waiting queue depth, KV-cache utilization, and prefix-cache hit rate with `worker`/`engine`/`model` labels. Track A has a scrape emitter that relabels vLLM/SGLang rows into the schema, including current vLLM `kv_cache_usage_perc` and prefix hit/query counters; Track B has the native emitter seam and the gateway's measured inference/admission signals feed the same row shape. | `internal/gateway/serving_metrics.go`; `internal/gateway/serving_metrics_test.go` | `vllm:time_to_first_token_seconds`, `vllm:time_per_output_token_seconds`, `vllm:num_requests_running/waiting`, `vllm:kv_cache_usage_perc`, `vllm:prefix_cache_hits/queries` |
| **MLX ride adapter (Apple-Silicon)** | **[SHIPPED] as a ride adapter** (#2724): `fak serve --engine mlx` fronts mlx-lm / vllm-mlx over the same OpenAI-compatible dispatch as the vLLM/SGLang/Dynamo adapters, with `ParseMLXPrometheus` re-tagging `vllm:*` rows under `engine="mlx"` and an empty-surface-fabricates-nothing fence. **No in-kernel MLX/Metal implementation**; external reference comparator evaluated against fak-native and llama.cpp in the head-to-head bench (#2723 / [`docs/notes/MAC-THREEWAY-BENCH-2026-09-03.md`](../notes/MAC-THREEWAY-BENCH-2026-09-03.md)) | `internal/engine/mlx.go:47` (`MLXEngineID`), `:198` (`abi.RegisterEngine`); `docs/supported/engines.md` (MLX row) | mlx-lm / vllm-mlx serving on Apple Silicon with unified-memory zero-copy CPU/GPU sharing |
| **Model-weight paging → disk/SSD tier** | **[SEAM-ONLY]** (#2726): `pagedRing` is a bounded per-weight LRU resident cache (byte budget, LRU victim, pinned-exemption, bit-equal to a resident `MatMul` on hit *and* miss) — but it is **standalone, off the live serve path**, f32-only, no async H2D, and links nothing new into the default binary. The **disk/SSD tier itself is still [GAP]**: nothing streams model weights from SSD on demand, so this does **not** back a >unified-memory serve claim | `internal/model/paging_ring.go:45` (`pagedRing`), `:67` (`newPagedRing`); `internal/model/paging.go` (page-in/page-out primitive, off serve path) | `ssd-llm` / KTransformers / MoE-Infinity stream layers or experts from SSD with predictive prefetch |

## 6. The four explicit honesty calls

So no downstream issue silently overclaims:

1. **The gateway has static replica dispatch, NOT fleet orchestration.**
   `internal/gateway/replica_router.go` and `fak serve --replica-base-url` can fan the
   served planner path across a fixed upstream set, round-robin. It still has no
   per-worker residency index (#41), node membership / health / drain / failover (#42),
   or load/KV-locality placement. The GPU-cluster path today can sit behind that static
   proxy, but fak still does **no** native multi-node compute. *[PARTIAL]*
2. **Streaming is live for prose, still gated for tools, and still partial.**
   `agent.StreamingPlanner` gives the gateway a content callback for OpenAI-compatible
   HTTP planners, and the Anthropic passthrough has a native SSE relay. Those paths stream
   prose deltas before the full turn finishes, while native `tool_calls` /
   Anthropic `tool_use.input` bytes remain buffered until the complete proposed call set
   clears adjudication. The offline mock, Gemini stream, and native in-kernel model still
   synthesize SSE from the finished turn. This is enough for real prose TTFT, not yet the
   full vLLM/SGLang per-step engine lifecycle. *[PARTIAL]*
3. **PP is SEAM-ONLY for serving.** `internal/model/pipeline.go:159` `MarshalHidden`/
   `UnmarshalHidden` (bit-exact hidden-state codec) and `internal/model/pipeline_transport.go:31`
   `TCPTransport` are real and loopback-proven byte-identical — but `TCPTransport`'s only peer
   is `EchoFrames` (`pipeline_transport.go:106`), an identity echo, not a band-running worker.
   No cross-node serve loop exists. #30
   / #85 reduce this to "implement one
   `Send` + a worker serve loop", still unwritten. *[SEAM-ONLY]*
4. **L7 trust DEGRADES to whole-prefix flush on Track A.** fak's bit-exact middle-span `Evict`
   ships (`internal/model/kvcache.go:94`, single-rotation re-RoPE from `Kraw`) — the thing
   vLLM/SGLang structurally **cannot** do. But `internal/enginecache/enginecache.go:272`
   `SupportsExactSpan` returns **false** for `EngineSGLang`, `EngineVLLM` **and** the MLX
   adapter added by #2724, so on a ridden
   engine exact-span collapses to a whole-prefix flush (`/flush_cache`, `/reset_prefix_cache`).
   Bit-exact span eviction is a **Track-B-only** guarantee. *[SHIPPED in-kernel / DEGRADED on ride]*

## 7. Track B fleet-scale is NOT co-equal near-term

Native parallelism (TP, EP, the device-mesh / collective-comms layer) is **greenfield**, and
that is the named blocking prerequisite for Track B fleet-scale:

- `compute.Backend`'s base interface (`internal/compute/compute.go:337`) exposes no collective
  op; the optional `CollectiveBackend` (`compute.go:431`) declares
  `AllReduceSum/AllGather/ReduceScatter/AllToAll`, and in the **default** build its only
  implementation is still the single-box CPU reference.
- **Corrected @`32586e6bc` (#2728):** a real NCCL/rank-as-process communicator *does* now exist —
  `internal/compute/cuda_nccl_pg.cu` (`ncclGetUniqueId` → `ncclCommInitRank` → `ncclAllReduce`)
  with its Go bridge `internal/compute/cuda_collective_pg.go`, plus per-rank device binding
  (`fcuda_set_device` / `fcuda_malloc_on`, `cuda_kernels.cu:233`/`:242`). The earlier
  "no NCCL communicator / no per-rank device binding / no rank-as-process anywhere in the tree"
  bullets were **true at `89abc5d` and are false now**; they are retired here.
- What still gates the conclusion: that substrate compiles **only** under `-tags cuda,nccl`
  with `FAK_CUDA_NCCL=1`, is excluded from the default `go build ./cmd/fak` *and* from a plain
  `-tags cuda` build, and both files self-declare `STATUS: unverified on a GPU-free host`.
  Reduction order is NCCL ring/tree, so it is an **Approx** peer (argmax-exact + cosine),
  never `max|Δ|=0` vs the host reduce it replaces.
- EP is explicitly fail-closed (`internal/model/tensor_parallel_forward.go:82` — re-verified
  exact at `32586e6bc`).

Therefore: **Track B is co-equal only at single-node → small-cluster scope.** Fleet-scale
native serving is sequenced *behind* the shared spine and the collective-comms substrate, and
is **scope-gated** (the TP+EP device-mesh design is a later, gated lever). Any throughput
comparison vs vLLM/SGLang is **unmeasured** and would need a measured bench run before it can
be claimed.

## 8. The child map — RIDE / NATIVE / shared

Every child of epic #50, mapped to its
track. (Shared spine is in [§4](#4-the-shared-spine-built-once-both-tracks-plug-in).)

**Track A — RIDE (orchestrate + govern; do NOT fork engine internals):**

| Issue | Deliverable |
|---|---|
| #40 | vLLM-V1 adapter behind the `EngineDriver` seam (HTTP + KV-events) |
| #39 | SGLang adapter behind the seam (RadixAttention signal + scheduler metrics) |
| #38 | Dynamo interop — **resolved as fak-governs / Dynamo-routes**: `engine.DynamoEngine` dispatches through Dynamo's public OpenAI-compatible frontend, normalizes Dynamo P/D worker metrics, and records the trust boundary in [`dynamo-interop.md`](dynamo-interop.md). |
| #37 | Orchestrate external P/D disaggregation + govern KV-transport bridge (NIXL/Mooncake/LMCache) — governance contract documented, live wiring is a later step |
| #2724 | MLX ride-adapter fronting mlx-lm / vllm-mlx on Apple Silicon — *not* a child of #50; it arrived via the Mac epic #2722 and lands in Track A because it rides an external engine behind the same seam |
| *(rides free)* | Speculative decoding inherited from the ridden engine (native verify/accept is a later Track-B lever; not separately filed in this block) |

**Track B — NATIVE (single-node → small-cluster; NOT chasing raw single-GPU tok/s vs vLLM):**

| Issue | Deliverable |
|---|---|
| #36 | Continuous-batching iteration scheduler over `StepBatch` |
| #35 | Admission control + priority + fairness gate |
| #34 | Paged/block KV allocator carrying the bit-exact `Evict` value-add |
| #33 | Design: prove bit-exact middle-span `Evict` survives paged/block KV |
| #32 | Real per-step request cancellation — thread `ctx` into the decode loop |
| #31 | Preemption + KV swap-to-host/recompute under memory pressure |
| #30 | Network PP serve loop — band-running worker replacing `EchoFrames` |
| #85 | Network `StageTransport` — PP stage handoff over the wire (NCCL/RPC under the proven seam) |
| #29 | Native `KVCache→bytes` serializer + RDMA/UCX KV transport (Track-B P/D data plane) |
| #28 | Native prefill/decode role split over the continuous-batching scheduler |

**L3 KV-governance value-add — rides ON TOP of the base items (out of scope for base-item parity, see [§9](#9-non-goals)):**

| Issue | Deliverable |
|---|---|
| #53 | epic(agentl3): external L3 disaggregated cache as a fak-governed tier |
| #55–#58 | L3RegionBackend seam, per-span DeletionCertificate, verified cross-tenant prefix-sharing, referee-sidecar |
| #414 | GLM-5.2 DSA: exact-span remote KV/index eviction *when an engine exposes it* (the inverse of honesty-call 4) |

## 9. Non-goals

- **Design/decision only.** No scheduler, router, transport, or kernel code lands in this doc.
- **Base-item parity framing.** The L3 KV-governance value-adds (the §8 L3 family) and the
  trust/exact-eviction layer ride **on top of** the base items. This doc must not let a parity
  item depend on a value-add, nor claim a value-add as parity.
- **Track A: do NOT fork** vLLM/SGLang/Dynamo/external-L3-cache internals — orchestrate + govern behind the
  `EngineDriver`/gateway seam.
- **Track B: not chasing raw single-GPU throughput parity vs vLLM** — native targets the
  single-node → small-cluster path and the kernel-owned-KV value-add unlock.
- **No benchmark number is produced here.** Measurement is the bench-harness sibling's job
  (#44); the epic's acceptance gates
  *every* parity claim on a MEASURED number against the best shipped SOTA setup.

## 10. De-dup map — and a correction the migration forced

The epic #50 body and several seed
*titles* cite "build-on" issue numbers (`#287/#149/#297/#292/#353/#373/#152/#274/#493/#285/
#348/#280/#504/#532/#129/#135/#105/#495/#533/#534`). **These are pre-migration internal-tracker
IDs carried verbatim into the GitHub bodies — they do NOT correspond to the GitHub issues of
the same number.** On GitHub today those numbers resolve to unrelated (mostly docs) issues,
closed issues, or nothing. Citing them as written would make this doc dishonest — the exact
failure mode it exists to prevent. The live serving work is the **#28–#50 seed block + #85 +
#414 + the #53–#58 L3 family**. The corrected map:

| Cited # (epic/seed body) | What the epic *meant* | What `#N` actually is on GitHub `@89abc5d` | Live equivalent |
|---|---|---|---|
| #287 / #149 / #297 | continuous-batching scheduler lineage | Vulkan opt / status-doc / Qwen2.5 checkpoints | #36, #35, #28 |
| #292 | paged/block KV | GGUF format completion | #34, #33 |
| #353 / #373 | metrics / dashboards plumbing | PHI policy example / readme-stale doc | #43 |
| #152 | router / fleet-dispatch | policy-doc deny-coupling | #45, #41, #42 |
| #274 | tensor parallelism | readme jargon doc | native TP — **no filed seed** (design-only, scope-gated R3; [§7](#7-track-b-fleet-scale-is-not-co-equal-near-term)) |
| #493 | network PP serve loop | **closed** policy hot-reload | #30, #85 |
| #285 | speculative decoding | claude-doc contradiction | rides free on Track A; native verify/accept unfiled in this block |
| #348 / #280 | streaming / detokenizer | **missing** / readme safety doc | #47, #48 |
| #504 / #532 | L3 KV-governance value-add | **missing** | #53–#58 |
| #129 | exact-span when engine exposes it | partition-doc wave markers | #414 |
| #135 | engine-cache reset / proxy-native asymmetry | partition-doc package list | #411 |
| #105 / #495 / #533 / #534 | bench-harness build-on | link-protocol doc / minimax / **missing** | #44 |

**Action for maintainers:** the seed titles' "(build-on #NNN)" suffixes should be re-pointed to
the live equivalents above; until then, **trust this table over the inline numbers.**

## 11. How to re-verify

Line numbers drift. Re-anchor any claim with ripgrep from the repo root:

```bash
rg -n 'func \(c \*KVCache\) Evict'            internal/model/kvcache.go            # honesty-call 4
rg -n 'func SupportsExactSpan'                internal/enginecache/enginecache.go  # honesty-call 4
rg -n 'writeChatCompletionStream|segmentContent' internal/gateway/http.go         # honesty-call 2
rg -n 'BaseURL|planner +agent.Planner'        internal/gateway/gateway.go          # honesty-call 1
rg -n 'EchoFrames|type TCPTransport'          internal/model/pipeline_transport.go # honesty-call 3
rg -n 'func \(bs \*BatchSession\) StepBatch'  internal/model/batch_step.go         # scheduler seam
rg -n 'ServingScrapeEmitter|fak_serving_kv_cache_usage_perc' internal/gateway        # serving metrics
rg -n 'cudaSetDevice|fcuda_set_device'        internal/compute/cuda_kernels.cu     # per-rank device binding
rg -n 'MLXEngineID|engine.cache.whole-prefix' internal/engine/mlx.go               # MLX ride adapter (#2724)
rg -n 'type pagedRing|func newPagedRing'      internal/model/paging_ring.go        # weight paging (#2726)
rg -n 'ncclCommInitRank' internal/compute/cuda_nccl_pg.cu                          # expect: REAL, build-gated
rg -n 'go:build cuda && nccl' internal/compute/cuda_collective_pg.go               # ...the gate that fences it
```

**Two anchors above were repaired by #2728 and one expectation was inverted.** At `89abc5d`
the first and sixth commands pointed at `internal/model/kv.go` and `internal/model/batch.go`;
both symbols have since moved file (`kvcache.go`, `batch_step.go`), so the sweep silently
matched **nothing** and stopped catching drift — the failure mode a re-verify block exists to
prevent. The last command previously read `rg -n 'ncclCommInitRank|world_size|DeviceMesh' .`
with the comment `# expect: no real substrate`; that expectation is now **false** (see
[§7](#7-track-b-fleet-scale-is-not-co-equal-near-term)), so it is replaced by the two commands
that assert the honest current shape: the substrate is real **and** it is build-gated.

## 12. The Mac humility fence (#2691) — blocked ladder retired by #2723 measured results

Issue #2728 existed to lift the #2691 humility fence once the Mac offload epic (#2722) landed
real numbers. That working spine has now run and landed: issue [#2723](https://github.com/anthony-chaudhary/fak/issues/2723)
captured empirical on-device measurements on `node-macos-a` (Apple M3 Pro 36GB, macOS Darwin arm64)
evaluating `Qwen3.8-27B Q4_K_M` across fak-native Metal, llama.cpp Metal, and MLX Metal in
[`docs/notes/MAC-THREEWAY-BENCH-2026-09-03.md`](../notes/MAC-THREEWAY-BENCH-2026-09-03.md).

**Retirement evidence — the blocked ladder is retired:**
The five child issues under epic #2722 are now fully accounted for with verified code and empirical artifacts:

| Child | State | Bearing on the fence / result |
|---|---|---|
| #2723 head-to-head fak vs llama.cpp vs MLX | **CLOSED** | Landed measured results on `node-macos-a` in [`docs/notes/MAC-THREEWAY-BENCH-2026-09-03.md`](../notes/MAC-THREEWAY-BENCH-2026-09-03.md): decode parity achieved (7.61 vs 7.38 tok/s, +3.1%), prefill 48.54 vs 52.74 tok/s, shared TTFT 12.60 ms |
| #2724 MLX ride adapter | CLOSED | Shipped ride adapter in `internal/engine/mlx.go`; MLX evaluated as external reference comparator in #2723 |
| #2725 prefill root-cause | CLOSED | Shipped isolation split diagnostic in `internal/model/metal_prefill_hybrid_core.go` and [`docs/notes/MAC-QWEN36-27B-PREFILL-ISOLATION-2026-07-07.md`](../notes/MAC-QWEN36-27B-PREFILL-ISOLATION-2026-07-07.md) |
| #2726 weight paging | CLOSED | Shipped `pagedRing` weight cache seam in `internal/model/paging_ring.go` |
| #2727 Mac cache-value P&L | **CLOSED** | Landed under epic #3809 / #3813 in [`docs/notes/MAC-MANYAGENT-CACHE-VALUE-2026-09-03.md`](../notes/MAC-MANYAGENT-CACHE-VALUE-2026-09-03.md): 88.2% compute reduction, flat 180 ms TTFT p50, 3.5× density gain |

**Measured comparison summary (Qwen3.8-27B Q4_K_M on Apple M3 Pro):**
- **Decode parity achieved:** In single-stream decode, fak-native achieves **7.61 tok/s** (p50 ITL 131.17 ms), leading llama.cpp Metal (7.38 tok/s) by **+3.1%** and reaching 94.3% of MLX (8.07 tok/s). On Apple Silicon unified memory, decode throughput across all three engines is tightly bound by the hardware memory bandwidth ceiling (~150 GB/s peak, ~120–128 GB/s effective).
- **Prefill compute:** fak-native achieves **48.54 tok/s** (TTFT 2652.00 ms for 128 ctx tokens), trailing llama.cpp Metal (52.74 tok/s) by -8.0% and MLX (64.10 tok/s) by -24.3%.
- **RadixAttention multi-agent TTFT:** While reference engines re-evaluate the full prompt prefix on every agent turn, fak-native with in-kernel RadixAttention prefix caching evaluates the preamble once globally and holds multi-agent shared TTFT flat at **12.60 ms** (>190× speedup).

The blocked ladder is therefore retired by the committed #2723 benchmark packet (`experiments/benchmark/runs/by-machine/node-macos-a/20260903T050000Z-macbench-threeway/packet.json`, validated via `fak macbench validate-comparison`).

---

*Parent epic:* epic(serving) #50. ·
*This doc closes* docs(serving) #49. ·
*Capability table + §11 sweep re-verified by* docs(mac) #2728 *(epic #2722).* ·
*Cross-linked with #2723 measured results by docs(mac) #5306.*
