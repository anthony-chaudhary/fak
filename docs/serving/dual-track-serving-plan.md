---
title: "fak dual-track serving: RIDE + NATIVE over one shared spine"
description: "fak's authoritative dual-track serving plan: ride best-in-class engines (vLLM, SGLang) plus a native in-kernel engine, over one shared track-neutral spine."
---

# Dual-track serving plan — RIDE + NATIVE over one shared spine

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
> `file:line` pointer verified against the working tree at commit `89abc5d` (2026-06-22).
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
   (`internal/model/kv.go:60`) — quarantine eviction the SOTA engines structurally **cannot**
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

Native multi-node parallelism is **greenfield**. The real distributed substrate does not
exist in the tree: an `rg` sweep finds **no** `ncclCommInitRank` / `rcclAllReduce` /
`nvshmem`, **no** `world_size`/`worldSize`, **no** `init_process_group` / `DeviceMesh`. What
*does* exist is a single-box **simulation** + a named swap-in seam (`model.Collective` /
`LocalCollective` `internal/model/tensor_parallel.go:140`, `compute.CollectiveBackend`
`internal/compute/compute.go:347` with a CPU-reference impl), and a CUDA backend that
hardcodes one device (`cudaSetDevice(0)` `internal/compute/cuda_kernels.cu:52`). Expert
parallelism fails closed (`ForwardTP` errors on MoE, `internal/model/tensor_parallel_forward.go:82`).

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
| **Gateway topology** | **[PARTIAL]** single-upstream proxy: one `BaseURL`, one `agent.Planner` seam — no replica set, no router | `internal/gateway/gateway.go:65` (`BaseURL`), `:188` (`planner`) | SGLang Router / vLLM router / LiteLLM front N replicas — fak is the single-engine front, not that router |
| **Engine seam** | **[PARTIAL]** one-shot buffered `Complete`; `ctx` never consulted inside decode → no cancel/stream | `internal/abi/registry.go:568`; `internal/modelengine/modelengine.go:139` (`Complete`), `:151` (`sess.Generate` one-shot); `internal/agent/chat.go:440` (`Planner`, no token callback) | vLLM-V1 `EngineCore` admit→per-step-decode→stream→reclaim lifecycle |
| **Streaming** | **[PARTIAL]** synthesized **post-adjudication** from the finished turn; TTFT == whole-turn latency | `internal/gateway/http.go:338` (`writeChatCompletionStream`), `:395` (`segmentContent` word-split) | vLLM-V1 / SGLang flush each decoded token live → real-TTFT SSE, inter-token gaps == TPOT |
| **Incremental detokenizer** | **[GAP]** whole reply detokenized once; no streaming detokenizer | — (prerequisite, #48) | streaming detokenizer feeding per-token SSE |
| **Continuous-batching scheduler** | **[SEAM-ONLY]** `StepBatch` per-step primitive exists; **no** admit/evict loop. `GenerateBatch` runs static fixed-B and re-feeds EOS into finished slots | `internal/model/batch.go:1122` (`StepBatch`), `:1148` (`GenerateBatch`) | vLLM-V1 `EngineCore` / SGLang `Scheduler` iteration loop: admit, retire, rebuild running batch each step |
| **Chunked prefill** | **[PARTIAL]** rectangular equal-length panel ≤512 tokens; ragged batches fall back to serial prefill | `internal/model/batch.go:761` (`rectangularPrefillLen`); `internal/model/batch.go:47` (`batchRectPrefillMaxTokens=512`) | chunked prefill (vLLM-V1 default) packs different requests' prefill chunks + decode into one ragged varlen batch |
| **Request admission / priority** | **[GAP]** "budget" == matmul worker count (thread-pool width), not request admission | `internal/model/budget.go:50` (`SetWorkerBudget`) | waiting queue + FCFS/priority policy + KV/token budget + preemption + `max-num-seqs` |
| **Paged / block KV** | **[GAP]** contiguous-append flat `[]float32` per layer; `Evict` memmoves to compact | `internal/model/kv.go:18` (`KVCache`) | vLLM PagedAttention `BlockManager` / SGLang token-to-KV-pool: fixed blocks + block table, O(1) evict, COW prefix share |
| **Radix prefix cache** | **[SHIPPED]** real RadixAttention trie (longest-prefix walk, edge split, LRU, ref-count leases) — but **single-process, in-memory** | `internal/radixkv/radixkv.go:72` (`Tree`) | SGLang RadixAttention (same algo) + a *distributed* residency index (Mooncake / LMCache) sharded across replicas |
| **Bit-exact middle-span Evict** | **[SHIPPED]** single-rotation re-RoPE from stored pre-RoPE `Kraw` → cache byte-identical to one that never saw the span | `internal/model/kv.go:60` (`Evict`) | **None.** vLLM `reset_prefix_cache` / SGLang `flush_cache` drop whole prefixes/LRU leaves only — a genuine structural value-add |
| **Exact-span on a ridden engine** | **[PARTIAL]** degrades to whole-prefix flush — `SupportsExactSpan` is false for SGLang **and** vLLM | `internal/enginecache/enginecache.go:132` (`SupportsExactSpan`), `:148` (`/flush_cache`), `:156` (`/reset_prefix_cache`) | engines expose only coarse whole-prefix reset over HTTP; neither can evict a named middle span |
| **KV transport / P/D data plane** | **[SEAM-ONLY]** metadata-only descriptor; `BytesMoved` is a reported counter, **no path copies KV bytes** | `internal/cachemeta/kvtransfer.go:54` (`FromKVTransfer`) | vLLM `KVConnector` (NIXL/LMCache), SGLang + Mooncake: RDMA/NVLink byte movers on the wire |
| **Pipeline parallelism** | codec **[SHIPPED]** / transport **[SEAM-ONLY]**: real bit-exact hidden-state codec + a `TCPTransport`, but its only peer is `EchoFrames` (identity echo), no band-running worker | `internal/model/pipeline.go:159` (`MarshalHidden`); `internal/model/pipeline_transport.go:30` (`TCPTransport`), `:105` (`EchoFrames`) | per-rank worker process that runs its band on GPU and forwards to the next rank over NCCL P2P |
| **Tensor parallelism** | **[SEAM-ONLY] / [GAP]**: single-box host-array **simulation** + swap-in seam shipped; real NCCL/world-size/device-mesh/per-rank-device GAP | `internal/model/tensor_parallel.go:140` (`Collective`/`LocalCollective`); `internal/compute/compute.go:347` (`CollectiveBackend`, cpu-ref only); `internal/compute/cuda_kernels.cu:52` (`cudaSetDevice(0)`) | vLLM/SGLang: `init_distributed_environment` builds TP/PP/EP groups over a NCCL world (world_size/rank/local_rank) + custom all-reduce |
| **Expert parallelism (MoE)** | **[GAP]** `ForwardTP` fails closed on MoE | `internal/model/tensor_parallel_forward.go:82` | all-to-all expert dispatch (e.g. DeepEP) |
| **Serving metrics** | **[PARTIAL]** Prometheus plumbing + an inference family (`fak_gateway_inference_*`) is good; **no** TTFT/TPOT/ITL/goodput/queue-depth/KV-util/per-token series | `internal/gateway/metrics.go:24` (`gatewayMetrics`) | `vllm:time_to_first_token_seconds`, `vllm:time_per_output_token_seconds`, `num_requests_running/waiting`, `gpu_cache_usage_perc` |

## 6. The four explicit honesty calls

So no downstream issue silently overclaims:

1. **The gateway is a single-upstream proxy today, NOT fleet orchestration.**
   `internal/gateway/gateway.go:65` fronts one `BaseURL` upstream through one `agent.Planner`
   seam (`gateway.go:188`); there is no replica set or router. The GPU-cluster path today runs
   one SGLang TP=8 upstream behind that single seam — fak does **no** native multi-node compute.
   *[PARTIAL]*
2. **Streaming is SYNTHESIZED from the finished adjudicated turn, not token-by-token.**
   `internal/gateway/http.go:338` (`writeChatCompletionStream`) runs strictly *after*
   `planner.Complete` + adjudication, then fakes word-granular deltas with `segmentContent`
   (`http.go:395`). TTFT == whole-turn latency on both wires. The `agent.Planner` seam
   (`chat.go:440`) has no token-callback method, so this is structural, not a missing flag.
   *(Note: the issue body described a "2-chunk SSE"; the synthesizer has since grown
   word-granular splitting — the chunk count changed, the load-bearing fact did not: it is
   post-adjudication synthesis, never incremental detokenization.)* *[PARTIAL]*
3. **PP is SEAM-ONLY for serving.** `internal/model/pipeline.go:159` `MarshalHidden`/
   `UnmarshalHidden` (bit-exact hidden-state codec) and `internal/model/pipeline_transport.go:30`
   `TCPTransport` are real and loopback-proven byte-identical — but `TCPTransport`'s only peer
   is `EchoFrames` (`pipeline_transport.go:105`), an identity echo, not a band-running worker.
   No cross-node serve loop exists. #30
   / #85 reduce this to "implement one
   `Send` + a worker serve loop", still unwritten. *[SEAM-ONLY]*
4. **L7 trust DEGRADES to whole-prefix flush on Track A.** fak's bit-exact middle-span `Evict`
   ships (`internal/model/kv.go:60`, single-rotation re-RoPE from `Kraw`) — the thing
   vLLM/SGLang structurally **cannot** do. But `internal/enginecache/enginecache.go:132`
   `SupportsExactSpan` returns **false** for `EngineSGLang` and `EngineVLLM`, so on a ridden
   engine exact-span collapses to a whole-prefix flush (`/flush_cache`, `/reset_prefix_cache`).
   Bit-exact span eviction is a **Track-B-only** guarantee. *[SHIPPED in-kernel / DEGRADED on ride]*

## 7. Track B fleet-scale is NOT co-equal near-term

Native parallelism (TP, EP, the device-mesh / collective-comms layer) is **greenfield**, and
that is the named blocking prerequisite for Track B fleet-scale:

- `compute.Backend`'s base interface (`internal/compute/compute.go:294`) exposes no collective
  op; the optional `CollectiveBackend` (`compute.go:347`) declares
  `AllReduceSum/AllGather/ReduceScatter/AllToAll` but its **only** implementation is the
  single-box CPU reference — no device communicator behind it. The CUDA backend does not
  implement it.
- `internal/compute/cuda_kernels.cu:52` hardcodes `cudaSetDevice(0)` — no per-rank device
  binding.
- There is **no** NCCL/RCCL communicator, **no** `world_size`/rank-as-process, **no**
  device-mesh anywhere in the tree. "rank" is exclusively a loop/shard index.
- EP is explicitly fail-closed (`internal/model/tensor_parallel_forward.go:82`).

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
| #38 | Dynamo interop — fak router stands beside/in front of Dynamo's P/D router |
| #37 | Orchestrate external P/D disaggregation + govern KV-transport bridge (NIXL/Mooncake/LMCache) — **fak moves zero KV bytes** |
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
rg -n 'func \(c \*KVCache\) Evict'            internal/model/kv.go                 # honesty-call 4
rg -n 'func SupportsExactSpan'                internal/enginecache/enginecache.go  # honesty-call 4
rg -n 'writeChatCompletionStream|segmentContent' internal/gateway/http.go         # honesty-call 2
rg -n 'BaseURL|planner +agent.Planner'        internal/gateway/gateway.go          # honesty-call 1
rg -n 'EchoFrames|type TCPTransport'          internal/model/pipeline_transport.go # honesty-call 3
rg -n 'func \(bs \*BatchSession\) StepBatch'  internal/model/batch.go              # scheduler seam
rg -n 'cudaSetDevice'                         internal/compute/cuda_kernels.cu     # parallelism greenfield
rg -n 'ncclCommInitRank|world_size|DeviceMesh' .                                   # expect: no real substrate
```

---

*Parent epic:* epic(serving) #50. ·
*This doc closes* docs(serving) #49.
