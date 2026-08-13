---
title: "Native device mesh and collective seam for TP and EP"
description: "R3+ design contract and build-gated status for fak's native DP x TP x PP x EP device mesh: process groups, rank/world-size, CollectiveBackend primitives, CPU-ref single-rank behavior, and the dependency chain from native TP to MoE expert parallelism."
---

# Native device mesh and collective seam

This is the design contract for issue #25: a shared device-mesh and collective-comms
substrate under `compute.Backend`, so native tensor parallelism (TP) and MoE expert
parallelism (EP) do not grow two incompatible communication layers.

**Scope:** issue #25 remains the design contract; it did not itself land an NCCL/RCCL
binding, multi-device CUDA change, or MoE expert sharding. Follow-on work has since landed a
real multi-process NCCL AllReduce process group and per-rank CUDA device binding. That
substrate is **[PARTIAL]**, not [SHIPPED]: it enters only under `-tags cuda,nccl` with
`FAK_CUDA_NCCL=1`, is absent from the default and plain `-tags cuda` builds, and self-declares
`unverified on a GPU-free host`. The default binary contains no NCCL device communicator;
production device TP/EP therefore still rides external Track A (vLLM/SGLang/Dynamo workers),
and this document makes no raw single-GPU parity claim.

## Ground truth

| Claim | Status | Pointer |
|---|---|---|
| Base `compute.Backend` whole-op surface is still the forward-loop target (`MatMul`, `BatchedMatMul`, `RMSNorm`, `RoPE`, `SwiGLU`, `Attention`, `Argmax`). | [SHIPPED] | `internal/compute/compute.go:352-378` |
| The tensor collective seam exists as the optional `compute.CollectiveBackend` interface with `AllReduceSum`, `AllGather`, `ReduceScatter`, and `AllToAll`. | [SEAM-ONLY] | `internal/compute/compute.go:477-515` |
| CPU reference collectives are single-box, rank-order exact, fail closed on malformed parts, and the single-rank case is identity. | [SHIPPED] | `internal/compute/collective.go:115` (`AllReduceSum`), `:132` (`ReduceScatter`), `:153` (`AllGather`), `:191` (`AllToAll`) |
| Cross-process host-f32 collectives exist through `model.DistComm`, but they are not a device/NCCL communicator. | [PARTIAL] | `internal/model/dist_collective.go`, `docs/serving/multi-node-compute.md` |
| CUDA has per-rank device binding and a real multi-process NCCL process group (`ncclGetUniqueId` -> `ncclCommInitRank` -> `ncclAllReduce`), but only behind the opt-in CUDA+NCCL build gate. The default/plain-CUDA builds exclude it, its source says `unverified on a GPU-free host`, and NCCL ring/tree reduction makes it an Approx peer (argmax-exact + cosine), never a `max|Delta|=0` peer. | [PARTIAL] | `internal/compute/compute.go:434` (`ProcessGroupBackend`); `internal/compute/cuda_collective_pg.go:1-16`; `internal/compute/cuda_nccl_pg.cu:13-30`, `:46-80`; `internal/compute/cuda_kernels.cu:63` (device-0 probe), `:233-244` (per-rank binding/allocation); `internal/compute/build_cuda.sh:212-231` |
| MoE routing and selected experts run in-process: router top-k, then per-expert SwiGLU, then weighted accumulation. | [PARTIAL] | `internal/model/moe.go:357` (`route`), `:449` (`moeFFN`) |
| Native TP explicitly refuses MoE/GLM-MoE-DSA instead of mis-serving those decompositions. | [GAP] | `internal/model/tensor_parallel_forward.go:79-82` |

Line numbers drift; re-anchor with:

```bash
rg -n 'type Backend interface|type ProcessGroupBackend interface|type CollectiveBackend interface' internal/compute/compute.go
rg -n 'func \(c \*cpuBackend\) (AllReduceSum|ReduceScatter|AllGather|AllToAll)' internal/compute/collective.go
rg -n 'cudaSetDevice|fcuda_set_device|fcuda_malloc_on' internal/compute/cuda_kernels.cu
rg -n 'ncclGetUniqueId|ncclCommInitRank|ncclAllReduce|STATUS:|GPU-free' internal/compute/cuda_nccl_pg.cu
rg -n 'go:build cuda && nccl|STATUS:|GPU-free' internal/compute/cuda_collective_pg.go
rg -n 'FAK_CUDA_NCCL|GO_TAGS="cuda,nccl"' internal/compute/build_cuda.sh
rg -n 'func route\(|type moeFFN|ForwardTP does not yet shard MoE' internal/model
```

## Collective seam

`compute.CollectiveBackend` is the additive method surface under `compute.Backend`.
Backends that do not implement it remain valid single-device backends. Backends that do
implement it are bound to one communicator/process group, so tensors reduced by that backend
are rank-local shards in the same group.

The process-group layer owns:

- `rank`: the current process's rank within the group.
- `world_size`: the number of ranks in the group.
- `local_rank` / `device_id`: the rank's device within a node.
- `group_name`: the semantic group (`tp`, `ep`, `pp`, `dp`, or a composed subgroup).
- `mesh_coord`: the rank coordinate in the DP x TP x PP x EP mesh.
- `backend`: the `compute.CollectiveBackend` implementation for the rank's resident tensors.

A future concrete shape can be a runtime object rather than fields on base `Backend`:

```go
type CollectiveGroup struct {
	Name      string
	Rank      int
	WorldSize int
	LocalRank int
	DeviceID  int
	Coord     MeshCoord
	Backend   compute.CollectiveBackend
}
```

The four primitives stay exactly the HAL primitives already named:

- `AllReduceSum`: row-parallel partial sum; equal-length parts; rank-order reference.
- `AllGather`: column-parallel output concatenation; rank-ordered shards.
- `ReduceScatter`: sequence-parallel activation reduction while keeping only each rank's
  shard; must satisfy `AllReduceSum == AllGather(ReduceScatter(parts))`.
- `AllToAll`: layout transpose; the primitive EP needs for expert dispatch/combine and TP
  needs for sequence/head layout changes; must be an involution.

The CPU reference group is the degenerate group: `rank=0`, `world_size=1`, `local_rank=0`,
`device_id=0`. All four collectives are identity for one rank, preserving the bit-exact
single-device path. A real NCCL/RCCL backend must preserve that single-rank identity and the
same fail-closed shape/type checks. Multi-rank NCCL ring/tree reduction does not preserve the
CPU reference's rank-ascending addition order, so the build-gated CUDA implementation is an
Approx peer judged by argmax-exact + cosine, not byte equality, before any throughput claim.

## Device mesh

The native mesh coordinate is:

```text
rank := (dp, pp, tp, ep)
world_size := DP * PP * TP * EP
```

The planner derives process groups from fixed coordinates:

| Group | Varies | Holds fixed | Communication |
|---|---|---|---|
| DP | `dp` | `pp,tp,ep` | Inference replicas and placement/routing; no per-token collective by default. |
| PP | `pp` | `dp,tp,ep` | Point-to-point stage handoff through `StageTransport`; deliberately not a collective. |
| TP | `tp` | `dp,pp,ep` | `AllGather`, `AllReduceSum`, and later `ReduceScatter` inside dense/attention blocks. |
| EP | `ep` | `dp,pp,tp` | `AllToAll` dispatch to expert owners and `AllToAll`/local reduce combine back to token order. |

Topology is part of placement, not an afterthought:

- TP and EP are innermost axes and should stay inside an NVLink/NVSwitch island where possible,
  because they run per layer and can require all-reduce or all-to-all on the critical path.
- PP is outermost across nodes/fabric. It moves activations between layer bands and can tolerate
  the higher latency of TCP/RDMA/UCX/NCCL point-to-point better than TP/EP collectives can.
- DP spans replicas and failure domains. It is a gateway/router concern first; any weight-sync or
  metric sync is out of the decode hot path.
- Cross-node EP is a last resort. If the expert set does not fit inside one island, the planner
  must price the inter-node all-to-all explicitly rather than hiding it behind the same label.

## Dependency chain

1. **Collective seam / device mesh** - multi-engineer-month. The optional HAL plus a
   build-gated NCCL AllReduce/process-group and per-rank device-binding rung now exist. The
   remaining work is a GPU witness, complete process-group collectives/mesh placement, and an
   RCCL peer, all checked against the CPU reference at the backend's declared correctness
   class. This issue (#25) is the design gate.
2. **Native TP** - multi-engineer-month. Consume the seam for Megatron-style column/row sharding
   over real device tensors. The host/CPU-ref decomposition, communicator, and per-rank device
   binding exist, but the device path remains build-gated and GPU-unverified; a measured
   multi-GPU run is still required before a parity or throughput claim.
3. **EP-for-MoE** - multi-engineer-month after TP. Add expert ownership and load-aware routing on
   top of the same mesh. The EP delta over TP is `AllToAll` token dispatch to expert owners,
   per-owner expert execution, and dispatch-combine back to token order. It does not mint a
   second collective substrate.

## Coordination notes

The migrated #25 body cites old internal tracker IDs `#274` and `#492`. On live GitHub those
numbers resolve to unrelated issues. The corrected live coordination targets are:

- Native TP: live GitHub #295 documents multi-GPU tensor parallelism; #25 is its lower
  communicator/device-mesh design gate, not a duplicate TP issue.
- GLM-DSA / MoE backend consumer: live GitHub #86 is the GLM-DSA `compute.Backend` consumer.
  EP-for-MoE remains downstream of #25 because it needs the shared `AllToAll` substrate.

Do not post coordination comments to live GitHub #274 or #492 for this topic; doing so would
attach TP/EP design notes to unrelated tickets. If those internal tracker IDs are ever imported
with their original meanings, this doc should be cited there.

## Non-goals

- No NCCL/RCCL implementation in this issue.
- No CUDA multi-device selection change in this issue.
- No MoE expert placement or all-to-all dispatch implementation in this issue.
- No new duplicate TP or EP issue minted from this design.
- No raw single-GPU throughput parity claim.
