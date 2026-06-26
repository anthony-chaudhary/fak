---
title: "Multi-GPU tensor parallelism in fak: architecture, API, and setup"
description: "How fak's native tensor-parallel (multi-GPU) path is built — the Megatron-style column/row sharding, the four-collective HAL seam, the in-process and cross-process collectives, and the exact NCCL/RCCL swap-in point. Documents what runs and is bit-exact today (host-free) and the hardware-gated residual: a real device communicator and a 2-/4-GPU run."
---

# Multi-GPU tensor parallelism in fak

Tensor parallelism (TP) splits a *single* layer's matmuls across several GPUs so a
model that does not fit on one device can be served across many. This page is the
single setup/architecture reference for fak's **native** TP path: the API you call,
the collective seam you implement to reach real hardware, what is proven today, and
exactly what is not yet shipped.

> **Status banner (read first).** The TP decomposition, the four-collective seam, and
> two collective implementations (in-process and **real cross-process over TCP**) are
> shipped and **bit-exact** against a single-device reference — all witnessable on a
> CPU with no GPU. What is **not** shipped is a **device communicator** (an NCCL /
> RCCL `CollectiveBackend`) and therefore a live **2×/4× GPU run** with throughput
> scaling. That residual is hardware-gated; it is a backend that implements two/four
> methods behind the seam below, **not** a rewrite of the decomposition. This tracks
> GitHub **#295** (`feat(gpu): Multi-GPU Tensor Parallelism [A-007]`).

---

## 1. The shape: pipeline × tensor

A real multi-GPU serving plan is a **grid of pipeline stages × tensor-parallel ranks**.
The two axes are orthogonal and compose:

- **Pipeline parallelism** splits the layer *stack* across workers and crosses a hidden
  state at each stage boundary (`internal/model/partition.go`, `pipeline.go`). The wire
  is the `StageTransport` seam; `TCPTransport` (`internal/model/pipeline_transport.go`)
  is a real socket implementation, byte-identical to the in-process `LocalTransport`.
- **Tensor parallelism** splits a *single* layer's matmuls across workers and crosses a
  collective (AllGather / AllReduce) *inside* the layer. This page is about this axis.

You can run either alone or both together; this doc covers the tensor axis end to end.

---

## 2. The decomposition (Megatron-style, made honest)

fak uses the canonical Megatron-LM decomposition, and the repo's existing numeric
discipline (`internal/model/parallel.go`) is what keeps it honest:

- **Column-parallel** (shard the **output** features): `y = x·Wᵀ`, with `W` split into
  row-bands `[W_0; W_1; …]`. Rank `r` computes its output band `y_r = x·W_rᵀ`; the parts
  are **AllGather**-concatenated in rank order. Each output element is computed by exactly
  **one** rank in the **same inner order** as the monolithic matmul, so column-parallel is
  **bit-exact** vs single-device (`max|Δ| = 0`).
- **Row-parallel** (shard the **contraction** dim): `W` split into column-bands, `x` into
  matching segments. Rank `r` computes a **partial** `y` over its slice; the parts are
  **AllReduce**-summed. This *reassociates* the reduction, so it is not bit-exact vs the
  monolith — it drifts ~`1e-6`, the same non-associativity `parallel.go` already documents
  for `fdot`. It **is** bit-exact vs a **shard-grouped reference** (the rank-ordered sum of
  each shard's `fdot`), which is the invariant the gate pins.

Megatron composes exactly these: attention is QKV column-parallel (shard heads) then
output-proj row-parallel; the FFN is gate/up column-parallel then down row-parallel — one
AllReduce per block, the intermediate never gathered. `TensorParallelFFN` and
`TensorParallelAttention` (`internal/model/tensor_parallel.go`,
`tensor_parallel_attn.go`) are those composed blocks.

The algebra the wired forward path obeys:

```
ForwardTP(ranks=1)  ==(bit-exact, max|Δ|=0)        Forward
ForwardTP(ranks=N)  ==(AllReduce reassociation)    ForwardTP(ranks=1)   (~1e-6, rank-order pinned)
```

The `ranks=1` leg is the **"bit-exact vs the single-GPU path"** rung — and it is
witnessable on a CPU, with **no multi-GPU hardware**.

---

## 3. The API surface

| You call | Where | What it does |
|---|---|---|
| `Model.ForwardTP(ids, TPConfig{AttnRanks, FFNRanks, Coll})` | `internal/model/tensor_parallel_forward.go` | The wired TP forward. `AttnRanks` shards attention over the kv-head groups; `FFNRanks` shards the FFN over the intermediate dim. `Coll == nil` → `LocalCollective`. |
| `NewTPPlan(dim, ranks)` → `TPPlan` / `TPShard` | `internal/model/tensor_parallel.go` | Validated tiling of one dimension into `ranks` contiguous, non-overlapping, complete shards. Fails closed on a degenerate plan (a rank with no work). |
| `Collective` (`LocalCollective`) | `internal/model/tensor_parallel.go` | The host-`[]float32` AllGather/AllReduce seam; `LocalCollective` is the single-box, bit-exact default. |
| `CollectiveBackend` | `internal/compute/compute.go` | The **device-tensor** cross-rank seam at the HAL — the swap-in point for real hardware (see §5). |
| `BackendCollective` | `internal/model/collective_bridge.go` | Bridges `model.Collective` onto a HAL `CollectiveBackend`, byte-identical to `LocalCollective`. |
| `DistComm` | `internal/model/dist_collective.go` | A **real cross-process** communicator (a process group): a star rooted at rank 0 over framed TCP. The distributed twin of `LocalCollective`. |
| `TCPTransport` | `internal/model/pipeline_transport.go` | The pipeline-axis cross-process wire (real socket, byte-identical to in-process). |

Minimal call (single box, the bit-exact default):

```go
act, err := m.ForwardTP(ids, model.TPConfig{AttnRanks: 2, FFNRanks: 2}) // Coll nil → LocalCollective
```

---

## 4. The collective seam — four primitives

A device collective implements the cross-rank reduction. The HAL interface
`compute.CollectiveBackend` declares the four canonical Megatron collectives; the
CPU reference (`internal/compute/collective.go`) is the single-box, **exact** default
that a real communicator must reproduce **byte-for-byte**:

- **AllReduceSum** — element-wise sum of equal-length per-rank partials, added in rank
  order. Post-block reduction for row-parallel.
- **AllGather** — rank-ordered concatenation of per-rank shards. Recombines a
  column-parallel output.
- **ReduceScatter** — the AllReduceSum result scattered into equal per-rank shards; the
  dual of AllGather. `AllReduceSum ≡ AllGather∘ReduceScatter` (the identity the reference
  pins). Lets sequence-parallel TP keep only a `1/P` slice of the activation.
- **AllToAll** — the transpose collective (a different shard to each peer). An involution
  (`AllToAll∘AllToAll == identity`); `ReduceScatter` is recoverable as `AllToAll` + a local
  per-rank reduce. Turns a sequence-sharded activation into a head-sharded one.

Every method **fails closed** at the boundary: no parts, ragged partials, a non-F32 part,
an unready part, or a part owned by a *different* backend (the cross-backend reduction a
real communicator rejects — a CUDA tensor cannot be all-reduced against a host tensor) is
refused, never silently mis-reduced. Indivisible inputs (real NCCL requires
`sendcount % nranks == 0`) fail closed too.

---

## 5. Reaching real hardware — the swap-in point

Adding NCCL (NVIDIA) or RCCL (AMD) is **a backend that implements `CollectiveBackend`**,
discovered by a type-assert in the forward loop (with a cheap `Caps().Collective`
pre-check) — never an edit to the forward loop:

1. Implement `AllReduceSum`, `AllGather`, `ReduceScatter`, `AllToAll` over device-resident
   tensors on one communicator (`ncclAllReduce` / `ncclAllGather` / `ncclReduceScatter` /
   `ncclAllToAll`, or the RCCL equivalents).
2. It is **correct iff it reproduces the reference bytes** — the rank-order spec in
   `collective.go` and the identities above are the conformance target. The CPU reference
   is your test oracle on a single box before you ever touch two GPUs.
3. For a **multi-process** topology (N processes, rank `r` holds only its own part — *why*
   real NCCL serving runs N processes), `DistComm` already pins the cross-process protocol:
   a star rooted at rank 0, one framed connection per worker, each collective a single
   gather→reduce→scatter round reduced through `LocalCollective`'s rank-order spec, so the
   result is byte-identical to the in-process gate by construction.

What you need for the **live** run (the hardware-gated residual of #295):

- 2× (or 4×) GPUs with NCCL/RCCL on the host (e.g. 2× RTX 4090, or the 8-GPU lane in
  [`docs/HARDWARE-MATRIX.md`](../HARDWARE-MATRIX.md)).
- The `CollectiveBackend` implementation from step 1, registered as a backend cap.
- A 70B checkpoint sharded across the ranks per `NewTPPlan`.

Until that backend lands, fak serves on a single GPU or CPU; there is **no** device-side
multi-GPU all-reduce yet. This doc does not claim otherwise.

---

## 6. What is proven today (host-free witnesses)

All of these run under `go test ./internal/model/... ./internal/compute/...` on a CPU —
no GPU required:

| Property | Witness |
|---|---|
| Column-parallel matmul bit-exact vs monolith | `model.TestColumnParallelMatMulBitExact` |
| Row-parallel matmul == shard-grouped reference | `model.TestRowParallelMatMulMatchesShardReference` |
| TP FFN / attention / full layer == monolith | `model.TestTensorParallelFFNMatchesMonolith`, `TestTensorParallelAttentionMatchesMonolith`, `TestTensorParallelLayerMatchesMonolith` |
| `ForwardTP(ranks=1)` == `Forward` (**bit-exact vs single-device**) | `model.TestForwardTPMatchesForward`, `TestTPForwardRanks1MatchesLive` |
| Reduction order is rank-order pinned | `model.TestTPForwardReductionRankOrderPinned` |
| HAL collective bridge == `LocalCollective` byte-for-byte | `model.TestBackendCollectiveMatchesLocal`, `TestForwardTPViaBackendCollective` |
| **Real cross-process** `ForwardTP` over `DistComm` == single-process `ForwardTP` | `model.TestForwardTPDistCommRanksMatchLocalForwardTP`, `TestForwardTPViaDistCommCollective` |
| `DistComm` over a real wire == `LocalCollective` (+ fail-closed on desync/ragged) | `model.TestDistCommAllReduceSumMatchesLocal`, `TestDistCommFailsClosedOpDesync`, … |
| The four device collectives + identities + fail-closed | `compute.TestCollectiveAllReduceSumRankOrder`, `TestCollectiveAllGatherRankOrder`, `TestCollectiveReduceScatter`, `TestCollectiveAllToAll`, `TestCollectiveFailsClosed` |

The cross-process row (`DistComm`) is the strongest host-free rung: it is a **genuine
multi-process collective over a socket** proven byte-identical to the single-process path,
on hardware that exists. The device backend swaps in behind the same contract.

---

## 7. #295 acceptance status (honest)

| Acceptance item | Status |
|---|---|
| Design TP API (`compute.Backend` extension) | ✅ Shipped — `CollectiveBackend` seam + `ForwardTP`/`TPConfig`/`TPPlan`. |
| All-reduce collective implementation | ✅ Shipped — CPU reference (`collective.go`) + cross-process `DistComm`, bit-exact. |
| Bit-exact vs single-GPU path | ✅ Host-free rung shipped (`ForwardTP(ranks=1) == Forward`). Device-side bit-exact awaits the GPU backend. |
| Documentation for multi-GPU setup | ✅ **This page.** |
| NCCL/RCCL backend | ❌ Not shipped — the `CollectiveBackend` device impl (§5). **Hardware-gated.** |
| Run 70B across 2× RTX 4090 · near-linear 1.8× scaling | ❌ Not run — depends on the backend above + a 2-GPU host. **Hardware-gated.** |

The math primitive and both collectives are landed and proven; the **device communicator
and the real multi-GPU run** are the remaining work, and they need multi-GPU hardware.

---

## 8. Where the code lives

```
internal/model/tensor_parallel.go          # TPPlan/TPShard, TensorParallelFFN, Collective/LocalCollective
internal/model/tensor_parallel_attn.go     # TensorParallelAttention
internal/model/tensor_parallel_forward.go  # ForwardTP, TPConfig (the wired forward)
internal/model/collective_bridge.go        # BackendCollective (model→HAL bridge)
internal/model/dist_collective.go          # DistComm (cross-process communicator over TCP)
internal/model/pipeline_transport.go       # TCPTransport (pipeline-axis wire)
internal/compute/compute.go                # CollectiveBackend interface (the device seam)
internal/compute/collective.go             # CPU reference implementation (the conformance oracle)
```

Related reading:
[`docs/comm-as-mpi-split.md`](../comm-as-mpi-split.md) (the agent-layer lease ≈ `MPI_Comm_split`, and why it is *not* this tensor-layer collective),
[`docs/explainers/sota-optimizations.md`](sota-optimizations.md) (where TP sits among serving optimizations),
[`docs/HARDWARE-MATRIX.md`](../HARDWARE-MATRIX.md) (the multi-GPU serving lane).
