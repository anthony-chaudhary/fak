---
title: "fak collectives: the MPI reduce/allreduce/bcast family mapped to its real fak symbols, and the agent-vs-tensor honesty line"
description: "The canonical anti-conflation map for the MPI collective family (MPI_Reduce / MPI_Allreduce / MPI_Bcast) onto the shipped fak symbols they are shaped like — the AGENT layer (non-bit-exact, scope-bounded: modelroute.Combine + the Reduce* set, gateway.dispatchEnsemble, abi.ShareScope as the broadcast bound) versus the TENSOR layer (model.DistComm.AllReduceSum / AllGather, real cross-process HOST float32, explicitly NOT NCCL / not-multi-GPU). Quotes the dist_collective.go and modelroute all_reduce disclaimers verbatim. MPI is the design lens, never an HPC number borrowed."
---

# collectives: the MPI reduce/allreduce/bcast family, mapped honestly

fak has collective-shaped surfaces in **two different rank spaces**, and the single
biggest overclaim risk in the MPI-shaped epic (#639) is conflating them. This doc is the
canonical map every other collective-shaped child links: one row per MPI collective
primitive → its real fak symbol → which **layer** it lives in, so the rank-space
distinction is written down once instead of inferred from terse inline comments. It is
part of the MPI-shaped message-passing epic (#639).

> **Honesty caveat (read first).** fak borrows the **structure** of MPI collectives and
> the **vocabulary**, never an MPI/HPC number. There are two distinct rank spaces:
>
> - The **AGENT layer** is non-bit-exact and scope-bounded. `modelroute.Combine` folds
>   many *models'* answers into one; its determinism is pinned to the routing decision and
>   the fold over fixed votes, **not** to the members' outputs (those come from non-bit-exact
>   engines). `abi.ShareScope` *bounds* where a shared result may become visible — it does
>   not move bytes. Ranks here index *agents/roles*.
> - The **TENSOR layer** is `model.DistComm` — a **real** cross-process collective that does
>   move bytes, but over **HOST float32**, and it is explicitly **NOT NCCL and NOT
>   multi-GPU**. Ranks here index tensor-parallel shards of **one** model.
>
> MPI is the **design lens and vocabulary** that tells us where these boundaries are and
> what to call them. It is **not** a claim that fak is MPI or inherits any HPC throughput,
> latency, message-rate, or wire-protocol property.

---

## The map

| MPI primitive | fak symbol | Layer |
|---|---|---|
| `MPI_Reduce` / `MPI_Allreduce` (the general fold) | [`modelroute.Combine`](https://github.com/anthony-chaudhary/fak/blob/main/internal/modelroute/modelroute.go) over the `Reduce*` set — `ReduceFirst` / `ReduceVote` / `ReduceBestOf` / `ReduceAllReduce` / `ReduceConcat` | **AGENT** (deterministic on structure only) |
| `MPI_Allreduce` (the *named* all-reduce) | [`modelroute.ReduceAllReduce`](https://github.com/anthony-chaudhary/fak/blob/main/internal/modelroute/modelroute.go) — weighted mean of the members' **scalar** outputs | **AGENT** (scalars, not tensors) |
| The live fan-out that *produces* the votes Combine folds | [`gateway.dispatchEnsemble`](https://github.com/anthony-chaudhary/fak/blob/main/internal/gateway/gateway.go) — N independently-adjudicated `Kernel.Syscall` calls in member order (#597) | **AGENT** (each member crosses the default-deny floor) |
| `MPI_Bcast` (the broadcast *bound*) | [`abi.ShareScope`](https://github.com/anthony-chaudhary/fak/blob/main/internal/abi/types.go) — `ScopeAgent` / `ScopeFleet` / `ScopeTenant` | **AGENT** (authorizes visibility, moves no bytes) |
| `MPI_Allreduce` (real cross-process sum) | [`model.DistComm.AllReduceSum`](https://github.com/anthony-chaudhary/fak/blob/main/internal/model/dist_collective.go) | **TENSOR** (real cross-process HOST float32) |
| `MPI_Allgather` (real cross-process gather) | [`model.DistComm.AllGather`](https://github.com/anthony-chaudhary/fak/blob/main/internal/model/dist_collective.go) | **TENSOR** (real cross-process HOST float32) |

---

## The AGENT layer — non-bit-exact, scope-bounded

This layer folds and bounds **agent** outputs. It is deterministic on **structure** (the
routing decision, the reduce order, the scope partition), never on the member text/scalar
a non-bit-exact engine produced.

### `modelroute.Combine` + the `Reduce*` set ≈ `MPI_Reduce` / `MPI_Allreduce`

`Combine(reduce, votes)` (`internal/modelroute/modelroute.go:559`) is the ensemble
**reduce**: it folds many members' outputs into one `Result` under a CLOSED, additive set
of reductions —

- `ReduceFirst` — first member's output (fastest-wins / fallback chain).
- `ReduceVote` — weighted-majority over discrete answers (self-consistency / quorum).
- `ReduceBestOf` — the highest-scored member (a judge/verifier picks).
- `ReduceAllReduce` — the weighted **mean** of the members' **scalar** outputs.
- `ReduceConcat` — concatenate the members' outputs (fan-out gather).

It is pure and deterministic: every tie is broken by a stable key, and the caller MUST
pass votes in `Plan.Members` order, so the same votes always fold the same way. That is
the MPI analogue's load-bearing honesty: like `MPI_Reduce`, the **fold** is deterministic
on its inputs — but the inputs (member answers) come from non-bit-exact engines, so
determinism is pinned to the decision and its reduce, **never** to end-to-end answer
reproducibility.

### `gateway.dispatchEnsemble` — the fan-out that produces the votes

`Combine` is the pure fold; the live dispatch that PRODUCES the votes is
`dispatchEnsemble` (`internal/gateway/gateway.go:1142`, issue #597). It runs each member
as its **OWN** independently-adjudicated kernel call — carrying that member's model in
`abi.ToolCall.Engine` — gathers the ALLOWED members' outputs in `Plan.Members` order, and
folds them with `modelroute.Combine`. The MPI-shaped invariant it honors: an ensemble
expands to **N independently-adjudicated `Kernel.Submit` calls**, never one fan-out that
bypasses the default-deny floor. A member bound for a REMOTE model still crosses the
residency/policy gate and is denied for a tenant/sensitive payload; on a full wipeout
(every member refused) it fails closed, surfacing the last refusal verdict rather than a
silent empty success.

### `abi.ShareScope` ≈ `MPI_Bcast` — but it is the broadcast *bound*, not a broadcast

`ShareScope` (`internal/abi/types.go:93`) is the CLOSED, additive isolation scope a shared
`Ref` carries:

| `ShareScope` | meaning | broadcast analogue |
|---|---|---|
| `ScopeAgent` | private to one agent (the fail-closed default) | not broadcast — rank-private |
| `ScopeFleet` | shareable across the fleet's trusted partition | the fleet broadcast bound |
| `ScopeTenant` | shareable within a tenant boundary | the tenant broadcast bound |

The default `ScopeAgent` is fail-closed (private): a value becomes visible to a wider
audience only when its scope is **explicitly widened**. This is the *bound* on a
broadcast, not the broadcast itself — widening a scope **authorizes** a later share, it
does not transport data. A one-sided write can never widen sharing past its `ShareScope`
(an `Accumulate` into a `ScopeFleet` window cannot publish at `ScopeTenant`); the default
stays `(TaintTainted, ScopeAgent)`.

---

## The TENSOR layer — real cross-process HOST float32 (NOT NCCL, NOT multi-GPU)

`model.DistComm` (`internal/model/dist_collective.go`) is the first **REAL**
cross-process collective on fak: a coordinator-rooted process group where each rank holds
**only its own part**, performing `AllReduceSum` / `AllGather` over a real wire, proven
byte-identical to the in-process default. Its ranks are tensor-parallel **host-float32
shards of ONE model** — a completely different rank space from the agent layer above.

This layer carries the load-bearing disclaimer for the whole epic. Quoted **verbatim**
from the package, this is the rank space that must be held apart from the agent layer:

> **`internal/model/dist_collective.go` (HONESTY), verbatim:**
>
> This is a cross-PROCESS collective over HOST float32 — it is NOT multi-GPU and
> is NOT NCCL. "Multi-GPU" stays unclaimable until a non-cpu-ref compute.CollectiveBackend
> (the NCCL/RCCL device backend) all-reduces a DEVICE tensor across 2 GPUs and matches
> cpu-ref on the GPU server. DistComm proves the distributed architecture above the device
> line; the device line is the next, GPU-node rung. Following the repo's own TCPTransport
> precedent, the gate runs the ranks as goroutines over a loopback socket — a genuine
> cross-process send, verifiable on one box.

---

## The all_reduce caveat — the same disclaimer at the agent layer

The agent-layer `ReduceAllReduce` borrows the distributed-systems *name* but is a scalar
reduce, not a tensor one. Quoted **verbatim** from the package
(`internal/modelroute/modelroute.go:221`):

> **`modelroute.ReduceAllReduce`, verbatim:**
>
> ReduceAllReduce numerically aggregates the members' SCALAR outputs into their
> weighted mean — the map-reduce / all-reduce form for numeric answers (a score,
> a count, a probability). It is NOT a tensor all-reduce: outputs that do not
> parse as a float are an error, not a silent guess. (Name borrows the
> distributed-systems term for the scalar reduce family; the scope is scalars.)

This is the borrow-the-term / disclaim-the-scope template the whole epic copies: take the
MPI *word*, then state exactly what scope it does and does **not** cover.

---

## What this is NOT

- **The agent rank space is not the tensor rank space.** `modelroute.Combine` /
  `dispatchEnsemble` / `ShareScope` index *agents and roles*; `model.DistComm` indexes
  *tensor-parallel shards of one model*. "fak has MPI collectives" is **not** a performance
  claim — the agent layer moves no tensor bytes, and the tensor layer is host float32, not
  a device collective.
- **`ReduceAllReduce` is a scalar mean, not a tensor reduce.** It folds numeric scalar
  answers (a score, a count, a probability); a non-numeric output is an error, never a
  silent guess.
- **`ShareScope` is a visibility bound, not a `MPI_Bcast`.** Widening a scope authorizes a
  later share; no collective transports the data at this layer.
- **`DistComm` is not multi-GPU.** It is cross-process HOST float32; "multi-GPU" stays
  unclaimable until a non-cpu-ref device `CollectiveBackend` all-reduces a DEVICE tensor
  across 2 GPUs and matches cpu-ref on the GPU server (see the verbatim disclaimer above).
- **No MPI/HPC number is borrowed.** MPI is the design lens and the vocabulary; fak
  inherits no MPI/HPC throughput, latency, message-rate, or wire-protocol property.

---

## See also

- [`internal/modelroute/modelroute.go`](https://github.com/anthony-chaudhary/fak/blob/main/internal/modelroute/modelroute.go) — `Combine` + the `Reduce*` set (the agent-layer reduce).
- [`internal/gateway/gateway.go`](https://github.com/anthony-chaudhary/fak/blob/main/internal/gateway/gateway.go) — `dispatchEnsemble`, the live N-submit ensemble fan-out (#597).
- [`internal/abi/types.go`](https://github.com/anthony-chaudhary/fak/blob/main/internal/abi/types.go) — `ShareScope`, the broadcast bound.
- [`internal/model/dist_collective.go`](https://github.com/anthony-chaudhary/fak/blob/main/internal/model/dist_collective.go) — `DistComm.AllReduceSum` / `AllGather`, the real cross-process tensor collective.
- [comm-as-mpi-split.md](comm-as-mpi-split.md) — the lane lease as `MPI_Comm_split`, `topobench` as `MPI_Cart_create` (sibling MPI-analogue doc).
- [model-routing.md](model-routing.md) — per-aspect + ensemble routing, the `fak route` surface the reduce sits behind.
- [explainers/multi-gpu-tensor-parallelism.md](explainers/multi-gpu-tensor-parallelism.md) — the tensor-parallel path, the device-collective HAL seam, and the exact NCCL/RCCL swap-in point `DistComm` sits below.
- [explainers/vdso-revoke-as-comm-revoke.md](explainers/vdso-revoke-as-comm-revoke.md) and [proofs/async-addressing.md](proofs/async-addressing.md) — sibling MPI-analogue docs in the same epic (#639).
