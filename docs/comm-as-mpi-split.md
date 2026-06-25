---
title: "fak comm: the lane lease as MPI_Comm_split, topobench as MPI_Cart_create (search, not declare)"
description: "Maps fak's shipped coordination surfaces onto MPI communicator primitives — the dos-arbitrate lane lease as MPI_Comm_split (color = lane, disjoint file-tree = no overlap, enforced by refusal), abi.ShareScope as the communicator isolation scope, and topobench's TopologyGenome search as the MPI_Cart_create/Graph_create counterpart that OPTIMIZES a shape where agenttopo DECLARES one — with the honest line that no bytes move at this layer."
---

# comm: the lane lease as a communicator split

This maps three **shipped** fak coordination surfaces onto the MPI communicator
primitives they are shaped like, so the analogy is documented in one place instead of
inferred from terse inline comments. It is part of the MPI-shaped message-passing epic
(#639).

> **Honesty caveat (read first).** This is structural analogy, **not** a message-passing
> transport. A lane lease is a *coordination decision enforced by refusal* (two overlapping
> leases serialize; one is denied), deterministic and `dos`-verifiable — **no collective
> moves bytes between agents at this layer.** `topobench`'s "savings" are MEASURED replay
> token counts capped at the corpus divergence frontier, never an HPC throughput/latency
> number. And the agent-layer rank/size here is **not** the tensor-layer rank/size of
> `internal/model`'s `DistComm` serving collective — see "What this is NOT".

---

## 1. `dos arbitrate` + `dos.toml [lanes]` ≈ `MPI_Comm_split`

`MPI_Comm_split(comm, color, key)` partitions a communicator: every rank passing the same
`color` lands in the same sub-communicator, disjoint from the others. fak's lane lease is
the same partition, decided by **admission** rather than rendezvous:

- **color = the lane.** `dos.toml [lanes]` (`dos.toml:32`) declares the lane taxonomy, and
  `[lanes.trees]` (`dos.toml:135`) gives each lane its canonical, prefix-disjoint file
  tree. A worker requests a lane; `dos arbitrate` admits it iff its tree is disjoint from
  every live lease (PARTITION.md: "one lease per tree").
- **the split is enforced by refusal.** Where `MPI_Comm_split` *places* a rank, the
  arbiter *admits or denies* one: two workers whose trees intersect cannot both hold an
  exclusive lease, so they **serialize** (one waits) instead of colliding. The partition is
  a decision the kernel can refuse, not a barrier all ranks must reach — which is why a
  single absent worker never deadlocks the others.
- **deterministic and verifiable.** The admission is a pure function of the requested lane,
  mode, tree, and the live-lease set (the `dos arbitrate` kernel), so the same inputs yield
  the same verdict — `dos`-checkable, not a timing-dependent race.

The load-bearing difference from `MPI_Comm_split`: a communicator split is a collective all
ranks call together; a lane lease is an **asymmetric admission** one worker requests and the
arbiter grants or refuses. No bytes cross between the workers in the same lane — the lease
coordinates *who may write which files*, it does not transport a message.

## 2. `abi.ShareScope` ≈ the communicator isolation scope

`abi.ShareScope` (`internal/abi/types.go:93`) is the CLOSED, additive isolation scope a
shared resource (a `Ref`) carries — the analogue of *which communicator a name is visible
within*:

| `ShareScope` | meaning                                              | communicator analogue                    |
|--------------|------------------------------------------------------|------------------------------------------|
| `ScopeAgent` | private to one agent (the fail-closed default)       | a rank-private buffer (no communicator)  |
| `ScopeFleet` | shareable across the fleet's trusted partition       | the fleet communicator                   |
| `ScopeTenant`| shareable within a tenant boundary                   | a tenant sub-communicator                 |

The default `ScopeAgent` is fail-closed (private): a value is shared only when its scope is
explicitly widened, the same discipline that keeps a name from leaking across a
communicator boundary it was never published to. This is *visibility scope*, not a
broadcast — widening a scope authorizes a later share, it does not move data.

## 3. `topobench` `TopologyGenome` ≈ `MPI_Cart_create` / `MPI_Graph_create` — but it SEARCHES

`MPI_Cart_create` / `MPI_Graph_create` *impose* a topology a caller declares. fak has both
halves, and the distinction is the point of this section:

- **DECLARE — `internal/agenttopo`.** `agenttopo` declares a *named, validated DAG* over a
  `comm.Group`: who may hand a result to whom, every edge endpoint checked against the
  group, cycles refused, declaration order preserved (`internal/agenttopo/doc.go`). This is
  the direct `MPI_Graph_create` analogue: a fixed adjacency the caller asserts.
- **SEARCH — `cmd/topobench` + `turnbench.TopologyGenome`.** `topobench` does NOT declare a
  shape; it **optimizes an anonymous one**. A `TopologyGenome` is three searchable levers —
  `Width` (fan-out), `SubTurns` (orchestrator→worker depth), `Lanes` (the leaf-lane
  partition the workers serialize on) (`internal/turnbench/toposearch.go:92`) — and the
  search ranks genomes by `CreditedSavingsTokens`, the measured prefix-reuse + dedup token
  saving **capped at the corpus frontier width** (`min(Width, Wmax)`), so an extrapolated
  topology can never be crowned by a saving the corpus never recorded.

So: `agenttopo` is "here is my graph" (`MPI_Graph_create`); `topobench` is "find me a good
graph, and prove the win is measured, not extrapolated" — the search counterpart to a
declared topology. The "savings" are **token counts** off `RunFanoutCell`'s measured
halves, capped at the divergence frontier — not an HPC metric.

---

## What this is NOT

- **No bytes move at this layer.** A lane lease serializes writers; a `ShareScope` authorizes
  a later share; a `TopologyGenome` ranks fleet shapes by measured token savings. None of
  the three transports a message between agents — there is no collective moving data here,
  unlike a real `MPI_Bcast`/`MPI_Allreduce`.
- **Agent rank/size ≠ `DistComm` rank/size.** `internal/model`'s `DistComm` (the serving
  collective in `internal/model/dist_collective.go`) carries real cross-process host
  float32 between *devices*; its rank/size index GPUs/workers in a tensor collective. The
  agent-layer "rank" here indexes *agents/roles* in a coordination graph. They are different
  ranks over different objects — do not conflate the two.
- **`pipelinegen` is the tensor/device layer, not this one.** `cmd/pipelinegen` runs fak's
  NATIVE engine across pipeline-parallel transformer stages (`internal/model/pipeline.go`):
  hidden state crossing a serialize→bytes→deserialize boundary between layer-bands, the
  stand-in for a real NCCL/RPC worker hop. That is a *model-internal serving* demonstrator
  (device collectives), **not** an agent-topology or communicator-split surface. The
  agent-layer mappings above (`agenttopo`/`topobench`/lane lease) are unrelated to it; they
  inherit no HPC latency, throughput, or progress guarantees from the tensor layer.

---

## See also

- [`dos.toml`](https://github.com/anthony-chaudhary/fak/blob/main/dos.toml) — the `[lanes]` taxonomy and `[lanes.trees]` partition the arbiter splits on.
- [PARTITION.md](https://github.com/anthony-chaudhary/fak/blob/main/PARTITION.md) — "one lease per tree": the disjoint-tree rule behind the split.
- [`internal/agenttopo`](https://github.com/anthony-chaudhary/fak/blob/main/internal/agenttopo/doc.go) — the DECLARED agent-topology DAG (the `MPI_Graph_create` half).
- [`internal/turnbench/toposearch.go`](https://github.com/anthony-chaudhary/fak/blob/main/internal/turnbench/toposearch.go) — `TopologyGenome` / `TopologyFitness`, the SEARCH half.
- [vdso-revoke-as-comm-revoke.md](explainers/vdso-revoke-as-comm-revoke.md) and [proofs/async-addressing.md](proofs/async-addressing.md) — sibling MPI-analogue docs in the same epic.
