---
title: "Multi-node compute: the runnable witness and the path to high performance"
description: "fak cluster runs a real cross-node collective over the DistComm process group on any two CPU hosts today; this doc is the rung ladder from that host-layer witness to GPU-speed multi-node serving."
---

# Multi-node compute — what runs today, and the path to high performance

fak's tensor-parallel and pipeline-parallel seams have been bit-exact on CPU for a
while, and the cross-process collective (`DistComm`) has been proven byte-identical to
the in-process reference. But that proof lived entirely inside a test that spins the
ranks as goroutines over a loopback socket on one box. There was no command an operator
could launch on two *separate* machines.

`fak cluster` is that command. It is the first runnable witness that fak compute crosses
a real machine boundary — two laptops on a LAN, two cloud VMs, two GPU boxes — not a
simulation inside one process.

## Run it on two machines

The host-layer collective is CPU-only, so any two hosts work. On node A (the
coordinator), and node B (a worker), run the fak binary built from this repo:

```bash
# node A — bind a port, wait for the workers, hold rank 0's vector:
fak cluster coordinator --listen 0.0.0.0:7777 --size 2 --vec 1,2,3

# node B — dial node A, join as rank 1, hold rank 1's vector:
fak cluster worker --coord A.B.C.D:7777 --rank 1 --size 2 --vec 4,5,6
```

Both nodes print `5,7,9` — the element-wise sum, reduced across the wire. Every rank
holds only its own `--vec`; the sum is computed by the `DistComm` process group, not by
one node that already had all the data. `--op allgather` (with a shared `--widths`
tiling) instead concatenates each rank's shard in rank order. Scale past two by giving
every node the same `--size` and a distinct `--rank`.

Before you touch a second box, prove the path on one:

```bash
fak cluster selftest
# PASS: cross-process allreduce + allgather bit-exact vs LocalCollective for sizes 1..4 (max|Δ|=0)
```

`selftest` runs the ranks over a loopback socket and asserts each one's result is
byte-for-byte equal to the in-process `LocalCollective`. It is the same wire codec and
the same orchestration the two-node launch uses; only the address differs from a real
interface. A regression in the reduction order, the frame format, or the rank placement
fails it at `max|Δ|=0`.

## What this proves — and what it does not

`fak cluster` is a cross-**process**, cross-**node** collective over **host float32**.
It exercises the distributed plumbing that real serving stands on: rank coordination, a
framed wire protocol, rank-order reduction across machines, and the fail-closed contract
(a ragged reduce, a mis-width gather, or a process-group op desync is refused on every
rank without deadlocking a peer). All of that runs today on commodity CPUs.

It is **not** multi-GPU and **not** NCCL. The bytes move over TCP between Go processes,
the reduction runs on the host, and the collective is the cpu-reference one. Pointing the
same seam at a device-tensor collective on real GPUs is a separate rung (below), and
"multi-GPU" stays unclaimable until that rung is witnessed on hardware. This doc and the
command are honest about which line they sit above.

## The rung ladder to high performance

Each rung swaps one implementation in behind a seam the rung below already proved, so the
correctness gates carry forward and only the new piece is under test. The authoritative
sequencing lives in [dual-track serving](dual-track-serving-plan.md) and
[`THROUGHPUT-TRUST-SHARED-SPINE`](../notes/THROUGHPUT-TRUST-SHARED-SPINE-2026-06-24.md);
this is the compute-data-plane slice of it.

| Rung | What it adds | Seam it plugs into | Status / issue |
|---|---|---|---|
| **0. Host collective across nodes** | `fak cluster` — AllReduce/AllGather over TCP, each rank holds its own part | `model.DistComm` | **runnable today** (this doc) |
| **1. Map the shipped analogues** | name DistComm allreduce / Combine reduce / Scope bcast as the MPI-shaped set | — | #639, #652 |
| **2. Wire ForwardTP per-rank across DistComm** | run the tensor-parallel forward as N processes, each holding only its shard, reducing over the wire (today `ForwardTP` runs every rank in one process via `LocalCollective`) | `model.Collective` → per-rank `DistComm` | native TP (#25), scope-gated |
| **3. Band-running pipeline worker** | replace the `EchoFrames` peer with a worker that loads a layer band and runs `ForwardBand`, so `TCPTransport` drives a real distributed pipeline | `model.StageTransport` | #85, #30 |
| **4. KV → bytes byte mover** | serialize a `KVCache` and ship it between nodes (the P/D disaggregation data plane) | `StageTransport` (TCP-first, then RDMA/UCX) | #29 |
| **5. Device CollectiveBackend** | NCCL/RCCL all-reduce of a **device** tensor across 2+ GPUs, bit-exact vs cpu-ref | `compute.CollectiveBackend` | hardware-gated (#305, #706); needs a GPU bench node (#12, #18) |
| **6. Health/cache-aware routing** | turn the round-robin `ReplicaRouter` into residency- and health-aware placement across remote replicas | `gateway.ReplicaRouter` | #41, #42 (ride path) |

Rung 0 is the floor the rest stand on: the wire protocol, rank coordination, and the
fail-closed contract are now an operable command rather than a test fixture. Rungs 2–4 are
CPU-runnable and turn "a collective crosses nodes" into "a model forward crosses nodes."
Rung 5 is the only piece that genuinely needs GPUs, and it is a backend swap behind the
collective seam rung 0 already proved — not a redesign.

## Re-verify

```bash
go test ./cmd/fak/ -run TestCluster              # the selftest + parsing + fail-closed gates
go test ./internal/model/ -run 'DistComm|Pipeline|TP'   # the bit-exact collective/pipeline witnesses
fak cluster selftest                             # the operator-facing loopback proof
```

The two-node run is the same code with a real address in `--coord`. If both nodes print
the same vector and it equals the reduction of the inputs, the kernel computed it across
the machine boundary.
