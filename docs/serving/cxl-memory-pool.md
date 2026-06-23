# Multi-tenant CXL memory pool: fleet reuse + a cross-tenant trust gate

> Status: the **policy/metadata plane** is shipped and tested (`internal/cachemeta/pool.go`,
> `cmd/cxlpooldemo`). It decides *what a shared pooled tier is worth across a fleet* and
> *who may reuse a pooled cell* — and emits no bytes. The physical CXL.mem map / RDMA
> transfer is performed by the engine adapter that consumes the placement directives;
> this plane touches no KV tensor. The demo numbers are a deterministic cost model over
> representative (or operator-supplied) tier profiles, not a hardware measurement.

This is the multi-tenant counterpart of the [hardware-aware cache](hardware-aware-cache.md)
plane. That layer reasons about one host's ladder and demotes a hot prefix one tier
colder instead of evicting it. This layer answers the question a *shared* memory fabric
turns on: when the **same** hot prefix is wanted by a whole fleet of tenants, what is a
switch-pooled tier actually worth, and who is allowed to reuse a cell in it?

## The three pooling regimes — `pool.go`

A residency tier has a `PoolProfile` describing its pooling character: how many `Hosts`
can attend it, whether it is `Coherent`, and the zero-copy `ShareKind` across the fabric.
Two derived properties drive the economics:

- **`Reachable()`** — a cell can be reached by a host other than the one that wrote it
  (`Hosts > 1`): a shared pool, not host-private memory.
- **`FabricShareable()`** — one resident copy is attendable zero-copy by every host in
  the pool (`Hosts > 1` **and** coherent **and** a real zero-copy share kind).

`PlanFleetReuse` computes, for one hot prefix wanted by *N* tenants, the three regimes
side by side:

| Regime | Prefill | Resident copies | Saves |
|---|---|---|---|
| host-private (unreachable) | N× (each tenant rebuilds) | N | nothing — the baseline |
| reachable, copy-only (e.g. RDMA) | 1× (owner builds, others stage a copy) | N | the re-prefill |
| **coherent, zero-copy (CXL.mem)** | **1×** | **1** | **both axes** |

Only a coherent, zero-copy pool collapses *N* prefills **and** *N* copies into one — the
reason a shared memory fabric exists. The choice is made purely from the `PoolProfile`,
so plugging in a real fabric's topology changes the verdict, not the code.

## The cross-tenant trust gate — `PoolReuseVerdict`

A shared address space is not a license to alias bytes. Before a tenant may attend a
pooled cell another tenant wrote, `PoolReuseVerdict` checks, in order, and fails closed
on any miss:

1. **not poisoned** — a quarantined cell leaves the pool, never re-served;
2. **declared shareable** — the producer marked it beyond the private `ScopeAgent` default;
3. **adjudicated trusted** — only `TaintTrusted` bytes may cross a tenant boundary;
4. **exact key match** — same model / tokenizer / serializer / position / policy /
   admitter (a KV span built under one model is garbage under another); an incomplete
   key fails closed.

Every failure is a typed, non-serveable verdict, so pooled dedup is *honest* — never a
blind alias of a mismatched or poisoned cell. This is the layer the rack-scale academic
CXL KV caches leave out: they assume mutual trust, with no attribution, revocation, or
quarantine.

## What you can run

```
go run ./cmd/cxlpooldemo
```

A no-model, no-GPU, deterministic proof: the pool topology (which tier is
fabric-shareable), the three-way fleet economics for 8 tenants sharing one 4000-token
prefix (a coherent CXL pool saves **28000 prefill tokens** and **448 MB** of copies), and
the trust gate refusing a poisoned / private / wrong-model cell.

To compute the same economics over **your** fabric's measured numbers, pass a calibration
file:

```
go run ./cmd/cxlpooldemo -profiles cmd/cxlpooldemo/calibration.example.json
```

The calibration overrides only the tiers/fields you measured (latency, bandwidth,
capacity, host count); everything else keeps its representative default. It is still a
cost model over the supplied profiles — fak moves no bytes, and this is not a measurement
of fak on any particular hardware.

## Honest boundaries

- The plane is **payload-free**: `PlanFleetReuse` and `PoolReuseVerdict` are pure
  functions; the demo emits no `KVTransfer` and moves no KV. The physical pooling (a CXL
  HDM map, an RDMA copy) is the engine adapter's job.
- The demo numbers are a **mechanism proof** — a count of prefill tokens and copies a
  blind-LRU baseline would spend versus the tiered/pooled policy — not a throughput, TPS,
  or hardware result. fak's industry scorecard rates multi-tier KV *offloading* (the
  byte-moving half) as a no-claim head-to-head for exactly this reason: fak owns the
  placement-and-trust decision, not the data movement.
- `FabricShareable` and the savings are **host-count and coherence driven**, not
  latency/bandwidth driven: a coherent multi-host pool collapses the copies regardless of
  how fast it is; the latency/bandwidth profile only sets the per-tenant attend/stage
  cost.
