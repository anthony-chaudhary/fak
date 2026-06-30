---
title: "Neo-silicon internal pilot pitch - fak for accelerator vendors"
description: "A one-page pitch for chip and accelerator teams evaluating fak as a vendor-neutral agent-kernel binding layer."
---

# Neo-silicon internal pilot pitch - fak for accelerator vendors

**Audience:** accelerator product, compiler/runtime, ecosystem, developer
relations, and cloud partnerships.
**Ask:** run a one-day pilot that registers one accelerator backend behind
`internal/compute.Backend`, proves it did not fall back to CPU, and publishes an
honest support row.

## Why this is attractive to a chip maker

Agent workloads are not just matmuls. They also need tool governance, audit,
tenant policy, context management, cache provenance, refusal reasons, and a
fallback story. A chip team can build all of that, or it can bind into a kernel
that already treats those surfaces as first-class.

| Vendor need | fak binding-layer value |
|---|---|
| Show useful agent workload support | The backend plugs into an agent kernel, not only a standalone benchmark. |
| Avoid per-agent integration forks | The same backend contract sits under CLI, gateway, policy, and future conformance surfaces. |
| Be honest about numerics | `Reference` and `Approx` are typed classes with different evidence. |
| Avoid CPU-fallback ambiguity | `-require-non-reference` makes silent fallback a failed run. |
| Explain memory limits | `CapacityProbe` and `HostCapacityProbe` distinguish real capacity from unknown. |
| Give clouds a route row | Backend name, caps, capacity, correctness class, fallback, and conformance status can be scored uniformly. |

## One-day pilot

| Step | Evidence |
|---|---|
| Register backend | `compute.Register` exposes a stable backend name. |
| Implement hot path | `MatMul`, `Attention`, and `Argmax` run on the accelerator for one supported model shape. |
| Keep the floor | Unsupported ops delegate or fail visibly; `cpu-ref` remains the reference witness. |
| Prove non-reference | `cmd/modelbench -backend <name> -require-non-reference` refuses CPU-only fallback. |
| Publish support row | Hardware, driver/runtime, model, caps, correctness class, parity result, throughput, and gaps are recorded. |

The pilot succeeds only if both facts are true: the accelerator path was used,
and the parity/support row names what is still not certified.

## The support row vendors should want

| Field | Example |
|---|---|
| Backend id | `vendor-x` |
| Correctness class | `Approx` |
| Hardware | device SKU, memory, interconnect |
| Runtime | driver/compiler/runtime versions |
| Model shape | model name, dtype, context, batch/decode geometry |
| Caps | exact `Caps` values advertised by the backend |
| Capacity | reported bytes if probe is present, else `unknown` |
| Parity | argmax/cosine or reference rungs used |
| Fallback | named fallback and the condition that triggers it |
| Status | `FENCED`, `PARTIAL`, or `not-yet`; not `fak-certified` until the kit exists |

## Residual risks to name up front

- The conformance kit and `fak-certified` mark are not shipped yet (#1684).
- The backend scaffold generator is not shipped yet (#1685).
- Cross-vendor collectives should start as homogeneous backend groups until the
  collective evidence exists.
- Capacity probes report local facts; they are not a fleet scheduler by
  themselves.
- A benchmark win is not reportable unless it is net-true: real baseline, real
  cost, reproducible witness, and scope stated.

## Decision the vendor can make

The narrow decision is:

> Treat `compute.Backend` as the binding target for agent-serving silicon, and
> publish support rows in fak's evidence vocabulary while the conformance mark is
> being built.

That gives cloud partners and agent products a way to compare accelerators
without letting generic benchmark claims replace route-level evidence.
