---
title: "Neo-cloud reference architecture - one control plane for heterogeneous accelerator pools"
description: "Reference architecture for exposing bring-any-silicon backend pools through fak's agent-kernel binding layer."
---

# Neo-cloud reference architecture - one control plane for heterogeneous accelerator pools

**Audience:** neo-cloud operators, accelerator marketplaces, infra teams running
mixed GPUs, NPUs, XPUs, and experimental silicon.
**Goal:** expose many accelerator backends through one agent-kernel control plane
without pretending all hardware is the same.

## Status fence

This is a reference architecture, not a shipping multi-vendor fleet product. The
shipped pieces are the compute HAL seam, backend registry, policy/audit gateway,
and the existing backend witnesses documented in
[hardware portability](../explainers/hardware-portability.md).

The open pieces are called out below: backend conformance (#1684), scaffolding
(#1685), hardware-shape-neutral scorecard rows (#1688), and fleet scheduling
evidence across heterogeneous devices.

## Shape

```text
agent clients
    |
    v
fak gateway / policy / audit / context
    |
    v
backend router
    |
    +-- backend group: cpu-ref reference floor
    +-- backend group: cuda approximate devices
    +-- backend group: vulkan approximate devices
    +-- backend group: vendor <name> approximate devices
```

The control plane owns admission, policy, audit, context, cache scoring, and
the backend selection contract. Backend groups own device execution and expose
only the facts the control plane can safely use: backend name, correctness
class, caps, capacity reports, conformance status, and measured model support.

## Required planes

| Plane | Responsibility | Must not overclaim |
|---|---|---|
| Admission and policy | Decide whether an agent effect may run before any backend work starts. | Hardware speed never bypasses tool governance. |
| Backend inventory | List registered backends, correctness class, caps, model shapes, runtime version, and evidence date. | A device family capability is not a backend capability. |
| Capacity | Use `DeviceCapacity` and `HostCapacity` only when the backend advertises the matching caps. | Unknown capacity is not infinite capacity. |
| Routing | Pick backend groups by model, cap needs, correctness class, tenancy, and current load. | A CPU fallback must be visible; it cannot masquerade as accelerator execution. |
| Cache and context | Preserve provider/engine/local cache provenance and O(1) context fault semantics. | Provider warmth is cost telemetry, not proof of local reuse. |
| Audit | Record the backend id, route decision, policy verdict, and witness row. | Do not log raw secrets or raw model context just to prove a route happened. |

## Heterogeneous routing contract

A route decision should be explainable in one row:

| Field | Example |
|---|---|
| `model` | `SmolLM2-135M` |
| `backend` | `cuda`, `vulkan`, `vendor-x` |
| `class` | `Approx` |
| `caps_required` | `DeviceMemory`, `CapacityProbe` |
| `caps_seen` | from `Backend.Caps()` |
| `capacity` | total/free bytes when probe is present, else `unknown` |
| `conformance` | `passed`, `partial`, or `not-yet` |
| `fallback` | `cpu-ref` or another named backend |
| `reason` | `fits`, `missing-cap`, `over-capacity`, `no-conformance`, `policy-block` |

This row is the fleet version of fak's local honesty rule: a route is a fact only
when it is backed by a witness the backend did not merely narrate.

## Smoke design

The first multi-backend smoke should stay smaller than the final product:

1. Run one reference group (`cpu-ref`) and one non-reference group behind the same
   gateway.
2. Serve the same small model shape on both groups.
3. Route a known prompt to the non-reference group only when
   `-require-non-reference` can prove the accelerator path was used.
4. Re-run the prompt on `cpu-ref` and record the approximate parity witness.
5. Force the non-reference group unavailable and prove the route falls back with a
   visible reason, not a silent success label.

That smoke proves the control plane can select hardware, keep the cold path
correct, and disclose fallback. It does not prove optimal scheduling.

## Known gaps

| Gap | Why it matters | Honest next step |
|---|---|---|
| Cross-vendor collectives | Tensor-parallel collectives usually require one communicator and one tensor ownership model. | Start with homogeneous backend groups; require `CollectiveBackend` evidence before cross-rank routes. |
| Per-tenant policy distribution | A neo-cloud control plane needs tenant-specific policy without leaking one tenant's cache, route, or audit data to another. | Keep tenant scope in the gateway and cache metadata; test cross-tenant denial before multi-tenant launch. |
| Fleet capacity scheduling | `CapacityProbe` gives local memory facts, not a full scheduler. | Build a scheduler that consumes capacity rows and treats unknown as unknown. |
| Backend conformance mark | Operators need a simple support row, but the mark is not live yet. | Use `passed/partial/not-yet` until #1684 defines `fak-certified`. |
| Benchmark catalog vocabulary | Mixed hardware needs neutral names for control-plane host, model worker, backend group, and accelerator. | Use the terms below consistently in benchmark docs and scorecards. |

## Benchmark vocabulary

Use neutral roles rather than vendor names as architecture terms:

| Term | Meaning |
|---|---|
| Control-plane host | The process running gateway, policy, audit, routing, and score aggregation. |
| Model worker | A process that owns one model instance and one selected backend. |
| Backend group | A homogeneous set of workers using the same backend id, driver/runtime, and support row. |
| Accelerator pool | The cloud inventory behind one or more backend groups. |
| Reference floor | `cpu-ref`, used for correctness witnesses and fallback disclosure. |
| Approx backend | A backend judged by approximate parity gates such as argmax and logit cosine, not byte identity. |

## Operator decision

The near-term neo-cloud decision is not "standardize every accelerator." It is:

> Require every accelerator lane to bind through the same backend contract and
> publish the same route, capacity, cap, conformance, and fallback evidence.

That lets a cloud add hardware without re-arguing the agent governance layer each
time a new device appears.
