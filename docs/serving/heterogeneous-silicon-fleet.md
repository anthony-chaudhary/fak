---
title: "Heterogeneous silicon fleets - one agent-kernel control plane"
description: "Public reference architecture for exposing CUDA, Vulkan, vendor accelerators, and future fak-certified backends through one agent-kernel control plane."
---

# Heterogeneous silicon fleets

This page is the public entry point for neo-cloud teams searching for
`heterogeneous accelerator agent kernel`, `neo-cloud agent serving`,
`bring-your-accelerator agent serving`, or `vendor-neutral inference backend`.

The goal is one control plane over many backend groups. The control plane owns
policy, audit, context, cache provenance, route decisions, and support evidence.
Each backend group owns device execution behind `internal/compute.Backend`.

## Reference shape

```text
agent clients
    |
    v
fak gateway / policy / audit / context / cache scoring
    |
    v
backend router
    |
    +-- cpu-ref reference floor
    +-- cuda backend group
    +-- vulkan backend group
    +-- vendor backend group
```

The control-plane invariant is that every route can be explained without trusting
a vendor string alone.

## Route evidence

| Field | Example |
|---|---|
| Backend id | `cuda`, `vulkan`, `vendor-x` |
| Correctness class | `Reference` or `Approx` |
| Required caps | `DeviceMemory`, `CapacityProbe`, `Collective` |
| Seen caps | Exact `Caps` returned by the backend |
| Capacity | total/free bytes, or `unknown` when no probe is advertised |
| Conformance | `passed`, `partial`, or `not-yet` until the conformance kit exists |
| Fallback | `cpu-ref` or another named backend |
| Route reason | `fits`, `missing-cap`, `over-capacity`, `no-conformance`, `policy-block` |

This route row is the fleet analogue of fak's local refusal vocabulary: a route
is a claim only when it carries the evidence needed to replay or reject it.

## First smoke

A useful heterogeneous smoke is small:

1. Run `cpu-ref` and one non-reference backend group behind the same gateway.
2. Serve one small model shape on both groups.
3. Require `cmd/modelbench -backend <name> -require-non-reference` for the
   accelerator route.
4. Record the approximate parity witness against `cpu-ref`.
5. Force the accelerator route unavailable and prove fallback is visible.

That proves route selection, parity, and fallback disclosure. It does not prove
optimal fleet scheduling.

## Honest gaps

| Gap | Current stance |
|---|---|
| Cross-vendor collectives | Start with homogeneous backend groups. Cross-backend reduction should fail closed until collective evidence exists. |
| Tenant policy distribution | Keep tenant scope in gateway policy, audit, and cache metadata; do not share one tenant's cache facts across another tenant. |
| Fleet capacity scheduling | `CapacityProbe` reports local memory facts; a scheduler still has to consume them and treat unknown as unknown. |
| `fak-certified` mark | Planned with the Backend Conformance Kit; use `passed`, `partial`, or `not-yet` until the kit ships. |

For the backend support vocabulary, see
[supported silicon backends](../supported/silicon-backends.md). For the
low-level seam, see
[hardware portability via the compute HAL](../explainers/hardware-portability.md).
