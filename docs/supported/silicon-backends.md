---
title: "Supported silicon backends - fak backend conformance and accelerator portability"
description: "Public landing page for fak's vendor-neutral inference backend path: compute.Backend, Caps, correctness classes, backend conformance, and accelerator support vocabulary."
---

# Supported silicon backends

This page is the public entry point for accelerator and chip teams searching for
`fak backend conformance`, `fak-certified backend`, `neo-silicon agent kernel`,
`bring-your-accelerator agent serving`, or `vendor-neutral inference backend`.

The short version: fak has a binding layer for model execution. A new accelerator
does not fork the agent kernel; it implements `internal/compute.Backend`, registers
a backend name, advertises exact `Caps`, and proves its correctness class against
the reference floor.

## What is shipped

| Surface | Status |
|---|---|
| `internal/compute.Backend` | Shipped whole-op interface for model execution. |
| `Caps` | Shipped optional capability vocabulary: async, fused attention, graph compile, dtype upload, device memory, collectives, and capacity probes. |
| `cpu-ref` | Shipped `Reference` backend and correctness floor. |
| Device backends | CUDA and Vulkan are documented with hardware witnesses in the HAL explainer. |
| Non-reference gate | `cmd/modelbench -backend <name> -require-non-reference` fails closed when a run silently falls back to CPU. |

The detailed contributor-facing seam is documented in
[hardware portability via the compute HAL](../explainers/hardware-portability.md).

## What is not yet a mark

`fak-certified backend` is the intended public mark, but it should not be used as
a claim until the Backend Conformance Kit lands. Until then, support rows should
use plain evidence labels:

| Label | Meaning |
|---|---|
| `Reference` | Exact-reference rungs are green. |
| `Approx` | Approximate parity gates, such as argmax and logit cosine, are the right witness. |
| `FENCED` | The backend has a named seam, cap, or support row that bounds the claim. |
| `PARTIAL` | Some methods, model shapes, or caps work, and the gaps are listed. |
| `not-yet` | The path is designed or planned but lacks a witness. |

## Vendor-neutral support row

A backend support row should name the facts a router and a customer can actually
use:

| Field | Why it matters |
|---|---|
| Backend id | Stable selector used by the registry and by route logs. |
| Correctness class | Prevents an approximate accelerator from being judged as byte-identical reference. |
| Caps | Says what the backend implements, not what the device family can do in theory. |
| Hardware and runtime | Makes measurements reproducible. |
| Model shape | Keeps one model's witness from becoming a universal claim. |
| Capacity | Real bytes only when `CapacityProbe` or `HostCapacityProbe` is advertised. |
| Parity witness | Shows the backend was compared to the `cpu-ref` floor. |
| Fallback | Names when and where a route falls back. |

## Why this matters for agent serving

Agent serving is not only throughput. A backend sits under policy, audit, tool-call
admission, context, cache provenance, and refusal reasons. The value of the seam is
that a silicon vendor can focus on kernels and runtime evidence while fak keeps the
agent-kernel boundary stable.

For cloud operators building heterogeneous fleets, see
[heterogeneous silicon fleets](../serving/heterogeneous-silicon-fleet.md).
