---
title: "Neo-silicon onboarding - make an accelerator useful to fak agents"
description: "Vendor-facing guide for adding a compute backend behind fak's agent-kernel binding layer."
---

# Neo-silicon onboarding - make an accelerator useful to fak agents

**Audience:** accelerator vendors, backend authors, compiler/runtime teams.
**Goal:** turn a new device into a fak backend without forking the agent kernel,
the tool-call policy layer, or the model forward pass.

## Status fence

The binding seam is shipped. `internal/compute.Backend` is the interface, `Caps`
advertises optional behavior, and the registry lets a host pick a backend by
name. The repo already carries the CPU reference floor plus device backends
documented in [hardware portability](../explainers/hardware-portability.md).

The product packaging is not complete yet. The Backend Conformance Kit (#1684)
and `fak backend scaffold <name>` generator (#1685) are the intended vendor
entry points. Until those land, this guide names the contract and the evidence a
backend will need.

## The contract to implement

`compute.Backend` is a whole-op interface. It is deliberately small enough for a
vendor team to own, and high-level enough that fak does not need to know whether
the device is a GPU, NPU, XPU, dataflow chip, or host CPU extension.

| Method group | Methods | Vendor responsibility |
|---|---|---|
| Identity and evidence | `Name`, `Tier`, `Class`, `Caps` | Give the backend a stable id, expose private device tiering, and declare whether it is `Reference` or `Approx`. |
| Residency and dtype | `Upload`, `Host`, `Read`, `Free`, `NewKV` | Move tensors to the device, narrow dtype only when advertised, fence reads, release storage, and keep KV state behind the seam. |
| Core math | `MatMul`, `BatchedMatMul`, `RMSNorm`, `RoPE`, `SwiGLU`, `AddInPlace`, `AddBias` | Lower the model's whole ops to native kernels without changing the forward loop. |
| Attention and decode | `Attention`, `Argmax` | Keep attention and greedy token selection on device when possible; `Read` is the explicit host fence. |

Optional extensions are discovered by type assertion and by `Caps`:

- `CollectiveBackend` adds `AllReduceSum`, `AllGather`, `ReduceScatter`, and
  `AllToAll` for tensor-parallel device tensors.
- `RankUploader` uploads a tensor to a specific rank for multi-device layouts.
- `CollectiveInitializer` bootstraps a collective world before advertising it.
- `DeviceCapacity` and `HostCapacity` report finite memory only when
  `Caps.CapacityProbe` or `Caps.HostCapacityProbe` says the report is real.

## What `Caps` means

`Caps` is a promise, not a marketing label. Leave a capability false until the
backend can satisfy it under the conformance harness.

| Capability | Meaning when true |
|---|---|
| `Async` | Backend methods may enqueue work; `Read` and `Argmax` are the host fences. |
| `FusedAttn` | `Attention` lowers to a fused attention primitive rather than the reference decomposition. |
| `FusedFFN` | Feed-forward subgraphs can fuse norm, gate/up, activation, down, and residual work. |
| `GraphCompile` | The backend can consume a recorded op list or graph as a static device program. |
| `UploadDtype` | `Upload(t, as)` honors dtype narrowing instead of ignoring `as`. |
| `DeviceMemory` | Resident tensors are not host-addressable; `Host` returns `(nil, false)`. |
| `Collective` | The backend implements the collective extension over device tensors. |
| `CapacityProbe` | Device memory total/free bytes are reported by the backend, not guessed. |
| `HostCapacityProbe` | Host-scoped offload/DDR capacity is reported by the backend, not guessed. |

## Minimum useful backend

A minimum backend still implements the full interface. The useful shortcut is
not to pretend unsupported ops exist; it is to delegate them to the reference
floor until the device implementation is ready.

The practical first slice is:

1. Register a stable backend name.
2. Return `Class()==Approx` unless the backend can prove byte-identical reference
   arithmetic.
3. Implement `Upload`, `Read`, `MatMul`, `Attention`, and `Argmax` for one model
   shape.
4. Delegate or explicitly fence the remaining ops rather than advertising caps.
5. Run parity against `cpu-ref`, then run the non-reference gate:

```bash
cmd/modelbench -backend <name> -require-non-reference
```

When the scaffold generator lands, it should make this split mechanical: every
method compiles, every unsupported path is visible, and the conformance kit says
which claims are still unfenced.

The compiling minimum example is
[`internal/compute/minimal_backend_example_test.go`](https://github.com/anthony-chaudhary/fak/blob/main/internal/compute/minimal_backend_example_test.go).
It is intentionally small: `MatMul` and `Attention` are the hot ops the wrapper
owns, `Caps()` advertises nothing, and every other method delegates to `cpu-ref`
until a device implementation is ready. That is the shape a day-1 vendor backend
should preserve: compile the whole contract first, then turn on capabilities only
after a witness proves them.

## What a chip team inherits for free

Once the backend sits behind the seam, fak's higher layers do not need a fork:

| Inherited layer | Why it matters to silicon vendors |
|---|---|
| Tool-call governance | The accelerator is sold into agent workloads that already need policy, refusal reasons, audit, and quarantine. |
| Runtime registry | A backend is selected by name; host products do not need hard-coded per-vendor forks. |
| Approx/reference gates | A vendor can ship an `Approx` backend with honest argmax/cosine evidence instead of claiming bit identity. |
| Cache and KV seams | Device-resident KV and prefix reuse can be implemented behind `NewKV` and cache witnesses. |
| Capacity probes | Fleet schedulers can reason about real memory only when the backend proves it can report it. |
| Documentation vocabulary | Vendor claims can say `FENCED`, `UNDEFINED`, `SHIPPED`, or `not yet` instead of overclaiming. |

That is the binding-layer value proposition: the chip team owns kernels and
runtime evidence; fak keeps the agent-kernel contract, policy, audit, and
operator vocabulary stable.

## One-day accelerator pilot

The intended vendor pilot is intentionally small:

| Step | Output |
|---|---|
| Scaffold | A backend package with every `compute.Backend` method present and registered under a stable name. |
| Fill two hot ops | Device-backed `MatMul` and `Attention` for one model geometry, with `Argmax` staying device-local if possible. |
| Run CPU parity | `cpu-ref` remains the witness floor; the device backend is checked as `Approx` unless proven reference. |
| Run non-reference gate | `cmd/modelbench -backend <name> -require-non-reference` proves the host did not silently fall back to CPU. |
| Publish support row | The vendor support page lists caps, correctness class, model shape, hardware, driver/runtime version, and residual gaps. |

## Claims to avoid

- Do not call a backend `fak-certified` until the conformance kit exists and the
  backend passes it.
- Do not advertise `Reference` unless exact-reference rungs are green.
- Do not advertise capacity, collective, async, fusion, or graph compile caps
  because the device family can do them in theory. Advertise only what this
  backend reports or implements.
- Do not describe provider cache, local KV reuse, and device-resident KV as one
  generic "cache hit." They have different trust and economics.
