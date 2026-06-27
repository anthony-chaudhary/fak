---
title: "The cross-platform spine: one agent kernel from IoT to hyperscaler"
description: "Why the same fak kernel is the invariant spine across the whole deployment spectrum — IoT, edge, laptop, hyperscaler — the way Linux is one kernel under an Android phone and a datacenter. The hardware changes; the workload shape and the invariants do not."
slug: cross-platform-spine
keywords:
  - cross-platform agent kernel
  - deployment spectrum
  - IoT edge hyperscaler
  - workload shape invariant
  - one binary laptop to fleet
  - hardware abstraction layer
  - portable by construction
---

# The cross-platform spine — one kernel from a sensor to a datacenter

Linux is one kernel. It runs the phone in your pocket, the Wi‑Fi router on your
shelf, the car's infotainment head unit, and the rack of GPUs training the model
on that phone. The silicon underneath spans four orders of magnitude in power and
price, and almost none of the kernel changes to cross that range. The drivers at
the bottom differ; the system-call contract in the middle does not. That stable
middle — the same `read`, `write`, `mmap`, the same process and scheduling model
on every box — is the **spine**. It is why "learn Linux once" pays off from a
Raspberry Pi to a hyperscaler, and why an application written against the contract
runs on hardware its author never owned.

Agents are arriving at the same shape, and they need the same kind of spine.

The thesis of this page: **`fak` is that spine for the agentic workload.** The
deployment target changes enormously — a battery-powered IoT box, an edge gateway,
a laptop, a fleet of datacenter GPUs — but the *workload shape* (an agent running a
loop that proposes tool calls a kernel must adjudicate) and the *invariants the
kernel keeps* (default-deny on every call, bit-exact reuse of work already done, a
tamper-evident line per decision) do not change shape with the hardware. So the
same kernel is present at every point on the spectrum, and an operator who learns
it on a laptop already knows it on a fleet. This is the cross-platform claim, drawn
one axis wider than the two the rest of the docs draw.

*Who this is for:* anyone deciding where an agent will run and worrying they'll
need a different stack at each scale. No prior `fak` knowledge needed beyond the
one-line idea — the model proposes a tool call, the kernel disposes. By the end
you'll be able to name what stays invariant across the whole deployment spectrum,
why that invariance is structural rather than aspirational, and the honest edge of
where it stops.

## The three axes (and the one this page adds)

The repo already draws two axes of "the same rule, everywhere":

- **The scale axis** ([engineering is building loops](engineering-is-building-loops.md)):
  the same observe → decide → act → verify shape, and the same trust invariant,
  recurs from one tool call up through the turn, the session, the fleet, and the
  loop that improves the loop. That axis is *internal* — it's about how much of the
  stack lives in one address space.
- **The depth axis** ([hardware portability](hardware-portability.md)): the
  in-kernel forward pass runs on a CPU reference, then CUDA, Vulkan, or Metal, by
  *registration* against the `internal/compute` HAL rather than a re-fork. That
  axis is about *which silicon runs the matmul*.

This page draws a third: **the deployment-substrate axis.** Not "which chip runs
the matmul," but "what *kind of box* is this, and how big" — from a microcontroller-
class edge node to a multi-GPU datacenter host. The claim is that the kernel is
invariant along this axis the same way it is along the other two, and for the same
structural reason: the part that changes is pushed below a contract, and the part
above the contract is the spine.

```
            THE DEPLOYMENT-SUBSTRATE AXIS
   (hardware specifics change ->; the spine does not)

   IoT / MCU-class   edge gateway     laptop / desktop   hyperscaler
   sensor+actuator   Pi / Jetson      dev box            8-GPU host, fleet
   --------------    -------------    ----------------    ---------------
   ^ HARDWARE: power, RAM, ISA, accelerator, network — all change 1000x

   ====================== THE SPINE (invariant) =======================
   tool call = syscall   |  default-deny capability floor (fail closed)
   result quarantine     |  bit-exact KV reuse / addressable eviction
   tamper-evident audit  |  one static pure-Go binary, CGO_ENABLED=0
   deterministic verdict |  same wire (OpenAI / Anthropic / MCP)
   ====================================================================

   v WORKLOAD SHAPE: an agent loop proposing tool calls — same at every box
```

## Why the workload shape is invariant even when the hardware isn't

The goal that motivates this page is a real observation about the field: the
*hardware specifics* of where agents run are diverging fast (NPUs on phones,
microcontrollers with TinyML, Vulkan desktops, Ampere/Hopper datacenters), but the
*shape of the work* is converging. Wherever it runs, an agent is a loop:

1. it observes some bytes (a sensor reading, a file, an API response),
2. it orients (assembles context for this step),
3. it **proposes an action** — a tool call,
4. something must decide whether that action is allowed, and
5. it acts and verifies.

Step 3 → 4 is the load-bearing seam, and it has the *same security and economic
structure on every box*. On a $35 edge node a malicious sensor reading can drive
the local agent to exhaust the battery or actuate a relay it shouldn't; on a
datacenter host a poisoned tool result can walk into a shared context and corrupt a
fleet. Different blast radius, identical mechanism: an untrusted program proposing
an effect that a gate must adjudicate **before** it happens, failing closed, and
leaving an auditable record. That is a syscall boundary, and it does not get
simpler or more complex with the size of the box — it just *is* the shape of the
work. A kernel that owns that boundary is therefore the same kernel at every scale.

This is exactly the Linux insight. The reason one kernel spans a phone and a
datacenter is not that Linux is magic; it's that "a process makes a system call the
kernel must mediate" is the invariant shape of computation on shared hardware, and
Linux is the thing that owns that shape. fak's bet is that "an agent proposes a
tool call the kernel must adjudicate" is the invariant shape of *agentic*
computation, and that owning that shape — not the token throughput below it — is
what earns a spine.

## What the spine actually is — five invariants, all shipped

The spine is not a slogan; it is a specific, small set of properties that are the
same artifact on every target. Each is grounded in shipped code, and each is
*hardware-independent by construction*, which is what lets it be invariant across
the substrate axis.

| Invariant | Why it doesn't change with the box | Where it lives |
|---|---|---|
| **One static pure-Go binary, zero deps, `CGO_ENABLED=0`** | Nothing to link or port; the same ~13 MB artifact cross-compiles to every target. The arm64 NEON Q8 path is the same one Apple silicon ships, so the edge build is not a special build. | `go.mod` (stdlib only, no `go.sum`); `release-artifacts.yml` (5 static targets incl. `linux/arm64`); `internal/model/quant_arm64.go` |
| **Default-deny capability floor, fail closed** | The lever is never wired up rather than caught after the fact, so it costs the same whether the model is a 1.5B on a Pi or a frontier API. The same booby-trapped policy walls identically across a weak cloud model and a local CPU model. | `internal/adjudicator`, `internal/policy`, `POLICY.md`, `docs/repro-packet.md` |
| **Result quarantine at the write-time boundary** | A poisoned tool result is held out of context by structure, not by a classifier — identical logic whether the "tool" is a sensor on an untrusted bus or a cloud API. | `internal/ctxmmu/mmu.go` |
| **Bit-exact KV reuse + addressable eviction** | The deterministic metrics (token-count reuse, evict == never-saw at `max\|Δ\|=0`) are hardware-independent and reproduce **byte-for-byte** across arm64 and x86_64. | `internal/model/kvcache.go`, `internal/kvmmu`; cross-platform witness in `HARDWARE-MATRIX.md` |
| **Tamper-evident audit: append-only, SHA-256 hash-chained** | One verifiable line per decision, the same on a regulated edge device and a fleet host; verified offline with `fak audit verify`. | `internal/journal/journal.go`, `docs/proofs/journal.md` |

Notice what is *not* on that list: tokens per second. The spine is the governance,
reuse, and provenance band — the half of the stack a fast token engine
([vLLM/SGLang/llama.cpp](one-binary-one-surface.md)) deliberately leaves empty.
Below the spine, the substrate-specific half is exactly where the **HAL**
([hardware portability](hardware-portability.md)) and the **engine seam**
(`EngineDriver`, the `--base-url` proxy) live: a CUDA kernel on the datacenter
host, a Metal kernel on the laptop, a phone-native NPU runtime behind the gate on a
handset, a CPU-only forward pass on the Pi. The hardware-specific part is pushed
*below the contract*; the part above it is the same everywhere. That split is the
whole portability mechanism, and it is the same split Linux uses: drivers below,
syscall contract above.

## Reading the spectrum end to end

The same kernel, four very different boxes, one contract:

- **IoT / constrained edge node** (battery, small RAM, an MCU-class or low-end
  arm64 SoC, often air-gapped). The win here is the part fak ships and the part the
  platform leaves thin: a default-deny gate the on-device model can't argue past, a
  poisoned-result fence, and a tamper-evident log — all CPU-only, all offline. The
  compute is somebody else's (a vendor NPU runtime behind the gate, or a tiny CPU
  model). *Honest edge: there is no measured RAM/power footprint on a real Pi or
  Jetson yet, and no 32-bit-ARM or phone-NDK binding — these are named net-new work
  in the [mobile/edge/IoT strategy note](../notes/MOBILE-EDGE-IOT-STRATEGY-2026-06-24.md),
  not shipped claims.*
- **Edge gateway** (Pi 5 / Jetson Orin / arm64 industrial gateway). With
  `linux/arm64` now a published release target, "download fak on an arm64 edge box"
  is a true statement backed by an official binary, not a build step. The
  determinism guarantee says the *verdicts* are bit-identical to the laptop, so
  porting risk is structurally low.
- **Laptop / desktop** (the dev box). The canonical adoption rung: `fak serve`
  fronts a local Ollama/llama.cpp/LM Studio, or runs the small in-kernel reference
  model. Same binary, same flags as production minus the hardening switches.
- **Hyperscaler / fleet** (multi-GPU host, many sessions). The same `fak serve`
  plus `--policy floor.json`, `--require-key-env`, Prometheus scrape, and the
  cross-session shared-KV reuse that pays off most when the fan-out is widest. The
  multi-GPU serving lane is the [hardware matrix](../HARDWARE-MATRIX.md) Platform 4.

You don't graduate from a dev tool to a different production system as you climb,
and you don't strip down to a different embedded build as you descend. You add or
remove flags. That "same binary, two scales" property
([one binary is the whole surface](one-binary-one-surface.md)) is exactly the same
property as "same kernel, two ends of the substrate axis" — this page is just that
claim drawn all the way down to the constrained end, not only up to the fleet.

### The datacenter end keeps the same invariants — and that's the witness, not throughput

The interesting claim at the hyperscaler end is *not* a throughput number. It is
that the five invariants above are the **same artifact** on a multi-GPU fleet host
as on the laptop — and for the two that are deterministic, "same" means
**byte-for-byte**, by construction. The bit-exact KV reuse and addressable-eviction
metrics (`max\|Δ\|=0`, evict == never-saw) are pure-Go logic with no hardware
dependency, so they don't merely *approximate* the laptop result on a bigger box —
they reproduce it exactly, the same way the [hardware matrix](../HARDWARE-MATRIX.md)
already witnesses them reproducing across arm64 and x86_64. A datacenter host is one
more point on that determinism axis: a faster forward pass below the contract, the
identical verdict above it. The default-deny floor and the SHA-256 hash-chained
audit line are likewise pure-Go and hardware-independent, so an offline
`fak audit verify` over a fleet host's journal is the same check it is on a Pi.

What is **witnessed today** vs. what is a **TARGET**, kept provenance-honest:

| Datacenter-end claim | Status |
|---|---|
| Deterministic invariants (bit-exact KV reuse, addressable eviction, default-deny, hash-chained audit) reproduce byte-for-byte on any box, fleet host included | **Witnessed by construction** — the metrics carry no hardware dependency, and the cross-ISA reproduction is recorded in `HARDWARE-MATRIX.md`. The same logic on a bigger box yields the same numbers; there is nothing silicon-specific left to drift. |
| A *dedicated* datacenter-scale run that re-records those same invariants on a multi-GPU fleet host, alongside [hardware matrix](../HARDWARE-MATRIX.md) Platform 4 | **TARGET / not-yet-witnessed** — the multi-GPU serving lane exists, but a fleet-host re-run of the determinism witness has not been captured yet. Until it is, the byte-for-byte claim rests on the construction argument plus the cross-ISA witness, not on a recorded datacenter row. |
| Any hyperscaler *throughput* result | **Not claimed.** The spine is the governance/reuse/provenance band; tokens per second is the substrate-specific half the HAL and engine seam own (see the fences below). |

## Why this is structural, not a marketing reframe

Three properties make the spine real rather than aspirational, and all three are
the *same* properties that make Linux portable:

1. **The hardware-specific part is below a typed contract.** The seven CPU-monoculture
   assumptions are lifted into the `internal/compute` types, so a new accelerator is
   a `Backend` registration, not an edit to the forward loop. A new wire is a new
   handler, not a new core. A new engine is an `EngineDriver`, not a fork. The spine
   above the contract never sees the silicon below it.
2. **The invariants are independent of the silicon by construction.** Default-deny,
   quarantine, and the hash-chain are pure-Go logic with no hardware dependency;
   the deterministic reuse metrics are *proven* byte-for-byte identical across two
   ISAs (the Mac↔Windows reproduction in `HARDWARE-MATRIX.md`). Invariance isn't
   claimed — it's witnessed.
3. **The artifact is one statically-linked binary with no dependency tree.** There
   is no Python env to drift, no CUDA/PyTorch pin to match per target, no libc to
   match. The supply-chain surface is identical on a Pi and a fleet host: one file
   to pin and audit. That is what makes "the same artifact everywhere" literally
   true rather than "a similar artifact, rebuilt per platform."

## The honest fences

This page widens an existing, honest story; it does not smuggle in new claims.

- **The spine is the governance/reuse/provenance band, not throughput.** fak is not
  a faster token engine on *any* box, large or small, and does not try to be
  (`README.md`, [`FAQ`](../FAQ.md)). The compute half is the substrate-specific
  half the HAL and the engine seam own; on constrained hardware that half is the
  vendor's runtime, not fak.
- **The constrained end is partly published, partly net-new.** `linux/arm64` is now
  a first-class release target and the determinism story makes the verdicts
  portable by construction — but there is no measured footprint on real Pi/Jetson
  hardware, no 32-bit ARM, and no Android-NDK/iOS bindings yet. Those gaps are
  enumerated, not hidden, in the [strategy note](../notes/MOBILE-EDGE-IOT-STRATEGY-2026-06-24.md).
- **The cache-reuse multiples are self-host only.** An app that merely *calls* a
  frontier API from an edge box gets the gate, the quarantine, and the audit line,
  but not the KV-reuse savings — those need fak to own the cache.
- **0 of 29 primitives are novel** (`CLAIMS.md`). The contribution here is the same
  as everywhere in the repo: the *assembly*. The new framing is that the assembly is
  invariant across the deployment substrate too, not only across the internal scale
  and hardware-depth axes — the crossing point is one kernel present at the most
  *kinds of box*, carrying the same invariant through all of them.

## Read next

- [One binary is the whole surface](one-binary-one-surface.md) — the same-binary,
  laptop-to-fleet operational claim this page extends down to the IoT end.
- [Hardware portability via the compute HAL](hardware-portability.md) — the depth
  axis: how a new accelerator plugs in by registration, the contract under the spine.
- [Engineering is building loops](engineering-is-building-loops.md) — the scale
  axis and the two-axis grid this page adds a third axis to.
- [Mobile / edge / IoT strategy](../notes/MOBILE-EDGE-IOT-STRATEGY-2026-06-24.md) —
  the go-to-market for the constrained end, with the honest net-new inventory.
- [Hardware matrix](../HARDWARE-MATRIX.md) — every box fak's gates have been proven
  on, and the cross-platform bit-exact determinism witness.
