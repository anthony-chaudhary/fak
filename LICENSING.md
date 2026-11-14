# Netra Fused Agent Kernel (`fak`) — the layered-licensing stack

> This file replaces **decision-by-omission** (which silently defaults to
> *donate-the-category*) with a deliberate, written choice. It is a statement of intent and
> structure, not a legal opinion; the binding texts are [`LICENSE`](LICENSE) (Apache-2.0),
> [`CLA.md`](CLA.md) (the contributor grant), and any future per-layer license files. The
> change process for this document lives in [`GOVERNANCE.md`](GOVERNANCE.md).

## The stack, layer by layer

fak is licensed in **layers**, not as one undifferentiated blob. Each layer has a different
job, a different audience, and therefore a different license. The table is the single source
of truth for "what may I do with this part?"

| Layer | What it is | License **today** | Reserved option | Invariant |
|---|---|---|---|---|
| **Spec / ABI** | The frozen public ABI surface and conformance contract (`internal/abi`, the golden contract, the verdict matrix) | Apache-2.0 + patent **non-assert** for conformant implementations | — | Stays a **spec anyone may implement**; never proprietary. |
| **Importable kernel** | The adjudication core a third party links against (the ABI driver surface, the capability floor, plancfi, KV ownership) | **Apache-2.0** | — | **Never source-available-only and never proprietary.** This is the load-bearing promise of the whole project. |
| **Serving surface** | `fak serve` and the gateway/proxy binary as a deployable product | Apache-2.0 | **AGPL-3.0 + commercial dual-license**, held in reserve | May be relicensed *forward* under the reserved option; the kernel it embeds stays Apache-2.0. |
| **Value-capture product** | Managed control plane, hosted observability, fleet orchestration, certification services | Proprietary or source-available (TBD per product) | — | Built **on** the open kernel; does not subtract from it. |

### The one invariant that governs all the others

> **The kernel is never source-available-only and never proprietary.**

Every other choice (relicensing the serving surface, charging for the managed product,
reserving AGPL) is downstream of keeping the kernel open under Apache-2.0. A contributor,
an auditor, or a provider can rely on that single sentence. The CLA's sublicense right
([`CLA.md`](CLA.md) §2) exists to enable the *serving-surface and product* options — **not**
to ever close the kernel.

## Layer → path map

The layers above are mapped to concrete paths so that "which code sits in which layer"
is checkable, not asserted. Every path listed exists on trunk; adding a package to a
layer, or moving one between layers, is itself a governance change (see
[`GOVERNANCE.md`](GOVERNANCE.md)) and lands as a `docs(licensing):` update to this map.

| Layer | Paths (globs) |
|---|---|
| **Spec / ABI** | `internal/abi/**`, `pkg/abi/**`, the golden contract `internal/abi/testdata/abi_v0.1.golden`, the conformance suite `internal/conformance/**` |
| **Importable kernel** | `pkg/abi/**` (the ABI driver surface), `internal/kernel/**` (the adjudication core), `internal/plancfi/**`, `internal/kv/**`, `internal/kvmmu/**`; the rest of `pkg/**` is the import surface and joins a layer only when named here |
| **Serving surface** | `cmd/fak/**` (the `fak serve` entry point), `internal/gateway/**`, `internal/l3server/**`, `pkg/routerfront/**`, `pkg/gatewayauth/**` |
| **Value-capture product** | No committed paths yet (TBD per product). Best-effort current set: `internal/fleet/**` (fleet orchestration), `internal/ops/**` (managed control plane); updated through the governance change process as products are carved out. |

A new package does not silently join a layer: the change that lands it names its layer
in this map, or a follow-up `docs(licensing):` change does.

## Relicensing notice period (commitment)

The reserved AGPL-3.0 + commercial dual-license option on the serving surface is held
in reserve deliberately, and it may be exercised deliberately slowly. This is a
commitment to downstream users, hosts, and providers who build on `fak serve`:

- **Minimum 12-month written notice.** Before the first release carrying new license
  terms on any serving-surface path, the steward publishes a dedicated announcement
  (a labeled GitHub issue plus release notes) naming the affected paths from the
  layer → path map above and the effective date, at least twelve months earlier.
- **The last Apache-2.0 release stays available.** The final Apache-2.0 release of each
  affected component remains downloadable and supported (security fixes included) for
  at least twelve months after the announcement, so a host can remain on the last
  permissive release while it decides.
- **No retroactive reach.** Already-published Apache-2.0 releases stay Apache-2.0; the
  notice period governs future fixes and new releases only.
- **Scope is the serving surface.** The spec/ABI and kernel layers have no reserved
  option; the kernel invariant above binds regardless of this commitment.

Shortening the notice period requires a documented decision in
[`GOVERNANCE.md`](GOVERNANCE.md) with the reason; the default direction is longer, not
shorter. Exercising the option additionally follows the comment window in
[`GOVERNANCE.md`](GOVERNANCE.md).

## Why layered, and why now

- **Apache-2.0 on the kernel** maximizes adoption and makes the frozen ABI worth
  implementing against — a permissive floor is what lets a provider trust the drop-in.
- **A reserved AGPL-3.0 + commercial option on the serving surface** keeps a value-capture
  path open without retroactively taking anything from the open kernel. It is *held in
  reserve*: not exercised today, documented so it is a deliberate lever rather than an
  accident.
- **The CLA before the first external contribution** is the only time-sensitive,
  irreversible move (you cannot retroactively collect a relicense grant from past
  contributors). Hence DCO + CLA from day one — see [`CONTRIBUTING.md`](CONTRIBUTING.md).

## The value-capture decision (recorded)

The deliberate decision, recorded here per the project's positioning:

1. **Capture value in the product layer (managed/hosted/certification), not by closing the
   kernel.** Open kernel + open spec + proprietary product is the chosen shape.
2. **Do not donate the category by default.** Foundation donation is a *triggered* action,
   not a drift — see the donation trigger in [`GOVERNANCE.md`](GOVERNANCE.md). Donating
   prematurely would forfeit the stewardship needed to keep the layers coherent.
3. **Certification is a value-capture lever** tied to a behavioral conformance suite (the
   `fak-certified` mark, see [`TRADEMARK.md`](TRADEMARK.md)) — the one lever a permissive
   license cannot erode, because trademark is licensed separately from copyright.

## What this means for you

- **Using fak / building a driver against the ABI** → Apache-2.0, no strings beyond the
  Apache terms. Build, ship, embed, commercialize.
- **Contributing** → Apache-2.0 inbound = outbound, **plus** the CLA grant to the Steward
  ([`CLA.md`](CLA.md)). The kernel you contribute to stays open, guaranteed by the invariant.
- **Reselling a managed fak service** → permitted under Apache-2.0 today; the serving
  surface carries a reserved relicensing option for future versions, exercisable only
  under the [relicensing notice period](#relicensing-notice-period-commitment) below.

---

_License of record: [Apache-2.0](LICENSE) for the kernel. This document records intent and
structure; where it and a binding license file disagree, the binding file controls._
