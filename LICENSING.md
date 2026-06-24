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
- **Reselling a managed fak service** → permitted under Apache-2.0 today; be aware the
  serving surface carries a reserved relicensing option for future versions.

---

_License of record: [Apache-2.0](LICENSE) for the kernel. This document records intent and
structure; where it and a binding license file disagree, the binding file controls._
