# GOVERNANCE.md — who may change what, and how

> This file records the change processes that keep the project's promises honest: the
> **additive-only ABI** process, **who may bump the ABI minor version**, and the
> **foundation-donation trigger**. It is the companion to [`LICENSING.md`](LICENSING.md)
> (the *what* of the layered stack) — this is the *how it may change*.

## Stewardship

While **Netra Systems** is the steward, the steward is accountable for keeping the
[`LICENSING.md`](LICENSING.md) invariant true — **the kernel is never source-available-only
and never proprietary** — and for the change processes below. Stewardship is a
responsibility, not a license to close the kernel; the invariant binds the steward.

## The ABI is additive-only and human-owned

The public ABI (`internal/abi`, pinned by the golden contract
`internal/abi/testdata/abi_v0.1.golden`, currently **Major 0 / Minor 1**) is the substrate's
load-bearing promise: a driver written against ABI vX.Y keeps working. Therefore:

1. **Additive-only within a major version.** New ops, new verdict kinds, new fields, new
   `Register*` seams may be **added**. Existing ops/verdicts/fields are **never removed,
   renumbered, or repurposed** within a major version. A removal or a breaking change
   requires a **major** bump and is a deliberate, announced event.
2. **Human-owned.** The ABI is not auto-generated and not edited by an autonomous agent
   acting alone. A change to `internal/abi` is reviewed by a human maintainer, and the
   golden contract is regenerated and committed in the same change so the conformance pin
   moves atomically with the surface.
3. **Enforced below the agent layer.** `internal/architest` fails the build on an
   upward/cross-tier import; the golden test fails on any drift between the surface and the
   pinned contract. "It compiled and the golden matched" is the gate, not anyone's say-so.

### Who may bump `ABIMinor`

- A **minor** bump (`ABIMinor`, additive surface) may be proposed by any contributor but is
  **merged only by a human maintainer** with steward authority, together with: (a) the
  regenerated golden contract, (b) a conformance-suite run that stays green, and (c) a note
  in the change describing the added surface. The minor bump and the golden update land in
  the **same commit**.
- A **major** bump (`ABIMajor`, any breaking change) is reserved to the steward and is
  announced ahead of time. It cuts a new conformance pin (a new `abi-vX.0` tag — see the
  conformance program) and starts a new additive-only line.
- An **agent** (Claude Code / Codex / any autonomous session) may *draft* an additive ABI
  change and open it for review, but the merge of an `internal/abi` change is gated on a
  human maintainer. The trunk/commit guards do not, by themselves, authorize an ABI change.

## Conformance pin

"Certified at ABI vX.Y" must reference something immutable. The pin is a **git tag**
(`abi-v0.1`, …) at the commit whose golden contract defines that ABI version. The tag is
cut by a maintainer at the time the surface is declared stable, and the conformance suite is
SLA'd to the ABI cadence — a conformance suite that lags the kernel is a public attestation
of a stale floor, which is worse than no mark. See [`TRADEMARK.md`](TRADEMARK.md) for how
the `fak-certified` mark binds to a passing conformance run at a named ABI version.

## The foundation-donation trigger

Donating the project (or the spec) to a neutral foundation is a **triggered** action, never
a default drift. The steward donates **only** when at least one of these holds:

1. **A second independent implementation exists** — another module implements the ABI
   (claims an `OpsVendor`/`VerdictsVendor` number) end-to-end and is maintained
   independently. Neutral stewardship is meaningful once there is more than one
   implementer to be neutral *between*.
2. **A serious provider names neutral stewardship as an explicit precondition** for
   adoption — i.e. the donation unlocks real adoption that single-steward governance blocks.

Absent a trigger, the steward retains stewardship deliberately (see the value-capture
decision in [`LICENSING.md`](LICENSING.md)). Donation is one-way: it is taken only when the
conditions that make it safe and load-bearing are actually present, not pre-emptively.

## Change process for the governance and licensing docs

Changes to this file, [`LICENSING.md`](LICENSING.md), [`CLA.md`](CLA.md), or
[`TRADEMARK.md`](TRADEMARK.md) are made by the steward (or a maintainer with steward
authority), in the open, on the trunk, with a `docs(governance):` / `docs(licensing):`
subject. Legal-text changes to `CLA.md` carry the "pending legal review" caveat until
counsel signs off.

---

_This document governs process, not code semantics. The binding artifacts are the
[`LICENSE`](LICENSE), the golden ABI contract, and the conformance tags; this file records
who may move them and under what conditions._
