---
title: "Generational ABI and Schema Compatibility Policy"
description: "How fak artifacts that cross generations — the frozen wire ABI and the versioned schema ledgers — stay compatible without a per-generation branch, so long-horizon architecture work can influence current APIs safely."
---

# Generational ABI and Schema Compatibility Policy

**Issue:** #1667.
**Parent:** #1625.
**Stream:** `gen/future`.
**Milestone:** Generation G3 - Future.
**Status:** research memo / decision model — a compatibility policy a later
`gen/second-next` or `gen/next` stream can enforce with code, not yet a runtime
gate.

This memo is the handoff a future agent can use without rereading the whole
generation epic. It answers one question: when a fak artifact has to be read by
a worker from a *different* product generation than the one that wrote it, what
compatibility promise binds the two, and what evidence moves that promise
between generation streams? The canonical stream taxonomy is in
[`docs/generation.md`](generation.md); this memo is the compatibility half of
the "second-next needs a compatibility policy before it can ship" rule stated
there.

## Why a cross-generation contract at all

Long-horizon architecture work (`gen/future`, `gen/second-next`) cannot
influence current APIs until it can promise that an artifact written today will
still be readable by a worker built for a later horizon — and that a later
worker will not silently mis-read an older artifact. Without that promise, every
architecture bet forces a flag day, which the shared-trunk rule forbids. The
industry analogue is the same one that seeds epic #1625: HBM3E and HBM4 ship
concurrently because the interface between generations is a published, additive
contract, not a fork.

fak already has the two artifact classes that cross generations. This memo names
them and states the promise for each.

## The two cross-generation surfaces

### 1. The frozen wire ABI (`internal/abi`)

`internal/abi` is the wave-0 spine every fleet worker imports: a closed,
byte-stable set of wire enums (verdict kinds, status, outcome, taint, scope,
ref-kind, fallback, the reason vocabulary) plus the `FoldRank` restrictiveness
lattice. Its compatibility rule is already proven and gated in
[`docs/proofs/abi+architest.md`](proofs/abi+architest.md): **additive-only**. A
new generation may register new enum values; it may never renumber, remove, or
repurpose an existing one, and every unknown value must fold fail-closed
(`FallbackDeny`, rank 100).

**Cross-generation promise (ABI):**

- *Backward read* — a worker from generation *N+1* reading an artifact written by
  generation *N* must recognise every value *N* could emit. Guaranteed because
  values are never removed.
- *Forward read* — a worker from generation *N* reading an artifact written by
  *N+1* must not crash or silently accept an unknown value. Guaranteed because
  unknown enum values fold to the most-restrictive rank (fail-closed), which is
  the safe default for a trust kernel.
- *No renumber* — a registration-block id is permanent. `FoldRank` orders by the
  restrictiveness lattice, never by raw enum value, so adding values never
  reorders folds.

The witness already exists: `TestFoldSitesOrderByFoldRank` and
`TestFoldRankOrdering` (`internal/architest`, `internal/abi`). This memo does not
change that gate; it *names it as the ABI arm of the cross-generation policy* so
a future stream does not re-invent it.

### 2. Versioned schema ledgers (the `/N` JSONL rows)

The second surface is the append-only JSONL ledgers under `docs/nightrun/` and
adjacent (`fak-memory-value-ledger/1`, gateway-usage, cache-savings,
harness-resources, and peers). Each row carries an explicit schema tag with a
version suffix (`.../1`). Their compatibility rule mirrors the ABI:

**Cross-generation promise (schema):**

- A schema version is immutable once a row of it is committed. New fields are
  added as *new optional keys within the same version* only when a reader that
  ignores unknown keys stays correct; anything that changes the meaning of an
  existing key or makes a previously optional key required is a **new version
  suffix** (`.../2`), never an in-place edit of `/1`.
- Readers must tolerate unknown keys (forward compatibility) and unknown schema
  versions (skip-with-witness, never crash — the same fail-closed posture as the
  ABI).
- A ledger may carry multiple schema versions concurrently, exactly as the wire
  ABI carries multiple enum generations concurrently. A fold/score consumer
  (e.g. `internal/memvaluescore`) must dispatch on the version tag, not assume
  the newest.

## Orthogonality (the generation invariants this artifact must restate)

This policy is metadata and promise, not a branch, a priority, or a runtime
switch. Concretely:

- **Orthogonal to priority.** The compatibility promise binds `gen/now` product
  code and `gen/future` research artifacts identically. A `gen/future` label does
  not lower the bar (an unversioned experimental ledger still cannot break a
  reader); a `gen/now` label does not raise it (product code still gets only the
  additive-only promise, nothing stronger). Priority answers "how urgent"; this
  policy answers "what will still parse."
- **Orthogonal to shared trunk.** All generations land on `main`, by explicit
  path, under the same DCO and ship-stamp rules. This policy explicitly forbids
  the per-generation branch (see #1667 non-goals): the whole point of an additive
  wire contract is that concurrent generations share one trunk without a flag
  day. Compatibility is enforced by the additive rule + `dos arbitrate` path
  scope, not by isolation.
- **Orthogonal to runtime feature gates.** A feature gate decides whether a code
  path is reachable at runtime; this policy decides whether a *serialized
  artifact* is readable across horizons. New-generation code can land inert
  behind a default-off gate while still emitting a new, additive, versioned
  schema that older readers safely skip. The gate controls exposure; the ABI/
  schema contract controls durability. Neither substitutes for the other.

## Promotion evidence (future → second-next → next → now)

This memo promotes when a later stream can *enforce* the promise, not just state
it:

- **future → second-next:** a compatibility test fixture that pins a `/1` row and
  a `/2` row and proves one consumer reads both — i.e. the schema arm gets the
  same kind of gate the ABI arm already has (`internal/architest`). Naming that
  fixture is the promotion witness.
- **second-next → next:** a machine-checked schema registry (analogous to
  `internal/abi`'s enum registry) that refuses an in-place edit of a shipped
  schema version, wired into `make ci` / a `fak hygiene` gate.
- **next → now:** the registry runs default-on in CI and the nightrun ledger
  producers emit their version tag through it, so a drift is caught at commit
  time rather than in review.

## Demotion / retirement evidence

- **Demote** if the additive-only rule proves too strict in practice — e.g. a
  wire value must be *retired* because it is a security liability, forcing a
  fail-closed removal path this memo currently forbids. That failure would push
  the policy back to `gen/future` for a redesign that distinguishes
  "never-renumber" from "may-quarantine-with-witness."
- **Retire** the memo if `internal/abi`'s existing additive proof plus a single
  schema-registry gate fully subsume it; then this doc collapses into a pointer
  from [`docs/generation.md`](generation.md) and the proof, with no standalone
  policy left to carry.
- **Retire** if generation-crossing artifacts stop existing (all consumers
  become single-generation), which would make the cross-generation promise
  vacuous.

## Invalidating assumptions (kill criteria)

State them so a later agent can check them cheaply:

1. **The additive-only rule is sufficient.** This memo assumes no
   cross-generation artifact ever *needs* a breaking change — that every real
   evolution can be expressed as a new enum value or a new schema version. If a
   concrete case appears that cannot (a field whose *removal* is required for
   correctness or security, with no additive path), the policy is invalidated for
   that surface and must gain an explicit, witnessed breaking-change protocol
   (deprecation window + reader-version floor). **This is the assumption most
   likely to fail.**
2. **Readers already ignore unknown keys/values.** The forward-compatibility
   promise assumes every current reader tolerates unknowns. This is *proven* for
   the wire ABI (fail-closed fold) but only *asserted* for the JSONL consumers —
   until the promotion fixture exists, a consumer that hard-fails on an unknown
   key would silently violate the schema arm.
3. **Two surfaces are enough.** This memo enumerates the wire ABI and the JSONL
   ledgers. If a third cross-generation artifact class emerges (e.g. an on-disk
   cache format, a persisted lease journal), the policy must be extended, not
   assumed to cover it by analogy.

## Handoff (continue from here without the epic)

A future agent picking this up should: (a) write the `/1`-vs-`/2` schema
compatibility fixture named under "promotion evidence" — that is the smallest
next increment and the concrete promotion witness; (b) leave `internal/abi`'s
additive proof untouched (it is the ABI arm and already green); (c) file the
follow-on as a real `gen/second-next` issue under #1625, not a note. The
compatibility promise here is planning data until that fixture makes the schema
arm enforceable.
