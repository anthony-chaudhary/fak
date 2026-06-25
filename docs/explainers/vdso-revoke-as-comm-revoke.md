---
title: "vdso.Revoke is the MPI_Comm_revoke analogue (invalidate-on-refutation)"
description: "fak's shipped vdso.Revoke path is the integrity-direction eraser: a refuted witness evicts every entry admitted under it, bumps a monotone trust clock, refuses re-admission, and publishes to registered subscribers. It is the structural analogue of ULFM's MPI_Comm_revoke — with one load-bearing difference: the revoked unit is a cache witness, not a live communicator, so revocation can only ever turn a would-be hit into a miss. It can never strand live computation."
slug: vdso-revoke-as-comm-revoke
keywords:
  - MPI_Comm_revoke
  - ULFM
  - fault tolerance
  - cache coherence
  - revocation
  - trust epoch
  - invalidate on refutation
  - cross-agent propagation
date: 2026-06-24
---

# vdso.Revoke is the MPI_Comm_revoke analogue (invalidate-on-refutation)

> **TL;DR:** `vdso.Revoke` is fak's shipped invalidate-on-refutation path. A refuted
> world-state witness evicts every cache entry admitted under it, refuses any future
> re-admission under that witness, advances a monotone *trust* clock, and publishes the
> refutation to registered subscribers so peer agents are causally evicted too. That is
> the shape of ULFM's `MPI_Comm_revoke` — with the bright line that the revoked unit is a
> **cache witness**, not a live communicator, so revocation is sound by construction: it
> only ever turns a would-be hit into a miss.

This is part of the [MPI-shaped message-passing epic](https://github.com/anthony-chaudhary/fak/issues/639).
It documents an **already-shipped** structural analogue (the symbols in
[`internal/vdso/revoke.go`](https://github.com/anthony-chaudhary/fak/blob/main/internal/vdso/revoke.go)),
not a new build. It pairs with the genuine fault-tolerance gap tracked in the
[cohort build issue](https://github.com/anthony-chaudhary/fak/issues/648) (`Shrink`/`Agree`),
which is the sibling that re-forms a *working group* after a crash — the piece fak does
not yet have at the agent layer.

## The MPI analogue

ULFM (the User-Level Failure Mitigation extension to MPI) adds `MPI_Comm_revoke`:
when a failure is detected on a communicator, any process may *revoke* it. Every
subsequent operation on that communicator fails immediately with a communicated
error, and the revocation **propagates** to every rank that holds it, so no
survivor keeps using a now-invalid handle. The collective point of `MPI_Comm_revoke`
is *invalidate-and-propagate*: turn a stale handle into an immediate, broadcast
miss, so a survivor cannot accidentally build on refuted state.

fak's integrity-direction eraser has the same shape, applied to the cache instead
of to a communicator:

| ULFM `MPI_Comm_revoke`            | fak `vdso.Revoke` (`internal/vdso/revoke.go`)                                                                                                       |
| --------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| revoke a communicator             | `Revoke(witness)` — refute an external world-state witness (a git commit / blob hash / lease epoch a cache entry was admitted under)                |
| subsequent ops fail on it         | every entry admitted under `witness` is **evicted now**, and the witness is marked refuted so the durable CAS cannot silently re-serve those bytes   |
| monotone failure detection clock  | `TrustEpoch()` — the monotone **integrity** clock, advanced once per refutation                                                                      |
| total refutations observed        | `Revocations()` — the integrity-bus event count                                                                                                     |
| propagation to every holding rank | `SubscribeRevocations(func(Revocation))` — registered subscribers (a peer's private cache, a "what changed" feed, an audit log) are notified        |
| one revocation event              | `Revocation{Witness, Evicted, TrustEpoch, Seq}` — the refuted witness, the local consumer-set size, the bumped epoch, and a shared-bus sequence      |

The mapping is structural, not implementational. fak does not speak the MPI wire
protocol, hold network ranks, or run a failure detector. `Revoke` names the same
*invalidate-on-refutation + propagate* boundary that `MPI_Comm_revoke` names, at
the cache-coherence layer the kernel already owns.

## Two clocks, dual erasers

fak's cache has two independent invalidation axes, and `Revoke` is the one MPI's
analogue lights up. [`revoke.go`](https://github.com/anthony-chaudhary/fak/blob/main/internal/vdso/revoke.go)
draws the duality explicitly:

- `worldVer` — the monotone **consistency** epoch, bumped by writes. This is the
  eraser for "the data changed." (`scope.go`.)
- `trustEpoch` — the monotone **integrity** epoch, bumped by refutations. This is
  the eraser for "a later witness refuted an entry that was already pooled — it was
  poisoned, or the source it claimed never existed — even though nothing *wrote* to
  it." (This file.)

The second axis is load-bearing precisely because the first cannot express it. The
durable, content-addressed CAS beneath the cache is frozen: the same content keeps
the same digest forever, and `worldVer` only advances on a real mutation. Neither
can retire a pooled entry whose *world* is unchanged but whose *provenance* turned
out to be a lie. Integrity therefore needs its own clock — dual to the consistency
one — and `Revoke` is the trigger that advances it.

## The mechanism, end to end

`Revoke(witness)` does four things in one locked pass, then publishes outside the
lock (the code in [`revoke.go`](https://github.com/anthony-chaudhary/fak/blob/main/internal/vdso/revoke.go)):

1. **Evict the consumer set.** Every entry admitted under `witness` is removed from
   the LRU and the cache map, its CAS pin released. This is targeted, not a blunt
   full flush: sibling witnesses stay warm.
2. **Refuse re-admission.** The witness is recorded as refuted, so the durable CAS
   cannot silently repopulate the evicted bytes on the next read (without this, the
   consistency key would still match and the same poisoned bytes would come back).
3. **Advance the trust clock.** `trustEpoch` is bumped atomically, and the
   refutation is appended to the exact revoked-witness ledger (bounded; on overflow
   the vDSO fails closed, treating unknown witness-bearing entries as revoked).
4. **Publish.** A `Revocation{Witness, Evicted, TrustEpoch, Seq}` is handed to every
   registered subscriber outside the lock, so a peer's private cache or a "what
   changed" feed can evict its own copies. `Seq` is shared with the write-bus
   `Mutation` sequence, so a subscriber can interleave writes and refutations into
   one total order without a wall clock.

`SubscribeRevocations` is the *propagation* half — the integrity-direction
companion to the write-bus `Subscribe`. The cache's own eviction already happened
inside `Revoke`; subscribers are *additional* observers. Only refutations fire it,
never the read hot path.

## Why this is sound (the load-bearing honesty)

`MPI_Comm_revoke` is delicate: revoking a communicator mid-collective can strand
live computation, so ULFM pairs it with shrink/agree to re-form a working group.
fak's revocation has no such hazard, and the reason is structural rather than
careful. Quoted verbatim from
[`revoke.go`](https://github.com/anthony-chaudhary/fak/blob/main/internal/vdso/revoke.go):

> Soundness ("a hit equals a fresh call", preserved). Revocation is sound BY
> CONSTRUCTION and trivially so: it only ever turns a would-be hit into a MISS (-> the
> engine, a fresh call) and only ever refuses a fill. It can never cause a stale serve
> because it never serves anything — it is the SAFE direction of the cache-coherence
> tradeoff. The witness binding is purely ADDITIVE to the consistency key (it is NOT a
> component of keyLocked), so an entry with no witness behaves exactly as v0.1, and an
> entry WITH a witness is gated by BOTH the consistency key AND the not-revoked check —
> two gates that can only ever remove serves, never add one.

That asymmetry is the whole story. Revoking a *communicator* is dangerous because a
collective might be mid-flight on it; revoking a *cache witness* is safe because a
cache miss is not a fault — it is a delegation back to the engine, a fresh call. The
revocation axis can only ever convert serves into non-serves. It is the safe
direction of the cache-coherence tradeoff, the way a write-invalidate MESI probe is
safe compared with a write-back race.

## What this is NOT

This documents an analogy, not an MPI implementation. Quoted verbatim from
[`revoke.go`](https://github.com/anthony-chaudhary/fak/blob/main/internal/vdso/revoke.go),
the file's own scope caveat:

> What this is NOT (honesty, per PRIOR-ART-fak-partb-residue): this builds C4 (causal
> refutation eviction across the recorded consumer set) and opens the C3-external seam
> (the witness is an external world-state token, not the internal worldVer counter), but
> it does not yet bind the witness into the tier-2 KEY, so two agents reading under
> different witnesses still share by (tool,args,worldVer). Witness-keying is the natural
> follow-on; the revocation axis is the load-bearing half and is what this file ships.

The bright lines, stated plainly so they cannot be misread as a performance or
consensus claim:

- **The revoked unit is a cache witness, not a communicator.** A witness is an
  external world-state token — a git commit, blob hash, or lease epoch the
  orchestration substrate already holds. `Revoke` refutes that token. It does not
  tear down a live conversation, renumber ranks, or interrupt a collective.
- **`Revoke` is sound precisely because it only ever turns a would-be hit into a
  MISS** — it can never strand live computation the way revoking a communicator
  mid-collective does.
- **Cross-agent propagation is an in-process, registered-subscriber publish, not a
  network failure broadcast.** `SubscribeRevocations` hands a `Revocation` to
  observers that explicitly registered; it sends no packets, detects no partitions,
  and reaches no process that did not opt in. A "peer" here is a registered
  subscriber in the same pool's coherence bus.
- **fak inherits no MPI/HPC throughput, latency, message-rate, or wire-protocol
  property.** This borrows the *structure* of `MPI_Comm_revoke`
  (invalidate-on-refutation + propagate) and adds what MPI lacks: the revocation is
  adjudicated inside the kernel's single integrity-notification point, the trust
  clock is monotone, and the consumer-set eviction is targeted. The only measured
  numbers are the evicted-entry count and the epoch — both deterministic and
  replay-checkable.

The fault-tolerance story this doc surfaces is the *under-told* half: fak already
has invalidate-on-refutation at the cache layer. What it does **not** yet have is
the agent-layer shrink/agree that re-forms a working group after a real crash —
that is the [cohort build issue](https://github.com/anthony-chaudhary/fak/issues/648),
the genuine gap this shipped analogue is paired with.

## Where to go deeper

- The source of truth for every symbol above:
  [`internal/vdso/revoke.go`](https://github.com/anthony-chaudhary/fak/blob/main/internal/vdso/revoke.go).
- The consistency-direction eraser (`worldVer`, the other clock):
  [`scope.go`](https://github.com/anthony-chaudhary/fak/blob/main/internal/vdso/scope.go).
- The cache-coherence context this sits inside: [frozen trajectory cache cliff](frozen-trajectory-cache-cliff.md).
- The one-sided shared-result pool whose coherence fence this revocation path guards:
  [addressable KV cache](addressable-kv-cache.md).
- The parent [MPI-shaped message-passing epic](https://github.com/anthony-chaudhary/fak/issues/639)
  and the paired fault-tolerance gap: the [cohort build issue](https://github.com/anthony-chaudhary/fak/issues/648).
