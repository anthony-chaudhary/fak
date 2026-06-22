# The enforcement-topology argument

*Companion to [the core note](expire-by-default.md). The core claims the contribution is
an enforcement-topology one — "a fail-closed tag on the write" — not a memory semantics.
This note makes that precise: which layer the decision belongs to, which obligations are
forced vs. free, and where the boundary actually bites.*

## Which layer: the decision is a *semantics* property, not a *routing* one

fak's [memory-layers framing](../docs/MEMORY-LAYERS-EXPLAINER.md) separates four things that
get conflated under the word "memory":

- **Routing** — where the cell physically lives, how fast it's served (KV/prefix caches,
  LMCache, the serving engines own this).
- **Addressing** — how a cell is named so two readers hit the same one.
- **Fusion** — how cells combine into a context.
- **Semantics** — *what may be done to the cell*: who may read it, what write invalidates
  it, whether it should persist at all.

Durability — *how long is this true, should it become memory* — is a **semantics**
question, and that placement is the whole reason it can't live in a serving cache. A
serving cache sees an anonymous token stream; it can tell you bytes are warm or cold but
not that "prefers afternoons" outlives "it's 3pm," because the distinction was never *in*
the bytes. The structure the decision needs — span type (tool result vs user text vs model
reasoning), principal, turn boundary, provenance — exists at the agent's syscall boundary
and is gone by the time anything looks like a cache. So the durability decision belongs at
the same layer fak already adjudicates trust, and for the same reason: that's the only
layer that can *see* what it's deciding about.

## Three obligations, and only one is a real architectural claim

The original post fused three things. Pulling them apart is most of the honesty.

**Capture is timing-forced.** Structure not otherwise logged — principal, span, turn
boundary, provenance — is irretrievable once the blob flattens into a token stream. It has
to be recorded *before* that. "At write" is the natural site, but not a unique one: if a
transport already records principal + span, capture is satisfied wherever that record
lives. This is forced by information loss, nothing more.

**Deciding is free.** The durability *verdict* can be computed at write or lazily at read.
An earlier version implied write-time decision was forced; it isn't. (In fact, lazy
read-time expiry — *don't return a fact whose validity has passed* — is often simpler and
sufficient than an active eviction scheduler, and avoids building a cron daemon for memory.
A staff-engineer reading made exactly this point, and it's right: a stamped-but-resident
expired fact costs nothing if retrieval won't return it.)

**Enforcement on the sole path is the claim that bites.** The gate must sit on the *only*
path to the durable store, so a text-emitting model cannot route around it. This is not a
property of being the kernel — any sole writer (an app framework owning the only write
path) is equally fail-closed against a model producer. The kernel is a *convenient,
defensively-deeper* site, not the only possible one. And the guarantee is scoped to the
model-producer threat — it is not a general "all writers are gated" claim. What the
argument earns is exactly: *mandatory, on the sole path, against a model producer.* No
more.

## Why enforced, not advisory — without borrowing a security threat

An advisory durability hint a downstream component can strip is not a lifecycle guarantee,
and the reason needs no threat model richer than *components have bugs*. A buggy compactor
that copies a `turn`-scoped span into a durable summary produces a standing belief nobody
decided to keep — and a `durability: turn` hint is the first thing such a component drops.
That is the entire case for enforcement, and it's strictly weaker (therefore harder to
attack) than the OWASP-poisoning appeal an earlier draft made and then
[retracted](_adversarial-review.md): it justifies non-bypassable enforcement of *any*
lifecycle property without claiming the durability gate defends a security boundary it
doesn't.

The posture already exists in shipped code: the trust gate (`internal/ctxmmu.Admit`)
holds quarantined bytes *structurally* out of context rather than flagging-and-hoping, and
`internal/provenance` makes `Untrusted` the fail-closed default keyed to a host-registered
source class the model can't forge. Durability is one more admit-time verdict riding the
same enforced posture — the additive `Verdict.Meta` / `Result.Meta` seam exists for
exactly this, so the schema cost is zero and the work is the classifier and policy behind
the tag.

## Eviction is the dual of admission — and it must be equally fail-closed

The sharpest gap, surfaced by an external security review: the whole apparatus defends the
*write-in* direction. Expire-by-default makes *eviction* a security-relevant event, and a
byte-exact, audit-erasing forget is a privileged *write-out* an adversary will target — to
induce a silent, unrecoverable false negative (mis-route a standing safety rule into
`session` so it byte-exact-evicts; or skew the clock a TTL depends on to force early
expiry). The kernel is not naked: `recall` re-screens every page on the way back in and
honors a `vdso` trust-epoch revocation, and `RequestContextChange` records tombstones as
durable ledger rows rather than silent mutations. But the *expiry* path does not yet carry
the same default-deny treatment admission does. The rule the threat model needs, stated
plainly:

> **Eviction is the dual of admission and must be equally fail-closed: a span carrying a
> safety/durable class is default-deny to evict, and expiry state must survive resume —
> or "expire-by-default" becomes "forget-on-command" for whoever controls the classifier,
> the freshness oracle, or the clock.**

That sentence is the one this whole note exists to earn, and it is currently a `[PROPOSED]`
obligation, not shipped — the [review ledger](_adversarial-review.md) keeps it honest.

---

*Back to [the core](expire-by-default.md) · [prior art](_prior-art.md) ·
[what fell under review](_adversarial-review.md).*
