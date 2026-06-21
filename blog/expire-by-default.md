# A Fail-Closed Tag on the Write: Durability as an Enforcement Problem, Not a Memory Semantics

> *The hard part of agent memory was never the TTL bucket. It was deciding where the
> keep-or-forget choice is **enforced** — and proving a downstream component can't route
> around it.*
>
> An engineering note from the **fak** project. This is the tight core; three companion
> notes carry the parts that don't belong in a first read:
> [the prior-art map](_prior-art.md), [the enforcement-topology argument](_enforcement-topology.md),
> and [the adversarial-review ledger](_adversarial-review.md) — what fell when this thesis
> was attacked, and the wounds still open. The durability classifier and expire policy
> below are `[PROPOSED]`; the gate they ride on **ships** (`internal/ctxmmu`).

---

## The bug everyone has shipped

It's 3pm. Your agent knows that — correctly, usefully, for this turn. The question that
decides whether your memory system is sound is what happens to "it's 3pm" three weeks
from now.

An operator put it in one sentence: *"The context is that it's 3pm, but I don't want that
in memory in general."* "It's 3pm" is real and relevant — and false within the hour. A
memory system that promotes it to durable storage hasn't helped; three weeks later the
agent recalls "it's 3pm" and serves it back as standing truth, with the same confidence
it serves your name. Nobody asked it to forget, because nobody knew it had remembered.
The write pipeline asked *"is this useful / important / relevant?"* — and "it's 3pm"
passes every one of those filters.

The filter you actually want is a different axis: **how long the fact stays true.** "It's
3pm" stays true for minutes; "I prefer afternoon meetings" stays true for years. Same
size, same turn, identical to an importance score — opposite sides of the memory
boundary. (Note the axis is not salience: *your current account balance is $40* has short
truth-duration but high recurrence and relevance — so a recency/importance ranker keeps
it, and it must still never freeze as durable. Truth-duration and salience point in
different directions, and only truth-decay produces the silent stale-as-current error.)

## The claim, scoped up front

Here is what this note argues, and the scope is part of the claim, not a footnote.

**The contribution is an enforcement-topology claim, not a new memory semantics.** On the
happy path, a write-time durability classifier is the *same kind* of LLM/lexical judgment
Mem0's ADD/UPDATE/NOOP and Zep's extraction already make — there is no new science in the
guess. What's different is one property: the keep-or-forget decision is **enforced on the
sole path to the durable store**, fail-closed, so a buggy or compromised component can't
route around it. An *advisory* lifecycle hint — the kind every framework has — is the
first thing a sloppy compactor drops when it copies a `turn`-scoped span into a durable
summary. That is the whole move: not a better classifier, a non-bypassable gate. The
[enforcement-topology note](_enforcement-topology.md) makes the precise version of this
(capture-at-write is forced; *deciding* can happen at read; the kernel is a convenient,
not unique, site).

**Expire-by-default is the right default for the modal agent — attended, reading before
it acts, with reachable sources — and that scope is load-bearing.** The argument is an
asymmetry of failure costs. A *false negative* (forgot a durable fact) is recoverable and
self-announcing: the fact is still true, the miss shows up at read time, the user
restates it, the cost is one re-ask. A *false positive* (promoted an ephemeral fact as
durable) is silent, persistent, and surfaced as confident truth — "I'm in a hurry today"
freezing into "the user always wants speed," which nobody knows to correct. One error is
a cheap re-ask; the other is an undetectable wrong belief. Bias toward the cheap error.

But the scope is real: outside the attended regime the asymmetry can **invert**. In a
headless, stated-once, one-shot pipeline, a byte-exact-evicted false negative — a
kill-switch threshold dropped before the action fires — leaves *no auditable artifact*,
while a stale false positive at least leaves a value a later read could catch. There the
false negative is the silent one. So "expire-by-default" is not a universal law; it is
the loss-minimizing default *for the regime where the modal product lives*, and the honest
form names the regime that flips it. (An earlier version of this argument claimed it was
"forced, not tuned" in every regime. That was wrong; see the
[review ledger](_adversarial-review.md).)

## The honest default is three arms, not two

Because the asymmetry is scoped, the default is not a binary expire/persist switch. The
surviving invariant is narrower and more defensible: **persist is never the right default
for an *unclassified* span.** What to do instead depends on what kind of span it is:

- **First-party user assertions → persist, by authority.** "I prefer afternoons" is
  written because the principal who owns the durable store said it. Defaulting a user's
  own stated preference to `turn` is what manufactures the forgetful-assistant failure
  users hate — so principal is part of the decision, not an afterthought.
- **Oracle-checkable facts → persist *with a non-strippable freshness stamp* + read-time
  re-verification.** A SHA, a file state, a flag: keep it, but re-check it at read against
  the ground truth, and decay it when the check fails. This *dominates* hard-expire
  wherever the freshness check is itself fail-closed — and it is the production answer for
  most ambiguous-but-useful writes ("the repo uses pnpm not npm").
- **Genuinely unclassified residual → expire-or-escalate.** Expire the throwaway; for the
  high-stakes unattended one-shot, **refuse/escalate** rather than silently expire *or*
  silently persist. (This arm is the residual novelty and the least-built — it needs
  read-time *action* context the write-side gate doesn't have. It is honestly the weakest
  part; the [review ledger](_adversarial-review.md) says why.)

"Expire-by-default" was always shorthand for the prohibition (*never silently persist the
unclassified*), and hard-expire was one remedy among these three, never the only one.

## What's actually hard (and unsolved)

Two honest open problems, stated so they aren't mistaken for shipped:

1. **The representation, not just the default.** "Route uncertainty to the cheap error"
   needs a *calibrated posterior* over truth-duration to threshold against the
   cost-sensitive operating point — but a hard four-way class (`turn|session|bounded|durable`)
   on an open string tag *throws away the very uncertainty the decision rule consumes*. The
   real object is a distribution whose *width* picks the arm, not a bucket. Getting the
   output type right is upstream of getting the policy right, and the policy math assumes a
   representation the bucket can't carry. This is the sharpest unsolved piece.
2. **The classifier is fallible and that's the argument, not the hole.** The STALE
   benchmark ([arXiv:2605.06527](https://arxiv.org/abs/2605.06527)) finds the best model
   only ~55% accurate at knowing when its *own* memory went stale. A write-time classifier
   is no more oracular. But a fallible judge under expire-by-default routes its uncertainty
   to the *cheap* error; the same judge as a read-time "should I trust what I stored?"
   check routes it to the *expensive* one. Same fallibility, opposite blast radius, decided
   by which way the default points — for oracle-*less* facts. For oracle-checkable ones,
   arm 2 removes the fallibility from the loop entirely.

## The backbone is real, the policy is proposed

What separates this from a roadmap: the *enforcement* half already runs as adjudicated
code in the kernel's hot path, with test witnesses (`internal/ctxmmu`). Tool results pass
a **write-time admission gate** (`Admit`) that returns `Allow` / `Transform` /
`Quarantine` from a closed set; a poisoned or secret result is structurally held out of
context, not flagged-and-hoped. Removal is **byte-exact**: `model.KVCache.Evict`
re-rotates the survivors so the context is bit-identical to one that never saw the span
(`max|Δ|=0`, proven against the reference). Trust is decided by `internal/provenance` —
`Untrusted` is the fail-closed default, derived from a host-registered source class, never
the model's own tag — which is why **memory poisoning is the trust gate's job, not the
durability gate's** (a competent attacker injects a durable-*if-true* payload a correct
classifier keeps anyway; the durability axis is neither necessary nor sufficient against
it). And eviction is not default-allow: `recall` re-screens every page on the way back in
and honors a trust-epoch revocation, and `RequestContextChange` lets the agent tombstone a
span it judges stale — so the *forget* path is already gated, not wide open.

What is **`[PROPOSED]`**: the durability classifier, the expire-by-default promotion
policy, and any TTL clock. The gate they would ride on ships; the tag does not. The
durability value is an additive key on the existing open `Verdict.Meta` / `Result.Meta`
seams — no change to the closed verdict set — so the schema cost is genuinely zero and the
*behavior* is the entire proposal.

## The one-line version

Trust decides whether a value may enter memory; durability decides whether it *should*,
and for how long. fak ships the gate for the first as enforced, byte-exact, fail-closed
code; durability is the proposed second tag on the same gate, and its honest default is
*persist is never right for an unclassified span* — expire-or-escalate the residual,
persist-with-freshness what an oracle can re-check. The next thing to build is not more
machinery but the measurement: a stale-as-current A/B that prices the silent failure
directly.

---

*Companions: [prior art & where the gap really is](_prior-art.md) ·
[the enforcement-topology argument](_enforcement-topology.md) ·
[what fell under adversarial review](_adversarial-review.md). Design doc:
[`../CONTEXT-IS-NOT-MEMORY.md`](../CONTEXT-IS-NOT-MEMORY.md).*
