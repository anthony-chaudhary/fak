# Prior art, and where the gap really is

*Companion to [the core note](expire-by-default.md). This is the related-work map, moved
out of the core so a first read isn't a wall of unfamiliar proper nouns. A contribution
you can't place against the field is one nobody can trust; here is the honest placement,
drawn so the one surviving gap is visible at the seams.*

State the gap up front, so you can hold each citation against it: **estimated
truth-duration as an *enforced write-time admission gate* that refuses promotion by
default.** Each result below has one or two of those properties; none, as of this writing,
has all three — though several are one design iteration away, because this is a field of
live preprints, not a settled landscape.

## The estimator this builds on

**Zhang & Choi, *"Mitigating Temporal Misalignment by Discarding Outdated Facts"*** (EMNLP
2023, [arXiv:2305.14824](https://arxiv.org/abs/2305.14824)) defined **fact duration
prediction** — predict how long a fact stays true, so a model can tell stable knowledge
from volatile and discard what's gone stale. That is precisely the estimator a durability
classifier consumes, named years before this writeup. The contribution here is strictly
downstream: turn the estimate into an enforced, expire-default write gate. The ML primitive
is theirs.

The older root of the same instinct is **bitemporal data modeling** (Snodgrass; SQL:2011):
every fact carries a *valid-time* (true in the world) distinct from a *transaction-time*
(recorded by the system). "It's 3pm" is a fact with a one-hour valid-time; storing it as a
timeless assertion is a modeling error the database world fixed decades ago. Agent memory
is, in large part, re-deriving bitemporality.

## The closest neighbors, and the exact line each doesn't cross

- **Springdrift** (Brady, 2026, [arXiv:2604.04660](https://arxiv.org/abs/2604.04660)) — an
  auditable persistent runtime; case-based reasoning over **append-only** memory replayed
  chronologically, weighted by outcome. Closest *mechanism*. Two decisive differences:
  opposite default (it keeps every record; any down-weighting is *post-write*), and
  post-write ranking versus write-time admission (a discounted case is still resident and
  can resurface; an un-admitted span never was). The difference between "we'll discount it
  later" and "it never got in."
- **Cloudflare Agent Memory** (April 2026) — a genuine, partial truth-duration gate, and
  the one I owe the most concession. An **eight-check write-time verifier** classifies into
  Facts / Events / Instructions / Tasks; Facts **supersede rather than delete** on
  staleness; Tasks are session-scoped and **ephemeral by design**. Most of the machinery
  this argues is missing. Two lines it doesn't cross: the bucket is a fixed content *type*
  (assigned by what the thing *is*, orthogonal to duration — a Fact can be ephemeral, a Task
  can encode a durable preference), and the default is **persist-and-supersede**, not
  expire-unless-earned. Closest in the field; the distinction is narrow on purpose.
- **Zep / Graphiti** (Rasmussen et al., 2025, [arXiv:2501.13956](https://arxiv.org/abs/2501.13956))
  — the strongest production port of bitemporality: a bi-temporal knowledge graph stamping
  `(t_valid, t_invalid)` and **invalidating rather than deleting** on contradiction. The
  "isn't this just Zep?" instinct deserves a precise answer. Zep *does* filter at write
  (dedup + contradiction invalidation); what it does *not* do is gate on truth-duration —
  every extracted fact is admitted, then revised, with no expire-by-default. Zep tells you
  *when a written fact stopped being true*; it never asks *whether the fact should have been
  written*. That unasked question is the gate.
- **Beyond Dialogue Time** ([arXiv:2601.07468](https://arxiv.org/abs/2601.07468), Jan 2026)
  — splits memory into **point-wise** (atomic facts on a semantic timeline) and **durative**
  (enduring patterns consolidated from point-wise memories *after the fact*). The same
  *axis*, cited as confirmation it's real and independently arrived at — not a "first," and
  not a write-time gate (durative memory is post-hoc abstraction over already-stored
  episodes).

## The three dominant production families

Step back and shipped systems fall into three write-policy families, none a principled
enforced durability gate:

1. **Capacity / overflow-driven** — MemGPT / Letta (Packer et al., 2023), LlamaIndex.
   Promotion fires when the window fills: summarize the oldest turns. The trigger is *"it
   doesn't fit,"* unrelated to whether the facts stay true; overflow launders the ephemeral
   into the permanent by sweeping timestamp, mood, and preference together by age.
2. **LLM-judgment** — Mem0's ADD/UPDATE/DELETE/NOOP; Letta block edits; the auto-synthesis
   behind ChatGPT, Claude, and Gemini memory. An LLM decides per fact — the one place "is
   this transient?" is even implicitly asked — but it's *advisory*, and the systems concede
   it misclassifies. (Note: on *abstain*, the common incumbent behavior is to write
   **nothing**, which is observationally identical to expire-by-default for that span; the
   persist failure occurs when the judge *fires* and mis-promotes — which is the happy path
   that is identical to a fak-style classifier. So the differentiator is the *enforcement
   posture*, not a better classifier. This is conceded in the [review ledger](_adversarial-review.md).)
3. **Score-based, at read time** — Generative Agents (Park et al., 2023): importance +
   recency + relevance score *retrieval*, everything written first. MemoryBank (Zhong et
   al., 2023): Ebbinghaus decay reinforced on recall, so a transient fact that happens to
   get recalled is reinforced *as if* durable. Both approximate durability post-hoc, after
   the ephemeral fact is resident.

## The deepest version is the oldest

**ACT-R**'s base-level activation (frequency/recency, context-*independent* → durable)
cleanly separated from spreading activation (context-*dependent* → contextual) is the
principled analog the LLM systems gesture at and don't implement. And **Tulving (1972)**
split *episodic* memory (specific, time-and-place indexed) from *semantic* memory
(decontextualized gist); the operation *between* them — context-stripping an episode into
a standing fact — is the promotion decision itself. Consolidation is selective
transformation, not archival copying; forgetting the ephemeral is adaptive. The AI field
imported the storage-hierarchy metaphor and skipped the selective consolidation that is the
entire point.

## The seam, exactly

Not noticing time matters (cognitive science, bitemporal DBs). Not predicting how long a
fact stays true (Zhang & Choi). Not naming the axis (Beyond Dialogue Time). Not decaying or
replaying a store (Springdrift), typing memory by lifespan (Cloudflare), or revising facts
at retrieval (Zep). All covered, and covered well. The one thing none does — *as of this
writing* — is enforce estimated truth-duration **as a write-time admission gate that
refuses promotion by default**, fail-closed, that a downstream component cannot route
around. That is the [enforcement-topology claim](_enforcement-topology.md) — a reference-
monitor posture, not a new memory semantics — and it is a narrow, conceded-as-small
contribution riding a gate that ships.

---

*Back to [the core](expire-by-default.md) · [enforcement topology](_enforcement-topology.md) ·
[what fell under review](_adversarial-review.md).*
