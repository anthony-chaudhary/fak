# fak/blog

External-facing engineering posts drawn from fak's internal design docs and research
sweeps. These are the *publishable* form of the in-repo strategy/design notes — written
for technically literate builders of LLM agents, cited against primary sources, and held
to the same `CLAIMS.md` honesty discipline (SHIPPED vs PROPOSED, no overclaim, prior art
cited generously).

Posts are **modular**: a tight core note plus separable companion notes (prefixed `_`),
each cross-linked and independently revisable, re-pathable, or removable. The single-file
form proved too long and litigated its own concessions twice; the split keeps the core
confident and quarantines the retreats into one ledger. Dated redirect stubs preserve any
URL that was already public.

## Posts

### Expire by Default — durability as a write-time enforcement problem (2026-06-20)

- **[Core: A Fail-Closed Tag on the Write](expire-by-default.md)** — the bug everyone
  ships ("it's 3pm" promoted to durable memory), the scoped claim (an *enforcement-
  topology* claim, not a new memory semantics), the three-armed default whose invariant is
  *persist is never right for an unclassified span*, what's actually hard (the
  representation gap), and the shipped backbone vs. the proposed policy.
- [Companion — Prior art, and where the gap really is](_prior-art.md) — Zhang & Choi 2023,
  Springdrift, Cloudflare, Zep/Graphiti, Beyond Dialogue Time, the three production
  families, ACT-R/Tulving; the exact line each doesn't cross.
- [Companion — The enforcement-topology argument](_enforcement-topology.md) — which layer
  the decision belongs to (semantics, not routing); capture-forced / decide-free /
  enforce-on-the-sole-path; and *eviction is the dual of admission and must be equally
  fail-closed*.
- [Companion — What fell under adversarial review](_adversarial-review.md) — the single
  concession ledger: what fell (forced→scoped, poisoning severed, kernel-exclusivity
  dropped, semantics→topology) and the six residual live wounds, stated plainly.
- [Redirect stub](2026-06-20-expire-by-default-truth-duration.md) — the original dated URL,
  now pointing at the core.

Companion to the design doc `../CONTEXT-IS-NOT-MEMORY.md`.
