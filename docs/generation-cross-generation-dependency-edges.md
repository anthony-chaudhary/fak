---
title: "Cross-Generation Dependency Edges"
description: "The metadata and reporting shape for a dependency that crosses a generation boundary — a gen/now leaf that waits on a gen/second-next seam, or a second-next bet whose prerequisites are already shipping in gen/next. Defines how to record the edge (issue-body + commit sidecar), which direction it points, and how it surfaces in the milestone report's generation debt so a stalled cross-horizon bet is visible without rereading the epic."
---

# Cross-Generation Dependency Edges

**Issue:** #1655.
**Parent:** #1625.
**Stream:** `gen/second-next`.
**Milestone:** Generation G2 - Second Next Gen.
**Status:** design memo / metadata + reporting contract for dependencies that
cross a generation boundary. Architectural option: expose only through this doc
and an optional, default-off report field; never a default runtime gate and
never a per-generation branch.

This memo answers one question #1625 raised but did not resolve: work at one
horizon routinely depends on work at another — a `gen/now` leaf that cannot ship
until a `gen/second-next` seam lands, or a `gen/second-next` bet whose promotion
trigger is "these prerequisites are all shipped" (the *dependency edge resolving*
promotion trigger named in
[`docs/generation-second-next-option-contracts.md`](generation-second-next-option-contracts.md)
§4). The generation contract names that trigger but never says **how the edge is
recorded** or **how a stalled cross-horizon dependency becomes visible**. This
memo defines both, without forking the trunk or inventing a new label taxonomy.

The canonical stream taxonomy, promotion verbs, and the flat debt metric live in
[`docs/generation.md`](generation.md); this memo is the *edge* refinement of the
debt metric's `unpromoted_bets` and `missing_witnesses` inputs.

## What a cross-generation edge is

A **cross-generation dependency edge** is a directed relationship between two
tracked generation items whose stream labels differ. It has three parts:

- a **from** item (the dependent — the work that is blocked),
- a **to** item (the dependency — the work that must land first), and
- a **direction across the horizon**: whether the dependent is *nearer* to `now`
  than the dependency (a **forward bet**) or *farther* (a **prerequisite pull**).

The two directions carry different risk and are reported differently:

| Direction | Shape | Example | Risk it signals |
|---|---|---|---|
| **forward bet** | nearer-horizon `from` waits on farther-horizon `to` | a `gen/now` leaf blocked on a `gen/second-next` seam | current-product work has taken a dependency on an *unpromoted* architectural option — the near work cannot ship until the far bet promotes, which the far bet's contract may never let it do. This is the dangerous edge. |
| **prerequisite pull** | farther-horizon `from` waits on nearer-horizon `to` | a `gen/second-next` bet whose promotion trigger is "these `gen/next` prerequisites ship" | normal and healthy — the far bet is correctly waiting on nearer foundations. When every `to` resolves, the edge *fires* the option's promotion trigger. |

The load-bearing distinction: a **forward bet** is a latent trunk hazard (near
work mortgaged to a far, unpromoted option), while a **prerequisite pull** is the
ordinary way a second-next option earns promotion. The reporting below weights
them accordingly.

## Metadata: how to record an edge

GitHub has no native typed, directed issue-to-issue dependency that agents can
read cheaply and portably, so the edge rides the mechanisms the generation
contract already uses — no new label, no new project field, no schema break.

### Issue-body field

The dependent (`from`) issue names the edge in its body, one line per edge:

```text
Gen-dep: blocked-by #1664 (gen/second-next)
```

- `Gen-dep:` is the key. `blocked-by` is the direction from the dependent's point
  of view (the only two verbs are `blocked-by` and, for a prerequisite pull that
  the far item records on itself, `blocks`).
- `#NNNN` is the other item; the parenthesized `(gen/*)` is that item's stream so
  a reader can classify the edge without a second issue fetch.
- Multiple edges are multiple `Gen-dep:` lines. An edge whose two endpoints carry
  the **same** stream is not a cross-generation edge and does not belong here — it
  is ordinary intra-lane sequencing.

### Commit sidecar

A commit that resolves a dependency records the fact in the body, alongside the
existing `Generation:` sidecar (never in the subject — subjects stay optimized
for the witness path, per [`docs/generation.md`](generation.md) §Commits And
Releases):

```text
feat(metrics): add cross-generation dependency edge metadata #1655 (fak metrics)

Generation: gen/second-next
Gen-dep: blocks #1655 (gen/now)
```

The sidecar reuses the normalization already applied to `Generation:` (`now` /
`next` / `second-next` / `future` → the matching `gen/*` label). A malformed
`Gen-dep:` line is advisory, not a commit blocker, so old commits keep working —
the same failure posture the `Generation:` sidecar already uses.

## Reporting: how an edge surfaces

The reporting home is the milestone report's generation section, computed in
[`internal/milestonereport`](../internal/milestonereport/milestonereport.go).
Today `GenerationRow.DebtScore` folds four cheap inputs:

```text
debt_score = stale_issues + 3*missing_witnesses + 2*unpromoted_bets + 2*label_ship_mismatches
```

A cross-generation edge maps onto this surface **without a new weight** until a
stronger witness is cheap to read:

- A **forward bet** (near work blocked on a farther, *unpromoted* item) is
  counted as an `unpromoted_bet` on the **dependency's** lane — the far item is,
  by definition, an option that nearer work is already waiting on, which is
  exactly what `unpromoted_bets` weights (`2×`). The dependent near item is *not*
  double-counted.
- A **prerequisite pull** whose `to` endpoints are all shipped is *promotion
  evidence*: it fires the option-contract promotion trigger and should **reduce**
  the far lane's debt, not add to it.
- An edge naming a `to` item the report cannot read (deleted, mislabeled, or in a
  view it cannot see) is a `missing_witness` (`3×`) — an unreadable dependency is
  the most dangerous kind, because the bet cannot be safely promoted or demoted.

The smallest honest reporting increment a future agent can land is a
`GenerationRow.ForwardBets int` count (forward-bet edges pointing *into* this
lane) surfaced in `DebtReason`, defaulting to `0` when no `Gen-dep:` lines are
present so a repo with no edges reports exactly as it does today. That keeps the
edge *observable* before it is *scored*, matching the debt metric's own
"cheap-proxy-first" posture.

## Orthogonality

An edge is planning metadata, not a branch, a priority, or a runtime switch.

- **Orthogonal to priority.** An edge records *sequence* ("this cannot land until
  that does"), never *value*. A high-priority near leaf can be blocked by a
  low-priority far seam and vice versa; the edge changes neither issue's priority.
  Priority still comes from labels, milestone, and operator decision.
- **Orthogonal to shared trunk.** Both endpoints land on `main` by explicit path
  under the normal DCO and ship-stamp rules. An edge never authorizes a
  per-generation branch, a side worktree, or a long-lived integration lane — the
  non-goal #1655 names explicitly. The reversible-seam discipline in the
  [option-contracts memo](generation-second-next-option-contracts.md) is what
  lets the far item touch current APIs additively instead of branching.
- **Orthogonal to runtime feature gates.** An edge says *why* one item waits on
  another; a feature gate decides whether the resolved code is reachable at
  runtime. A far seam can land inert behind a default-off gate while the edge that
  pointed at it resolves — the gate owns exposure, the edge owns sequence.

## Promotion evidence

An individual edge *resolves* (promoting the dependent) when:

- Every `to` endpoint of a **prerequisite pull** is shipped and witnessed, firing
  the far item's promotion trigger; the recheck note records the resolving
  commits. This is the `dependency edge` promotion trigger from
  [`docs/generation-second-next-option-contracts.md`](generation-second-next-option-contracts.md)
  §4, now with a metadata record instead of prose.
- A **forward bet** resolves when the far dependency promotes to a nearer horizon
  *on its own contract's evidence* (a passing simulation or an additive
  compatibility fixture) — never by the near work forcing the far label. If the
  near work needs to ship before the far bet can honestly promote, that is a
  demotion signal on the edge, not a promotion.

The mechanism itself promotes `second-next → next` when the `ForwardBets` count
lands in `GenerationRow` with a focused `internal/milestonereport` test proving a
forward-bet edge raises the dependency lane's `DebtReason` and a resolved
prerequisite pull does not.

## Demotion / retirement evidence

The edge mechanism (or an individual edge) demotes or retires under the closed
sunset-trigger vocabulary in
[`docs/generation-future-sunset-criteria.md`](generation-future-sunset-criteria.md),
cited by exactly one token with the four-piece retirement evidence:

- `ASSUMPTION_FIRED` — the invalidating assumption below fails: cross-generation
  edges turn out rare or unstable enough that the flat `unpromoted_bets` proxy is
  already sufficient and the `Gen-dep:` field is dead metadata.
- `SUPERSEDED` — GitHub ships a readable native typed-dependency surface (or the
  project adopts one) that agents can query more cheaply than parsing issue
  bodies; retire the `Gen-dep:` convention in favor of it.
- `STALE_RECHECK` / `CARRY_EXHAUSTED` — the edges recorded rot (endpoints closed,
  relabeled, or abandoned) and no one re-witnesses them within the recheck
  cadence, so the metadata misleads more than it informs.

A retirement names the token and the witness that fired it; moving labels alone
is a hidden demotion and is re-opened.

## Invalidating assumption

**The dangerous edges are rare and mostly forward bets.** This memo assumes that
cross-generation dependencies worth recording are (a) infrequent enough that
per-edge issue-body metadata is cheaper than a dedicated tracking surface, and
(b) dominated by the forward-bet direction, which is why the reporting folds them
into the existing `unpromoted_bets` weight rather than adding a new one. **This is
the assumption most likely to fail.** If real backlogs turn out to be dense webs
of cross-generation edges, or if prerequisite-pull edges dominate and need their
own resolution tracking, then the flat debt fold hides more than it shows and the
edge should graduate to a first-class structure — a `fak generation edges` view
that reads every `Gen-dep:` line into a directed graph with per-edge status — not
a fifth term bolted onto `debt_score`. Retire this memo's issue-body convention
in favor of that view if the assumption fires; do not defend the convention with
hand-maintained edge lists.

Secondary assumptions, stated so a later agent can check them cheaply:

1. **Issue bodies stay agent-readable and editable.** The `Gen-dep:` field
   assumes `gh issue view`/`edit` remain cheap. If they do not, the commit
   sidecar is the durable fallback and the issue-body field is dropped.
2. **Two directions are enough.** `blocked-by` / `blocks` assume every honest
   cross-generation edge is a simple precedence. A genuinely bidirectional or
   optional ("would benefit from, not blocked by") relationship is out of scope
   and should be modeled as ordinary issue text, not stretched onto this field.

## Handoff (continue from here without the epic)

A future agent picking this up should:

1. **File the reporting increment as its own `gen/next` issue** — add
   `GenerationRow.ForwardBets` to `internal/milestonereport` with a focused test
   (forward bet into a lane raises `DebtReason`; resolved prerequisite pull does
   not), reading `Gen-dep:` lines from the same issue specs the report already
   ingests. This is the code witness that promotes this memo `second-next → next`.
2. **Add a `Gen-dep:` line to the two or three real open cross-generation edges**
   already latent under [#1625](https://github.com/anthony-chaudhary/fak/issues/1625)
   (any `gen/now` leaf whose body already says "blocked on the seam from …") so
   the first report run has live input, not a synthetic fixture.
3. **Leave the option-contract, compatibility-policy, and sunset-vocabulary docs
   untouched** — this memo reuses their promotion trigger, additive-seam promise,
   and closed kill-trigger vocabulary; it does not fork them.

Until step 1 lands, this is planning metadata: the edge is *recordable* and
*orthogonal* today, and *scored* once `ForwardBets` ships.
