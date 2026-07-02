---
title: "The queried-harness-overlay schema — fill the overlay tier by query, not menu, and the base holds no bodies"
description: "The engine-free selection contract for Rung 3 of the system-prompt MMU (#1261 / C3): given a turn's INTENT and a token BUDGET over a catalog of at-rest capability cards (refs only, body-on-demand), which cards are FAULTED IN, which are MASKED, and is an identical re-invocation a HIT? It returns fault | hit | refuse(reason), the reason from a closed vocabulary. Three closed sets make it portable: the capability-kind set, the disposition set, and the refuse-reason set — an out-of-set token is refused at the authoring boundary, never selected silently. It is fail-closed and pointers-only: a card may NOT carry a body at rest (invariant 4), an empty intent is refused (query, not menu), a HIT re-faults nothing, the breakpoint does not move when the overlay changes (the Rung-2 assertion stays green), and a CapRef is defined ONCE — the canonical capindex.CapRef (the #1144 fold). The honest fence: this is the CONTRACT Rung 3 implements; the live caller of contextq.QueryCapabilities / capindex.Catalog does NOT exist yet — Rungs 1 (#1259, the internal/syspromptmmu overlay tier) and 2 (#1260, the splice breakpoint) are open, so the keystone has no request-path home to append after — but the substrate it calls (QueryCapabilities, Catalog.Query, the CapabilityLedger, SkillContextRecord's HIT-on-reinvocation) is shipped and offline-witnessed."
---

# The queried-harness-overlay schema

An agent's tool menu degrades as it grows. Past a few dozen descriptors the model picks
worse, not better — the listing itself crowds out the work, and retrieval becomes the new
ceiling (RAG-MCP, arXiv 2505.03275; Toolshed, 2410.14594). The field's answer is *active
discovery*: do not paste the whole menu into the head; let the turn's **intent** query a
cheap index and fault in only the cards it needs (MCP-Zero). Anthropic's Agent Skills make
the same move structurally — ~100 tokens of metadata per skill resident, the body read only
on trigger (progressive disclosure). The overlay tier of the
[system-prompt MMU](../notes/SYSTEM-PROMPT-MMU-2026-06-29.md) is exactly this discipline
applied to the *harness's* contributions: its skill / MCP / A2A menu and situational rules
are `capindex.CapCard`s at rest, appended **after** the cache breakpoint, faulted in by the
turn's query and masked — never mutated — when availability changes.

fak already ships the machinery — `capindex.Catalog.Query(intent)` ranks cards,
`contextq.QueryCapabilities(resolvers, req, ledger)` faults winners under a budget, the
`CapabilityLedger` tracks residency, and `contextq.SkillContextRecord` serves an identical
re-invocation as a HIT. What it does *not* yet have is a **live caller** on the request
path: the resolvers have no turn-path home, because the overlay tier they would append to
([`internal/syspromptmmu`](../notes/SYSTEM-PROMPT-MMU-2026-06-29.md), Rung 1 / `#1259`) and
the splice breakpoint they would append *after* (Rung 2 / `#1260`) are not built yet. This
page is the contract that pins what that caller must do — written domain-free so the
system-prompt MMU's Rung 3 (`#1261` / `C3`) implements it and any agent runtime with a
queried capability overlay can satisfy the same floor. It is the selection sibling of [the
witness-gated system-prompt-mutation schema](system-prompt-mutation-schema.md) (which gates
an *edit* to the base) and [the portable context-contract schema](context-contract-schema.md)
(which certifies a *view* is a witnessed fold); same recipe — closed vocabulary,
evidence-bound, fail-closed, data-not-code — pointed at a third decision: *which cards does
this turn's intent fault into the overlay, and is an identical re-invocation a hit?*

## The schema

A queried overlay is a small set of nouns. None of them runs a model — the schema is the
*selection* shape, and the selection is taken before any body is faulted in.

### OverlayQuery — the turn's intent against the at-rest catalog

```
OverlayQuery {
  version       : "fak-overlay/v1"   // schema tag (optional)
  intent        : string             // the turn's query (MCP-Zero active discovery); REQUIRED, non-empty
  token_budget  : int >= 0           // the overlay budget; winners faulted in up to it, the rest masked; REQUIRED
  breakpoint_at : int >= 0           // the resident-base token count = the Rung-2 cache-breakpoint offset; REQUIRED
  catalog       : [CapCard]          // the at-rest cards — refs + trigger only, NO bodies; REQUIRED, may be empty
  prior_digest  : string             // OPTIONAL: a prior overlay digest, for the HIT-on-reinvocation check
}
```

`intent` is required and non-empty: the overlay is filled **by query, not menu**, so an
empty intent (a request to dump the whole catalog) is refused `NO_INTENT`, never expanded.
`breakpoint_at` is the immovable seam — the overlay is appended *after* it, and a selection
that moves it is refused `BREAKPOINT_MOVED` (the Rung-2 byte-prefix assertion must stay
green). `catalog` may grow without bound; the resident base stays flat (the 0-for-∞ property
below), because the cards are pointers.

### CapCard — the tiny at-rest card (≈100 tokens, body-on-demand)

```
CapCard {
  ref         : CapRef     // {kind, name, version} — the canonical capindex.CapRef
  digest      : string     // sha256 over the body (the sync key + the HIT key)
  trigger     : string     // the trigger clause the query ranks against; REQUIRED
  tags        : [string]   // OPTIONAL ranking/filtering tags
  card_tokens : int >= 0   // the at-rest card cost (~100); bounded — page the index once it grows
}
```

A `CapCard` is a **closed object with no `body` field** (`additionalProperties: false`). That
is the structural form of invariant 4: the base holds pointers, not bodies. A card that
carries its body at rest is refused `BODY_AT_REST` at the authoring boundary — the body
faults in only when the query selects the card (`capindex.Capability.Materialize`, at most
once per capability). The resident descriptor set is *itself* bounded (`card_tokens` is a
cost, not free); once it grows, the index pages too.

### CapRef — the one canonical reference (the #1144 fold)

```
CapRef {
  kind    : CapKind   // CLOSED: "skill" | "mcp-tool" | "a2a-agent"  (the shipped capindex.CapKind)
  name    : string    // the capability name; REQUIRED
  version : string    // OPTIONAL; empty = latest
}
```

`CapRef` is defined **once** in the schema (`$defs/CapRef`) and referenced everywhere a
capability is named — in a card, in `selected`, in `masked`. That is the schema-level form of
the `#1144` reconciliation: there is **one** `capindex.CapRef`, never the duplicate copy that
lives in `contextq` today. A payload that resolves a ref to two distinct definitions is
refused `DUPLICATE_CAPREF`. `kind` is the shipped `capindex.CapKind` taxonomy exactly
(`skill` / `mcp-tool` / `a2a-agent`); the harness's situational rules ride as cards of one of
these kinds. An out-of-set kind is `UNKNOWN_KIND`.

### OverlayDecision — the reviewable fault | hit | refuse

```
OverlayDecision {
  query           : OverlayQuery    // OPTIONAL echo of the input
  disposition     : "fault" | "hit" | "refuse"
  selected        : [CapRef]        // the faulted-in winners (refs only) — only what the intent needs
  masked          : [CapRef]        // the masked-out cards — masked, NOT removed from the resident array
  overlay_digest  : string          // the digest of the selected set — the key an identical re-invocation matches
  base_tokens     : int >= 0        // the resident-base count AFTER selection — MUST equal breakpoint_at (flat)
  selected_tokens : int >= 0        // the overlay tokens faulted in — MUST be <= token_budget
  faulted         : int >= 0        // how many bodies were paged in — 0 on a HIT (the no-re-fault proof)
  reason          : RefuseReason    // REQUIRED on "refuse"; from the closed set below
}
```

Resolving an `OverlayQuery` yields an `OverlayDecision`: the `fault`/`hit`/`refuse` verdict,
the faulted-in `selected` refs, the `masked` refs, the resulting `overlay_digest`, and the
accounting that makes the acceptance checkable. The decision is data: reviewable, diffable,
produced with no model in the loop. A `hit` MUST have `faulted == 0` (the schema enforces it)
— an identical re-invocation re-faults nothing. `base_tokens` MUST equal the query's
`breakpoint_at` regardless of catalog size; `selected_tokens` MUST be within the budget.

### The refuse vocabulary

`RefuseReason` is **closed and additive** — a new reason is a new named value plus a decision
arm, never a free-text field. Every token names a load-bearing constraint of the overlay
tier:

| `reason` | when | constraint it holds |
|---|---|---|
| `NO_INTENT` | `intent` is empty — a menu dump, not a query | fill by query, not menu (MCP-Zero) |
| `BODY_AT_REST` | a catalog card carries a body at rest | inv. 4 (the base is pointers, not bodies) |
| `UNKNOWN_KIND` | a `ref.kind` not in `{skill, mcp-tool, a2a-agent}` — fail-closed | the closed `capindex.CapKind` set |
| `DUPLICATE_CAPREF` | a `CapRef` resolves to two distinct definitions | `#1144` (one canonical `capindex.CapRef`) |
| `BREAKPOINT_MOVED` | the realized `base_tokens != breakpoint_at` | the Rung-2 byte-prefix assertion stays green |
| `OVER_BUDGET` | the faulted set's `selected_tokens > token_budget` | the budget bounds the overlay (mask the overflow) |

A token outside this set is **not** silently coerced to a select. A malformed query fails at
the authoring boundary (the closed enums / `additionalProperties: false` reject it) or returns
a fail-closed refusal at selection time — never a quiet pass.

### The decision table — domain-free, deterministic, fail-closed

The selection is a pure function of the query shape. Read top to bottom; the first matching
arm wins; every non-select arm refuses, never a silent serve:

```
1. intent is empty                                    -> Refuse(NO_INTENT)        (query, not menu)
2. a catalog card carries a body at rest              -> Refuse(BODY_AT_REST)     (inv. 4: pointers, not bodies)
3. a card ref.kind ∉ {skill, mcp-tool, a2a-agent}     -> Refuse(UNKNOWN_KIND)
4. a CapRef resolves to two distinct definitions      -> Refuse(DUPLICATE_CAPREF) (#1144: one canonical CapRef)
5. prior_digest == the selected-set overlay_digest    -> Hit (faulted = 0; an identical re-invocation re-faults nothing)
6. rank the catalog by intent; fault winners until the next would exceed token_budget, mask the rest:
     a. the faulted set's tokens > token_budget        -> Refuse(OVER_BUDGET)      (never serve over budget)
     b. the realized base_tokens ≠ breakpoint_at        -> Refuse(BREAKPOINT_MOVED) (the overlay must not move the seam)
     c. otherwise -> Fault(selected, masked, base_tokens = breakpoint_at, overlay_digest)
```

It is **append-and-mask, never mutate**: an unselected card is `masked` (kept in the
resident array, hidden from the turn), not deleted — availability changes by masking the
overlay set, not by editing the resident array (invariant 2). It is **fail-closed**: the
absence of an affirmative `fault`/`hit` is a refusal. And it is **flat in the catalog size**:
`base_tokens` is `breakpoint_at` whether the catalog holds three cards or three thousand,
because only pointers are resident.

## The four contracts (the acceptance, made checkable)

This selection is portable because it holds four properties an external runtime can verify
without fak's kernel — the issue's four acceptance clauses, each a check over the on-disk
fixtures:

1. **Fault only what the intent needs; the base holds no bodies, flat as the catalog grows
   (the 0-for-∞).** `selected` is a subset of the catalog refs;
   [`overlay-decision-fault.json`](fixtures/overlay-decision-fault.json) faults in 2 of 3
   cards and masks the third, with `base_tokens == breakpoint_at`. Doubling the catalog to
   six cards ([`overlay-query-grown.json`](fixtures/overlay-query-grown.json) →
   [`overlay-decision-grown.json`](fixtures/overlay-decision-grown.json)) selects the **same**
   two and leaves `base_tokens` **unchanged** — the resident base is flat as the catalog
   grows. And no card may carry a body at rest (the `CapCard` `additionalProperties: false`
   fence; [`card-body-at-rest.json`](fixtures/queried-overlay-invalid/card-body-at-rest.json)
   is rejected).
2. **Re-invocation with an identical digest is a HIT (no re-fault).**
   [`overlay-query-hit.json`](fixtures/overlay-query-hit.json) carries a `prior_digest` equal
   to the fault case's `overlay_digest`; the decision
   ([`overlay-decision-hit.json`](fixtures/overlay-decision-hit.json)) is `hit` with
   `faulted == 0` — the `contextq.SkillContextRecord` HIT-on-reinvocation property, enforced
   by the schema's `disposition == hit ⇒ faulted == 0` conditional.
3. **One canonical `CapRef` (the #1144 fold).** The schema defines `CapRef` exactly once and
   `$ref`s it from the card, from `selected`, and from `masked` — there is no second copy to
   drift. A runtime that carries the `contextq` duplicate is refused `DUPLICATE_CAPREF`.
4. **The breakpoint does not move when the overlay changes (the Rung-2 assertion stays
   green).** `base_tokens` is identical across the fault, hit, and grown decisions and equal
   to each query's `breakpoint_at` — the overlay is appended after the seam, and the seam does
   not move when the overlay does.

The vocabularies are closed and validatable: the kind set, the disposition set, and the
refuse-reason set are finite enums; a validator decides membership with a finite switch, not a
lookup against a live service. The schema is published as a machine-checkable JSON Schema —
[`queried-overlay-schema.json`](queried-overlay-schema.json) (Draft 2020-12) — so any runtime
authors and validates a query with an off-the-shelf validator, **no fak engine present**: the
positive fixtures validate against it, the five
[negative fixtures](fixtures/queried-overlay-invalid/) are each rejected at the boundary, and
the [validation recipe](fixtures/queried-overlay-invalid/README.md) runs the whole round-trip
(three queries + three decisions accepted, five negatives rejected) **and asserts the four
acceptance clauses as cross-field checks** — so "checkable" is checked, not asserted.

## The round-trip, as data

The schema's whole claim is that author → select → review is data, not narration. The
fixtures under [`fixtures/`](fixtures/) are the on-disk witness — each `OverlayQuery` paired
with the `OverlayDecision` a selector with no model and no fak engine produces:

- [`overlay-query-fault.json`](fixtures/overlay-query-fault.json) →
  [`overlay-decision-fault.json`](fixtures/overlay-decision-fault.json) — the **fault** case:
  a diff-review intent over a 3-card catalog faults in the two diff-related cards and masks
  the release-notes agent, within budget, `base_tokens` flat at the breakpoint.
- [`overlay-query-hit.json`](fixtures/overlay-query-hit.json) →
  [`overlay-decision-hit.json`](fixtures/overlay-decision-hit.json) — the **hit** case: the
  same intent with a matching `prior_digest`, served as `hit` with `faulted == 0`.
- [`overlay-query-grown.json`](fixtures/overlay-query-grown.json) →
  [`overlay-decision-grown.json`](fixtures/overlay-decision-grown.json) — the **0-for-∞**
  case: the catalog doubled to six cards, the same two selected, `base_tokens` unchanged — the
  resident base is flat as the catalog grows.

A reviewer reads the six files and the table binding them; no model and no fak engine are
needed to confirm the selection.

## Reference implementation and witness

The contract rests on a query + residency + cache-coherence substrate that is **shipped and
offline-witnessed**, and a *live caller* on the request path that is **not yet** — Rungs 1–4
of the system-prompt MMU are open, this issue (`#1261`) being Rung 3. The table is honest
about which is which:

| Schema element | Reference stick | Status |
|---|---|---|
| The query primitive (intent → ranked cards, fault under budget) | [`contextq.QueryCapabilities`](https://github.com/anthony-chaudhary/fak/blob/main/internal/contextq/contextq.go) (MCP-Zero active discovery) + the `CapabilityLedger` | [SHIPPED] |
| The 0→∞ overlay index (harness items as queryable cards) | [`capindex.Catalog.Query(intent)`](https://github.com/anthony-chaudhary/fak/blob/main/internal/capindex/catalog.go) + [`capindex.CapCard`](https://github.com/anthony-chaudhary/fak/blob/main/internal/capindex/capindex.go) (cards at rest, body-on-demand via `Materialize`) | [SHIPPED] |
| The canonical `CapRef` (the #1144 fold) | [`capindex.CapRef`](https://github.com/anthony-chaudhary/fak/blob/main/internal/capindex/capindex.go) `{Kind, Name, Version}` | [SHIPPED] (the `contextq` duplicate is the debt this folds) |
| HIT-on-reinvocation (no re-fault on an identical digest) | [`contextq.SkillContextRecord`](https://github.com/anthony-chaudhary/fak/blob/main/internal/contextq/skillmemory.go) (procedural-memory view keyed by invocation digest) | [SHIPPED] |
| The pager for an evictable overlay body | [`ctxmmu`](https://github.com/anthony-chaudhary/fak/tree/main/internal/ctxmmu) `PageOutBody` + CAS-pinned eviction | [SHIPPED] |
| The cache-breakpoint vocabulary the overlay is appended after | [`cachemeta.PromptSegment`](https://github.com/anthony-chaudhary/fak/tree/main/internal/cachemeta) + `InjectBreakMarker` | [SHIPPED] |
| `internal/syspromptmmu` base-context overlay tier (where cards append) | Rung 1 (`#1259`, keystone) | **not yet** |
| Segment-plan → `promptmmu` splice adapter (the breakpoint to append after) | Rung 2 (`#1260`) | **not yet** |
| The live caller that drives `QueryCapabilities` from the turn on the request path | Rung 3 (`#1261`, this issue) | **not yet** (the substrate above is request-path-dead until this lands) |

## Honest fences

- **This page is the CONTRACT, not the live caller.** The substrate
  (`contextq.QueryCapabilities`, `capindex.Catalog`, the `CapabilityLedger`,
  `SkillContextRecord`) is built and tested; what does not exist is a turn-path caller, because
  the overlay tier it appends to (`internal/syspromptmmu`, Rung 1 / `#1259`) and the splice
  breakpoint it appends after (Rung 2 / `#1260`) are open. This page pins the selection
  decision Rung 3 must satisfy *before* the wiring lands, so the acceptance is a fixed target,
  not a post-hoc rationalization — the same discipline the sibling
  [system-prompt-mutation](system-prompt-mutation-schema.md) and
  [context-contract](context-contract-schema.md) schemas keep.
- **The schema validates shape, not ranking quality.** The contract gates *that* the selector
  is pointers-only, fail-closed, budget-bounded, breakpoint-stable, and HIT-correct; it does
  not decide *which* cards a given intent should rank highest — that is the ranker's job
  (`capindex.Catalog.Query`), measured separately. A runtime that faults in the wrong cards
  still satisfies the shape; the shape is the floor, not the ranker's quality bar.
- **`base_tokens == breakpoint_at` is asserted by the decision, not derived by the schema.**
  JSON Schema cannot cross-reference the query's `breakpoint_at` from inside the decision
  document, so the flat-base and budget invariants live in the decision table and are checked
  by the [recipe's cross-field assertions](fixtures/queried-overlay-invalid/README.md), not by
  `is_valid` alone. The schema enforces what it can structurally (no body at rest, closed
  kinds, `hit ⇒ faulted == 0`); the recipe enforces the rest.
- **`DUPLICATE_CAPREF` is a contract refusal, not yet a code-level merge.** The schema models
  the single-`CapRef` end state and refuses a payload that carries two; the actual deletion of
  the `contextq` duplicate `CapRef`/`Capability` types is the `#1144` code follow-on in
  `internal/contextq`, on the `model`/`contextq` lane — not this docs increment.

## Cross-references

- [`queried-overlay-schema.json`](queried-overlay-schema.json) — the machine-checkable JSON Schema (Draft 2020-12): author and validate an overlay query with any off-the-shelf validator, no fak engine present.
- [`fixtures/queried-overlay-invalid/README.md`](fixtures/queried-overlay-invalid/README.md) — the engine-free validation recipe (three queries + three decisions accepted, five negatives rejected, the four acceptance clauses asserted) that makes the contract checkable.
- [The system-prompt MMU design note](../notes/SYSTEM-PROMPT-MMU-2026-06-29.md) — the epic (`#1258`) this is Rung 3 of: the fak-first base-context layout, the five load-bearing invariants, and the six rungs.
- [The witness-gated system-prompt-mutation schema](system-prompt-mutation-schema.md) — the Rung-5 sibling (gates an *edit* to the base); same recipe, a different decision.
- [The portable context-contract schema](context-contract-schema.md) — the materialized-view sibling (certifies a *view* is a witnessed fold); same recipe, a different decision.
- [`internal/contextq`](https://github.com/anthony-chaudhary/fak/blob/main/internal/contextq/contextq.go) · [`internal/capindex`](https://github.com/anthony-chaudhary/fak/blob/main/internal/capindex/capindex.go) — the shipped query primitive, the at-rest card index, and the canonical `CapRef` this contract drives and folds.
