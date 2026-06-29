---
title: "The portable agent-routing schema — per-aspect routing + ensembles as data any orchestrator can author"
description: "The engine-free routing schema that lifts fak's per-aspect routing + ensembles (internal/modelroute) into a domain-free contract another fleet can author against, and the `dos route` verb implements. The routed unit is an *aspect* of a request (the whole request, one tool call, a sub-query, a planner state, a reasoning step), not the whole request; an ensemble (a set of models folded by a closed Reduction: first / vote / best_of / all_reduce / concat) is a first-class plan. Three contracts make it portable: (1) it round-trips author → plan → review as data with no model and no fak engine in the loop; (2) the Reduction set is a closed, validatable vocabulary — an out-of-set token fails the manifest loud (the UNCLASSIFIED-refuse form), never a silent default; (3) the plan is deterministic over the request shape — the first matching rule wins, a fail-closed default applies when none match, so the same Subject yields the same Plan. The honest fence: this lifts the routing *decision/schema*, which is SHIPPED and offline-witnessed by `internal/modelroute`; live multi-model dispatch is [STUB], and the `dos route` verb is not yet."
---

# The portable agent-routing schema

Most LLM routers — OpenRouter, Portkey, LiteLLM's Router, Unify, Martian,
NotDiamond — answer one question: *given a whole request, which single model
should serve it?* They route at one granularity (the request) and pick one model.
fak routes at a finer granularity: the routed unit is an **aspect** of a request —
the whole request, **one tool call**, a sub-query, a planner state, or a reasoning
step — each to a different model, and an **ensemble** (a set of models folded by a
[`Reduction`](#the-reduction-vocabulary)) is a first-class plan, not a bolt-on. All
of it is one declarative, version-tagged file. That spine is shipped and offline-
witnessed in [`internal/modelroute`](../model-routing.md).

The grammar gap this page closes: that manifest is **fak-specific**. There was no
portable schema another fleet could author against without adopting fak's engine.
This page is that schema — the aspect taxonomy and the ensemble/reduction vocabulary
written as a domain-free contract. It is `G8` of the
[agent-programming-grammar epic](../notes/CONCEPT-AGENT-PROGRAMMING-GRAMMAR-2026-06-28.md)
(#1215), the routing sibling of [net-true-value](net-true-value.md) (a value claim),
[the observer-effect contract](observer-effect.md) (a cost number), and
[the support-maturity honesty fence](support-maturity-honesty-fence.md) (a maturity
claim). The verb that walks this schema over a request is `dos route`; its home is
the installed DOS package, and it is **not yet** (see the [honest fences](#honest-fences)).
fak's [`internal/modelroute`](../model-routing.md) is the reference implementation —
the offline witness that the schema below is real, not a wish.

## The schema

A routing policy is a small set of nouns. None of them mentions a model output, a
network, or a fak package — the schema is the *decision* shape, and the decision is
taken before any model runs.

### Subject — the thing being routed, at any granularity

```
Subject {
  aspect        : Aspect        // OPEN set — the granularity axis
  tool          : string        // the tool name when aspect == "tool_call"
  prompt_tokens : int           // estimated prompt length
  latency       : Latency       // CLOSED: "" | "interactive" | "batch"
  complexity    : Complexity     // CLOSED ordered: "" < "low" < "medium" < "high"
  labels        : map[str]str   // OPEN signals: domain, language, tenant, taint, …
}
```

`aspect` is an **open** set — the named values (`request`, `tool_call`, `query`,
`state`, `step`, `scout`) are the well-known granularities, but a deployment may
route its own named stage by using any string. This openness is the whole point:
routing one request at *more than one* level is what a request-only router cannot
express. `latency` and `complexity` are **closed** vocabularies, and `complexity` is
ordered so a rule can set a floor (route only the *hard* aspects to the big model).
`labels` carries open signals a deployment matches on without a code change.

### Plan — a single pick, or an ensemble + a reduction

```
Member { model: string, weight: float, role: string }   // role: primary | drafter | verifier | judge | …

Plan {
  members : [Member]    // len == 1 is a single PICK (the SOTA shape); len > 1 is an ENSEMBLE
  reduce  : Reduction   // REQUIRED when len(members) > 1
  scout   : string      // optional cheap model that classifies the subject FIRST
  reason  : string      // free-text note surfaced in the decision trace
}
```

`len(members) == 1` is the request-level router's whole world — pick one model.
`len(members) > 1` is an ensemble: a set of models on one item, folded into one
answer by a `Reduction`. The ensemble is a first-class plan in the schema, so an
orchestrator that authors this schema gets ensembles for free, not as a separate
tool.

### The Reduction vocabulary

`Reduction` is how an ensemble's member outputs fold into one result. It is a
**closed, additive** set — a new reduction is a new named value plus a fold rule,
**never** a free-text field:

| `reduce` | fold | use |
|---|---|---|
| `first` | take the first member's output | fastest-wins / fallback chain |
| `vote` | weighted majority over discrete answers | self-consistency / quorum |
| `best_of` | the highest-scored member's output | a judge/verifier scores; best wins |
| `all_reduce` | weighted mean of the members' **scalar** outputs | numeric answers (a score, a count, a probability) — *scalars only*; a non-numeric output is an error, not a silent guess |
| `concat` | concatenate the members' outputs in member order | fan-out gather |

A token outside this set is **not** silently coerced to a default. Validation
rejects the manifest loud — the schema's form of "an out-of-set token is
`UNCLASSIFIED` and refused conservatively." A misconfigured fold must fail at the
authoring boundary, never fall through to a quiet behaviour at dispatch.

### Match → Rule → Manifest — the ordered, fail-closed policy

```
Match { aspect, tool, min_prompt_tokens, max_prompt_tokens, latency, min_complexity, labels }
        // every SET field must hold (logical AND); an unset field is a wildcard;
        // `tool` allows a single trailing "*" prefix wildcard ("write_*").

Rule     { name: string, match: Match, plan: Plan }   // names are unique

Manifest {
  version : string    // identifies the implementation; absent == current
  default : Plan       // REQUIRED, fail-closed — applied when NO rule matches
  rules   : [Rule]     // evaluated top-to-bottom; FIRST match wins
}
```

Rules are ordered and the **first match wins**, so the most specific rule goes
first. When no rule matches, the **fail-closed `default`** plan applies — a manifest
with no default, or a plan with zero members, is invalid. That is the routing
analogue of the capability floor's default-deny: absence of an affirmative route is
not a free pick, it is the declared default.

### Decision — the reviewable output

Routing a `Subject` through a `Manifest` yields a `Decision`: the echoed subject,
which rule fired (or that the default was used), and the chosen `Plan`. The decision
is data — reviewable, diffable, and produced with no model in the loop.

## The three contracts (the acceptance, made checkable)

This schema is portable because it holds three properties an external authoring tool
can check without fak's engine:

1. **It round-trips with no engine.** author (write a `Manifest`) → plan (route a
   `Subject` → a `Decision`) → review (read the `Decision` as data) is a pure
   data transform. No model runs; no network is touched. The
   [round-trip below](#the-round-trip-as-data) is the witness, on disk as fixtures.
2. **The reduction set is a closed, validatable vocabulary.** The five values in the
   [table above](#the-reduction-vocabulary) are the whole set; an ensemble must carry
   one; an unknown token fails validation. A validator decides membership with a
   finite switch, not a lookup against a live service.
3. **The plan is deterministic and reviewable.** Routing is a pure function of the
   subject shape and the manifest: the same `Subject` against the same `Manifest`
   always yields the same `Decision`. "Deterministic" is scoped to the routing
   *decision* and the *fold* — never to the member models' (non-bit-exact) outputs.
   The fold preserves member order, so the order-sensitive reductions (`concat`,
   `vote` tie-break) stay deterministic.

## The round-trip, as data

The schema's whole claim is that author → plan → review is data, not narration. The
fixtures under [`fixtures/`](fixtures/) are the on-disk witness:

- [`route-schema-manifest.json`](fixtures/route-schema-manifest.json) — the **author**
  artifact: a portable manifest with a fail-closed default and rules covering a single
  pick plus a `vote`, a `best_of` (with a scout), and a `concat` ensemble.
- [`route-schema-plan.json`](fixtures/route-schema-plan.json) — the **review**
  artifact: the `Decision` produced by routing one `Subject` (a `tool_call` to a
  `write_*` tool) through that manifest. Same subject shape ⇒ this same plan, every
  time.

A reviewer reads the two files and the rule binding them; no model and no fak engine
are needed to confirm the route. fak's [`internal/modelroute`](../model-routing.md)
parses the same manifest, routes the same subject, and produces the same decision —
offline, witnessed by `go test ./internal/modelroute/`, which is the reference
implementation's proof that the three contracts hold.

## Reference implementation and witness

| Schema element | Reference stick (`internal/modelroute`) | Status |
|---|---|---|
| Subject / Aspect / Latency / Complexity | `Subject`, `Aspect` (open), `validLatency` / `validComplexity` (closed) | [SHIPPED] |
| Plan / Member / ensemble | `Plan`, `Member`, `Plan.IsEnsemble()`, `Plan.Primary()` | [SHIPPED] |
| Closed Reduction vocabulary | `Reduction` constants + `knownReduction` + `Combine` fold | [SHIPPED] |
| Ordered, fail-closed Manifest | `Manifest.Route` (first-match-wins), `validatePlan` (≥ 1 member) | [SHIPPED] |
| Round-trip (author ↔ review) | `Manifest.JSON` / `ParseManifest` (`DisallowUnknownFields`) | [SHIPPED] |
| Offline determinism witness | `go test ./internal/modelroute/` (no model in the loop) | [SHIPPED] |
| Portable `dos route <request>` verb | the installed DOS package | **not yet** |
| Live multi-model dispatch | writes the chosen model to `abi.ToolCall.Engine`, runs the ensemble | [STUB] |

## Honest fences

- **This lifts the decision/schema, which is shipped; it does not claim live
  dispatch.** The routing *decision* and the ensemble *fold* are [SHIPPED] and
  offline-witnessed in `internal/modelroute`. The live multi-model **dispatch** — the
  layer that writes the chosen model to `abi.ToolCall.Engine`, runs each ensemble
  member as its own adjudicated call, and feeds the outputs into the fold — is
  [STUB], tracked on the model-routing claim itself in [`CLAIMS.md`](../../CLAIMS.md).
  This page is the *contract* that dispatch (and any external orchestrator) authors
  against, not a claim that dispatch runs.
- **The `dos route` verb is not yet.** The portable verb that walks this schema over
  a request lives in the installed DOS package (the grammar's home; see the
  [epic](../notes/CONCEPT-AGENT-PROGRAMMING-GRAMMAR-2026-06-28.md)). This page is the
  schema it implements; the verb is a named follow-on, not shipped here.
- **The wiring contract is load-bearing and stated, not yet enforced end to end.**
  When dispatch lands it must (a) **route before adjudicate** — write the model to
  `abi.ToolCall.Engine` *before* the kernel admits the call, so the residency floor
  adjudicates the real route and a sensitive payload bound for a remote model is
  denied, never fail-open; (b) expand an ensemble to **N independently-adjudicated
  calls**, never one fan-out that bypasses the floor; (c) gather member outputs **in
  member order** into the fold. These are the conditions any conforming implementation
  must keep; the schema names them so the later wiring cannot silently regress the
  default-deny floor.
- **"Deterministic" is scoped to the decision and the fold.** The member models'
  outputs come from non-bit-exact engines, so end-to-end answer reproducibility is
  not claimed — only that the same subject shape yields the same plan, and the same
  votes fold the same way.

## Cross-references

- [Model routing — first-class at every level](../model-routing.md) — fak's reference implementation of this schema, with the per-aspect spine, the surveyed-router comparison, and the live-dispatch `[STUB]`.
- [Routers & gateways](../integrations/routers.md) — why fak is a complement to a request-level router, the three topologies, and the residency floor that holds for every router.
- [The agent-programming grammar](../notes/CONCEPT-AGENT-PROGRAMMING-GRAMMAR-2026-06-28.md) — the epic this schema is `G8` of, and the recipe every lift keeps (closed vocabulary, evidence-bound, fail-closed, data-not-code, pays on both lenses).
- [Net-true-value](net-true-value.md) · [The observer-effect contract](observer-effect.md) · [The support-maturity honesty fence](support-maturity-honesty-fence.md) — the sibling standards in `docs/standards/`.
- [Claims ledger](../../CLAIMS.md) — shipped vs stub, claim by claim.
