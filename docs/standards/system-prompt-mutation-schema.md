---
title: "The system-prompt-mutation schema — witness-gated runtime modification of the base context, where the agent never edits its own spine"
description: "The engine-free admission contract for Rung 5 of the system-prompt MMU (#1263 / C5): a self-authored edit to the live base context (the `fak prompt` add/promote/demote/version verbs) may become resident only if it passes an INDEPENDENT witness, never targets the immutable spine or the edit-governing meta-rules, is an append-mostly versioned delta (never a full rewrite), and takes effect at the next prefix rebuild (never mid-prefix, so the warm cache is never busted). It returns admit | refuse(reason) | demote, the reason from a closed vocabulary. Four closed sets make it portable: the verb set, the tier set, the delta-kind set, and the refuse-reason set — an out-of-set token is refused at the authoring boundary, never admitted silently. It is fail-closed and append-mostly: the spine is off-limits to every verb, a full rewrite is inexpressible, an activating agent edit with no independent witness PASS is refused, and the agent never grades its own edit (invariant 5). A resident learned rule that later correlates with worse witnessed outcomes is auto-demoted (the guard-RSI worst-bucket shape) and leaves a journal row. The honest fence: this is the CONTRACT Rung 5 implements; the `fak prompt` verbs and internal/syspromptmmu are NOT yet — Rungs 1–5 are all open — but the witness substrate it rests on (the guard journal, `dos verify`, the promptmmu byte-identity prefix floor, `internal/sessionreset`) is shipped and offline-witnessed."
---

# The system-prompt-mutation schema

A self-improving agent that can rewrite its own base context has a failure mode no
benchmark catches until it is too late: it edits the very text that keeps it safe. The
literature names the two ways this goes wrong. *Context collapse* and *brevity bias* —
when an agent re-writes its whole instruction block in one pass, it compresses, and the
first thing a compressor drops is the load-bearing-but-verbose safety text (ACE,
2510.04618). And *misevolution* — a learned rule that scores well on the proxy the agent
optimizes drifts the agent away from the objective the operator actually holds (*Your
Agent May Misevolve*, 2509.26354). The field's answer is not "don't let the agent learn";
it is **Voyager's acceptance gate**: the agent may *propose* a skill, but an *independent*
verifier decides whether it is kept (2305.16291).

fak already ships that verifier discipline — the hash-chained guard journal
([`internal/abi`](https://github.com/anthony-chaudhary/fak/blob/main/internal/abi/events.go) `EvDecide`/`EvDeny`), the truth syscall
`dos verify`, and the guard-RSI worst-bucket routing
([`internal/guardrsi`](https://github.com/anthony-chaudhary/fak/blob/main/internal/guardrsi/guardrsi.go) `WorstBucket`). What it does
*not* yet have is a base context the agent can edit at all — so the gate has had nothing
to gate. This page is the contract that closes that gap: the admission decision for a
self-authored edit to the live base context, written domain-free so the
[system-prompt MMU's](../notes/SYSTEM-PROMPT-MMU-2026-06-29.md) Rung 5 (`#1263` / `C5`)
implements it and any agent runtime that grows a *live* system prompt can satisfy the same
floor. It is the runtime-modification sibling of [the portable context-contract
schema](context-contract-schema.md) (which certifies a *view* is a witnessed fold) and
[the portable taint-check schema](taint-check-schema.md) (which gates a *value* into a
sink); same recipe — closed vocabulary, evidence-bound, fail-closed, data-not-code —
pointed at a third decision: *may this edit to my own head become resident?*

The verb that walks this schema is `fak prompt`; it and its package
[`internal/syspromptmmu`](../notes/SYSTEM-PROMPT-MMU-2026-06-29.md) are **not yet** (see the
[honest fences](#honest-fences) — Rungs 1–5 are all open). This page is the contract they
must satisfy, grounded in the shipped witness + cache-stability substrate below — the
offline witness that the floor is real, not a wish.

## The schema

A system-prompt mutation is a small set of nouns. None of them mentions a model, a token,
or a fak package — the schema is the *decision* shape, and the decision is taken before
any byte of the base context changes.

### PromptEdit — the proposed self-modification

```
PromptEdit {
  verb          : Verb          // CLOSED: "add" | "promote" | "demote" | "version"
  target_tier   : Tier          // CLOSED: "spine" (OFF-LIMITS) | "policy" | "overlay"
  block_id      : string        // the base-context block being edited (never the spine)
  meta_rule     : bool          // is block_id an edit-GOVERNING meta-rule? OFF-LIMITS if true; default false
  delta_kind    : DeltaKind     // CLOSED: "append" | "version_swap" | "mask"  (NEVER a full rewrite)
  base_version  : int           // the version this delta applies to (the rollback chain); REQUIRED for version_swap
  author        : Author        // CLOSED: "operator" | "agent"
  witness       : Witness       // {grader, verdict}; REQUIRED for an activating agent edit
  effect_timing : EffectTiming  // CLOSED: "next_rebuild" (default) | "mid_prefix" (REFUSED)
  outcome       : Outcome       // OPTIONAL observed correlation of a RESIDENT rule: improved|neutral|regressed|unknown
}
```

The shape encodes the design note's three load-bearing invariants directly. `target_tier`
is the [system-prompt-MMU layout](../notes/SYSTEM-PROMPT-MMU-2026-06-29.md): the **spine**
is the immutable fak-authored head (the attention-sink anchor, byte-identical every turn);
the **policy** floor is the versioned deny/allow tier below it; the **overlay** is the
queried harness cards appended *after* the cache breakpoint. The spine is off-limits to
every verb — *no* `fak prompt` verb can target it, even with a witness pass (invariant 5).
`meta_rule` fences the same way over the *edit-governing* rules: the rules that decide what
may be edited are out of the agent's reach, so the agent cannot widen its own authority.

`delta_kind` is closed to the three *append-mostly* edits — a full rewrite is **not a
value**, so "re-author the whole block" cannot even be expressed (invariant 2 / the ACE
context-collapse defense). `base_version` is the rollback chain anchor: a `version_swap`
names the version it supersedes, the prior version is preserved (never overwritten), and
**rollback re-pins a prior `base_version` bit-for-bit**.

`author` + `witness` carry the Voyager gate. An `agent`-authored *activating* edit
(`add`/`promote`/`version` — anything that makes a rule active) must carry an independent
witness; a `demote` is removal and is always safe, so it needs none. `effect_timing`
encodes invariant 2's timing: an edit takes effect at the **next prefix rebuild**
(`internal/sessionreset` / `--reset-on-budget`), never mid-prefix — a `mid_prefix` edit is
refused because it would mutate the resident array mid-session and bust the warm cache.

### Witness — the independent acceptance gate

```
Witness {
  grader  : Grader   // CLOSED: "dos_verify" | "guard" | "operator" | "self"
  verdict : Verdict  // CLOSED: "pass" | "fail" | "absent"
}
```

The witness is *who graded the edit and what they said*. `dos_verify` and `guard` are the
independent witnesses; `operator` is a human reviewer. `self` is the editing author grading
its own edit — expressible so the check can **explicitly refuse it** for an agent author
(`SELF_GRADED`), rather than pretend it cannot happen. The success signal comes from a
party that is not the author (invariant 5).

### EditDecision — the reviewable admit | refuse | demote

```
EditDecision {
  edit         : PromptEdit     // the echoed input
  decision     : "admit" | "refuse" | "demote"
  reason       : RefuseReason   // REQUIRED on "refuse"/"demote"; from the closed set below
  effective_at : "next_rebuild" // when an admit takes effect (never mid-prefix)
  journal      : bool           // a demote / refused agent edit leaves a hash-chained journal row
  witness      : string         // a bounded, payload-free note (verb + tier + reason), never the block bytes
}
```

Checking a `PromptEdit` yields an `EditDecision`: the echoed edit, an
`admit`/`refuse`/`demote` verdict, and — on anything but `admit` — a closed reason. The
decision is data: reviewable, diffable, produced with no model in the loop. An `admit`
always takes effect at the next rebuild and is byte-stable until then (invariant 1). A
`demote` (and a refused agent edit) leaves a **journal row**, so the demotion signal is
durable and independent of the agent's self-report.

### The refuse/demote reason vocabulary

`RefuseReason` is **closed and additive** — a new reason is a new named value plus a
decision arm, never a free-text field. Every token names a load-bearing invariant of the
system-prompt MMU:

| `reason` | when | invariant it holds |
|---|---|---|
| `UNKNOWN_VERB` | `verb` not in `{add, promote, demote, version}` — fail-closed | closed verb set |
| `UNKNOWN_TIER` | `target_tier` not in `{spine, policy, overlay}` — fail-closed | closed tier set |
| `SPINE_OFF_LIMITS` | a verb targets `target_tier == spine` — hard refusal, even with a witness pass | inv. 5 (the agent never edits its own spine) |
| `META_RULE_OFF_LIMITS` | `meta_rule == true` — the edit-governing rules are out of the agent's reach | inv. 5 (the meta-rules are off-limits) |
| `FULL_REWRITE_FORBIDDEN` | `delta_kind` not in `{append, version_swap, mask}` — a full rewrite | inv. 2 (ACE: never a full rewrite) |
| `MID_PREFIX_EDIT` | `effect_timing == mid_prefix` — would mutate the resident array mid-session | inv. 2 (next rebuild, never mid-prefix) |
| `WITNESS_REQUIRED` | an activating agent edit with witness absent / `verdict == absent` | inv. 5 (Voyager acceptance gate) |
| `SELF_GRADED` | an agent edit with `grader == self` — the agent graded its own edit | inv. 5 (never grades its own edit) |
| `WITNESS_FAILED` | an activating agent edit with `verdict == fail` | inv. 5 (the witness refuted it) |
| `AUTO_DEMOTED` | a resident agent rule with `outcome == regressed` — demoted, journal row written | guard-RSI worst-bucket routing |

A token outside this set is **not** silently coerced to `admit`. A malformed edit fails at
the authoring boundary (the closed enums reject it) or returns a fail-closed refusal at
check time — never a quiet pass.

### The decision table — domain-free, deterministic, fail-closed

The whole check is a pure function of the edit shape. Read top to bottom; the first
matching arm wins; every non-`admit` arm refuses or demotes, never a silent serve:

```
1.  verb          ∉ {add, promote, demote, version}   -> Refuse(UNKNOWN_VERB)
2.  target_tier   ∉ {spine, policy, overlay}          -> Refuse(UNKNOWN_TIER)
3.  target_tier   == spine                            -> Refuse(SPINE_OFF_LIMITS)        (hard: no verb edits the spine)
4.  meta_rule     == true                             -> Refuse(META_RULE_OFF_LIMITS)
5.  delta_kind    ∉ {append, version_swap, mask}      -> Refuse(FULL_REWRITE_FORBIDDEN)  (never a full rewrite)
6.  effect_timing == mid_prefix                       -> Refuse(MID_PREFIX_EDIT)         (next rebuild only)
7.  verb == demote:
      a. outcome == regressed                         -> Demote(AUTO_DEMOTED)            (guard-RSI worst-bucket; journal row)
      b. otherwise                                    -> Admit                           (removal is always safe; effective next rebuild)
8.  author == agent  (verb ∈ {add, promote, version} — an ACTIVATING edit):
      a. witness absent OR verdict == absent          -> Refuse(WITNESS_REQUIRED)
      b. witness.grader == self                       -> Refuse(SELF_GRADED)             (inv. 5: never grades its own edit)
      c. witness.verdict == fail                      -> Refuse(WITNESS_FAILED)
9.  otherwise                                         -> Admit (effective next_rebuild, prefix re-pinned, byte-stable until then)
```

It is **append-mostly and fail-closed**: an agent edit faces strictly more checks than an
operator edit (arm 8 is agent-only), and the *absence* of an affirmative witness pass is a
refusal, never a silent admit. It is **monotone in authority**: tightening the author from
`operator` to `agent` only ever *adds* restriction; nothing an agent may do is something an
operator may not. The one non-refusing edit that needs no witness — `demote` (arm 7) — only
ever *removes* reach, so it cannot widen the agent's authority.

## The three contracts (the acceptance, made checkable)

This check is portable because it holds three properties an external runtime can verify
without fak's kernel:

1. **It round-trips with no engine.** author (write a `PromptEdit`) → check (apply the
   table → an `EditDecision`) → review (read the `EditDecision` as data) is a pure data
   transform. No model runs; no network is touched. The
   [round-trip below](#the-round-trip-as-data) is the witness, on disk as fixtures — an
   agent `add` with a `dos_verify:pass` witness *admits*; the same agent, targeting the
   spine, is *refused* `SPINE_OFF_LIMITS`; graded by `self` it is *refused* `SELF_GRADED`;
   a resident rule with `outcome == regressed` is *demoted* `AUTO_DEMOTED`.
2. **The vocabularies are closed and validatable.** The verb set, the tier set, the
   delta-kind set, the witness sets, and the refuse-reason set are finite enums; a
   validator decides membership with a finite switch, not a lookup against a live service.
   The schema is published as a machine-checkable JSON Schema —
   [`system-prompt-mutation-schema.json`](system-prompt-mutation-schema.json) (Draft
   2020-12) — so any runtime authors and validates an edit with an off-the-shelf validator,
   **no fak engine present**: the [positive fixtures below](#the-round-trip-as-data)
   validate against it, and five on-disk
   [negative fixtures](fixtures/system-prompt-mutation-invalid/) — an out-of-set verb (a
   full rewrite as a verb), an out-of-set tier, an out-of-set delta (`full_rewrite`), an
   activating agent edit with *no witness*, and an unknown field — are each rejected at the
   boundary. The
   [validation recipe](fixtures/system-prompt-mutation-invalid/README.md) runs the whole
   round-trip (eight positives accepted, five negatives rejected) with a stock Draft 2020-12
   validator, so the "validatable" claim is checkable, not asserted.
3. **It is fail-closed and evidence-bound.** Absence of an affirmative admit is a refusal:
   an unknown verb/tier/delta, a spine or meta-rule target, a mid-prefix edit, or an
   activating agent edit with no independent witness pass each refuses, never passes. And
   the `witness` verdict is the *independent* witness's — `dos verify` or the guard journal,
   not the agent's self-report. A runtime that lets the agent author its own `pass`
   (`grader == self`) has voided the contract; the check refuses exactly that as
   `SELF_GRADED`.

## The round-trip, as data

The schema's whole claim is that author → check → review is data, not narration. The
fixtures under [`fixtures/`](fixtures/) are the on-disk witness — one `PromptEdit` per
disposition, each paired with the `EditDecision` a check with no model and no fak engine
produces:

- [`prompt-edit-admit.json`](fixtures/prompt-edit-admit.json) →
  [`prompt-decision-admit.json`](fixtures/prompt-decision-admit.json) — the **admit** case:
  an agent `add` of an overlay card with an independent `dos_verify:pass` witness, effective
  at the next rebuild. The proposal path the agent *is* allowed.
- [`prompt-edit-spine-refuse.json`](fixtures/prompt-edit-spine-refuse.json) →
  [`prompt-decision-spine-refuse.json`](fixtures/prompt-decision-spine-refuse.json) — the
  **hard refusal**: a `version` swap that targets `tier == spine`. Well-formed, *with* a
  witness pass — and still refused `SPINE_OFF_LIMITS`. The spine cannot be targeted by any
  verb (acceptance criterion 1).
- [`prompt-edit-self-graded.json`](fixtures/prompt-edit-self-graded.json) →
  [`prompt-decision-self-graded.json`](fixtures/prompt-decision-self-graded.json) — the
  **invariant-5 refusal**: an agent `version` of a policy block whose witness is
  `grader: self`. Refused `SELF_GRADED` — the agent never grades its own edit (acceptance
  criterion 4).
- [`prompt-edit-auto-demote.json`](fixtures/prompt-edit-auto-demote.json) →
  [`prompt-decision-auto-demote.json`](fixtures/prompt-decision-auto-demote.json) — the
  **auto-demote** case: a resident learned rule whose later `outcome == regressed`. Demoted
  `AUTO_DEMOTED` with `journal: true` — a rule that correlates with worse witnessed outcomes
  is demoted and leaves a journal row (acceptance criterion 3).

A reviewer reads the eight files and the table binding them; no model and no fak engine are
needed to confirm the check.

## Reference implementation and witness

The contract rests on a witness + cache-stability substrate that is **shipped and
offline-witnessed**, and a *runtime* that walks it which is **not yet** — Rungs 1–5 of the
system-prompt MMU are all open, this issue (`#1263`) being Rung 5. The table is honest about
which is which:

| Schema element | Reference stick | Status |
|---|---|---|
| The spine is byte-identical every turn (inv. 1) | [`cachemeta.SegStable`](https://github.com/anthony-chaudhary/fak/tree/main/internal/cachemeta) ("system prompt — byte-identical every turn") + [`promptmmu`](https://github.com/anthony-chaudhary/fak/tree/main/internal/promptmmu) byte-identity prefix floor (fail-safe identity on any drift) | [SHIPPED] |
| An edit takes effect at the next prefix rebuild (inv. 2) | [`internal/sessionreset`](https://github.com/anthony-chaudhary/fak/tree/main/internal/sessionreset) `Contributor` registry (folds a drained transcript into a fresh seed; re-pins the spine) | [SHIPPED] |
| The independent witness (Voyager acceptance gate) | `dos verify` (the truth syscall) + the hash-chained guard journal ([`internal/abi`](https://github.com/anthony-chaudhary/fak/blob/main/internal/abi/events.go) `EvDecide`/`EvDeny`) | [SHIPPED] |
| Auto-demote of a worse-correlating rule | [`internal/guardrsi`](https://github.com/anthony-chaudhary/fak/blob/main/internal/guardrsi/guardrsi.go) `WorstBucket` (worst-bucket routing) | [SHIPPED] |
| The segment vocabulary for a fak-owned base context | [`cachemeta.PromptSegment`](https://github.com/anthony-chaudhary/fak/tree/main/internal/cachemeta) `{Kind, Tokens, Content, Witness}` | [SHIPPED] |
| `internal/syspromptmmu` base-context spine + segment model | Rung 1 (`#1259`, keystone) | **not yet** |
| Segment-plan → `promptmmu` splice adapter (proves inv. 1 + 2 e2e) | Rung 2 (`#1260`) | **not yet** |
| Residency policy (safety-critical always resident; spine never paged) | Rung 4 (`#1262`) | **not yet** |
| The portable `fak prompt {add, promote, demote, version}` verbs + the versioned-delta store + the auto-demote pass | Rung 5 (`#1263`, this issue) | **not yet** |

## Honest fences

- **This page is the CONTRACT, not the runtime.** The `fak prompt` verbs and the
  `internal/syspromptmmu` package do not exist yet — Rung 5 (`#1263`) builds them, and it
  depends on Rungs 1, 2, and 4, which are also open. This page pins the admission decision
  Rung 5 must satisfy *before* the code lands, so the acceptance is a fixed target, not a
  post-hoc rationalization. It is the same discipline the sibling
  [context-contract](context-contract-schema.md) and [taint-check](taint-check-schema.md)
  schemas keep: the contract ships first, the verb is a named follow-on.
- **The witness must be the kernel's, not the agent's self-report.** The check assumes the
  `witness.verdict` is `dos verify`'s or the guard journal's reading, not the agent's
  assertion. The schema can carry only a `{grader, verdict}` in the right shape; it cannot
  itself prove the verdict was independently produced — that proof is the calling runtime's
  obligation. The contract refuses the *expressible* self-grade (`grader == self`); it
  cannot detect a runtime that forges a `dos_verify:pass`. (Same fence as
  [taint-check's kernel-authored taint](taint-check-schema.md#honest-fences).)
- **The contract is data; declaring an edit introduces no spontaneous refusal.** A
  `PromptEdit` is a binding, not a runtime gate — authoring one fires no check until the
  `fak prompt` verb walks it at an opt-in surface. The mechanism stays in the verb; only
  policy crosses into the tree.
- **The spine fence is structural, not advisory.** `SPINE_OFF_LIMITS` is a hard refusal of
  *any* verb on `tier == spine`, with no witness escape — there is deliberately no
  `authorized`-style override on the spine, because the spine is the one region whose
  byte-identity invariant (1) the whole cache economy depends on. An operator who must
  change the spine does so out-of-band (a new build), not through `fak prompt`.
- **`base_version` enables rollback but does not itself store the bytes.** The schema names
  the rollback chain anchor; the *bit-for-bit* restoration is the versioned-delta store's
  obligation (Rung 5's preserve-prior-versions property), witnessed by the splice adapter's
  `bytes.Equal(prefix)` floor at the next rebuild — not by this schema, which only declares
  the version a delta supersedes.

## Cross-references

- [`system-prompt-mutation-schema.json`](system-prompt-mutation-schema.json) — the machine-checkable JSON Schema (Draft 2020-12): author and validate an edit with any off-the-shelf validator, no fak engine present.
- [`fixtures/system-prompt-mutation-invalid/README.md`](fixtures/system-prompt-mutation-invalid/README.md) — the engine-free validation recipe (eight positives accepted, five negatives rejected) that makes contracts (2) and (3) checkable.
- [The system-prompt MMU design note](../notes/SYSTEM-PROMPT-MMU-2026-06-29.md) — the epic (`#1258`) this is Rung 5 of: the fak-first base-context layout, the five load-bearing invariants, and the six rungs.
- [The portable context-contract schema](context-contract-schema.md) — the materialized-view sibling (certifies a *view* is a witnessed fold); same recipe, a different decision.
- [The portable taint-check schema](taint-check-schema.md) — the IFC sibling (gates a *value* into a sink); same recipe, a different decision.
- [`internal/guardrsi`](https://github.com/anthony-chaudhary/fak/blob/main/internal/guardrsi/guardrsi.go) · [`internal/abi`](https://github.com/anthony-chaudhary/fak/blob/main/internal/abi/events.go) — the shipped guard-RSI worst-bucket routing and the hash-chained decision journal the auto-demote + witness arms rest on.
- [`internal/promptmmu`](https://github.com/anthony-chaudhary/fak/tree/main/internal/promptmmu) · [`internal/sessionreset`](https://github.com/anthony-chaudhary/fak/tree/main/internal/sessionreset) · [`internal/cachemeta`](https://github.com/anthony-chaudhary/fak/tree/main/internal/cachemeta) — the shipped cache-stable splice floor, the rebuild re-pin, and the segment vocabulary.
