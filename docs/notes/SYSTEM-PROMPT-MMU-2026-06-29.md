# The system-prompt MMU — fak owns its own base context

*Dated design note, 2026-06-29. Planning + ticket scope only; no implementation
lands with this doc. Sibling of the inbound prompt-MMU
([INBOUND-PROMPT-MMU-2026-06-25](INBOUND-PROMPT-MMU-2026-06-25.md), epic #751),
the queried skill loader ([SKILL-LOADER-QUERY-EPIC](../SKILL-LOADER-QUERY-EPIC.md),
epic #1103), and the context-safety observability spine (epic #1217).*

## Thesis

Today fak runs *inside* a harness whose system prompt is frozen at session
start. The harness authors the preamble — its identity, its tool-use rules, its
environment block, its skill menu — and fak is one capability mentioned inside
it. The base context is **harness-first, fak-incidental**.

Invert it. The base context should be **fak-first**: the irreducible, always-
resident head is fak's own concepts (what the gate is, what the journal proves,
what a capability is), and *everything the harness needs* — its rules, its skill
menu, its environment, its per-turn reminders — becomes a **queryable, paged,
dynamically-modifiable overlay** that fak assembles below an immutable spine.

Two end-states define "done":

- **(a) The minimal base.** In the best case the irreducible resident base
  context is *only* fak — fak concepts plus a retrieval affordance (pointers, an
  index, a query verb), not the bodies. Harness items are faulted in on demand
  and evicted under pressure, the way the result-side `ctxmmu` already pages a
  tool result.
- **(b) The live base.** The system prompt itself becomes a first-class object
  fak can rewrite at runtime — add a rule, promote a skill from paged to
  resident, swap a v1 instruction block for v2 — **without rebuilding the whole
  prompt and without busting the provider's prefix cache**, under a witness gate
  so a self-authored edit cannot silently corrupt the spine.

This is not "another prompt builder." It is the application of fak's own MMU
doctrine — page, pin, evict, witness — to the one region of the context window
fak does not yet own: the head.

## Why this is the dual of #751, not a duplicate

There are three adjacent epics. They are layers, not overlaps:

| Epic | Owns | Verb |
|---|---|---|
| **#751 promptmmu** | the inbound prompt as it *passes through* `fak guard -- claude` | **prunes** (drops denied tool defs, cache-prefix-safe) |
| **#1103 skill loader** | capabilities (skill / MCP tool / A2A agent) as paged objects | **faults in** by query, not menu |
| **#1217 context-safety** | the *observability* of what got paged/cached/evicted | **witnesses** (divergence = alarm) |
| **this epic (system-prompt MMU)** | the *authorship and layering* of the base context itself | **assembles** (fak spine + queried harness overlay) |

`promptmmu` takes a harness-authored prompt and removes what can't be used.
This epic takes ownership of the *composition* of the head: fak authors the
immutable spine, and the harness's contributions (rules, environment, skill
menu) are assembled below it from the same paged-capability substrate the skill
loader already provides. #751 is subtractive on someone else's prompt; this is
the constructive layer that decides what the prompt *is*.

The clean seam: a system-prompt MMU **emits** a layered `[]cachemeta.PromptSegment`
(spine → policy → queried overlay → tail), and `promptmmu` remains the
cache-safe **splicer** that realizes a segment plan into wire bytes without
moving the prefix boundary. This epic produces the plan; #751's machinery
executes it. They compose; neither subsumes the other.

## The hard constraint (this is what makes it engineering, not a wish)

The system prompt is the **head** of the context window. Two independent bodies
of evidence say the same thing about the head, and together they *are* the
design:

1. **The cache prefix is byte-immutable or it is cold.** A KV / provider prefix
   cache reuses a contiguous run from token 0 up to the first byte that differs;
   change anything ahead of the stable part and everything after it re-prefills.
   Manus reports a measured ~10× gap between cached and uncached input — so
   "dynamically modifiable head" and "warm prefix cache" are in direct tension,
   and the *only* resolution the field endorses is **append-and-mask, never
   edit-in-place** (Anthropic context-engineering; Manus, *Lessons from Building
   Manus*). fak already models this exactly: `cachemeta.SegStable` is literally
   commented *"system prompt, static instructions — should be byte-identical
   every turn,"* and `promptmmu`'s floor is `bytes.Equal(raw[:prefixEnd],
   out[:prefixEnd])` or fail-safe identity.

2. **The head is where attention concentrates.** Attention-sink work
   (StreamingLLM) shows the first few tokens absorb persistent attention mass
   regardless of content; *Lost in the Middle* shows accuracy is U-shaped over
   position (best at head and tail, weakest in the middle); heavy-hitter work
   (H2O/SnapKV) shows the tail query predicts which earlier positions matter.

These resolve into one physical layout — and, crucially, **fak-first ordering
is also the cache-stable, attention-optimal ordering**:

```
[ fak spine ]      immutable, always resident — the attention sink + heavy-hitter
                   anchor, never compressed, never evicted, byte-identical every turn
[ policy floor ]   the deny/allow rules + safety-critical instructions; resident,
                   versioned, changes only at a marked cache breakpoint
─ cache breakpoint ─────────────────────────────────────────────────────────────
[ harness overlay ] queried, paged capability cards (skills/MCP/A2A descriptors,
                   the harness's situational rules) — APPENDED after the breakpoint,
                   masked not mutated, faulted in by the turn's query
[ … messages … ]
[ tail: live query + the few items it faulted in ]   the high-recall zone
```

The fak-first inversion the goal asks for is not a stylistic preference — it is
forced by the cache and the attention geometry. The thing that must never move
(the spine) goes at the head; the thing that changes per turn (the overlay) goes
after a breakpoint where appending it is free; modification happens by
**page-in + mask**, never by rewriting the spine.

## Load-bearing invariants (every rung holds all five)

Stated as the promptmmu epic states its five, because the discipline is the
same and a rung that breaks one is not shippable.

1. **The spine is byte-identical every turn — proven, not hoped.** The fak spine
   + policy floor re-serialize to the same bytes through the last
   `cache_control`-bearing element; realized through `promptmmu`'s existing
   `bytes.Equal(prefix)` guard. Any drift ⇒ fail-safe to the prior prompt.
2. **Append-and-mask, never edit-in-place.** A new or changed overlay item is
   paged in *after* the breakpoint and gated by masking; the resident array is
   never mutated mid-session. Modification of a *resident* block is a
   versioned swap that takes effect at the next prefix rebuild (session
   start / `--reset-on-budget`), never mid-prefix.
3. **Safety-critical instructions are always resident, never paged.** The
   dominant failure mode of self-paging on weaker models is *silent
   under-retrieval* (MemGPT degrades on GPT-3.5; the agent thinks it has the
   full picture and does not). Anything load-bearing for the deny floor or
   identity stays in the spine/policy tier and is excluded from the evictable
   set by construction.
4. **The base is pointers, not bodies — and the index itself is bounded.**
   Progressive disclosure's metadata-first/body-on-demand is the shape
   (Anthropic Agent Skills: ~100 tokens/skill resident, body read on trigger,
   the listing truncated at 1,536 chars). The resident descriptor set is *not*
   free; page the index too once it grows. A capability is `capindex.CapCard`
   at rest, faulted to `capindex.Capability` only when the query needs it.
5. **Self-modification is witness-gated, versioned, append-mostly — the agent
   never rewrites its own spine and never grades its own edit.** Voyager's
   verification gate + ACE's incremental deltas (never full rewrites; full
   rewrites cause *context collapse* and *brevity bias* that erodes safety
   text) + the *"Your Agent May Misevolve"* warning. fak's witness / `dos
   verify` discipline and guard-RSI worst-bucket routing are exactly the
   acceptance gate and the demotion path this requires. The base context and
   the meta-rules that govern its editing are *off-limits* to the agent's own
   edits.

## fak already ships the substrate (this is mostly *wiring*, not greenfield)

The substrate map is unambiguous on two points: **(1)** no fak-authored system
prompt exists today — the authorship spine is the one genuinely new surface;
**(2)** the paging / residency / query / cache-coherence machinery is *built*,
and the skill-loader keystone is **request-path-dead** (no live serve caller).
The biggest lever is to wire what exists into a real base-context assembler.

| fak package | already does | role in this epic |
|---|---|---|
| `internal/cachemeta` (tier 1) | `PromptSegment{Kind,Tokens,Content,Witness}`, `SegmentKind` incl. `SegStable`="system prompt"; §A3 stability + §A4 coherence (`InjectBreakMarker`, `ShapeGLMTurn…`) | the **vocabulary** for a fak-owned base context + the breakpoint placement |
| `internal/promptmmu` (tier 1, #751) | `CompactInboundTools`, `bytes.Equal(prefix)` floor, splices bytes (never re-marshals), honors a `system`-only breakpoint | the **cache-safe splicer** that realizes a segment plan |
| `internal/ctxplan` (tier 1) | lossless `Store` + budgeted `Optimize`/`Plan` (`Faithful` witness), `Layout{base,current,recent,deep}`, `DemandPage` | the **budgeted planner** — base-context layout is its `base` area |
| `internal/capindex` (tier 2) + `capindexgw` (tier 4) | `CapCard`/`Capability`/`Resolver`/`Catalog.Query(intent)`, SHA-256 `Diff` CRUD | the **0→∞ overlay index**: harness items as queryable cards |
| `internal/contextq` (tier 3) | `QueryCapabilities` (MCP-Zero active discovery), `CapabilityLedger`, `SkillContextRecord` HIT-on-reinvocation | the **query primitive** for the overlay |
| `internal/ctxmmu` (tier 2) | page-in/out, CAS-pinned eviction, witness-clear gate, `PageOutBody` | the **pager** for an evictable overlay body |
| `internal/ctxresidency` (tier 3) | resident/evictable/held + `MeasureBlastRadius` + `EvictColdest` | the **residency + eviction-cost** read |
| `internal/architest` | layered-DAG tier gate | the **layering contract** new packages obey |
| `internal/sessionreset` | `Contributor` registry folds a drained transcript into a fresh seed | where a **base-context rebuild** re-pins the spine |

Two debts to reconcile on the way (both already named): the duplicate
`CapRef`/`Capability` in `contextq` vs `capindex` (part of **#1144**), and the
fact that the skill-loader resolvers have *no live caller* — this epic is the
caller that gives them a turn-path home.

## Rungs (child issues — planning only)

Numbered so the keystone lands first and each rung holds all five invariants.

- **Rung 1 — `internal/syspromptmmu` base-context spine + segment model.**
  A fak-owned, pure package that emits an ordered `[]cachemeta.PromptSegment`
  for the head: `SegStable` fak spine + a versioned policy floor, each segment
  carrying its `Witness`. No wire mutation here — it produces a *plan*. The one
  genuinely new authorship surface. Holds invariants 1–4. (tier 2; imports only
  `cachemeta`/`abi`.)

- **Rung 2 — segment-plan → `promptmmu` splice adapter.** Realize a Rung-1
  segment plan into wire bytes via `promptmmu`, anchoring the breakpoint between
  the policy floor and the overlay, asserting `bytes.Equal(prefix)` and
  fail-safe identity. Proves invariants 1 and 2 end-to-end on the
  `fak guard -- claude` passthrough.

- **Rung 3 — queried harness overlay (wire the dead keystone).** Drive
  `contextq.QueryCapabilities` / `capindex.Catalog` from the turn's intent to
  select overlay cards under a token budget, append them *after* the breakpoint,
  mask rather than mutate. This is the first live caller of the skill-loader
  substrate. Holds invariant 4; folds in the #1144 `CapRef` reconciliation.

- **Rung 4 — residency policy: what stays in the spine vs pages to queryable.**
  The MemGPT/Letta decision rule (resident iff used-every-turn ∧ small ∧
  safety-load-bearing; else paged), enforced so safety-critical text is
  excluded from the evictable set. Holds invariant 3. Uses
  `ctxresidency.MeasureBlastRadius` to refuse an evict whose blast radius
  crosses the spine.

- **Rung 5 — witness-gated runtime modification (the live base).** `fak prompt`
  verbs to add/promote/demote/version a base-context block, each edit a
  versioned delta gated by `dos verify` / the guard witness, taking effect at
  the next prefix rebuild (Rung-2 splice + `sessionreset`). The agent may
  *propose* via the same path but the spine and the edit-governing meta-rules
  are off-limits; a learned rule that later correlates with worse witnessed
  outcomes is auto-demoted (guard-RSI shape). Holds invariant 5.

- **Rung 6 — observability surface (feeds #1217).** A `fak debug` / `fak prompt
  show` view: the live segment plan, what's resident vs paged, the prefix
  divergence series, and a re-derivation check that the realized wire prefix
  equals the planned spine bytes (divergence = alarm). Consumes the
  context-safety spine; does not blend its own numbers into it.

## What this epic is *not*

- Not a per-turn prompt *rewriter* — the spine is immutable; only the
  after-breakpoint overlay changes per turn, by append+mask.
- Not prompt *compression* of the base — the fak spine is never run through a
  lossy compressor (page, don't compress; compression is reserved for verbose
  overlay examples, never the canonical fak definition).
- Not an agent that edits its own core — runtime modification is a
  witness-gated, versioned, append-mostly store over the *overlay and policy*
  tiers; the spine and the meta-rules are out of the agent's reach.
- Not a replacement for #751/#1103/#1217 — it is the authorship layer that sits
  on their machinery and gives the skill-loader keystone its first live caller.

## Rung 5 — witness-gated runtime modification: the live base (pinned contract, #1263)

Rung 5 ([#1263](https://github.com/anthony-chaudhary/fak/issues/1263)) is the
*live-base* rung — the system prompt becomes a first-class object fak rewrites at
runtime, under a witness gate, **never edit-in-place**. This section does for
Rung 5 what an epic's per-child pinned contracts do for the rest of a ladder: fix
the contract's shape — the four `fak prompt` verbs, the versioned-delta record,
the witness gate, the spine-off-limits hard refusal, the auto-demote journal row
— so the eventual code build is **wiring against a fixed contract**, and fence
honestly what blocks the live verb today (Rungs 1/2/4 are unbuilt — there is no
`internal/syspromptmmu` package yet, so the verbs have nothing to edit). This is
a *docs-lane* increment: it pins the contract; it does **not** itself satisfy the
acceptance tests, which are a named code follow-on (§R5.5).

### R5.1 The substrate (the inputs, each with its current build state)

The honest accounting: the *cache-safe splice floor*, the *rebuild seam*, and the
*witness* are built and tested today — Rung 5 is **the versioned-store + verb
surface on top of them**, not greenfield in those parts — but the tier model the
verbs target and the residency rule that bounds promote/demote are not yet in-tree.

| Input | Source (current) | What it contributes to Rung 5 | Build state |
|---|---|---|---|
| the tiered spine + policy + overlay model | `internal/syspromptmmu` segment plan (Rung 1, [#1259](https://github.com/anthony-chaudhary/fak/issues/1259)) | the `SegStable` spine / versioned policy floor / overlay tiers a verb may (or may not) target | OPEN ([#1259](https://github.com/anthony-chaudhary/fak/issues/1259)) — **blocker** (no package yet) |
| the next-rebuild seam (when a swap takes effect) | Rung-2 splice adapter ([#1260](https://github.com/anthony-chaudhary/fak/issues/1260)) + `internal/promptmmu` `bytes.Equal(prefix)` floor + `internal/sessionreset` `Contributor` re-pin | where a versioned swap lands at the *next prefix rebuild*, never mid-prefix (invariants 1–2) | promptmmu + sessionreset **BUILT**; the Rung-2 adapter is OPEN ([#1260](https://github.com/anthony-chaudhary/fak/issues/1260)) — **blocker** |
| the witness gate (acceptance) | `dos verify` / the `fak guard` decision journal | the independent gate a self-authored delta must pass before it can become resident — the agent never grades its own edit (invariant 5) | **BUILT** (`dos verify` + the hash-chained guard journal are live) |
| the residency rule (what may move) | Rung-4 residency policy ([#1262](https://github.com/anthony-chaudhary/fak/issues/1262)) over `ctxresidency.MeasureBlastRadius` | which blocks are promotable/demotable vs spine-pinned; refuses an edit whose blast radius crosses the spine | OPEN ([#1262](https://github.com/anthony-chaudhary/fak/issues/1262)) — **blocker** for `promote`/`demote` |
| the auto-demote routing | guard-RSI worst-bucket routing (`cmd/fak/guardrsi.go`) | demote a learned rule that later correlates with worse witnessed outcomes, and leave a journal row | **BUILT** as a routing shape; needs the syspromptmmu version-log binding |
| the versioned-delta store + rollback | NEW in `internal/syspromptmmu` (this rung's own surface) | the append-mostly version log (ACE incremental deltas, never a full rewrite) that preserves prior versions for bit-for-bit rollback | the rung's one genuinely new surface |

### R5.2 The `fak prompt` verb surface (the four verbs)

Each verb edits a *base-context block* as a **versioned delta** — ACE-style
incremental, never a full rewrite (full rewrites cause context collapse + brevity
bias that erodes safety text). A verb may target the **overlay** or the
**versioned policy floor**; targeting the **spine** or the **edit-governing
meta-rules** is a hard refusal.

| Verb | Does | May target | Witness requirement |
|---|---|---|---|
| `fak prompt add <block>` | append a new overlay block as version `v1` (proposed) | overlay / policy floor | gated — proposed→resident only on an independent witness pass |
| `fak prompt promote <block>` | paged → resident (pin a block into the always-resident set) | overlay / policy floor (bounded by Rung-4 residency) | gated — a promotion that would cross into the spine tier is **refused** |
| `fak prompt demote <block>` | resident → paged (the reversible inverse of `promote`); the auto-demote path uses this on a worse-witnessed-outcome signal | overlay / policy floor | the auto variant leaves a journal row (§R5.4) |
| `fak prompt version <block>` | swap `v_n` for `v_{n+1}` as a versioned delta, preserving `v_n` for rollback | overlay / policy floor | gated — the new version is byte-stable until the next rebuild, then takes effect (§R5.5 AC2) |

Hard refusal (tested): any verb whose target resolves to the **spine tier** or the
**edit-governing meta-rules** returns a structured refusal (`SPINE_OFF_LIMITS`)
and exits non-zero — the agent may *propose* over the overlay/policy tiers, but
the spine and the meta-rules that govern editing are out of its reach by
construction, not by convention (invariant 5; "What this epic is *not*", bullet 3).

### R5.3 The versioned-block record shape (golden-testable)

A stable, deterministic per-block record — schema `fak.prompt-block.v1` — so a
frozen block round-trips byte-identically and *that* is the golden test; rollback
is then "re-pin `parent_version`'s bytes," provable bit-for-bit.

| Field | Meaning |
|---|---|
| `block_id` | the base-context block this record versions |
| `tier` | `policy` or `overlay` — never `spine` (a `spine` value is rejected at write) |
| `version` / `parent_version` | the monotone version + its predecessor (the rollback target) |
| `delta` | the ACE incremental change from `parent_version` (never a full-rewrite snapshot) |
| `witness` | the `dos verify` / guard verdict id that admitted this version to resident |
| `state` | `proposed` / `resident` / `paged` / `demoted` |
| `demoted_reason` | set when auto-demote fires — the worse-witnessed-outcome signal that routed it down |

Rollback restores `parent_version` bit-for-bit (the prior `delta` chain re-applied
to the same base = the same bytes); an auto-demote transitions `state`→`demoted`,
stamps `demoted_reason`, and appends a journal row — it never rewrites history.

### R5.4 The witness gate + auto-demote journal (invariant 5)

- **propose → witness → resident.** A self-authored delta is `proposed`; it becomes
  `resident` *only* after an **independent** witness (`dos verify` / the guard
  journal) accepts it (the Voyager acceptance-gate shape). The agent never grades
  its own edit — the success signal comes from the witness, not the author.
- **spine + meta-rules off-limits.** Enforced by the §R5.2 tier check in every
  verb (hard refusal), and by the §R5.3 write-time rejection of a `spine` tier
  value — two independent fences, so a bug in one does not open the spine.
- **auto-demote leaves a journal row.** A `resident` rule whose subsequent
  witnessed outcomes regress is demoted automatically (the guard-RSI worst-bucket
  routing shape), writing a `demoted` record with its `demoted_reason` — the
  demotion is itself a witnessed event, auditable after the fact.

### R5.5 Acceptance, and what blocks it today

The issue's four acceptance clauses, each tied to the witness that proves it:

- **AC1 — a self-proposed overlay rule is rejected unless it passes the witness;
  the spine cannot be targeted by any `fak prompt` verb (hard refusal, tested).**
  Proven by two tests: a proposed delta with a failing witness never reaches
  `resident` (§R5.4), and every verb against a spine-tier target returns
  `SPINE_OFF_LIMITS` and exits non-zero (§R5.2).
- **AC2 — a version swap is byte-stable until the next rebuild, then takes effect
  with the prefix re-pinned (invariant 1 holds across the swap).** `fak prompt
  version` stages `v_{n+1}` but the realized wire prefix stays byte-identical
  (the `internal/promptmmu` `bytes.Equal(prefix)` floor) until the Rung-2 splice
  re-pins at the next `sessionreset` rebuild — asserted by a before/after prefix
  equality test across the swap.
- **AC3 — rollback restores the prior version bit-for-bit; an auto-demoted rule
  leaves a journal row.** The §R5.3 `parent_version` re-pin is a byte-equality
  test; the §R5.4 auto-demote writes an auditable `demoted` row.
- **AC4 — the agent never grades its own edit (invariant 5).** The acceptance
  signal is the independent witness verdict, structurally separated from the
  proposing path (§R5.4).
- **Blocked-on (the honest fence).** AC1's "becomes resident" half, AC2 end-to-end,
  and the `promote`/`demote` residency bound are blocked on **Rung 1
  ([#1259](https://github.com/anthony-chaudhary/fak/issues/1259), OPEN)** — the
  tiered segment model the verbs target does not exist yet (`internal/syspromptmmu`
  is unbuilt) — on **Rung 2 ([#1260](https://github.com/anthony-chaudhary/fak/issues/1260),
  OPEN)** — the splice adapter that realizes a swap at the next rebuild — and on
  **Rung 4 ([#1262](https://github.com/anthony-chaudhary/fak/issues/1262), OPEN)** —
  the residency rule that says what may move. The witness (`dos verify` + the guard
  journal) and the rebuild seam (`promptmmu` + `sessionreset`) are **built**, so
  once Rungs 1/2/4 land Rung 5 is **wiring against this fixed contract**: stand up
  the `fak.prompt-block.v1` version log in `internal/syspromptmmu`, add the four
  `cmd/fak` `prompt` verbs over it, gate `proposed→resident` on the witness, and
  bind auto-demote to the guard-RSI worst-bucket signal.
- **Lane for the build:** the verbs are `cmd` (`cmd/fak` `prompt` + the version
  log in `internal/syspromptmmu`), the witness binding is `gateway`/`guard` — **not**
  `docs`. This docs increment pins the contract only; it does not itself satisfy
  AC1–AC4.

### R5.6 Reproduce (the contract today; the live verb once Rungs 1/2/4 land)

Today the contract is checkable as text: the four verbs (§R5.2), the
`fak.prompt-block.v1` record (§R5.3), and the two off-limits fences (§R5.4) are
fixed, and the witness + rebuild seam they bind to are already green
(`go test ./internal/promptmmu/... ./internal/sessionreset/...`). Once Rung 1's
segment model and Rung 2's splice adapter land, `fak prompt version <block>`
followed by a `--reset-on-budget` rebuild reproduces AC2 (prefix byte-stable, then
re-pinned), and `fak prompt add` with a deliberately failing witness reproduces
AC1 (never reaches `resident`).

## Sources

Context as a finite budget / just-in-time retrieval — Anthropic, *Effective
context engineering for AI agents* (Sep 2025). KV-cache economics + append-and-
mask — Manus, *Context Engineering: Lessons from Building Manus*. Context
degradation with length/distractors — Chroma, *Context Rot* (Jul 2025).
Progressive disclosure / metadata-first — Anthropic, *Agent Skills*. Tool-RAG
(menu degrades as it grows; retrieval is the new ceiling) — RAG-MCP
(arXiv 2505.03275), Toolshed (2410.14594), Gorilla (2305.15334). Virtual-context
paging + pressure eviction — MemGPT (2310.08560), Letta memory blocks.
Head-attention geometry — StreamingLLM (2309.17453), Lost in the Middle
(2307.03172), H2O (2306.14048) / SnapKV (2404.14469). Witness-gated runtime
self-modification — Voyager (2305.16291), Agentic Context Engineering /
ACE (2510.04618), *Your Agent May Misevolve* (2509.26354).
