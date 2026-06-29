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
