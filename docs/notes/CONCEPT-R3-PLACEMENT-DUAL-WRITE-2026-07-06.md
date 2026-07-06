---
title: "R3 placement dual-write: one structure, two projections (residency ⇄ breakpoints)"
description: "The derivation contract for autoctx rung R3 (#2201): cache_control breakpoint placement and SegStable residency segments derive from ONE structure, so prefix stability is a contract in code, not discipline. Specifies the single derived structure, the advisory shadow-mode check that logs a would-be stable-segment edit without changing anything, the two derivation test targets (#555 splice, promptmmu prune), and the fak_harness_coherence witness. Ships no code; grounds the code dispatch."
date: 2026-07-06
---

# R3 placement dual-write: one structure, two projections

Status: derivation contract for rung **R3** of the zero-knob automatic-context
epic [#2198](https://github.com/anthony-chaudhary/fak/issues/2198) — issue
[#2201](https://github.com/anthony-chaudhary/fak/issues/2201). Spine:
[`CONCEPT-AUTOMATIC-CONTEXT-2026-07-01.md`](CONCEPT-AUTOMATIC-CONTEXT-2026-07-01.md)
(§R3, §L4). This note **ships no code** — like the parent spine it binds
already-shipped surfaces into one contract and specifies the smallest code
increment plus the witness that would close #2201. It is the design half; the
`internal/promptmmu` / `internal/cachemeta` code dispatch is the other half
(named under *What remains* below).

## Horizon classification (the intake this note repairs)

R3/#2201 is **`gen/now` + `managed-context` + `prompt-caching`**, milestone
`Generation G0 - Now / Immediate` — not `needs-triage`.

- **Promotion evidence:** the spine's "Generation classification of the rungs"
  section reclassified the whole R1–R8 program to `gen/now` on *applied*
  evidence — the keystone R1 (#2199) closed carrying `managed-context`,
  `generation`, `gen/now` on the `Generation G0` milestone, and the newest
  sibling #2947 was filed `gen/now`. Under [`docs/generation.md`](../generation.md)
  a rung whose labels disagree with its applied peers is intake drift to fix
  before dispatch. R3 improves the *current* default guard loop (a stable-prefix
  contract) with a code witness, not a future architecture bet — the `gen/now`
  test.
- **Demotion / retirement evidence:** if a reviewer shows R3's derivation
  depends on relay mechanics (#1860) that are themselves `gen/next`, or the
  shadow-mode advisory never reaches a live guard session (its witness is
  structurally unreachable in-fleet), R3 demotes to `gen/next` and this note's
  classification moves with that evidence. The advisory-only, change-nothing
  first step is precisely what keeps the demotion cheap: no wire behavior rides
  on it yet.
- **Invalidating assumption (load-bearing):** this reads the *applied* labels on
  R1/#2947 as the program's true horizon. If those labels were themselves a
  bulk-apply artifact rather than a reviewed decision, the whole program horizon
  is unproven and every rung inherits `needs-triage` again.

## The gap (grounded, at HEAD)

Prefix stability is discipline, not contract. Three surfaces already speak the
same nouns but derive their spans independently:

- `cachemeta.PromptSegment{Kind, Tokens, Content, Witness}` with the
  `SegmentKind` set `SegStable | SegToolSchema | SegVolatile | SegToolResult |
  SegMessage | SegSealed` (`internal/cachemeta/prefix_stability.go`). `Diverge`
  computes the cacheable prefix and stops it at the first content mismatch **or**
  a `SegSealed` span.
- `promptmmu.PlanBreakpoints(turn, score) → BreakpointPlan` (#1603,
  `internal/promptmmu/breakpoint.go`) already derives the residency plan:
  `ProtectedPrefix` / `MutableTail` / `UnsafeToCompact` spans over the turn's
  segments. This IS the residency structure R3 needs — it is just not yet
  consumed by breakpoint placement.
- `syspromptmmu.BaseContext() → []Segment{Tier, PromptSegment}` authors the base
  context as `SegStable` segments with `NonEvictable(Tier)` pins
  (`internal/syspromptmmu/syspromptmmu.go`) — but it is **dormant** (no live
  importer, per the spine's honest map).

Nothing derives `cache_control` positions and the stable residency segments from
*one* structure, so a residency edit (a splice/prune boundary) can land inside a
`SegStable` span and silently bust warmth — law **L4** violated at the prefix
boundary, invisibly.

## The contract: one derived structure

R3 is satisfied when a single value is the source of truth both sides read:

1. **The residency plan is the structure.** `BreakpointPlan.ProtectedPrefix`
   already names the stable leading run in segment coordinates. R3 adds the dual
   projection: the *only* legal `cache_control` breakpoint position is the
   `ProtectedPrefix.End` boundary (end-exclusive segment index), i.e. a
   breakpoint sits between the last protected segment and the first mutable one —
   never inside a `SegStable` / non-evictable (`syspromptmmu.NonEvictable`) span.
   Placement is a *read* of the plan, not a second derivation.
2. **A stable-segment set falls out of the same plan.** The protected span plus
   any `SegStable`/`SegToolSchema` segment and any `syspromptmmu` spine/policy
   (`TierSpine`/`TierPolicy`) segment form the *stable set*: the byte ranges a
   transform must carry verbatim. `SegSealed` is already handled (refusal rule 3,
   `UnsafeToCompact`); R3 extends the same refusal shape to a stable-segment
   edit.

The one-structure test: `PlanBreakpoints` output, and nothing else, decides both
"where may a breakpoint sit" and "which segments are immutable this turn."

## The advisory shadow-mode check (change nothing, log the would-be violation)

The first increment is **advisory** (L7: advisory first, gate later). Define a
pure check with a closed reason vocabulary, in the same style as
`promptmmu.UnsafeSpan`:

```
CheckStableEdit(plan BreakpointPlan, edit Span) → (ok bool, reason string)
```

- `edit` is the half-open segment range a transform proposes to rewrite/drop.
- Reason `stable-segment-edit` when `edit` intersects the stable set strictly
  inside `ProtectedPrefix` (the L4 hazard). Reason `sealed-segment-edit` reuses
  the existing `UnsafeSealed` semantics. `ok == true` (empty reason) when the
  edit lies wholly within `MutableTail` and touches no stable/sealed segment.
- **Shadow mode:** the caller LOGS a non-`ok` result and proceeds unchanged —
  attributed, one structured record per would-be violation, zero wire change.
  The gate (refuse, or re-price the breakpoint) is a later rung once the shadow
  log reads clean.

### The two derivation test targets (the #2201 done condition's unit half)

The derivation is unit-tested against the two transforms that actually edit the
prefix today, asserting each is correctly classified:

- **The #555 splice** — `promptmmu` splices `tools[]` past the last
  `cache_control` breakpoint (`internal/architest` leaf note: "splices tools[]
  past the last cache_control breakpoint"; the Anthropic twin anchors the
  protected prefix on the FIRST breakpoint, `internal/agent/anthropic_compact.go`).
  A splice bounded to `MutableTail` must classify `ok`; a splice whose boundary
  crosses into `ProtectedPrefix` must classify `stable-segment-edit`. This is the
  by-construction proof that the splice preserves the cached prefix.
- **The promptmmu prune** — `CompactInboundTools`/`CompactInboundSystem`, the
  inbound twin of #555 (`cmd/fak/guard.go`: "prune tool DEFINITIONS the floor can
  never admit"). Pruning a floor-denied tool schema outside the protected prefix
  must classify `ok`; pruning one inside a protected `SegToolSchema`/`SegStable`
  run must classify `stable-segment-edit`.

## The witness

R3's live witness is already scraped: the
`fak_harness_coherence_events_total{event=...}` family
(`internal/gateway/harness_coherence.go`), folding
`compactcohere.PrefixEvent` (`internal/compactcohere/compactcohere.go`). The
fak-attributed break events are `fak_cut` (`EventFakCut`) and `fak_world_break`
(`EventFakWorldBreak`). R3 is witnessed when, on a real guard session:

```
fak_harness_coherence_events_total{event="fak_cut"}        == 0
fak_harness_coherence_events_total{event="fak_world_break"} == 0
```

while `event="harness_rewrite"` (and the prune/shed counters) keep firing on the
same session — fak-attributed prefix breaks are zero while shed/prune stay
active. A non-zero fak-attributed count is a real L4 violation the shadow log
must have attributed line-by-line.

## What remains (the code + live-session gate — the honest fence)

This note is the design contract only. To *close* #2201 a code dispatch on the
`promptmmu`/`cachemeta` lane must:

1. Add `CheckStableEdit` (pure, closed reason set) beside `PlanBreakpoints`, and
   the shadow-mode log attribution at the two call sites (#555 splice, promptmmu
   prune).
2. Land the unit test against those two transforms (fails before the check
   exists, passes after) — the achievable half of the done condition.
3. Run the shadow mode on a **real guard session** and show the two
   fak-attributed counters at 0 while `harness_rewrite` fires — the half that
   needs a live session on real hardware (a host capability not present on the
   native-Windows dev box, where even `go test` routes through WSL).

Until (3) has a captured witness, #2201 is `not yet`: the derivation and its unit
test are landable now; the live shadow-mode/metric witness is the gate that
remains.

## Next checkable step

Dispatch the `promptmmu`-lane code increment above; check
`go test ./internal/promptmmu -run StableEdit` for the unit half, then a guarded
session's `/metrics` scrape for the two `fak_harness_coherence` counters.
