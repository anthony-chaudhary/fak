---
title: "Generation classification: modver policy-manifest key space (#2462)"
description: "Grooming artifact classifying #2462 (modver: version policy manifests) as gen/now, with promotion, demotion/retirement, and invalidating-assumption evidence, the applied label intake, and the smallest-next implementation increment for a modver/cmd-lane worker."
---

# Generation Classification — modver policy-manifest key space (issue #2462)

Grooming artifact for issue #2462
(`modver: version policy manifests (examples/*.json, guard-default-policy)`). It
classifies the generation horizon from issue evidence so a later,
implementation-authorized worker starts warm. It is **not** a resolution: #2462's
witnessable acceptance (a policy-manifest key space in `internal/modver` plus an
advisory rev surfaced beside each policy name in `fak policy` output, landed with a
unit test + a live stamp showing policy rows) is not built, so #2462 stays
**open**. Its own closure binding requires "a ship commit whose diff lands the
named witness in the same commit"; this docs note is a `#2462` **mention**, not a
close.

Parent: #2458 (`version-everything`, milestone `version-everything`), a child of the
version-everything program.
Snapshot date: 2026-07-11.

## Why this is a classification note and not the feature

The dispatch routed #2462 to the **`docs` lane** under an **unclassified**
generation intent, with the triage frame's `allowed risk = triage only` and
`proof bar = classify the horizon from issue evidence before implementation`. That
state authorizes classification + a note (or label/milestone repair) only — not a
cross-lane code edit.

The feature's acceptance touches `internal/modver` (a new policy-manifest key
space folded through `Snapshot`/`DeltaRows`) and the `fak policy` verb surface (the
advisory rev display). Those paths are **outside** the docs lane's lease tree
(`dos arbitrate --lane docs` granted `docs/notes/GENERATION-MODVER-POLICY-KEYSPACE-2026-07-11.md`,
not `internal/**`/`cmd/**`), and their ship stamp is `(fak modver)` / a `cmd`-leaf
stamp, not `(fak docs)`. Implementing them here would collide with a code lane and
mis-stamp the commit. So this worker does the in-lane, gate-passable half (classify
+ record the map, repair the labels) and leaves the code to a modver/cmd-lane
worker, warmed by the increment map and the corrected assumptions below.

## Classification

| Field | Value |
|---|---|
| Issue | #2462 |
| Stream | `gen/now` |
| Generation milestone (contract) | `Generation G0 - Now / Immediate` |
| Also carries | `generation` |
| Parent epic | #2458 (`version-everything`), milestone `version-everything` |
| Intake status | **labels applied** — `generation` + `gen/now` added to #2462 on 2026-07-11. Milestone left as `version-everything` (see the milestone tension below). |

### Why `gen/now` (from issue evidence)

`gen/now` is the generation contract's "improves the current product, operator
loop, or trunk hygiene with a clear witness and no dependency on a future
architecture bet" stream, and its worked example is literally *"add a report field
that immediately helps the operator ... with a captured command output as
witness."* #2462 is exactly that shape:

- It **improves the current operator loop today**: it lets an operator cite
  "policy X at rev N" the way they already cite a binary build — an advisory rev
  surfaced beside each policy name in `fak policy` output.
- It has a **clear code/CLI witness**: the issue names "unit test + live stamp with
  policy rows," the same witness class the shipped `fak version modules` surface
  already carries.
- It has **no future-architecture dependency**: the version-everything **spine is
  already shipped and verified present in-tree** — `internal/modver/modver.go` +
  `internal/modver/trend.go` (the `Snapshot`/`DeltaRows`/`JoinScores` core), the
  `cmd/fak/version_modules.go` shell, and the `fak-module-versions/1` ledger at
  `docs/nightrun/module-versions.jsonl` (schema id referenced across
  `internal/modver` and `cmd/fak/version_*_test.go`). This child is **additive** to
  that shipped `/1` ledger; the issue's out-of-scope explicitly bars any
  non-additive `/1` schema change (a breaking row is a `/2` with its own contract).
- The rev display is **purely advisory** — it changes no policy enforcement
  semantics — so it is reversible and inert with respect to the capability floor.

### Why not the other streams

- **Not `gen/next`**: it needs no new schema (it is additive to `/1`), no
  default-exposure runtime gate, and no dogfood run before an operator can rely on
  the advisory rev. There is no default-off route to earn.
- **Not `gen/second-next`**: no cross-generation compatibility policy, adapter
  conformance, or simulation — it composes existing single-repo seams.
- **Not `gen/future`**: the next action is code against named seams
  (`internal/modver` + the `policy` verb), not research, a hardware witness, or a
  market/standards analogue.

## Promotion evidence (what confirms `gen/now` / closes #2462)

- A focused `internal/modver` unit test that folds a fixed policy-manifest set into
  derived revs and asserts the rows appear in the report/ledger — fails before the
  key space exists, passes after (the repro is the proof).
- A live `fak policy` stamp (captured command output) showing the advisory rev
  beside each policy name, plus a `fak-module-versions/1` ledger diff that appends
  policy rows **without** editing an existing `/1` row (additive-only witness).
- Operator-surface before/after: a policy name that previously rendered bare now
  renders with a citable rev, so an operator can pin "policy X at rev N."

## Demotion / retirement evidence (what pushes #2462 back or retires it)

- The additive-only constraint cannot be met — deriving a policy rev turns out to
  require a non-additive `/1` row change → the rev-display slice **demotes** (the
  breaking part becomes a `/2` child with its own contract, per the issue's own
  out-of-scope), and only the advisory read-side can stay `gen/now`.
- A sibling modver-* child (#2459–#2461, #2463–#2466) already lands the
  policy-manifest key space under a stronger witness → **retire** #2462's key-space
  half into that child and keep only the `fak policy` display, or close as
  duplicate.
- The `fak policy` surface is superseded (the verb is refactored out of
  `cmd/fak/main.go` into a dedicated handler by another owner) → re-target the
  advisory display into that new owner rather than the current `case "policy"`.

## Invalidating assumptions (found against the live tree, 2026-07-11)

The issue's "Likely files" and In-scope carry two assumptions that **do not hold
against the current tree** — a future worker must correct them before dispatching,
or the implementation will target the wrong seam:

1. **`cmd/fak/policy.go` does not exist.** The issue names it as the surface the
   advisory rev is added to, but there is no such file. The `fak policy` verb is
   dispatched from `cmd/fak/main.go:265` (`case "policy":`); the nearest
   policy-named source files are `cmd/fak/dispatch_model_policy.go` (a different
   concern — model-routing policy) and the manifest `cmd/fak/guard-default-policy.json`.
   → The display change lands against the `main.go` policy handler (a `cmd` leaf),
   stamped `(fak cmd)` / the real leaf, **not** the issue's assumed `cmd/fak/policy.go`
   and **not** `(fak modver)` for the CLI half.
2. **The policy-manifest "key space" is not the `examples/*.json` glob.** That glob
   both **over-includes** non-policy JSON that happens to live in `examples/`
   (`examples/observability/dashboard.json`, `examples/model-routing.example.json`,
   `examples/fanbench/sample-model-config.json`, `examples/mcp/.mcp.json`, the
   `examples/grammar-repair*/testdata/*.json` fixtures) and **under-includes**
   nested policy manifests (`examples/autogen-groupchat/policy.json`,
   `examples/crewai-crew/policy.json`, `examples/admit-and-log/research-batch-policy.json`,
   `examples/deny-in-60s/policy.json`, …). This is precisely the issue's own stated
   "Confusion risk: Missing a policy manifest location outside `examples/*.json` /
   guard-default-policy." → The key space needs a **schema-shape predicate** (a JSON
   whose structure matches a policy manifest — the loader in `internal/policy`
   already defines that shape), not a path glob, plus `cmd/fak/guard-default-policy.json`.

## Milestone tension (the one intake sub-decision left open)

The generation contract says the stream label and its **generation milestone**
should agree, and the `generation-now` issue view requires **both** the `gen/now`
label and the `Generation G0 - Now / Immediate` milestone. But a GitHub issue holds
exactly one milestone, and #2462 is already milestoned to the active
`version-everything` **program** roadmap. Rebinding to `Generation G0` would
satisfy the generation view but drop #2462 from the version-everything milestone's
child list (whether `fak milestone report` reads program membership from the
milestone or the `version-everything` **label** — which #2462 also carries — was
not confirmed here, so the effect is uncertain). Because that is a
program-tracking-vs-generation-view trade with an unverified reporting dependency,
this note **does not** rebind the milestone by guess. Recommended operator/epic-owner
decision: either (a) let program children keep the program milestone and have the
generation view key off the `gen/now` label alone, or (b) rebind to `Generation G0`
after confirming `fak milestone report` tracks the program by the `version-everything`
label rather than the milestone. Until then #2462 carries the `gen/now` label and
the `version-everything` milestone — a deliberate, recorded state, not silent drift.

## Smallest next increment (advisory, for a modver/cmd-lane worker)

1. Define the policy-manifest key space in `internal/modver` (or a thin
   `internal/policy` accessor it calls) as a **schema-shape predicate** over
   `examples/**` + `cmd/fak/guard-default-policy.json`, not an `examples/*.json`
   glob (assumption 2). Fold each manifest through the existing `Snapshot` so a
   policy gets a derived rev exactly like a module does — additive to
   `fak-module-versions/1`, no `/1` row edited.
2. Surface the derived rev **advisory-only** beside each policy name in the
   `fak policy` output, at the real handler (`cmd/fak/main.go` `case "policy"`,
   assumption 1) — no change to enforcement semantics.
3. Witness in the **same commit**: an `internal/modver` unit test folding a fixed
   policy set and asserting the rows land in the report/ledger, plus a captured
   `fak policy` run showing revs beside policy names. Ship with `Closes #2462` and
   the `(fak modver)` (core fold) / `(fak cmd)` (display) stamp on the matching
   paths.

Do not surface the rev as authoritative for enforcement — the issue's Confusion
risk #1 is exactly that conflation; it is a citable display number, not a gate
input.
