---
title: "next() — the cross-family actuation primitive: one witnessed move that both operator control ops and the autonomous Stop-hook continuation lower to"
date: 2026-07-15
status: FILED 2026-07-15 as epic #4992 + children #4993 (next/seam), #4994 (next/lower-stophook)
family_key: fak-next-key
supersedes: none
prior_art: ["#2753", "#2754", "#2765", "#3559", "#2768", "#2208", "#2396", "#2388", "#4107", "#4108", "#3529", "#2540", "#2533"]
---

# next() — the residual after #2753

**One line.** Several drivers already decide "what the model does next" — the
operator control bus, the trajctl regime nudge, the GatewaySteer channel, the
guard **Stop-hook** continuation, and the re-spawn/relaunch family. The #2753
family already unifies the first three onto one closed operator-control
vocabulary. The **residual** this note scopes is the part #2753 does *not* cover:
(1) the autonomous **Stop-hook continuation** lives in a separate hook/loop
family and is unified with nothing, and (2) there is no **domain-free kernel
primitive** that both the operator ops *and* the autonomous continuation lower
to, with **one witness-of-applied row**. This note is honest that this is a
*narrow superset* of #2753, not a fresh unification.

> **Provenance.** Grounded by a 5-way driver survey + a dedup sweep, a synthesis
> pass, and a 3-lens adversarial review (correctness / doctrine / dedup). All
> three reviewers returned **revise** (not reject), `doctrineSafe=true`,
> `isRealResidual=true`. Their corrections are folded in below — including
> **dropping the original third leaf** (lowering steer/redirect/trajctl-nudge),
> which they showed is ~60% a re-do of open #2765/#3559/#2768.

---

## 1. The honest driver map (ground truth, file-anchored)

The native loop splices the model's next move at **three** sites, in this
source order (verified in `internal/agent/loop.go`):

| Driver | Decider | Payload | Transport | Landing | Witness | Anchor |
|---|---|---|---|---|---|---|
| **GatewaySteer** (freeform) | operator / self-naming machine guard (screened) | freeform prose | `POST /v1/fak/session/{id}/steer` → `drainSteer()` | turn boundary, `RoleUser` | fire-and-forget `202 + logf` (gap) | `loop.go:481‑482` |
| **sessionctl redirect / set-objective** (#2755, closed) | operator (capability-checked) | typed `ControlOp`, `Objective.Directive()` | `internal/sessionctl` bus → `applyRedirect()` | turn boundary, `RoleSystem` | `WitnessDirective`; apply-time refusals **silently dropped** (gap) | `loop.go:493‑494`, `loop_redirect.go:12` |
| **trajctl re-anchor nudge** (#2540 closed / #2533) | `State.DecideNudge` (regime-gated) | the objective's own statement | #2765 bridge → sessionctl → `contextNudge()` | turn boundary, `RoleUser` | `SteerDecision` JSONL | `loop.go:503‑504`, `cmd/fak/trajctl_nudge.go` |
| **guard Stop-hook continuation** | the hook's own ladder (gateway `/metrics` gauges + completion-gate stack) | **exit code + stderr prose** (not a typed op) | Claude Code `Stop` hook, exit 2 = continue / exit 0 = allow-stop | **Stop boundary only** (cannot fire mid-turn or on idle) | `guardStopRecord` JSONL | `cmd/fak/guard_stophook.go` |
| **re-spawn / relaunch family** (loop-drive `loopRunChild`, resume-watchdog `--resume`, guard cap/auth park→`--continue`, cron prompt-assembly) | `loopgate.Adjudicate` / governor | `FAK_GOAL_*` env / seed prompt | fresh child **session start**, not a live boundary | next child's start | per-producer | `internal/agent/loop.go`, `cmd/fak/resume_watchdog_cli.go` |

Two structural facts fall out:

- The first three land as **messages spliced into the running turn**; the
  Stop-hook lands as an **exit code that re-opens an ended turn**; the re-spawn
  family lands at a **fresh child's start**. These are three different physical
  actuation classes, not one.
- A session is driven by **native RunArm** *or* **foreign Claude Code harness**,
  never both — GatewaySteer's `STEER_NO_OWNED_LOOP` gate refuses any session
  with no native loop, so a foreign (Stop-hook-driven) session cannot also take
  a native splice.

## 2. The residual (partial — most of the ambition is already OPEN work)

The repo's **own** dedup gate (`fak issue dedup`, cosine 0.62 over 200 open
issues, 11 clusters) found **no near-duplicate epic** — the control/steer/trajctl
issues aren't even dup-clustered, and the literal term `next()` returns zero
issues. But a *human* reading the naive "unify all the drivers" pitch would
correctly call it a re-file, because **~70% is already OPEN**:

- **#2753** *(epic, OPEN)* — out-of-band operator control: unifies GatewaySteer +
  the OOB control bus + the trajctl re-anchor onto **one** closed control-op
  vocabulary (`internal/sessionctl`; spine shipped in **#2754**, closed).
- **#2765** wires the trajctl driver onto that bus; **#3559** makes steer +
  redirect share one send-right; **#2768** is the explicit *"unify + document the
  control plane"* child; **#2756** adds add-constraint. **#2208** is a parallel
  control-plane epic on the operator-intent-knob axis.

**What no open issue covers (the genuine residual):**

1. **The Stop-hook continuation is unified with nothing.** It is autonomous
   (non-operator), lives in the **#2396 hookbus** / **#2388 owned-turn** family,
   and no issue folds its continue/halt decision together with steer / redirect /
   re-anchor. *Caveat:* #2396 already proposes to make lifecycle hooks
   (including `Stop`) **journaled kernel adjudication rungs** — so the *journal
   the Stop-hook* half is #2396 territory; only the **cross-family actuation**
   (one move type spanning operator ops **and** the autonomous continuation) is
   truly unclaimed.
2. **There is no domain-free kernel `next()` primitive** that all drivers route a
   decided move through, with one witness-of-applied row. #2753 is framed as
   *human* out-of-band control; #2388/#4107 frame the turn as a lifecycle /
   transaction (**TurnRecord #4108** overlaps the witness row) — none as a
   unified driver seam.

**Verdict: `partially_covered`.** A `next()` epic is defensible **only** as the
*kernel-primitive + Stop-hook-inclusive superset of #2753*, citing the prior art
and scoping to (1)+(2) above. Anything wider is a re-file.

## 3. The design (rescoped to the residual)

**The primitive.** A single typed **move** with a **closed vocabulary** and one
**witness-of-applied** row, that every producer lowers a *decided* move to:

- **Closed move vocabulary:** `{continue, redirect, annotate, re-anchor, halt}`.
  An unknown kind is refused `UNCLASSIFIED` at enqueue.
- **Witness-of-applied:** every drained **or refused** move appends exactly one
  `NextRecord` row (op, render, producer, gate, reason, `applied`, trace,
  head-before/after). This **strictly increases** witnessing: today's
  fire-and-forget `202+logf` (plain steer) and silently-dropped redirect
  refusals become ledgered.
- **Router, not recognizer:** the move carries a **non-empty `Gate` attestation**
  of the producer-side gate that already fired. `next()` validates the field's
  *presence*; it **never re-runs** a gate and hosts **no** decision logic
  (`DecideNudge`, the stophook ladder, `loopgate` all stay upstream).

**Where it lives — an explicit, deferred design gate (do not pre-decide here).**
The adversarial review showed the new-leaf rationale ("must sit low enough that
both `internal/agent` and `cmd/fak` import it") **does not hold**: both already
import `internal/sessionctl` today (`loop_redirect.go:12`,
`resume_watchdog_cli.go`, `trajctl_nudge.go`). So the **default** is to *extend
`sessionctl`'s closed vocabulary* with `continue`/`halt`/`annotate` ops + a
witness-of-applied contract (the #2755/#2756 precedent; `trajctl_nudge.go`
already notes annotate "would fit better" there) — **not** to invent a second
closed vocabulary beneath the first. A new `internal/nextctl` leaf is taken
**only** if a concrete import-cycle/tier proof forces it. This keeps the
one-vocabulary doctrine intact.

**Two sinks, one move type, one ledger — single authority made structural.**

- **Native turn-boundary sink:** `loop.go` collapses its three splice sites into
  **one** drain, preserving the **source-site render order** (steer `RoleUser`,
  then redirect `RoleSystem`, then nudge `RoleUser` — i.e. **User, System,
  User**; the byte order is guarded by a golden test). Producers: annotate /
  redirect / re-anchor. *(There is no native `continue` producer — that was a
  phantom in the first draft.)*
- **Stop-hook exit-code sink:** `guard-stophook` constructs `continue→reopen`
  (exit 2 + stderr reason) / `halt→stop` (exit 0) through the **same**
  constructor + witness; the adapter renders the exit code.
- **Enforced XOR (not asserted):** a trace's render class is **bound to its
  session class at enqueue** — a trace may only ever carry one render class — so
  native-splice and Stop-hook-reopen structurally cannot co-drive one session. A
  test witnesses this; it is **not** left to prose.

**The re-spawn/relaunch family is witness-only, by explicit scope.** loop-drive /
resume-watchdog / guard-cap-park / cron lower **only their decision + `NextRecord`
row**, *not* their render (the `FAK_GOAL_*` env / seed-prompt transport stays the
producer's, landing at a fresh child's start). So for that family `next()` is a
**ledger schema, not a router**. We therefore **do not** claim "every driver
routes a move through one router" — only the native-boundary sink and the
Stop-hook sink actually route.

## 4. Doctrine reconciliation

- **No babysitting / interrupt-driven:** no poll loop; producers *enqueue* when
  their own event fires; the loop *drains* at the existing turn boundary.
- **Router-not-recognizer:** enforced by the non-empty-Gate rule + zero decision
  logic in the seam.
- **Healthy-curve-never-nudged (#2533):** `DecideNudge` returns `ActionNone`
  **before** a move is constructed, so a healthy session produces **zero**
  `next()` traffic — the rule physically cannot drift into the router.
- **One authority:** exactly one native drain call; XOR-bound render class means
  no second injection path races the loop.
- **Earned envelope / consent:** every authority gate (operator send-right,
  owned-loop + ingress screen, redirectable-objective, regime gate, `loopgate`
  witness, budget/handoff ceilings) stays **upstream** of enqueue, unchanged.

**Honest limits (folded from the review):**

- **`NextRecord.Gate` is a producer-*claimed* gate, not proof one fired.** The
  witnessing win is real ("strictly more than today's `202+logf` / dropped
  refusals") but the Gate column is **not** forgery-resistant. Do not oversell it.
- **Failure semantics are not unified across sinks:** the Stop-hook sink is
  **fail-open** (undecidable → halt/exit-0, so a hook can never wedge the agent);
  the native sink is **fail-closed-drop** (mailbox always drained, refusals
  `applied=false`). "Unified" holds at the type/ledger layer, not the
  failure-handling layer.
- **Explicit non-goal:** the `now/next/later` + query-bit steer *scheduling
  class* is **out of** the move vocabulary (a named vocabulary-extension
  follow-on, not this epic).
- **trajctl routing:** re-anchor lowers **via sessionctl** (default
  sessionctl-extension design), reconciling with — not superseding — open #2765.

## 5. Adversarial verification result (recorded honestly)

Three independent lenses, each told to refute:

| Lens | Verdict | doctrineSafe | isRealResidual |
|---|---|---|---|
| Correctness | **revise** | ✅ | ✅ |
| Doctrine | **revise** | ✅ | ✅ |
| Dedup/residual | **revise** | ✅ | ✅ |

**Load-bearing corrections (all folded above):** drain order is source-site
(User/System/User), not "SystemDirective-first"; the re-spawn family is
witness-only (drop the "every driver routes through one router" claim); the
phantom native `continue` producer removed; shadow / fail-open Stop-hook
dispositions carry a shadow flag so `Render=Stop` while the would-be disposition
is preserved in `Reason`; the XOR is an **enforced** precondition + test;
`now/next/later` declared a non-goal; **the third leaf (lower steer/redirect/
trajctl-nudge) is dropped** as #2765/#3559/#2768 work; cite #2396/#2388/#4107/
#4108/#3529 as prior art. The `design:repair` pass that would have re-emitted the
revised design hit the workflow's StructuredOutput retry cap; the revisions are
applied by hand here instead.

## 6. Filing decision — RESOLVED: Option A (filed 2026-07-15)

**Chosen: Option A** — a new `fak-next-key` epic positioned as the Stop-hook-
inclusive kernel superset of #2753, plus its two children. Filed:
- **Epic #4992** — `next/epic` (labels `epic`, `dispatch`)
- **#4993** — `next/seam` (labels `enhancement`, `dispatch`; contract `ready`)
- **#4994** — `next/lower-stophook` (labels `enhancement`, `dispatch`; contract `ready`)

The reviewers surfaced this fork; the maintainer chose A:

- **Option A — new `fak-next-key` epic, positioned as the superset of #2753.**
  Matches the house pattern (family-scoped epics: #2753, #4477, #2208). The epic
  **cites and scopes** to the residual and declares it will **subsume** #2753's
  unification children as they land. *Recommended* — the dedup gate found no
  dup-cluster, and the residual is cross-family (spans #2753 **and** #2396), so
  it does not fit cleanly under either parent alone.
- **Option B — no new epic; file the two residual pieces as children of #2396
  (hookbus) + a cross-epic bridge issue** to #2753. Lower-surface, but splits a
  single coherent seam across two parents.

The ready bodies below are written for **Option A** (with the #2396 relationship
stated inside the Stop-hook child). Switching to B = re-parent the two children
and drop the epic.

---

## 7. Ready-to-file bodies (contract-shaped; dedup marker per body)

> Filing convention (per #2540's Batch policy): each body embeds one
> `<!-- fak-next-key: <slug> -->` marker; the write-time near-dup gate (#2504)
> **updates** on re-file rather than duplicating. Re-validate any edit with
> `fak issue contract` before updating a filed issue.
>
> **Contract verdict — measured 2026-07-15** (`fak issue contract --from-issues`,
> default gates; `refused: 0`):
>
> | Body | Verdict | Dispatchable | Lane | Steps / Scale |
> |---|---|---|---|---|
> | `next/seam` | **`ready`** | ✅ | sessionctl | 4 / S1 |
> | `next/lower-stophook` | **`ready`** | ✅ | cmd | 5 / S1 |
> | `next/epic` | `needs_scope` (`ISSUE_NOT_DISPATCH_LEAF`) | — (parent) | cmd | 2 / S1 |
>
> Both children pass the spine-first contract and are dispatch-ready. The epic's
> `needs_scope` is the **correct** parent verdict (an epic is never a dispatch
> leaf — it decomposes into the two children, which it does). Advisory-only flags
> not enforced by default gates: `model_tier`/`agent_context`/`batch_policy`
> metadata absent on all three (add before filing if the repo's `--strict-*`
> gates are armed at sync time).

### Epic — `next/epic` → **filed #4992**

Filed via `gh issue create` labels `epic`, `dispatch` (repo has no `priority/*`
labels). Children checklist (#4993/#4994) appended to the live body.

<!-- fak-next-key: next/epic -->

**Title:** `epic(next): the cross-family actuation primitive — one witnessed move that operator control ops and the autonomous Stop-hook continuation both lower to`

**Dispatch** · the model's "next move" is decided by five drivers across three families; #2753 unifies three of them, the Stop-hook continuation is unified with nothing, and no kernel primitive + witness spans both families.

#### Parent context
Superset of the #2753 out-of-band control family. Design note:
`docs/notes/CONCEPT-NEXT-PRIMITIVE-UNIFY-2026-07-15.md`. **Prior art (must be
honored, not re-done):** #2753/#2754/#2765/#3559/#2768/#2756 (operator control
plane), #2208 (intent-knob plane), #2396 (hookbus — where the Stop-hook lives),
#2388/#4107/#4108 (owned/managed turn + TurnRecord), #3529/#2540/#2533 (trajctl
re-anchor). This epic **subsumes** #2753's unification children as they land; it
does not re-open their work.

#### Current state
Three native drivers (GatewaySteer, sessionctl redirect, trajctl nudge) already
converge on the `internal/sessionctl` vocabulary (#2753; spine #2754 shipped) and
splice at the turn boundary (`loop.go:481/493/503`). The **guard Stop-hook**
continuation decides autonomously via exit-code protocol
(`cmd/fak/guard_stophook.go`), is not a sessionctl op, and is unified with
nothing. There is no domain-free primitive both families lower a *decided* move
to, and the witness is scattered (`guardStopRecord`, `SteerDecision`,
fire-and-forget `202+logf`, silently-dropped refusals).

#### Why now
#2753's spine is shipped and its children are in flight, so the operator side is
converging. The Stop-hook continuation is the one named driver #2753 structurally
cannot absorb (it is autonomous, exit-code, foreign-harness). Folding it in now —
as a thin kernel move + witness the operator ops *also* lower to — closes the
cross-family residual before a fourth control surface accretes.

#### Working spine
A single closed **move vocabulary** `{continue, redirect, annotate, re-anchor,
halt}` + a **witness-of-applied** `NextRecord` row, with **two sinks** (native
turn-boundary splice; Stop-hook exit-code) sharing one move type and one ledger.
**Default implementation: extend `internal/sessionctl`** (both `internal/agent`
and `cmd/fak` already import it) rather than a second vocabulary; a new leaf only
if a cycle/tier proof forces it. Router-not-recognizer (each move attests the
gate that already fired; the seam re-runs none). Single authority via an
**enforced** native-XOR-foreign render-class binding.

#### In scope
Child `next/seam` (the shared move + witness contract) and child
`next/lower-stophook` (fold the Stop-hook continuation + the re-spawn family's
decision/witness onto it, prove single authority).

#### Out of scope
Lowering steer / redirect / trajctl-nudge onto the seam — that is **#2765 /
#3559 / #2768** and must be consumed, **not re-executed**. Drive-state ops
(pause/resume/cancel/throttle/budget) stay on the sessionctl control route. The
`now/next/later` scheduling class + query bit (a later vocabulary extension).
Making `Gate` forgery-resistant (it records a producer *claim*).

#### Done condition
A single closed move type + `NextRecord` witness exists; the Stop-hook continue/
halt decision and the re-spawn family's decision both lower to it; the operator
ops lower via the same contract as #2753's children land; a structural test
proves the native loop drains exactly one seam and the two sinks never co-drive a
trace.

#### Witness
Per child. Epic-level: `dos_verify(NEXT, next/seam)` and
`dos_verify(NEXT, next/lower-stophook)` both SHIPPED; the single-authority
structural test green.

#### Acceptance gate
Both children merged and green; the #2765/#3559/#2768 relationship reconciled in
the docs (#2768).

#### Work unit
Epic — two worker-ready children below.

#### Expected steps
2 (one per child).

#### Assumptions
- Extending `sessionctl` (vs a new `internal/nextctl`) is sufficient — both
  callers already import it. *(Falsifiable: an import cycle forces a new leaf.)*
- The Stop-hook's exit-code disposition and the sessionctl ops can share one
  move type despite rendering differently (message splice vs exit code).

#### Confusion risks
- This is **not** a re-file of #2753. #2753 unifies the three *operator* drivers;
  this adds the *autonomous Stop-hook* + a kernel move/witness both families share.
  If a change here starts lowering steer/redirect/trajctl-nudge, it has crossed
  into #2765/#3559/#2768 — stop.
- "One seam" = one move type + one ledger with **two renders** (splice XOR exit
  code), not one physical drain. Reading the two-sink design as a single-authority
  violation is the #5 misread; the XOR is enforced, not asserted.
- `NextRecord.Gate` is a producer-*claimed* gate, not proof one fired.

#### Coordination
- `internal/sessionctl`, `internal/agent`, and `cmd/fak` guard are contended
  lanes — arbitrate via `dos_arbitrate` before writing. Reconcile with #2396
  (Stop-hook journaling) and #2768 (control-plane docs).

#### Lane
cmd

#### Likely files
- `internal/sessionctl/` (default seam home)
- `internal/agent/loop.go`
- `cmd/fak/guard_stophook.go`

#### Closure binding
Closes when `next/seam` and `next/lower-stophook` are merged and green and the
single-authority + XOR structural tests pass.

#### Trigger
Design note parked 2026-07-15; residual after #2753.

---

### Child 1 — `next/seam` → **filed #4993** (contract `ready`)

Filed via `gh issue create` labels `enhancement`, `dispatch`.

<!-- fak-next-key: next/seam -->

**Title:** `feat(next): the shared move contract — closed vocabulary {continue,redirect,annotate,re-anchor,halt} + witness-of-applied NextRecord, no producer wired`

**Dispatch** · stand up the primitive both families lower to, with zero behavior change and no producer attached, so the fold in the next leaf is a pure rewire.

#### Parent context
Child of `next/epic`. Implements §3 of the design note. **Default: extend
`internal/sessionctl`'s closed vocabulary** (per #2755/#2756) — both
`internal/agent` and `cmd/fak` already import it; take a new `internal/nextctl`
leaf **only** with a concrete import-cycle proof.

#### Current state
`internal/sessionctl` (#2754, shipped) is a closed control-op vocabulary with a
witness-of-applied contract, but it has no `continue`/`halt` op and no move type
the *autonomous* Stop-hook could lower to; the Stop-hook's disposition is a bare
exit code with a `guardStopRecord` side-ledger.

#### Why now
The move contract must land **before** any producer is rewired, so the fold
(child 2) is a mechanical rewire against a stable, tested seam rather than a
big-bang change.

#### Working spine
Add the closed move kind `{continue, redirect, annotate, re-anchor, halt}` and a
closed render `{user-splice, system-directive, reopen, stop}` (a `shadow` flag
carries a would-continue-but-allowed Stop-hook disposition so `render=stop` while
the intent is preserved in `reason`). Add `enqueue` (validates shape + closed
vocabulary + **non-empty gate**, never re-runs a gate) / one boundary `drain`
(deterministic **source-site order**: user, system, user) / `witness` (one
`NextRecord` row per move, incl. no-ops/refusals). Bind a move's render class to
the session class at enqueue (the XOR precondition). No producer is wired in this
leaf.

#### In scope
The move + render enums, the enqueue/drain/witness surface, the render-class↔
session-class binding, a completeness test naming any kind lacking a render/
witness mapping, and (if a new leaf is taken) the architest tier row + pureRoot
entry.

#### Out of scope
Wiring any producer (child 2). Lowering steer/redirect/trajctl-nudge
(#2765/#3559/#2768). Drive-state ops.

#### Done condition
The seam package compiles and `go test` is green; the completeness test fails
loudly if any move kind lacks a render/witness mapping; enqueue rejects an
empty-gate move and an out-of-vocabulary kind; **no file under `internal/agent`
or `cmd/fak` changes in this leaf**. If a new leaf: `TestEveryPackageDeclaresTier`
passes with it tiered.

#### Witness
A table test constructs one move per kind, drains, and **re-reads the appended
`NextRecord` JSONL bytes** (not an in-memory return) asserting each row's op is in
the closed set, gate is non-empty, and refused moves are `applied=false`; plus a
test asserting a trace cannot carry two render classes. `dos_commit_audit` grades
the diff-witnessed commit; `dos_verify(NEXT, next/seam)` SHIPPED.

#### Acceptance gate
Done condition + `make ci` green.

#### Work unit
One worker owns the move contract, the witness row, and the tests.

#### Expected steps
4

#### Assumptions
- Extending `sessionctl` avoids an import cycle (both callers already import it).

#### Confusion risks
- Do **not** wire a producer or delete an existing splice site here — this leaf
  is the contract only; the rewire is child 2. A second closed vocabulary under
  `sessionctl` is a doctrine smell — prefer extension.

#### Coordination
- Touches `internal/sessionctl` — arbitrate the lane first.

#### Lane
sessionctl

#### Likely files
- `internal/sessionctl/` (move kind + render + witness)
- (only if a new leaf is forced) `internal/nextctl/`

#### Closure binding
The `Witness` test above, green on a merged PR whose body carries `next/seam`.

#### Trigger
Epic `next/epic`.

---

### Child 2 — `next/lower-stophook` → **filed #4994** (contract `ready`)

Filed via `gh issue create` labels `enhancement`, `dispatch`.

<!-- fak-next-key: next/lower-stophook -->

**Title:** `feat(next): lower the guard Stop-hook continuation + the re-spawn family's decision/witness onto the shared move, and prove single authority`

**Dispatch** · fold the one autonomous driver #2753 cannot absorb onto the shared move + witness, preserving the Stop-hook fail-open invariant, and witness the single-authority XOR.

#### Parent context
Child of `next/epic`; consumes child `next/seam`. **Reconciliation with #2396
(hookbus):** #2396 owns making lifecycle hooks journaled kernel rungs; this leaf
does **not** re-implement that — it lowers the Stop-hook's *continue/halt
disposition* onto the shared move so the autonomous continuation is witnessed in
the **same** schema as the operator ops. If #2396 lands first, this projects
`guardStopRecord` through its journal; if not, `NextRecord` is the spine and
#2396 folds it later. State the chosen relationship in the PR body.

#### Current state
`guard-stophook` returns a bare exit code (2 = continue via stderr reason, 0 =
allow-stop) with fail-open on gateway unavailability, ledgered only in
`guardStopRecord`. The re-spawn family (loop-drive, resume-watchdog, guard
cap-park, cron) decides continue/halt via `loopgate` and lands at a fresh child's
start, ledgered per-producer. Neither shares the operator ops' witness.

#### Why now
Once the shared move exists (child 1), folding the Stop-hook is a bounded rewire
that closes the cross-family residual and makes single authority testable.

#### Working spine
`guard-stophook` constructs `continue→reopen` (exit 2 + existing stderr reason as
payload) / `halt→stop` (exit 0) through the shared constructor + witness; a
`shadow` flag carries would-continue-but-allowed dispositions so no
kind/render contradiction and the projection loses no `fail_open_*` signal. The
re-spawn family records **decision + witness only** (render stays its
`FAK_GOAL_*`/seed transport). `guardStopRecord` becomes a superset projecting the
unified row. Add the single-authority structural test.

#### In scope
The Stop-hook move constructor (fail-open preserved), the re-spawn family's
`NextRecord` decision rows, `guardStopRecord`→unified-row projection, and the
single-authority + XOR structural tests.

#### Out of scope
Lowering steer/redirect/trajctl-nudge (#2765/#3559/#2768). Re-implementing #2396's
hook journaling. Folding the half-landed per-session handoff ceiling. Changing any
completion oracle (`loopgate` stays the decider).

#### Done condition
A structural test proves each producer reaches the loop only via the shared
`enqueue` and the native loop exposes exactly one drain call site; the Stop-hook
**fail-open** test stays green (exit 0 + a `NextRecord(applied=false)` on any
undecidable/malformed state); `guard-stops` folds the unified rows; the XOR test
proves a trace never carries both a reopen and a splice; **`go build ./...` is
green on a non-skewed base** (the `guard_stophook_test.go` 6-arg vs 3-arg handoff
skew is **not** folded in).

#### Witness
A structural test over the parsed call graph (not a claim) enumerates every
producer and asserts the single `enqueue`/single-drain shape; a fail-open test
injects a malformed stophook decision and asserts **exit 0 + `NextRecord(applied
=false)`**; `dos_commit_audit` diff-witnessed under `(fak manage)`;
`dos_verify(NEXT, next/lower-stophook)` SHIPPED.

#### Acceptance gate
Done condition + `make ci` green on a non-skewed base.

#### Work unit
One worker owns the Stop-hook lowering, the re-spawn witness rows, and the
structural/fail-open tests.

#### Expected steps
5

#### Assumptions
- The Stop-hook exit-code disposition maps cleanly to `continue/halt` + `shadow`
  without losing a `fail_open_*`/shadow-mode signal.
- A non-skewed `cmd/fak` base is available (or HEAD-overlay-pin the skewed files).

#### Confusion risks
- Preserve **fail-open**: any enqueue/validate failure or undecidable state must
  render `halt`/exit-0 so a Stop hook can never wedge the agent. Do **not** move
  the stophook ladder or `loopgate` into the seam (router-not-recognizer).
- Do not fold the half-landed handoff ceiling; land on green.

#### Coordination
- `cmd/fak` guard + `internal/agent` loop are contended — arbitrate first;
  reconcile with #2396.

#### Lane
cmd

#### Likely files
- `cmd/fak/guard_stophook.go`
- `internal/agent/loop.go`
- `cmd/fak/resume_watchdog_cli.go`

#### Closure binding
The single-authority + fail-open tests above, green on a merged PR carrying
`next/lower-stophook`.

#### Trigger
Epic `next/epic`.

---

## 8. Honesty fences (what this does NOT do)

- It does **not** unify all five drivers into one router — the re-spawn family is
  **witness-only**, and steer/redirect/trajctl-nudge stay #2765/#3559/#2768's job.
- It does **not** prove a gate fired — `NextRecord.Gate` is a producer claim.
- It does **not** journal lifecycle hooks (that is #2396); it lowers the
  Stop-hook *disposition* onto a shared move.
- It is **partially covered** prior art (~70% is open #2753-family work); it is
  defensible only as the narrow Stop-hook-inclusive kernel superset.

**Next checkable step:** run `fak issue contract --from-issues` over the three
bodies, record the measured verdicts in §7, then take the §6 filing decision.
