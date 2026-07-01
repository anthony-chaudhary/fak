---
title: "Perpetual sessions: bounded-context relays instead of compaction"
description: "A doctrine and epic spine for running a goal forever in O(1) peak context — stop at a safe point, externalize every load-bearing fact to a durable store, hand a small typed baton to a fresh session, and re-derive the rest by query. Explicitly not compaction."
date: 2026-07-01
---

# Perpetual sessions (the relay)

Status: concept note plus GitHub epic spine. Nothing here is shipped yet. The
staged sketch below decomposes into the child issues listed under **The rungs**;
the doctrine is written to be enforced the fak way — closed vocabulary, evidence
over self-report, data over code, disjointness before fan-out, advisory before
enforcement, fail-closed.

## The problem

Some goals do not fit in a context window and never will: an overnight backlog
drain, a migration across a thousand call sites, a benchmark harness that runs
for days, a "keep this green" watch. The work is unbounded in the transcript
even when it is perfectly bounded in the *artifacts* it produces (commits, ledger
rows, issue closures).

The industry's default answer is **compaction**: when the window fills, summarize
the transcript in place and continue. Anthropic ships this server-side, and Claude
Code fires it automatically near the limit. Compaction is the right tool for a
*conversation* a human is steering. It is the wrong tool for a *perpetual machine
goal*, for four reasons:

- **It is lossy under pressure.** Auto-compaction fires reactively at ~95% of the
  window — exactly when the model is already operating in a degraded, context-rot
  state — and the summary it writes is made under that same pressure. It routinely
  keeps mundane tool output and drops the architectural decision.
- **It re-derives what is already durable.** For this use case every fact that
  matters is *already* written somewhere queryable: the commit is in git, the
  progress is in the intent ledger, the decision is a memory file, the closure is
  a GitHub issue. Summarizing the transcript re-encodes, lossily, a shadow of
  records that already exist losslessly outside it.
- **It carries poison forward.** A wrong turn, a stale assumption, a dead-end
  exploration — compaction faithfully preserves a blurred copy of all of it. The
  fresh window inherits the confusion instead of shedding it.
- **It still busts the cache and is non-deterministic.** Rewriting the middle of
  the prefix invalidates the prompt cache from the compaction point anyway, and
  the summary is a model sample, so the same run is not reproducible.

So for the perpetual use case we do not want a *better* compaction. We want **no
compaction**. The thesis of this note:

> A load-bearing fact is either a durable record outside the context (and is
> queried on demand), or it does not survive at all. The transcript is disposable.
> When a leg of the goal fills its window, it does not shrink — it **ends cleanly
> and a fresh leg takes over**, seeded only by a small typed baton and a query
> interface into the durable store.

We call this a **relay**: one goal, run as a sequence of bounded **legs**, each
handing a **baton** to the next. The invariant we are buying is:

> **Flat context (the "O(1)" the operator asked about):** peak resident tokens in
> a relay are bounded by the per-leg ceiling, *independent of goal duration or
> total work done.* A relay that runs for a week peaks no higher than one that
> runs for an hour.

## Where the state of the art is

The pattern is not exotic; it is the hardened, witnessed form of things people
already do by hand.

- **Anthropic's platform primitives** are three: *context editing* (clears stale
  tool results in place, `clear_tool_uses_20250919`), the *memory tool* (a
  file-based store *outside* the window with just-in-time retrieval), and
  *compaction* (server-side summarization). Their own guidance for agents that
  span sessions is the tell: *"design your state artifacts so that context
  recovery is fast when a new session starts."* The relay takes that to its limit
  — recovery is a query, not a summary. We borrow the memory tool's
  just-in-time-retrieval shape and reject compaction for this mode.
- **Claude Code** persists each session as JSONL and starts every new session with
  a *fresh* window; the community's answer to context limits is the **`HANDOFF.md`
  habit** — write goal/state/files/decisions/next-step to a markdown file, then
  hand it to a fresh session. The relay is that habit made a first-class,
  schema'd, *witnessed* mechanism: the baton is a `HANDOFF.md` that cannot lie
  about progress because progress is re-verified from git, not trusted from the
  note.
- **MemGPT / Letta** frame the window as RAM and page to disk via tool calls, with
  the model self-editing its own memory. Their documented tradeoff is the warning
  we design against: *"memory quality depends entirely on the model's judgment. If
  the model fails to save something, it's gone,"* and every page-op costs
  inference. The relay deliberately does **not** ask the model to curate memory
  mid-flight. Carryover is not model-sampled at rotation time; it is a projection
  of records the model *already committed* through the normal, witnessed work
  path. The durable store is git and the ledger, not a summary the model wrote
  under deadline.

The one-line contrast: compaction and self-paging both ask a pressured model to
decide *what to remember*. The relay never asks that question — it asks *what did
this leg durably ship*, and that has a witnessed answer.

## What fak already has (we are extending, not starting)

This is the crucial part: the substrate is largely built under the managed-context
epic **#1570**. The relay is a *mode* over it, plus a few genuinely new gates.

| Existing surface | What it already gives the relay |
|---|---|
| `internal/ctxplan` (`Plan`, `Faithful`, `ObjectivePin`) | Treats the turn as an O(1) view over a **lossless** store and *already refuses lossy compaction* (`Faithful` fails any plan that elides a span with no recovery handle). This is the relay's philosophy already in code. |
| `internal/sessionreset` (`BuildSeed`, `Contributor`, `Seed`) | Deterministic, de-LLM-ified carryover via a **contributor registry** (`durabilityFacts`, `taskDistill`, `warmPrefix`, `verbatimTail`). The relay is a *stricter contributor policy* over this seam. |
| `session.Recontinue` / `Generation` / `ContinuationID` / `ParentTrace` | The re-arm verb and lineage: mint a fresh live session under a child trace, link to parent, `Generation++`, leave parent `Stopped`. A relay leg **is** a generation. |
| `session.ResetTransaction` | The audit ledger of a reset: old/new trace, seed digest, contributors, **omitted spans with reasons**, budget re-arm. The baton's provenance record already exists. |
| `internal/sessionimage` (`fak.session.v1`) | Portable, integrity-checked offload/restore across host and model. The relay's durable checkpoint format. |
| `session.Descriptor` + `FileStore` | The small durable index of a session, outside the transcript. |
| `session.TimeBudget` / `Envelope` / `Budget` / `Decide`→`Verdict` | Multi-axis budget with **closed stop-reason tokens** (`BUDGET_CONTEXT_EXHAUSTED`, `DRAINING`, …) and wall-clock accounting that survives process restart. The relay's rotation trigger reuses these axes. |
| Handoff-to-close chain (**#1462**, `taskmgr.Handoff`, `dos commit-audit`) | An end-to-end, hermetically-tested path from a handoff through routing, commit, and witness. The relay's close arm binds here. |
| `internal/corelocks`, guard, witness, `safecommit` | Where "is it safe to stop right now" is already adjudicated. |

The gap the relay closes is **not** persistence, lineage, or carryover — those
exist. It is three things that do not yet exist:

1. a **pointer-only carryover policy** (a baton that is O(1) and re-verified, not a
   distilled recap that grows with the work);
2. a **safe-stop-at-a-work-boundary** predicate and an **externalize gate** that
   fails closed if any load-bearing fact lives only in the transcript;
3. **perpetual-goal semantics** — a done-check, anti-thrash hysteresis, and a
   no-progress escape — so a relay terminates on the goal, not on the window.

## Vocabulary

A closed, small vocabulary (DOS lesson #1). Recommended names, with alternatives
called out in **Open questions**.

- **Relay** — a single goal executed as an ordered sequence of legs. The unit the
  operator starts and the goal the done-check is measured against.
- **Leg** — one bounded session (one `Generation`) of the relay. A leg has a
  window ceiling and ends at a safe point, never mid-action.
- **Baton** — the forward-carried, **pointer-only** handoff a leg writes for its
  successor. O(1) in size. Contains the objective pin, a re-verifiable progress
  cursor, the single next action, open questions, and *pointers* (SHAs, ledger
  ids, memory slugs, issue numbers, file globs) into the durable store — never
  transcript bytes, never a model-written recap of the work.
- **Tombstone** — the closing leg's typed exit record: a closed reason token
  (why it ended) plus an honest, witness-backed self-assessment. The baton
  *carries* the tombstone as its header; a leg that could not reach a safe point
  writes a `RELAY_PARKED_UNSAFE` tombstone instead of a clean one.
- **Externalize gate** — the fail-closed check, run before a rotation is allowed,
  that every load-bearing fact of this leg is already in the durable store
  (committed, ledgered, or filed). If not, the rotation is refused.
- **Safe point / quiescence** — a work boundary at which a leg may end without
  losing or corrupting work: no in-flight tool call, no half-written commit, the
  tree is green or explicitly parked, and the next action is expressible as a
  single baton line.
- **Durable store** — the union of the records that already survive a session:
  git (commits, notes), the intent ledger / run registry, agent memory files, and
  GitHub issues. Everything the relay carries forward is a *pointer* into one of
  these, and each pointer resolves to something a witness can check.
- **Flat-context invariant** — peak resident tokens bounded by the leg ceiling,
  independent of goal duration. The property being purchased.

## The baton schema

The baton is data, not prose (DOS lesson #3), and it is the *least* trusted signal
in the system, so its progress half is never a claim — it is a cursor the next leg
**re-verifies** (DOS lesson #2). A deliberately closed schema:

```jsonc
{
  "schema": "fak.relay.baton.v1",
  "relay_id": "RLY-...",          // stable across all legs of the goal
  "leg": 7,                        // monotonic; == session.Generation
  "parent_trace": "…",             // lineage link (session.ParentTrace)

  "objective": {                   // == ctxplan.ObjectivePin, carried verbatim
    "pin_id": "…", "text": "…", "digest": "…"
  },
  "done_when": "…",                // how a fresh leg checks 'am I already done?'

  "progress_cursor": {             // RE-VERIFIED, never trusted. No 'claimed' field.
    "start_sha": "…",              // ground-truth anchor for dos_status/verify
    "ledger_ref": "…",             // intent-ledger id to re-read verified progress
    "held_region": ["glob", …]     // lease region to re-acquire (disjointness)
  },

  "next_action": "…",              // the single next step, one line
  "open_questions": ["…"],         // unresolved decisions (pointers, not essays)

  "artifacts": [                   // pointers into the durable store, that's all
    {"kind": "commit",  "ref": "…"},
    {"kind": "issue",   "ref": "#1234"},
    {"kind": "memory",  "ref": "slug"},
    {"kind": "ledger",  "ref": "…"},
    {"kind": "file",    "ref": "path/glob"}
  ],
  "do_not_rederive": ["#dead-end-1", "memory:tried-X"],  // pointers to closed paths

  "tombstone": {                   // the closing leg's typed death note
    "reason": "RELAY_ROTATED",     // closed token (see below)
    "at_sha": "…",
    "note": "…"                    // honest, short, witness-backed
  }
}
```

Two properties are load-bearing and mirror the fail-closed A2A digest shape of
`dos_status`:

- **No `claimed` field, by construction.** The baton hands the successor a
  *pointer* to verified progress (`start_sha`, `ledger_ref`), never a number the
  closing leg asserted. A successor structurally cannot pick up a self-report it is
  never handed.
- **Re-verified at read, like a recalled memory.** A baton is a frozen self-report
  — the same trust class as an agent memory. The successor runs the
  `dos_recall`/`dos_status` discipline on it before building: does the cursor still
  match git? Are the cited commits ancestors of HEAD? If not, the baton is
  `RELAY_BATON_STALE` and the leg re-derives from the durable store rather than
  trusting the note.

## The relay lifecycle

One leg's loop, with the new gates in **bold**:

1. **Reload + re-verify.** Start a fresh window seeded only by the baton + a query
   tool. **Re-verify the baton** against git/ledger (dos_recall discipline). On
   mismatch, mark `RELAY_BATON_STALE` and rebuild the cursor from ground truth.
2. **Done-check.** Evaluate `done_when` against the durable store *first*. If the
   goal is already satisfied, end the relay with `RELAY_GOAL_DONE` and write no new
   leg. (Idempotent restart: a relay that is done stays done.)
3. **Work.** Do the next action and beyond, committing/ledgering/filing as normal.
   Every result lands in the durable store through the ordinary witnessed path —
   this is what makes the externalize gate cheap.
4. **Arm at threshold.** When a rotation trigger crosses its soft mark (context %,
   turns, wall-clock, spend — reused `Envelope` axes), set `RELAY_ARMED`. Arming
   does not stop anything; it just says "rotate at the next safe point."
5. **Reach a safe point.** Continue only to the next quiescence boundary. Do not
   rotate mid-action.
6. **Externalize gate (fail-closed).** Before rotating, confirm nothing
   load-bearing lives only in the transcript. If uncommitted/unfiled state exists,
   refuse with `RELAY_NOT_EXTERNALIZED` — commit/file it, or park.
7. **Write baton + tombstone, rotate.** Project the pointer-only baton, stamp the
   tombstone reason, `Recontinue` into leg N+1. The old leg is `Stopped`; its
   transcript is discarded, not summarized.
8. **Hard ceiling.** If the window ceiling is hit before a safe point ever arrives,
   fail closed: write a `RELAY_PARKED_UNSAFE` tombstone and stop. Never blow the
   window to keep going; a parked relay is resumable by a careful next leg or an
   operator, an overrun one is not.

## Closed reason vocabulary (data-only until a floor consumes it)

Proposed tokens, in the DOS style — each emittable, verifiable, and refusable.
Data-only until a named check opts into enforcing them (DOS lesson #6):

Authoritative `summary` + `fix` rows live in
[`RELAY-REASON-VOCABULARY-2026-07-01.md`](RELAY-REASON-VOCABULARY-2026-07-01.md);
the table below is the short spine view.

| Token | Category | Meaning |
|---|---|---|
| `RELAY_ARMED` | advisory | soft threshold crossed; rotate at next safe point |
| `RELAY_ROTATED` | TRUE_DRAIN | clean rotation at a safe boundary (normal leg end) |
| `RELAY_GOAL_DONE` | TRUE_DRAIN | `done_when` satisfied against the durable store; relay ends |
| `RELAY_NOT_EXTERNALIZED` | STALE_CLAIM | refuse to rotate: load-bearing state lives only in the transcript |
| `RELAY_PARKED_UNSAFE` | OPERATOR_GATE | hit the hard ceiling before a safe point; parked, needs careful resume |
| `RELAY_BATON_STALE` | STALE_CLAIM | reload re-verification found the baton no longer matches ground truth |
| `RELAY_NO_PROGRESS` | OPERATOR_GATE | N consecutive legs made no verified progress; stop and escalate |

## Thresholds and triggers

Rotation is **two-phase** (arm, then fire) so it never lands mid-action:

- **Arm** at a *soft* mark, well below the wall. SOTA consensus is to rotate around
  50–70% of the window, not 95% — a leg that arms at 60% still has clean headroom
  to reach a safe point and externalize. Reuse the `Envelope` axes: context tokens
  (primary), turns, wall-clock, spend. The soft mark is policy data, not a magic
  number in code (DOS lesson #3) — a `[relay]` table in `dos.toml` or an `Envelope`
  field.
- **Fire** at the next safe point after arming.
- **Hard ceiling** as a fail-closed backstop (the `RELAY_PARKED_UNSAFE` path).

Anti-thrash / hysteresis (a failure mode in its own right, below): a leg must make
some minimum verified progress before it is allowed to arm again, a relay caps
rotations per wall-clock hour, and `RELAY_NO_PROGRESS` fires if K consecutive legs
close with no forward `progress_cursor` movement — the relay stops and escalates
rather than spinning fresh windows forever.

## Safe-stop-point detection

A safe point reuses existing adjudication rather than inventing a new one. It is
the conjunction:

- **No in-flight tool call** — the turn boundary model already guarantees this
  (`Decide` gates between turns; `Draining` runs one more boundary then stops,
  never mid-decode).
- **No half-written commit / green-or-parked tree** — `safecommit` / guard already
  know whether the working tree is at a committable boundary.
- **Next action is baton-expressible** — the leg can name its single next step in
  one line. If it cannot, it is mid-thought, not at a boundary.

The relay does not get a bespoke stop mechanism; it composes `Draining` +
`safecommit` + a one-line-next-action predicate. The `RELAY_ARMED` → safe-point →
externalize sequence is the only new control flow.

## Lessons applied from DOS

The relay is deliberately built to the same doctrine as core-locks:

1. **Closed vocabulary.** Rotation and tombstone reasons are a fixed set; a relay
   never refuses with prose.
2. **Evidence, not claims.** The baton carries no `claimed` progress; the
   successor re-verifies the cursor from git/ledger before trusting it.
3. **Data, not code.** Rotation policy (soft marks, caps, done-check hooks) lives
   in `dos.toml`/`Envelope`, not scattered constants.
4. **Disjointness before fan-out.** A relay holds a lane across legs; the baton
   carries `held_region` so leg N+1 re-acquires the *same* lease and does not
   collide with peers on the shared tree.
5. **Advisory before enforcement.** Ship as shadow first: emit "would rotate here"
   and score baton fidelity for weeks before any auto-fire. (This reuses the
   PreCompact shadow-hook posture already in the tree.)
6. **No spontaneous refusal from vocabulary alone.** Adding the reason tokens
   blocks nothing until a named gate (the externalize gate) consumes them.
7. **Both lenses.** The relay is also an optimization: flat context means a stable,
   cacheable steady-state prefix (system + O(1) baton), lower token spend than a
   growing-then-compacting window, and no context-rot accuracy decay.

## Failure modes and anti-patterns

| Failure | Mitigation |
|---|---|
| **Tombstone rot / stale handoff** — baton says progress that git no longer reflects | Re-verify at read (`dos_recall` discipline); `RELAY_BATON_STALE` forces re-derivation from the durable store. |
| **Thrash** — rotating so often no leg makes progress | Hysteresis: min verified progress before re-arm; per-hour rotation cap; `RELAY_NO_PROGRESS` escape after K empty legs. |
| **Hidden-state loss** — a load-bearing fact lived only in the transcript | The externalize gate fails closed (`RELAY_NOT_EXTERNALIZED`); rotation is impossible until it is durable. |
| **Mid-action rotation** — window hit during a tool call or half-commit | Two-phase arm/fire; safe-point predicate; hard-ceiling parks rather than cuts. |
| **Goal drift** — successor pursues a subtly different goal | Objective pin carried verbatim + content-digested (`ctxplan.ObjectivePin`, `ReconcileObjective`); surfaces a typed outcome on mismatch, never a silent rewrite. |
| **Infinite relay** — a goal with no real end condition | `done_when` re-checked each leg; a relay is bounded by the goal or an operator `max-legs`/`max-spend` envelope. |
| **Cache thrash** — every leg pays a cold prefix | Steady-state prefix is small and stable (system + baton), so it caches; contrast compaction, which rewrites the middle every time anyway. |
| **Poison carryover** — dead ends re-inherited | Only pointers cross the boundary; `do_not_rederive` is a pointer index, so the fresh window sheds the confusion instead of blurring it forward. |

## Implementation sketch (phases)

Stageable without destabilizing the tree, mirroring the core-locks rollout:

- **Phase 0 — observe only.** Instrument context growth per leg; emit a shadow
  "would rotate here at soft mark X" signal and a would-be baton; change no
  behavior. Reuse the PreCompact shadow-hook seam.
- **Phase 1 — baton schema, offline.** `fak.relay.baton.v1` type + a pure
  `Parse`/project like `session.Envelope`: deterministic, no I/O, witness test on
  round-trip. `fak relay handoff` writes it; `fak relay resume` reads it.
- **Phase 2 — reload re-verification.** The `dos_recall`-style freshness check on a
  baton at leg start; `RELAY_BATON_STALE` on mismatch.
- **Phase 3 — safe-point predicate.** Compose `Draining` + `safecommit` +
  one-line-next-action into a `relay safe-point?` check.
- **Phase 4 — externalize gate (fail-closed).** Refuse rotation with
  `RELAY_NOT_EXTERNALIZED` when transcript-only state exists.
- **Phase 5 — rotation policy as data.** `[relay]` table (soft marks, caps,
  `done_when` hook) in `dos.toml`/`Envelope`; two-phase arm/fire; hysteresis.
- **Phase 6 — the relay driver.** Orchestrate leg → arm → safe-point → externalize
  → baton → `Recontinue`, with lease continuity across legs and the done-check.
- **Phase 7 — pointer-only carryover policy.** A `sessionreset` contributor set
  restricted to durable pointers (drop `verbatimTail`, forbid model-written recap).
- **Phase 8 — dogfood + QA.** Below.

## Dogfooding and QA

A relay is only real if a long goal survives many rotations with its progress
intact and its cost bounded. The QA bar:

- **Fidelity across N rotations.** Run a real multi-hour goal (e.g. a backlog
  drain) through ≥10 forced rotations; assert every closed issue / commit is still
  attributable and the objective pin is byte-identical end to end.
- **No-transcript-only-state proof.** Adversarially plant a load-bearing fact only
  in the transcript and assert the externalize gate refuses to rotate.
- **Tombstone-rot / stale-baton test.** Mutate git under a baton and assert the
  reload re-verification flags `RELAY_BATON_STALE` and re-derives correctly.
- **Thrash test.** Drive the trigger to fire repeatedly and assert hysteresis +
  `RELAY_NO_PROGRESS` stop the spin.
- **Cost vs compaction.** Measure steady-state token spend and cache-hit rate of a
  relay against the same goal run with auto-compaction; the relay should show flat
  peak context and no accuracy decay.
- **Determinism.** Same durable store + same baton → same reload plan.
- **Hermetic chain smoke.** Extend the #1462 handoff-to-close smoke to a
  handoff→rotate→resume→close chain that never mocks the witness.

## The rungs

GitHub epic **#1860**. The 50 child leaves, by track — each a `worker-ready-issue`
with lane, path hints, dependencies, a done condition, and a witness:

- **A — Concept & contract:** #1861 A1, #1862 A2, #1863 A3, #1864 A4
- **B — Observe (Phase 0):** #1865 B1, #1866 B2, #1867 B3, #1868 B4, #1869 B5
- **C — Baton schema & IO (Phase 1):** #1870 C1, #1871 C2, #1872 C3, #1873 C4, #1874 C5, #1875 C6, #1876 C7
- **D — Reload re-verification (Phase 2):** #1877 D1, #1878 D2, #1879 D3
- **E — Safe-stop predicate (Phase 3):** #1880 E1, #1881 E2, #1882 E3, #1883 E4
- **F — Externalize gate (Phase 4):** #1884 F1, #1885 F2, #1886 F3, #1887 F4
- **G — Rotation policy (Phase 5):** #1888 G1, #1889 G2, #1890 G3, #1891 G4, #1892 G5, #1893 G6
- **H — Relay driver (Phase 6):** #1894 H1, #1895 H2, #1896 H3, #1897 H4, #1898 H5, #1899 H6, #1900 H7
- **I — Pointer-only carryover (Phase 7):** #1901 I1, #1902 I2
- **J — Dogfood / QA / docs (Phase 8):** #1903 J1, #1904 J2, #1905 J3, #1906 J4, #1907 J5, #1908 J6, #1909 J7, #1910 J8

The rollup manifest with per-rung lane, paths, and dependencies is
`docs/milestones/perpetual-sessions-epic-tickets-2026-07-01.json`.

## Success criteria

The relay is working only if all hold:

- a goal runs across many legs with **peak context flat** and no measured accuracy
  decay from context rot;
- **no load-bearing fact is ever lost** to a rotation — the externalize gate makes
  transcript-only state unrotatable;
- the baton carries **no trusted claim** — progress is re-verified from git/ledger
  at every leg start;
- rotations land **only at safe points**; a window ceiling parks, never corrupts;
- a done goal **terminates the relay**, and a stuck one **escalates** rather than
  thrashing;
- an operator can see, per relay: legs, rotation reasons, baton fidelity, and cost
  vs a compaction baseline;
- the whole thing is **cheaper and more faithful** than compaction for this use
  case, or it is not worth its complexity.

## Open questions

- **Naming.** Recommended: *relay / leg / baton / tombstone*, invariant
  *flat-context*. Alternatives considered: *rotation / rollover / continuation*
  (collides with the existing `ContinuationID`), *session cycling*, *O(1)
  sessions*. Is "relay" the right operator-facing word, or should it align tighter
  with the existing `Generation`/`Recontinue` vocabulary as "generational relay"?
- **Where does policy live** — a new `[relay]` table in `dos.toml`, an `Envelope`
  extension, or generated from the existing budget axes?
- **How strict is pointer-only?** Is a *tiny* verbatim tail (last 1–2 turns) a
  pragmatic exception, or does any transcript carryover reopen the compaction door
  we are closing? (Lean: none — the baton's `next_action` replaces the tail.)
- **Done-check for open-ended goals.** A "keep X green" watch has no terminal
  `done_when`. Is the relay bounded only by an operator envelope there, and how is
  `RELAY_NO_PROGRESS` distinguished from "correctly idle, nothing to do"?
- **Relationship to `sessionreset` contributors.** Is the relay a new strict
  contributor policy, or a separate carryover path that bypasses the seed builder
  entirely?
- **Cross-host relays.** `sessionimage` is portable; does a leg boundary double as
  a host-migration boundary, and does the baton subsume the image or point at it?

## The doctrine

Compaction shrinks a pressured window by asking a degraded model what to remember.
The relay never asks that question. It ends a leg cleanly at a safe boundary, makes
the machine prove every load-bearing fact is already durable, hands the next leg a
small typed baton of *pointers* it must re-verify, and throws the transcript away.
The goal runs forever; the window stays flat; and nothing survives a rotation that
could not survive a witness.
