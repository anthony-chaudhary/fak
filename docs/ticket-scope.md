---
title: "The scope of a GitHub ticket"
description: "What the scope of one GitHub issue is — the six axes that decide whether a ticket is a single dispatchable unit of agent work (structure, size on the S0–S4 ladder, atomicity, path/write scope, cohort placement, work-class), each mapped to the fak tool that measures it, the closed reason it fails with, and the fix. The front door for the ticket-scope toolkit and the /ticket-scope skill."
---

# The scope of a GitHub ticket

In a fleet that dispatches issues to headless workers, **an issue is a unit of
work, and its scope is the contract for that unit** — what one worker is expected
to change, prove, and finish in one session, and what it must *not* touch. Scope
is the difference between a ticket a worker resolves in one clean pass and one
that strands the worker: a two-feature ticket that no single commit can witness,
an epic that should have been ten leaves, a ticket whose write boundary overlaps
another in-flight worker's lease.

This page is the single front door for that concept. The tooling that measures
ticket scope already exists — it was just scattered across a smallness linter, a
contract validator, a cohort planner, and section 1 of the
[dispatch runbook](agentic-issue-dispatch.md). Here it is one toolkit with one
definition.

> **A well-scoped ticket is a *leaf*: one deliverable, one done-condition, one
> witness, a bounded write-scope, routed to exactly one lane.** Anything larger
> is an epic to decompose before dispatch; anything under-specified is triage
> debt to complete before dispatch.

## The six axes of scope

Scope is not one number — a ticket can be perfectly small yet unroutable, or
perfectly routed yet secretly two features. There are six independent axes, each
with its own measuring tool and its own closed failure reason.

| # | Axis | What it asks | Tool | Fails with |
|---|------|--------------|------|-----------|
| 1 | **Structure** | Does the body carry the sections a worker needs to act? | `fak issue contract` | `ISSUE_SCOPE_INCOMPLETE` (`missing_sections`) |
| 2 | **Size** | Is it a leaf, or a feature/epic that must decompose first? | `fak issue contract` (S0–S4 scale readout) | `ISSUE_NOT_DISPATCH_LEAF`, `ISSUE_OVERSIZED_EXPECTED_STEPS` |
| 3 | **Atomicity** | Is it *one* deliverable with *exactly one* witness? | `fak dispatch issue-smallness-lint` | smallness `fail` (≥3 deliverables, or witness ≠ 1) |
| 4 | **Write-scope** | Is the file/path boundary bounded and collision-free? | `dos arbitrate` · `fak orient` | lane collision / unknown expected paths |
| 5 | **Cohort** | Where does it sit in a batch — a wave, or the split-first queue? | `fak issue cohort` | pulled into the split-first queue |
| 6 | **Work-class** | Is it plumbing, a product leaf, or the public front door? | [work-class axis](work-class-axis.md) · `fak orient` | routed to a fenced/private lane |

An issue is **dispatchable** only when all six agree: complete structure, leaf
size, atomic, bounded and free write-scope, wave-placeable, and routed to a lane
a worker may take.

---

### 1. Structure — the five required sections

A worker-ready body must let a headless worker act without rediscovering the
assignment. The [dispatch runbook](agentic-issue-dispatch.md#required-worker-body-sections)
requires these five sections (this page and that one are the same contract):

- **`Current state`** — what is already true and what gap remains.
- **`Scope`** — either one `Scope` section, or both `In scope` and `Out of scope`.
- **`Done condition`** — the observable end state for *this one* issue.
- **`Witness`** — the focused command, captured artifact, or read-back that
  proves done.
- **`Likely files`** — concrete path hints (aliases: `Path hints`, `Paths`,
  `Files`).

```sh
fak issue contract --from-issues issues.json --json
```

Each review row reports `missing_sections` for these five and `missing_fields`
for the stricter issue-contract fields. A row missing any required section is
triage debt, not a launch candidate — it fails **exit 3** with
`ISSUE_SCOPE_INCOMPLETE`. The overlay backfills body sections only; a scope
reason that comes from a *label* (e.g. `ISSUE_NOT_DISPATCH_LEAF` set by an
`epic` label) needs an operator label edit, not a body rewrite.

### 2. Size — the S0–S4 work-size ladder

Structure can be complete and the ticket still be too *big*. Every contract
review also reports a **scale readout** on the S0–S4 ladder:

| Rung | Size | Dispatchable? |
|------|------|---------------|
| **S0** | step | yes |
| **S1** | leaf | yes — the target |
| **S2** | feature | **no** — decompose first |
| **S3** | epic | **no** — decompose first |
| **S4** | program | **no** — decompose first |

The readout carries the *declared* size, the size *derived* from the step budget
and work-unit shape, the *effective* size, and whether the witness KIND matches
the scale (a feature/epic "done" witnessed only by a single commit or test flags
a scale mismatch). **A feature-or-larger scale (S2+) is always held off dispatch
as `ISSUE_NOT_DISPATCH_LEAF` — it must decompose into leaves first.** An oversized
step budget flags `ISSUE_OVERSIZED_EXPECTED_STEPS`.

`--strict-scale` turns an *undeclared* size or an under-scale witness triage-only
(`ISSUE_SCALE_UNDECLARED` / `ISSUE_WITNESS_SCALE_MISMATCH`) instead of leaving it
advisory. Decomposition is the [`dos-replan`](../.claude/skills/dos-replan/SKILL.md)
and [`study-repo`](../.claude/skills/study-repo/SKILL.md) job: an epic becomes an
epic issue plus N leaf issues, never one "adopt everything" monolith.

#### Capability-aware sizing: smaller models and scoped fail-to-abstain

Model capability directly bounds dispatchable ticket size. While frontier models
can reason across moderately complex multi-file leaves, **smaller models (such as
local 7B/14B models, fast/flash models, or bounded worker subagents) require tight
S0/S1 leaf boundaries, scoped fail-to-abstain discipline, and persistence through recoverable blockers**:

- **Subdivide into focused leaves**: Smaller models must receive packets restricted
  to single-concern changes, touching at most 1–3 files within a single package or
  lane, with exactly one deterministic witness. Decompose compound work into
  sequential steps: establish a reproduction test first, commit the minimal fix,
  and verify the package.
- **Scoped fail-to-abstain on high-difficulty aspects**: When a task involves subtle
  concurrency invariants, lock ordering, frozen ABI modifications (`internal/abi`),
  SIMD/CUDA kernel mechanics, or security policy gates that exceed reliable reasoning,
  scope abstention strictly to the hard boundary. Land partial verified evidence for
  safe, solvable sub-components (such as reproduction tests or doc stubs) and emit a
  structured `ABSTAIN` verdict for the isolated aspect rather than guessing, emitting
  speculative diffs, or abandoning the entire prompt.
- **Persist through recoverable blockers**: A kernel refusal or environment hurdle
  (e.g., `POLICY_BLOCK`, `FILE_ADMISSION`, `MERGE_IN_PROGRESS`) is runtime feedback, not
  a session stop. Query `fak recover <TOKEN>` for structured recovery, wait out transient
  concurrency locks, and pivot to sanctioned alternate routes (e.g., Go sub-modules or
  structured file tools) while preserving safety invariants.
- **Emit structured evidence**: Record an explicit `ABSTAIN` verdict with a typed
  refusal token or boundary description alongside landed partial witnesses, cleanly
  escalating only the isolated out-of-depth aspect.

### 3. Atomicity — the smallness lint

Size asks "how big is the work"; atomicity asks "is it really *one* thing". The
smallness lint (`internal/issuesmallness`, schema `fak-issue-smallness-lint/1`)
counts distinct deliverables and witnesses:

```sh
fak dispatch issue-smallness-lint --issue 1234       # one live issue
fak dispatch issue-smallness-lint --body-file draft.md   # a local draft (or - for stdin)
fak dispatch issue-smallness-lint --open --json      # the whole open backlog, as a report
```

The verdict is mechanical, not a model judgment:

- **≥ 3 distinct deliverables** in the `Goal` / `Done condition` section → **`fail`**:
  split before dispatch.
- **2 deliverables** → **`warn`**: confirm they are genuinely one unit of work.
- **≤ 1 deliverable** → **`pass`** on the deliverable axis.
- **Witness count ≠ 1** → **`fail`**: exactly one witness is required. A ticket
  with no witness cannot be proven done; a ticket with three witnesses is three
  tickets.

`--open` folds the backlog into a pass/warn/fail report; `--open --scorecard`
emits a control-pane payload for `fak scoreboard post`. Exit 1 on any `fail`.

### 4. Write-scope — the path boundary and collision check

A leaf-sized, atomic ticket can still be **unsafe to dispatch** if its write
boundary is unknown or overlaps another worker. The write-scope axis is the file
tree the worker is licensed to touch:

- **Derive** the expected paths (`fak orient <globs>` resolves a path glob to its
  lane, arch tier, owning tests, ship-stamp, and any live lease — the same
  read every worker should do before editing).
- **Arbitrate** same-wave safety with `dos arbitrate`: two workers may run
  concurrently only when their trees are disjoint (shared/shared may overlap;
  any exclusive holder must be tree-disjoint). An issue with unknown expected
  paths or a private/exclusive lane is held for triage, not dispatched.

The write-scope is also what the ship-stamp names: the `(fak <leaf>)` trailer
must match the code lane the paths touch, not the issue's routing lane.

### 5. Cohort — placement in a batch

At creation time you rarely scope one ticket in isolation — you scope a *batch*.
`fak issue cohort` plans a whole cohort (1..1000 candidates):

```sh
fak issue cohort --from-issues issues.json --json --max-wave 8
```

It partitions the dispatchable leaves into concurrency-safe **waves** (the same
disjoint-tree rule `dos arbitrate` uses), pulls oversized/non-leaf rows into a
**split-first queue** with a child-issue budget, buckets the rest into **triage**,
and reports duplicate marker keys. It is a *planner, not a gate* — exit 0 on a
valid plan, exit 2 on bad input. A ticket that lands in the split-first queue is
telling you its size axis (2) failed at batch scale. Retrospective duplicate
scope — "is this ticket a re-file of an open one?" — is `fak issue dedup`.

### 6. Work-class — which kind of work, which front door

The last axis is orthogonal to size: *what kind* of work is this? The
[work-class axis](work-class-axis.md) derives one of three classes from the lane
the issue routes to — **`class:infra`** (fleet plumbing), **`class:dev`** (product
leaves, the clean default dispatch surface), **`class:frontdoor`** (the public
release path — review-heavier, the fenced bucket). Scope-wise this decides
*which surface* a ticket belongs on and how heavily it is reviewed: a front-door
ticket is in scope for release review even when it is a perfectly small leaf.

---

## The one-pass scope check

To scope-check a single ticket end to end, two verbs cover all six axes (the
[`/ticket-scope`](../.claude/skills/ticket-scope/SKILL.md) skill runs exactly this
and reads back a single verdict):

```sh
# axes 1, 2, 4, 6 — structure, size, routing, class
fak issue contract --from-issues one-issue.json --json

# axis 3 — atomicity / smallness
fak dispatch issue-smallness-lint --issue <N> --json
```

For a batch, add `fak issue cohort --from-issues <batch>.json --json` for axis 5.

## How the axes compose into a verdict

```
             ┌─ structure complete? ────── no → triage (add sections)
             ├─ leaf size (S0/S1)? ──────── no → decompose (dos-replan)
 dispatch ⟵ ─┼─ atomic (1 deliverable,
             │   1 witness)? ───────────── no → split (smallness fail)
             ├─ write-scope bounded &
             │   collision-free? ────────── no → triage (derive paths / wait on lease)
             └─ lane a worker may take? ─── no → route / fence (work-class)
```

Every "no" is a *typed* outcome with a fix, not a vague "needs work": a missing
section is backfillable, an S2+ scale decomposes, a smallness fail splits, an
unknown path derives, a fenced lane routes. Scope debt is always actionable.

## See also

- [Agentic issue dispatch](agentic-issue-dispatch.md) — the full runbook that
  consumes a well-scoped ticket: select → derive paths → arbitrate → launch →
  witness → close. Section 1 is the structure axis; this page is all six.
- [Issue-dispatch loop](dispatch-loop.md) — the witness-gated backlog driver.
- [Work-class axis](work-class-axis.md) — axis 6 in full.
- [`/ticket-scope`](../.claude/skills/ticket-scope/SKILL.md) — the repeatable
  skill that runs the one-pass check and proposes the fix.
- [`/issue-triage`](../.claude/skills/issue-triage/SKILL.md) — the backlog
  gardening pass (labels, staleness, dedup) that scopes the *set*, where this
  scopes the *ticket*.
- [`/dos-replan`](../.claude/skills/dos-replan/SKILL.md) — decompose an S2+ epic
  into dispatchable leaves.
