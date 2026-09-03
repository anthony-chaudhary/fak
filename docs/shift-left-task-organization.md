---
title: "Shift-left task organization"
description: "The human-readable contract for turning a new outcome into dispatchable work before a worker starts."
---

# Shift-left task organization

Use this page when new work is being proposed, split, filed, or handed to an agent. The rule is simple:

> **Move every decision that is already knowable to the task-creation boundary.** A worker should discover the solution, not rediscover the task's value, intent, ownership, dependencies, done condition, or proof.

The Feynman test is the first gate: explain the work in words a new operator can repeat accurately. Name **who has what problem, how they handle it today, and why the smallest end-to-end change should be better**. Complexity that cannot yet be explained is uncertainty to resolve, not detail to hide behind. “Better” means a witnessed user outcome against the real next-best alternative—not more machinery, more files, or a larger scorecard.

First use the canonical [problems fak exists to solve](problems-we-solve.md) in two ways:
classify the work's **problem centrality** (`Core`, `Enabling`, `Stewardship`, or
`Peripheral`), then answer **all four** P1-P4 checks for context, net value, bounded
adaptation, and integrated operations. Centrality is a portfolio signal; the P rows are
all-work design and review checks, not competing priority labels. The value frame still has
to name the concrete pain.

This is an authoring and reading contract over the repository's existing sources of truth. It does not replace GitHub Issues, DOS plans, [`fak issue contract`](agentic-issue-dispatch.md), lane leases, or the [`shared-task/1`](shared-task-record-contract.md) runtime record.

## The 60-second operator view

Read work at four levels. Do not flatten them into one checklist.

| Level | Human question | Durable record | Done when |
|---|---|---|---|
| **Outcome** | What operator-visible result are we pursuing? | epic, plan, or milestone | its stated effect is witnessed |
| **Leaf** | What coherent, independently closable change advances it? | one issue and one eventual commit | acceptance and witness both pass |
| **Attempt** | Who is acting now, where, and under which lease? | lane lease plus run/task record | it ships, yields, or records a typed stop |
| **Witness** | What evidence can another reader independently check? | test, captured render, effect read-back, or witnessed commit | the claimed effect is corroborated |

A human queue should lead with **outcome and leaf**. PID, account, token, and heartbeat details belong in the attempt drill-down, not in the primary task title.

## Create the task before scheduling it

At creation time, fill every field that is knowable. Use `unknown(<reason>)` only when the answer genuinely depends on investigation; never use omission to mean unknown.

```markdown
## Value
- For: <the person or operator who benefits>
- Problem: <observable pain or unmet need>
- Today: <the real next-best way they handle it now>
- Better because: <plain-language reason the smallest spine should win>
- Witness: <artifact that would prove the operator burden changed>
- Centrality: <Core | Enabling(named Core outcome) | Stewardship(obligation) | Peripheral>
- P1 Context: <advanced | preserved | N/A — reason>
- P2 Net value: <advanced | preserved | N/A — reason>
- P3 Adaptation: <advanced | preserved | N/A — reason>
- P4 Operations: <advanced | preserved | N/A — reason>

## Outcome
<one sentence describing what becomes observably better for that person>

## Scope
- Owns: <lane/tree/API or named surfaces>
- Excludes: <explicit non-goals>

## Dependencies
- Requires: <issue, phase, decision, or `none`>
- Unblocks: <issue, phase, or `unknown(reason)`>

## Acceptance
- [ ] <observable effect, not an implementation step>
- [ ] <compatibility or readability invariant>

## Witness
- <test/render/effect read-back/commit verification that proves acceptance>

## Placement
- Parent: #<epic/outcome>
- Class / priority / generation: <typed labels>
- Milestone: <delivery horizon>
- Suggested lane: <tree ownership, not a worker name>
```

Then run the repository's contract and placement surfaces rather than asking the worker to infer them later:

```text
fak issue contract <N> --json
fak issue graph --issue <N> --json
fak issue cohort --from-plan <plan>
fak issue fanout --title <T> --leaf <L> --spine <sha|cmd|doc> --json
```

The exact available verbs evolve; `fak help --all` is authoritative. The invariant is stable: **classify and validate before dispatch**.

### Machine readiness read-back

`fak issue contract --file CANDIDATE.json --json` and `--from-issues ISSUES.json
--json` identify schema `fak-task-brief-readiness/1`. Each review's
`brief_readiness.fields` object reports `beneficiary`, `problem`, `alternative`,
`advantage`, `outcome`, `scope`, `dependencies`, `acceptance`, `witness`, and
`placement` as exactly one of:

- `present`;
- `unknown` with a non-empty `reason`; or
- `missing` with a concrete `repair_action`.

For the copyable brief above, any `missing` field makes the leaf non-ready and the
review verdict `needs_brief`. A `Value` heading opts into all four value checks; this
keeps legacy issues readable while making newly value-framed work mechanically honest.
Existing issue contracts that predate this vocabulary
remain compatible (`brief_readiness.enforced=false`) until migrated; once a brief uses
`Scope / tree`, `Witness / proof`, or `Placement`, all six fields are enforced.

## Shift-left checklist for new work

1. **Explain the value simply.** Name the beneficiary, observable problem, current next-best alternative, and why the smallest spine should be better. If a new operator cannot repeat it accurately, keep scoping.
2. **Name the outcome.** Prefer a user/operator effect over a component noun; “build X” is an implementation, not an outcome.
3. **Find the parent and duplicates.** Search open and closed issues before creating another source of truth.
4. **Ship or name the spine.** Follow [`spine-first-defaults.md`](spine-first-defaults.md); a broad fan-out without an end-to-end witness is planning debt.
5. **Cut coherent leaves.** One issue should be independently closable and should map to one eventual commit/leaf. Split by acceptance boundary, not by file count.
6. **Declare ownership and collisions.** Name the lane/tree before a worker starts. Scheduling may choose the worker; it must not invent scope.
7. **Declare dependencies.** Use explicit `requires`/`unblocks` edges. Parallel-looking work with a hidden prerequisite is serial work described badly.
8. **Write acceptance before implementation.** Checkboxes describe observable effects. Implementation notes may suggest a path but cannot redefine done.
9. **Choose the witness.** Visual work needs a captured render; behavior needs a before/after repro; runtime/CLI/protocol work needs meaningful execution (dogfood or integration test run, not mock-only); shipped claims need independent commit/effect verification.
10. **Type uncertainty.** File a decision or investigation leaf when an unknown blocks a contract. Do not bury the unknown in a worker prompt.
11. **Place the work.** Apply class, priority, generation, milestone, and parent at creation or update time.
12. **Dispatch only ready leaves.** A worker receives the issue contract plus current lease/attempt state, not an improvised prose brief.
13. **File discovered follow-ups.** New work becomes a deduplicated issue with a done condition before the run ends; otherwise it does not leave the run.

## Right proof at the right stage

Shift left does **not** mean putting the release gate before the first runnable path. It means answering each question at the cheapest stage where the answer is reliable:

| Stage | Decide now | Minimum faithful witness | Do not lead with |
|---|---|---|---|
| **Proposal** | beneficiary, problem, current alternative, expected advantage, explicit unknowns | a plain-language value frame plus a checked existing-work search | architecture, exhaustive matrices, or unmeasured gain claims |
| **Spine** | smallest safe end-to-end path that reaches the user outcome | one captured live path or real-object test | broad fan-out while the primary outcome is still “almost there” |
| **Hardening** | real failure modes and operating envelope exposed by the spine | focused boundary, failure, compatibility, and render witnesses | hypothetical completeness unrelated to the working path |
| **Optimization** | whether the change beats the tuned next-best alternative net of its costs | reproducible A/B evidence under the same quality and workload constraints | naive baselines or proxy metrics presented as user value |
| **Release** | supportability, discoverability, rollback, and outcome retention | clean committed-tip gates plus operator read-back | shipping because components exist while the end-to-end effect is unwitnessed |

At every stage ask: **What is the simplest accurate explanation? What is the cheapest witness that could disprove it now? What later proof is intentionally not due yet?** This preserves quality without putting hardening or optimization before value and spine.

## Readable status, typed status

Use a small durable lifecycle for the leaf and keep execution state separate:

| Leaf state | Meaning | Next human action |
|---|---|---|
| `DRAFT` | contract, dependency, or decision is incomplete | finish authoring; do not dispatch |
| `READY` | scope, acceptance, witness, placement, and prerequisites are explicit | schedule on its lane |
| `ACTIVE` | at least one current attempt holds the required lease | inspect attempt drill-down only if needed |
| `HELD` | a typed dependency, decision, soak, or policy gate prevents pickup | perform the named unblock action |
| `DONE` | acceptance is independently witnessed | close and update the parent/outcome |

`blocked`, `waiting`, and `stalled` are not interchangeable. Record a typed reason and the next action. The task manager's runtime states remain authoritative for an individual attempt; see [`task-manager.md`](task-manager.md).

## Human-readable queue rendering

The primary view should fit one line per leaf:

```text
#6418 READY  P1/gen-now  docs  Shift-left task organization
  outcome: operators can understand and dispatch new work without reconstructing intent
  requires: none  witness: docs links + path-scoped validation  parent: portfolio hygiene
```

Expand only on demand to show attempts, account/profile, lease timestamps, heartbeats, or raw payloads. This preserves machine detail without making humans parse scheduler telemetry to learn what the work is.

## Existing contracts this page composes

- [`spine-first-defaults.md`](spine-first-defaults.md): prove an end-to-end path before broad fan-out.
- [`agentic-issue-dispatch.md`](agentic-issue-dispatch.md): turn issue labels and contracts into dispatch fuel.
- [`shared-task-record-contract.md`](shared-task-record-contract.md): carry one task across runtimes without losing identity or stop state.
- [`multi-agent-coordination-protocol.md`](multi-agent-coordination-protocol.md): arbitrate lanes and reconcile concurrent attempts.
- [`task-manager.md`](task-manager.md): manage live execution, heartbeats, budgets, and typed attempt outcomes.

The shift-left test is: **could a new operator explain why this leaf exists, whether it is ready, and what proves it done without reading a worker transcript?** If not, improve the task record before adding capacity.
