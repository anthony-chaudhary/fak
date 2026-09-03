---
title: "Spine-first + fan-out â€” the two defaults for new work"
description: "The two defaults that fire for every new feature, leaf, verb, demo, or process change in fak unless explicitly waived: spine-first, then fan-out."
---

# Spine-first + fan-out: the two defaults for any new unit of work

These are defaults, not ceremonies: they fire for **every** new feature, leaf,
verb, demo, or process change unless explicitly waived â€” the same way "default:
ship" and "proof by default" fire in [`AGENTS.md`](../AGENTS.md).

## Default 1 â€” the minimal working end-to-end spine ships first

In the **same session** the work starts, ship the smallest runnable path that
exercises the **real seam** end to end â€” as minimal and as *working* as possible:

- **User-facing surface** â†’ meet the LCD demo bar
  ([`docs/run-the-demos.md`](run-the-demos.md)): one command, deterministic, no
  key, no network, no GPU, with a `-selfcheck` invariant.
- **Library leaf / verb** â†’ a test that drives the real object **plus** one
  captured live run of the verb (a `--json`/`--dry-run` invocation or loopback integration run is fine; mocks hide integration bugs).
- **Process / doctrine change** â†’ the machinery that makes it a default (a
  gate, a skill, a verb), not a paragraph asking agents to remember it.

**If a working spine is not achievable this session with high confidence, the
spine itself becomes the first issue** â€” filed `gen/now`, milestoned at
creation, with the missing witness named. The spine is never silently deferred:
either it exists as a witness, or it exists as a tracked issue. `fak issue
fanout` enforces this mechanically â€” it **refuses to plan** without a
`--spine` witness.

### Shift the finish line left at issue creation

Every armed issue must make two judgments before implementation begins:

- **Core through-line** — the shortest causal path from the proposed change, through the real
  seam, to one observable user/operator outcome and its witness. If removing a step still
  produces the stated outcome and witness, that step is not core.
- **Gold-plating boundary** — name tempting work that improves completeness, elegance, breadth,
  or hypothetical future reuse but is not needed for that causal path. This is a routing
  decision, not a dismissal: file independently valuable items as follow-ons instead of silently
  expanding the current issue.

Use this counterfactual at triage: **if we omit this item, can the named user still traverse the
working spine and can the named witness still prove the outcome?** Yes means gold plating for
this issue; no means core. Safety, compatibility, and fail-closed behavior required to run the
spine are core. Broad edge matrices, abstractions for uncommitted consumers, polish without an
outcome change, and optimization without a measured bottleneck are gold plating by default.

The issue contract renders these decisions as `## Core through-line` and
`## Gold-plating boundary` (legacy `In scope` / `Out of scope` headings remain readable). The
`Done condition` must close the through-line; the `Witness` must observe its final outcome. This
keeps "core" from becoming a synonym for everything the author wants and makes scope creep
visible before dispatch. `fak-dev issue create` enforces this by default and canonicalizes legacy
headings to the decision names; `--raw-body` is the explicit escape hatch for a deliberate
non-contract administrative issue.

Why: a spine is a witness; a plan is a claim. The repo refuses unproven claims
(`not yet` discipline) â€” the spine is what converts "we will" into "it does".
It is also the cheapest moment to discover the design is wrong.

### The order of operations: applied implementation, then exhaustive proof

"Spine first" also orders work *inside* an issue. When scope permits, use this
sequence and keep the first two steps in the same session:

1. **Name the end-to-end outcome and its smallest real seam.** State what a user,
   caller, or operator can actually do when the path works.
2. **Make one representative path work end to end.** Prefer an applied vertical
   slice over broad scaffolding, a comparison matrix, or an optimized component
   that is not yet connected to the outcome.
3. **Capture the spine witness.** Record the runnable command/test and its result;
   this is the anchor that later evidence compares against.
4. **Expand proof systematically.** Add failure paths, edge cases, platforms,
   concurrency, soak, and the rest of the operating envelope. These prove where
   the working path holds; they do not substitute for establishing it.
5. **Optimize last, against the working path.** Measure the real end-to-end
   alternative, identify the limiting seam, change it, and keep only a net-true
   gain. A faster disconnected component or an exhaustive "almost there"
   comparison is not primary progress.

This is a sequencing default, not permission to skip correctness or safety.
Build the smallest path that is safe to run; if fail-closed behavior is part of
that path, it belongs in the spine. Exceptions are explicit: a prerequisite
investigation may lead only when the unknown makes an applied spine impossible,
but it must end in a checkable spine issue rather than an open-ended comparison.

## Defaults are interventions

When the spine adds or widens behavior users receive without deliberately selecting it, complete the [benefit–harm default admission record](standards/benefit-harm-defaults.md) before calling the spine shipped. Compare against doing nothing and the strongest practical alternative; name side effects, uncertainty, contraindications, minimum effective scope, operator control, surveillance, and a tested stop/rollback rule. Choose `DEFAULT`, `CONDITIONAL DEFAULT`, `OPT-IN`, or `EXCLUDE`. A measured average gain alone does not justify broad exposure, and follow-on monitoring cannot repair a missing rollback path.

## Default 2 â€” the follow-on backlog is filed at creation time (3..50+)

The moment a spine ships, fan out its hardening backlog **while context is
hot** â€” not "later", which on a shared trunk means never:

```bash
fak-dev issue fanout --title "my feature" --leaf myleaf \
    --spine <commit-sha | demo command | doc path> [--parent '#<epic>'] --json
```

The planner ([`internal/issuefanout`](../internal/issuefanout/issuefanout.go))
expands a fixed 15-template taxonomy â€” **qa** (edge sweep, failure paths,
determinism/race), **dogfood** (self-run on the repo's own work, usage ledger),
**product** (CLI reference, LCD demo, error-message UX), **observability**
(outcome counters, scorecard fold), **integration** (advisory guard gate,
dos.toml wiring, super-loop hookup), **docs** (doctrine + doc-map linkage), and
**release** (CLAIMS.md tag + note) â€” into candidates that each carry the *full*
[`issuepolicy`](../internal/issuepolicy/contract.go) scope contract
(working spine, done condition, witness, acceptance gate, closure binding,
route, step budget). Every candidate is **dispatchable the moment it is
filed**, proven by `go test ./internal/issuefanout`.

- **3 is the floor, not the target** (`issuefanout.MinFanout`); the full
  taxonomy is 15, and a large feature composes several fan-outs (one per leaf).
- File with `gh` â€” **milestone + labels at creation** (the issue-hygiene
  default) â€” or wave-plan first: `fak-dev issue fanout ... --json > plan.json &&
  fak-dev issue cohort --from-plan plan.json` gives concurrency-safe, leased waves.
- Dedupe before filing: the candidates carry `fanout-<leaf>-*` marker keys, so
  a rerun is detectable against existing issues.

## How this binds into the machinery (shipped vs planned)

| Seam | State |
|---|---|
| `fak-dev issue fanout` planner + verb (`internal/issuefanout`) | **shipped** `5b8f0bd1` |
| Spine-first as an *issue* property (`issuecontract.WorkingSpine` required + spine-priority scoring) | shipped (pre-existing) |
| Wave planning over the fan-out (`fak-dev issue cohort --from-plan`) | shipped (pre-existing) |
| `/spine-fanout` skill (agent front door) | **shipped** (`.claude/skills/spine-fanout/`) |
| Advisory pre-commit nudge (PRIOR_ART pattern, `internal/hooks`) when a new leaf ships spine-less | planned â€” #2521 |
| `dos.toml` lane for `issuefanout` + advisory reason token | **shipped** â€” #2522 |
| Super-loop / dispatch hookup (fan-out fires automatically at spine-ship) | **shipped** â€” #2523 |
| `fak new-leaf` scaffold prompting the spine + fan-out | planned â€” #2530 |
| Adoption scorecard (spines shipped vs fan-outs filed) | **shipped** â€” #2532 |

This doctrine was dogfooded on itself: the planner above **is** the minimal
working spine of the concept, its own fan-out was generated by the verb, and
the "planned" rows above were filed from that output as **epic #2510**'s
children (#2511â€“#2532). Each planned row is dispatchable by its issue number;
pick one up with `fak-dev issue contract --issue <n>`.
