---
title: "One map for continual background work: dispatch, superloop, dos loop, and the rest"
description: "Inventories every mechanism fak has for getting continual background work done, organizes them by role (timer / selector / executor / witness), names the name-collisions and overlaps, and proposes how to combine, align, or disambiguate them."
date: 2026-07-02
---

# Continual background work: the loop-family map

Status: concept note — an inventory plus proposals. Nothing here changes runtime
behavior. The conceptual taxonomies already exist
([engineering-is-building-loops](../explainers/engineering-is-building-loops.md)
gives the ring altitudes, [super-loops](../super-loops.md) gives the intent layer
above them); what does **not** exist is one page that says which *operational*
mechanism to reach for, how the mechanisms relate, and which names collide. This
note is that page's seed.

## The problem

The repo has grown at least **six families** of "keep work happening in the
background" machinery, built at different times for different work surfaces:

- the **issue-dispatch loop** (scheduled tasks + `fak dispatch tick/wave`),
- the **`/super-loop`** bulk detached wave launcher (plus `/wave-harvest`),
- the **DOS loop family** (`/dos-dispatch-loop` → `/dos-dispatch` →
  `/dos-next-up` / `/dos-replan`, and the `dos loop --enact` supervisor),
- the **`fak loop`** substrate (`run`/`drive`/`admit`/`status` + the
  `.fak/loops.jsonl` ledger),
- the **selector meta-loops** (`fak superloop`, `fak bench-loop`,
  `fak nightrun next`, the bottleneck map),
- the **self-maintenance loops** (RSI/dojo/guard-RSI keep-or-revert, garden,
  cadence, the scorecard ratchets) and a bench of **janitor watchdogs**.

Each is individually documented and individually sound. Together they exhibit
three failure classes this note names and addresses:

1. **Name collisions** — the same word means two different mechanisms
   ("superloop" is the worst offender; see the table below).
2. **Role duplication** — two mechanisms can hold the same role over the same
   resource pool (two spawn authorities on one seat pool = the double-dispatch
   anti-pattern `/super-loop` warns about in prose).
3. **No closed index** — the health surfaces disagree about what the set of
   loops even *is*: `internal/loopfleet` folds 5 ledgers, the durable registry
   (`tools/loop-registry.json`) registers 2 jobs, while ~18 scheduled tasks
   actually fire on a fleet host. A loop that no fold can see is a dark loop by
   construction.

## The inventory

Grouped by the role each mechanism plays (roles defined in the next section).
Cadences are the documented defaults; a host's installed cadence can drift, and
the status card — not the doc — is the witness for what is actually firing.

### Work-getters (the fleet ring: spawn workers, ship commits, close issues)

| Mechanism | Work surface | Driven by | Stops on | Witness |
|---|---|---|---|---|
| [Issue-dispatch loop](../dispatch-loop.md) (`fak dispatch tick/wave/progress` + 3 scheduled tasks) | GitHub issues, routed to `dos.toml` lanes | OS scheduler (spawn ~10 min, close ~15 min, doc ~30 min) | standing; cap-bounded (`SPAWN_OK` preflight) | per-SHA `dos commit-audit`; deterministic close arm |
| `/super-loop` skill (bulk wave launcher) | ranked ready leaves (live triage query) | operator turn; MARATHON = a cadence of waves | plan/approval gates; stop signals (flat throughput, drained, capped) | none of its own — "a launch is not a ship"; defers to `dos commit-audit` |
| `/wave-harvest` skill | the wave's durable markers | operator turn after a wave | one reconciliation pass | `dos commit-audit` / `dos verify` per claimed leaf |
| `/dos-dispatch-loop` skill (+ `/dos-dispatch`, `/dos-next-up`, `/dos-replan` phases) | **plan units** (`PLAN-*.md` portfolio — empty in this repo: `PLAN_SURFACE_EMPTY`) | in-session, one lane lease | 8 typed kernel verdicts (`loop_decide`: drained-twice, pick-cooldown, not-ratcheting, …) | `dos reconcile` / `dos verify`; outer ratchet via `dos improve` |
| `dos loop --enact` supervisor | plan/lane work via the DOS kernel | 5-min watchdog task keeps it alive | supervisor target | DOS ledger + liveness |
| `fak loop drive` | **one** `GOAL.md` goal-spec | manual or scheduler; re-reads the goal fresh each turn | `witnessed_done` or budget spent | per-turn DOS exit-gate (`internal/loopgate`) |

### Selectors (decide *what next*; read-only by construction)

| Mechanism | Selects over | Output |
|---|---|---|
| `fak superloop walk` | member loops / scorecards / gardens under an operator intent | worst-first worklist with enter hints |
| `fak bench-loop next/walk` | benchmark registry + run catalog + nightrun ledger | the single next benchmark action |
| `fak nightrun next/plan` | feasible-here data-collection backlog | `{task, run command, acceptance}` |
| [Bottleneck map](../bottleneck-map-loop.md) | fleet capacity vs backlog evidence | the current limiter + first-true next process |
| `issue_triage.py` / `fak dispatch route` | open issues | ranked, lane-routed queue |
| [Idea-scout](../idea-scout.md) (daily) · `fak maturity route` | arXiv/GitHub · maturity-ladder gaps | **new** backlog items (intake feeders) |

### Self-improvement and maintenance executors

| Mechanism | Cadence | Keep discipline |
|---|---|---|
| [RSI loop](../rsi-loop.md) (`rsiloop`/`rsicycle`), dojo-RSI, guard-verdict RSI, docfresh-RSI | manual + CI `track` | non-forgeable keep-bit (`shipgate.Evaluate`); breaker → ESCALATE |
| `fak garden` / `garden tick` | daily CI + on push | idempotent remediation on ACTION/RED members only |
| `fak cadence` | weekly workflow | append-only `standing_score` ledger |
| Scorecard control pane | per run / CI ratchet | debt re-derived from disk; `--check` vs pinned baseline |
| `fak nightrun run --apply --loop` | overnight, operator-armed | OBSERVED ledger rows; never fabricates a number |

### Substrate and health (record, gate, and fold the loops themselves)

| Mechanism | Role |
|---|---|
| `fak loop run` / `admit` / `status` / `health` / `recover` + `.fak/loops.jsonl` | the durable, hash-chained ledger + admission governor every timed loop should route through |
| `internal/loopfleet` (`cmd/loophealth`) | cross-ledger live/stale/dark fold — today: loopmgr, nightrun, dojo, cadence, dispatch; **not yet**: rsiloop, guardrsi |
| `fak loop-score` | durability scorecard: are firing loops registered, non-dark, outcome-recorded, guard-wrapped |
| `tools/loop-registry.json` | the durable job registry (2 jobs registered today) |

### Janitors (keep the loops and the host alive; not work loops themselves)

Control-pane tick (5 min), supervisor watchdog (5 min), DOS-dispatch watchdog
(5 min), resume watchdog (10 min), session checkpoint (20 min), proc-resource
guard / runaway reaper, stale-work watchdog (6 h), worktree doctor (daily + 4 h),
dispatch-log audit (daily), Slack status/beat (30 min / 3 h), bench-plan doc
(12 h). All installed by `tools/register_*.ps1`; only some route through
`fak loop run`.

## The name-collision table (disambiguate on sight)

| Name | Thing 1 | Thing 2 | Risk |
|---|---|---|---|
| **superloop / super-loop** | `fak superloop` — read-only **intent walker**; interior node, mutates nothing | `/super-loop` skill — **bulk detached wave launcher**; spawns fleets | Same name, *opposite* risk profiles. "Run a super loop" is ambiguous between a safe orientation pass and a fleet spawn. |
| **loop** | `fak loop` — a **ledger + governor**, not a loop | `fak loop drive` — an actual (Ralph) loop; also "loop" the generic ring | "Is the loop on?" has no referent. |
| **dos loop** | `dos loop --enact` — the DOS supervisor process | `/dos-dispatch-loop` — the in-session skill cadence | Two different lifetimes and owners. |
| **bench-loop** | `fak bench-loop` — the benchmark control surface | `internal/loopgate/loopbench.go` — the verified-vs-naive exit-gate micro-bench | Unrelated code paths. |
| **loop index** | `fak loop-index-scorecard` — Orient→…→Learn **stage** coverage | an index **of** loops (does not exist) | The missing artifact shares a name with an existing, different one. |
| **dispatch** | `fak dispatch` — the issue loop | `/dos-dispatch` — the generic single-lane plan skill | Different work surfaces (issues vs plan units). |

## The organizing model: four roles, one rule

Every mechanism above plays one of four roles:

1. **Timer** — owns the cadence: an OS scheduled task, a CI cron, an operator
   turn, or a self-paced wakeup.
2. **Selector** — owns *what next*: a read-only ranking over some work surface.
3. **Executor** — does **one unit** and exits: a detached worker, a skill turn,
   a CLI verb (`nightrun run --apply`, `garden tick`, one RSI iteration).
4. **Witness / closer** — decides the unit actually landed, from evidence the
   executor cannot forge: `dos commit-audit`, `dos verify`, `shipgate.Evaluate`,
   the deterministic close arm, `/wave-harvest`.

The rule that keeps the families composable: **each mechanism plays exactly one
role, and no two mechanisms hold the same role over the same resource pool.**
The repo has already made this decision once, explicitly: dispatch-loop.md #419
("the dispatch⇄replan cadence is a supervisor concern, not a worker concern" —
worker single-shot, timer owned by the supervisor). The failure modes we
observe are exactly violations of this rule:

- **Two timers, one executor pool** — hand-launching waves while the dispatch
  cron is live (the double-dispatch anti-pattern; today it is prevented by a
  prose warning in `/super-loop` and operator memory, not by structure).
- **A launcher named like a selector** — the superloop collision puts a
  fleet-spawning executor-launcher and a read-only selector under one name.
- **An executor that grows its own timer** — a worker running a private
  multi-iteration loop instead of exiting; #419 fixed this for the opencode
  backend, and `loop_decide`'s typed stops bound it for the Claude backend.

Two axes complete the map, both already documented but not carried on the
mechanisms themselves:

- **Ring altitude** (engineering-is-building-loops): tool-call → turn →
  session → fleet → RSI → intent. `fak loop drive` is a turn/session-ring
  driver; dispatch is the fleet ring; RSI/dojo are the improvement ring;
  `fak superloop` is the intent ring.
- **Work surface**: issues | plan units | data/benchmarks | quality debt |
  repo hygiene | *the loops themselves*. Note the plan-unit surface is empty in
  this repo, which makes the whole `/dos-dispatch-loop` family **dormant here**
  (the gap dispatch-loop.md exists to close) — a fact currently discoverable
  only by reading two docs against each other.

## Proposals (combine, align, disambiguate)

Ordered by value over cost; each carries its cheapest checkable next step.

**R1 — Kill the superloop collision (disambiguate).** Rename the `/super-loop`
skill to `/wave-launch`, pairing it naturally with `/wave-harvest` (launch half /
close half). Skills are prompt files, so the rename is cheap; `fak superloop`
(code + docs + registry) keeps the name. Until the rename lands, both surfaces
should carry a two-line "not the other superloop" stanza. *Next step: rename the
skill directory, update the description trigger phrases, and grep the docs for
`/super-loop` references.*

**R2 — One spawn authority (align).** Every worker launch — the dispatch cron
tick, a wave, `dos loop --enact`, a hand launch — should pass the **same**
admission verb before spawning. The native preflight inside `fak dispatch tick`
(host guard ∧ seat route ∧ cap) is that authority; exposing it as a standalone
`fak dispatch admit` (exit 0/3, same shape as `fak loop admit`) lets the wave
launcher and the DOS supervisor consume it instead of their parallel Python
preflights. This converts the double-dispatch rule from operator memory into
structure. *Next step: extract the `internal/dispatchtick` preflight evaluation
behind one verb and point `launch_goal_detached.ps1` at it.*

**R3 — One loop registry, closed (combine).** Extend `tools/loop-registry.json`
(or a `dos.toml`-adjacent table) so **every** standing loop — scheduled task,
CI cron, skill cadence — is declared in one data file, with `ring:`,
`surface:`, `role:`, and `ledger:` fields. Have each `tools/register_*.ps1`
installer write its row through a `fak loop register` verb at install time. Then
add the no-drift test the superloop registry already models: every registry row
must be foldable by `loopfleet` (or explicitly marked ledger-less), and every
ledger `loopfleet` folds must have a registry row. This makes `fak loop-score`'s
"registered" check meaningful (today: 0/5) and gives the fleet a *closed* answer
to "what loops exist?". *Next step: wire the two missing `loopfleet` adapters
(rsiloop, guardrsi) — the smallest concrete drift already named in code.*

**R4 — One selector contract (align).** `fak nightrun next` has the
best-formed selector output: `{id, run command, requires, acceptance}` — work to
do, never a result, so it cannot overclaim. Adopt that shape for every
selector: `superloop walk` enter-hints, `bench-loop next`, the dispatch tick
pick. Once selectors share a contract, the named-but-unbuilt rung in
super-loops.md ("actually *entering* the worst-first member") becomes one
generic driver over `{run, acceptance}` instead of per-family glue. *Next step:
define the task shape as a type in `internal/superloop` and emit it from
`walk --json`.*

**R5 — Declare the work surface (disambiguate).** Each loop's front-door doc
gets one frontmatter-adjacent line naming its surface (issues / plans / data /
debt / hygiene / loops) and its role. The dormant-here arms get labeled as such
where they live: `/dos-dispatch-loop` and `FleetIssueDispatch -Mode loop` are
plan-surface mechanisms on a plan-empty repo, "dormant until `PLAN-*.md`
ship" — dispatch-loop.md says this once; the skill itself should too. *Next
step: add the stanza to the dos-* skill fronts.*

**R6 — Promote this map (combine).** Once R1's rename settles, promote this
note to a front-door `docs/loops-map.md` — the canonical anti-conflation map
for the loop family, the role [`docs/collectives.md`](../collectives.md) plays
for the MPI family — and link it from `llms.txt` and WORK-MAP.md's ongoing-work
section. *Next step: this note; then the promotion is a move + index refresh.*

## Honest scope

This note is an inventory and a set of proposals; none of R1–R6 is shipped by
it. The inventory was read from the docs and code cited inline on 2026-07-02;
cadences and member sets drift, and the live status surfaces (`fak loop
status`, `loophealth`, the dispatch status card) outrank this page wherever
they disagree. The four-role model is descriptive, not enforced — R2 and R3
are what would make two of its clauses structural.
