---
name: fleet-wave
description: 'Run ONE wave of N fak-guarded ultracode sessions against the top open issues under a closing target and a wall-clock deadline — price, render fuel, launch, monitor, reconcile from git, release. N defaults to 30 (30 issues, 30 sessions, 4 hours). Use when the operator says "spawn N ultracode sessions", "fleet wave", "close the top 30 issues in 4 hours", or asks for a bulk guarded fan-out with a stated goal. The goal-shaped single door over /super-loop (the raw launcher).'
allowed-tools: Read, Bash, Write
metadata:
  opencode: claude-only   # commit-by-explicit-path, the honesty boundary, and the seat-hygiene rules are load-bearing and not portable per-skill
---

# /fleet-wave — N guarded ultracode sessions on the top open issues

> **One call, one wave, one stated goal.** `/fleet-wave` takes a target — *close N top
> issues in H hours* — and runs the whole arc: price the fan-out against the live
> ceilings, render the ultracode fuel, launch N detached **`fak guard`ed** workers,
> watch them, reconcile what actually shipped from git, and release what they held.
>
> **Default N = 30.** `/fleet-wave` alone means *30 issues, 30 sessions, 4 hours*.
> `/fleet-wave 12` retargets it. `/fleet-wave 30 8h` retargets the deadline too.

> **Codex is the default worker backend.** Native `fak dispatch wave --backend codex` is
> the closest currently supported Ultracode-style shape: one guarded headless Codex process
> per admitted live issue and switcher seat. Do not use raw `codex exec` or share one
> interactive `CODEX_HOME` across children.

**Relation to the family — this skill does not re-implement any of it.**

| | |
|---|---|
| [`super-loop`](../super-loop/SKILL.md) | the raw bulk launcher and its regimes. `/fleet-wave` is the **goal-shaped** door over it: one wave, one closing target, one deadline. |
| [`wave-harvest`](../wave-harvest/SKILL.md) | the reconcile half. Phase 5 **calls it**; it is not copied here. |
| [`refusals.md`](refusals.md) | the worker refusal + park rules — the one canonical copy, read by every worker **by path**. |
| [`fuel-wave.md`](fuel-wave.md) | the per-wave **operator receipt** template — records target, deadline, and ownership rules; native dispatch renders bounded child fuel from each live issue. |
| [`monitor.md`](monitor.md) | the read-only watcher preamble. |
| [`RATIONALE.md`](RATIONALE.md) | the evidence behind every rule here. **No worker loads it** — keep it that way. |

---

## ⛔ The four that make this not a one-liner

Settle all four before Phase 3. Each one has silently voided a whole wave.

**1. The native `fak dispatch wave` contract is the one launch path.**
The PowerShell `launch_wave_detached.ps1` / `launch_goal_detached.ps1` pair remains a
legacy launcher for direct `/goal` pointers, but `/fleet-wave` does **not** call it. Native
dispatch owns account allocation, issue contracts, guarded worker commands, and bounded
per-issue fuel. Mixing the two paths caused stale `-PointerFile`, `-ExtendStanding`, and
`-Launch` instructions to survive after the CLI had moved on. ⛔ **Do not pass launcher-only
flags to `fak dispatch wave`, and do not substitute the legacy launcher after a native
plan refusal.** A refusal is the result to report or fix, not permission to change control
planes.

**2. `fuel-wave.md` is an operator receipt, not a `/goal` condition.**
Render it under `.fak/wave/` to preserve the wave id, deadline, target, and rules used by
the orchestrator. It may exceed the legacy launcher's 4000-character `/goal` cap because
it is never passed to that launcher. Each native dispatch tick renders its own bounded
worker fuel from the selected live GitHub issue. The regression witness in
`internal/wavefuel` refuses any skill text that routes this receipt into `-PointerFile` or
revives stale launcher flags.

**3. The N you asked for is an aspiration, and the binding term is usually NOT the cap.**
Every spawn re-checks `dispatch_preflight.py`, whose cap is
`min(FAK_MAX_WORKERS, host_cap, dos [supervise].target, seats)`. But the wave then has to
buy *distinct-account session slots* on top of that. Measured dry-run of this exact skill,
2026-08-08, `-Count 30`:

```
WAVE PLAN  requested=30  allocation_requested=12  granted=12  shortfall=18  distinct_pools=3
  preflight: SPAWN_OK  live=0  cap=24  headroom=24  seat_free=12
```

⇒ **A 30-session ask grants 12 on this box** — bounded by `seat_free=12` across 3 distinct
t1 account pools, *not* by the cap of 24, and not by the host (`host_cap=32`, cores-bound).
⛔ **Never quote the cap as the wave size.** Run the plan and read `granted`/`shortfall`;
the binding term moves with seat health (2 seats sat `auth_failed` at this reading).
⭐ **N closes come from refill throughput, not concurrency**: workers exit, `fak dispatch
auto --live` tops the population back toward target. A `REFUSE_*` is never something to
route around; `-SkipPreflight` is an operator-only override that removes the floor for the
whole wave.

⭐ **`-Workspace` — FIXED (#5895, `2b2710ee71`).** Both launchers used to default to a
stale sibling clone missing `tools/proc_resource_guard.py`, so preflight fail-safed to
**`REFUSE_INSPECT`, cap=4, granted=0** and refused the whole wave; `launch_wave_detached.ps1`
pinned that same sibling's `.goal-runs`, which is how a probe reads *another* wave's
artifacts. Both now derive from `$PSScriptRoot` (`tools/` → repo root), and the guard test
refuses any absolute-path default. A bare invocation targets **this** checkout and reaches
the gate. **The workspace is still the tree your workers will share** — pass it explicitly
whenever that is not the checkout you are running from.

**4. Workers here SELF-SELECT their leaf — so only `fak intent` stops double-work.**
Unlike an orchestrator-assigned wave, fak's fuel has each worker pick the top-ranked ready
leaf routed to *its own* lane. The orchestrator therefore **cannot** pre-claim 30 specific
tickets. A lane lease stops two workers editing the same FILES and says nothing about two
workers doing the same TICKET in different files. The fuel makes `fak intent claim`
mandatory per worker for exactly this reason — do not weaken it.

---

## Phase 0 — Gate: is something already gardening? (decide BEFORE anything else)

The commonest way this skill wastes a turn is marching all the way to launch and only then
noticing a standing dispatcher already owns the fleet. One read settles it:

```bash
python tools/dispatch_status.py --fast
```

| card says | do |
|---|---|
| `FleetIssueDispatch` installed + enabled and its action verifies for this workspace | Proceed through native `fak dispatch wave`; it joins the global worker/account/admission and issue-intent control plane instead of creating a second queue owner. |
| task/process present but the extension contract does **not** verify | ⛔ **STOP — AUTOMATION_COLLISION.** A name or an empty worker snapshot is not permission to share the queue. |
| no standing gardener | Proceed to Phase 1 in standalone mode. |

⛔ **`lane lease: N held — dead-holder=N` is NOT a blocker and must NOT be reaped.**
Measured 2026-08-08: 48 of 48 lane leases were dead-holder, some 559 h old. The admission
fold `live_leases(config, expire_dead=True)` already elides them at read time for every
consumer, so they block exactly one thing — `lane_lease.acquire()` on that specific lane.
`dos lease-lane release` runs **no liveness check** and its `--owner ""` matches any
holder, so a blind sweep evicts live work (#5859). Watch for a repeated acquire REFUSE on
one lane and escalate *that* lane; do not sweep.

## End-to-end ownership default

A wave dispatches **issue owners**, not finishers waiting for already-complete work. Every selected
issue starts in an unknown state: unstarted, partial, implemented-but-unwitnessed, or shipped. The
owner inspects and moves the actual state to a witnessed ship or a durable, clean handoff.

Price work shape as well as worker count:

- `BOUNDED`: one worker can perform root implementation and witness the issue within its context and
  deadline. It works directly and does not orchestrate.
- `BROAD`: at least two independently executable, tree-disjoint packets are required, or the whole
  issue cannot credibly fit one worker's remaining context/deadline. Dispatch a parent ISSUE_OWNER
  and reserve child capacity. Guarded delegation is mandatory when it is needed and available.
- `LEAF_CHILD`: one bounded child packet. It never orchestrates and never closes the parent.

The parent keeps the ticket claim and closure authority, starts the smallest executable root spine,
prices child tree collisions, and records an ignored execution map under the wave run directory:
acceptance item, evidence, implementation step, exact trees, witness, owner, child identity, lease,
status, and artifact. One-level fan-out is the safety boundary: enough decomposition for broad work
without fork bombs or unowned descendants.

A child-launch or lease refusal lowers concurrency only; it does not erase agent-accessible root
work. The owner continues safe root steps itself. Clean exit requires independent read-back of every
child effect, every child verified/parked/stopped, all owned intents/leases released, and either a
coherent ship or a durable park naming implemented state, missing witness, and exact next command.

## Phase 1 — Price the wave, and state the shortfall out loud

⭐ **The Phase 3 dry-run IS the pricing instrument — do not price by hand.** It folds the
preflight, the seat pool and account distinctness into one line you can quote. Run it
first (Phase 2 renders the fuel it needs), and record the base sha:

```bash
git rev-parse HEAD > .fak/wave/BASE.sha          # ⭐ unrecoverable later — Phase 5 needs it
```

Read `requested` / `granted` / `shortfall` / `distinct_pools` and the preflight's binding
term. Then say the arithmetic plainly, before spawning anything:

> *"You asked for 30. The plan grants 12 — bound by `seat_free=12` across 3 account pools,
> not by the cap of 24. So 12 launch now and the refill door tops up as they exit."*

⛔ A wave that silently under-fills reads as a launch of 30 and gets scored against a
target it never had the capacity to attempt.

**Pricing the closing target.** N closes in H hours is a *throughput* claim, not a
concurrency one. The honest form: `closes_needed / H` versus the witnessed close rate the
box has actually shown — read it from `fak dispatch progress --target <N>` and the
`watch:`/`supply:` folds of the Phase 0 card, never from a worker's log. ⛔ Do not promise
the target off worker count; promise it off rate, and report the gap when the rate is
short. Measured 2026-08-08 the card read `supply: DRAINING arrival −0.167/h vs service
0.083/h` with `loop-closed 106→106` flat over the window — a 30-in-4h target is well above
that rate, which makes it fine to *pursue* and dishonest to *promise*.

## Phase 2 — Render the operator receipt

```bash
WAVE=fw$(date -u +%m%d%H%M)                       # single-use; never reuse a wave id
DEADLINE=$(date -u -d '+4 hours' +%Y-%m-%dT%H:%MZ)
mkdir -p .fak/wave
sed -e "s/{{WAVE}}/$WAVE/g" -e "s/{{DEADLINE}}/$DEADLINE/g" \
    .claude/skills/fleet-wave/fuel-wave.md > ".fak/wave/$WAVE.md"
```

⭐ **This file is an operator receipt, not child fuel.** Keep it for attribution and
reconciliation, but never pass it as `-PointerFile` and never compare it with the legacy
4000-character `/goal` ceiling. Native dispatch records issue/lane/account/PID/run paths
in its typed receipt and renders bounded per-issue worker fuel itself.

## Phase 3 — DRY-RUN, then launch **once**

Confirm the installed native contract and account offer first:

```powershell
fak dispatch wave --help
fak fleet-accounts wave --count 30 --work-kind codex --product codex --json
fak fleet-accounts status --provider codex --json
```

The `fleet-accounts wave` receipt—not account-directory count—is the account-pool authority,
but it is not sufficient by itself. Quote the preflight's OS process census (`live`, `headroom`,
`os_worker_procs`) beside `seat_free`, plus `granted`, `shortfall`, `distinct_pools`, and each
lane's `config_dir`, `pool`, and `session_slot`. A large leased/live claim that disagrees with
top-level Codex process trees is a **pricing defect**, not a reason to stop: preserve both
readings, use the independently witnessed process count for host headroom, and fix/file the
stale registry or sidecar source. If allocation underfills, launch the available tranche and
let the refill controller reprice on cadence. If status says zero while the offer grants seats,
preserve the offer and surface the inconsistency.

Current `dispatch wave` is dry-run unless `--live` is present:

```powershell
fak dispatch wave --count 30 --backend codex --work-kind codex --max-workers 30 `
  --goal high-priority --workspace . --json
```

Read the typed plan before launch. It must select real issue contracts, pairwise-safe lanes,
distinct switcher seats, guarded worker commands, and explicit refusal/downsize reasons.
Each dispatch tick renders bounded fuel from the live GitHub issue and requires full
implement-test-witness-commit-push ownership. The wave fuel file is an operator receipt, not
a child prompt argument.

Do not pass stale `--deadline`, `--fuel-dir`, `--dry-run`, `--launch`, or `--accounts` flags
unless installed help advertises them. The deadline remains the monitor/refill stop condition.
The launcher defaults to a 60-second capacity recheck for four hours
(`-RefillCadenceSeconds 60 -RefillForMinutes 240`): do not add `-NoRefill`, shorten that window,
or stop after the first tranche unless the operator explicitly changes the deadline. `Count` is
the total workers to launch over the window, not a promise that all N must fit concurrently;
completed processes create refill headroom.

Launch exactly once after explicit intent and a clean dry run:

```powershell
fak dispatch wave --count 30 --backend codex --work-kind codex --max-workers 30 `
  --goal high-priority --workspace . --live --json
```

When the verified standing gardener is present, this native command joins the shared
account/admission/intent control plane; there is no separate `-ExtendStanding` flag in the
installed contract. `--count` requests account-session slots; `--max-workers` is the hard process ceiling. The
switcher binds each child to its assigned `CODEX_HOME`. Record wave ID, issue/lane/account,
PID, and run/log paths from the receipt. Never globally switch the interactive account, copy
credentials, or replace this path with raw `Start-Process` calls.

## Phase 4 — Monitor. Watchers see what workers inside the tree cannot.

```bash
RUNS="$(git rev-parse --show-toplevel)/.goal-runs"          # ⭐ the workspace you launched with
ls -t "$RUNS"/$WAVE-*.pid | wc -l                           # spawned
cat "$RUNS"/$WAVE-*.out.log                                 # ⭐ READ THIS FIRST — see below
tail -5 "$RUNS"/$WAVE-*.err.log                             # guard's own exit summary
fak dispatch status --runs-dir "$RUNS" --json
```

⛔ **Derive the runs dir from the workspace you launched with — never hardcode the
sibling checkout.** `launch_goal_detached.ps1` sets `$LogDir = Join-Path $Workspace
'.goal-runs'`, so probing a different tree silently reads *another* wave's artifacts. The
default sibling's `.goal-runs` held **941** of them at last count and ids recycle — that
is the stale predecessor vouching for the corpse, self-inflicted.

⭐ **`out.log` is the highest-value read in the whole phase and it is ~117 bytes.** It
carries the upstream refusal verbatim. When the wave dies together, three causes present
identically and this one line separates them:

| out.log says | cause | next |
|---|---|---|
| nine **distinct** `session <id>` + `BUDGET_CONTEXT_EXHAUSTED` | budget starvation | join the guard session id from `err.log` onto `.fak/nightrun/compaction-health.jsonl` |
| **one** session id repeated across all workers | seat/env collapse | the launcher's strip-then-pin regressed — do not hand-wrap |
| empty, processes actually alive | broken probe | bash `kill -0` lies on native Windows pids; use PowerShell `Get-Process -Id` |

⛔ **Healthy compaction does NOT exonerate the context path.** `anchor_starved=0`,
`solvency_forced=0` and all-correct bail reasons (`under_budget`, `too_few_msgs`,
`burst_unprofitable`) mean the compactor worked *and the session died anyway* — which
indicts the CUMULATIVE drain ceiling (`--context-budget-tokens`), a different knob on a
different scale from the per-turn shed-line. A worker reads a comfortable
`ctx:83.2k/96.0k` on the turn its cumulative budget dies. Never compare the two numbers.

Spawn 2–3 **independent read-only** monitor sessions with [`monitor.md`](monitor.md).
Their job is the three things no worker can see from inside: a **committed-red trunk**, a
**ticket with two owners**, and a **silent death**.

⛔ **A monitor told to "sample every 3 minutes" DIES after pass one.** A session ends when
the model stops calling tools — announcing a wait *is* exiting. The sleep must live inside
one Bash call: `for i in 1 2 3; do { <checks>; } >> $LOG 2>&1; sleep 180; done`, then the
model reads the chunk, judges, and immediately issues the next one.

⛔ **Verify every alert before acting.** A watcher that reports the whole wave dead is far
more often a broken probe than a dead wave — require an explicit `ALL_CLEAR` enumeration
too, or silence is indistinguishable from a broken watcher.

⛔ **Never `Read` a worker's raw transcript** — it is JSONL and will blow up your context.
Read the `.err.log` tail and the report fields.

## Phase 5 — Reconcile from git, then release. Reports understate the work.

```bash
/wave-harvest                                                  # the reconcile half — call it
git log "$(cat .fak/wave/BASE.sha)"..HEAD --pretty='%h %s' --name-only
fak dispatch progress --target 30 --json                       # closes vs the stated goal
fak dispatch closure-audit --workspace . --json
```

- ⭐ **Build the ledger from `git log`, not from reports.** A long, healthy report routinely
  undercounts its own author, because the worker writes its `LANDED:` line before its last
  commits land and never returns. Reconcile dispatched ids against the range diff and treat
  every report as commentary.
- ⭐ **Read every parked tag message — that is where a wave's best work hides.**
  `for t in $(git tag -l "hold/$WAVE-*"); do git log -1 --format='=== %S%n%B' "$t"; done`.
  Then **post those accounts to the tickets yourself**: the worker that did the work is
  gone, and this is the orchestrator's job precisely for that reason.
- ⛔ **Push the `hold/*` refs — a worker physically cannot.** `git push` of a raw ref is
  `REQUIRE_WITNESS`-gated, so every park is durable on this box only until you close the
  loop. Refs live in **two** namespaces and a check over one reads a strand as
  over-published:
  ```bash
  git for-each-ref --format='%(refname)' refs/tags/hold refs/hold | wc -l
  git ls-remote origin 'refs/tags/hold/*' 'refs/hold/*' | sed 's/\^{}$//' | sort -u | wc -l
  ```
  Re-run **both counts after pushing** and report the closed number — the push is not the
  evidence, the recount is.
- **Release the ticket claims.** A lane release does not touch `fak intent`. Finish with
  `fak intent list` showing none of your wave's holders:
  ```bash
  fak intent list | jq -r '.[] | select(.holder|startswith("'"$WAVE"'-")) | .target'
  ```
  The leak is soft — a 3600 s TTL and `fak garden` reap it — but inside that hour they read
  **live** to every peer, which is exactly the window the next wave ranks in.
- ⛔ **Report the target honestly.** *"30 sessions launched"* is not *"30 issues closed."*
  Close the wave with: launched / still live / witnessed closes against the target / parked
  but unlanded / and the shortfall named. A launch is not a ship.

---

## Trap index

| # | trap | tell |
|---|---|---|
| 1 | mixing native and legacy launchers | stale `-PointerFile`, `-ExtendStanding`, or `-Launch` appears beside `fak dispatch wave` — use the native dry-run/`--live` pair only |
| 2 | treating the operator receipt as child fuel | a 4000-character gate is applied to `.fak/wave/$WAVE.md` — native dispatch renders bounded child fuel itself |
| 3 | asking for N > what seats allow | plan shows `granted` ≪ `requested`; the binding term is usually `seat_free`, **not** `cap` |
| 3b | ~~default `-Workspace` names a sibling clone~~ **fixed #5895** | was: `REFUSE_INSPECT … guard not found: …\tools\proc_resource_guard.py`, cap=4, granted=0. Both launchers now derive from `$PSScriptRoot` |
| 4 | hand-wrapping workers in your own `fak guard` | whole wave dies together when the parent gateway exits |
| 5 | re-running the launcher to "top up" | the per-spawn gate can't see starting siblings — use `fak dispatch auto` |
| 6 | reaping dead-holder lane leases | `dos lease-lane release` has no liveness check; `--owner ""` evicts live work |
| 7 | two workers, one ticket, lanes both green | lanes partition FILES, not TICKETS — gate on `fak intent claim` |
| 8 | `fak intent claim … \| head` | `$?` is the pipe's: a live collision reads rc=0 and stays on the roster |
| 9 | reusing a wave id / pointer name | last wave's logs and pid crumbs answer as if they were this run's |
| 10 | monitor announcing a wait | one pass, clean exit, no further samples — sleep must be inside the Bash call |
| 11 | treating the first capacity sample as final | underfilled tranche exits while seats later open — keep the launcher's 60s refill loop alive to the deadline |
| 12 | trusting leased-seat counts without a process census | "nearly full" with few top-level Codex trees — quote `live`/`os_worker_procs`, repair stale sidecars, and reprice |
| 13 | trusting a worker's `LANDED:` line | report says none; `git log` for that lane says three |
| 14 | `hold/*` refs never pushed | `for-each-ref` count ≫ `ls-remote` count — and only if you check BOTH namespaces |
| 15 | an OPEN issue read as unstarted | the fix is already on the trunk under a commit that never cited it |
| 16 | reporting launches as closes | "30 sessions" ≠ "30 issues" — only a witnessed `Fixes #N` on the trunk closes |
