---
title: "fak issue-dispatch loop: witness-gated agent fleet"
description: "The fak issue-dispatch loop spawns capped, witness-gated workers that resolve GitHub issues, ship #N commits, and close tickets only when verified."
---

# The issue-dispatch loop (`dispatch-loop`)

The issue-dispatch loop is fak's witness-gated driver for a GitHub-issue backlog: it spawns a capped fleet of detached `claude -p` workers, each scoped to one open issue, and closes a ticket only after a commit citing `#N` is bound to it and re-verified per-SHA by `dos commit-audit` — never on the worker's word. Because this repo ships no `PLAN-*.md` portfolio, the open issues themselves are the work surface, routed to the `dos.toml` lane whose file-tree each one touches. The whole loop runs unattended on three OS scheduled tasks, bounded so the live-worker population can never exceed its cap (the no-DoS guarantee). It defaults to dry-run; `--live` is the explicit opt-in to autonomous spawning and closing.

> The fleet's GitHub-issue backlog driver, **closed and witness-gated**. This repo
> ships no `PLAN-*.md` (`dos` reports `PLAN_SURFACE_EMPTY`), so the backlog lives in
> GitHub *issues*, not a plan portfolio. The loop spawns a worker at one concrete
> open issue, the worker ships a commit citing `#N`, a witness binds the commit to
> the issue, and a deterministic close arm drives the resolved ticket to CLOSED —
> each close re-verified per-SHA by [`dos commit-audit`](https://github.com/anthony-chaudhary/fak/blob/main/tools/issue_resolve_witnessed.py),
> never by the worker's word. The whole thing runs unattended on three OS scheduled
> tasks, bounded so the live-worker population can never exceed a cap (the no-DoS
> guarantee). An operator-local, human-readable view is rendered to
> `.dispatch-runs/dispatch-status.md` (gitignored), refreshed by the loop itself.

## The gap this closes

The generic `/dos-kernel:dos-dispatch-loop` worker resolves *plan units* from the
plan portfolio. On a plan-empty repo it has no work surface and closes nothing —
workers spin and produce nothing. The DOS supervisor dispatches by **lane**, and a
lane-worker picks plan work; issues are invisible to it. So a live supervisor run
only resolves tickets that happen to ride along on plan-lane work; it cannot *target*
the backlog. [`issue_closure_audit.py`](https://github.com/anthony-chaudhary/fak/blob/main/tools/issue_closure_audit.py) proved the
cost: closure rate sat near zero because nothing aimed the fleet at tickets.

This loop is the missing aim. It treats **the open-issue backlog as the work
surface**, routes each issue to the lane whose file-tree it touches, and dispatches a
scoped worker per issue — while keeping every safety primitive the plan path had.

## The parts → the pipeline

| Stage | Tool | What it does |
|---|---|---|
| 0. **Gate** | `fak dispatch tick` (`internal/dispatchtick` preflight evaluator) | `SPAWN_OK` iff native host process guard clean ∧ native account routing finds a free worker ∧ native seat-pool admission has headroom ∧ live workers < cap. The account route reads `tools/_registry/sessions.json` plus host-local `route_weights`; the seat pool reads live `.account` sidecars. The cap bound is the no-DoS proof. The legacy [`dispatch_preflight.py`](https://github.com/anthony-chaudhary/fak/blob/main/tools/dispatch_preflight.py) / [`proc_resource_guard.py`](https://github.com/anthony-chaudhary/fak/blob/main/tools/proc_resource_guard.py) / `fleet_accounts.py route|seats` path remains for compatibility and standalone operator modes; `fak dispatch tick` no longer shells to them. |
| 1. **Route** | `fak dispatch route` / `fak dispatch tick` (`internal/dispatchtick` router) | Maps each open `gh` issue → a `dos.toml` lane via a confidence ladder (path-confirmed > exact-scope > alias > label > none). `UNROUTED` is first-class; exclusive lanes are never auto-routed. `route --json` exposes the same lanes payload that `tick` consumes. The legacy [`issue_lane_router.py`](https://github.com/anthony-chaudhary/fak/blob/main/tools/issue_lane_router.py) remains for older Python dispatch entry points; native dispatch no longer shells to it. |
| 2. **Spawn** | `fak dispatch tick` / `fak dispatch wave` | `tick` picks the busiest lane's first non-skipped open issue, renders the prompt, and launches ONE detached worker on the routed account. `wave` allocates N distinct native account pools in one call, stamps rank/wave membership, then feeds each lane through the same tick path. Generation-aware selection keeps `gen/now` and scoped `gen/next` launchable by default, holds `gen/second-next`/`gen/future` unless explicitly requested, and preserves lane pressure plus priority as the dominant ordering signals; the policy is pinned in [`docs/notes/GENERATION-DISPATCH-SELECTION-2026-06-30.md`](notes/GENERATION-DISPATCH-SELECTION-2026-06-30.md). Anti-churn cooldown + in-flight de-dup so it *walks* the backlog instead of re-storming one un-landable issue. The scheduled `FleetIssueDispatch -Mode resolve` task now runs `fak dispatch tick`; the legacy [`issue_resolve_dispatch.py`](https://github.com/anthony-chaudhary/fak/blob/main/tools/issue_resolve_dispatch.py) path remains for older Python entry points. |
| 2a. **Prompt** | `fak dispatch tick` (`internal/dispatchtick` prompt renderer) | Renders the per-issue resolution prompt: the smallest correct change, the git laws (trunk-only, commit `-s` by path), honest-block-first, and the load-bearing **`#N`-in-subject** rule. The legacy [`issue_worker_prompt.py`](https://github.com/anthony-chaudhary/fak/blob/main/tools/issue_worker_prompt.py) remains as a compatibility shim for older Python dispatch entry points. |
| 3. **Witness** | [`issue_closure_audit.py`](https://github.com/anthony-chaudhary/fak/blob/main/tools/issue_closure_audit.py) | Binds each issue to its resolving commit(s) from the commit text, grades through `dos commit-audit`: `TRUE_RESOLVED` / `CLAIMED_CLOSED` / `OPEN_WITNESSED` / `OPEN`. `closure_rate = TRUE / (TRUE + CLAIMED)`. |
| 4. **Close** | [`issue_resolve_witnessed.py`](https://github.com/anthony-chaudhary/fak/blob/main/tools/issue_resolve_witnessed.py) | The deterministic close arm — no model, no edit. For each `OPEN_WITNESSED` issue it **re-runs** `dos commit-audit <sha>` at close time and closes via `gh issue close` citing the SHA iff `OK` ∧ `diff-witnessed`. Reversible with `gh issue reopen`. |
| 5. **Harvest** | `fak dispatch progress` / [`issue_resolve_progress.py`](https://github.com/anthony-chaudhary/fak/blob/main/tools/issue_resolve_progress.py) | Native `progress` snapshots open / closed-by-loop / witnessed counts to `.dispatch-runs/progress.jsonl` (the curve), records the baseline, and emits loop-ledger witness rows. The legacy Python progress script still drives `--close` until the native witnessed close arm lands. Counts only closes carrying the close-arm's signature as the loop's own work. |
| 6. **Surface** | [`dispatch_status.py`](https://github.com/anthony-chaudhary/fak/blob/main/tools/dispatch_status.py) | One-touch operator card; `--md` writes the operator-local `.dispatch-runs/dispatch-status.md` (gitignored; backlog-by-lane, closure honesty, silent-worker scan). |

Close-arm dry-runs render the per-candidate decision table before any GitHub
mutation:

```text
resolve-witnessed: PLANNED (ok)  live=False  candidates=2 planned=2
  issue   sha        audit                  decision  reason
  #1800   45d88e28ab OK/diff-witnessed      close     witness ok; dry-run only
  #1801   abcdef1234 DENY/?                 hold      commit-audit verdict=DENY witness=?
  -> closed=0 would_close=1 skipped=1 unpushed=0 failed=0  (gate=active, closure_rate before=0.91)
  DRY-RUN - re-run with --live to execute the gh closes
```

For generation-specific concurrent loop scheduling, use
[`generation-loop-scheduling.md`](generation-loop-scheduling.md). It defines the
held/default-admitted buckets, shared-lease contention behavior, and operator
override evidence without changing the dispatch loop's shared-trunk rules.

Use the [dispatch SLO glossary](dispatch-slo-glossary.md) for report and status
terms shared by the loop, close arm, and operator summaries.

## Capacity equation for 400 issues/hour

Use this equation before raising worker count or claiming the fleet can hit the
400 issues/hour target:

```text
effective_workers = min(worker_count, host_cap, seat_cap, lease_cap, routeable_issue_cap)
cycle_sec = median_session_sec + median_witness_latency_sec
raw_attempts_per_hour = effective_workers * 3600 / cycle_sec
net_issues_per_hour = raw_attempts_per_hour * close_rate / (1 + retry_rate)
```

Variables:

- `worker_count`: the operator-requested worker population.
- `host_cap`: the host resource ceiling from CPU/RAM/process headroom.
- `seat_cap`: routable account seats available for worker launches.
- `lease_cap`: the DOS/lane lease ceiling for non-overlapping file trees.
- `routeable_issue_cap`: open issues with enough scope, path, and dependency
  clearance to launch now.
- `median_session_sec`: median wall time from worker spawn to exit.
- `median_witness_latency_sec`: median parent-side time to rerun tests,
  `dos commit-audit`, issue-state checks, and close-arm verification.
- `close_rate`: witnessed closes divided by completed attempts. Worker
  self-reports do not count.
- `retry_rate`: retry attempts per witnessed close. Use `0.20` when every five
  witnessed closes consumed one additional retry attempt.

The target test is:

```text
net_issues_per_hour >= 400
```

The reverse form tells you how many effective workers are needed:

```text
required_effective_workers =
  ceil(400 * cycle_sec * (1 + retry_rate) / (3600 * close_rate))
```

Example that reaches the target:

```text
worker_count=120, host_cap=110, seat_cap=105, lease_cap=100, routeable_issue_cap=140
effective_workers = min(120, 110, 105, 100, 140) = 100
cycle_sec = 600s median session + 30s witness latency = 630s
raw_attempts_per_hour = 100 * 3600 / 630 = 571.4
net_issues_per_hour = 571.4 * 0.85 / (1 + 0.15) = 422.4
result: reaches 400 issues/hour
```

Example that misses the target:

```text
worker_count=120, host_cap=80, seat_cap=60, lease_cap=90, routeable_issue_cap=100
effective_workers = min(120, 80, 60, 90, 100) = 60
cycle_sec = 720s median session + 60s witness latency = 780s
raw_attempts_per_hour = 60 * 3600 / 780 = 276.9
net_issues_per_hour = 276.9 * 0.75 / (1 + 0.25) = 166.2
required_effective_workers = ceil(400 * 780 * 1.25 / (3600 * 0.75)) = 145
result: misses 400 issues/hour; the next limiter is seat_cap, not worker_count
```

## The load-bearing invariants

These are the rules that make it safe to hand autonomous spawning to an unattended
loop. Each one is a hard guarantee, not a best effort:

- **DoS cap.** The live worker population is provably ≤ `cap = min(--max-workers,
  dos [supervise].target, host_cap, seats)`, where `live = MAX(kernel lease count, OS
  process scan for the worker marker)` — so neither a stale lease nor an unleased
  orphan can hide capacity. `--max-workers` (default **4**) is only the operator's
  outer ceiling; the binding safety terms are `host_cap` (#1337, the box's adaptive
  cores/RAM/thread headroom — it auto-throttles a loaded host and recovers as load
  clears) and `seats` (#1336, one routable account per worker, so a spawn can never
  double-book a rate limit). Because those two can only *lower* the effective cap,
  doubling the static ceiling 2→4 raises concurrency exactly as far as the box and
  the account pool allow and no further. The preflight `REFUSE_AT_CAP` / `REFUSE_NO_SEAT`
  is correct steady-state behavior, not a failure.
- **`#N`-in-subject binding.** The commit→issue link is reconstructed **only** from
  the commit subject/body (`close/fix/resolve #N`, or `#N` in the subject), because
  this repo runs no PR-keyword workflow. A resolved issue whose commit omits `#N` can
  never be witnessed-closed — which is why the worker prompt bakes the rule in.
- **Per-SHA re-verify at close.** The close arm never trusts the audit's bucket; it
  re-asks `dos commit-audit` per SHA at close time. No keep on a self-authored claim
  (the same discipline as the [RSI loop's](rsi-loop.md) non-forgeable keep-bit).
- **Anti-churn cooldown.** An issue attempted within `--cooldown-min` (default 120)
  is skipped so the picker advances down the lane instead of re-storming a known
  drain; in-flight de-dup separately skips an issue with a live worker.
- **Dry-run by default.** Every tool plans only until `--live`; the scheduled tasks
  install dry-run unless `-Live` is passed. `--live` is the explicit opt-in to
  autonomous spawning / closing.

## Before spawning: map the limiter

Run the [bottleneck map loop](bottleneck-map-loop.md) before turning on a dispatch
window or when the loop reports `AT_CAP`/low throughput. That fold answers whether
the next bottleneck is fleet capacity/recovery or the issue backlog itself.
Before increasing `--max-workers` or a scheduled task's `-MaxWorkers`, run the
[safe-to-raise-cap checklist](safe-to-raise-cap-checklist.md); it requires green
seats, host cap, lease health, rate budget, and closure honesty, and records a
raise/hold decision row.

If fleet health is CRITICAL/HIGH from account throttles or auth failures, treat it
as a **transient dispatch gate**: cap the spawn arm and recheck after reset/relogin
instead of elevating it to the top strategic problem. If the CRITICAL/HIGH row is
recovery plumbing, watchdog, auto-resume, or surfacing backlog, treat it as
semi-durable process debt and fix it before broad dispatch. In both cases, keep the
open-work lens visible: `/issue-triage` may still need to cut taxonomy debt or an
ownership pass may still need to claim/defer orphan P0/P1 work before issue-dispatch
spawns the next worker.

## Run it

```bash
# the operator status card (add --json for machine output, --fast to skip gh folds)
python tools/dispatch_status.py

# progress toward the target (snapshot only)
go run ./cmd/fak dispatch progress --target 50

# spawn ONE issue worker now (cooldown-aware; busiest lane's next fresh issue)
go run ./cmd/fak dispatch tick            # dry-run / plan
go run ./cmd/fak dispatch tick --live      # spawn

# feed public-routeable maturity-ladder gaps into the issue backlog the dispatcher drains
# private-boundary lanes stay visible in `fak maturity next` and are skipped here
go run ./cmd/fak maturity route --fetch-existing --limit 3   # dry-run: create/update plan
go run ./cmd/fak maturity route --live --limit 3             # create/update public issues

# close every witnessed-but-still-open issue now (each re-verified per-SHA)
python tools/issue_resolve_witnessed.py            # dry-run / plan
python tools/issue_resolve_progress.py --close --live

# render the operator-local status doc (gitignored; never committed)
python tools/dispatch_status.py --md .dispatch-runs/dispatch-status.md
```

## The always-on tasks (the "keep going" loop)

Three Windows Scheduled Tasks drive the loop on a cadence. Each installs **dry-run by
default**; `-Live` opts into the side effect.

| Task | Installer | Cadence | Arm |
|---|---|---|---|
| `FleetIssueDispatch` | [`register_issue_dispatch.ps1`](https://github.com/anthony-chaudhary/fak/blob/main/tools/register_issue_dispatch.ps1) | 10 min | SPAWN — one native `fak dispatch tick` issue worker per tick (`-Mode resolve`, default). `-Mode loop` runs the legacy plan-portfolio arm instead (dormant until `PLAN-*.md` ship). |
| `FleetResolveProgress` | [`register_resolve_progress.ps1`](https://github.com/anthony-chaudhary/fak/blob/main/tools/register_resolve_progress.ps1) | 15 min | CLOSE / harvest — snapshot the curve and close `OPEN_WITNESSED` issues. DoS-free (no worker spawned). |
| `FleetDispatchStatusDoc` | [`register_dispatch_status_doc.ps1`](https://github.com/anthony-chaudhary/fak/blob/main/tools/register_dispatch_status_doc.ps1) | 30 min | DOC — render the gitignored, operator-local `.dispatch-runs/dispatch-status.md`. Read-only fold; never committed. |

All three tasks are installed through `fak loop run`; the spawn task's default
resolve arm also runs the native `fak dispatch tick` child instead of the legacy
Python dispatcher. The Task Scheduler fire/start/end wrapper rows land in
`.fak/loops.jsonl` under
`issue-resolve-dispatch/task-scheduler/<backend>`, `issue-resolve-progress/task-scheduler`,
and `dispatch-status-doc/task-scheduler`; the native spawn child records its own
admission/spawn rows under `issue-resolve-dispatch/<backend>`, while the progress
producer records progress/witness rows under `issue-resolve-progress`.
(`FleetDispatchStatusDoc` is a read-only render, so it adds only the wrapper run
rows — enough to see in `fak loop status` that the doc actually refreshed.)

```powershell
# install all three live (bounded autonomous spawn + close + doc refresh)
.\tools\register_issue_dispatch.ps1     -Workspace C:\work\fak -Mode resolve -Live -MaxWorkers 4
.\tools\register_resolve_progress.ps1   -Workspace C:\work\fak -Live -Target 50
.\tools\register_dispatch_status_doc.ps1 -Workspace C:\work\fak -EveryMinutes 30

# status / remove any of them
.\tools\register_issue_dispatch.ps1 -Action preview
.\tools\register_issue_dispatch.ps1 -Action status
.\tools\register_issue_dispatch.ps1 -Action remove
```

Together: **spawn → ship `#N` commit → witness → close → refresh the doc**, unattended
and cap-bounded.

## Recently-created feature dogfood

When a new local feature lands, run the same dogfood packet instead of inventing a
one-off proof. It exercises the current loop ledger, vCache score/refutation
surface, benchmark catalog, avoided-call economics tests, prompt-tool-pruning
tests, code-slop scorecard, and dogfood coverage scorecard, then writes a JSON
evidence bundle under `.fak/recent-feature-dogfood/`.

```bash
# quick local run
python tools/recent_feature_dogfood.py

# scheduler/manual run with OS-edge loop rows
go run ./cmd/fak loop run --loop recent-feature-dogfood/manual --source manual -- \
  python tools/recent_feature_dogfood.py

# cron/launchd/systemd helper form
tools/fak_loop_run.sh recent-feature-dogfood/cron cron -- \
  python tools/recent_feature_dogfood.py
```

The scorecards may report ACTION/debt and still pass this dogfood packet when the
machine payload is valid. The pass condition is repeatable use of the feature and
valid evidence, not pretending existing repo debt is already gone.

### As a CI gate (`.github/workflows/dogfood.yml`)

The same packet runs on a clean checkout in CI (issue #798), so the recently-shipped
CLI surfaces are proven to work on every push — not just locally. The
[`dogfood`](https://github.com/anthony-chaudhary/fak/blob/main/.github/workflows/dogfood.yml) workflow builds a real `fak` into
`tools/.bin/`, runs the packet into a fixed evidence dir, fails the build when a
**required** probe fails, and uploads `report.json` + the vCache score artifact as build
artifacts (with the human report written to the run's step summary). It runs on push to
`main`/`master`, on pull requests, on a daily `schedule:`, and on demand via
`workflow_dispatch`. The gate preserves the local semantics: a scorecard reporting
ACTION/debt does **not** fail the packet — only an invalid machine payload does.

```bash
# the exact gate command CI runs (writes evidence under .fak/recent-feature-dogfood/ci):
python tools/recent_feature_dogfood.py --out-dir .fak/recent-feature-dogfood/ci --json

# trigger the workflow manually from a branch:
gh workflow run dogfood.yml
```

## A note on opaque workers

A `claude -p` worker buffers all stdout until its final message, so a detached
worker's log is 0 bytes until it finishes — a killed or timed-out worker also shows 0
bytes. Don't read "0-byte log" as "did nothing while running." The robust progress
signal is **git commits**, not the worker log. `dispatch_status.py` folds a
*silent-worker* scan (a 0-byte log whose pid is already dead) into the status doc so
the genuinely-produced-nothing case is visible to an operator instead of silent; a
single hard issue (often an epic) that one pass can't land is expected, and the
cooldown advances the picker past it.

## Backends: the Claude skill-chain vs. the opencode single-shot worker

The loop spawns its per-issue worker on one of two backends, and they express
the dispatch cadence differently:

- **Claude** drives a *chain of plugin slash-commands* —
  `/dos-dispatch-loop` → `/dos-dispatch` → `/dos-next-up`, with `/dos-replan`
  on a drain. Each `/dos-*` is a dos-kernel plugin skill that loads more
  instruction text into context. The *multi-iteration* loop (the typed
  `drained-twice` / `pick-cooldown` / `pick-held-invariant` stop conditions)
  and the refill-on-drain (`/dos-replan`) live inside `dos-dispatch-loop`, so a
  Claude worker can run its own bounded 10-iteration loop in one process.
- **opencode** has **no plugin slash-command-to-skill loading**, so that chain
  has no 1:1 port. The opencode worker (`.opencode/agent/dos-dispatch.md`, in
  the sibling fleet repo) instead calls the underlying `dos` CLI verbs directly
  (`dos doctor` / `dos arbitrate` / `dos enumerate` / `dos gate` / `dos verify`
  / `dos lease-lane release`) and is **intentionally single-shot**: it discovers
  → takes a lane → snapshots → gates → ships one packet → verifies → releases,
  then exits cleanly.

**Decision (#419): option (b) — the opencode backend is single-shot by design;
the dispatch⇄replan cadence is a supervisor concern, not a worker concern.**
There is deliberately no in-worker opencode expression of `/dos-replan` or the
multi-iteration stop conditions. The refill-on-drain and the
spawn-again-next-tick cadence are owned by the supervisor: the kernel already
holds the loop state (`dos loop_decide`, the WAL, liveness), and the
[always-on tasks](#the-always-on-tasks-the-keep-going-loop) above respawn a
fresh worker each tick at the busiest lane's next fresh issue. A worker that
ships one packet and exits — respawned by the supervisor — is easier to make
resilient than one that runs its own long loop, and it keeps loop state in the
one place (the kernel) that survives a worker crash.

So the gap is **named, not silent**: on the opencode backend the worker's
`gate → DRAIN` is a clean stop, and the *replan* that would refill the backlog
happens on the next supervisor tick, not inside the worker. An unattended
always-on opencode loop is therefore the **supervisor cadence × the single-shot
worker**, not a worker running its own `/dos-dispatch-loop`.

## Extending it / adopting it elsewhere

The loop reads its repo shape — lane names, file-trees, ship-stamp grammar — entirely
from `dos doctor --json`, so it generalizes to any repo whose backlog is GitHub
issues. A standalone, config-driven extraction (single-account default, pluggable
switcher, cross-platform scheduler) is published separately as **`dos-dispatch`**, a
companion to [`dos-kernel`](https://github.com/anthony-chaudhary/dos-kernel); the fak
copy under `tools/` is the reference implementation it was generalized from. The
witness rung, the cap bound, the `#N` binding, and the dry-run discipline carry over
unchanged — the loop is the harness, your issue backlog is the payload.
