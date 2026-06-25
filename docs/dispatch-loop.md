---
title: "fak issue-dispatch loop: witness-gated agent fleet"
description: "The fak issue-dispatch loop spawns capped, witness-gated workers that resolve GitHub issues, ship #N commits, and close tickets only when verified."
---

# The issue-dispatch loop (`dispatch-loop`)

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
| 0. **Gate** | [`dispatch_preflight.py`](https://github.com/anthony-chaudhary/fak/blob/main/tools/dispatch_preflight.py) | `SPAWN_OK` iff host guard clean ∧ an account is free ∧ live workers < cap. The cap bound is the no-DoS proof. |
| 1. **Route** | [`issue_lane_router.py`](https://github.com/anthony-chaudhary/fak/blob/main/tools/issue_lane_router.py) | Maps each open `gh` issue → a `dos.toml` lane via a confidence ladder (path-confirmed > exact-scope > alias > label > none). `UNROUTED` is first-class; exclusive lanes are never auto-routed. |
| 2. **Spawn** | [`issue_resolve_dispatch.py`](https://github.com/anthony-chaudhary/fak/blob/main/tools/issue_resolve_dispatch.py) | Picks the busiest lane's first non-skipped open issue, renders the prompt, launches ONE detached `claude -p` worker on the switcher-pinned account. Anti-churn cooldown + in-flight de-dup so it *walks* the backlog instead of re-storming one un-landable issue. |
| 2a. **Prompt** | [`issue_worker_prompt.py`](https://github.com/anthony-chaudhary/fak/blob/main/tools/issue_worker_prompt.py) | Renders the per-issue resolution prompt: the smallest correct change, the git laws (trunk-only, commit `-s` by path), honest-block-first, and the load-bearing **`#N`-in-subject** rule. |
| 3. **Witness** | [`issue_closure_audit.py`](https://github.com/anthony-chaudhary/fak/blob/main/tools/issue_closure_audit.py) | Binds each issue to its resolving commit(s) from the commit text, grades through `dos commit-audit`: `TRUE_RESOLVED` / `CLAIMED_CLOSED` / `OPEN_WITNESSED` / `OPEN`. `closure_rate = TRUE / (TRUE + CLAIMED)`. |
| 4. **Close** | [`issue_resolve_witnessed.py`](https://github.com/anthony-chaudhary/fak/blob/main/tools/issue_resolve_witnessed.py) | The deterministic close arm — no model, no edit. For each `OPEN_WITNESSED` issue it **re-runs** `dos commit-audit <sha>` at close time and closes via `gh issue close` citing the SHA iff `OK` ∧ `diff-witnessed`. Reversible with `gh issue reopen`. |
| 5. **Harvest** | [`issue_resolve_progress.py`](https://github.com/anthony-chaudhary/fak/blob/main/tools/issue_resolve_progress.py) | Snapshots open / closed-by-loop / witnessed counts to `.dispatch-runs/progress.jsonl` (the curve) and drives the close arm. Counts only closes carrying the close-arm's signature as the loop's own work. |
| 6. **Surface** | [`dispatch_status.py`](https://github.com/anthony-chaudhary/fak/blob/main/tools/dispatch_status.py) | One-touch operator card; `--md` writes the operator-local `.dispatch-runs/dispatch-status.md` (gitignored; backlog-by-lane, closure honesty, silent-worker scan). |

## The load-bearing invariants

These are the rules that make it safe to hand autonomous spawning to an unattended
loop. Each one is a hard guarantee, not a best effort:

- **DoS cap.** The live worker population is provably ≤ `cap = min(--max-workers,
  dos [supervise].target)`, where `live = MAX(kernel lease count, OS process scan for
  the worker marker)` — so neither a stale lease nor an unleased orphan can hide
  capacity. The preflight `REFUSE_AT_CAP` is correct steady-state behavior, not a
  failure.
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
python tools/issue_resolve_progress.py --target 50

# spawn ONE issue worker now (cooldown-aware; busiest lane's next fresh issue)
python tools/issue_resolve_dispatch.py            # dry-run / plan
python tools/issue_resolve_dispatch.py --live      # spawn

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
| `FleetIssueDispatch` | [`register_issue_dispatch.ps1`](https://github.com/anthony-chaudhary/fak/blob/main/tools/register_issue_dispatch.ps1) | 10 min | SPAWN — one guarded issue worker per tick (`-Mode resolve`, default). `-Mode loop` runs the plan-portfolio arm instead (dormant until `PLAN-*.md` ship). |
| `FleetResolveProgress` | [`register_resolve_progress.ps1`](https://github.com/anthony-chaudhary/fak/blob/main/tools/register_resolve_progress.ps1) | 15 min | CLOSE / harvest — snapshot the curve and close `OPEN_WITNESSED` issues. DoS-free (no worker spawned). |
| `FleetDispatchStatusDoc` | [`register_dispatch_status_doc.ps1`](https://github.com/anthony-chaudhary/fak/blob/main/tools/register_dispatch_status_doc.ps1) | 30 min | DOC — render the gitignored, operator-local `.dispatch-runs/dispatch-status.md`. Read-only fold; never committed. |

```powershell
# install all three live (bounded autonomous spawn + close + doc refresh)
.\tools\register_issue_dispatch.ps1     -Workspace C:\work\fak -Mode resolve -Live -MaxWorkers 2
.\tools\register_resolve_progress.ps1   -Workspace C:\work\fak -Live -Target 50
.\tools\register_dispatch_status_doc.ps1 -Workspace C:\work\fak -EveryMinutes 30

# status / remove any of them
.\tools\register_issue_dispatch.ps1 -Action status
.\tools\register_issue_dispatch.ps1 -Action remove
```

Together: **spawn → ship `#N` commit → witness → close → refresh the doc**, unattended
and cap-bounded.

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
