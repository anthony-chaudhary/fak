# The issue-dispatch loop (`dispatch-loop`)

> The fleet's GitHub-issue backlog driver, **closed and witness-gated**. This repo
> ships no `PLAN-*.md` (`dos` reports `PLAN_SURFACE_EMPTY`), so the backlog lives in
> GitHub *issues*, not a plan portfolio. The loop spawns a worker at one concrete
> open issue, the worker ships a commit citing `#N`, a witness binds the commit to
> the issue, and a deterministic close arm drives the resolved ticket to CLOSED —
> each close re-verified per-SHA by [`dos commit-audit`](../tools/issue_resolve_witnessed.py),
> never by the worker's word. The whole thing runs unattended on three OS scheduled
> tasks, bounded so the live-worker population can never exceed a cap (the no-DoS
> guarantee). The committed, human-readable view is
> [`docs/dispatch-status.md`](dispatch-status.md), refreshed by the loop itself.

## The gap this closes

The generic `/dos-kernel:dos-dispatch-loop` worker resolves *plan units* from the
plan portfolio. On a plan-empty repo it has no work surface and closes nothing —
workers spin and produce nothing. The DOS supervisor dispatches by **lane**, and a
lane-worker picks plan work; issues are invisible to it. So a live supervisor run
only resolves tickets that happen to ride along on plan-lane work; it cannot *target*
the backlog. [`issue_closure_audit.py`](../tools/issue_closure_audit.py) proved the
cost: closure rate sat near zero because nothing aimed the fleet at tickets.

This loop is the missing aim. It treats **the open-issue backlog as the work
surface**, routes each issue to the lane whose file-tree it touches, and dispatches a
scoped worker per issue — while keeping every safety primitive the plan path had.

## The parts → the pipeline

| Stage | Tool | What it does |
|---|---|---|
| 0. **Gate** | [`dispatch_preflight.py`](../tools/dispatch_preflight.py) | `SPAWN_OK` iff host guard clean ∧ an account is free ∧ live workers < cap. The cap bound is the no-DoS proof. |
| 1. **Route** | [`issue_lane_router.py`](../tools/issue_lane_router.py) | Maps each open `gh` issue → a `dos.toml` lane via a confidence ladder (path-confirmed > exact-scope > alias > label > none). `UNROUTED` is first-class; exclusive lanes are never auto-routed. |
| 2. **Spawn** | [`issue_resolve_dispatch.py`](../tools/issue_resolve_dispatch.py) | Picks the busiest lane's first non-skipped open issue, renders the prompt, launches ONE detached `claude -p` worker on the switcher-pinned account. Anti-churn cooldown + in-flight de-dup so it *walks* the backlog instead of re-storming one un-landable issue. |
| 2a. **Prompt** | [`issue_worker_prompt.py`](../tools/issue_worker_prompt.py) | Renders the per-issue resolution prompt: the smallest correct change, the git laws (trunk-only, commit `-s` by path), honest-block-first, and the load-bearing **`#N`-in-subject** rule. |
| 3. **Witness** | [`issue_closure_audit.py`](../tools/issue_closure_audit.py) | Binds each issue to its resolving commit(s) from the commit text, grades through `dos commit-audit`: `TRUE_RESOLVED` / `CLAIMED_CLOSED` / `OPEN_WITNESSED` / `OPEN`. `closure_rate = TRUE / (TRUE + CLAIMED)`. |
| 4. **Close** | [`issue_resolve_witnessed.py`](../tools/issue_resolve_witnessed.py) | The deterministic close arm — no model, no edit. For each `OPEN_WITNESSED` issue it **re-runs** `dos commit-audit <sha>` at close time and closes via `gh issue close` citing the SHA iff `OK` ∧ `diff-witnessed`. Reversible with `gh issue reopen`. |
| 5. **Harvest** | [`issue_resolve_progress.py`](../tools/issue_resolve_progress.py) | Snapshots open / closed-by-loop / witnessed counts to `.dispatch-runs/progress.jsonl` (the curve) and drives the close arm. Counts only closes carrying the close-arm's signature as the loop's own work. |
| 6. **Surface** | [`dispatch_status.py`](../tools/dispatch_status.py) | One-touch operator card; `--md` writes [`docs/dispatch-status.md`](dispatch-status.md) (backlog-by-lane, closure honesty, silent-worker scan). |

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

# render the committed status doc, then commit it by path
python tools/dispatch_status.py --md docs/dispatch-status.md
git commit -s -- docs/dispatch-status.md
```

## The always-on tasks (the "keep going" loop)

Three Windows Scheduled Tasks drive the loop on a cadence. Each installs **dry-run by
default**; `-Live` opts into the side effect.

| Task | Installer | Cadence | Arm |
|---|---|---|---|
| `FleetIssueDispatch` | [`register_issue_dispatch.ps1`](../tools/register_issue_dispatch.ps1) | 10 min | SPAWN — one guarded issue worker per tick (`-Mode resolve`, default). `-Mode loop` runs the plan-portfolio arm instead (dormant until `PLAN-*.md` ship). |
| `FleetResolveProgress` | [`register_resolve_progress.ps1`](../tools/register_resolve_progress.ps1) | 15 min | CLOSE / harvest — snapshot the curve and close `OPEN_WITNESSED` issues. DoS-free (no worker spawned). |
| `FleetDispatchStatusDoc` | [`register_dispatch_status_doc.ps1`](../tools/register_dispatch_status_doc.ps1) | 30 min | DOC — render `docs/dispatch-status.md`. Read-only fold; writes the working-tree doc but never commits it. |

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

## Extending it / adopting it elsewhere

The loop reads its repo shape — lane names, file-trees, ship-stamp grammar — entirely
from `dos doctor --json`, so it generalizes to any repo whose backlog is GitHub
issues. A standalone, config-driven extraction (single-account default, pluggable
switcher, cross-platform scheduler) is published separately as **`dos-dispatch`**, a
companion to [`dos-kernel`](https://github.com/anthony-chaudhary/dos-kernel); the fak
copy under `tools/` is the reference implementation it was generalized from. The
witness rung, the cap bound, the `#N` binding, and the dry-run discipline carry over
unchanged — the loop is the harness, your issue backlog is the payload.
