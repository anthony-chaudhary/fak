---
title: "Long-running agent loops: make fak the loop kernel, not another cron"
description: "A strategy and implementation map for rolling cron jobs, cloud/remote execution, other nodes, phone/chat control, OS notifications, foreground/background sessions, and green-thread-like agent loops into one fak-shaped control plane."
---

# Long-running agent loops - fak as the loop kernel

Date: 2026-06-25.

This note is the missing complement to three existing docs:

- [`engineering-is-building-loops`](../explainers/engineering-is-building-loops.md) explains the nested loop ladder.
- [`RESEARCH-cloud-vm-remote-agent-landscape-2026-06-23`](RESEARCH-cloud-vm-remote-agent-landscape-2026-06-23.md) says the market moved compute into cloud/VMs and control onto phones/chat.
- [`SESSION-CONTROL-STATE-AS-FIRST-CLASS`](SESSION-CONTROL-STATE-AS-FIRST-CLASS-2026-06-24.md) makes one live session's drive state queryable and writable.

The remaining problem is operational: an unattended loop is not just an agent prompt. It is
a scheduled, resumable, observable, remotely steerable process that can run while nobody is
watching. Today fak has many good pieces, but the shape is scattered across Windows
Scheduled Tasks, launchd templates, watchdog scripts, Slack lab helpers, `fak guard`,
`fak serve`, `taskmgr`, session images, and DOS witness gates.

The product line should be:

> fak is the kernel for long-running agent loops. Cron, launchd, Task Scheduler, Slack,
> WhatsApp, GitHub, and phones are just interrupt sources. fak owns the capability floor,
> loop identity, run ledger, leases, witness, resume state, and notification discipline.

> **Status update (2026-06-28): the in-kernel RUNTIME + tick source landed.** Rungs 1–2
> below (the durable `loopmgr` ledger and the OS-scheduler adapters) drive every tick from
> OUTSIDE the kernel — `loopmgr` itself "schedules, spawns, notifies, authorizes nothing".
> `internal/bgloop` + `fak bgloop` add the piece that was missing on this whole ladder: an
> in-kernel `Supervisor` that RUNS registered loops in their own goroutines on the
> `fak serve` lifecycle, so a loop progresses BECAUSE the kernel is up, with no external
> scheduler firing it. It contains panics/errors with capped backoff (a crashing loop never
> takes the kernel down), joins cleanly on shutdown, and is observable at `GET /v1/fak/loops`,
> via the `fak_bgloop_*` Prometheus family, and through `fak bgloop status` / `fak bgloop demo`
> (the offline witness). It folds into this note's substrate without coupling — `WithObserver`
> pushes ticks into the `loopmgr` ledger (so an in-kernel loop also shows in `fak loop status`)
> and `WithAdmit` gates fires through `loopmgr.Governor.Admit`. This is the runtime + read-only
> observability complement to Rung 3 below; the authenticated `POST /v1/fak/loops/{id}/fire|signal`
> control bridge remains the next step. See `CLAIMS.md` (Gateway) and `internal/bgloop/doc.go`.

## External signals

Current public products all point in the same direction:

- OpenAI Codex cloud runs coding tasks in the background, including in parallel, in a cloud environment: https://developers.openai.com/codex/cloud
- OpenAI ships Codex in Slack, where a channel mention chooses an environment and returns a cloud-task link: https://openai.com/index/codex-now-generally-available/
- Anthropic Routines package a prompt, repos, and connectors, then run from schedules, API `/fire`, or repository events: https://code.claude.com/docs/en/routines
- GitHub describes Copilot coding agent as an asynchronous background agent and shows mobile issue assignment/review from a phone: https://github.blog/changelog/2026-02-17-assign-issues-to-copilot-coding-agent-from-raycast/ and https://github.blog/developer-skills/github/completing-urgent-fixes-anywhere-with-github-copilot-coding-agent-and-mobile/
- Cursor's Faire case study is the operational version: thousands of automated cloud-agent runs per week, many agents in parallel, and local/cloud handoff: https://cursor.com/blog/faire
- Claude Code hooks now include HTTP hooks, so shared audit/control services are becoming a first-class integration shape: https://code.claude.com/docs/en/hooks-guide

The important read is not "build another cloud agent." The field already has VMs,
dashboards, IDEs, and phone apps. The unresolved layer is the one fak is already good at:
what may run, where it may run, how it is admitted, how it is observed, how it proves done,
and what state survives the next tick.

## Current fak substrate

fak already has most primitives, but not one named loop-control surface.

| Need | Existing substrate | Honest gap |
|---|---|---|
| Foreground guarded session | `fak guard -- <agent>` launches a loopback gateway and child-only base URL | Tied to one local child process; lifecycle ends with the child |
| Always-on gateway | `fak serve`, deployment docs, auth, metrics, `/debug/vars`, audit journal | Gateway is a front door, not yet a loop scheduler |
| Host schedulers | Windows Scheduled Tasks installers, launchd plists, Mac keep-awake script, GCP dogfood plan | Each is bespoke; no common loop record or run ledger |
| Fleet admission | DOS lane leases, `dos_arbitrate`, issue-dispatch preflight and cooldowns | Strong in the dispatch path, not exposed as a general loop admission API |
| Progress and resource view | `internal/taskmgr`, `fak task sample`, gated `GET /v1/fak/tasks` | Process-local only; not durable, cross-PID, or fleet-wide |
| Session control | `internal/session` drive state, budgets, continuation ids, reset-on-budget | A served-session primitive, not yet the outer loop's scheduler state |
| Durable resume | `internal/sessionimage`, `internal/snapshot`, recall core images | Captures session state, not the loop's schedule, owner, run count, and remote trigger history |
| Notifications | `tools/notify.ps1`, Slack benchmark bridge, dispatch status docs | Output sinks exist, but no normalized notification/ack contract |
| Node operation | `node-macos-a` runbook, dogfood coverage, hardware matrix | No node registry that says "this loop can run here, under this policy, with this gateway" |

## The missing abstraction: a Loop Record

One durable loop record should sit above `taskmgr` and `session`:

```json
{
  "schema": "fak.loop.v1",
  "loop_id": "issue-dispatch/default",
  "owner": "fleet",
  "mode": "background",
  "trigger": {"kind": "schedule", "spec": "every 10m"},
  "target": {"kind": "local", "node": "workstation-a", "workspace": "C:/work/fak"},
  "policy": "examples/dev-agent-policy.json",
  "state": "armed",
  "next_fire_unix": 1782422400000000000,
  "last_fire_unix": 1782421800000000000,
  "runs": {
    "attempted": 411,
    "admitted": 392,
    "succeeded": 173,
    "witnessed": 167,
    "refused": 19
  },
  "last_run": {
    "run_id": "run_20260625_101500_abc123",
    "status": "witnessed_done",
    "reason": "",
    "evidence": [{"kind": "commit", "ref": "8469c56"}]
  },
  "notify": [{"sink": "windows_toast"}, {"sink": "slack:#fleet"}]
}
```

This is not a replacement for a task snapshot. `taskmgr` answers "what is this process
doing right now?" A loop record answers "what recurring logical loop exists, when did it
fire, where did it run, what happened, and who has been told?"

## Foreground, background, and green threads

The OS analogy helps if it stays precise.

| Agent loop concept | OS analogy | fak meaning |
|---|---|---|
| Foreground session | foreground process | A user-attached `fak guard -- claude` or CLI run. Terminal owns attention; exit ends the loop. |
| Background loop | daemon/service | A durable loop record with an owner, schedule, policy, budget, run ledger, and notification sinks. It can survive logout, host sleep, and a fresh process. |
| Loop tick | timer interrupt | A scheduler event from cron/launchd/Task Scheduler/K8s/GitHub/HTTP. It requests a run; fak still decides admission. |
| Agent green thread | cooperative user-space thread | A logical loop fiber multiplexed over scarce workers. It is cheap to keep as state, only binds an OS process/VM while executing a tick, and yields at turn/session boundaries. |
| Worker process/VM | kernel thread / execution context | The concrete child process, container, microVM, remote SSH session, or cloud task that runs one admitted slice. |
| Signal | notification/control event | Pause, drain, resume, speed up, reduce budget, request status, or request witness. Signals mutate loop/session state; they are not arbitrary shell commands. |

"Green thread" is the right model for the long-running fleet: keep thousands of logical
loops as small records, but run only the few admitted by leases, accounts, budgets, and
node capacity. The scheduler is cooperative because the safe yield point is a fak boundary:
before a run, between turns, at reset/drain, or after a witness check. It should never kill
mid-tool-call unless the state is already corrupt and the kill is recorded as such.

## What fak should own

### 1. Loop identity and durable run ledger

Every background loop needs an append-only, hash-chained ledger, similar in spirit to the
guard audit journal, but at loop granularity:

- `loop.armed`: schedule/control source accepted.
- `loop.fire`: cron/API/chat/GitHub event requested a run.
- `loop.admit`: policy, node, lease, budget, and cooldown admitted or refused.
- `loop.start`: execution target created.
- `loop.heartbeat`: optional progress pulse with task snapshot digest, not payload bytes.
- `loop.end`: process/session ended, with exit status.
- `loop.witness`: independent evidence accepted/refused/unavailable.
- `loop.notify`: notification sent and optionally acknowledged.

This is how fak answers "how often did the background loop actually run?" without scraping
OS scheduler logs or trusting the worker's final sentence.

### 2. Trigger adapters, not trigger trust

Cron, launchd, Windows Scheduled Tasks, systemd timers, Kubernetes CronJobs, GitHub events,
Slack, WhatsApp, phone actions, and HTTP `/fire` should all lower into the same control
event:

```json
{"event": "fire", "loop_id": "issue-dispatch/default", "source": "slack", "principal": "alice", "nonce": "..."}
```

The source authenticates the request, but does not decide the run. fak still checks:

- Is the principal allowed to fire this loop?
- Is the loop armed?
- Is the target node alive and eligible?
- Is there a lane/account/worker lease?
- Is the policy floor available?
- Is this fire inside budget and outside cooldown?

That split is the fak-shaped part. Phone and chat are convenient control panes, not new
trust roots.

### 3. Node registry and execution targets

Add one registry that says what each node can host:

```json
{
  "node_id": "node-macos-a",
  "platform": "darwin/arm64",
  "scheduler": "launchd",
  "gateway_url": "http://127.0.0.1:8080",
  "labels": {"metal": "true", "always_on": "true"},
  "capacity": {"workers": 4, "interactive": 1},
  "heartbeat_unix": 1782421800000000000
}
```

Targets should be typed, not stringly:

- `local`: run on the current host.
- `worktree`: run in a checked-out worktree, concurrency-safe but not a security boundary.
- `container`: run with shared-kernel isolation.
- `microvm`: run with a hardware isolation boundary supplied by a sandbox provider.
- `remote_ssh`: run on a named node over SSH/Tailscale.
- `cloud_task`: run in an external cloud-agent environment.

fak should not become a VM provider. It should make the boundary travel to whichever target
the operator already uses.

### 4. Notifications and acknowledgements

Notifications need a normalized contract:

- `level`: info / action / warning / critical.
- `reason`: closed vocabulary, for example `WITNESS_REFUSED`, `AUTH_REQUIRED`,
  `AT_CAP`, `LOOP_STUCK`, `POLICY_DENIED`, `DONE_WITNESSED`.
- `loop_id`, `run_id`, `trace_id`.
- `summary`: bounded text, no prompt/tool-result payload.
- `actions`: constrained verbs such as `pause`, `resume`, `drain`, `rerun`, `open_status`,
  each mapped back through policy.

Sinks can be Windows toast, macOS notification, Slack, WhatsApp Business/webhook, email,
GitHub comment, A2A task event, or a local status file. Sinks are interchangeable because
they do not carry authority by themselves. Authority is in the control event that comes
back, authenticated and re-adjudicated.

### 5. Witnessed completion by default

Long-running loops are where self-report is most tempting and least useful. A run can
finish in four distinct states:

- `claimed_done`: worker exited and said done, no witness yet.
- `witnessed_done`: independent evidence confirms the effect.
- `witness_refused`: evidence contradicts the claim.
- `witness_unavailable`: witness could not read back the effect.

This mirrors `internal/taskmgr.WitnessRecord` and the DOS discipline already used by
issue dispatch. It keeps "green status" from meaning "the worker stopped talking."

## Proposed ladder

### Rung 0 - name the product surface

Document `fak loop` as the umbrella for long-running loops:

- `fak loop status`: list loop records and latest run state.
- `fak loop fire LOOP`: request one run through the same admission path as a schedule.
- `fak loop pause|resume|drain LOOP`: write loop/session drive state.
- `fak loop runs LOOP`: print run ledger rows.
- `fak loop witness RUN`: re-run the witness for a completed run.

This can be documentation-first. The point is to stop treating each watchdog script as a
different product.

### Rung 1 - durable loop ledger

Implement the loop record and JSONL ledger with a small stdlib-only package. First host:
wrap existing issue-dispatch and dogfood loops without changing their behavior. Success
means a loop can answer:

- how many scheduled fires occurred;
- how many were admitted/refused and why;
- how many completed;
- how many were independently witnessed;
- last success, last failure, next fire, and drift.

### Rung 2 - OS scheduler adapters

Keep OS schedulers as thin drivers:

- Windows: Scheduled Tasks.
- macOS: launchd plus `caffeinate`.
- Linux: systemd timers, cron, or Kubernetes CronJob.
- GitHub Actions: external scheduled runner.

Each adapter should only fire a loop id and then record the fire result. All policy,
leases, and witnesses live above it.

### Rung 3 - control/notification bridge

Expose a network-safe control endpoint only after auth is strong enough:

- `POST /v1/fak/loops/{id}/fire`
- `POST /v1/fak/loops/{id}/signal`
- `GET /v1/fak/loops`
- `GET /v1/fak/loops/{id}/runs`

Then bind Slack/phone/WhatsApp/GitHub to those routes. The order matters: a remote control
surface before real auth is an anti-feature.

### Rung 4 - node registry and remote targets

Add node heartbeats and typed execution targets. Start with local and remote SSH/Tailscale.
Add container/microVM/cloud-task as adapters, not as fak-owned infrastructure.

Admission should be:

```text
fire -> auth -> loop policy -> node eligibility -> lease/capacity -> spawn target -> witness -> notify
```

### Rung 5 - cache-aware green-thread scheduler

Once logical loops are cheap records, schedule them with fak's differentiators:

- prefer nodes where the gateway/cache is warm for that repo/session;
- batch loops that share a prefix;
- hold or drain loops whose witness rate falls;
- lower priority for loops that are burning budget without commits/evidence;
- let urgent foreground sessions preempt background green threads at the next yield point.

This is where fak's security and reuse boundary become one operational advantage, not two
separate features.

## What not to build

- Do not build another generic cron. Use OS schedulers as interrupt sources.
- Do not build a VM/cloud-agent product. Treat VMs and cloud tasks as targets.
- Do not build a chat bot that shells out. Chat creates signed control events.
- Do not treat worktrees as a security boundary. They are a concurrency boundary.
- Do not make notifications authoritative. They are outputs; inbound actions still pass
  the same control policy.
- Do not call a run done because the agent says done. Completion needs a witness rung.

## First practical wedge

The first Rung 1 slice is shipped as of 2026-06-25:

- `internal/loopmgr` defines `fak.loop-event.v1`, an append-only SHA-256 hash-chained
  JSONL event ledger for `armed`, `fire`, `admit`, `start`, `heartbeat`, `end`,
  `witness`, and `notify` rows.
- `fak loop append --loop ID --kind KIND ...` is the canonical writer for schedulers,
  scripts, and future chat/phone bridges.
- `fak loop run --loop ID -- CMD ...` is the generic OS-scheduler wrapper: cron,
  launchd, systemd timers, and Windows Scheduled Tasks can wrap an existing command
  and get `fire`, `admit`, `start`, and `end` rows plus child exit-code/duration
  metrics without each script reimplementing the ledger.
- `tools/fak_loop_task.ps1` is the shared Windows Scheduled Task action helper:
  it probes for a usable `fak loop` binary, falls back to `go run ./cmd/fak` for
  source-tree installs, and keeps Task Scheduler's Execute and Argument fields split.
- `tools/fak_loop_run.sh` is the shared Unix scheduler shim for cron, launchd,
  and systemd: it probes for a usable `fak loop` binary, skips stale binaries that
  do not support the loop verb, falls back to `go run ./cmd/fak`, and execs the
  child command through `fak loop run`.
- `tools/register_issue_dispatch.ps1` and `tools/register_resolve_progress.ps1` now
  install the Windows `FleetIssueDispatch` and `FleetResolveProgress` Scheduled Tasks
  through `fak loop run`, using OS-edge loop ids
  `issue-resolve-dispatch/task-scheduler/<backend>` and
  `issue-resolve-progress/task-scheduler` before the Python producers emit rows.
- `tools/register_mac_watchers.sh`, `tools/com.fleet.dispatch-supervisor.plist`,
  `tools/com.fak.dogfood-fleet.plist`, and the GCP `fak-dogfood-fleet.service`
  template in `scripts/gcp-dogfood-control-vm.sh` now lower their cron/launchd/systemd
  fires through `fak loop run` before running the existing watchdog or guarded dispatch
  tick.
- `fak loop status --ledger FILE [--json]` folds that ledger into a read-only
  `fak.loop-status.v1` view with per-loop fire/admit/refusal/run/witness/notification
  counts and last-run evidence.
- `tools/issue_resolve_dispatch.py` appends dispatcher-tick rows by default:
  `fire`, `admit`, `start` for live spawns, and `end` for successful ticks. The rows
  describe the dispatcher tick, not the worker's eventual issue-resolution claim.
- `tools/issue_resolve_progress.py` appends progress/close proof rows by default:
  `fire`, admitted/refused `admit`, `end`, and a `witness` row when the GitHub open-count
  plus closure-audit read-back exists. Audit outages become `witness_unavailable`
  instead of turning a progress snapshot into a fake success.

Honest fence: this is not yet a scheduler, node registry, chat/phone bridge, or general
notification plane. `fak loop run` wraps commands and the main Windows/Mac/GCP dogfood
scheduler templates now call it, but fak still does not install or own cron/launchd/systemd
generally. Only the issue-resolution dispatcher and progress/close proof ticks are wired
as producers; worker-effect completion still belongs to the independent close/audit arm.

Remaining Rung 1 work:

1. Reuse `internal/taskmgr.WitnessRecord` vocabulary for run completion.
2. Reuse existing notification sinks only after the run row exists, so notifications are
   audit-backed.

This gives the operator the thing the current system lacks most: "what loops exist, when did
they last actually run, what did they prove, and what should page me?"

## Success metrics

The loop surface should emit these standing KPIs:

| KPI | Meaning |
|---|---|
| `loop_fire_total` | Trigger events received, by source. |
| `loop_admit_total` | Fires admitted/refused, by reason. |
| `loop_run_total` | Runs started/ended, by status. |
| `loop_witness_rate` | Witnessed done / claimed done. |
| `loop_schedule_drift_seconds` | Actual fire time minus intended fire time. |
| `loop_stuck_age_seconds` | Oldest running run without heartbeat or evidence. |
| `loop_notify_total` | Notifications sent/acked, by sink and reason. |
| `loop_cache_reuse_ratio` | Cache/prefix reuse for loops sharing a gateway. |
| `loop_budget_burn_without_evidence` | Tokens/time spent before first independent effect. |

These are the loop-level counterparts to existing gateway and task-manager surfaces. They
turn long-running automation from "a script probably ran" into a governed kernel object.

## Verdict

fak is positioned well precisely because unattended loops widen the distance between a human
and an action. The more a run moves into cloud VMs, other nodes, cron, phones, Slack, or
WhatsApp, the more valuable a structural kernel boundary becomes.

The concrete gap is not another planner. It is a loop kernel surface: durable loop identity,
trigger normalization, node/target admission, task/resource snapshots, witnessed completion,
bounded notifications, and cache-aware scheduling. Build that and the existing scripts stop
being one-off automation; they become user-space drivers for one governed agent OS.
