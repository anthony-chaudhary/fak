# Recurring loops inventory

_Generated 2026-07-11T15:59:16Z by `tools/loops_inventory.py` — a read-only fold over `tools/register_*.ps1` (OS Scheduled Tasks) and `.github/workflows/*.yml` (cron). Do not hand-edit; re-run the tool._

**52 loops**: 24 scheduled tasks · 28 cron workflows. Reporting to (inferred): 21 → Slack, 4 → GitHub issues, 1 → repo doc, 1 → operator-local.

## OS Scheduled Tasks (operator-local, Windows)

| Task | Cadence | Reports to | Runs | Purpose | Source |
| --- | --- | --- | --- | --- | --- |
| FleetControlPaneTick | every 1 min | action | tools/fleet_control_pane.py | register_control_pane_tick.ps1 - install/remove the one Scheduled Task that runs the portable fleet control pane tick | `tools/register_control_pane_tick.ps1` |
| FleetDOSDispatchWatchdog | every 5 min | action | — | install/remove the OS-level Scheduled Task that runs fleet_dos_dispatch_watchdog.ps1 every 5 minutes, so FLEET's own generic-DOS dispatch supervisor (`dos loop --enact --target N`) is kept alive forev | `tools/register_dos_dispatch_watchdog.ps1` |
| FleetIssueDispatch | every 5 min | action | tools/dispatch_status.py | install/remove the OS Scheduled Task that keeps the DoS-SAFE issue dispatcher always-on | `tools/register_issue_dispatch.ps1` |
| FleetRunawayReaper | every 5 min | action | — | register_runaway_reaper.ps1 - install/remove the Scheduled Task that runs the runaway search-process reaper every 5 min (kills Git-Bash find/grep that have run away into /proc/registry or /mnt/c junct | `tools/register_runaway_reaper.ps1` |
| FleetSupervisorWatchdog | every 5 min | action | — | install/remove the OS-level Scheduled Task that runs fleet_supervisor_watchdog.ps1 every 5 minutes (plus at logon), so the job-fleet supervisor is kept alive forever with zero human intervention | `tools/register_supervisor_watchdog.ps1` |
| FleetProcResourceGuard | every 10 min | action | — | register_proc_resource_guard.ps1 - install/remove the Scheduled Task that runs the process-resource guard on a standing interval, so the host has a durable watch for the runaway classes the level watc | `tools/register_proc_resource_guard.ps1` |
| FleetResumeWatchdog | every 10 min | action | tools/fleet_resume_watchdog.ps1 | register_resume_watchdog.ps1 - install/remove the Scheduled Task that runs the cross-account resume watchdog every 10 min (refreshes the on-disk session registry each tick = "extract in advance", and  | `tools/register_resume_watchdog.ps1` |
| FleetPushLagPusher | every 15 min | action | tools/auto_push_on_lag.py | install/remove the OS Scheduled Task that runs the push-lag BACKSTOP on a cadence: tools/auto_push_on_lag.py checks fresh_status's git pane and, only when the oldest unpushed commit has been waiting p | `tools/register_push_lag_pusher.ps1` |
| FleetResolveProgress | every 15 min | github | tools/issue_resolve_progress.py | install/remove the OS Scheduled Task that runs the issue-resolve PROGRESS + CLOSE tick on a cadence (the harvesting watchdog) | `tools/register_resolve_progress.ps1` |
| FakSessionCheckpoint | every 20 min | ? | tools/session_checkpoint.py | install/remove the OS Scheduled Task that writes a durable, off-host SESSION WORK-STATUS CHECKPOINT to GitHub every few minutes | `tools/register_session_checkpoint.ps1` |
| FleetDispatchStatusDoc | every 30 min | local-doc | tools/dispatch_status.py | install/remove the OS Scheduled Task that keeps the operator-local issue-dispatch STATUS DOC fresh (gitignored .dispatch-runs/dispatch-status.md) | `tools/register_dispatch_status_doc.ps1` |
| FleetSession0Sweep | every 30 min | action | tools/session0_orphan_sweep.py | register_session0_sweep.ps1 - install/remove the Scheduled Task that runs the elevated Session-0 orphan sweep (issue #2338) | `tools/register_session0_sweep.ps1` |
| FleetSlackStatus | every 30 min | slack | tools/fleet_slack_status.py | install/remove the OS Scheduled Task that posts the WHOLE fleet status to Slack on a cadence: the dispatch_status card (dispatcher + supervisor + watchdog-installed + backlog + closure + throughput) A | `tools/register_fleet_slack_status.ps1` |
| FleetSlackBeat | every 3h | slack | — | install/remove the OS Scheduled Task that posts the Slack LIVENESS BEAT on a cadence (#1426, epic #1425): `fak slack beat` folds the Slack-surface health (resolution + auth + a real conversations.hist | `tools/register_slack_beat.ps1` |
| FleetWorktreeDoctor | every 4h (+ daily 03:30) | action | tools/worktree_doctor.py | register_worktree_doctor.ps1 - install/remove/status the Scheduled Task that runs tools/worktree_doctor.py on a cadence, so this box stays at "one worktree on the trunk" (auto-detected, e.g | `tools/register_worktree_doctor.ps1` |
| FleetGrafanaRollup | every 6h | slack | — | install/remove the OS Scheduled Task that posts the #grafana dashboard/debug-link ROLLUP to Slack on a cadence, so the channel stays populated without a manual `fak grafana post` | `tools/register_grafana_rollup.ps1` |
| FleetStaleWorkWatchdog | every 6h | action | tools/stale_work_watchdog.py | register_stale_work_watchdog.ps1 - install/remove the Scheduled Task that runs the stale-work watchdog (tools/stale_work_watchdog.py) on a cadence: it GCs this clone's own gitignored per-session ephem | `tools/register_stale_work_watchdog.ps1` |
| FleetBenchPlanDoc | every 12h | repo-doc | tools/bench_plan_tick.ps1 | install/remove the OS Scheduled Task that keeps the committed hardware bench-plan doc fresh (docs/bench-plan.md) | `tools/register_bench_plan_doc.ps1` |
| FleetLoopsInventory | every 12h | slack | tools/loops_inventory.py | install/remove the OS Scheduled Task that keeps the committed recurring-loops inventory fresh (docs/loops-inventory.md) and, opt-in, posts a compact rollup to Slack | `tools/register_loops_inventory.ps1` |
| FleetDispatchLogAudit | daily 09:30 | github | tools/dispatch_log_audit.py | install/remove the OS Scheduled Task that runs the daily dispatch-log-audit (tools/dispatch_log_audit.py): scan .dispatch-runs/ worker logs, classify failure signatures (panic/traceback, hook-failure  | `tools/register_dispatch_log_audit.ps1` |
| FleetDispatchSessionAudit | daily 09:50 | github | tools/dispatch_log_audit.py | install/remove the OS Scheduled Task that runs the daily dispatch SESSION audit (`fak dispatch audit`): fold the .dispatch-runs/ worker sessions into a per-worker OUTCOME classification (SHIPPED / WAS | `tools/register_dispatch_session_audit.ps1` |
| FleetIdeaScout | daily 09:00 | github | tools/idea_scout.py | install/remove the OS Scheduled Task that runs the daily idea-scout (tools/idea_scout.py): search arXiv + GitHub for ideas RELATED to fak, and file the genuinely-new, on-topic hits as GitHub issues -- | `tools/register_idea_scout.ps1` |
| FleetScoutLoop | daily 10:30 | ? | tools/launch_goal_detached.ps1 | install/remove the OS Scheduled Task that runs the `scout-loop` research->backlog loop (.claude/skills/scout-loop/SKILL.md) on a daily cadence | `tools/register_scout_loop.ps1` |
| LearningDocsFreshness | daily | action | tools/learning_scorecard.py | install/remove the OS Scheduled Task that runs the durable learning-docs freshness loop | `tools/register_learning_docs_freshness.ps1` |

## GitHub Actions (cron)

| Workflow | Cadence | Reports to | Purpose | Source |
| --- | --- | --- | --- | --- |
| Release cadence | every 2h | ci | Release cadence | `.github/workflows/release-cadence.yml` |
| issue-lane-router | every 6h | ci | issue-lane-router | `.github/workflows/issue-lane-router.yml` |
| slack-beat | every 6h | slack | slack-beat | `.github/workflows/slack-beat.yml` |
| bench-feed | daily 08:27 UTC | slack | bench-feed | `.github/workflows/bench-feed.yml` |
| bench-signal | daily 08:03 UTC | slack | bench-signal | `.github/workflows/bench-signal.yml` |
| blockers-feed | daily 09:07 UTC | slack | blockers-feed | `.github/workflows/blockers-feed.yml` |
| cachevalue-feed | daily 09:37 UTC | slack | cachevalue-feed | `.github/workflows/cachevalue-feed.yml` |
| cachevalue-gate | daily 09:53 UTC | ci | cachevalue-gate | `.github/workflows/cachevalue-gate.yml` |
| capacity-feed | daily 08:47 UTC | slack | capacity-feed | `.github/workflows/capacity-feed.yml` |
| dispatch-session-audit-feed | daily 09:43 UTC | slack | dispatch-session-audit-feed | `.github/workflows/dispatch-session-audit-feed.yml` |
| dogfood | daily 06:41 UTC | ci | dogfood | `.github/workflows/dogfood.yml` |
| dogfood-coverage | daily 07:13 UTC | ci | dogfood-coverage | `.github/workflows/dogfood-coverage.yml` |
| dojo-feed | daily 08:37 UTC | slack | dojo-feed | `.github/workflows/dojo-feed.yml` |
| dojo-rsi-feed | daily 08:43 UTC | slack | dojo-rsi-feed | `.github/workflows/dojo-rsi-feed.yml` |
| garden | daily 06:23 UTC | ci | garden | `.github/workflows/garden.yml` |
| gate-signal | daily 07:53 UTC | slack | gate-signal | `.github/workflows/gate-signal.yml` |
| node-usage-feed | daily 08:57 UTC | slack | node-usage-feed | `.github/workflows/node-usage-feed.yml` |
| score-signal | daily 07:41 UTC | slack | score-signal | `.github/workflows/score-signal.yml` |
| scoreboard-feed | daily 08:17 UTC | slack | scoreboard-feed | `.github/workflows/scoreboard-feed.yml` |
| slack-watchdog | daily 10:19 UTC | slack | slack-watchdog | `.github/workflows/slack-watchdog.yml` |
| steering-guard | daily 07:33 UTC | ci | steering-guard | `.github/workflows/steering-guard.yml` |
| trajctl-signal | daily 09:13 UTC | slack | trajctl-signal | `.github/workflows/trajctl-signal.yml` |
| backlog-feed | weekly 09:11 UTC | slack | backlog-feed | `.github/workflows/backlog-feed.yml` |
| cachevalue-weekly | weekly 09:47 UTC | ci | cachevalue-weekly | `.github/workflows/cachevalue-weekly.yml` |
| cadence report | weekly 08:17 UTC | ci | cadence report | `.github/workflows/cadence.yml` |
| fanout-cadence | weekly 09:47 UTC | ci | fanout-cadence | `.github/workflows/fanout-cadence.yml` |
| product-feed | weekly 09:23 UTC | slack | product-feed | `.github/workflows/product-feed.yml` |
| security-audit | weekly 06:17 UTC | ci | security-audit | `.github/workflows/security-audit.yml` |

_"Reports to" is inferred from each loop's declaration text; the Source column is the ground truth._
