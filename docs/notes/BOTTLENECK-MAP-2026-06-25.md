---
title: "Bottleneck map and durable process loops - 2026-06-25"
description: "Evidence-backed snapshot of the current system bottlenecks, GitHub/open-work bottlenecks, and the durable skills/loops that should keep both visible."
---

# Bottleneck map and durable process loops - 2026-06-25

This is the current map for the objective: identify the system bottlenecks and
the GitHub/open-work bottlenecks, then make the check repeatable as part of the
ongoing process.

## Evidence gathered

Commands run from the repo root on 2026-06-25:

```bash
python tools/fleet_bottleneck.py report --no-audit
python tools/fleet_bottleneck.py json --no-audit --out tools/_registry/bottleneck-map-latest.json
python tools/issue_triage.py --markdown --out docs/_audits/issue-triage-2026-06-25.md
python tools/issue_triage.py --actions --out docs/_audits/issue-actions-2026-06-25.json
gh issue list --state open --label "priority/P0" --limit 50 --json number,title,labels,assignees,updatedAt,url
python tools/fleet_bottleneck.py report --no-audit
python tools/issue_triage.py --json
```

The `docs/_audits/` and `tools/_registry/` files are gitignored process output.
This tracked note keeps the durable summary.

Future passes should treat the `fleet_bottleneck.py` CLI as a `tools/_registry`
refresh: run it only when the `tools` lane/control-pane loop owns that mutation.
When `tools` is held, import `collect(audit=False)` and `rank_bottlenecks()`
read-only and record the summary here.

Publication rule: this tracked note is public-repo material. Keep GitHub issue
numbers/titles/labels and aggregate fleet counts here, but do not publish account
directory names, auth details, session IDs, reset timestamps, private host paths,
or token/cost details. Those belong in ignored operator-local artifacts.

## System bottleneck map

`fleet_bottleneck.py` reported fleet health as **CRITICAL** at
`2026-06-25T09:01:36Z`. The registry was fresh enough for this pass
(about 6 minutes old, 10 hour window).

| Rank | Bottleneck | Evidence | Durable move |
|---:|---|---|---|
| 1 | Rate-limit / throttle saturation | 39/90 in-flight workers are on rate-limited accounts; 5/8 accounts throttled; 46 stopped on a limit. | Pause/cap dispatch to throttled accounts; spread only to real free accounts. |
| 2 | Auth-blocked workers | 5 sessions blocked on authentication. | Re-login the owning accounts before treating capacity as available. |
| 3 | Crash-resume backlog | 12 workers queued for resume; watchdog activity not seen on disk. | Verify the resume watchdog is scheduled/live and process the resume queue. |
| 4 | Recovery plumbing stale | No recovery-watchdog activity on disk while 75 workers need attention. | Fix supervisor/watchdog liveness before launching more work. |
| 5 | Account load imbalance | 50% of active workers are on one account; even split would be 12% across 8 accounts. | Rebalance dispatch after throttles/auth clear. |
| 6 | Stalled capacity | 6/90 active workers parked or ambiguous, average age about 7.5h. | Inspect parked workers; resume real waits and close stuck sessions. |

Interpretation: the actual system limiter is the operating loop, not a missing
code feature. Adding workers before throttles/auth/recovery are handled increases
pressure on the constrained path and hides the real bottleneck.

### Refresh at 2026-06-25T09:07:47Z

A read-only in-process refresh of `fleet_bottleneck.collect(audit=False)` still
reports **CRITICAL** health. Some pressure shifted: throttled accounts fell from
5/8 to 3/8 and only 6/94 in-flight workers are now on throttled accounts, but
the CRITICAL rows remain because 45 workers are stopped on limits, 5 sessions
are auth-blocked, 12 workers are still queued for auto-resume, and recovery
watchdog activity is still absent. A new high row appeared: 17 interactive
crashed/stopped workers need human surfacing (`surface_backlog`, score 60).

Updated interpretation: recovery/auth remains the first limiter. The backlog
can improve in parallel, but broad dispatch should stay capped until recovery
is live and the SURFACE/AUTO_RESUME rows are processed.

### Refresh at 2026-06-25T09:18:38Z

`fleet_bottleneck.collect(audit=False)` plus `rank_bottlenecks()` still reports
**CRITICAL** health. The read-only import path was used, so this refresh did not
write `tools/_registry`. Slots: active 93, live 10, supervised 3, hanging 6,
auto-resume 10, surface 15, auth-blocked 5, workers-on-throttled 15,
rate-limited/stopped 45, deferred throttle 42, done 210.

The ranked limiter is unchanged:

| Rank | Bottleneck | Evidence |
|---:|---|---|
| 1 | Rate-limit / throttle saturation | 15/93 in-flight workers on throttled accounts; 4/8 accounts throttled; 45 stopped on a limit. |
| 2 | Auth-blocked workers | 5 sessions blocked on expired auth/login. |
| 3 | Crash-resume backlog | 10 workers queued for auto-resume; watchdog activity still absent. |
| 4 | Recovery plumbing stale | 58 workers need attention and no recovery-watchdog activity was found. |
| 5 | Dead-crash surfacing backlog | 15 interactive crashed/stopped workers need human review. |
| 6 | Account load imbalance | 48% of active workers are on one account. |
| 7 | Stalled / hanging capacity | 6/93 active workers parked/ambiguous, average age about 4.5h. |

Interpretation: do not increase broad dispatch. Account throttles and auth
failures are transient dispatch gates: they matter for whether to spawn now, but
they should be rechecked after reset/relogin before becoming strategic debt.
Watchdog absence, auto-resume backlog, and surfacing backlog are semi-durable
recovery debt and remain the system-process fix. The structural open-work problem
below remains even after the transient gate clears.

## GitHub/open-work bottleneck map

The issue triage snapshot found:

| Signal | Count |
|---|---:|
| Open issues | 400 |
| Needs priority | 245 |
| Needs area | 217 |
| Needs kind | 21 |
| Orphan P0/P1 | 70 |
| Stale | 0 |
| Dormant question | 0 |
| Likely duplicate | 7 |
| Bare | 4 |

The latest read-only `issue_triage.py --json` refresh showed progress but did
not change the bottleneck class: open issues stayed at 397, needs-priority stayed
at 242, needs-area fell to 173, needs-kind fell to 19, and orphan P0/P1 stayed
at 70.

The action manifest now contains 316 actions, all review-only and zero
mechanical actions. That means stale cleanup is not the backlog bottleneck
today. The bottleneck is judgment and ownership: missing taxonomy labels plus
unclaimed high-priority work.

Current P0 rows are all unassigned/orphaned:

| Issue | Area implied by labels/title | Why it matters |
|---|---|---|
| #295 `feat(gpu): Multi-GPU Tensor Parallelism [A-007]` | GPU/model performance | Hardware-gated execution parity; already has partial-state comments, still not complete. |
| #45 `feat(gateway): fleet router skeleton` | Agentic serving/gateway | Dispatch substrate for multiple upstreams. |
| #298 `feat(model): Production Llama 3.x Checkpoints [A-004]` | Model support | Production checkpoint support. |
| #294 `feat(model): HuggingFace Hub Direct Loading [A-008]` | Loader/model support | Repo-level HF resolution remains after single-file support. |
| #289 `perf(prefill): Close Prefill Throughput Gap [B-001]` | CUDA/GPU performance | Throughput parity path. |
| #287 `perf(vulkan): Vulkan Backend Optimization [B-002]` | Vulkan/GPU performance | AMD backend performance path. |
| #255 `feat(benchmark): N=100-1000 Agent Benchmark [D-001]` | Agentic serving/benchmark | Fleet-scale comparison still lacks cross-framework/task-success axis. |
| #50 `epic(serving): large-scale disaggregated serving` | Serving substrate | Portfolio epic; sequencing needs active ownership. |

The top P1 rows are also unassigned, starting with #440, #438, #430, #399,
#388, #78, #71, #70, #69, and #67. The open-work limiter is therefore not just
P0 scarcity; it is the absence of an ownership/defer pass across high-priority
work.

## Scope, Horizon, And Weight

| Surface | Evidence scope | Horizon | Weight |
|---|---|---|---|
| Rate limits / auth blockers | Operator-private; publish aggregate counts only. | Transient, unless repeated across passes after reset/relogin. | Dispatch-gating now; not the highest strategic problem by itself. |
| Recovery watchdog / auto-resume / surfacing | Operator-private; publish aggregate counts only. | Semi-durable operating-loop debt. | Process-debt: fix before broad dispatch. |
| GitHub issue taxonomy | Public issue tracker. | Structural until labels are added or issue policy changes. | Strategic backlog debt. |
| Orphan P0/P1 ownership | Public issue tracker. | Structural until claimed, deferred, or closed by witnessed commits. | Strategic backlog debt and the top durable open-work limiter. |

## Durable skills and loops

The durable front door is now `/bottleneck-map`:

- It runs `fleet_bottleneck.py` and `issue_triage.py` together.
- It records system capacity/recovery constraints before recommending dispatch.
- It routes the next action into `/issue-triage`, issue dispatch, or recovery
  maintenance instead of letting each pass rediscover the same state.

The standing runbook is `docs/bottleneck-map-loop.md`.

Use the loops in this order:

1. **Bottleneck map:** run `/bottleneck-map` at the start of a dispatch window
   and after any fleet stall. If system health is CRITICAL, fix capacity/recovery
   first.
2. **Issue triage:** run `/issue-triage --scope priority` or `--scope area` to
   cut the 242 / 173 / 19 taxonomy gaps. Apply no GitHub write without explicit
   operator approval.
3. **Ownership pass:** claim or explicitly defer the orphan P0/P1 rows before
   launching broad work. Today the floor is 70 orphan P0/P1 rows.
4. **Dispatch:** only after the system bottleneck is below critical, dispatch one
   issue per worker and require `#N` commit binding plus `dos commit-audit`.
5. **Learning refresh:** keep the dated note and skill text updated when a pass
   reveals a repeated blocker. Repeated blockers should become a small process
   issue or a helper/skill change, not another one-off reminder.

The standing-process handoff is `tools/control_pane.loops.json`, whose catalog
is maintained with `python tools/fleet_control_pane.py loop-add ... --apply`.
That file is in the `tools` lane; during this pass the `tools` lane was already
held by live work, so the durable hook was deliberately not edited. When the
lane is free, add a read-only `bottleneck-map` loop that reports ACTION when
fleet health is CRITICAL/HIGH, orphan P0/P1 is nonzero, or the label-debt counts
regress above the last baseline.

## Next-pass done conditions

The next useful pass should prove at least one of these with fresh output:

- fleet health is below CRITICAL, or the transient throttle/auth gate has been
  rechecked after reset/relogin and the semi-durable recovery rows have concrete
  operator action recorded;
- `needs_priority`, `needs_area`, or `needs_kind` falls below the latest
  baseline of 242 / 173 / 19;
- orphan P0/P1 falls below 70 by explicit claim or explicit defer;
- a top P0/P1 issue is resolved by a `#N`-cited commit and witnessed through
  `dos commit-audit`;
- the `bottleneck-map` control-pane loop is added when the tools lane is safe;
- `/bottleneck-map` is run again and its note updates this baseline rather than
  relying on memory.
