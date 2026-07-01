---
title: "Bottleneck map and durable process loops - 2026-07-01"
description: "Evidence-backed snapshot of the current system bottlenecks, GitHub/open-work bottlenecks, and the durable skills/loops that should keep both visible."
---

# Bottleneck map and durable process loops - 2026-07-01

Current map for the objective: name the actual limiter right now, the open work
blocked behind it, and the durable loop that should run next.

## Evidence gathered

Commands run from the repo root on 2026-07-01 (fleet snapshot at
`2026-07-01T15:11:59Z`):

```bash
# read-only import (the tools lane was not confirmed free, so the CLI that
# mutates tools/_registry was NOT run):
python -  # fleet_bottleneck.collect(audit=False) + rank_bottlenecks()
python tools/issue_triage.py --markdown --out docs/_audits/issue-triage-2026-07-01.md
python tools/issue_triage.py --actions  --out docs/_audits/issue-actions-2026-07-01.json
```

`docs/_audits/` is gitignored process output; this tracked note keeps the durable
summary. Publication rule honored: aggregate fleet counts and public issue
numbers/titles/labels only — no account directory names, session IDs, auth
details, reset timestamps, or private host paths.

## System bottleneck map

`fleet_bottleneck` reported fleet health **CRITICAL** at `2026-07-01T15:11:59Z`.
Slots: active 73, live 19, done 96, hanging 5, dead 7, auto-resume 8, surface 10,
api-error 2, auth-blocked **0**, rate-limited/stopped **0**.

| Rank | Bottleneck | Score/Sev | Evidence | Durable move |
|---:|---|---|---|---|
| 1 | Crash-resume backlog | 100 / CRITICAL | 8 workers queued AUTO_RESUME, not yet recovered. Watchdog **is** live (last ran ~1m ago). | Confirm the watchdog is in LIVE mode (not DRY-RUN) and that resume_plan.json is actually draining — it is fresh but the queue is not clearing. |
| 2 | Dead-crash surfacing backlog | 60 / HIGH | 10 crashed/stopped **interactive** workers flagged SURFACE — not auto-resumable. | Triage with `python tools/fleet_sessions.py resume` (SURFACE block); resume keepers, close the rest. |
| 3 | API-error stalls | 24 / LOW | 2 workers stopped on a transient API/transport error. | Usually self-clears on retry; watch for a climbing count before adding load. |
| 4 | Stalled / hanging capacity | 23.6 / LOW | 5 of 73 active parked/ambiguous (avg ~3h). | Inspect parked workers; resume real waits, close stuck ones. |
| 5 | Rate-limit / throttle saturation | 1.6 / OK | 1/5 accounts throttled (a since-deleted account); **0** in-flight workers on it. | No action — not a live gate this pass. |

Interpretation: the limiter is **Recovery-layer**, not account pressure. This is
the key change from the 2026-06-25 baseline, where the top rows were
throttle-saturation + auth-blocked (transient account pressure). Those transient
gates have **cleared** (auth-blocked 0, rate-limited 0, throttle OK). What remains
is the semi-durable recovery debt: 18 workers (8 AUTO_RESUME + 10 SURFACE) are
lost throughput. The watchdog is now running (it was absent on 06-25), so the fix
is no longer "start the watchdog" but "prove it resumes in LIVE mode and the
queue drains."

## GitHub / open-work bottleneck map

| Signal | Count (2026-07-01) | vs 2026-06-25 baseline |
|---|---:|---|
| Open issues | 389 | 400 (↓) |
| Needs priority | 238 | 242 (≈) |
| Needs area | 203 | 173 (↑ regressed) |
| Needs kind | 77 | 19 (↑↑ regressed) |
| Orphan P0/P1 | 81 | 70 (↑ regressed) |
| Stale | 0 | 0 |
| Dormant question | 0 | 0 |
| Likely duplicate | 10 | 7 |
| Bare | 7 | 4 |

The action manifest holds **0 mechanical actions** — every row is review-only
(`cmd: null`). Stale cleanup is therefore **not** the backlog bottleneck; the
bottleneck is judgment/ownership.

The needs-kind (19→77), needs-area, orphan (70→81), and bare regressions trace to
a large recent filing burst of unlabeled/orphan epics filed idle-0d: the `relay`
perpetual-sessions family (#1860, #1866–#1910), the `generation(*)` set
(#1625, #1648–#1677), the `cache-default[01–50]` wall (#1519–#1568), and the
harness-profile children (#1951–#1957). They are well-formed but arrived without
priority/kind/area or an owner.

Top orphan P0 (both idle 1d, unassigned):

| Issue | Why it matters |
|---|---|
| #897 `epic(terminal-bench): reach rank 1 on Terminal-Bench 2.1` | Flagship benchmark epic; no owner, no area. |
| #50 `epic(serving): large-scale disaggregated serving (RIDE + NATIVE)` | Portfolio serving substrate; sequencing needs active ownership. |

## Scope, horizon, and weight

| Surface | Evidence scope | Horizon | Weight |
|---|---|---|---|
| Crash-resume + surfacing backlog | Operator-private; aggregate counts only. | Semi-durable operating-loop debt. | Process-debt: dispatch-gating **now**, fix before broad dispatch. |
| API-error / stalled capacity | Operator-private; aggregate counts only. | Transient. | Advisory; recheck next pass. |
| Throttle saturation | Operator-private; aggregate counts only. | Transient / already cleared. | Not gating this pass. |
| Issue taxonomy (priority/kind/area) | Public tracker. | Structural until labeled. | Strategic backlog debt (regressing). |
| Orphan P0/P1 ownership | Public tracker. | Structural until claimed/deferred. | Strategic; top durable open-work limiter. |

## Durable loops — the two outputs (allowed to differ)

**Stop/continue dispatch now → CAP broad dispatch.** Health is CRITICAL from the
Recovery layer. Do not launch more workers as the first move: 18 workers are
already stuck in resume/surface. First move is recovery, not capacity.

**What durable loop should improve next → the recovery loop.**
1. Confirm the resume watchdog is in LIVE mode and `resume_plan.json` is draining
   (it is fresh but the 8-deep AUTO_RESUME queue is not clearing — LIVE-vs-DRY-RUN
   is the first thing to check).
2. Triage the 10 SURFACE interactive crashes: `python tools/fleet_sessions.py
   resume`, resume keepers, close the rest.
3. Only after recovery is draining and health is below CRITICAL: run
   `/issue-triage --scope kind` then `--scope priority` to reverse the taxonomy
   regression (needs-kind 19→77 is the sharpest), and claim-or-defer the top
   orphan P0s (#897, #50). No GitHub write without operator approval — 0 mechanical
   actions means there is nothing to auto-close.

## Step 5 handoff — standing loop still pending

`tools/control_pane.loops.json` has **no** `bottleneck-map` loop registered
(grep: no match). The `tools` lane was not confirmed free this pass, so the hook
was deliberately not added. When the lane is safe, register a read-only loop that
reports ACTION when fleet health is CRITICAL/HIGH, orphan P0/P1 is nonzero, or the
taxonomy counts regress above baseline:

```bash
python tools/fleet_control_pane.py loop-add bottleneck-map --scope repo \
  --status-cmd "python tools/fleet_bottleneck.py report --no-audit" --apply
```

## Next-pass done conditions

The next pass should prove at least one with fresh output:

- fleet health below CRITICAL, or the AUTO_RESUME (8) and SURFACE (10) counts
  fall with concrete operator action recorded (watchdog LIVE-mode confirmed);
- needs-kind falls back below ~30 (reverse the 19→77 regression);
- orphan P0/P1 falls below 70 by explicit claim or defer;
- a top orphan P0 (#897 or #50) gets an owner or an explicit defer;
- the `bottleneck-map` control-pane loop is registered once the tools lane is safe.
