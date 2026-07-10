---
title: "Bottleneck map — 2026-07-09"
description: "Bottleneck map: the CRITICAL health is transient account-throttle churn, not a broken watchdog; the durable limiter is the MaxPerTick=4 resume cap."
---

# Bottleneck map — 2026-07-09

Goal context: drive throughput of the issue-completing loops; find the limiter.

## 1. Evidence gathered

- `tools/fleet_bottleneck.py` imported read-only (the `tools` lane was dirty, so
  the CLI was not run) — `collect(audit=False)` + `rank_bottlenecks()` at
  `2026-07-09T22:04Z`. Raw fold: `docs/_audits/bottleneck-map-2026-07-09-probe.json`
  (gitignored).
- Resume watchdog doctor: `register_resume_watchdog.ps1 -Action status` /
  `-Action assert-live`.
- Watchdog log tail: `%LOCALAPPDATA%\Fleet\watchdog\resume_watchdog.log`.
- `tools/fleet_sessions.py resume` (AUTO_RESUME + SURFACE blocks).

## 2. System bottleneck map (ranked)

| # | Bottleneck | Layer | Score | Sev |
|---|---|---|---|---|
| 1 | Crash-resume backlog (13 AUTO_RESUME queued) | Recovery | 100 | CRITICAL |
| 2 | Dead-crash surfacing backlog (13 SURFACE) | Recovery | 60 | HIGH |
| 3 | API-error stalls (6 STOPPED_APIERR) | Provider | 60 | HIGH |
| 4 | Rate-limit / throttle saturation (3/8 accts) | Account ceiling | 43.5 | MED |
| 5 | Account load imbalance (25% on july10) | Dispatch | 15.7 | LOW |

Health: **CRITICAL**. Slot counts: live=24, done=57, dead=11, auto_resume=13,
surface=13, rate_limited=5, api_error=6.

## 3. What the headline actually is (not a watchdog break)

The resume watchdog is **healthy**: `State=Running mode=LIVE`, 10-min S4U tick,
`assert-live OK`, `resume_plan.json` 1 min fresh. The log shows it resuming +
RE-HOME-ing onto healthy accounts every tick and pruning closed rows; the plan
drained 40 → 15 over four ticks. Old sids that looked "stuck" (64m, 118m) were in
fact resumed (`SKIP … resume took`).

So the CRITICAL is **transient account-pressure churn**, not a broken recovery
layer: deaths are dominated by STOPPED_LIMIT (3 accounts throttled —
july14/july12/day26NEW, all reset **3:50pm PT**) and STOPPED_APIERR (transient
upstream). A healthy watchdog recovers them at the per-tick cap.

## 4. The durable limiter, if it persists past reset

`MaxPerTick=4` at a 10-min tick ⇒ ~**24 resumes/hr**, and "per-tick cap reached (4)"
fires every tick — the cap, not the source governor, is binding. When the death
rate exceeds 24/hr the backlog holds at 25–40 and dead workers idle several ticks.

**Do NOT raise `MaxPerTick` during a throttle window.** Only 5/8 accounts are
healthy; resuming faster onto them pushes *those* toward their own limits — a
STOPPED_LIMIT feedback loop. Revisit the cap only if the backlog stays > ~20
*after* the 3:50pm reset clears the throttled accounts.

## 5. Scope / horizon / weight

- Evidence: mixed. Public = the throughput mechanism + counts (this note).
  Operator-private = account names / reset times (kept coarse here).
- Horizon: **transient** (account ceiling, resets 3:50pm) sitting on a
  **semi-durable** recovery-capacity question (cap vs death rate).
- Weight: dispatch-gating now; becomes process-debt only if the backlog is still
  > 20 next pass after reset.

## 6. Decision

- **Dispatch now:** continue, but do **not** add workers or raise the resume cap
  while 3 accounts are throttled (skill rule: CRITICAL from transient account
  pressure ⇒ cap, don't add load).
- **Next durable loop:** issue-dispatch / complete ready leaves — add to `done`
  without adding recovery load. The watchdog needs no intervention.

## 7. Next-pass done condition

After 3:50pm PT reset: `auto_resume` < 8 **and** `resume_plan.json` plan < 10 **and**
throttled accounts = 0. If `auto_resume` is still ≥ 20 with 0 throttled accounts,
*then* the cap is the real limiter — raise `MaxPerTick` (e.g. 4→8) and recheck.
