---
title: "fak loop scorecard"
description: "Whether fak's always-on agentic background loops are first-class durable processes — they survive a system restart (auto-restart), self-report their own status, and run through fak's own tooling — re-derived from the loop ledger + job registry fak writes, never a self-report."
---

# fak loop scorecard

**loopscore_debt: 4**; composite **33/100 (F)**; durability 50/100; self-report 0/100; dogfood 50/100

> loop-score carries 4 debt (durability 50/100, self-report 0/100, dogfood 50/100, composite 33 F): firing_loops_registered, no_dark_loop, outcome_recorded, guard_wrapped

The question: are fak's always-on background loops — the issue dispatchers, the resolve-progress tracker, the freshness cadences, the smoke loops — *first-class durable processes*, or fire-and-forget scripts that vanish on the next reboot and report nothing? Every number is re-derived from the loop ledger (`.fak/loops.jsonl`) and the job registry (`tools/loop-registry.json`) fak's own `fak loop` tooling writes, folded with the same `loopmgr` projection `fak loop status` / `fak loop health` use — so the score moves only when the loops actually become more durable, observable, and fak-native.

## Orientation

Audience: operators and agents responsible for keeping background fak loops durable across
restarts. Prereq: read [`../dispatch-loop.md`](../dispatch-loop.md) for the dispatch-loop
contract and [`../../AGENTS.md`](../../AGENTS.md) for the shared-trunk recovery rules. TL;DR:
this scorecard tells you which loops are registered, observable, and routed through fak's own
guard/witness path; after running it, fix the named loop gap before claiming loop health.

## Durability — auto-restart on system restart

| ok | criterion | detail |
|---|---|---|
| no | every loop that actually fires is registered with a cadence, so the OS scheduler re-arms it after a reboot | 0/5 (0%) firing loops are registered |
| yes | registered jobs are armed (a stopped/disabled job will NOT re-fire after a restart) | 2/2 (100%) registered jobs armed |
| yes | registered jobs carry a cadence `fak cron emit` can project to a launchd/systemd/task-scheduler unit | 2/2 (100%) cron-emittable |

## Self-report — does each loop surface its own status?

| ok | criterion | detail |
|---|---|---|
| no | no active loop is dark — every registered/firing loop is observably ticking, not silent | 7 dark loop(s) across 7 active loop(s) |
| no | loops that fire also record an end outcome in the ledger (not fire-and-forget) | 17/30 (57%) fires recorded an end |
| no | loops self-report liveness via heartbeat/notify events, not just fire/end | 0/7 loop(s) emit a heartbeat or notify self-report |

## Dogfood — does the loop run THROUGH fak's tooling?

| ok | criterion | detail |
|---|---|---|
| no | loop runs route through `fak guard` (guard_enabled=1) — the containment wrapper, not raw exec | 0/2 (0%) runs guard-wrapped |
| yes | the loops append to the canonical hash-chained loop ledger (fak loop), not an ad-hoc log | ledger present with events |
| yes | loop runs reach a witnessed-done verdict — the loop dogfoods the witness contract | 1/7 loop(s) reached a witnessed-done verdict |

## Run it

```bash
go run ./cmd/fak loop-score             # score this box's background loops
go run ./cmd/fak loop-score --markdown  # regenerate this doc
go run ./cmd/fak loop-score --json      # control-pane payload (corpus.loopscore_debt)
go test ./internal/loopscore/...        # prove the fold over a fragile vs durable corpus
```

Expected checkpoint:

```text
loopscore_debt: reports the current loop durability gaps
durability: shows registered firing loops
self-report: shows heartbeat/notify coverage
dogfood: shows guard-wrapped and witnessed loop runs
```

## The 3× program — make the loops durable, observable, and fak-native

The debt is concentrated in one structural gap: the loops that actually fire are driven by **external schedulers + ad-hoc logging**, not by `fak loop run`. This is an OPEN improvement program, not a shipped multiplier claim. So they are unregistered (a reboot re-launches nobody), often dark (the registry's own jobs have never been observed ticking), and unguarded (no `guard_enabled=1` run). A 3× is NOT hand-appending events during the measurement window (that is the data-gaming pattern every fak scorecard refuses) — it is making the durability + observability a **byproduct of how the loop is driven**, so the score rises structurally:

1. **Register every firing loop.** Add the issue-dispatch / resolve-progress / smoke loops to `tools/loop-registry.json` with a cadence (`fak loop` registry). A registered job re-arms at boot — that is the auto-restart on system restart. Then `fak cron emit --target launchd|systemd|taskscheduler` projects it to a real OS unit that survives the reboot.
2. **Drive them through `fak loop run`.** Replacing the raw scheduler call with `fak loop run --loop ID --source cron -- <cmd>` records fire/admit/start/**end** around every run under `fak guard` (guard_enabled=1) and posts a witnessed dispatch-result card — closing the self-report and dogfood gaps in one move.
3. **Let them witness.** A run that ends with a `fak loop append --kind witness --status witnessed_done` (the resolve-progress loop already does this once) makes the keep-rate real, so the loop dogfoods the witness contract instead of trusting its own exit code.

Re-run after a loop session and `--compare` against a pinned `--json` baseline: the verdict reports the multiple on the composite (the lever), so a real 3× (composite ~33 → ~99, debt → 0) is provable, not asserted.

**Next:** retire worst-first: firing_loops_registered — 0/5 (0%) firing loops are registered
