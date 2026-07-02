---
title: "Safe concurrency headroom on the agent host (2026-07-01)"
description: "Witnessed capacity audit of the one-box fleet: where the effective worker cap actually binds today, ranked safe levers to raise live concurrency, and the liveness bug that has the cap pinned at zero."
---

# Safe concurrency headroom on the agent host (2026-07-01)

Question: what are the *safe* ways this one machine can run more concurrent
agents? Answer from live probes, not theory. Every number below was witnessed
on 2026-07-01 on the agent-host box (see
`experiments/benchmark/catalog.json`); the decision frame is
[`docs/safe-to-raise-cap-checklist.md`](../safe-to-raise-cap-checklist.md):

```text
effective cap = min(configured_max, dos_target>0?, host_cap, seats, rate_budget)
```

## Witnessed state (2026-07-01)

| Dimension | Value today | Binding? |
|---|---|---|
| Hardware | Ryzen 9 9950X 16C/32T, 253.6 GB RAM (178.7 free), CPU 37% | no — huge headroom |
| `host_cap` (adaptive, #1337) | **16**, binding component = cores (32÷2); threads dim ≈ 67 (18,449 live vs 32,000 budget, `FAK_HOST_THREADS_PER_CORE=1000` already set), RAM dim ≈ 121 | no |
| Configured static caps | claude task `--max-workers 3`, codex task `--max-workers 4`, GLM-docs target 12 | sometimes |
| Seats | 6 dirs → 4 distinct Anthropic accounts; **2 available now** (gem7, july1), 2 usage-limited until tonight (9:10pm / 12:10am PT); +1 codex ambient; +2 opencode (GLM provider dead until 2026-07-04, 28/28 stub logs) | **yes — the real multiplier** |
| Preflight verdict | **REFUSE_HOST — effective cap 0**, dispatcher STALLED at 0 live | **yes — total freeze** |

The freeze cause: `proc_resource_guard` flagged **WindowsTerminal pid 85884
(2,767 threads > 2,000/proc)**. It is PROTECTED — the guard rightly refuses to
kill the operator's terminal — but `dispatch_preflight.host_check()` folds
*any* flagged row into `safe=False`, so one foreign, un-reapable process
freezes every spawn indefinitely. The box is at 37% CPU with 178 GB free while
the fleet runs zero workers.

## Ranked levers (safe-first)

1. **Give REFUSE_HOST a liveness carve-out (0 → cap immediately).**
   A flagged process that is PROTECTED and not fleet-spawned should be
   *advisory* (its threads already discount `host_cap` through the threads
   dimension); only a flagged *fleet-reapable* process should hard-refuse
   spawns. Operator workaround today: restart Windows Terminal. This is the
   only lever that changes today's number, because the current cap is 0.
2. **Add seats — the true multiplier.** Cap math tops out at seats long before
   host_cap=16. Each seat is safe by construction (one worker per account
   cannot double-book a rate limit). Today: 2→4 claude seats return by
   midnight on their own; GLM's 2 return 2026-07-04; one more paid Anthropic
   account ≈ +33% standing fleet. The machine has room for ~16.
3. **Raise static ceilings to the design intent.** `dispatchtick.go` says the
   static cap should sit *above* the adaptive gates (they can only lower).
   With host_cap=16, the claude task's `--max-workers 3` binds before the box
   does the moment ≥3 seats are green. Raise 3→4–6 via the checklist gates
   only (seats, host, leases, rate, closure honesty all green).
4. **Keep the thread baseline from decaying: an elevated zombie reaper.**
   Session-0 S4U-spawned retry-loop zombies (68 opencode.exe ≈ 2,067 threads
   on 2026-06-30) are un-killable from a non-elevated shell and eat the
   threads dimension. A SYSTEM-scheduled `proc_resource_guard --enact`
   variant scoped to *known agent binaries only* (never foreign processes,
   PROTECTED list intact) keeps ~67-worker thread headroom real.
5. **Recalibrate per-worker charge constants from measurement.** host_cap
   charges 200 threads + 1,500 MB per worker — conservative guesses. Measured
   footprints of a `fak guard -- claude -p` worker would let the constants be
   honest instead of cautious.
6. **Scale the safety scaffolding with the worker count** (this is what makes
   the levers above *safe*): fak guard fronts every worker (9,690 witnessed
   decisions across 157 dispatch sessions today: 98 DENY, 447 QUARANTINE);
   waves of ~3 spawns (burst fan-out has tripped provider limits at ~28);
   lane leases + pairwise-disjoint lane trees for collision isolation
   (lane_leases currently 0/0 — workers should hold them); witnessed-close
   gates so more workers means more *verified* throughput, not more claims.

## Honest fences

- ~~`not yet`: lever 1 is a code change nobody has made~~ → **shipped
  2026-07-02** as 69439d98 + c06f9bbf (#2227/#2252, both closed): `host_check()`
  now hard-refuses only on an ACTIONABLE flag (non-protected, or a fleet agent
  image); a PROTECTED foreign breach surfaces as ADVISORY on a SAFE host, with
  tests in `tools/dispatch_preflight_test.py`. Both commits diff-witnessed via
  `dos_commit_audit`. Same-day confirmation of the note's predictions: the
  guard reads `flagged 0`, the freeze is gone, and the seat pool moved 2→3
  available (gem8NEW's reset landed on schedule).
- The status card prints `max=2` while the scheduled task passes
  `--max-workers 3` — the card probes its own default instead of the task's
  live args; fix the readout before reasoning from it.
- Rate budget was `n/a (--fast)` in today's probe; the checklist requires a
  fresh rate row before any static-cap raise.

## Ceiling estimate

Today: 0 (frozen). After the terminal fix + tonight's seat resets: ~4–5
(claude 3-cap + codex). After 2026-07-04 (GLM back): ~7. With lever 3 and one
added seat: ~8–10. The machine itself stays green to host_cap 16 — on this
box, concurrency is bounded by seats and one guard fold, not by hardware.
