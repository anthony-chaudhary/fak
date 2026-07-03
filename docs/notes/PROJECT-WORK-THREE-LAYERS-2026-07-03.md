# Pulling the project work apart: three layers, one verb each (2026-07-03)

Project work in this repo is easy to read as one undifferentiated pile — a
dirty 52-path shared tree, a 699-issue backlog, and a roadmap, all in motion at
once. This note pulls it apart into the three layers it actually consists of,
and names the repo verb that enumerates each layer, so the separation is
mechanical rather than a judgment call:

1. **Planning** — deciding what the work *is* and how it is measured.
2. **Known work queued** — what is already known to need doing, validated and
   waiting, but not yet being done.
3. **Implementation itself** — edits in flight in the working tree.

A unit of work sits in exactly one layer at a time, and each layer has a
different honesty rule: plans are measured as a frontier or a completion %,
queued work is measured by contract-readiness, and implementation is measured
by witnessed commits. Mixing the layers is how an unproven claim gets read as
shipped; keeping them apart is the point.

## Layer 1 — Planning (`fak milestone report`)

The planning layer is owned by the worktype split in
[`internal/worktype`](../../internal/worktype/worktype.go) (see AGENTS.md,
"Planning: two kinds of work"): an **ongoing program** is never done and is
tracked as a frontier + trend; a **discrete epic** has a definition of done and
converges on 100%. `fak milestone report` renders both without conflating them.

Snapshot (2026-07-03, @22d4beb5):

- **Climb (programs' frontier):** 21/56 cells matured to M4+, 36.0%
  (+0.5% and +2 matured vs 2026-06-30).
- **Roadmap:** 42.4% across 5 discrete epics; 2 ongoing programs
  (cache-value P&L roll-up; a flagship model through the kernel).
- **Generation lanes:** `now` 4 tracked / 60.9% over 3 discrete;
  `next` 3 tracked / 0.0% over 2 discrete.

Standing planning artifacts also live here: the auto-generated hardware bench
plan (`docs/bench-plan.md`, refreshed by `tools/bench_plan.py`) and the two
program spines (`docs/perf-parity-rsi-loop.md`, `docs/cache-value-rollup.md`).
These *decide and rank* work; none of them is work being done.

## Layer 2 — Known work queued (contracts, overlays, debt counters)

This layer is what is already known to need doing — specified well enough to
dispatch — but not in any working tree yet. Three queues hold it:

- **Parked ready-to-file issues.** The worker-unit conservation backlog
  ([`WORKER-UNIT-CONSERVATION-BACKLOG-2026-07-02.md`](WORKER-UNIT-CONSERVATION-BACKLOG-2026-07-02.md))
  carries three contract-validated issue bodies (stall-reap, attempt-budget
  wiring, SPAWN_FAILED lease release), each graded `ready` 100/100 by
  `fak issue contract --from-issues`. As of today they remain parked —
  specified, deduped, unfiled.
- **Contract-repair overlays.** `.dispatch-runs/contract-overlays/` holds 8
  authored repairs (issues 1411, 1507, 1511, 1515, 1648, 2272, 2340, 2372)
  awaiting verify/sync — work whose *content* exists but whose landing is
  queued behind a gate.
- **Roadmap debt counters.** The milestone report's own queue pressure:
  `now`-lane debt 12 (9 stale-risk issues, 1 missing witness); `next`-lane
  debt 35 (10 stale-risk, 1 missing witness, 11 unpromoted bets). Backlog
  pressure from the 2026-07-02 conservation reading: 699 open vs a 483
  baseline, with contract-holds — not spent units — the dominant reason
  capacity went unused.

The honesty rule for this layer: an item belongs here only once it passes the
issue contract (`ready`, scored); a thin idea is planning, not queue.

## Layer 3 — Implementation itself (`fak sweep --json`)

The implementation layer is the dirty shared tree, and `fak sweep --json` pulls
it apart by lane. Snapshot (2026-07-03): **52 dirty paths → 13 lane groups**
plus one no-lane operational artifact (a dispatch overlay, gitignored class).

| Lane | Paths | Character of the in-flight work |
|---|---|---|
| `cmd` | 17 | dispatch-tick preflight, watchdog autoheal + render-witness tests, memory recall, a live-model demo (`trychatdemo`), repoguard CLI |
| `docs` | 11 | scorecard/demo doc refreshes, nightrun telemetry, bench-plan auto-refresh, backlog-note touch-ups |
| `tools` | 10 | a new popularization-readiness scorecard (+ data), dispatch preflight, scorecard control pane |
| `gateway` | 2 | proposed-call adjudication + wiring |
| `memq` | 2 | notes backend + tests |
| `repoguard` | 2 | new `decisions.go` + tests (new package surface) |
| `logvault` | new pkg | untracked new internal package |
| `adjudicator` · `policy` · `model` · `devindex` · `dos` · `claude` | 1 each | single-file edits |

Each group ships separately — `fak sweep --apply --lane <lane>` or
`fak commit --path …` — because on a shared multi-session tree the lane group,
not the whole diff, is the unit of implementation. The honesty rule for this
layer: nothing here counts as done until it is a witnessed commit
(`dos commit-audit` / the `(fak <leaf>)` stamp); a dirty edit is in-flight, not
shipped.

## How the layers connect (and why the separation pays)

Planning names the lane, the witness, and the measure; the queue holds
contract-validated bodies that any authorized worker can pick up cold; the
implementation layer is lane-partitioned so concurrent workers don't collide.
Each hand-off between layers is a *gate with a verb*: planning → queue is
`fak issue contract` (ready or it stays a plan); queue → implementation is the
dispatch tick + `dos arbitrate` (a lease or it stays queued); implementation →
done is the witnessed commit (a stamp `dos verify` can bind, or it stays
in-flight).

Read the three snapshots above together and the day's honest status falls out
without any self-report: what the project intends (36.0% climb, 42.4% roadmap),
what it already knows it owes (3 parked issues, 8 overlays, debt 12+35), and
what is actually moving (13 lanes in flight). When a status update conflates
those — reporting a queued item as moving, or an in-flight edit as shipped —
this decomposition is the correction.
