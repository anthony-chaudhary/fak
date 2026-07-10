# Bounded park-and-retry for the no-seat transient (#3523)

**Status:** kernel SHIPPED in `internal/seatpark` (13 tests) AND **wired live into BOTH
dispatch front-doors** — the `fak garden dispatch` bridge (`cmd/fak/garden_dispatch.go`,
`garden_dispatch_seatpark_test.go`, 9 tests) AND the `fak dispatch sweep` queue-drain
front-door (`cmd/fak/dispatch_sweep.go`, `dispatch_sweep_seatpark_test.go`, incl. a full
`runDispatchSweep` park + exhaust proof). Both derive the no-seat park tail from the same
durable loop ledger and park BEFORE running a tick. The only site left is the raw
`internal/dispatchtick` preflight where `REFUSE_NO_ACCOUNT` is *produced* (peer-dirty at
author time) — but every loop that *consumes* it now parks; see Open.

## Wired live (garden dispatch bridge)

`fak garden dispatch --apply` was the textbook burst: on a `REFUSE_NO_ACCOUNT` it stopped
the run, and the next scheduled invocation re-drove immediately against the same wall.
Now a **Gate 1.5** sits between the loop-governor admit and the candidate load: it derives
the consecutive no-seat park tail from the durable loop ledger the bridge already writes
(`deriveSeatParkState` — a run that stopped on a seat refuse records the `SEAT_NO_SEAT`
reason; a deferred run records `SEAT_PARKED` and is neutral in the tail; an exhausted one
records `SEAT_EXHAUSTED` and resets the cycle), feeds it to `seatpark.Decide`, and on
`SEAT_PARKED`/`SEAT_EXHAUSTED` returns **before loading a candidate or probing preflight**
— so a parked run adds zero load (it cannot manufacture a `REFUSE_INSPECT`). Keyed on the
seat-refuse verdicts specifically (`gardenDispatchSeatRefuses` = `REFUSE_NO_ACCOUNT` /
`WEEKLY_CAPPED`), so a drained queue, a fault, or the worker-slot cap (`REFUSE_AT_CAP`, a
worker-count wall, not an account-seat wall) never arms the park. Dry-run (inspection) is
never parked. Reuses the loop ledger + governor rather than inventing a second throttle.

## What shipped

`internal/seatpark/seatpark.go` — a pure, deterministic bounded park-and-retry
fold for the `REFUSE_NO_ACCOUNT` (no free Claude seat) preflight refusal. Today
dispatch maps that refusal to the `"no-seat"` skip reason
(`cmd/fak/garden_dispatch.go:317`) and re-checks it on the very next tick — a
*burst* against a wall only a peer finishing can move. This kernel replaces the
unbounded busy-retry with a bounded exponential backoff:

- `Decide(Input) Decision` returns a closed `Status`:
  - `SEAT_READY` — no active park (first encounter, or the backoff window
    elapsed): attempt a launch now.
  - `SEAT_PARKED` — seat-blocked and still inside the current backoff window:
    skip this tick cheaply.
  - `SEAT_EXHAUSTED` — the bounded retry budget (`Policy.MaxParks`) is spent:
    stop re-offering this cycle, surface for the next one.
- Geometric backoff `min(cap, base·factor^(parks−1))` with documented defaults:
  base 30s, factor 2, **cap 300s**. The cap is anchored to a real in-repo
  precedent, not invented: a no-seat refusal is the same shape of wall as
  `internal/attemptbudget`'s `FailureClassRateLimit` — "a shared capacity window
  reopening on its own" (`attemptbudget.go:121`, 5m) — so the longest a task
  waits between no-seat retries matches that window.

## Design choices (and why)

- **Clock as data.** `Decide` reads no clock; the caller supplies `NowUnix`,
  matching `internal/attemptbudget` and `internal/dispatchorder`. Same input →
  same verdict, trivially testable.
- **Fail toward READY.** A first encounter (`Parks==0`) and a missing clock
  (`NowUnix==0`) are `SEAT_READY`, never a silent stall — the kernel only ever
  *adds* a bounded wait, it never blocks a launch it cannot justify.
- **Distinct from `attemptbudget`, and composes with it.** `attemptbudget` folds
  an issue's POST-attempt failure history (a worker ran and failed); `seatpark`
  folds the PRE-attempt seat-contention refusal, where no worker launched and no
  `Attempt` is ever recorded. `seatpark` decides whether a launch may be tried;
  `attemptbudget` decides what to do once one has run.
- **Closed, typed status.** `Status` is a `known…`-validated newtype (mirrors
  `attemptbudget.Status` and `modelroute.Reduction`), so the dispatcher's
  skip-reason surface stays a verifiable vocabulary rather than free text.
- **New clean package.** Zero edits to existing files — collision-proof in the
  shared tree, and it builds/tests green even while the trunk tip does not.

## Open (honestly not done here)

- **`fak dispatch sweep` front-door — NOW WIRED.** `cmd/fak/dispatch_sweep.go` gained the
  same Gate-1.5 park (its own `dispatch-issue-sweep` loop id + `deriveSweepSeatParkState`, a
  deliberate mirror of the garden fold on a different loop id), so the primary queue-drain
  loop now parks instead of bursting. The only site still open is the RAW
  `internal/dispatchtick` preflight where `REFUSE_NO_ACCOUNT` is *produced*
  (`dispatch_tick_preflight.go`, peer-dirty at author time — a by-path commit there would
  sweep the peer's adjacent seat-refuse-transparency work). Every dispatch loop that
  *consumes* the verdict (garden + sweep) already parks; adding it at the production site is
  a belt-and-suspenders refinement, not a missing behavior, and drops in once that file
  settles.
- **Headroom-derived park duration.** The issue prefers the park window be derived from
  the live `fak accounts headroom` / cooldown signal (park until a seat is *projected* to
  free) rather than the fixed geometric backoff. `accounts_headroom.go` was peer-dirty;
  `seatpark.Policy` already accepts a caller-supplied schedule, so this is a drop-in
  refinement once that surface is free.
