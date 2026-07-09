# Bounded park-and-retry for the no-seat transient (#3523)

**Status:** pure kernel SHIPPED in `internal/seatpark` (clean package, compiles &
tests independently of the current red trunk); dispatcher wiring remains OPEN
(see below). Diff-witnessed by `internal/seatpark/seatpark_test.go` (13 cases).

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

- **Dispatcher wiring.** Consulting `seatpark.Decide` on the `REFUSE_NO_ACCOUNT`
  path in `cmd/fak/garden_dispatch.go` / `internal/dispatchtick`, and persisting
  each task's `Parks`/`LastParkUnix` across ticks (the caller's bookkeeping the
  kernel folds over). `garden_dispatch.go` is clean; `internal/dispatchtick`'s
  `dispatch_tick.go` neighbour was mid-edit by another agent at author time, and
  the trunk tip did not compile (`internal/gateway` `upstream4xxStatus`), so
  wiring + a cmd-level test run was deferred rather than landed against a red
  tree.
- **Park-state store.** Where `Parks`/`LastParkUnix` live between ticks (the loop
  ledger, or a small sidecar) is the wiring layer's call; the kernel is
  storage-agnostic by construction.
