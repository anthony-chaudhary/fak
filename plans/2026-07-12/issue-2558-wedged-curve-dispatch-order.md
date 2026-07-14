# Plan: #2558 — dispatch-tick admission input: deprioritize wedged objectives by curve

- Author: claude worker (lane `trajctl`), 2026-07-12
- Issue: #2558 (epic #2533; blocked-by #2538 SHIPPED at 4c7e852d2; related #2550 SHIPPED at d6d2acba7/a3e667f07 on the same seam)
- Generation: gen/next — additive, fail-open, dogfood-wired; no default behavior changes when no curve signal is present.

## What exists today (verified read-only)

- `internal/trajctl/curve.go` ships the closed curve vocabulary from #2538: `HEALTHY / STALL / DRIFT / DETOUR_OVERRUN`; `trajctl.Fold(ReadLedgerFile(root/docs/nightrun/trajctl.jsonl)).CurveFor("issue-N")` yields a per-objective signal. STALL is already a *windowed* judgment (flat delta + active divergence), i.e. the "sustained flat curve" wedge.
- `internal/dispatchorder.Plan` is the pure ordering decision the wave path (`cmd/fak/dispatch_wave.go:516`) and the CLI (`fak dispatch order`, `fak dispatch skip-ledger`) run. It has no curve input: a wedged objective competes equally with fresh work.
- `internal/skipledger.Record` persists one row per candidate (disposition + reason) verbatim — the "ledgered reason" channel already exists; a new Reason token flows through with zero changes there.
- `#2550` wired per-issue `child_curve` read-back into the tick payload (`cmd/fak/dispatch_tick.go:820`, `internal/dispatchtick/childcurve.go`) — read-back only, not an ordering input. Our change is in `dispatchorder.Plan`'s candidate schema + sort, a different function than #2550's prompt/payload seam, so no clobber.

## Change (smallest correct)

### 1. Leaf: `internal/dispatchorder/dispatchorder.go`

- Add `Candidate.CurveSignal string` (JSON `curve_signal`) — the trajctl closed vocabulary carried as data (string, keeping this package's "imports nothing internal" purity). Empty = no witnessed curve = fresh (fail-open, byte-identical ordering for every existing caller).
- Add `ReasonWedgedDeprioritized = "wedged_deprioritized"` to the closed reason vocabulary.
- Wedge predicate: normalized `CurveSignal == "STALL"`. STALL only — DRIFT/DETOUR_OVERRUN are steering's business (`trajctl/steer.go`), not admission's; the issue names sustained-STALL.
- In `Plan`: a kept-and-wedged unit KEEPS `DispKeep` and stays in `Keep`/`Pick` eligibility (deprioritization must not become implicit refusal), but its `Reason` becomes `wedged_deprioritized` instead of `freshest`.
- Final sort gains one tier: kept-first (existing) → **non-wedged before wedged (new)** → Priority desc → recency/PreferOldest/ID. The wedge tier sits ABOVE Priority deliberately: a wedged P0 re-fed ahead of fresh P2s is exactly the worker-unit burn the issue targets; an operator can still pin `tick --target-issue`.
- Collision admission (`applyCollisionPrice`): wrap the admit order so non-wedged admit first — a wedged unit must not win a collision slot and serialize fresh work behind it.
- `Result.WedgedKeptCount int` for the one-line summary.

### 2. Dogfood wiring: `cmd/fak/dispatch_wave.go`

- Fold the trajctl ledger ONCE per wave build (`trajctl.Fold(trajctl.ReadLedgerFile(filepath.Join(root, trajctl.DefaultLedgerRel)))`), and stamp `CurveSignal` on each issue candidate via `CurveFor("issue-<N>")`. Absent ledger / absent curve → empty signal → plan byte-identical to today. This is the "dispatch-tick working path" feed the spine names, on the path that actually calls `dispatchorder.Plan`.

### 3. Operator/agent surface: `cmd/fak/dispatch_order.go`

- `parseCandidates` picks up `curve_signal` for free (JSON tag). Document it in the usage text (the agent-runnable schema artifact for gen/next) and render wedged kept rows explicitly in `renderDispatchOrder` (e.g. a `wedged` note line naming `wedged_deprioritized`), plus the count in the summary header.

### 4. Witness (the issue's done condition)

`internal/dispatchorder/dispatchorder_test.go`:
- `TestWedgedObjectiveSortsBelowFreshCandidates` — a STALL candidate that is fresher AND higher-priority still ranks below fresh candidates; its `Reason == "wedged_deprioritized"`; it is still present in `Keep` (ordering, not exclusion); `Pick()` returns the fresh unit.
- `TestWedgedOnlyCandidatesStillDispatch` — an all-wedged set still yields keeps + a pick (no starvation → not a refusal).
- `TestCurveSignalAbsentOrHealthyNoRegression` — empty/HEALTHY/DRIFT signals leave order and reasons byte-identical to today.
- Collision case: wedged collider loses the safe-set slot to the fresh one.
- Small `cmd/fak` test for the wave curve-signal helper (plant a temp ledger, assert the per-issue signal map; hermetic).

## Out of scope (unchanged from issue)

- Killing workers (#2559), picker rewrite.
- The Python `dispatch_status.silent_workers` curve tie-in: partially covered — the wedge now surfaces as a ledgered `wedged_deprioritized` reason in `fak dispatch order`/`skip-ledger`, and Go `fak dispatch status` already renders per-objective curve lines; the per-silent-row curve join in `tools/dispatch_status.py` is named in the final report as the follow-on seam (it lives in the Python tools lane, not this leaf).

## Gate

- `go test ./internal/dispatchorder ./internal/skipledger -count=1` + touched `cmd/fak` tests (peer-dirty-tree caveats per fleet memory: verify leaf in isolation, `-vet=off` if peer vet debt).
- `make ci` / `scripts/ci.ps1`; scorecard control-pane `--check` non-regression.

## Ship

- Single commit by explicit path on `main`:
  `feat(dispatch): deprioritize wedged objectives by curve in dispatch order (#2558) (fak trajctl)`
  body carries `Closes #2558`.
- Paths: `internal/dispatchorder/dispatchorder.go`, `internal/dispatchorder/dispatchorder_test.go`, `cmd/fak/dispatch_wave.go` (+ helper test file), `cmd/fak/dispatch_order.go`.

## Generation evidence (gen/next contract)

- Promotion evidence toward gen/now: skip-ledger rows showing `wedged_deprioritized` on real fleet ticks while wedged issues stop being re-picked first (worker-units conserved), with no starvation complaints (wedged units still eventually dispatch when fresh work drains).
- Demotion/retirement evidence: fleet ticks where the STALL signal is noisy (false wedges deprioritizing genuinely-progressing work) or where the ledger is too sparse to ever fire — then the tier should demote to advisory render-only.
- Invalidating assumption: that `trajctl`'s STALL fold is available and honest at wave-build time on fleet hosts (the ledger `docs/nightrun/trajctl.jsonl` is populated per issue objective). If dispatch hosts never carry per-issue curves, the input is permanently empty and this rung is dead code — that is checkable from the skip-ledger (zero `wedged_deprioritized` rows over a week of ticks).
