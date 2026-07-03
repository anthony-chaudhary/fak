

---

<!-- dispatch-contract sections appended 2026-07-02; derived from the issue prose above + repo state. No intent change. Generation triage recorded here because the fleet worker's capability floor refuses outward `gh issue edit`/`gh issue comment` (preview-confirm gate; the confirm-token echo is stripped by the harness) — an operator or labeled-token pass must apply the labels/milestone named below. -->

## Generation stream

gen/now. Classified from issue evidence per docs/generation.md intake rules: this is a current-product operator-loop defect with a live host witness (the 2026-07-02 host audit in the body: one WindowsTerminal process holding 87 OpenConsole panes, 39 of them `fak info` dashboards, at 2,829 threads / 54k handles / 2 GB working set) and acceptance criteria checkable today. No dependency on a future architecture bet. Labels to apply: `generation`, `gen/now`, `priority/P2` (repo equivalent of the suggested priority/medium; the suggested `area/fak` / `kind/fix` labels do not exist in this repo's taxonomy). Milestone to bind: `Generation G0 - Now / Immediate`.

Promotion evidence: the host audit above plus the #2252 history (REFUSE_HOST liveness carve-out already shipped because this leak was freezing spawns).
Demotion evidence that would move it: if wave/dispatch launchers stop opening per-session info panes by default (fix option 2 lands as the launcher default), the per-pane self-exit (options 1/3) demotes to a hygiene backstop; if WT pane accumulation stops reproducing after #2252-adjacent host changes, re-witness before implementing.
Invalidating assumption: "gateway stopped answering for N consecutive intervals" is a reliable proxy for "the guarded session ended". If gateways restart in place or stall transiently under load during normal operation, exit-on-unreachable kills live dashboards; N and the option-2 multiplexed surface must be chosen against real gateway lifecycle evidence (e.g. `.dispatch-runs` session logs).

## Parent context

No parent epic named on the issue. Nearest program surface: milestone 8 "Fleet observability you can trust"; predecessor issue #2252 (REFUSE_HOST liveness carve-out) documents the same host pressure this issue removes at the source.

## Current state

`fak info` (`cmd/fak/info.go`) polls `--gateway-url` on `--interval` forever; `runGuardInfoOverlay` has no exit path when the gateway stops answering, and no `--max-idle` flag exists. `fak guard --split` (`cmd/fak/guard_split.go`) opens one 20% `fak info` pane per guarded session via `wt -w 0 split-pane`; when the guard exits, the info pane keeps polling a dead URL for the life of the WT window. No commit cites #2340, so closure is unbound.

## Why now

(gen/now: immediate operator-loop cost) Every fleet session leaks one permanent OpenConsole pane; an 11 h fleet day measured 2 GB / 2,829 threads / 54k handles on the WT host and per-tick proc_resource_guard flag noise. The only workaround (restart the terminal window) kills all live session panes.

## Working spine

Make the per-session `fak info` pane exit within ~30 s of its gateway dying: in the `runGuardInfoOverlay` poll loop, count consecutive `--gateway-url` failures and exit with a final one-line summary after N consecutive misses (N*interval ≈ 30 s default), plus an optional `--max-idle` backstop flag. Launcher-side multiplexing (fix option 2) is a follow-on, not the spine.

## In scope

Fix option 1 (exit on N consecutive gateway failures, with a final summary line) and option 3 (`--max-idle` backstop) in `cmd/fak/info.go`, with a focused test proving the loop terminates when the poll target stops answering.

## Out of scope

Fix option 2 (wave/dispatch launchers reusing one multiplexed pane or opening none for headless waves) — a launcher-policy change touching `cmd/fak/guard_split.go` and dispatch tooling; split into its own issue if the spine fix does not return WT pane counts to baseline. Also out: proc_resource_guard threshold tuning (#2252 owns that surface).

## Done condition

After a guarded session ends, its paired `fak info` pane exits within ~30 s; WT pane count returns to interactive baseline instead of growing monotonically; proc guard stops flagging the terminal host under normal fleet operation.

## Witness

A test that fails before the fix and passes after: drive the info poll loop against a gateway that stops answering and assert the loop returns (with the summary line) within N intervals. Plus a before/after host readout (pane count / thread count on the WT process) attached to the issue.

## Acceptance gate

`go test ./cmd/fak -run Info -count=1` green (under WSL `./test.ps1` on a native-Windows host) and `make ci` green.

## Closure binding

Resolving commit cites #2340 in the subject and carries the ship trailer the dispatcher names for its lane (this dispatch named `(fak metrics)`; see Coordination for the routing tension).

## Lane

metrics (as dispatched). NOTE — likely misroute: every likely file lives under `cmd/fak/**`, not `internal/metrics/**`, so the metrics cluster lease does not actually cover the edit surface. Re-route to the lane that owns `cmd/**` before the implementing dispatch, or the lease protects nothing.

## Work unit

leaf

## Expected steps

5 — add failure-counter + `--max-idle` exit to the info loop, write the terminating-loop test, run the focused gate, capture the before/after host readout, commit citing #2340.

## Assumptions

- The paired gateway becoming unreachable is how an ended session presents to its info pane (see the invalidating assumption under Generation stream — pick N against real gateway lifecycle evidence).
- WT panes close when the hosted process exits (no `wt` keep-open profile setting on fleet hosts); if fleet WT profiles set closeOnExit=never, the process fix will not reap panes and option 2 becomes the spine.

## Confusion risks

- Do not conflate this with proc_resource_guard tuning: #2252 already carved out liveness; this issue removes the leak at the source.
- `fak info --once` already exists and exits; the defect is only the watch mode.
- Exit-on-unreachable must not fire during gateway startup (pane opens before the gateway binds) — the failure counter should only arm after the first successful poll, or N must exceed worst-case startup.

## Coordination

- `cmd/fak/info*.go` is shared with any open info/TUI work (info_visual, watchdog autoheal both touch adjacent files; watchdog_autoheal.go is dirty in the shared tree right now).
- Lane/tree mismatch: this dispatch leased `metrics` (`internal/metrics/**`) but the edit surface is `cmd/fak/**` — take the covering lane before implementing.

## Trigger

Host audit 2026-07-02 (87-pane WT process) filed as #2340; groomed by the metrics-lane triage dispatch of 2026-07-02 under a triage-only generation frame.

## Batch policy

One issue for the pane-leak defect; launcher-side multiplexing (option 2) splits into its own issue only if the spine fix leaves pane counts above baseline. Re-grooms update this overlay in place instead of filing duplicates.

## Likely files

- `cmd/fak/info.go`
- `cmd/fak/info_test.go`
- `cmd/fak/guard_split.go` (option 2, out of scope here)
