# Wiring audit — internal/worktree (debtwave2-internal-worktree)

Read-only wiring audit for the `worktree` debt lane (ticket
`docs/tickets/debtwave2-internal-worktree/TICKET-debtwave2-internal-worktree.md`,
baseline census 2026-11-02). Audited 2026-11-02 at HEAD `21b6512ec8`; independently
re-verified 2026-11-03 at the same HEAD by two fresh subagents (import-chain audit:
6/6 claims HOLDS; scoring-model analysis: no in-boundary lever exists, see below).
Third independent re-verification 2026-11-03 at origin/main `59df3efc1d` (~80 trunk
commits past the audit base): import-chain audit 6/6 claims HOLDS plus a dynamic-path
sweep (no registry/reflection/string load of the package); scoring-model audit
confirms NO_IN_BOUNDARY_LEVER (raw 7.5 capped at 5.0 by curve.go:80-82; total_debt >=
6.48 while gap > 0); `go test ./internal/worktree/... -count=1` ok, `go vet` + gofmt
clean.

## Finding: ABSENT-with-reason (disconnected, deliberately not wired here)

The lane's only health issue is `disconnected_wiring`: no production call path invokes
this package. Presence != invokability, so the audit establishes the full importer chain:

| Question | Answer | Re-check command (repo root) |
|---|---|---|
| Who imports `internal/worktree`? | Exactly one: `internal/supervise/reaper.go:13` | `rg -n '"github.com/anthony-chaudhary/fak/internal/worktree"' -g '*.go'` |
| Who imports `internal/supervise`? | Nobody — zero matches repo-wide (orphan chain) | `rg -n '"github.com/anthony-chaudhary/fak/internal/supervise"' -g '*.go'` |
| Is there a cmd/ entry point to it? | No. BFS from `cmd/**/*.go` imports cannot reach it (`internal/debtlane/scan.go` `scanReachableFromCmd`) | `fak debt-lanes --workspace . --target-repo fak --lane worktree --json` → `evidence.integrated=false` |
| Where does production worktree machinery live? | `internal/workerworktree` (+ `internal/worktreereapcli`, `pkg/worktreecheck`) — distinct packages, wired into `cmd/fak` (`commit_lifecycle.go`, `dispatch_tick.go`, `worktree_worker_reap.go`, …) | `rg -n 'workerworktree\.' cmd/fak/ -g '!*_test.go'` |

The orphan shape is architecturally pinned, not accidental:
`internal/architest/architest_test.go:752` records `"supervise": 3 // contract reaper and
stuck-loop detector (#11171); imports leaseref(2)+worktree(2), off the hot path.`

## Why no wiring was landed in this ticket

1. **Boundary.** The ticket permits edits ONLY under `internal/worktree/`. A minimal real
   connection (a `fak` verb driving `Manager.Sweep`, or wiring `internal/supervise` into
   `cmd/dispatchworker`) necessarily edits `cmd/fak/*` or `internal/supervise/*` — sibling
   packages, off-limits here.
2. **No hollow wiring.** A blank import or presence-only registration would flip
   `integrated=true` without any call path — exactly the gaming this audit forbids.
3. **Companion coverage.** The supervise side has its own wave ticket
   (`debtwave2-internal-supervise`); wiring belongs to a ticket whose boundary spans both
   packages plus the verb table.

## Reader impact (why the numbers are frozen until wiring lands)

While `integrated=false`, `internal/debtlane/curve.go` hard-caps the lane at the tested
ceiling (5.0) regardless of in-lane quality (`if !e.Integrated && score > 5.0 { score = 5.0 }`,
curve.go L79-82), and `interest.go` floors the rate at the enabling base (0.08).
Baseline: maturity 5.0/8.0, principal 6.0, carrying 0.48, total debt 6.48.
In-lane quality is already at the curve's max for a disconnected unit (raw 7.5:
4.0 tested base + 0.5 testfiles>1 + 1.0 dogfooded + 1.0 benchmarked + 0.5 clean comments
+ 0.5 dependents<=25), so no edit inside `internal/worktree/` can move maturity or
total_debt. The census confirms the landed audit is inert by construction:
`inspectUnitEvidence` skips non-Go files (scan.go L1061), so committing this document
changes no scored evidence field.

Once a future out-of-boundary wiring flips `integrated=true`, the same evidence scores
7.5 + 1.5 integration + 1.0 production-grade bonus = 10.0 (rung production_grade) >=
target 8.0, retiring the lane automatically — nothing else in-lane blocks the target.

## Next checkable step

The seam already exists on the companion side: ticket `debtwave2-internal-supervise`
produced `cmd/fak/supervise.go` (imports `internal/supervise`; witness: `git show
1e8f4c9ab6 --stat` — local as of 2026-11-03, not yet on origin/main). Once that commit
lands, the orphan chain resolves — `cmd/fak -> internal/supervise -> internal/worktree`
becomes reachable from cmd — and this lane flips `integrated=true`: maturity 10.0,
total_debt 0. That commit's remaining 2-line `cmd/fak/main.go` dispatch hunk
(`case "supervise"`) was held for PEER_WIP_COLLISION against
ticket-debtwave2-internal-agentsched; it affects CLI usability, not import-graph
reachability. After landing, re-run
`fak debt-lanes --workspace . --target-repo fak --lane worktree --json` and expect
maturity 10.0 / total_debt 0; update the architest import-shape pin
(`internal/architest/architest_test.go:752`) only if the supervise edge count changes.

## Closing receipt — lane at target maturity (2026-11-03)

The unblock named by the ABSENT-with-reason finding is present on this branch
(branch-local; not yet on origin/main `b0b49d9f78`): commit `ab29ca688f` (supervise
lane; patch-id-identical twin `1e8f4c9ab6` in the sibling
ticket-debtwave2-internal-supervise worktree) adds `cmd/fak/supervise.go`, which
imports both `internal/supervise` and `internal/worktree`. In this branch's tree the
orphan chain is resolved: `cmd/fak -> internal/supervise -> internal/worktree` is
reachable from cmd (and `cmd/fak -> internal/worktree` directly). No hollow wiring was
introduced: the connection is a real call path (`cmdSupervise` drives
`Supervisor.Tick` against the worktree manager), not a blank import; CLI dispatch
remains held (see landing notes).

Witness (this worktree, 2026-11-03; origin/main `b0b49d9f78`):

| Metric | Baseline (2026-11-02) | Now |
|---|---|---|
| integrated | false | true |
| maturity | 5.0 / 8.0 (tested ceiling) | 10.0 / 8.0 (production_grade) |
| total_debt | 6.48 | 0 |
| health | degraded (0.85, disconnected_wiring) | healthy (1.0, wiring_status=integrated) |

Reader: `go run ./cmd/fak debt-lanes --workspace . --target-repo fak --lane worktree --json`
→ `next_action: "worktree is at target maturity (10.0/8.0); keep tests and benchmarks green"`.
Package witnesses: `go test ./internal/worktree/... -count=1` ok; `go vet` clean;
gofmt clean. Companion checks: `go test ./internal/supervise/... -count=1` ok;
`go test ./internal/architest/ -run TestArch -count=1` ok (import-shape pin at
architest_test.go:752 still holds: supervise imports leaseref+worktree, off the hot
path).

Landing notes:
- `ab29ca688f` is branch-local at writing time (not on origin/main); it lands with this
  branch through the flow landing, or is recognized LAND_ALREADY_ON_TRUNK if the
  supervise lane's patch-id-identical twin `1e8f4c9ab6` lands first.
- The 2-line `cmd/fak/main.go` dispatch hunk (`case "supervise"`) remains uncommitted in
  both worktrees, held for PEER_WIP_COLLISION against ticket-debtwave2-internal-agentsched.
  It affects CLI usability only; import-graph reachability (what flips `integrated`) does
  not depend on it.
- If a future supervise-lane change alters the supervise edge count, update the architest
  import-shape pin (`internal/architest/architest_test.go:752`) in that owning lane.
