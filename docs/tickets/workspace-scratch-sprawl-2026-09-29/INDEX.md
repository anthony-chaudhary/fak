<!-- fak-dualsync-key: workspace-scratch-sprawl-2026-09-29-index -->
# Workspace scratch sprawl — public tickets (2026-09-29)

Three public-repo tickets bounding scratch checkouts that accumulate beside the
workspace root when a bounded/killed land skips its deferred cleanup.

| # | Subject | Lane | Paths |
| --- | --- | --- | --- |
| 01 | crash-safe, owner-named exec-witness scratch + sweep-before-create | witness | `internal/witness/execution.go` |
| 02 | multi-parent `.fak-cand-validate-*` sweep + prune-on-remove | workerworktree | `internal/workerworktree/candidate.go`, `land.go`, `cmd/fak/worktree_worker_reap.go` |
| 03 | reuse block-clone / `--no-checkout` / pool for per-copy checkouts | witness | `internal/witness/execution.go`, `internal/workerworktree/*` |

Order: 02 first (bounds the candidate leak), then 01 (same pattern for the
witness scratch), then 03 (shrinks the bytes both materialize). 01 and 03 share
a materialization seam; isolate with `fak-flow start` if landed concurrently.

Witness per ticket: `go test ./internal/witness/... -count=1` and/or
`go test ./internal/workerworktree/... -count=1`.
