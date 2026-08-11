# Agent status checkpoints

For work lasting more than one meaningful step, leave a compact durable trail with `fak agent checkpoint`. Write `started`, material `progress`, `blocked`/`handoff`, and `done` rows; do not log each tool call or no-change heartbeat.

```powershell
fak agent checkpoint `
  --actor "worker-issue-123" --scope "issue #123" --state progress `
  --stage-current 2 --stage-total 4 --stage-name "implementation" `
  --summary "Added stale-lease classification" `
  --evidence "internal/leases/lease_test.go::TestStale" `
  --next "Run supervisor integration tests"
```

The append-only local log defaults to `.fak/agent-status.jsonl`. Each row has `timestamp`, `actor`, `scope`, `state`, optional typed `stage` (`current`, `total`, `name`, computed `percent`), `summary`, `evidence[]`, `next`, `blockers[]`, and optional `github`. `progress` requires a valid ordinal stage; `blocked` requires a blocker; every non-`done` state requires one next action. Use `--log` to select another ledger and `--json` to echo the appended record.

If an issue or PR already is the canonical coordination surface, mirror only material milestones there. Do not create an issue solely to report status. After a crash, inspect the latest row for the actor/scope and independently verify its evidence before resuming.
