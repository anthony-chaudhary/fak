# Developer status checkpoints

`fak dev checkpoint` records implementation and repository-work milestones. It is deliberately under the `dev` namespace: it does **not** checkpoint an end-user agent runtime, conversation, tool state, or resumable inference session.

For development work lasting more than one meaningful step, write `started`, material `progress`, `blocked`/`handoff`, and `done` rows. Do not log each tool call or no-change heartbeat.

```powershell
fak dev checkpoint `
  --actor "worker-issue-123" --scope "issue #123" --state progress `
  --stage-current 2 --stage-total 4 --stage-name "implementation" `
  --summary "Added stale-lease classification" `
  --evidence "internal/leases/lease_test.go::TestStale" `
  --next "Run supervisor integration tests"
```

The append-only local log defaults to `.fak/dev-status.jsonl`. Each row has `timestamp`, `actor`, `scope`, `state`, optional typed `stage` (`current`, `total`, `name`, computed `percent`), `summary`, `evidence[]`, `next`, `blockers[]`, and optional `github`. `progress` requires a valid ordinal stage; `blocked` requires a blocker; every non-`done` state requires one next action. Use `--log` to select another ledger and `--json` to echo the appended record.

If an issue or PR already is the canonical coordination surface, mirror only material milestones there. Do not create an issue solely to report status. After a crash, inspect the latest row for the actor/scope and independently verify its evidence before resuming.

## CLI placement rule

Place a command according to the state it acts on, not according to who happens to invoke it:

- `fak dev …` is for repository development, contributor coordination, CI/build workflows, and coding-worker bookkeeping.
- `fak agent …` is for the product's end-user agent runtime and behavior.
- Cross-cutting operational commands use a purpose-named top-level namespace only when neither domain owns them.

A generic data shape does not make a command product-generic. Before adding a subcommand, state its subject in one sentence and test that its help text cannot plausibly be mistaken for another domain. Compatibility aliases across these boundaries are avoided because they preserve the ambiguity.
