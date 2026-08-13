# Child-agent registration and lineage

Every executable child agent launched by `dispatchworker` or `fak guard` is registered durably **before** its process starts. Registration failure is fail-closed: no child work runs without an inspectable row.

## Record and lifecycle

The append-only JSONL store uses schema `fak-child-registration/1`. The default is the OS config directory under `fak/child-registrations.jsonl`; set `FAK_SESSION_REGISTRY` for an explicit shared store. Each latest record carries stable registration, parent, root, task/issue, attempt/resume, lane/lease, runtime/session/thread/process, scope, timestamps, terminal state, reason, and witness identities.

Lifecycle is `registered` (persisted before start) → `active` (PID plus process start time read back) → `completed|failed|cancelled|lost|reaped`. `unknown` is allowed only with a reason. Retries with the same immutable identity are idempotent; conflicting replay and a child whose parent is absent are refused. `fak guard` registers before the brokered command is started, reads back PID/start after the job boundary is armed, creates a distinct attempt for every budget/crash resume, and terminalizes normal, failed, cancelled/restarted, time-budget, and lost outcomes. A child independently receives its own `FAK_REGISTRATION_ID`/`FAK_ATTEMPT_ID`, the parent side as `FAK_PARENT_REGISTRATION_ID`/`FAK_PARENT_ATTEMPT_ID`, and root ancestry as `FAK_ROOT_REGISTRATION_ID`, `FAK_ROOT_OUTCOME`, `FAK_ROOT_ISSUE`, and `FAK_TASK_ID`. A nested launcher uses the current registration as its child's parent, preserving both sides of every edge.

## Inspect and trace

```bash
dispatchworker inspect --json
dispatchworker inspect --registration reg-...
```

The first command reads all latest registrations and returns counts by lifecycle and launch kind. Filters support `--root-issue`, `--parent`, `--session`, `--thread`, `--pid` plus `--process-start`, `--lane`, `--lease`, and `--witness`. `--observed processes.json` joins independently observed process identities and surfaces every unmatched PID/start pair as `UNREGISTERED_OBSERVED`. The second resolves the selected row to its root and renders the complete descendant tree, in human table form by default or stable JSON with `--json`. `FAK_WITNESS_REF` binds the terminal result to an external proof.

### Fleet-level goal metrics

`fak fleet metrics --registration-ledger PATH` joins this graph to the gateway usage ledger. Every registration is grouped by `root_registration_id`, while `root_issue` and `task` remain drill-down labels. A parent, dedicated headless child, nested child, or in-process micro-context therefore contributes to the same `fak_fleet_goal_*` series as long as it registers its execution and carries a session id. The exporter publishes registration/session counts, lifecycle-state counts, observed turns, input/output tokens, and adjudications for each root goal. `fak_fleet_registration_registry_readable` distinguishes an empty graph from an unreadable one.

The join is deliberately structural: ancestry comes from the prelaunch registration graph, not from process nesting or agent narration, and usage joins only on an explicit registered `session_id`. Rows without a session id still count toward the goal's registration lifecycle but do not receive guessed usage attribution.

The store is execution identity, not prompt storage: prompts, credentials, and full command lines are deliberately excluded. Restrict the store to its owner (the writer creates directories `0700` and files `0600`). Archive/retention and cross-host integrity are tracked by #6459. Broader thread/process/guard reconciliation consumes this same record in #6430. The lifecycle contract and dry-run cleanup plan are documented in [`session-lifecycle-reconciliation.md`](session-lifecycle-reconciliation.md); #6431 tracks applying owner-specific terminal repairs.

## Operator recovery

Never edit a row in place. If a process is independently witnessed dead while its row remains active, reconciliation appends `lost` or `reaped` with a reason. Preserve the JSONL for audit, or move the complete file to an access-controlled archive only after all active identities have been reconciled.
