---
title: "Nightly trajectory attribution receipt"
description: "Source-side local/fleet audit contract, versioned budgets, receipt history, and rollback."
---

# Nightly trajectory attribution receipt

`trajectory-attribution-nightly.yml` runs `fak trajectory nightly` daily on two
explicitly labelled self-hosted lanes: `fak-local` and `fak-fleet`. Raw Claude
and Codex transcripts stay on those hosts. The workflow uploads only the
content-free `fak-trajectory-attribution-receipt/1` row.

## Source and budget contract

Scheduled runs read roots from repository variables:

- `FAK_TRAJECTORY_LOCAL_CLAUDE_ROOT` and `FAK_TRAJECTORY_LOCAL_CODEX_ROOT`;
- `FAK_TRAJECTORY_FLEET_CLAUDE_ROOT` and `FAK_TRAJECTORY_FLEET_CODEX_ROOT`.

A manual dispatch can override all four roots. An unset root resolves to a
known-missing sentinel under `${{ runner.temp }}`; it does not fall back to a runner's
home directory. The resulting receipt is `no_data` with `root_present=false`,
so a bare checkout cannot claim local or fleet coverage. Permission, traversal,
or read failures produce `collection_failed`, which is distinct from `no_data`.

[`configs/trajectory-attribution-nightly.v1.json`](../../configs/trajectory-attribution-nightly.v1.json)
is the versioned budget consumed by the CLI and workflow. It bounds the recent
window, directory entries walked, files, bytes actually read, and path-hashed
sample coordinates. Known budgeted record subtypes remain named; unknown
transcript-authored subtype labels are hash-tokenized. Per-source subtype rows
are deduplicated and cardinality-bounded; an overflow is counted, sampled, and
budgeted instead of growing the JSONL history row without limit. The budget gates unknown share,
duplicates, malformed rows, unmatched tool events, schema-drift signals, and
files omitted by the scan ceiling. A budget breach appends the receipt and exits
3, making the scheduled job red without losing its diagnostic artifact.

## Producers and consumers

The producer writes a temporary latest receipt, appends one scrubbed row per
host to `~/.fak/nightrun/trajectory-attribution.jsonl`, then atomically publishes
the staged receipt. A failed latest publication rolls back that new history row;
a failed history append discards the staged success and publishes
`publication_failed`, separately from source `collection_failed`.
The prior same-corpus row supplies comparable metric deltas;
`no_data` and `collection_failed` rows are explicitly non-comparable. Budget
schema/version, window, source coverage, and scan exposure must also match;
otherwise the receipt records a typed non-comparability reason instead of a
misleading delta.

Consumers are the Actions step summary (rendered from the scrubbed receipt), the
30-day uploaded local/fleet receipt
artifacts, and operators reading the host-local append-only history. Neither the
workflow nor the CLI commits telemetry automatically. Reviewed publication is
an explicit by-path operation, and must never include raw transcripts.

## Cutover and rollback

Cut over by registering source-side runners with `trajectory-audit` plus their
`fak-local` or `fak-fleet` label, then set the four named root variables. Run a
manual dispatch first and confirm populated `files_scanned`, `records`, and a
`pass` or actionable `budget_failed` receipt before relying on the cron.

No matching Actions runners or `FAK_TRAJECTORY_*` variables are provisioned by
this repository. Until an operator performs that cutover, the scheduled jobs
remain dormant in the runner queue; a provisioned runner with unset roots emits
honest `no_data`. The checked workflow-entrypoint test is a populated local
contract witness, not a claim that a hosted scheduled run has occurred.

Rollback by disabling the workflow or removing the `trajectory-audit` runner
label. This stops new reads and appends without rewriting receipt history. To
roll back a budget revision, restore the prior checked-in budget file and
workflow reference together; old rows remain readable because every receipt
records both budget schema and version.
