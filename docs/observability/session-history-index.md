---
title: "Incremental session-history index"
description: "fak vcache session-mine can retain a privacy-safe index across runs instead of reparsing every historical transcript:"
---
# Incremental session-history index

`fak vcache session-mine` can retain a privacy-safe index across runs instead of reparsing every historical transcript:

```bash
fak vcache session-mine --days 7 --index ~/.fak/session-mine/index.json
```

The JSON response includes the full aggregate `report`, `parsed_files`, `reused_files`, and `new_candidates`. `new_candidates` contains only candidate fingerprints that crossed the current support threshold for the first time, so a scheduler can run the command repeatedly without re-notifying reviewed work.

The index stores normalized per-session counts and bounded tool trajectories plus one-way source fingerprints. It never stores transcript paths, prompts, tool arguments, results, or provider-private tool names. Writes use a same-directory temporary file and atomic rename; an unknown schema is rejected rather than overwritten.

This is the foundational background-process seam: Task Scheduler, cron, or a long-lived supervisor can invoke the bounded command on a cadence while aggregate and drill-down query surfaces read one durable, incrementally refreshed artifact.


## Retrospect from aggregate to one session

Read the index without touching provider transcripts:

```bash
fak vcache session-history --index ~/.fak/session-mine/index.json
fak vcache session-history --index ~/.fak/session-mine/index.json --provider codex --min-errors 1
fak vcache session-history --index ~/.fak/session-mine/index.json --tool view_image
fak vcache session-history --index ~/.fak/session-mine/index.json --session codex-0123456789ab
```

`indexed_at` is the newest normalized session-end timestamp represented by the index (or `all` when empty), not the wall-clock time a scan happened. `indexed_at` is the newest normalized session-end timestamp represented by the index (or `all` when empty), not the wall-clock time a scan happened. The first view returns corpus metrics and sessions ranked by tool errors, recency, then stable ID. Provider, minimum-error, and exact normalized `--tool STEP` filters recalculate the aggregate over the selected slice. The tool filter gives a direct aggregate to matching sessions to ID drill-down path without full-text transcript search. `--session` returns one normalized record and its bounded anonymous trajectory, providing a direct drill-down without reopening or exposing the raw transcript.


## Keep the index current

Run one refresh from a scheduler:

```bash
fak vcache session-history refresh --once
```

Or let an existing supervisor keep a bounded worker alive:

```bash
fak vcache session-history refresh --interval 15m --max-runs 96
```

The worker refreshes immediately before waiting, writes the same atomic privacy-safe index, and emits one `fak-session-history-refresh/1` receipt per run. Receipts contain only counts (`parsed_files`, `reused_files`, sessions, candidates); cancellation is a clean stop. `--index`, provider roots, horizon, support, and candidate limit remain configurable, while the defaults cover the normal local Codex and Claude stores under the user's home directory.
