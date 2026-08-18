# Incremental session-history index

`fak vcache session-mine` can retain a privacy-safe index across runs instead of reparsing every historical transcript:

```bash
fak vcache session-mine --days 7 --index ~/.fak/session-mine/index.json
```

The JSON response includes the full aggregate `report`, `parsed_files`, `reused_files`, and `new_candidates`. `new_candidates` contains only candidate fingerprints that crossed the current support threshold for the first time, so a scheduler can run the command repeatedly without re-notifying reviewed work.

The index stores normalized per-session counts and bounded tool trajectories plus one-way source fingerprints. It never stores transcript paths, prompts, tool arguments, results, or provider-private tool names. Writes use a same-directory temporary file and atomic rename; an unknown schema is rejected rather than overwritten.

This is the foundational background-process seam: Task Scheduler, cron, or a long-lived supervisor can invoke the bounded command on a cadence while aggregate and drill-down query surfaces read one durable, incrementally refreshed artifact.
