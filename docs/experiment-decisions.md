# Evaluate and replay native performance decisions

Use the existing transaction gate to evaluate a supplied profile, selected
obligation and matched baseline/candidate receipts:

```text
fak experiments native-performance evaluate --file request.json --json
fak experiments native-performance decisions --limit 20 --json
fak experiments native-performance replay --file request.json --decision journal.json --json
```

`evaluate` appends the validated decision to
`.fak/nativeperf-decisions.jsonl` under the repository root. Override this with
`--ledger FILE`. A valid KEEP or REJECT exits zero. Invalid requests exit nonzero
without appending. Append I/O errors can leave a record: inspect the ledger with
`decisions` before retrying. Duplicate decision identities are refused. KEEP
reflects the existing non-regression policy; it does not imply a measured speedup
or promotion. To replay, save the `journal` object from evaluation as
`journal.json`. Replay recomputes that object without writing the ledger.

The request uses the existing `NativePerfTransactionRequest` fields:
`profile`, `obligation_graph`, `obligation_graph_digest`, `obligation_id`,
`candidate_id`, `gate_request` and `default_readback`. The graph digest is
`sha256:` followed by the SHA-256 of its compact Go JSON encoding. Existing
profile, graph-selection, gate and readback requirements remain authoritative.
The profile and both gate receipts must explicitly supply a non-null
`execution.fallback_count`; an omitted or null value cannot establish zero.
Request and journal files each have a 1 MiB input limit and reject unknown fields
or trailing JSON values.

`decisions` validates a bounded ledger tail and returns its latest complete rows
in ledger order.
Its defaults are 20 rows and 1 MiB; `--limit` accepts 1–100 and `--max-bytes`
accepts 1 byte–16 MiB. It captures the file size and reads at most one additional
byte to align the first complete row. `window_start` and `omitted_bytes` report the
omitted prefix; `file_bytes` reports the captured size. An omitted prefix leaves
`total_rows` and `omitted_rows` unknown; `rows_read`, `bytes_read` and
`omitted_rows_at_least` describe only inspected evidence. Row and byte truncation
are reported separately. The reader refuses an invalid or incomplete final row,
or a byte window too small to retain the latest row. Increase `--max-bytes` in
that case. A missing ledger returns `present=false` with unknown totals without
creating a file; an existing empty file returns `present=true` with zero totals.

These commands validate supplied evidence. They do not run a model, authenticate
hardware execution or change serving. Output always states
`hardware_verified=false`, `default_observation=caller-supplied` and
`serving_mutated=false`. The current transaction graph supports its named
Qwen3.8 Metal and CUDA envelopes. DS/Vulkan and automatic correspondence with
compiled onboarding obligations require separate adapters; this command does
not supply those capabilities.


### SelectedCandidateID (native-performance decision)

NativePerfDecisionJournal.SelectedCandidateID identifies the selected obligation node promotion witness bound into one replayable native-performance gate decision.

**Distinct from:** This is the immutable decision journal binding to a supplied graph candidate; Candidate is a scored resident span considered by the context planner. The journal identifier does not establish hardware execution or serving promotion.
