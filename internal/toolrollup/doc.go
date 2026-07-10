// Package toolrollup folds a corpus of individual tool-call records into a
// per-tool-TYPE rollup — the read-only aggregate a session-analytics report
// renders (issue #2824, session-analytics C2).
//
// # The fold
//
// The input is a flat corpus of [ToolCall] records drawn from one or more agent
// trajectories: each record is one call of one tool, carrying its token cost, its
// wall duration, and whether it succeeded. [Rollup] groups those records by tool
// name and, for each distinct tool, aggregates the call count, the total and mean
// input/output tokens, the total and mean wall duration, the error count and rate,
// and the share of all calls the tool accounts for. The result is a deterministic
// slice of [ToolStat], sorted by call count descending then tool name ascending, so
// two runs over the same corpus render byte-identically.
//
// # Corpus shape
//
// [ReadCorpus] reads the on-disk corpus: JSONL, one JSON [ToolCall] per line, the
// same one-record-per-line shape the trajectory corpus (internal/trajectory's
// ExportTo) and the trajquery `--corpus` reader use. The record's JSON field names
// mirror internal/trajectory's Turn where they overlap (`tool`), and follow the
// wider repo convention for the rest (`input_tokens` / `output_tokens`,
// `duration_ms`, `ok`) so a row is legible alongside the other JSONL ledgers.
//
// The package is self-contained and stdlib-only: it imports nothing internal (not
// even internal/trajectory), so it sits at the foundation of the architest tier DAG
// and never imports "upward".
package toolrollup
