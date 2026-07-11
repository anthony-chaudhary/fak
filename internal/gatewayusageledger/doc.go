// Package gatewayusageledger provides a durable, append-only JSONL ledger for the
// gateway's FULL served-turn counter family (issue #1610, child B of epic #1601):
// kernel submits/vDSO-hits/denies/quarantines, provider-cache economy (read/write
// tokens), compaction, and tool-prune savings. It is the restart-durability sibling
// of internal/cachevalueledger (#1303), which persists ONLY the observed $-economics
// axis; this ledger persists the broader served-turn counter set that today lives
// exclusively in the in-memory gatewayMetrics struct and is lost on every restart.
//
// Rows are OBSERVED counter snapshots — counts and timings only, never prompt or
// secret content — appended with os.O_APPEND so concurrent writers (multiple `fak
// serve`/`fak guard` sessions against the same ledger file) never truncate or
// interleave-corrupt each other's rows. The writer pattern mirrors
// internal/cachevalueledger/ledger.go: NewRow builds one row from a live counter
// snapshot, Append serializes and appends it, ReadLedgerFile/ParseLedger fold the
// file back into a slice, and FoldTrend derives a simple before/after summary a
// caller can use to see counters trending across gateway restarts.
//
// Row idempotency (the ledger-family convention, #2507). The O_APPEND write path
// has no cross-process serialization, so a retried exit flush, a crash-then-rerun
// teardown, or a periodic and an exit flush landing in the same millisecond can
// write two rows for ONE snapshot. To keep trend reports (and, via the dedup census
// #2503, savings claims) from double-counting, every NewRow row carries a
// deterministic RowKey hashed over its identity + payload (schema, session_id, pid,
// unix_millis, and the counter snapshot — NOT the write-time labels, so a retried or
// periodic-vs-exit re-emission of one snapshot shares a key while two genuinely
// distinct snapshots do not). Readers dedupe by key at FOLD time (FoldTrend, via the
// exported DedupeByKey), keeping the first occurrence and reporting the collapse in
// Trend.RowsDedupedAtFold. The write path stays a pure append and the read path a
// pure parse; only the fold collapses duplicates. Dedupe is tolerant of legacy
// keyless rows — a row with an empty RowKey is never treated as a duplicate, so
// historical files (out of scope to rewrite) fold exactly as before. This RowKey +
// fold-time-dedupe discipline is the convention for the whole append-only nightrun
// ledger family; internal/cachevalueledger carries the same key and fold.
//
// Writers never truncate, so the file grows one row per session (plus opt-in
// periodic snapshots) for the whole fleet lifetime. The sanctioned bound is Cut
// (#3490, following the internal/journal cut discipline of #2457): an
// operator-invoked fold-and-truncate that collapses everything older than the
// newest N rows into per-(kind, session_type) carryforward rows whose Counters
// are the exact sum of what they replace, so whole-file counter totals survive
// the cut. Sessions themselves never cut; `fak nightrun cut` is the one door.
package gatewayusageledger
