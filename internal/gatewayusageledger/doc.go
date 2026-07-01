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
package gatewayusageledger
