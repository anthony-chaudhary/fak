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
// Writers never truncate, so the file grows one row per session (plus opt-in
// periodic snapshots) for the whole fleet lifetime. The sanctioned bound is Cut
// (#3490, following the internal/journal cut discipline of #2457): an
// operator-invoked fold-and-truncate that collapses everything older than the
// newest N rows into per-(kind, session_type) carryforward rows whose Counters
// are the exact sum of what they replace, so whole-file counter totals survive
// the cut. Sessions themselves never cut; `fak nightrun cut` is the one door.
package gatewayusageledger
