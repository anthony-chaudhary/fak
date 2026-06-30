// Package programreport folds the project's ONGOING optimization PROGRAMS — the
// work classes internal/worktype marks as never-"done" frontiers (kernel-optimization
// and cache-optimization) — into one read-only report envelope with a durable JSONL
// trend ledger. It is the ongoing-program sibling of internal/milestonereport.
//
// The distinction the two reports draw together (see internal/worktype for the law):
//
//   - milestonereport's ROADMAP measures DISCRETE epics by child-completion % — the
//     right lens for a deliverable that converges on 100% and closes.
//   - programreport measures the two ONGOING programs by a FRONTIER + a TREND — the
//     right lens for a process that is never done. There is no completion % here on
//     purpose; a "60% complete" line on kernel-opt or cache-opt is a category error.
//
// The two program frontier signals (both WITNESSED, never self-reported):
//
//   - KERNEL OPTIMIZATION — the trailing-window count of ships stamped on a perf/kernel
//     leaf (the decode/prefill/quant/parity lanes), read from git through the SAME
//     hooks.StampOf grammar the pre-commit lint binds to. It is an activity proxy,
//     honestly labeled: it asserts the program is being worked, not a tok/s number
//     (the throughput claim lives in the benchmark authority rows). A quiet window is
//     HOLDING, not regressed.
//   - CACHE OPTIMIZATION — the realized KV-reuse ratio over the dogfood cache-value
//     ledger, read through cachevalueledger's #1066-fenced trend gate (the same gate
//     `fak cachevalue` enforces, so the two never disagree). The honesty fence (the
//     marginal-over-tuned-warm-KV value family) is carried onto the signal.
//
// The report is a REPORT CONTRACT, not a second quality gate (the milestonereport
// posture): --check fails ONLY when the programs dimension could not be MEASURED. A
// regressed frontier is a MEASURED fact surfaced as an advisory line; the per-program
// ratchet (the cache-value trend gate, the perf-parity RSI loop) owns the real gate.
//
// The pure surface (the interpreter, the fold, the ledger parse/trend, render, gate)
// lives in programreport.go and is unit-testable with no process and no repo. The
// impure runners (the cache-value ledger read + the perf-lane git window) live in
// collect.go.
package programreport
