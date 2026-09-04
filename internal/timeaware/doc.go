// Package timeaware provides deterministic accounting and health signals for agent work.
// It records monotonic, half-open spans; attributes them to execution dimensions; and
// deliberately keeps elapsed time, aggregate effort, active-union time, and critical
// path separate.
//
// Invariant: time-aware span aggregation is fail-closed and deterministic.
// Any span failing schema validation, bearing negative monotonic timestamps, or specifying
// inverted start/end boundaries is excluded from duration rollups and counted as an invalid span.
//
// Guard: aggregation preserves strict separation between elapsed wall-clock duration,
// aggregate worker effort, active-union time, and critical-path latency. Waiting, stalled,
// or unclassified intervals are never conflated with active execution effort.
//
// Precondition: caller-supplied monotonic timestamps must be non-negative half-open bounds [start, end).
// Duplicate span identifiers are safely dropped into duplicate counters without corrupting metrics.
package timeaware
