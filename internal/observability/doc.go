// Package observability provides lightweight, pure-Go telemetry evaluation
// and alarm threshold monitoring across turn token budgets, turn latencies,
// and database health metrics.
//
// Invariant: telemetry evaluation functions operate deterministically and fail-closed;
// any exceeded thresholds, unparseable headers, or corrupt files trigger active warning alarms
// without masking underlying anomalies.
//
// Contract: health inspections never mutate underlying runtime state or disk files;
// header reads inspect only the fixed 100-byte SQLite header without requiring external CGO bindings.
//
// Precondition: caller-provided token counts, latency samples, and database paths must be
// valid references; non-existent or inaccessible file paths result in explicit error alarms rather
// than panics or silent omissions.
package observability
