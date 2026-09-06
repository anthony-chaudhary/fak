// Package observability provides lightweight, pure-Go telemetry evaluation
// and alarm threshold monitoring across turn token budgets, turn latencies,
// and database health metrics.
//
// Evaluations are deterministic and operate fail-closed. Database inspections
// read only the fixed 100-byte SQLite file header directly without CGO
// dependencies, locks, or state mutations.
package observability
