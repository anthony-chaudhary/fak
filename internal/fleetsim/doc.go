// Package fleetsim provides a deterministic synthetic-ledger model for the "safe 400
// GitHub issues/hour parallel-agent throughput" program (issue #1819, fleet-400iph).
//
// Invariant: fleet simulation replay is fail-closed and monotonic.
// Replay operations are pure functions of immutable configuration parameters
// and never produce optimistic close projections when parameters are missing or degenerate.
//
// Guard: simulation parameters enforce strict validation, and replay execution strictly
// halts without mutating shared state or claiming unverified progress.
package fleetsim
