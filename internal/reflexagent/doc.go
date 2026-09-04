// Package reflexagent provides lightweight, fast-spawning micro-agent execution profiles
// for atomic, leaf-level tasks requiring sub-millisecond setup latency and strict lane lease arbitration.
//
// Invariant: Reflex task execution enforces tree-disjoint lane leases; overlapping exclusive leases are refused fail-closed.
//
// Contract:
//   - Micro-agent spawn overhead is bounded and avoids heavy session allocation or multi-agent protocol handshakes.
//   - All lane lease acquisitions through ConcurrencyClassArbiter are released on task completion or failure.
//   - Parallel execution via RunParallel preserves independent lane disjointness per worker.
//
// Guard: Conflicting lane trees return a typed refusal verdict without executing task closures or mutating shared state.
package reflexagent
