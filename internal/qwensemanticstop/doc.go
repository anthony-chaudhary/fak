// Package qwensemanticstop validates server-side compute-reclamation receipts.
//
// Invariant: semantic stop evaluation is fail-closed and deterministic.
// Guard: evaluation rejects non-exact models, unproven cancellations,
// and post-cancellation token leakage before promoting to compute_reclaimed semantics.
//
// A receipt is admitted only when ten interleaved exact-model pairs demonstrate that
// disconnect signals cleanly reached the scheduler, stopped generation within the
// declared cancellation bound, and left next-request latency uncontaminated.
package qwensemanticstop
