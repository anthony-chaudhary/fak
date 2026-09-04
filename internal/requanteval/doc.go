// Package requanteval provides a versioned, neutral evaluation contract for
// ReQuant-style fixed-grid discrete refinement.
//
// Invariant: requant evaluations are fail-closed and deterministic.
// Guard: all inputs, grids, Hessians, and provenance fields must validate completely before optimization.
package requanteval
