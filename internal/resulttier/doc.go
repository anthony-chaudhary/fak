// Package resulttier provides standard bounded result tier allocation and pagination
// primitives for agent kernels and tool execution results.
//
// Invariant: result tier assignment is fail-closed and deterministic across all inputs.
// Guard: requests exceeding MaxStandardTier without an audited widening reason fail closed with ErrTierWideningRefused.
package resulttier
