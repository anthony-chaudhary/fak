// Package mixedprecision defines a neutral, deterministic contract for layerwise
// precision assignment, coverage and evidence. It adjudicates metadata; it does
// not execute quantization or infer hardware performance.
//
// Invariant: mixed precision operations are fail-closed and bounded.
// Undeclared combinations, ambiguous module patterns, unpinned versions, or
// unhandled fallbacks always refuse execution deterministically.
// Guard: precision assignments and parameter budgets are checked against overflow
// and validated against an explicit closed support matrix before acceptance.
//
// Tier: foundation (1) - see internal/architest.
package mixedprecision
