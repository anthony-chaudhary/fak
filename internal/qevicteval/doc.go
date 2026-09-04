// Package qevicteval evaluates the recoverable quantized KV eviction recipe named
// by QEvict (arXiv:2608.05326v1) against ordinary irreversible eviction.
//
// Invariant: Q-eviction evaluation is fail-closed and deterministic across all evaluation traces.
// Guard: Requests with unknown contract versions, mismatched recipes, or invalid artifact digests are rejected with DecisionAbstain.
// Precondition: Window events in a trace must have strictly monotonic step indices and valid byte allocations.
package qevicteval
