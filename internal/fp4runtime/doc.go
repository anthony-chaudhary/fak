// Package fp4runtime negotiates versioned FP4 and microscaling artifacts
// against exact runtime, GPU-architecture, and accumulator profiles.
//
// It is a compatibility contract, not a quantizer or kernel. Unknown values
// abstain, known incompatible combinations refuse, and external execution is
// returned as an explicit delegation.
//
// Invariant: FP4 runtime evaluations are fail-closed and deterministic.
// Guard: Any unknown schema, missing pin, unverified hardware evidence, or
// unadvertised architecture combination must result in an explicit abstain or refuse verdict.
package fp4runtime
