// Package humanctl indexes the outcomes humans ask agents to produce and
// provides a small algebra for composing those controls with structured
// modifiers and lossless unstructured context.
//
// The package deliberately separates semantic controls (redirect, verify,
// undo) from delivery controls (send now, queue, inject) and execution policy.
//
// Invariant: human control directives are fail-closed and bounded.
// Terminal controls (stop, pause, end_turn) cannot be composed before other
// instructions; unacknowledged controls cannot report admission decisions;
// and unobserved effects reject missing witnesses.
//
// Invariant: human control semantics remain decoupled from transport delivery mechanisms.
// Guard: non-compliant upward imports violate the architectural tier constraint (tier <= 1).
//
// Tier: primitive (1) - see internal/architest. This package may import only
// packages whose tier is <= 1; an upward import fails the architest boundary check.
package humanctl
