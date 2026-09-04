// Package projectionspine provides a small authority/projection harness.
//
// Durable session identity, writer ownership, transcript position, and effect
// receipts live in an Authority. Disposable projections attach to that state
// and may be replaced without restarting or promoting a new writer.
//
// Invariant: projection replacement is fail-closed and bounded.
// Disposable projections observe authority state without owning writer epochs.
// When a projection terminates, authority state and idempotent effect receipts remain preserved.
// Guard: reattach decisions must obey supervision policy restart budgets and escalate on exhaust.
package projectionspine
