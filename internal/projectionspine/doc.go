// Package projectionspine provides a small authority/projection harness.
//
// Durable session identity, writer ownership, transcript position, and effect
// receipts live in an Authority. Disposable projections attach to that state
// and may be replaced without restarting or promoting a new writer.
package projectionspine
