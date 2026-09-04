// Package toolbound is generic tool output bounding with managed spill files.
//
// Tier: primitive (1) - see internal/architest. This package may import only
// packages whose tier is <= 1; an upward import fails the architest gate.
// See AGENTS.md and internal/architest for the layering contract.
//
// Invariant: tool output bounding is fail-closed and memory-safe.
// Unbounded tool output streams are constrained by strictly enforced limits on
// both line counts and byte lengths, preventing denial-of-service and runaway
// buffer growth.
//
// Guard: complete original content is preserved in managed spill files when limits
// are exceeded, and temporary allocations on failure paths are deterministically cleaned up.
package toolbound
