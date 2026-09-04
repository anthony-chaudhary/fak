// Package memorycotravel manages cross-session memory transfer and synchronization
// across agent project directories with strict rollout controls and audit logging.
//
// Invariant: memory cotravel evaluation is fail-closed and bounded.
// Guard: operations default to shadow mode to eliminate unverified side effects.
// Contract: memory files are transferred only when allowed by the active strategy and gate.
package memorycotravel
