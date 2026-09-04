// Package faultlab provides a fault injection laboratory for agentic serving,
// network stream disruptions, JSON corruption, mid-turn truncation, and simulated kernel faults.
//
// Invariant: fault lab injection is fail-closed and deterministic across all failure modes.
// Guard: unmatched targets bypass interception cleanly without payload mutation or memory allocations.
// Precondition: fault rules must specify non-empty identifiers and recognized fault type enumerations.
// Postcondition: intercepted streams deliver deterministic disruptions or pass through untouched.
package faultlab
