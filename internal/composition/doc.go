// Package composition resolves an immutable pre-allocation execution graph.
//
// Invariant: composition resolution is fail-closed and bounded.
// Precondition: caller supplies a fully populated Snapshot with an explicit policy identifier.
// Guard: unsupported execution backends or undeclared resource capabilities fail closed before allocation.
package composition
