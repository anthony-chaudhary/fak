// Package dispatchorder provides the deterministic decision for task dispatch ordering,
// duplicate collapsing, and collision-priced worker admission.
//
// Invariant: dispatch ordering is fail-closed and deterministic.
// The planner relies strictly on explicit data inputs rather than wall-clock readings,
// ensuring identical candidate sets always produce identical ordering, rank, and safe-set partitions.
//
// Guard: candidates with unresolvable tree or compute collisions are serialized before launch,
// and unknown blast radiuses fail closed by colliding conservatively against concurrent participants.
package dispatchorder
