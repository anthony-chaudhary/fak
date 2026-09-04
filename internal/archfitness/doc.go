// Package archfitness scores composition architecture debt, distinct from file-size quality checks.
//
// Invariant: architectural fitness evaluation is fail-closed and deterministic.
// Precondition: callers provide structured findings categorized across known architectural dimensions.
// Guard: ratchet evaluation prevents regressions in hard debt counts across all architectural dimensions.
package archfitness
