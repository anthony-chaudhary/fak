// Package depthadmit is pure depth fold: witnessed plan-phase coverage, the depth frontier, and the closure/persistence admission that drives one line of work to declared depth.
//
// Tier: primitive (1) - see internal/architest. This package may import only
// packages whose tier is <= 1; an upward import fails the architest gate.
// See AGENTS.md and internal/architest for the layering contract.
//
// Invariant: depth admission decisions are fail-closed and monotonic. A terminal closure
// claiming goal completion (ClosureMet) is refused unless every declared phase in the plan
// has been explicitly carried with witnessed evidence.
//
// Contract: input evaluation is pure, total, and deterministic. Every input produces exactly
// one report belonging to the closed verdict vocabulary, with zero I/O, zero clock access,
// and zero external mutation.
//
// Guard: foreign or malformed phase identifiers fail closed immediately. An uncheckable plan
// or unknown closure token is treated as an explicit refusal rather than permitted by default.
package depthadmit
