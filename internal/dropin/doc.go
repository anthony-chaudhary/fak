// Package dropin is the canonical drop-in wire resolution + known-agent registry shared by fak guard and the entry-point demo.
//
// Tier: foundation (1) — see internal/architest. This package may import only
// packages whose tier is <= 1; an upward import fails the architest gate.
// See AGENTS.md and internal/architest for the layering contract.
//
// Invariants and contracts:
// - Invariant: drop-in plan generation is fail-closed, deterministic, and free of side effects or I/O.
// - Guard: unrecognized agents safely fall back to the default Anthropic wire without panic.
// - Precondition: gateway URLs provided to PlanFor or InjectedEnv must be valid loopback HTTP endpoints.
package dropin
