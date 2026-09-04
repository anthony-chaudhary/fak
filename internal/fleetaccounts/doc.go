// Package fleetaccounts provides account discovery, capability routing, status folding,
// and lifecycle tracking across Claude Code, Codex, and opencode worker accounts.
//
// Invariant: fleet account routing is fail-closed and secret-safe across all product families.
// Guard: blocked or unverified credentials immediately trigger refusal rather than silent fallback or auth leakage.
package fleetaccounts
