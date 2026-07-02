// Package laneadmit is the shared lane/tree admission decision every execution
// surface asks before acting on the shared tree: dispatch workers, `fak loop
// drive` runs, and manual sessions (via `fak loop coord`).
//
// It closes the gap where the fleet ran two lease vocabularies with no bridge:
// leaseref lock leases checked raw tree geometry only, while the dos.toml
// [lanes] taxonomy's semantics (a named lane serializes, an exclusive lane runs
// alone) lived only in the external `dos arbitrate`. Decide() is the in-binary
// twin of that arbitrate contract — lane modes + tree geometry + the live lease
// set, refusing with the closed-vocabulary COLLISION_RISK — so every surface
// can afford to ask the same question at its act boundary, and a loop, a
// dispatch worker, and a manual session become mutually visible through one
// lease namespace (refs/fak/locks/*, internal/leaseref).
//
// The package is pure: no clock, no I/O. Callers supply the live leases (from
// leaseref) and the taxonomy (ParseTaxonomy over dos.toml bytes). It shares
// leaseref's honest scope — local visibility, not cross-host atomicity.
//
// Tier: foundation (1) — see internal/architest. This package may import only
// packages whose tier is <= 1; an upward import fails the architest gate.
// See AGENTS.md and internal/architest for the layering contract.
package laneadmit
