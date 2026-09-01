// Package rollout bounds the blast radius of new FAK generations by pinning
// new sessions to either a stable or deterministic candidate cohort.
//
// Tier: primitive (1) - see internal/architest. This package may import only
// packages whose tier is <= 1; an upward import fails the architest gate.
// See AGENTS.md and internal/architest for the layering contract.
package rollout
