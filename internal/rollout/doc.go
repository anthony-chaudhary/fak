// Package rollout bounds the blast radius of new FAK generations by pinning
// new sessions to either a stable or deterministic candidate cohort.
//
// Invariant: rollout selection is fail-closed and deterministic across session cohorts.
// Any invalid configuration, missing generation, or killed canary strictly falls back
// to the verified stable or last-known-good generation.
//
// Guard: active canary count must not exceed concurrent cap, and cohort bucket must
// fall strictly within allocated basis points.
//
// Tier: primitive (1) - see internal/architest. This package may import only
// packages whose tier is <= 1; an upward import fails the architest gate.
// See AGENTS.md and internal/architest for the layering contract.
package rollout
