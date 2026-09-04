// Package dispatchauto is auto-size a multi-account dispatch wave from live ceilings; pure fold, no I/O.
//
// Invariant: automated dispatch planning is fail-closed and deterministic.
// Target concurrency is strictly bounded by the minimum of every set ceiling,
// and zero ready work or zero account session slots immediately yields zero target workers.
//
// Guard: unhealthy nodes contribute zero seat headroom, and workers are never
// placed onto saturated or full nodes. The fold executes without I/O or system clock dependencies.
//
// Tier: foundation (1) — see internal/architest. This package may import only
// packages whose tier is <= 1; an upward import fails the architest gate.
// See AGENTS.md and internal/architest for the layering contract.
package dispatchauto
