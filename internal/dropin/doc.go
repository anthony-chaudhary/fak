// Package dropin is the canonical drop-in wire resolution + known-agent registry shared by fak guard and the entry-point demo.
//
// Tier: foundation (1) — see internal/architest. This package may import only
// packages whose tier is <= 1; an upward import fails the architest gate.
// See AGENTS.md and internal/architest for the layering contract.
package dropin
