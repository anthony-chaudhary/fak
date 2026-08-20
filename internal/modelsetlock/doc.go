// Package modelsetlock persists canonical model-set selections with
// digest-bound, fail-closed readback.
//
// Tier: foundation-composite (2) - see internal/architest. This package may import only
// packages whose tier is <= 2; an upward import fails the architest gate.
// See AGENTS.md and internal/architest for the layering contract.
package modelsetlock
