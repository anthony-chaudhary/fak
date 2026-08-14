// Package humanctl indexes the outcomes humans ask agents to produce and
// provides a small algebra for composing those controls with structured
// modifiers and lossless unstructured context.
//
// The package deliberately separates semantic controls (redirect, verify,
// undo) from delivery controls (send now, queue, inject) and execution policy.
//
// Tier: primitive (1) - see internal/architest. This package may import only
// packages whose tier is <= 1; an upward import fails the architest gate.
package humanctl
