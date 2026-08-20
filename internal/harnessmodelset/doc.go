// Package harnessmodelset declares strict role-indexed model requirements for
// generated harnesses.
//
// Tier: primitive (1) - see internal/architest. This package may import only
// packages whose tier is <= 1; an upward import fails the architest gate.
// See AGENTS.md and internal/architest for the layering contract.
package harnessmodelset
