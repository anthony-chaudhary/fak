// Package schemaadapter is centralized multi-provider tool schema contract adapter.
//
// Tier: primitive (1) - see internal/architest. This package may import only
// packages whose tier is <= 1; an upward import fails the architest gate.
// See AGENTS.md and internal/architest for the layering contract.
//
// Invariant: schema adapter transformations are fail-closed and lossless across supported dialects.
// Guard: empty, malformed, or non-object schemas are rejected immediately before dialect sanitization.
package schemaadapter
