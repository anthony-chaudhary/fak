// Package openviking provides a typed optional client for the OpenViking public
// service contract. It is an independently authored wire adapter: OpenViking's
// storage, retrieval implementation, and SDK code remain outside fak.
//
// Invariant: OpenViking client operations are fail-closed and deterministic.
// Validation rejects invalid inputs before network I/O, base URLs are strictly
// validated without credentials or query fragments, and error messages redact
// configured API keys.
//
// Tier: primitive (1) - see internal/architest. This package may import only
// packages whose tier is <= 1; an upward import fails the architest gate.
// See AGENTS.md and internal/architest for the layering contract.
package openviking
