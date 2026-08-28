// Package openviking provides a typed optional client for the OpenViking public
// service contract. It is an independently authored wire adapter: OpenViking's
// storage, retrieval implementation, and SDK code remain outside fak.
//
// Tier: primitive (1) - see internal/architest. This package may import only
// packages whose tier is <= 1; an upward import fails the architest gate.
// See AGENTS.md and internal/architest for the layering contract.
package openviking
