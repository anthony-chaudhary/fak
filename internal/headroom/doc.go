// Package headroom is the context-compression seam: a pluggable Compressor area (headroom/native/noop plugins) folded into the result path as a ResultAdmitter.
//
// Tier: composer (3) — see internal/architest. This package may import only
// packages whose tier is <= 3; an upward import fails the architest gate.
// See AGENTS.md and internal/architest for the layering contract.
package headroom
