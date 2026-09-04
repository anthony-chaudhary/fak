// Package market validates discoverable extension descriptors without executing extension code.
//
// Invariant: market descriptor parsing and verification are fail-closed and deterministic.
// Guard: untrusted extension descriptors must never execute external code or load arbitrary host binaries.
// Precondition: caller provides isolated ABI version maps and rooted workspace directories.
//
// Tier: composer (3) - see internal/architest. This package may import only
// packages whose tier is <= 3; an upward import fails the architest gate.
package market
