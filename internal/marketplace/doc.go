// Package marketplace validates discoverable extension descriptors without executing extension code.
//
// Tier: composer (3) - see internal/architest. This package may import only
// packages whose tier is <= 3; an upward import fails the architest gate.
package marketplace
