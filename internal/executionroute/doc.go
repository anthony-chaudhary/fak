// Package executionroute composes harness, model, and session routing into one
// inspectable execution decision without collapsing their distinct policies.
//
// Tier: composer (3) - see internal/architest. This package may import only
// packages whose tier is <= 3; an upward import fails the architest gate.
package executionroute
