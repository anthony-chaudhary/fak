// Package vcachewarm is the vCache M3 dedicated-warming decision layer.
//
// The package is deliberately off the live provider path: it decides which
// warming primitive a caller may spend, where an Anthropic explicit breakpoint
// belongs, when the first-real-request ordering path is required, when a fanout
// barrier may release dependents, and how to account for a warm that never reads
// back from the provider cache. It issues no network calls and does not claim a
// live transport.
//
// Tier: mechanism (2) - see internal/architest. This package imports only the
// standard library; an upward import fails the architest gate. It is not
// registered into the kernel.
package vcachewarm
