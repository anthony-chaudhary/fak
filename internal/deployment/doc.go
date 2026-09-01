// Package deployment provides the gated gen/next contract for deterministic
// derivations, content-addressed realizations, and immutable activation generations.
//
// The three identities are deliberately separate: a derivation proves equal declared
// inputs, a realization proves equal output bytes, and a generation records an atomic
// machine-local selection. Behavioral equivalence always requires an independent witness.
// Importing or materializing a realization never activates it.
//
// Promotion evidence: the contract test must remain green on two machines and a remote
// substitution dogfood receipt must show a compatible realization reused before this moves
// to gen/now. Demote or retire it if generation recovery is not atomic on a supported
// filesystem or cross-machine receipts cannot distinguish build from substitution. The
// invalidating assumption is that rename within one activation filesystem is atomic.
//
// Tier: primitive (1) - see internal/architest. This package may import only
// packages whose tier is <= 1; an upward import fails the architest gate.
package deployment
