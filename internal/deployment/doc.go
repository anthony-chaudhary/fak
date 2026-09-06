// Package deployment provides deterministic derivations, content-addressed realizations,
// and immutable activation generations.
//
// The three identities are deliberately separate: a derivation proves equal declared
// inputs, a realization proves equal output bytes, and a generation records an atomic
// machine-local selection. Materializing a realization does not activate it.
//
// Tier: primitive (1) - see internal/architest.
package deployment
