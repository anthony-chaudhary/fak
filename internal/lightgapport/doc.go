// Package lightgapport audits portability swap points against committed CI witnesses.
//
// Invariant: lightgap swap witness checking is fail-closed and bounded.
// Guard: missing witnesses or unparseable files immediately return non-nil errors.
// Assumption: repository root points to a valid checkout containing committed test witnesses.
//
// Tier: foundation (1) - see internal/architest. This package may import only
// packages whose tier is <= 1; an upward import fails the architest gate.
package lightgapport
