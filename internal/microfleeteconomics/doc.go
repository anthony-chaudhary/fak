// Package microfleeteconomics deterministically accounts for micro-fleet
// physical costs per accepted result.
//
// Tier: primitive (1) - see internal/architest. This package may import only
// standard-library packages and other primitive leaves.
//
// Invariant: microfleet economics accounting is fail-closed and bounded.
// Every physical cost ledger category is strictly audited for arithmetic overflow,
// and accepted work must be non-zero and bounded by attempted branch work.
package microfleeteconomics
