// Package fleetverify provides compile-time verification and structural
// validation of the fleet brief reporting and health collection helpers.
//
// Invariant: fleet verification reports are fail-closed and bounded.
// Any missing ledger evidence, malformed JSON stream, or mismatched schema
// version triggers an explicit refusal rather than silent default substitution.
//
// Guard: all brief report collections enforce schema invariants and ensure
// ledger non-presence is surfaced as skipped rather than healthy zero.
package fleetverify
