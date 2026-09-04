// Package providerjobaccounting holds repository-level conformance tests for the
// provider-neutral completed-job accounting artifacts defined by issue #9575.
//
// Invariant: provider job accounting is fail-closed and deterministic. Any malformed
// records, unobserved fields, or inconsistent totals reject the ledger.
//
// Guard: ledger validation mandates explicit null counters when provider metrics are
// absent, preventing fabricated telemetry or unverified completion claims.
package providerjobaccounting
