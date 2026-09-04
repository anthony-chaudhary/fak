// Package shipgate implements RSI-as-ship-gate: propose -> measure -> keep-or-revert
// loops guarded by non-forgeable, evidence-backed keep decisions.
//
// Invariant: shipgate evaluation is fail-closed and evidence-verified. A candidate modification
// is kept only when an external, non-author witness confirms strict metric gains, a passing
// test suite, and clean truth assertions.
//
// Guard: missing, unmeasured, or inconclusive evidence must fail closed to REVERT or HOLD.
// In-band ship/release tool calls require witness corroboration before dispatch.
package shipgate
