// Package ultracodedogfood evaluates and verifies UltraCode dogfood lifecycle session replays.
//
// Invariant: UltraCode dogfood lifecycle sessions are fail-closed and deterministic.
// Guard: Ambiguous boundary evidence, missing provider cache receipts, or mismatched outcome digests force an ABSTAIN verdict or validation failure.
// Precondition: Lifecycle session envelope must match the canonical schema with non-empty metadata and valid cell boundaries.
// Postcondition: Scope avoided tokens and cache recoveries are monotonically validated across cold, warm, and clear boundaries.
package ultracodedogfood
