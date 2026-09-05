// Package stopgate unifies lifecycle stop gates across fak guard and the fak agent native harness (#11253).
//
// Both fak guard (the headless agent watchdog) and fak agent (the native harness) must
// make identical lifecycle decisions at turn boundaries:
//   1. Graduated deny-all ladder: auto-continue past capability floor refusals (nudge -> warn -> final),
//      with bounded give-up (stand-down) to prevent infinite loops.
//   2. Tool feedback bounds: bounded continuation for retryable/malformed tool calls.
//   3. Witness gating: require verified artifacts/commits before accepting self-narrated completion.
//   4. Clean wrap-up: recognize when an agent gracefully concludes on a protected boundary.
//
// This package is pure and deterministic (stdlib-only, tier 1), providing total decision
// evaluation and synthesized continuation guidance across all execution postures.
package stopgate
