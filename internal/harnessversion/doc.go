// Package harnessversion provides sub-harness runtime multi-versioning,
// sticky session routing, and weighted canary traffic splitting.
//
// Invariants:
//  1. Zero cross-version state corruption: each harness runtime execution
//     resolves to an isolated version descriptor with zero shared mutable
//     state across versions.
//  2. Sticky session pinning: once a session is mapped to a specific harness
//     version, subsequent requests with the same session ID are guaranteed
//     to route to the exact same pinned version until the session is
//     explicitly released.
//  3. Canary traffic splitting: unpinned sessions without explicit wire
//     negotiation are routed probabilistically across active registered
//     versions according to their configured weights, enabling safe canary
//     rollouts and progressive delivery.
//
// Tier: foundation (1) - see internal/architest. Stdlib-only implementation.
package harnessversion
