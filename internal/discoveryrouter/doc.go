// Package discoveryrouter coordinates multi-source discovery routing across
// documentation, active sessions, locator records, and fleet search adapters.
//
// Invariant: discovery routing resolution is fail-closed and deterministic.
//
// Guard: Plan.Run guards against partial failure and unbounded memory growth by clamping result limits, validating adapter responses, and marking coverage incomplete when adapters fail.
//
// Contract:
//   - Discovery routing resolution is fail-closed and deterministic across all query paths.
//   - Adapter failures do not panic; they are recorded as Unavailable with explicit error reasons.
//   - Results are stably ranked across attempted sources by score descending, then owner ascending.
//   - Output sets are strictly bounded by the requested limit.
package discoveryrouter
