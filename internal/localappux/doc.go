// Package localappux renders host-app language for local compute lifecycle states.
//
// Invariant: UX rendering is deterministic and fail-closed across all lifecycle states.
// Guard: Diagnostic export enforces strict redaction of sensitive identifiers, tokens, and prompt context before external serialization.
// Assumption: All state transitions resolve to well-defined lifecycle copy with safe fallback actions.
// Precondition: Caller supplies valid View configuration or structured Diagnostic values.
// Postcondition: Rendered strings and preview diagnostics contain zero unescaped credentials or raw prompt tokens.
package localappux
