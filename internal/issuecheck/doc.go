// Package issuecheck defines the pure contract for the agent-selected Top-5
// review that precedes implementation of a worker-ready GitHub issue.
//
// Invariant: issue check reviews are fail-closed and deterministic.
// Any schema mismatch, content tampering, or issue digest drift causes immediate rejection.
//
// Guard: comment actions refuse ambiguous multiple managed comments and never allow
// unauthenticated or malformed review payloads to mutate trusted state.
package issuecheck
