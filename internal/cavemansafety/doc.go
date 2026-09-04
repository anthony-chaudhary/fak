// Package cavemansafety evaluates policy value and safety guardrails for the Caveman agent.
//
// Invariant: caveman safety evaluation is fail-closed and deterministic.
// Guard: destructive actions, unauthorized writes, refunds, and prompt injection patterns trigger structural denial without model invocation.
package cavemansafety
