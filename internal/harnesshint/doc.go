// Package harnesshint emits zero-call, model-relative scope hints for agent harnesses.
//
// Tier: primitive (1) - see internal/architest. This package imports only the
// standard library and performs zero network, disk, or inference calls.
//
// Different language models exhibit sharply divergent operational envelopes when
// driving agent loops: small and flash models require heavy scaffolding, low turn
// bounds, and atomic S0/S1 task decomposition to resist looping and token sprawl;
// balanced mid-tier models support standard multi-step execution; and frontier
// reasoning models require tight turn bounds to avoid runaway per-token inference
// costs.
//
// harnesshint resolves an input model identifier to a deterministic ScopeHint
// through fast in-memory map lookups and prefix normalization.
package harnesshint
