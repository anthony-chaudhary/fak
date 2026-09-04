// Package quantlicense provides a neutral, evidence-driven gate for source-weight,
// derived-artifact, format, quantizer, and runtime license chains.
//
// Invariant: quant license compatibility evaluation is fail-closed and chain-verifiable.
// Every required permission, attestation, format license, and hardware envelope claim
// must be explicitly supplied and witnessed; unknown terms or unverified assertions
// trigger a refusal or typed abstention rather than an implicit fallback or permissive default.
//
// Guard: evaluate verifies that all source weights, derived artifacts, format specifications,
// and execution runtime components have non-empty licenses with explicit permissions.
// Any missing license evidence or unsupported request usage immediately yields a fail-closed result.
package quantlicense
