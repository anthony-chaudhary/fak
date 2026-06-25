// Package vcachescore composes the vCache proof leaves into an agent-facing
// benchmark scorecard.
//
// The lower vCache leaves answer narrow questions: calibration estimates warmth,
// star proves planned token savings, telemetry proves realized savings, chain
// recall proves/refutes rebuild economics, and the governor classifies warm-set
// policy. This package keeps those proofs intact and folds them into one
// operator artifact: "is this workload at least 2x better, what index should the
// agent build, and what action moves it closer?"
//
// It is still a pure off-path decision layer. It issues no provider calls, warms
// no prefix, and treats cache hits as rebates only. Correctness never depends on
// a provider cache hit.
//
// Tier: mechanism (2) - see internal/architest. It imports only the vCache
// mechanism leaves and the standard library, and is not registered into the
// request path.
package vcachescore
