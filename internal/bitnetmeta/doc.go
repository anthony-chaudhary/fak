// Package bitnetmeta describes BitNet-family model artifacts without conflating
// weight semantics with their storage, conversion recipe, runtime, or benchmark.
//
// Invariant: Evaluation operates fail-closed — unrecognized schemas, unknown weight
// semantics, or unverified hardware witnesses abstain or refuse rather than guessing.
// Assumption: Artifact descriptors must explicitly state format versions, origin,
// and discrete quantization levels to guard against precision collapse.
package bitnetmeta
