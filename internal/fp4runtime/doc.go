// Package fp4runtime negotiates versioned FP4 and microscaling artifacts
// against exact runtime, GPU-architecture, and accumulator profiles.
//
// It is a compatibility contract, not a quantizer or kernel. Unknown values
// abstain, known incompatible combinations refuse, and external execution is
// returned as an explicit delegation.
package fp4runtime
