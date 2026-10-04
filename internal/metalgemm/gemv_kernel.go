package metalgemm

import "strings"

// GEMVKernel identifies the P=1 (single-token decode) Q4_K/Q6_K GEMV kernel that reached Metal
// dispatch. Its values equal the native MG_GEMV_EXEC_* identities in q4k.m, so a receipt can
// carry the executed kernel without a translation table. It is portable (no cgo) because
// GraphReceipt, which reports it, exists on every build.
type GEMVKernel int

const (
	// GEMVKernelNone means no P=1 GEMV dispatched (selection declined or nothing encoded).
	GEMVKernelNone GEMVKernel = iota
	// GEMVKernelScalar is the legacy one-simdgroup-per-row q4k_gemv / q6k_gemv parity reference,
	// selected by FAK_Q4K_GEMV_KERNEL=legacy or when the mul_mv pipeline is unavailable.
	GEMVKernelScalar
	// GEMVKernelVectorized is the opt-in q4k_gemv_vectorized float4x4 experiment (Q4_K only).
	GEMVKernelVectorized
	// GEMVKernelMulMv is the default llama.cpp-shaped q4k_mul_mv / q6k_mul_mv kernel (fak#13599).
	GEMVKernelMulMv
)

// String names the kernel the way receipts and logs print it.
func (k GEMVKernel) String() string {
	switch k {
	case GEMVKernelNone:
		return "none"
	case GEMVKernelScalar:
		return "legacy"
	case GEMVKernelVectorized:
		return "vectorized"
	case GEMVKernelMulMv:
		return "mul_mv"
	default:
		return "unknown"
	}
}

// GEMVKernelSet is the set of P=1 GEMV kernels a graph actually encoded, one bit per GEMVKernel
// (bit k set when kernel k ran). It is empty when the graph encoded no P=1 GEMV projection.
type GEMVKernelSet uint32

// Has reports whether kernel k executed at least once.
func (s GEMVKernelSet) Has(k GEMVKernel) bool { return k > GEMVKernelNone && s&(1<<uint(k)) != 0 }

// String lists the executed kernels ("mul_mv", "legacy+mul_mv", or "none").
func (s GEMVKernelSet) String() string {
	var names []string
	for _, k := range []GEMVKernel{GEMVKernelScalar, GEMVKernelVectorized, GEMVKernelMulMv} {
		if s.Has(k) {
			names = append(names, k.String())
		}
	}
	if len(names) == 0 {
		return "none"
	}
	return strings.Join(names, "+")
}
