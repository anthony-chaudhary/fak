//go:build !amd64

package model

// quant_kquant_int8_q2k_noamd64.go — the Q2_K integer-reduction dispatch for archs with no SIMD
// Q2_K reducer. It is the scalar-fallback twin of quant_kquant_int8_q2k_amd64.go: amd64 has the
// AVX2/VNNI kernel; every other arch routes the per-sub-block reduction to q2kReduceRowScalar, the
// arch-neutral reference in quant_kquant_int8_q2k.go.
//
// MIT provenance: direct port/adapt of llama.cpp ggml_vec_dot_q2_K_q8_K at
// ggml-org/llama.cpp commit 83078fec0db82d6b5a00d9599062c38c39145755, source
// ggml/src/ggml-cpu/arch/x86/quants.c (MIT; upstream LICENSE at the pin; Copyright (c) 2023-2026 The ggml authors).
// Native Go, no cgo/FFI.

// q2kReduceRow computes the per-sub-block (I_s = Σ q2*qx, S_s = Σ qx) reductions for a Q2_K row.
func q2kReduceRow(row []byte, nblk int, qx []int8, IS, SS []int32) {
	q2kReduceRowScalar(row, nblk, qx, IS, SS)
}
