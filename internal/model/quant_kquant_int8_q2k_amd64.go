//go:build amd64

package model

// quant_kquant_int8_q2k_amd64.go — amd64 dispatch for the resident Q2_K int8 decode reduction, the
// Q2_K sibling of the Q5_K/Q6_K reducers in quant_amd64_kquant.go. The AVX2 kernel
// (quant_kquant_int8_q2k_amd64.s) computes the per-sub-block integer reductions
// (I_s = Σ q2*qx, S_s = Σ qx for a 16-lane sub-block) for a whole row; the float combine stays in
// shared Go (q2kCombineRow), so asm correctness reduces to "the int32 reductions match the scalar
// reference" (TestQ2KReduceAsmMatchesScalar). Integer addition is associative with no overflow on
// these ranges, so any SIMD lane order is bit-identical.
//
// MIT provenance: direct port/adapt of the integer reduction of ggml_vec_dot_q2_K_q8_K,
//   upstream:   ggml-org/llama.cpp
//   commit pin: 83078fec0db82d6b5a00d9599062c38c39145755
//   source:     ggml/src/ggml-cpu/arch/x86/quants.c
//   license:    MIT (upstream LICENSE at the pin); Copyright (c) 2023-2026 The ggml authors
// Native Go assembly only — no cgo/FFI.

// Args Isum/Ssum (not IS/SS): on amd64 "SS" is the x86 stack-segment register, so naming the
// frame slot SS makes the assembler parse SS+32(FP) as a register expression. The Go-side caller
// still passes the IS/SS reduction buffers; only the asm-visible parameter names differ.
//
//go:noescape
func q2kReduceRowAsmAVX2(row *byte, nblk int, qx *int8, Isum, Ssum *int32)

// q2kUseVNNI is read by the q2k asm inner dot (CMPB ·q2kUseVNNI(SB)) to pick the one-VPDPBUSD-per-
// dot VNNI fast path over the AVX2 sign-extend path — same q2kReduceRowAsmAVX2 entry/unpack either
// way, only the inner reduction differs. 1 when the box has AVX512-VNNI (and the tier isn't pinned
// down via FAK_QKERNEL). A byte, not a bool, so the asm's CMPB matches its width exactly.
var q2kUseVNNI byte = func() byte {
	if q4kVNNI { // same CPUID gate (AVX512 + VNNI ECX bit 11), same FAK_QKERNEL pin
		return 1
	}
	return 0
}()

// q2kReduceRow dispatches the Q2_K integer reduction to the AVX2/VNNI kernel when the resolved tier
// has AVX2, else the scalar reference. The asm picks VNNI vs AVX2 internally via q2kUseVNNI. IS/SS
// are sized nblk*16 (one I_s/S_s per 16-lane sub-block across all super-blocks).
func q2kReduceRow(row []byte, nblk int, qx []int8, IS, SS []int32) {
	if nblk > 0 && qtier >= tierAVX2 {
		q2kReduceRowAsmAVX2(&row[0], nblk, &qx[0], &IS[0], &SS[0])
		return
	}
	q2kReduceRowScalar(row, nblk, qx, IS, SS)
}
