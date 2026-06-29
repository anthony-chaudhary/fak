//go:build amd64

package model

import "os"

// quant_amd64_q4k.go — amd64 dispatch for the resident-Q4_K int8 decode reduction, the SIMD sibling
// of the arm64 NEON path (quant_arm64_q4k.go). The AVX2 kernel (quant_amd64_q4k.s) computes the
// per-sub-block integer reductions (I_s = Σ nibble*qx via VPMADDWD, S_s = Σ qx via a ones-vector
// VPMADDWD) for a whole row; the float combine stays in shared Go (q4kCombineRow), so asm
// correctness reduces to "the int32 reductions match the scalar reference" (TestQ4KReduceAsmMatchesScalar).
// Integer VPMADDWD is associative with no overflow on these ranges, so any lane order is bit-identical.
//
// This file owns the amd64 build of three symbols that quant_noasm_q4k.go owns for every other
// non-arm64 arch: q4kReduceRow, q4kSDOTEnabled, and the FAK_KQ_INT8 default. quant_noasm_q4k.go is
// tagged !arm64 && !amd64 so exactly one of {arm64, amd64, noasm} defines each. Falls back to the
// scalar reference when the resolved kernel tier is scalar (no AVX2) or nblk==0.

// Args Isum/Ssum (not IS/SS): on amd64 "SS" is the x86 stack-segment register, so naming the
// frame slot SS makes the assembler parse SS+32(FP) as a register expression. The Go-side caller
// still passes the IS/SS reduction buffers; only the asm-visible parameter names differ.
//
//go:noescape
func q4kReduceRowAsmAVX2(row *byte, nblk int, qx *int8, Isum, Ssum *int32)

//go:noescape
func q4kReduceRowAsmVNNI(row *byte, nblk int, qx *int8, Isum, Ssum *int32)

// q4kVNNI reports whether the AVX512-VNNI Q4_K reducer is usable: AVX512F+BW with OS ZMM state
// (detectAVX512, which the YMM-VNNI form's EVEX encoding still needs) AND CPUID.(7,0):ECX bit 11
// (AVX512_VNNI). On a VNNI box the per-sub-block dot is one VPDPBUSD instead of the AVX2 path's
// 4×VPMOVSXBW+2×VPMADDWD. Resolved once; FAK_QKERNEL=avx2/scalar pins it off for A/B measurement.
var q4kVNNI = func() bool {
	switch os.Getenv("FAK_QKERNEL") {
	case "scalar", "avx2":
		return false
	}
	return detectAVX512() && detectAVX512VNNI()
}()

// detectAVX512VNNI reports CPUID.(EAX=7,ECX=0):ECX bit 11 (AVX512_VNNI). It is a DIFFERENT register
// than detectAVX512's EBX AVX512F/BW bits — a Skylake-X has AVX512 but no VNNI, so emitting
// VPDPBUSD there would #UD. The caller pairs this with detectAVX512 (OS ZMM state + F/BW).
func detectAVX512VNNI() bool {
	_, _, ecx7, _ := cpuid(7, 0)
	const avx512vnni = 1 << 11
	return ecx7&avx512vnni != 0
}

// q4kInt8Default resolves FAK_KQ_INT8 once: the GLM-5.2 mixed-quant offloaded-expert lever. The
// SCALAR int8 reduction already beats the f32 dequant path on amd64 (skips the 256-f32 per-super-block
// dequant); the AVX2 kernel here accelerates it further. Default OFF — the int8 path is APPROXIMATE
// (activation quantization), so it rides FAK_KQ_INT8 until a real-weights witness clears it.
var q4kInt8Default = func() bool {
	switch os.Getenv("FAK_KQ_INT8") {
	case "1", "on", "true":
		return true
	}
	return false
}()

// q4kSDOTEnabled reports whether the resident-Q4_K int8 decode path is active. It does NOT depend on
// the SIMD tier — the scalar int8 reducer is the fallback when AVX2 is absent, and it still beats f32
// — so this tracks only the FAK_KQ_INT8 gate (and the test force). The kernel tier (q4kReduceRow)
// decides scalar-vs-AVX2 independently.
func q4kSDOTEnabled() bool {
	if q4kSDOTForce != 0 {
		return q4kSDOTForce > 0
	}
	return q4kInt8Default
}

func q4kExtractOnceGemmEnabled() bool {
	return false
}

func q4kGemmExtractOnceInt8IntoArch(qt *q4kTensor, qp *q8Panel, Y []float32) bool {
	return false
}

// q4kReduceRow dispatches the integer reduction to the AVX2 kernel when the resolved tier has it,
// else the scalar reference. IS/SS are sized nblk*8 (one I_s/S_s per sub-block across all super-blocks).
func q4kReduceRow(row []byte, nblk int, qx []int8, IS, SS []int32) {
	if nblk > 0 {
		if q4kVNNI {
			q4kReduceRowAsmVNNI(&row[0], nblk, &qx[0], &IS[0], &SS[0])
			return
		}
		if qtier >= tierAVX2 {
			q4kReduceRowAsmAVX2(&row[0], nblk, &qx[0], &IS[0], &SS[0])
			return
		}
	}
	q4kReduceRowScalar(row, nblk, qx, IS, SS)
}
