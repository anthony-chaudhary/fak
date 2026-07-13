package compute

// w8a8_expert_gemm.go — a pure-Go W8A8 (int8 activation × int8 weight) expert-GEMM
// *reference* for the GLM-5.2 MoE expert path (#3087). It is the laptop-composable
// first step of the sm_80 INT8 tensor-core expert GEMM: the GPU kernel is hardware-
// gated (needs an A100-class device with IMMA/DP4A int8 tensor cores), but the
// arithmetic it must reproduce is portable, so — exactly as the Q4_K CPU decode
// kernel landed before its GPU witness — the CPU reference + accuracy gate ship
// first and pin the numeric contract the device kernel is later held to.
//
// What a real sm_80 int8 tensor-core MoE expert GEMM computes, and what this
// reproduces bit-for-bit in scalar Go:
//
//   - Per-token activation quant: each routed token's activation row X[t,:] is
//     dynamically quantized to int8 with ONE per-row f32 scale sx[t] = amax/127.
//   - Per-channel weight quant: each expert output channel W[o,:] is quantized to
//     int8 with ONE per-row f32 scale sw[o] = amax/127.
//   - Whole-K int32 accumulate: the tensor core's IMMA/DP4A path multiplies int8×int8
//     and accumulates the ENTIRE contraction dimension into a single int32 (no per-32
//     sub-block reduction — that is the structural difference from cpuref's Q8_0
//     MatMul, whose scale is per-block(32); here the whole row shares one scale, so
//     the reference is a genuinely independent computation, not a copy of qdot8scalar).
//   - Combined-scale dequant: Y[t,o] = float32(acc) · (sw[o] · sx[t]) — one f32
//     multiply per output element, after the exact integer dot.
//
// The int32 accumulator is EXACT (never int64) on purpose: it mirrors the device's
// 32-bit MMA accumulator, including its overflow envelope. Each int8×int8 term is
// ≤ 127·127 = 16129, so the sum is exact for any contraction length in ≤ 2^31/16129
// ≈ 133k — comfortably above any GLM-5.2 expert in-dim. A device that overflowed here
// would too; the reference does not paper over that with a wider type.
//
// It reuses QuantizeQ8 (the shipped, tested per-block quantizer) for BOTH operands
// with block == in, so the per-block scheme collapses to exactly one scale per row —
// the per-token / per-channel scales an int8 tensor-core GEMM carries. It touches no
// device code and imports nothing beyond the package's own quant primitives, so it
// compiles on the win32 build host unchanged.

// w8a8ExpertCosineMin is the RECORDED Approx cosine floor for the W8A8 int8 expert-GEMM
// path (#3087) — the device-vs-cpuref-f32 (and this reference-vs-f32) output cosine a
// witness must clear. It is set at the SAME 0.999 the int8 (Q8) lane records
// (cudaQ8CosineMin in cuda_accuracy_gates.go), for the same reason that gate's comment
// gives: the int8 lane keeps a full-f32 per-group scale beside the 8-bit codes and the
// integer dot is exact before a single f32 scale multiply, so the only precision loss is
// the in-group 8-bit code rounding — which keeps the direction of the output vector very
// tight against the f32 reference.
//
// One honest caveat specific to THIS path: the tensor-core GEMM shares one scale across
// the WHOLE row (per-token / per-channel), coarser than Q8_0's per-block(32) grouping, so
// on ill-conditioned rows with wide dynamic range the realized cosine can sit below a
// per-block lane's. 0.999 is therefore a CONSERVATIVE floor for well-conditioned expert
// activations; the true realized value on GLM-5.2's quantized weights is measured by the
// DGX sm_80 tensor-core witness, not asserted here.
//
// IMPORTANT (honest handoff, identical to the cuda_accuracy_gates constants): this RECORDS
// the threshold; it does NOT assert the device kernel passes it. This file's test proves
// the REFERENCE clears the gate against the f32 reference on the win32 host; the sm_80
// tensor-core kernel giving ~2× tok/s is measured on a CUDA node. Do not read a device
// pass — or a speedup — from this value alone.
const w8a8ExpertCosineMin = 0.999

// W8A8ExpertGEMM computes the MoE expert GEMM Y[t,o] = Σ_i W[o,i]·X[t,i] the way an sm_80
// int8 tensor-core kernel would: per-token int8 activation scale, per-channel int8 weight
// scale, whole-K int32 accumulate, then a single combined-scale f32 multiply per output
// element. w is the expert weight [out, in] row-major; x is the P routed-token activations
// [P, in] row-major; the result is [P, out] row-major (Y[t*out+o]).
//
// It is a REFERENCE, not the device kernel: it delegates quantization to the shipped
// QuantizeQ8 and does the integer dot in scalar Go, so it is self-contained, portable, and
// exact against its own definition. be supplies the (host) backend the quantized tensors
// resolve on; Default() (cpu-ref) is the intended caller.
func W8A8ExpertGEMM(be Backend, w []float32, out, in int, x []float32, P int) []float32 {
	if out <= 0 || in <= 0 || P <= 0 {
		panic("compute: W8A8ExpertGEMM requires positive out, in, P")
	}
	if len(w) != out*in {
		panic("compute: W8A8ExpertGEMM weight length != out*in")
	}
	if len(x) != P*in {
		panic("compute: W8A8ExpertGEMM activation length != P*in")
	}

	// block == in => exactly one block per row, so QuantizeQ8's per-block(→per-row) scheme
	// yields the per-channel weight scale and per-token activation scale a tensor-core GEMM
	// carries. Reusing the shipped quantizer keeps the rounding byte-identical to the Q8 lane.
	wq := QuantizeQ8(be, []int{out, in}, w, in)
	xq := QuantizeQ8(be, []int{P, in}, x, in)
	wc, ws := hostQ8(wq) // codes [out*in], scales [out]
	xc, sx := hostQ8(xq) // codes [P*in],  scales [P]

	y := make([]float32, P*out)
	for t := 0; t < P; t++ {
		xr := xc[t*in : t*in+in]
		st := sx[t]
		for o := 0; o < out; o++ {
			wr := wc[o*in : o*in+in]
			var acc int32 // the device's int32 MMA accumulator, exact for in ≤ ~133k
			for i := 0; i < in; i++ {
				acc += int32(wr[i]) * int32(xr[i])
			}
			// Dequant with the combined per-token·per-channel scale.
			y[t*out+o] = float32(acc) * (ws[o] * st)
		}
	}
	return y
}

// hostQ8 pulls the int8 codes and per-row f32 scales out of a Q8 tensor built by
// QuantizeQ8, through the public HostBuffer door only (so it is backend-agnostic and
// fails closed on a non-host tensor rather than reaching into private storage).
func hostQ8(t Tensor) (codes []int8, scale []float32) {
	hb, ok := t.Buf().(HostBuffer)
	if !ok {
		panic("compute: W8A8ExpertGEMM requires a host-resident int8 quant (use the cpu reference backend)")
	}
	return hb.I8(), t.Quant.Scale
}
