// kernel_matrix.go declares the mixed-precision KERNEL MATRIX: the enumerable set of
// (weight, activation) -> accumulate matmul combinations the kernels actually execute,
// grouped per backend. It is DECLARED DATA, not enforcement:
//
//	(a) The dispatch is UNCHANGED. Nothing here is consulted by MatMul/BatchedMatMul and no
//	    existing path branches on it, so every shipped backend stays byte-identical (P3). The
//	    cpuref arms below are transcribed from cpuref.go's MatMul/BatchedMatMul switch; the
//	    device rows are transcribed from the kernels their backends already call.
//	(b) It exists so the per-pass schedule (child 03) can ASK which pairs a backend executes
//	    and REFUSE a level it cannot run, instead of silently widening an approximation.
//	(c) The undeclared-pair contract: Supports returns (Pair{}, false) for any combination a
//	    matrix does not name. There is no guess, no widening, and no implicit f32 fallback —
//	    an unknown pair fails closed.
//
// ACCUMULATE HONESTY (why the Q8_0 / int8 rows declare Accumulate: F32, not an int32):
// the Q8_0 kernels (cpuref's quantizeVecQ8 + qdot8scalar/qgemm8cell, W8A8ExpertGEMM, and the
// device q8_matmul/gemv tiles) DO reduce in exact int32 — but that int32 is an INTERNAL
// intermediate: every MatMul/BatchedMatMul in this package returns an F32 tensor, and the
// kernel's last step is a single f32 scale multiply. The declared output currency is
// therefore F32. We do not invent an int32 Dtype for a value no caller can ever observe.
package compute

// ---- Kernel-matrix types --------------------------------------------------------

// Pair is one executable mixed-precision matmul combination.
type Pair struct {
	Weight     Dtype
	Activation Dtype
	Accumulate Dtype
	Class      CorrectnessClass
	CosineMin  float64 // the recorded Approx gate, 0 for Reference
}

// KernelMatrix is the declared set of pairs a backend supports.
type KernelMatrix struct{ Pairs []Pair }

// Supports returns the FIRST Pair whose Weight and Activation match (w, a) in declaration
// order, with ok=true. If no pair matches it returns (Pair{}, false) and never guesses — the
// quarantined fallback contract for this file.
func (m KernelMatrix) Supports(w, a Dtype) (Pair, bool) {
	for _, p := range m.Pairs {
		if p.Weight == w && p.Activation == a {
			return p, true
		}
	}
	return Pair{}, false
}

// ---- Default-build mirror constants ----------------------------------------------

// cuda_accuracy_gates.go is //go:build cuda && cgo, so its cosine constants are INVISIBLE on
// the default (non-cuda, CGO_ENABLED=0) build. The digits below therefore MIRROR those
// constants for the default build, exactly as approx_distribution_test.go:35 mirrors
// cudaFP16CosineMin into approxDemoCosineFloor. Each mirror names the tagged source it
// mirrors; if the tagged value moves, move its mirror with it. They RECORD gates only —
// neither they nor the tagged originals assert that any device kernel passes.
const (
	kmCUDAQ8CosineMin   = 0.999 // mirrors cudaQ8CosineMin  (cuda_accuracy_gates.go:53)
	kmCUDAQ4KCosineMin  = 0.995 // mirrors cudaQ4KCosineMin (cuda_accuracy_gates.go:54)
	kmCUDAQ2KCosineMin  = 0.995 // mirrors cudaQ2KCosineMin (cuda_accuracy_gates.go:58)
	kmCUDAQ2CosineMin   = 0.999 // mirrors cudaQ2CosineMin  (cuda_accuracy_gates.go:67)
	kmCUDAFP16CosineMin = 0.997 // mirrors cudaFP16CosineMin(cuda_accuracy_gates.go:25)

	// kmCUDAFP8CosineMin is the FP8-E4M3 block-32 GEMV lane's recorded floor.
	// SOURCE: none — no tagged cudaFP8CosineMin constant exists in the tree. Conservative
	// placeholder pending an on-device measurement. It MIRRORS the Q8 lane's tight 0.999
	// floor: like Q8_0, FP8-E4M3 carries a per-block(32) f32 scale beside 8-bit codes, so
	// its dynamic range is preserved and only the block code rounds. The FP8 lane is DECODE /
	// GEMV-ONLY (spec table row GemvFP8Block32E8M0) — it does not back the batched prefill
	// GEMM. This is a conservative recorded floor, not a measured pass.
	kmCUDAFP8CosineMin = 0.999

	// kmROCmFP4CosineMin is the recorded floor for the ROCm FP4-E2M1 / F16 tensor-core lane
	// (spec table row v_wmma_f32_16x16x16_fp4; strix/dequant_wmma.go).
	// SOURCE: none — there is NO recorded ROCm cosine constant in the tree, so this borrows
	// the closest recorded value, the CUDA 4-bit k-quant lane's 0.995 (kmCUDAQ4KCosineMin),
	// which shares the WebMMA structure of dequant-fused 4-bit operands accumulating in F32.
	// It is a conservative placeholder PENDING a real on-device measurement, not a fabricated
	// measurement.
	kmROCmFP4CosineMin = 0.995

	// kmVulkanQ8CosineMin is the Vulkan Q8_0 lane's floor, sharing the Q8 int8 structure
	// (per-block(32) f32 scale, integer block dot, one f32 scale multiply) and therefore the
	// SAME 0.999 the CUDA Q8 lane records (kmCUDAQ8CosineMin).
	kmVulkanQ8CosineMin = 0.999

	// kmVulkanQ4KCosineMin is the Vulkan Q4_K lane's floor, mirroring the CUDA Q4_K lane's
	// 0.995 (kmCUDAQ4KCosineMin) for the same 4-bit k-quant structure reason.
	kmVulkanQ4KCosineMin = 0.995

	// kmVulkanQ2KCosineMin / kmVulkanQ6KCosineMin: SOURCE: none — Vulkan records no
	// per-dtype k-quant cosine constant. The Q2_K row borrows the nearest recorded 2-bit
	// k-quant floor (the CUDA Q2_K lane's 0.995) and the Q6_K row borrows the nearest
	// recorded k-quant floor (the CUDA Q4_K lane's 0.995), pending on-device Vulkan
	// measurements. Both are conservative recorded floors, not measured passes.
	kmVulkanQ2KCosineMin = 0.995
	kmVulkanQ6KCosineMin = 0.995
)

// ---- Per-backend matrices -------------------------------------------------------

// CPURefKernelMatrix returns the pure-Go cpu-ref matrix, transcribed from the
// MatMul (cpuref.go:160) / BatchedMatMul (cpuref.go:230) switch arms. Every row is
// Class: Reference with CosineMin 0 — the exact-rung rungs (R2/R14) apply, not a cosine.
//
// Q8_0 is the one dynamic-activation arm: the f32 activation is re-quantized in-kernel to
// int8 per block (quantizeVecQ8), so its declared Activation is Q8_0 while the declared
// Accumulate is F32 (see the file header on the int32 intermediate).
func CPURefKernelMatrix() KernelMatrix {
	return KernelMatrix{Pairs: []Pair{
		{Weight: F32, Activation: F32, Accumulate: F32, Class: Reference, CosineMin: 0},     // cpuref.go:169/:235
		{Weight: Q8_0, Activation: Q8_0, Accumulate: F32, Class: Reference, CosineMin: 0},   // cpuref.go:174/:243
		{Weight: Q4_K, Activation: F32, Accumulate: F32, Class: Reference, CosineMin: 0},    // cpuref.go:182/:259
		{Weight: Q5_K, Activation: F32, Accumulate: F32, Class: Reference, CosineMin: 0},    // cpuref.go:188/:270
		{Weight: Q6_K, Activation: F32, Accumulate: F32, Class: Reference, CosineMin: 0},    // cpuref.go:188/:270
		{Weight: Q2_K, Activation: F32, Accumulate: F32, Class: Reference, CosineMin: 0},    // cpuref.go:199/:284
		{Weight: IQ2_XXS, Activation: F32, Accumulate: F32, Class: Reference, CosineMin: 0}, // cpuref.go:206/:294
		{Weight: Q2_0, Activation: F32, Accumulate: F32, Class: Reference, CosineMin: 0},    // cpuref.go:213/:304
	}}
}

// CUDAKernelMatrix returns the declared cuda device matrix. Every row is Class: Approx. The
// Q8/Q4_K/Q2_K/Q2_0/F16 rows mirror the tagged cuda* gate constants (they live under
// //go:build cuda && cgo, so the kmCUDA* mirrors are what the default build sees); the FP8
// row records a conservative floor with no tagged original (see kmCUDAFP8CosineMin).
func CUDAKernelMatrix() KernelMatrix {
	return KernelMatrix{Pairs: []Pair{
		{Weight: Q8_0, Activation: Q8_0, Accumulate: F32, Class: Approx, CosineMin: kmCUDAQ8CosineMin},
		{Weight: Q4_K, Activation: F32, Accumulate: F32, Class: Approx, CosineMin: kmCUDAQ4KCosineMin},
		{Weight: Q2_K, Activation: F32, Accumulate: F32, Class: Approx, CosineMin: kmCUDAQ2KCosineMin},
		{Weight: Q2_0, Activation: F32, Accumulate: F32, Class: Approx, CosineMin: kmCUDAQ2CosineMin},
		{Weight: F16, Activation: F16, Accumulate: F32, Class: Approx, CosineMin: kmCUDAFP16CosineMin},
		// FP8-E4M3 block-32 GEMV lane (spec row GemvFP8Block32E8M0): DECODE / GEMV-ONLY,
		// no batched-prefill GEMM backing it. Floor mirrors the Q8 tight lane; see kmCUDAFP8CosineMin.
		{Weight: FP8, Activation: F32, Accumulate: F32, Class: Approx, CosineMin: kmCUDAFP8CosineMin},
	}}
}

// W8A8KernelMatrix returns the int8 tensor-core expert GEMM lane: Q8_0 weight, Q8_0 per-row
// int8 activation, F32 output (W8A8ExpertGEMM, w8a8_expert_gemm.go:71). CosineMin uses the
// REAL default-visible constant w8a8ExpertCosineMin (w8a8_expert_gemm.go:59) — no mirror.
func W8A8KernelMatrix() KernelMatrix {
	return KernelMatrix{Pairs: []Pair{
		{Weight: Q8_0, Activation: Q8_0, Accumulate: F32, Class: Approx, CosineMin: w8a8ExpertCosineMin},
	}}
}

// ROCmKernelMatrix returns the declared ROCm matrix. It carries the FP4-E2M1 / F16
// tensor-core lane (v_wmma_f32_16x16x16_fp4; strix/dequant_wmma.go) and the Q8_0 / int8
// lane. No ROCm cosine constant is recorded in the tree, so the FP4 floor reuses the closest
// recorded value (kmROCmFP4CosineMin) and the Q8 floor reuses the Q8 lane's 0.999. Both are
// recorded targets pending on-device measurement — not measured passes.
func ROCmKernelMatrix() KernelMatrix {
	return KernelMatrix{Pairs: []Pair{
		{Weight: FP4, Activation: F16, Accumulate: F32, Class: Approx, CosineMin: kmROCmFP4CosineMin},
		{Weight: Q8_0, Activation: Q8_0, Accumulate: F32, Class: Approx, CosineMin: kmVulkanQ8CosineMin},
	}}
}

// VulkanKernelMatrix returns the declared Vulkan matrix: the q8_matmul.comp lane (Q8_0 /
// dynamically-quantized int8 activation), and the q4k_matmul.comp / q2k_matmul.comp /
// q6k_matmul.comp f32-activation lanes. Floors mirror the CUDA per-dtype floors (see the
// kmVulkan* constants); they are recorded gates, not measured passes.
func VulkanKernelMatrix() KernelMatrix {
	return KernelMatrix{Pairs: []Pair{
		{Weight: Q8_0, Activation: Q8_0, Accumulate: F32, Class: Approx, CosineMin: kmVulkanQ8CosineMin},
		{Weight: Q4_K, Activation: F32, Accumulate: F32, Class: Approx, CosineMin: kmVulkanQ4KCosineMin},
		{Weight: Q2_K, Activation: F32, Accumulate: F32, Class: Approx, CosineMin: kmVulkanQ2KCosineMin},
		{Weight: Q6_K, Activation: F32, Accumulate: F32, Class: Approx, CosineMin: kmVulkanQ6KCosineMin},
	}}
}

// MetalKernelMatrix returns the declared Metal matrix. The metal backend shares the Q8_0
// dynamically-quantized activation lane with the other device backends, so it records the
// same Q8 structure and floor; no Metal-specific cosine constant is recorded in the tree.
func MetalKernelMatrix() KernelMatrix {
	return KernelMatrix{Pairs: []Pair{
		{Weight: Q8_0, Activation: Q8_0, Accumulate: F32, Class: Approx, CosineMin: kmCUDAQ8CosineMin},
	}}
}

// ---- Dispatch -------------------------------------------------------------------

// KernelMatrixFor returns the declared matrix for a backend, dispatching on be.Name().
// An UNKNOWN backend name (or a nil backend) returns an EMPTY KernelMatrix, so Supports
// returns false for every pair — fail closed, no fallback matrix is guessed.
func KernelMatrixFor(be Backend) KernelMatrix {
	if be == nil {
		return KernelMatrix{}
	}
	switch be.Name() {
	case "cpu-ref":
		return CPURefKernelMatrix()
	case "cuda":
		return CUDAKernelMatrix()
	case "vulkan":
		return VulkanKernelMatrix()
	case "rocm":
		return ROCmKernelMatrix()
	case "metal":
		return MetalKernelMatrix()
	default:
		return KernelMatrix{}
	}
}
