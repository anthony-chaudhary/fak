//go:build cuda

package compute

// cudaFP16CosineMin is the cuda backend's RECORDED Approx cosine floor for the fp16 (HGEMM /
// tensor-core) compute path (#484) — the device-vs-cpuref-f32 logit/GEMM cosine a witness must
// clear. It is deliberately LOOSER than the Q8 / int8 lane's 0.999 gate, for a recorded reason,
// not an assumed one:
//
//   - Q8_0 keeps a per-block(32) f32 scale beside the 8-bit codes (QuantSpec.Scale), and the
//     activation is dynamically re-quantized per block with its own f32 scale. The dynamic
//     range of every group is therefore carried in FULL f32; only the in-block code rounds, and
//     the dot is integer-exact before the single f32 scale multiply. That structure keeps the
//     int8 lane tight against the f32 reference (0.999).
//   - fp16 (IEEE binary16) rounds BOTH operands to a 10-bit mantissa (~2^-11 relative) with NO
//     per-block f32 scale to preserve magnitude structure, so the per-element rounding enters
//     the product directly and compounds along the contraction. cublasGemmEx accumulates in F32
//     (CUBLAS_COMPUTE_32F), which bounds the SUM error, but the INPUTS are already fp16-rounded
//     before the multiply — a drift source the scaled-int8 path does not have. So the fp16 gate
//     is set below the int8 gate as a conservative floor.
//
// IMPORTANT (honest handoff): this constant RECORDS the threshold; it does not assert the path
// passes it. The realized cosine is measured on a CUDA node by tools/run_484_acceptance_on_gpu.sh
// (the win32 build host has no CUDA toolkit / GPU). Do not read a pass from this value alone.
const cudaFP16CosineMin = 0.997

// cudaQ8CosineMin / cudaQ4KCosineMin are the cuda backend's RECORDED Approx cosine floors for the
// native quantized device GEMMs (#485) — the device-vs-cpuref-f32 GEMM cosine each per-dtype
// witness must clear. They are PER-DTYPE because the two formats sit at different points on the
// precision/footprint curve, and the floor must reflect the floor's own format, not a shared guess:
//
//   - cudaQ8CosineMin (0.999): Q8_0 keeps a per-block(32) f32 scale beside 8-bit codes (256 levels)
//     and the activation is dynamically re-quantized per block with its OWN f32 scale, so every
//     group's dynamic range is carried in full f32 and only the in-block code rounds; the per-block
//     dot is integer-exact before a single f32 scale multiply. That structure keeps the int8 lane
//     tight against the f32 reference — the SAME 0.999 the Q8 lane has always been held to (it is the
//     gate cudaFP16CosineMin's comment calls "the Q8 lane's 0.999").
//   - cudaQ4KCosineMin (0.995): Q4_K is a 4-bit k-quant — codes carry only 16 levels and the
//     per-32-sub-block (scale,min) is itself quantized to 6 bits under ONE f16 super-block (d,dmin)
//     pair shared across 256 elements. So a Q4_K weight reconstructs less of the original f32
//     magnitude structure than Q8_0's 8-bit+f32-scale grouping does; on a real model quantized FROM
//     full-precision f32 the 4-bit reconstruction error is genuinely larger than 8-bit, so the
//     recorded floor is set LOOSER than the Q8 lane. (The -tags cuda Q4_K gate isolates the device
//     dequant-fused tile's arithmetic against an f32 dequant of the SAME super-block bytes — it
//     witnesses the getScaleMinK4 geometry is reproduced bit-for-bit; the full-model true-f32 → Q4_K
//     cosine is measured on a GPU node, the same honest residual as every device number here.)
//
// IMPORTANT (honest handoff, identical to cudaFP16CosineMin): these constants RECORD the thresholds;
// they do NOT assert the paths pass them. The realized cosines are measured on a CUDA node by
// tools/run_485_acceptance_on_gpu.sh (the win32 build host has no CUDA toolkit / GPU). Do not read a
// pass from these values alone.
const (
	cudaQ8CosineMin  = 0.999
	cudaQ4KCosineMin = 0.995
	// cudaQ2KCosineMin is the cuda backend's RECORDED Approx cosine floor for the Q2_K
	// dequant-fused device GEMV / panel GEMM (#11945). The reference is the CPU dequant
	// of the exact same super-block bytes.
	cudaQ2KCosineMin = 0.995
	// cudaQ2CosineMin is the cuda backend's RECORDED Approx cosine floor for the packed-ternary
	// Q2_0 device GEMV (#4872). UNLIKE the Q8/Q4_K floors, the Q2_0 witness compares the device
	// GEMM against a cpuref f32 GEMV over an f32 dequant of the SAME packed ternary codes+scales
	// (not a true-f32 → ternary reconstruction), so the only residual is the device tile's
	// reduction/scale-fold order vs the host q2RowDot — no quantization error enters the gate.
	// That drift is pure f32 summation reordering, so the floor is set at Q8's 0.999 (tight), the
	// same class of order-only drift; the true-f32 → ternary reconstruction error is a separate,
	// model-level number, not what this device-vs-cpuref gate measures.
	cudaQ2CosineMin = 0.999
)

// cudaAWQCosineMin is the cuda backend's RECORDED Approx cosine floor for the AWQ 4-bit device
// matmul (#926 — the one device op family that previously had NO cpuref-parity witness) — the
// device-vs-cpuref cosine the AWQMatMul / AWQBatchedMatMul (fcuda_awq_gemv / fcuda_awq_gemm)
// witness must clear. AWQ is a 4-bit weight-only format: nibble-packed codes (2/byte) with a
// PER-CHANNEL f32 scale and a symmetric zero-point of 8 (weight = scale·(code−8)); the kernel
// dequant-fuses the nibble into the GEMM tile and accumulates in F32, exactly the structure of
// the Q4_K lane. The floor is therefore set at the SAME 0.995 the 4-bit Q4_K dequant-fused lane
// records: the reference is an f32 dequant of the SAME packed bytes + scales, so this gate
// isolates the device tile's arithmetic (reduction-order drift) against the host AWQ dequant —
// the same class of drift cudaQ4KCosineMin bounds, not the true-f32→4-bit reconstruction error.
// AWQ's per-CHANNEL f32 scale is in fact finer than Q4_K's per-SUB-BLOCK quantized 6-bit scale,
// so 0.995 is a conservative recorded floor for the path.
//
// IMPORTANT (honest handoff, identical to cudaQ4KCosineMin): this RECORDS the threshold; it does
// NOT assert the path passes it. The realized cosine is measured on a CUDA node by
// tools/run_926_acceptance_on_gpu.sh (the win32 build host has no CUDA toolkit / GPU); this floor
// is a reasoned target derived from the analogous 4-bit dequant-fused lane, and the first GPU run
// records the realized value. Do not read a pass from this value alone.
const cudaAWQCosineMin = 0.995

// cudaGPTQCosineMin is the cuda backend's RECORDED Approx cosine floor for the native packed
// GPTQ device GEMV (#3030 — the GPU remainder of #300's CPU-resident GPTQ spine). GPTQ is a
// 4/8-bit weight-only format: codes packed 32/bits per int32 along the input dim, per-GROUP
// (not per-channel) int32-packed zero-points, and per-group f32 scales, with the AutoGPTQ
// zero+1 convention (weight = (code-(zero+1))·scale[g,o]). fcuda_gptq_gemv dequant-fuses the
// unpack into the GEMV tile and accumulates in F32 — the same dequant-fused structure as the
// Q4_K / AWQ 4-bit lanes, so the floor is the SAME 0.995 those 4-bit lanes record. The witness
// reference is an f32 dequant of the SAME packed qweight/qzeros/scales (dequantGPTQWeight), so
// this gate isolates the device tile's arithmetic (reduction-order drift) against the host GPTQ
// dequant — the same class cudaQ4KCosineMin bounds, NOT the true-f32→4-bit reconstruction error.
//
// IMPORTANT (honest handoff, identical to cudaAWQCosineMin): this RECORDS the threshold; it does
// NOT assert the path passes it. The realized cosine is measured on a CUDA node (the win32 build
// host has no CUDA toolkit / GPU); on this host the -tags cuda witness skips, fails-closed, naming
// the missing runner. This floor is a reasoned target derived from the analogous 4-bit dequant-
// fused lanes; the first GPU run records the realized value. Do not read a pass from this value.
const cudaGPTQCosineMin = 0.995

// cudaFlashAttnCosineMin is the cuda backend's RECORDED Approx cosine floor for the fused
// flash/online-softmax attention kernel (#486) — the device-vs-cpuref-f32 logit cosine a witness
// must clear. The flash kernel computes the SAME math as the cpuref reference — softmax(scale·q·k)
// then ΣwV — only reordered into the streaming online-softmax form (running max/sum, the
// accumulator rescaled onto each new max instead of a single batched max+exp over a full scores
// row). So the ONLY difference from the reference is f32 reduction order, the same class of drift
// as the SGEMM lane (cuBLAS reorders the contraction); it carries no extra rounding the way the
// quantized/fp16 lanes do (no narrowed operand). The floor is therefore set at the SAME 0.999 the
// full forward-pass gate has always used — a conservative recorded value, not a measured pass: in
// isolation the attention-only cosine should sit far tighter (near the 0.9999 SGEMM op gate), but
// 0.999 is the floor the multi-layer forward witness (TestCUDAForwardMatchesRef, whose Attention is
// now this flash kernel) is held to end-to-end.
//
// IMPORTANT (honest handoff, identical to the constants above): this RECORDS the threshold; it does
// NOT assert the path passes it. The realized cosine + the fused-vs-naive speedup are measured on a
// CUDA node by tools/run_486_acceptance_on_gpu.sh (the win32 build host has no CUDA toolkit / GPU).
// Do not read a pass — or a speedup — from this value alone.
const cudaFlashAttnCosineMin = 0.999

// cudaDsaSparseAttnCosineMin is the cuda backend's RECORDED Approx cosine floor for the GLM-MoE-DSA
// sparse-attention kernel (k_dsa_sparse_attend) — the device-vs-cpuref-f32 cosine a witness must
// clear. The kernel computes the SAME math as model.glmDsaAttendCached — softmax(scale·q·k) then ΣwV,
// over the host-SELECTED keys — only reordered into the same online-softmax form as the flash kernel.
// Crucially, the key SELECTION (the f64 index-score dots + top-k) is computed HOST-side and handed in
// as gathered rows, so the device attends the SAME keys as the reference: there is no risk of a flipped
// top-k entry, and the ONLY difference from cpuref is f32 reduction order — the same class of drift as
// the flash lane, with no narrowed operand. The floor is therefore set at the SAME 0.999 the flash and
// full-forward gates use (a conservative recorded value; in isolation the cosine sits far tighter).
//
// IMPORTANT (honest handoff, identical to the constants above): this RECORDS the threshold; it does NOT
// assert the path passes it. The realized cosine is measured on a CUDA node by the GLM-DSA on-device
// witness (TestCUDAGLMMoeDsaBackendForward, run via tools/dgx_glm_gpu_witness.sh); the win32 build host
// has no CUDA toolkit / GPU. Do not read a pass from this value alone.
//
//slop:keep RECORDED contract value, checked only by the out-of-tree dgx GPU witness
const cudaDsaSparseAttnCosineMin = 0.999

// cudaDsaIndexSelectionExact records the cuda backend's gate for the GLM-MoE-DSA indexer
// score + top-k SELECTION kernel (k_dsa_index_score + k_dsa_index_topk). Unlike the GEMM /
// attention lanes — which are Approx, held to a COSINE floor because f32 reduction order may drift
// the values slightly — the indexer drives a DISCRETE top-k, so its gate is SET EQUALITY, not a
// cosine: the device-selected key positions must equal the host f64 selection EXACTLY. The kernel
// accumulates the per-key score dot in f64 (IndexHeadDim is tiny; A100 has native f64), so the
// device score matches the host f64 score bit-closely and the same total order (score desc, ties by
// lower position) yields the IDENTICAL set. true documents the contract the witness asserts; a
// device that returned a different set would be a correctness defect, not an Approx drift.
//
// IMPORTANT (honest handoff, like the cosine constants): this RECORDS the contract; it does NOT
// assert the path holds it. The realized selection equality is measured on a CUDA node by the
// GLM-DSA on-device witness (TestCUDAGLMMoeDsaIndexSelectMatches, run via the dgx witness script);
// the win32 build host has no CUDA toolkit / GPU. Do not read a pass from this value alone.
//
//slop:keep RECORDED contract value, checked only by the out-of-tree dgx GPU witness
const cudaDsaIndexSelectionExact = true
