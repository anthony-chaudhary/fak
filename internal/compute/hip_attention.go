package compute

import (
	"fmt"
)

// hip_attention.go — HIP / ROCm attention execution with pre-attention f16 KV contiguization (#11746).
//
// Provides HIP/ROCm kernel adapters and dispatch wrappers that apply F16KVContiguizationPass
// on AMD APUs (gfx1151 / Strix Halo) to prevent 16-channel LPDDR5X channel camping.

// HIPF16KVContiguizationPass wraps F16KVContiguizationPass for ROCm / HIP compute graphs.
type HIPF16KVContiguizationPass struct {
	base *F16KVContiguizationPass
}

// NewHIPF16KVContiguizationPass creates a HIP-specific contiguization pass wrapper.
func NewHIPF16KVContiguizationPass(arch string, nPos, nKV, headDim int, precision KVPrecision) *HIPF16KVContiguizationPass {
	return &HIPF16KVContiguizationPass{
		base: NewF16KVContiguizationPass(arch, nPos, nKV, headDim, precision),
	}
}

// ShouldExecute delegates to ShouldContiguizeF16KV.
func (p *HIPF16KVContiguizationPass) ShouldExecute() bool {
	return p.base.ShouldExecute()
}

// ScratchBytes reports memory required in bytes.
func (p *HIPF16KVContiguizationPass) ScratchBytes() int64 {
	return p.base.ScratchBytes()
}

// Execute performs the contiguization pass on strided f16 inputs.
func (p *HIPF16KVContiguizationPass) Execute(kStrided, vStrided []uint16) ([]uint16, []uint16, error) {
	return p.base.Execute(kStrided, vStrided)
}

// HIPContiguizeF16KVCache linearizes strided [nPos, nKV, hd] into head-contiguous [nKV, nPos, hd].
func HIPContiguizeF16KVCache(src, dst []uint16, nPos, nKV, headDim int) ([]uint16, error) {
	return ContiguizeF16KVCache(src, dst, nPos, nKV, headDim)
}

// ShouldHIPContiguizeF16KV is a convenience helper for ROCm/HIP pipelines.
func ShouldHIPContiguizeF16KV(arch string, nPos int, precision KVPrecision) bool {
	return ShouldContiguizeF16KV(arch, nPos, precision)
}

// ExecuteHIPAttentionWithContiguization dispatches attention execution, automatically
// inserting the pre-attention contiguization pass when gating criteria are met.
// Returns the attention output, a boolean indicating whether contiguization was applied, and any error.
func ExecuteHIPAttentionWithContiguization(
	q, kStrided, vStrided []float32,
	arch string,
	nPos, nQ, nKV, headDim int,
	precision KVPrecision,
) (out []float32, contiguized bool, err error) {
	if ShouldContiguizeF16KV(arch, nPos, precision) {
		// Linearize into head-contiguous scratch buffers
		kContig, err := ContiguizeF32KVCache(kStrided, nil, nPos, nKV, headDim)
		if err != nil {
			return nil, false, fmt.Errorf("hip: failed to contiguize K: %w", err)
		}
		vContig, err := ContiguizeF32KVCache(vStrided, nil, nPos, nKV, headDim)
		if err != nil {
			return nil, false, fmt.Errorf("hip: failed to contiguize V: %w", err)
		}

		// Execute contiguized attention kernel
		out, err = ComputeContiguizedAttention(q, kContig, vContig, nQ, nKV, nPos, headDim)
		if err != nil {
			return nil, false, fmt.Errorf("hip: contiguized attention failed: %w", err)
		}
		return out, true, nil
	}

	// Baseline path: direct strided attention
	out, err = ComputeStridedAttention(q, kStrided, vStrided, nQ, nKV, nPos, headDim)
	if err != nil {
		return nil, false, fmt.Errorf("hip: strided attention failed: %w", err)
	}
	return out, false, nil
}

// DiagnoseHIPChannelCamping simulates and diagnoses whether memory accesses will suffer
// from LPDDR5X channel camping on the target architecture.
func DiagnoseHIPChannelCamping(
	arch string,
	nPos, nKV, headDim int,
	precision KVPrecision,
) (isCampingRisk bool, diagnosis string, report ChannelEntropyReport) {
	if !isStrixHaloArch(arch) {
		report = SimulateChannelDistribution(nPos, nKV, headDim, false, DefaultInterleaveBytes)
		return false, fmt.Sprintf("Architecture %q does not use 16-channel LPDDR5X UMA interleaving; no channel camping risk", arch), report
	}

	if precision != KVPrecisionF32 {
		report = SimulateChannelDistribution(nPos, nKV, headDim, false, DefaultInterleaveBytes)
		return false, fmt.Sprintf("Precision %s is quantized; channel camping pass targets unquantized f16 KV caches", precision), report
	}

	if nPos < ContiguizationMinContext {
		report = SimulateChannelDistribution(nPos, nKV, headDim, false, DefaultInterleaveBytes)
		return false, fmt.Sprintf("Context length %d < threshold %d; cache fits within L2/MALL without sustained channel camping", nPos, ContiguizationMinContext), report
	}

	// At >= 32k context on gfx1151 with unquantized f16: channel camping risk detected
	report = SimulateChannelDistribution(nPos, nKV, headDim, false, DefaultInterleaveBytes)
	diagnosis = fmt.Sprintf(
		"LPDDR5X channel camping detected on %s: strided access camps on %d channels with entropy %.4f (< 0.25); pre-attention contiguization pass required",
		arch, report.ActiveChannels, report.Entropy,
	)
	return true, diagnosis, report
}
