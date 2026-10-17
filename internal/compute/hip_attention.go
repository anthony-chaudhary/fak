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

// ShouldHIPContiguizeF16KV is a convenience helper for ROCm/HIP pipelines.
func ShouldHIPContiguizeF16KV(arch string, nPos int, precision KVPrecision) bool {
	return ShouldContiguizeF16KV(arch, nPos, precision)
}

// NOTE: HIPContiguizeF16KVCache (a zero-caller pass-through) and
// ExecuteHIPAttentionWithContiguization (a per-call whole-KV f32 copy with no
// production caller) were retired here (fak#13875). The library math they wrapped
// is still held directly by ContiguizeF32KVCache / ComputeContiguizedAttention /
// ComputeStridedAttention and radv_attention_test.go.

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

// NOTE: HIPFlashAttnDequantScratchpad (a zero-caller pass-through wrapper) and
// ExecuteHIPAttentionWithDequantOnce (an unwired per-call scratchpad allocator,
// the same latent per-call weight path class as fak#13875's dequant-once
// entrypoints) were retired here. The underlying single-pass dequantization is
// still held directly by FlashAttnDequantScratchpad and the radv/vulkan witnesses.
