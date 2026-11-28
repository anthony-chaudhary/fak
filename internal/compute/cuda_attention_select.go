package compute

import (
	"fmt"
	"strconv"
	"strings"
)

// String returns the canonical snake_case regime tag for CUDAAttentionRegime.
func (r CUDAAttentionRegime) String() string {
	switch r {
	case CUDARegimeDecode:
		return "decode"
	case CUDARegimePrefill:
		return "prefill"
	default:
		return fmt.Sprintf("regime(%d)", uint8(r))
	}
}

// CUDAAttentionKernel identifies the CUDA kernel family selected for an
// attention execution step.
type CUDAAttentionKernel uint8

const (
	// CUDAAttentionUnsupported indicates the target compute capability (< sm_80)
	// or geometry is outside the supported CUDA attention matrix.
	CUDAAttentionUnsupported CUDAAttentionKernel = iota
	// CUDAAttentionScalarGeneric selects the generic tiled attention launcher
	// (attn_generic_launch / k_prompt_causal_attention_panel).
	CUDAAttentionScalarGeneric
	// CUDAAttentionFlashDecode selects the fused online-softmax decode flash
	// attention kernel (fcuda_flash_attention_f32).
	CUDAAttentionFlashDecode
	// CUDAAttentionFlashPrefillPanel selects the GPU-resident prefill panel
	// attention kernel (fcuda_causal_attention_panel_f32).
	CUDAAttentionFlashPrefillPanel
	// CUDAAttentionMMA256Prefill selects the tensor-core / MMA hd=256 prefill
	// panel kernel (fak_qwen38_fattn_mma_hd256_launch).
	CUDAAttentionMMA256Prefill
	// CUDAAttentionPagedFP8Prefill selects the Blackwell / SM120+ FP8 paged
	// prefill attention path aligned with SelectCUDAPagedPrefillPath.
	CUDAAttentionPagedFP8Prefill
	// CUDAAttentionTypedKV selects the narrowed-KV device attention kernel
	// (fcuda_kv_attention_typed) for F16/BF16/FP8/asymmetric KV storage.
	CUDAAttentionTypedKV
)

// String returns the stable snake_case identifier for telemetry and ledgers.
func (b CUDAAttentionKernel) String() string {
	switch b {
	case CUDAAttentionUnsupported:
		return "unsupported"
	case CUDAAttentionScalarGeneric:
		return "scalar_generic"
	case CUDAAttentionFlashDecode:
		return "flash_decode"
	case CUDAAttentionFlashPrefillPanel:
		return "flash_prefill_panel"
	case CUDAAttentionMMA256Prefill:
		return "mma_hd256_prefill"
	case CUDAAttentionPagedFP8Prefill:
		return "paged_fp8_prefill"
	case CUDAAttentionTypedKV:
		return "typed_kv"
	default:
		return fmt.Sprintf("cuda_attn_kernel(%d)", uint8(b))
	}
}

// GPUResident reports whether the chosen attention kernel executes on the GPU
// without staging KV tensors back to the host.
func (b CUDAAttentionKernel) GPUResident() bool {
	return b != CUDAAttentionUnsupported
}

// UsesTensorCoreMMA reports whether the kernel dispatches MMA tensor-core
// attention tiles rather than SIMT-only or generic fallback tiles.
func (b CUDAAttentionKernel) UsesTensorCoreMMA() bool {
	return b == CUDAAttentionMMA256Prefill || b == CUDAAttentionPagedFP8Prefill
}

// CUDAAttentionOptions carries optional geometry and KV attributes used to
// refine CUDA attention backend selection.
type CUDAAttentionOptions struct {
	HeadDim     int
	Tokens      int
	KVPrecision KVPrecision
	KVDtype     Dtype // optional physical KV dtype (e.g. FP8, NVFP4, F16, BF16, F32)
	Paged       bool
}

// CUDAAttentionSelection describes the resolved CUDA attention backend plus
// architectural coverage metadata for telemetry and cost modeling.
type CUDAAttentionSelection struct {
	Backend           CUDAAttentionKernel
	Regime            CUDAAttentionRegime
	Arch              string
	Coverage          CUDACoverage
	Blackwell         bool
	GPUResident       bool
	UsesTensorCoreMMA bool
	SupportsFP8KV     bool
}

// IsBlackwellCUDAArch reports whether (major, minor) belongs to the Blackwell
// architecture family targeted for single-node 1-8 GPU cards (sm_100, sm_103, sm_120).
func IsBlackwellCUDAArch(major, minor int) bool {
	if major == 10 {
		return minor == 0 || minor == 3 || minor >= 0
	}
	return major == 12 && minor >= 0
}

// SelectCUDAAttentionKernel chooses the optimal fak-native CUDA attention
// kernel family for compute capability (major, minor), execution regime
// (CUDARegimeDecode or CUDARegimePrefill), and optional geometry/KV options.
// It explicitly supports Blackwell architectures (sm_100, sm_103, sm_120) as
// well as Ampere/Ada/Hopper (sm_80, sm_89, sm_90), failing closed to
// CUDAAttentionUnsupported below sm_80 or on unrecognized regimes.
func SelectCUDAAttentionKernel(major, minor int, regime CUDAAttentionRegime, opts ...CUDAAttentionOptions) CUDAAttentionKernel {
	return ResolveCUDAAttentionSelection(major, minor, regime, opts...).Backend
}

// ResolveCUDAAttentionSelection returns the full selection record for a given
// CUDA SM capability and execution regime.
func ResolveCUDAAttentionSelection(major, minor int, regime CUDAAttentionRegime, opts ...CUDAAttentionOptions) CUDAAttentionSelection {
	var opt CUDAAttentionOptions
	if len(opts) > 0 {
		opt = opts[0]
	}
	cov := ClassifyCUDAArch(major, minor)
	blackwell := IsBlackwellCUDAArch(major, minor)
	arch := fmt.Sprintf("sm_%d%d", major, minor)
	if major <= 0 || minor < 0 {
		arch = "sm_unknown"
	}

	sel := CUDAAttentionSelection{
		Backend:       CUDAAttentionUnsupported,
		Regime:        regime,
		Arch:          arch,
		Coverage:      cov,
		Blackwell:     blackwell,
		SupportsFP8KV: major > 8 || (major == 8 && minor >= 9),
	}

	if cov == CUDAUncovered {
		return sel
	}

	switch regime {
	case CUDARegimeDecode:
		if opt.KVDtype == FP8 && opt.Paged && sel.SupportsFP8KV {
			sel.Backend = CUDAAttentionPagedFP8Prefill
		} else if (opt.KVPrecision != 0 && opt.KVPrecision != KVPrecisionF32) || (opt.KVDtype != 0 && opt.KVDtype != F32) {
			sel.Backend = CUDAAttentionTypedKV
		} else {
			sel.Backend = CUDAAttentionFlashDecode
		}
	case CUDARegimePrefill:
		// Blackwell (sm_100, sm_103, sm_120) and Hopper/Ada FP8 KV or paged prefill.
		if opt.Paged && SelectCUDAPagedPrefillPath(major, minor) == CUDAPagedPrefillFP8 {
			sel.Backend = CUDAAttentionPagedFP8Prefill
		} else if opt.KVDtype == FP8 && sel.SupportsFP8KV && opt.Paged {
			sel.Backend = CUDAAttentionPagedFP8Prefill
		} else if (opt.KVPrecision != 0 && opt.KVPrecision != KVPrecisionF32) || (opt.KVDtype != 0 && opt.KVDtype != F32) {
			sel.Backend = CUDAAttentionTypedKV
		} else if opt.HeadDim == 256 && (opt.Tokens == 0 || opt.Tokens >= 1) {
			// Qwen3.8 hd=256 MMA prefill panel kernel covers SM80+ through Blackwell sm_100/sm_103/sm_120.
			sel.Backend = CUDAAttentionMMA256Prefill
		} else if opt.HeadDim > 1024 {
			sel.Backend = CUDAAttentionScalarGeneric
		} else if blackwell || cov == CUDANative || cov == CUDAJITFromPTX {
			sel.Backend = CUDAAttentionFlashPrefillPanel
		} else {
			sel.Backend = CUDAAttentionScalarGeneric
		}
	default:
		return sel
	}

	sel.GPUResident = sel.Backend.GPUResident()
	sel.UsesTensorCoreMMA = sel.Backend.UsesTensorCoreMMA()
	return sel
}

// parseCUDATierMajorMinor extracts compute capability (major, minor) from tier
// strings such as "sm120", "sm_120", "sm103", "sm_100", or "CUDA sm90".
func parseCUDATierMajorMinor(tier string) (major, minor int, ok bool) {
	lower := strings.ToLower(strings.TrimSpace(tier))
	idx := strings.Index(lower, "sm")
	if idx < 0 {
		return 0, 0, false
	}
	rest := lower[idx+2:]
	rest = strings.TrimPrefix(rest, "_")
	digitsEnd := 0
	for digitsEnd < len(rest) && rest[digitsEnd] >= '0' && rest[digitsEnd] <= '9' {
		digitsEnd++
	}
	if digitsEnd < 2 {
		return 0, 0, false
	}
	code := rest[:digitsEnd]
	// Last digit is minor (e.g. "80" -> 8.0, "100" -> 10.0, "103" -> 10.3, "120" -> 12.0).
	majVal, err1 := strconv.Atoi(code[:len(code)-1])
	minVal, err2 := strconv.Atoi(code[len(code)-1:])
	if err1 != nil || err2 != nil || majVal <= 0 || minVal < 0 {
		return 0, 0, false
	}
	return majVal, minVal, true
}
