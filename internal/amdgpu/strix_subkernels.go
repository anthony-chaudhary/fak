package amdgpu

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"regexp"
	"strconv"
	"strings"
	"time"
)

const (
	StrixSubkernelParitySchema = "fak.strix.subkernel-parity/v1"
	StrixVulkanEngine          = "fak-native/vulkan"
	embeddingGatherSelector    = "qwen35-embedding-gather"
	embeddingGatherTestName    = "TestQwen35VulkanEmbeddingGatherOneDispatch"
)

// Closed oracle kind vocabulary
const (
	OracleExactArgmax     = "exact_argmax"
	OracleCosineMaxAbs    = "cosine_max_abs"
	OracleMaxAbs          = "max_abs"
	OracleCosineArgmax    = "cosine_argmax"
	OracleStateContinuity = "state_continuity"
	OracleHostContract    = "host_contract"
)

// IsKnownOracleKind returns true if kind is a recognized oracle kind token.
func IsKnownOracleKind(kind string) bool {
	switch kind {
	case OracleExactArgmax,
		OracleCosineMaxAbs,
		OracleMaxAbs,
		OracleCosineArgmax,
		OracleStateContinuity,
		OracleHostContract:
		return true
	default:
		return false
	}
}

var (
	ErrSubkernelParityAbsent      = errors.New("amdgpu: subkernel parity event absent")
	ErrSubkernelParityDuplicate   = errors.New("amdgpu: duplicate subkernel parity events")
	ErrSubkernelParityMalformed   = errors.New("amdgpu: malformed subkernel parity event")
	ErrSubkernelParityNonFinite   = errors.New("amdgpu: non-finite subkernel parity metric")
	ErrSubkernelParityWrongEngine = errors.New("amdgpu: wrong engine or device in subkernel parity event")
	ErrSubkernelParityMismatch    = errors.New("amdgpu: subkernel parity selector/test mismatch")
	ErrSubkernelParityOutOfBounds = errors.New("amdgpu: subkernel parity metric out of bounds")
)

// SubkernelParityBounds specifies the numerical and functional acceptance criteria for a subkernel.
type SubkernelParityBounds struct {
	MinCosine                  *float64 `json:"min_cosine,omitempty"`
	MaxAbsDelta                *float64 `json:"max_abs_delta,omitempty"`
	RequireSourceMutationCheck bool     `json:"require_source_mutation_check,omitempty"`
	MaxSourceDelta             *float64 `json:"max_source_delta,omitempty"`
	RequireArgmaxExact         bool     `json:"require_argmax_exact,omitempty"`
	RequireStateIdentity       bool     `json:"require_state_identity,omitempty"`
	RequireFinite              bool     `json:"require_finite,omitempty"`
}

// SubkernelParityContract maps a subkernel selector to its exact emitting test,
// closed oracle kind, required engine, physical device requirement, and verification bounds.
type SubkernelParityContract struct {
	Selector       string                `json:"selector"`
	TestName       string                `json:"test_name"`
	OracleKind     string                `json:"oracle_kind"`
	Engine         string                `json:"engine"`
	DeviceObserved bool                  `json:"device_observed"`
	Bounds         SubkernelParityBounds `json:"bounds"`
}

// StrixSubkernelObservedMetrics records the numerical and functional metrics measured during test execution.
type StrixSubkernelObservedMetrics struct {
	Cosine         *float64 `json:"cosine,omitempty"`
	MaxAbsDelta    *float64 `json:"max_abs_delta,omitempty"`
	MaxSourceDelta *float64 `json:"max_source_delta,omitempty"`
	ArgmaxExact    *bool    `json:"argmax_exact,omitempty"`
	StateIdentity  *bool    `json:"state_identity,omitempty"`
	FiniteOutput   *bool    `json:"finite_output,omitempty"`
}

func (m *StrixSubkernelObservedMetrics) UnmarshalJSON(data []byte) error {
	var raw struct {
		Cosine                *float64 `json:"cosine"`
		LogitCosineSimilarity *float64 `json:"logit_cosine_similarity"`
		MaxAbs                *float64 `json:"max_abs"`
		MaxAbsDelta           *float64 `json:"max_abs_delta"`
		MaxAbsoluteDelta      *float64 `json:"max_absolute_delta"`
		MaxSourceDelta        *float64 `json:"max_source_delta"`
		SourceMutationDelta   *float64 `json:"source_mutation_delta"`
		ArgmaxExact           *bool    `json:"argmax_exact"`
		StateIdentity         *bool    `json:"state_identity"`
		StateContinuous       *bool    `json:"state_continuous"`
		FiniteOutput          *bool    `json:"finite_output"`
		Finite                *bool    `json:"finite"`
	}
	if err := json.Unmarshal(data, &raw); err != nil {
		return err
	}
	if raw.Cosine != nil {
		m.Cosine = raw.Cosine
	} else if raw.LogitCosineSimilarity != nil {
		m.Cosine = raw.LogitCosineSimilarity
	}
	if raw.MaxAbsDelta != nil {
		m.MaxAbsDelta = raw.MaxAbsDelta
	} else if raw.MaxAbs != nil {
		m.MaxAbsDelta = raw.MaxAbs
	} else if raw.MaxAbsoluteDelta != nil {
		m.MaxAbsDelta = raw.MaxAbsoluteDelta
	}
	if raw.MaxSourceDelta != nil {
		m.MaxSourceDelta = raw.MaxSourceDelta
	} else if raw.SourceMutationDelta != nil {
		m.MaxSourceDelta = raw.SourceMutationDelta
	}
	if raw.ArgmaxExact != nil {
		m.ArgmaxExact = raw.ArgmaxExact
	}
	if raw.StateIdentity != nil {
		m.StateIdentity = raw.StateIdentity
	} else if raw.StateContinuous != nil {
		m.StateIdentity = raw.StateContinuous
	}
	if raw.FiniteOutput != nil {
		m.FiniteOutput = raw.FiniteOutput
	} else if raw.Finite != nil {
		m.FiniteOutput = raw.Finite
	}
	return nil
}

// StrixSubkernelParityEvent models a fak.strix.subkernel-parity/v1 event.
type StrixSubkernelParityEvent struct {
	Schema         string                        `json:"schema"`
	Selector       string                        `json:"selector"`
	TestName       string                        `json:"test_name"`
	OracleKind     string                        `json:"oracle_kind"`
	Engine         string                        `json:"engine"`
	DeviceObserved bool                          `json:"device_observed"`
	CaseCount      int                           `json:"case_count"`
	Passed         bool                          `json:"passed"`
	Observed       StrixSubkernelObservedMetrics `json:"observed"`
}

func (e *StrixSubkernelParityEvent) UnmarshalJSON(data []byte) error {
	type eventAlias StrixSubkernelParityEvent
	var raw struct {
		eventAlias
		Cosine                *float64 `json:"cosine"`
		LogitCosineSimilarity *float64 `json:"logit_cosine_similarity"`
		MaxAbs                *float64 `json:"max_abs"`
		MaxAbsDelta           *float64 `json:"max_abs_delta"`
		MaxAbsoluteDelta      *float64 `json:"max_absolute_delta"`
		MaxSourceDelta        *float64 `json:"max_source_delta"`
		SourceMutationDelta   *float64 `json:"source_mutation_delta"`
		ArgmaxExact           *bool    `json:"argmax_exact"`
		StateIdentity         *bool    `json:"state_identity"`
		StateContinuous       *bool    `json:"state_continuous"`
		FiniteOutput          *bool    `json:"finite_output"`
		Finite                *bool    `json:"finite"`
	}
	if err := json.Unmarshal(data, &raw); err != nil {
		return err
	}
	*e = StrixSubkernelParityEvent(raw.eventAlias)
	if e.Observed.Cosine == nil {
		if raw.Cosine != nil {
			e.Observed.Cosine = raw.Cosine
		} else if raw.LogitCosineSimilarity != nil {
			e.Observed.Cosine = raw.LogitCosineSimilarity
		}
	}
	if e.Observed.MaxAbsDelta == nil {
		if raw.MaxAbsDelta != nil {
			e.Observed.MaxAbsDelta = raw.MaxAbsDelta
		} else if raw.MaxAbs != nil {
			e.Observed.MaxAbsDelta = raw.MaxAbs
		} else if raw.MaxAbsoluteDelta != nil {
			e.Observed.MaxAbsDelta = raw.MaxAbsoluteDelta
		}
	}
	if e.Observed.MaxSourceDelta == nil {
		if raw.MaxSourceDelta != nil {
			e.Observed.MaxSourceDelta = raw.MaxSourceDelta
		} else if raw.SourceMutationDelta != nil {
			e.Observed.MaxSourceDelta = raw.SourceMutationDelta
		}
	}
	if e.Observed.ArgmaxExact == nil && raw.ArgmaxExact != nil {
		e.Observed.ArgmaxExact = raw.ArgmaxExact
	}
	if e.Observed.StateIdentity == nil {
		if raw.StateIdentity != nil {
			e.Observed.StateIdentity = raw.StateIdentity
		} else if raw.StateContinuous != nil {
			e.Observed.StateIdentity = raw.StateContinuous
		}
	}
	if e.Observed.FiniteOutput == nil {
		if raw.FiniteOutput != nil {
			e.Observed.FiniteOutput = raw.FiniteOutput
		} else if raw.Finite != nil {
			e.Observed.FiniteOutput = raw.Finite
		}
	}
	return nil
}

func floatPtr(v float64) *float64 {
	return &v
}

func boolPtr(v bool) *bool {
	return &v
}

// DefaultSubkernelParityContracts exhaustively binds every default selector in DefaultSubkernelSpecs
// to its exact emitting test, closed oracle kind, required engine, physical device requirement, and verification bounds.
var DefaultSubkernelParityContracts = map[string]SubkernelParityContract{
	"argmax": {
		Selector:       "argmax",
		TestName:       "TestVulkanArgmaxExact",
		OracleKind:     OracleExactArgmax,
		Engine:         StrixVulkanEngine,
		DeviceObserved: true,
		Bounds: SubkernelParityBounds{
			RequireArgmaxExact: true,
		},
	},
	"matmul_f32": {
		Selector:       "matmul_f32",
		TestName:       "TestVulkanMatMulApprox",
		OracleKind:     OracleCosineMaxAbs,
		Engine:         StrixVulkanEngine,
		DeviceObserved: true,
		Bounds: SubkernelParityBounds{
			MinCosine:   floatPtr(0.9999),
			MaxAbsDelta: floatPtr(1e-2),
		},
	},
	"matmul2_f32": {
		Selector:       "matmul2_f32",
		TestName:       "TestVulkanMatMul2Approx",
		OracleKind:     OracleCosineMaxAbs,
		Engine:         StrixVulkanEngine,
		DeviceObserved: true,
		Bounds: SubkernelParityBounds{
			MinCosine:   floatPtr(0.9999),
			MaxAbsDelta: floatPtr(1e-2),
		},
	},
	"matmul3_f32": {
		Selector:       "matmul3_f32",
		TestName:       "TestVulkanMatMul3Approx",
		OracleKind:     OracleCosineMaxAbs,
		Engine:         StrixVulkanEngine,
		DeviceObserved: true,
		Bounds: SubkernelParityBounds{
			MinCosine:   floatPtr(0.9999),
			MaxAbsDelta: floatPtr(1e-2),
		},
	},
	"q8_matmul": {
		Selector:       "q8_matmul",
		TestName:       "TestVulkanQ8MatMulApprox",
		OracleKind:     OracleCosineMaxAbs,
		Engine:         StrixVulkanEngine,
		DeviceObserved: true,
		Bounds: SubkernelParityBounds{
			MinCosine:   floatPtr(0.9999),
			MaxAbsDelta: floatPtr(1e-3),
		},
	},
	"q8_matmul_wide": {
		Selector:       "q8_matmul_wide",
		TestName:       "TestVulkanQ8MatMulWideInput",
		OracleKind:     OracleCosineMaxAbs,
		Engine:         StrixVulkanEngine,
		DeviceObserved: true,
		Bounds: SubkernelParityBounds{
			MinCosine:   floatPtr(0.9999),
			MaxAbsDelta: floatPtr(1e-3),
		},
	},
	"q8_matmul_vocab": {
		Selector:       "q8_matmul_vocab",
		TestName:       "TestVulkanQ8MatMulVocabHead",
		OracleKind:     OracleCosineMaxAbs,
		Engine:         StrixVulkanEngine,
		DeviceObserved: true,
		Bounds: SubkernelParityBounds{
			MinCosine:   floatPtr(0.9999),
			MaxAbsDelta: floatPtr(1e-3),
		},
	},
	"q4k_matmul": {
		Selector:       "q4k_matmul",
		TestName:       "TestVulkanQ4KMatMulMatchesCPUReference",
		OracleKind:     OracleCosineArgmax,
		Engine:         StrixVulkanEngine,
		DeviceObserved: true,
		Bounds: SubkernelParityBounds{
			MinCosine:          floatPtr(0.995),
			RequireArgmaxExact: true,
		},
	},
	"q2k_matmul": {
		Selector:       "q2k_matmul",
		TestName:       "TestVulkanQ2KMatMulMatchesCPUReference",
		OracleKind:     OracleCosineArgmax,
		Engine:         StrixVulkanEngine,
		DeviceObserved: true,
		Bounds: SubkernelParityBounds{
			MinCosine:          floatPtr(0.995),
			RequireArgmaxExact: true,
		},
	},
	"rmsnorm": {
		Selector:       "rmsnorm",
		TestName:       "TestVulkanRMSNormApprox",
		OracleKind:     OracleMaxAbs,
		Engine:         StrixVulkanEngine,
		DeviceObserved: true,
		Bounds: SubkernelParityBounds{
			MaxAbsDelta: floatPtr(1e-3),
		},
	},
	"rmsnorm_matmul": {
		Selector:       "rmsnorm_matmul",
		TestName:       "TestVulkanRMSNormMatMulApprox",
		OracleKind:     OracleCosineMaxAbs,
		Engine:         StrixVulkanEngine,
		DeviceObserved: true,
		Bounds: SubkernelParityBounds{
			MinCosine:                  floatPtr(0.9999),
			MaxAbsDelta:                floatPtr(1e-2),
			RequireSourceMutationCheck: true,
			MaxSourceDelta:             floatPtr(0.0),
		},
	},
	"rmsnorm_matmul2": {
		Selector:       "rmsnorm_matmul2",
		TestName:       "TestVulkanRMSNormMatMul2Approx",
		OracleKind:     OracleCosineMaxAbs,
		Engine:         StrixVulkanEngine,
		DeviceObserved: true,
		Bounds: SubkernelParityBounds{
			MinCosine:                  floatPtr(0.9999),
			MaxAbsDelta:                floatPtr(1e-2),
			RequireSourceMutationCheck: true,
			MaxSourceDelta:             floatPtr(0.0),
		},
	},
	"rmsnorm_matmul3": {
		Selector:       "rmsnorm_matmul3",
		TestName:       "TestVulkanRMSNormMatMul3Approx",
		OracleKind:     OracleCosineMaxAbs,
		Engine:         StrixVulkanEngine,
		DeviceObserved: true,
		Bounds: SubkernelParityBounds{
			MinCosine:                  floatPtr(0.9999),
			MaxAbsDelta:                floatPtr(1e-2),
			RequireSourceMutationCheck: true,
			MaxSourceDelta:             floatPtr(0.0),
		},
	},
	"swiglu": {
		Selector:       "swiglu",
		TestName:       "TestVulkanSwiGLUApprox",
		OracleKind:     OracleMaxAbs,
		Engine:         StrixVulkanEngine,
		DeviceObserved: true,
		Bounds: SubkernelParityBounds{
			MaxAbsDelta: floatPtr(1e-3),
		},
	},
	"swiglu_matmul_add": {
		Selector:       "swiglu_matmul_add",
		TestName:       "TestVulkanSwiGLUMatMulAddInPlaceApprox",
		OracleKind:     OracleCosineMaxAbs,
		Engine:         StrixVulkanEngine,
		DeviceObserved: true,
		Bounds: SubkernelParityBounds{
			MinCosine:   floatPtr(0.9999),
			MaxAbsDelta: floatPtr(1e-2),
		},
	},
	"rope": {
		Selector:       "rope",
		TestName:       "TestVulkanRoPEApprox",
		OracleKind:     OracleMaxAbs,
		Engine:         StrixVulkanEngine,
		DeviceObserved: true,
		Bounds: SubkernelParityBounds{
			MaxAbsDelta:                floatPtr(1e-3),
			RequireSourceMutationCheck: true,
			MaxSourceDelta:             floatPtr(0.0),
		},
	},
	"attention": {
		Selector:       "attention",
		TestName:       "TestVulkanAttentionApprox",
		OracleKind:     OracleCosineMaxAbs,
		Engine:         StrixVulkanEngine,
		DeviceObserved: true,
		Bounds: SubkernelParityBounds{
			MinCosine:   floatPtr(0.999),
			MaxAbsDelta: floatPtr(1e-2),
		},
	},
	"qwen35_gdn_decode": {
		Selector:       "qwen35_gdn_decode",
		TestName:       "TestVulkanQwen35GDNDecodeMatchesCPUOracleInPlace",
		OracleKind:     OracleStateContinuity,
		Engine:         StrixVulkanEngine,
		DeviceObserved: true,
		Bounds: SubkernelParityBounds{
			MaxAbsDelta:          floatPtr(3e-4),
			RequireStateIdentity: true,
			RequireFinite:        true,
		},
	},
	"qwen35_gdn_preprojected": {
		Selector:       "qwen35_gdn_preprojected",
		TestName:       "TestVulkanQwen35GDNPreprojectedParityAndStateContinuity",
		OracleKind:     OracleStateContinuity,
		Engine:         StrixVulkanEngine,
		DeviceObserved: true,
		Bounds: SubkernelParityBounds{
			MaxAbsDelta:          floatPtr(2e-4),
			RequireStateIdentity: true,
			RequireFinite:        true,
		},
	},
	"qwen35_sequence_prefill": {
		Selector:       "qwen35_sequence_prefill",
		TestName:       "TestVulkanQwen35SequenceQuantizedPanelsMatchCPU",
		OracleKind:     OracleMaxAbs,
		Engine:         StrixVulkanEngine,
		DeviceObserved: true,
		Bounds: SubkernelParityBounds{
			MaxAbsDelta:   floatPtr(2e-3),
			RequireFinite: true,
		},
	},
	embeddingGatherSelector: {
		Selector:       embeddingGatherSelector,
		TestName:       embeddingGatherTestName,
		OracleKind:     OracleMaxAbs,
		Engine:         StrixVulkanEngine,
		DeviceObserved: true,
		Bounds: SubkernelParityBounds{
			MaxAbsDelta:   floatPtr(0),
			RequireFinite: true,
		},
	},
	"f16_kv_contiguize": {
		Selector:       "f16_kv_contiguize",
		TestName:       "TestRADVContiguizeShader",
		OracleKind:     OracleHostContract,
		Engine:         "fak-native/host",
		DeviceObserved: false,
		Bounds:         SubkernelParityBounds{},
	},
}

// LookupSubkernelParityContract retrieves the parity contract registered for selector.
func LookupSubkernelParityContract(selector string) (SubkernelParityContract, bool) {
	contract, ok := DefaultSubkernelParityContracts[strings.ToLower(strings.TrimSpace(selector))]
	return contract, ok
}

// SubkernelSpec defines an executable sub-kernel test.
type SubkernelSpec struct {
	Name        string
	Description string
	TestPattern string
	Category    string
	ProfileEnv  string
}

// DefaultSubkernelSpecs lists the canonical Vulkan compute sub-kernels.
var DefaultSubkernelSpecs = []SubkernelSpec{
	{
		Name:        "argmax",
		Description: "Bit-exact argmax reduction with first-max tie break",
		TestPattern: "^TestVulkanArgmaxExact$",
		Category:    "reduction",
	},
	{
		Name:        "matmul_f32",
		Description: "Single-precision matrix multiplication (16x16 tile)",
		TestPattern: "^TestVulkanMatMulApprox$",
		Category:    "gemv",
	},
	{
		Name:        "matmul2_f32",
		Description: "Dual matrix multiplication (FFN gate+up projection)",
		TestPattern: "^TestVulkanMatMul2Approx$",
		Category:    "gemv",
	},
	{
		Name:        "matmul3_f32",
		Description: "Triple matrix multiplication (Q/K/V projection)",
		TestPattern: "^TestVulkanMatMul3Approx$",
		Category:    "gemv",
	},
	{
		Name:        "q8_matmul",
		Description: "8-bit quantized matrix multiplication with int8 arithmetic",
		TestPattern: "^TestVulkanQ8MatMulApprox$",
		Category:    "quant",
	},
	{
		Name:        "q8_matmul_wide",
		Description: "Wide-input Q8_0 matrix multiplication",
		TestPattern: "^TestVulkanQ8MatMulWideInput$",
		Category:    "quant",
	},
	{
		Name:        "q8_matmul_vocab",
		Description: "Full vocab-head Q8_0 projection (large dimension)",
		TestPattern: "^TestVulkanQ8MatMulVocabHead$",
		Category:    "quant",
	},
	{
		Name:        "q4k_matmul",
		Description: "Q4_K super-block quantized GEMV (6-bit min/scale)",
		TestPattern: "^TestVulkanQ4KMatMulMatchesCPUReference$",
		Category:    "quant",
	},
	{
		Name:        "q2k_matmul",
		Description: "Q2_K super-block quantized GEMV (2-bit weights, 84-byte superblock)",
		TestPattern: "^TestVulkanQ2KMatMulMatchesCPUReference$",
		Category:    "quant",
	},
	{
		Name:        "rmsnorm",
		Description: "Root-Mean-Square normalization with epsilon scaling",
		TestPattern: "^TestVulkanRMSNormApprox$",
		Category:    "norm",
	},
	{
		Name:        "rmsnorm_matmul",
		Description: "Fused RMSNorm + MatMul single projection",
		TestPattern: "^TestVulkanRMSNormMatMulApprox$",
		Category:    "fused",
	},
	{
		Name:        "rmsnorm_matmul2",
		Description: "Fused RMSNorm + Dual MatMul (gate+up)",
		TestPattern: "^TestVulkanRMSNormMatMul2Approx$",
		Category:    "fused",
	},
	{
		Name:        "rmsnorm_matmul3",
		Description: "Fused RMSNorm + Triple MatMul (Q/K/V)",
		TestPattern: "^TestVulkanRMSNormMatMul3Approx$",
		Category:    "fused",
	},
	{
		Name:        "swiglu",
		Description: "SwiGLU gated activation function",
		TestPattern: "^TestVulkanSwiGLUApprox$",
		Category:    "activation",
	},
	{
		Name:        "swiglu_matmul_add",
		Description: "Fused SwiGLU + MatMul down-proj + Residual Add",
		TestPattern: "^TestVulkanSwiGLUMatMulAddInPlaceApprox$",
		Category:    "fused",
	},
	{
		Name:        "rope",
		Description: "Rotary position embedding with complex rotation",
		TestPattern: "^TestVulkanRoPEApprox$",
		Category:    "positional",
	},
	{
		Name:        "attention",
		Description: "Multi-head attention softmax and value weighted sum",
		TestPattern: "^TestVulkanAttentionApprox$",
		Category:    "attention",
	},
	{
		Name:        "qwen35_gdn_decode",
		Description: "Gated Delta Net recurrent decode in-place token oracle",
		TestPattern: "^TestVulkanQwen35GDNDecodeMatchesCPUOracleInPlace$",
		Category:    "linear_attention",
	},
	{
		Name:        "qwen35_gdn_preprojected",
		Description: "Gated Delta Net preprojected convolution and recurrence",
		TestPattern: "^TestVulkanQwen35GDNPreprojectedParityAndStateContinuity$",
		Category:    "linear_attention",
	},
	{
		Name:        "qwen35_sequence_prefill",
		Description: "Whole-sequence Qwen3.5 hybrid prefill on Vulkan",
		TestPattern: "^TestVulkanQwen35SequenceQuantizedPanelsMatchCPU$",
		Category:    "prefill",
	},
	{
		Name:        embeddingGatherSelector,
		Description: "Qwen3.5 resident embedding-table row gather using one Vulkan multi-region copy",
		TestPattern: "^" + embeddingGatherTestName + "$",
		Category:    "embedding",
	},
	{
		Name:        "f16_kv_contiguize",
		Description: "Pre-attention f16 KV cache contiguization (eliminates LPDDR5X channel camping)",
		TestPattern: "^TestRADVContiguizeShader$",
		Category:    "kv_cache",
	},
}

// DefaultCreditableSubkernelSelectors registers the verified physical Vulkan compute
// sub-kernels that run on the live AMD Strix Halo appliance and earn parity credit.
var DefaultCreditableSubkernelSelectors = []string{
	"argmax",
	"matmul_f32",
	"matmul2_f32",
	"matmul3_f32",
	"q8_matmul",
	"q8_matmul_wide",
	"q8_matmul_vocab",
	"q4k_matmul",
	"q2k_matmul",
	"rmsnorm",
	"rmsnorm_matmul",
	"rmsnorm_matmul2",
	"rmsnorm_matmul3",
	"swiglu",
	"swiglu_matmul_add",
	"rope",
	"attention",
	"qwen35_gdn_decode",
	"qwen35_gdn_preprojected",
}

var executeOneSubkernelFn = executeOneSubkernel

// RunSubkernelTests executes a set of sub-kernel test specs on the target Strix Halo machine.
// It validates sub-kernel selectors and fails fast before attempting network or dispatch.
func RunSubkernelTests(ctx context.Context, target *StrixTarget, selected []string, gitTip ...string) ([]StrixSubkernelResult, error) {
	specs, err := FilterSubkernelSpecs(selected)
	if err != nil {
		return nil, err
	}
	if len(specs) == 0 {
		return nil, fmt.Errorf("amdgpu: zero subkernels selected for execution")
	}

	if target == nil || !target.Reachable {
		host := "unknown"
		if target != nil {
			host = target.Host
		}
		return nil, fmt.Errorf("amdgpu: target %s is not reachable", host)
	}

	if len(gitTip) > 0 && gitTip[0] != "" {
		ctx = WithSourceBinding(ctx, gitTip[0], "")
	}

	results := make([]StrixSubkernelResult, 0, len(specs))

	for _, spec := range specs {
		res := executeOneSubkernelFn(ctx, target, spec)
		results = append(results, res)
	}

	return results, nil
}

// FilterSubkernelSpecs resolves and filters sub-kernel specs by selector.
// Selectors may be a sub-kernel spec name, a category name, or "all" (case-insensitive).
// If selected is empty, all DefaultSubkernelSpecs are returned.
// Returns an error if any selector is unknown or if no specs match.
func FilterSubkernelSpecs(selected []string) ([]SubkernelSpec, error) {
	if len(selected) == 0 {
		return DefaultSubkernelSpecs, nil
	}

	knownNames := make(map[string]bool, len(DefaultSubkernelSpecs))
	knownCategories := make(map[string]bool)
	for _, spec := range DefaultSubkernelSpecs {
		knownNames[strings.ToLower(spec.Name)] = true
		if spec.Category != "" {
			knownCategories[strings.ToLower(spec.Category)] = true
		}
	}

	selMap := make(map[string]bool)
	var unknown []string
	hasAll := false

	for _, raw := range selected {
		trimmed := strings.TrimSpace(raw)
		norm := strings.ToLower(trimmed)
		if norm == "" {
			unknown = append(unknown, raw)
			continue
		}
		if norm == "all" {
			hasAll = true
			selMap["all"] = true
			continue
		}
		if !knownNames[norm] && !knownCategories[norm] {
			unknown = append(unknown, raw)
			continue
		}
		selMap[norm] = true
	}

	if len(unknown) > 0 {
		return nil, fmt.Errorf("amdgpu: unknown subkernel selector(s): %s", strings.Join(unknown, ", "))
	}

	if hasAll {
		return DefaultSubkernelSpecs, nil
	}

	var out []SubkernelSpec
	for _, spec := range DefaultSubkernelSpecs {
		if selMap[strings.ToLower(spec.Name)] || selMap[strings.ToLower(spec.Category)] {
			out = append(out, spec)
		}
	}

	if len(out) == 0 {
		return nil, fmt.Errorf("amdgpu: no subkernel specs matched selector(s): %s", strings.Join(selected, ", "))
	}

	return out, nil
}

func filterSubkernelSpecs(selected []string) ([]SubkernelSpec, error) {
	return FilterSubkernelSpecs(selected)
}

func executeOneSubkernel(ctx context.Context, target *StrixTarget, spec SubkernelSpec) StrixSubkernelResult {
	start := time.Now()
	res := StrixSubkernelResult{
		Name:       spec.Name,
		Status:     "SKIPPED",
		Iterations: 1,
		Parity: StrixParityVerdict{
			ReferenceGEMV: "CPU reference (" + spec.Name + ")",
			Passed:        false,
		},
		Metrics: make(map[string]any),
	}

	sb, ok := SourceBindingFromContext(ctx)
	if !ok || sb.WorkDir == "" || !sha256RE.MatchString(sb.BinarySHA256) || !sha256RE.MatchString(sb.ShaderBundleSHA256) {
		res.Status = "FAIL"
		res.Error = "source binding mismatch: missing prepared candidate source/build binding"
		return res
	}
	testCmd := fmt.Sprintf(`FAK_VULKAN_SPIRV=%s FAK_VULKAN_REQUIRE_DEVICE=1 FAK_VULKAN_EXPECT_DEVICE=8060S %s -test.run %s -test.v`, shellQuote(sb.WorkDir+"/build/spirv"), shellQuote(sb.WorkDir+"/build/compute.test"), shellQuote(spec.TestPattern))
	command := buildStrixAdmissionCommand(target, sb, testCmd)
	out, err := runStrixTargetCommand(ctx, target, command, nil)
	duration := time.Since(start)
	res.DurationUS = duration.Microseconds()
	outputStr := string(out)
	res.Evidence = executionEvidenceFromOutput(outputStr, sb)

	if err != nil {
		res.Status = "FAIL"
		res.Error = fmt.Sprintf("execution failed: %v\n%s", err, truncateOutput(outputStr, 2000))
		return res
	}

	if strings.Contains(outputStr, "--- SKIP:") {
		res.Status = "SKIPPED"
		return res
	}

	// Truthful parity verification: parse and validate the contract event.
	// PASS text alone, missing, duplicate, or mismatched events remain non-credit.
	event, parseErr := ParseStrixSubkernelParity(outputStr, spec.Name)
	if parseErr != nil {
		res.Status = "FAIL"
		res.Error = fmt.Sprintf("parity contract validation failed: %v\n%s", parseErr, truncateOutput(outputStr, 200))
		return res
	}

	res.Status = "PASS"
	res.Parity.Passed = event.Passed
	if event.Observed.ArgmaxExact != nil {
		res.Parity.ArgmaxExact = *event.Observed.ArgmaxExact
	}
	if event.Observed.Cosine != nil {
		res.Parity.LogitCosineSimilarity = *event.Observed.Cosine
	}
	if event.Observed.MaxAbsDelta != nil {
		res.Parity.MaxAbsoluteDelta = *event.Observed.MaxAbsDelta
	}
	contract, _ := LookupSubkernelParityContract(spec.Name)
	res.ParityEvents = []StrixParityEvent{receiptParityEventFromSubkernel(*event, contract)}
	res.Metrics["category"] = spec.Category
	res.Metrics["wall_ms"] = duration.Milliseconds()
	res.Metrics["oracle_kind"] = event.OracleKind
	res.Metrics["case_count"] = event.CaseCount
	res.Metrics["device_observed"] = event.DeviceObserved
	return res
}

func receiptParityEventFromSubkernel(event StrixSubkernelParityEvent, contract SubkernelParityContract) StrixParityEvent {
	result := StrixParityEvent{
		OracleKind:     StrixOracleKind(event.OracleKind),
		CaseCount:      event.CaseCount,
		DeviceObserved: event.DeviceObserved,
		Engine:         event.Engine,
		Passed:         event.Passed,
		Observed: StrixParityMetrics{
			ArgmaxExact:      event.Observed.ArgmaxExact,
			CosineSimilarity: event.Observed.Cosine,
			MaxAbsoluteDelta: event.Observed.MaxAbsDelta,
			MaxSourceDelta:   event.Observed.MaxSourceDelta,
			StateIdentity:    event.Observed.StateIdentity,
			FiniteOutput:     event.Observed.FiniteOutput,
		},
		Bounds: StrixParityBounds{
			ExactMatch:     boolPtrOrNil(contract.Bounds.RequireArgmaxExact),
			MinCosine:      contract.Bounds.MinCosine,
			MaxAbsDelta:    contract.Bounds.MaxAbsDelta,
			MaxSourceDelta: contract.Bounds.MaxSourceDelta,
			StateIdentity:  boolPtrOrNil(contract.Bounds.RequireStateIdentity),
			FiniteOutput:   boolPtrOrNil(contract.Bounds.RequireFinite),
		},
		Detail: event.Selector + ":" + event.TestName,
	}
	if result.Bounds.MinCosine != nil {
		result.Bounds.CosineComparison = ">="
	}
	if result.Bounds.MaxAbsDelta != nil {
		result.Bounds.MaxAbsComparison = "<="
	}
	if result.Bounds.MaxSourceDelta != nil {
		result.Bounds.SourceDeltaComparison = "<="
	}
	if event.OracleKind == OracleExactArgmax || event.OracleKind == OracleCosineArgmax {
		result.Bounds.Comparison = "=="
	}
	if event.OracleKind == OracleHostContract {
		result.Observed.ContractHolds = event.Observed.FiniteOutput
		result.Observed.ContractName = event.Selector
		result.Bounds.ContractExpected = boolPtr(true)
	}
	return result
}

func boolPtrOrNil(required bool) *bool {
	if !required {
		return nil
	}
	return boolPtr(true)
}

func executionEvidenceFromOutput(out string, sb SourceBinding) StrixExecutionEvidence {
	zero := 0
	e := StrixExecutionEvidence{
		SourceArchiveSHA256: sb.SourceArchiveSHA256,
		BinarySHA256:        sb.BinarySHA256,
		ShaderBundleSHA256:  sb.ShaderBundleSHA256,
		CommandSHA256:       markerValue(out, "FAK_STRIX_COMMAND_SHA256="),
		DeviceIdentity:      markerValue(out, "FAK_STRIX_DEVICE="),
		EngineIdentity:      markerValue(out, "FAK_STRIX_ENGINE="),
		ArtifactRehashed:    strings.Count(out, "FAK_STRIX_ARTIFACT_REHASH=1") == 1,
		LeasePathSHA256:     markerValue(out, "FAK_STRIX_LEASE_SHA256="),
		AdmissionWaitMS:     sb.AdmissionWait.Milliseconds(),
		RawOutputSHA256:     digestBytes([]byte(out)),
		RawOutputBytes:      len(out),
	}
	if raw := markerValue(out, "FAK_STRIX_DEVICE_TIMEOUT_MS="); raw != "" {
		e.DeviceTimeoutMS, _ = strconv.ParseInt(raw, 10, 64)
	}
	ordinal := 0
	for _, line := range strings.Split(out, "\n") {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		ordinal++
		if strings.Contains(line, "FAK_STRIX_ADMISSION_ACQUIRED=1") {
			if e.AcquireOrdinal == 0 {
				e.AcquireOrdinal, e.Acquired = ordinal, true
			} else {
				e.Acquired = false
			}
		}
		if strings.Contains(line, "FAK_STRIX_ADMISSION_RELEASED=1") {
			if e.ReleaseOrdinal == 0 {
				e.ReleaseOrdinal, e.Released = ordinal, true
			} else {
				e.Released = false
			}
		}
	}
	if raw := markerValue(out, "FAK_STRIX_EXIT="); raw != "" {
		if n, err := strconv.Atoi(raw); err == nil {
			zero = n
			e.ExitCode = &zero
		}
	}
	return e
}

// findParityEventCandidates scans output for candidate subkernel parity JSON blocks.
func findParityEventCandidates(output string) []string {
	return findJSONEventCandidates(output, "subkernel-parity")
}

// findJSONEventCandidates extracts balanced JSON objects containing token from
// noisy `go test -v` output. Callers remain responsible for schema and semantic
// validation; this helper only preserves the complete structured envelope.
func findJSONEventCandidates(output, token string) []string {
	var candidates []string
	idx := 0
	for {
		pos := strings.Index(output[idx:], token)
		if pos == -1 {
			break
		}
		tokenIdx := idx + pos

		// Scan backward to find the opening '{'
		openBrace := -1
		for i := tokenIdx; i >= idx; i-- {
			if output[i] == '{' {
				openBrace = i
				break
			}
		}

		if openBrace == -1 {
			candidates = append(candidates, output[tokenIdx:tokenIdx+len(token)])
			idx = tokenIdx + len(token)
			continue
		}

		// Scan forward from openBrace tracking balanced braces
		depth := 0
		inString := false
		escaped := false
		closeBrace := -1

		for i := openBrace; i < len(output); i++ {
			c := output[i]
			if escaped {
				escaped = false
				continue
			}
			if c == '\\' && inString {
				escaped = true
				continue
			}
			if c == '"' {
				inString = !inString
				continue
			}
			if !inString {
				if c == '{' {
					depth++
				} else if c == '}' {
					depth--
					if depth == 0 {
						closeBrace = i
						break
					}
				}
			}
		}

		if closeBrace != -1 {
			candidates = append(candidates, output[openBrace:closeBrace+1])
			idx = closeBrace + 1
		} else {
			candidates = append(candidates, output[openBrace:])
			idx = len(output)
		}
	}

	return candidates
}

// ParseStrixSubkernelParity extracts and validates a fak.strix.subkernel-parity/v1 event from test output.
// If expectedSelector is provided, it validates that the event matches that selector's registered contract.
// Otherwise, it validates against the contract registered for the event's own declared selector.
func ParseStrixSubkernelParity(output string, expectedSelector ...string) (*StrixSubkernelParityEvent, error) {
	candidates := findParityEventCandidates(output)
	if len(candidates) == 0 {
		return nil, ErrSubkernelParityAbsent
	}
	if len(candidates) > 1 {
		return nil, fmt.Errorf("%w: expected 1 parity event, found %d", ErrSubkernelParityDuplicate, len(candidates))
	}

	candidate := candidates[0]

	// Check for non-finite tokens in raw JSON
	if strings.Contains(candidate, `"NaN"`) || strings.Contains(candidate, `NaN`) ||
		strings.Contains(candidate, `"Inf"`) || strings.Contains(candidate, `"+Inf"`) ||
		strings.Contains(candidate, `"-Inf"`) || strings.Contains(candidate, `"Infinity"`) {
		return nil, fmt.Errorf("%w: non-finite metric encountered in parity event", ErrSubkernelParityNonFinite)
	}

	var event StrixSubkernelParityEvent
	if err := json.Unmarshal([]byte(candidate), &event); err != nil {
		return nil, fmt.Errorf("%w: %v", ErrSubkernelParityMalformed, err)
	}

	if event.Schema != StrixSubkernelParitySchema {
		return nil, fmt.Errorf("%w: invalid schema %q (want %q)", ErrSubkernelParityMalformed, event.Schema, StrixSubkernelParitySchema)
	}
	if event.Selector == "" || event.TestName == "" || event.OracleKind == "" {
		return nil, fmt.Errorf("%w: missing required event field (selector=%q, test_name=%q, oracle_kind=%q)", ErrSubkernelParityMalformed, event.Selector, event.TestName, event.OracleKind)
	}
	if event.CaseCount <= 0 {
		return nil, fmt.Errorf("%w: case_count must be positive, got %d", ErrSubkernelParityMalformed, event.CaseCount)
	}

	// Validate non-finite floats
	if event.Observed.Cosine != nil && (math.IsNaN(*event.Observed.Cosine) || math.IsInf(*event.Observed.Cosine, 0)) {
		return nil, fmt.Errorf("%w: cosine is non-finite: %v", ErrSubkernelParityNonFinite, *event.Observed.Cosine)
	}
	if event.Observed.MaxAbsDelta != nil && (math.IsNaN(*event.Observed.MaxAbsDelta) || math.IsInf(*event.Observed.MaxAbsDelta, 0)) {
		return nil, fmt.Errorf("%w: max_abs_delta is non-finite: %v", ErrSubkernelParityNonFinite, *event.Observed.MaxAbsDelta)
	}
	if event.Observed.MaxSourceDelta != nil && (math.IsNaN(*event.Observed.MaxSourceDelta) || math.IsInf(*event.Observed.MaxSourceDelta, 0)) {
		return nil, fmt.Errorf("%w: max_source_delta is non-finite: %v", ErrSubkernelParityNonFinite, *event.Observed.MaxSourceDelta)
	}

	// Resolve target contract
	targetSelector := event.Selector
	if len(expectedSelector) > 0 && expectedSelector[0] != "" {
		if event.Selector != expectedSelector[0] {
			return nil, fmt.Errorf("%w: selector %q does not match expected %q", ErrSubkernelParityMismatch, event.Selector, expectedSelector[0])
		}
		targetSelector = expectedSelector[0]
	}

	contract, ok := LookupSubkernelParityContract(targetSelector)
	if !ok {
		return nil, fmt.Errorf("%w: unknown subkernel selector %q", ErrSubkernelParityMismatch, targetSelector)
	}

	// Validate selector, test_name, oracle_kind agreement
	if event.TestName != contract.TestName {
		return nil, fmt.Errorf("%w: test_name %q does not match registered test %q for selector %q", ErrSubkernelParityMismatch, event.TestName, contract.TestName, contract.Selector)
	}
	if event.OracleKind != contract.OracleKind {
		return nil, fmt.Errorf("%w: oracle_kind %q does not match registered oracle %q for selector %q", ErrSubkernelParityMismatch, event.OracleKind, contract.OracleKind, contract.Selector)
	}

	// Validate engine and device requirements
	if contract.DeviceObserved {
		if !event.DeviceObserved {
			return nil, fmt.Errorf("%w: selector %q requires device_observed=true", ErrSubkernelParityWrongEngine, contract.Selector)
		}
		if event.Engine != StrixVulkanEngine {
			return nil, fmt.Errorf("%w: selector %q requires engine=%q, got %q", ErrSubkernelParityWrongEngine, contract.Selector, StrixVulkanEngine, event.Engine)
		}
	} else {
		if event.DeviceObserved {
			return nil, fmt.Errorf("%w: host contract %q cannot declare device_observed=true (cannot earn physical parity)", ErrSubkernelParityWrongEngine, contract.Selector)
		}
		if event.Engine == StrixVulkanEngine {
			return nil, fmt.Errorf("%w: host contract %q cannot declare engine=%q", ErrSubkernelParityWrongEngine, contract.Selector, StrixVulkanEngine)
		}
	}

	// Validate pass state
	if !event.Passed {
		return nil, fmt.Errorf("%w: event declared passed=false", ErrSubkernelParityOutOfBounds)
	}

	// Exact argmax must not carry fabricated cosine
	if contract.OracleKind == OracleExactArgmax && event.Observed.Cosine != nil {
		return nil, fmt.Errorf("%w: exact_argmax oracle must not carry a cosine metric", ErrSubkernelParityOutOfBounds)
	}

	// Validate bounds
	if contract.Bounds.RequireArgmaxExact {
		if event.Observed.ArgmaxExact == nil || !*event.Observed.ArgmaxExact {
			return nil, fmt.Errorf("%w: selector %q requires argmax_exact=true", ErrSubkernelParityOutOfBounds, contract.Selector)
		}
	}
	if contract.Bounds.MinCosine != nil {
		if event.Observed.Cosine == nil {
			return nil, fmt.Errorf("%w: selector %q missing required cosine metric", ErrSubkernelParityOutOfBounds, contract.Selector)
		}
		if *event.Observed.Cosine < *contract.Bounds.MinCosine {
			return nil, fmt.Errorf("%w: selector %q cosine %v < minimum %v", ErrSubkernelParityOutOfBounds, contract.Selector, *event.Observed.Cosine, *contract.Bounds.MinCosine)
		}
	}
	if contract.Bounds.MaxAbsDelta != nil {
		if event.Observed.MaxAbsDelta == nil {
			return nil, fmt.Errorf("%w: selector %q missing required max_abs_delta metric", ErrSubkernelParityOutOfBounds, contract.Selector)
		}
		if *event.Observed.MaxAbsDelta > *contract.Bounds.MaxAbsDelta {
			return nil, fmt.Errorf("%w: selector %q max_abs_delta %v > maximum %v", ErrSubkernelParityOutOfBounds, contract.Selector, *event.Observed.MaxAbsDelta, *contract.Bounds.MaxAbsDelta)
		}
	}
	if contract.Bounds.RequireSourceMutationCheck {
		if event.Observed.MaxSourceDelta == nil {
			return nil, fmt.Errorf("%w: selector %q missing required max_source_delta metric", ErrSubkernelParityOutOfBounds, contract.Selector)
		}
		if contract.Bounds.MaxSourceDelta != nil && *event.Observed.MaxSourceDelta > *contract.Bounds.MaxSourceDelta {
			return nil, fmt.Errorf("%w: selector %q max_source_delta %v > maximum %v", ErrSubkernelParityOutOfBounds, contract.Selector, *event.Observed.MaxSourceDelta, *contract.Bounds.MaxSourceDelta)
		}
	}
	if contract.Bounds.RequireStateIdentity {
		if event.Observed.StateIdentity == nil || !*event.Observed.StateIdentity {
			return nil, fmt.Errorf("%w: selector %q requires state_identity=true", ErrSubkernelParityOutOfBounds, contract.Selector)
		}
	}
	if contract.Bounds.RequireFinite {
		if event.Observed.FiniteOutput == nil || !*event.Observed.FiniteOutput {
			return nil, fmt.Errorf("%w: selector %q requires finite_output=true", ErrSubkernelParityOutOfBounds, contract.Selector)
		}
	}

	return &event, nil
}

var cosineRe = regexp.MustCompile(`cosine\s+([0-9\.]+)`)

func extractCosine(out string) float64 {
	matches := cosineRe.FindStringSubmatch(out)
	if len(matches) >= 2 {
		if c, err := strconv.ParseFloat(matches[1], 64); err == nil {
			return c
		}
	}
	return 0.0
}

func truncateOutput(s string, maxLen int) string {
	s = strings.TrimSpace(s)
	if len(s) > maxLen {
		return s[:maxLen] + "..."
	}
	return s
}
