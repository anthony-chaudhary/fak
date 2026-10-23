// Prior-art: llama.cpp Metal / MLX (https://github.com/ggml-org/llama.cpp)
// Oracle: cpuref (GEMV cosine)

//go:build !(darwin && arm64 && cgo)

package metalgemm

import (
	"errors"
	"fmt"
	"sync"
)

// ICBDispatchFallbackReason explains why a decode step bypassed ICB replay.
type ICBDispatchFallbackReason string

const (
	FallbackReasonNone             ICBDispatchFallbackReason = ""
	FallbackReasonDeviceIneligible ICBDispatchFallbackReason = "metal_icb_ineligible"
	FallbackReasonBatchSize        ICBDispatchFallbackReason = "variable_batch_size"
	FallbackReasonPrefill          ICBDispatchFallbackReason = "dynamic_prefill_ineligible"
	FallbackReasonForced           ICBDispatchFallbackReason = "forced_fallback"
	FallbackReasonGraphUnrecorded  ICBDispatchFallbackReason = "graph_not_recorded"
)

// ICBDispatchConfig configures the static decode execution graph for Metal ICB replay.
type ICBDispatchConfig struct {
	NumLayers                int
	HiddenDim                int
	NumHeads                 int
	NumKVHeads               int
	HeadDim                  int
	VocabSize                int
	MaxSeqLen                int
	BatchSize                int
	ForceFallback            bool
	StorageMode              ICBStorageMode
	MaxKernelBufferBindCount int
	CommandTypes             []ICBCommandType
}

// DefaultICBDispatchConfig returns a standard configuration for a specified layer count.
func DefaultICBDispatchConfig(layers int) ICBDispatchConfig {
	if layers <= 0 {
		layers = 64
	}
	return ICBDispatchConfig{
		NumLayers:                layers,
		HiddenDim:                5120,
		NumHeads:                 40,
		NumKVHeads:               8,
		HeadDim:                  128,
		VocabSize:                152064,
		MaxSeqLen:                4096,
		BatchSize:                1,
		ForceFallback:            false,
		StorageMode:              ICBStorageModeShared,
		MaxKernelBufferBindCount: 16,
		CommandTypes: []ICBCommandType{
			ICBCommandTypeConcurrentDispatch,
			ICBCommandTypeConcurrentDispatchThreads,
		},
	}
}

// ICBStepParams contains inputs and dynamic parameters for a single decode token step.
type ICBStepParams struct {
	StepL          int
	SeqLen         int
	BatchSize      int
	KVRowStride    int
	DynamicPrefill bool
	InputEmbed     []float32
	WantLogits     bool
}

// ICBDynamicState tracks the currently registered dynamic execution offsets.
type ICBDynamicState struct {
	StepL        int
	SeqLen       int
	KVRowStride  int
	KVByteOffset int
}

// ICBEvidenceClass distinguishes diagnostic data from measured execution.
type ICBEvidenceClass string

const (
	ICBEvidenceSynthetic   ICBEvidenceClass = "synthetic"
	ICBEvidenceUnavailable ICBEvidenceClass = "unavailable"
)

// ICBUnavailableError reports that the standalone API cannot execute model work.
// The live Q8 decode.m ICB path is separate and is not affected by this result.
type ICBUnavailableError struct{ Reason string }

func (e *ICBUnavailableError) Error() string { return "icb: execution unavailable: " + e.Reason }

// ICBDispatchReceipt describes the standalone diagnostic path. Zero durations
// with unavailable evidence are absent measurements, not zero-cost execution.
type ICBDispatchReceipt struct {
	EvidenceClass     ICBEvidenceClass          `json:"evidence_class"`
	UnavailableReason string                    `json:"unavailable_reason,omitempty"`
	CommandBuffers    int                       `json:"command_buffers"`
	Encoders          int                       `json:"encoders"`
	ICBDispatches     int                       `json:"icb_dispatches"`
	HostEncodeMs      float64                   `json:"host_encode_ms"`
	HostWaitMs        float64                   `json:"host_wait_ms"`
	GPUMs             float64                   `json:"gpu_ms"`
	TotalMs           float64                   `json:"total_ms"`
	ICBUsed           bool                      `json:"icb_used"`
	FallbackReason    ICBDispatchFallbackReason `json:"fallback_reason"`
	StepL             int                       `json:"step_l"`
}

// ICBStepOutput holds the outputs and receipts from a single decode forward step.
type ICBStepOutput struct {
	Receipt        ICBDispatchReceipt
	LastPre        []float32
	NewK           []float32
	NewV           []float32
	Logits         []float32
	ICBUsed        bool
	FallbackReason ICBDispatchFallbackReason
}

// ICBBenchmarkResult describes a requested standalone comparison. Its topology
// is diagnostic metadata, not evidence that dispatches or model work occurred.
type ICBBenchmarkResult struct {
	EvidenceClass             ICBEvidenceClass `json:"evidence_class"`
	UnavailableReason         string           `json:"unavailable_reason,omitempty"`
	TopologyLayers            int              `json:"topology_layers"`
	DispatchesPerStep         int              `json:"dispatches_per_step"`
	SequentialHostEncodeMs    float64          `json:"sequential_host_encode_ms"`
	ICBHostEncodeMs           float64          `json:"icb_host_encode_ms"`
	LatencyReductionMs        float64          `json:"latency_reduction_ms"`
	LatencyReductionPercent   float64          `json:"latency_reduction_percent"`
	SpeedupFactor             float64          `json:"speedup_factor"`
	NumericalDivergenceMax    float64          `json:"numerical_divergence_max"`
	PassesLatencyThreshold    bool             `json:"passes_latency_threshold"`
	PassesDivergenceThreshold bool             `json:"passes_divergence_threshold"`
}

// ValidateMeasuredEvidence fails closed: this standalone API has no model
// execution witness, even if a caller relabels its diagnostic fields.
func (r ICBDispatchReceipt) ValidateMeasuredEvidence() error {
	return &ICBUnavailableError{Reason: "standalone dispatch receipts are not measured hardware evidence"}
}

// ValidateMeasuredEvidence rejects diagnostic benchmarks as hardware acceptance.
func (r ICBBenchmarkResult) ValidateMeasuredEvidence() error {
	return &ICBUnavailableError{Reason: "standalone benchmarks are not measured hardware evidence"}
}

// ICBDispatcher manages standalone descriptor and dynamic-offset diagnostics.
// It does not execute model work or a sequential model fallback.
type ICBDispatcher struct {
	mu       sync.Mutex
	cfg      ICBDispatchConfig
	desc     *ICBDescriptor
	recorded bool
	closed   bool
	dynState ICBDynamicState
}

// ICBDispatchAvailable reports false on non-Darwin/non-Metal platforms.
func ICBDispatchAvailable() bool {
	return false
}

// NewICBDispatcher initializes an ICB dispatcher in stub environments.
func NewICBDispatcher(cfg ICBDispatchConfig) (*ICBDispatcher, error) {
	if cfg.NumLayers <= 0 {
		return nil, errors.New("icb: num layers must be positive")
	}
	if cfg.HiddenDim <= 0 {
		return nil, errors.New("icb: hidden dimension must be positive")
	}
	if cfg.MaxKernelBufferBindCount <= 0 {
		cfg.MaxKernelBufferBindCount = 16
	}
	if len(cfg.CommandTypes) == 0 {
		cfg.CommandTypes = []ICBCommandType{
			ICBCommandTypeConcurrentDispatch,
			ICBCommandTypeConcurrentDispatchThreads,
		}
	}
	if cfg.StorageMode == "" {
		cfg.StorageMode = ICBStorageModeShared
	}

	totalDispatches := cfg.NumLayers*15 + 2
	desc := &ICBDescriptor{
		CommandTypes:             cfg.CommandTypes,
		InheritBuffers:           false,
		InheritPipelineState:     false,
		MaxKernelBufferBindCount: cfg.MaxKernelBufferBindCount,
		MaxCallCount:             totalDispatches,
		StorageMode:              cfg.StorageMode,
	}
	if err := desc.Validate(); err != nil {
		return nil, fmt.Errorf("icb: invalid descriptor: %w", err)
	}

	return &ICBDispatcher{
		cfg:  cfg,
		desc: desc,
	}, nil
}

// Config returns the configuration of this dispatcher.
func (d *ICBDispatcher) Config() ICBDispatchConfig {
	d.mu.Lock()
	defer d.mu.Unlock()
	return d.cfg
}

// IsAvailable reports false in stub environments.
func (d *ICBDispatcher) IsAvailable() bool {
	return false
}

// IsRecorded reports whether the diagnostic descriptor has been validated.
func (d *ICBDispatcher) IsRecorded() bool {
	d.mu.Lock()
	defer d.mu.Unlock()
	return d.recorded
}

// Descriptor returns the allocated MTLIndirectCommandBuffer descriptor.
func (d *ICBDispatcher) Descriptor() (*ICBDescriptor, error) {
	d.mu.Lock()
	defer d.mu.Unlock()
	if d.closed {
		return nil, errors.New("icb: dispatcher is closed")
	}
	return d.desc, nil
}

// RecordStaticGraph validates the diagnostic descriptor; no model commands are encoded.
func (d *ICBDispatcher) RecordStaticGraph() error {
	d.mu.Lock()
	defer d.mu.Unlock()
	if d.closed {
		return errors.New("icb: dispatcher is closed")
	}
	if err := d.desc.Validate(); err != nil {
		return err
	}
	d.recorded = true
	return nil
}

// UpdateDynamicOffsets updates dynamic argument buffer parameters and sliding KV offsets.
func (d *ICBDispatcher) UpdateDynamicOffsets(stepL, seqLen, kvRowStride int) error {
	d.mu.Lock()
	defer d.mu.Unlock()
	if d.closed {
		return errors.New("icb: dispatcher is closed")
	}
	if stepL < 0 || seqLen < 0 || kvRowStride <= 0 {
		return errors.New("icb: invalid dynamic dimensions")
	}

	byteOff := stepL * kvRowStride * 2
	d.dynState = ICBDynamicState{
		StepL:        stepL,
		SeqLen:       seqLen,
		KVRowStride:  kvRowStride,
		KVByteOffset: byteOff,
	}
	return nil
}

// DynamicOffsets returns a snapshot of the registered step index and per-layer sliding KV byte offsets.
func (d *ICBDispatcher) DynamicOffsets() ICBDynamicState {
	d.mu.Lock()
	defer d.mu.Unlock()
	return d.dynState
}

// DispatchStep returns diagnostic selection metadata and a typed unavailable
// error. This abstraction has no model weights, encoded kernels or real fallback.
func (d *ICBDispatcher) DispatchStep(params *ICBStepParams) (*ICBStepOutput, error) {
	d.mu.Lock()
	defer d.mu.Unlock()
	if d.closed {
		return nil, errors.New("icb: dispatcher is closed")
	}
	if params == nil {
		return nil, errors.New("icb: nil step params")
	}

	fallbackReason := FallbackReasonDeviceIneligible
	if params.BatchSize > 1 || d.cfg.BatchSize > 1 {
		fallbackReason = FallbackReasonBatchSize
	} else if params.DynamicPrefill {
		fallbackReason = FallbackReasonPrefill
	} else if d.cfg.ForceFallback {
		fallbackReason = FallbackReasonForced

	}
	reason := "Metal ICB is unavailable on this host"
	return &ICBStepOutput{
		Receipt: ICBDispatchReceipt{
			EvidenceClass:     ICBEvidenceUnavailable,
			UnavailableReason: reason,
			FallbackReason:    fallbackReason,
			StepL:             params.StepL,
		},
		FallbackReason: fallbackReason,
	}, &ICBUnavailableError{Reason: reason}
}

// Benchmark returns the requested topology and a typed unavailable error.
// CPU busy loops and analytical estimates cannot witness model latency or parity.
func (d *ICBDispatcher) Benchmark(numSteps, topologyLayers int) (*ICBBenchmarkResult, error) {
	d.mu.Lock()
	defer d.mu.Unlock()
	if d.closed {
		return nil, errors.New("icb: dispatcher is closed")
	}
	if topologyLayers <= 0 {
		topologyLayers = 64
	}
	reason := "Metal ICB is unavailable on this host"
	return &ICBBenchmarkResult{
		EvidenceClass:     ICBEvidenceUnavailable,
		UnavailableReason: reason,
		TopologyLayers:    topologyLayers,
		DispatchesPerStep: topologyLayers*15 + 2,
	}, &ICBUnavailableError{Reason: reason}
}

// Close is a no-op in stub environments.
func (d *ICBDispatcher) Close() error {
	d.mu.Lock()
	defer d.mu.Unlock()
	d.closed = true
	return nil
}
