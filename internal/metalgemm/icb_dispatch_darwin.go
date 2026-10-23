// Prior-art: llama.cpp Metal / MLX (https://github.com/ggml-org/llama.cpp)
// Oracle: cpuref (GEMV cosine)

//go:build darwin && arm64 && cgo

package metalgemm

/*
#cgo CFLAGS: -x objective-c
#cgo LDFLAGS: -framework Metal -framework Foundation
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

typedef struct {
    int available;
    void *device;
    void *icb;
    void *arg_buffer;
    int max_commands;
    int max_binds;
} mg_icb_handle_t;

typedef struct {
    uint32_t step_l;
    uint32_t seq_len;
    uint32_t kv_row_stride;
    uint32_t flags;
    uint64_t active_x_buffer;
    uint64_t next_x_buffer;
    uint64_t page_table_base;
    uint64_t kv_buf_base_k;
    uint64_t kv_buf_base_v;
    uint64_t logits_buffer;
} mg_decode_step_params_t;

static int mg_icb_is_available(void) {
    @autoreleasepool {
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        if (dev == nil) return 0;
        MTLIndirectCommandBufferDescriptor *desc = [[MTLIndirectCommandBufferDescriptor alloc] init];
        desc.commandTypes = MTLIndirectCommandTypeConcurrentDispatch | MTLIndirectCommandTypeConcurrentDispatchThreads;
        desc.inheritBuffers = NO;
        desc.inheritPipelineState = NO;
        desc.maxKernelBufferBindCount = 16;
        id<MTLIndirectCommandBuffer> icb = [dev newIndirectCommandBufferWithDescriptor:desc maxCommandCount:1 options:MTLResourceStorageModeShared];
        return icb != nil ? 1 : 0;
    }
}

static mg_icb_handle_t mg_icb_create(int max_commands, int max_binds) {
    mg_icb_handle_t h;
    memset(&h, 0, sizeof(h));
    @autoreleasepool {
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        if (dev == nil) return h;

        MTLIndirectCommandBufferDescriptor *desc = [[MTLIndirectCommandBufferDescriptor alloc] init];
        desc.commandTypes = MTLIndirectCommandTypeConcurrentDispatch | MTLIndirectCommandTypeConcurrentDispatchThreads;
        desc.inheritBuffers = NO;
        desc.inheritPipelineState = NO;
        desc.maxKernelBufferBindCount = (max_binds > 0 && max_binds <= 31) ? max_binds : 16;

        id<MTLIndirectCommandBuffer> icb = [dev newIndirectCommandBufferWithDescriptor:desc
                                                                       maxCommandCount:(NSUInteger)max_commands
                                                                               options:MTLResourceStorageModeShared];
        if (icb == nil) return h;

        id<MTLBuffer> argBuf = [dev newBufferWithLength:sizeof(mg_decode_step_params_t) options:MTLResourceStorageModeShared];
        if (argBuf == nil) return h;

        h.available = 1;
        h.device = (void *)CFBridgingRetain(dev);
        h.icb = (void *)CFBridgingRetain(icb);
        h.arg_buffer = (void *)CFBridgingRetain(argBuf);
        h.max_commands = max_commands;
        h.max_binds = desc.maxKernelBufferBindCount;
        return h;
    }
}

static void mg_icb_release(mg_icb_handle_t *h) {
    if (h == NULL) return;
    if (h->arg_buffer) {
        CFRelease(h->arg_buffer);
        h->arg_buffer = NULL;
    }
    if (h->icb) {
        CFRelease(h->icb);
        h->icb = NULL;
    }
    if (h->device) {
        CFRelease(h->device);
        h->device = NULL;
    }
    h->available = 0;
}

static void mg_icb_update_dynamic_args(mg_icb_handle_t *h, uint32_t step_l, uint32_t seq_len, uint32_t kv_row_stride) {
    if (h == NULL || h->arg_buffer == NULL) return;
    id<MTLBuffer> buf = (__bridge id<MTLBuffer>)h->arg_buffer;
    mg_decode_step_params_t *params = (mg_decode_step_params_t *)buf.contents;
    params->step_l = step_l;
    params->seq_len = seq_len;
    params->kv_row_stride = kv_row_stride;
    params->flags = 0;
}

*/
import "C"

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
// It does not record or execute the live decode.m model graph.
type ICBDispatcher struct {
	mu           sync.Mutex
	cfg          ICBDispatchConfig
	desc         *ICBDescriptor
	nativeHandle C.mg_icb_handle_t
	recorded     bool
	closed       bool
	available    bool
	dynState     ICBDynamicState
}

// ICBDispatchAvailable reports true if the host Apple Silicon device supports compute ICBs.
func ICBDispatchAvailable() bool {
	return Available() && C.mg_icb_is_available() == 1
}

// NewICBDispatcher initializes standalone descriptor and dynamic-offset diagnostics.
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

	available := ICBDispatchAvailable()
	var handle C.mg_icb_handle_t
	if available {
		handle = C.mg_icb_create(C.int(totalDispatches), C.int(cfg.MaxKernelBufferBindCount))
		available = handle.available == 1
	}

	return &ICBDispatcher{
		cfg:          cfg,
		desc:         desc,
		nativeHandle: handle,
		available:    available,
	}, nil
}

// Config returns the configuration of this dispatcher.
func (d *ICBDispatcher) Config() ICBDispatchConfig {
	d.mu.Lock()
	defer d.mu.Unlock()
	return d.cfg
}

// IsAvailable reports whether the underlying hardware and runtime support ICB replay.
func (d *ICBDispatcher) IsAvailable() bool {
	d.mu.Lock()
	defer d.mu.Unlock()
	return d.available
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

	if d.available && d.nativeHandle.available == 1 {
		C.mg_icb_update_dynamic_args(&d.nativeHandle, C.uint32_t(stepL), C.uint32_t(seqLen), C.uint32_t(kvRowStride))
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

	fallbackReason := FallbackReasonNone
	if params.BatchSize > 1 || d.cfg.BatchSize > 1 {
		fallbackReason = FallbackReasonBatchSize
	} else if params.DynamicPrefill {
		fallbackReason = FallbackReasonPrefill
	} else if d.cfg.ForceFallback {
		fallbackReason = FallbackReasonForced
	} else if !d.available {
		fallbackReason = FallbackReasonDeviceIneligible
	} else if !d.recorded {
		fallbackReason = FallbackReasonGraphUnrecorded
	}
	reason := "standalone model execution is not implemented; live Q8 decode uses a separate path"
	if !d.available {
		reason = "Metal ICB is unavailable on this host"
	}
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
	reason := "standalone model benchmarking is not implemented"
	if !d.available {
		reason = "Metal ICB is unavailable on this host"
	}
	return &ICBBenchmarkResult{
		EvidenceClass:     ICBEvidenceUnavailable,
		UnavailableReason: reason,
		TopologyLayers:    topologyLayers,
		DispatchesPerStep: topologyLayers*15 + 2,
	}, &ICBUnavailableError{Reason: reason}
}

// Close frees any allocated Metal buffers or ICB handles.
func (d *ICBDispatcher) Close() error {
	d.mu.Lock()
	defer d.mu.Unlock()
	if d.closed {
		return nil
	}
	d.closed = true
	if d.nativeHandle.available == 1 {
		C.mg_icb_release(&d.nativeHandle)
	}
	return nil
}
