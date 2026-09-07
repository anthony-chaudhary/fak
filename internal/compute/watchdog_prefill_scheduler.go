// Package compute implements hardware abstraction, tensor computation, memory slab management,
// and zero-copy device interconnect acceleration for the fak agent kernel.
package compute

import (
	"context"
	"errors"
	"fmt"
	"math"
	"strings"
	"time"
)

// watchdog_prefill_scheduler.go — Bytes-aware and FLOP-aware command buffer submission
// scheduler for deep-context prefill (>128k–262k tokens) on AMD APUs (#11904, #11572).
//
// Linux AMDGPU Ring Watchdog Background:
// The Linux kernel amdgpu driver (amdgpu_job.c) enforces a hardware ring timer
// (amdgpu.lockup_timeout), defaulting to 10,000 ms (10 seconds). If a single compute
// queue command buffer submission executes continuously without yielding past a
// synchronization fence for >10 seconds, the kernel triggers:
//   "[drm:amdgpu_job_timedout] *ERROR* ring gfx timeout"
// and forcibly resets the GPU (amdgpu_gpu_recover), dropping all in-flight state and
// crashing the process with VK_ERROR_DEVICE_LOST or HSA_STATUS_ERROR (GOTCHA_RING_TIMEOUT, #11241).
//
// On AMD APUs (such as Strix Halo gfx1151, Strix Point gfx1150, and Phoenix gfx1103),
// monolithic prefill of deep prompts (>128k to 262k tokens) takes tens of seconds.
// Furthermore, command buffer packing based purely on FLOPs fails because zero-FLOP memory
// operations (such as F16 KV contiguization, rotary transposition, and activation streaming)
// take hundreds of milliseconds on the UMA memory bus at 136k+ tokens.
//
// This file implements WatchdogPrefillScheduler:
// 1. Calibrated chunking profiles for single-sequence deep context (default 1024 tokens)
//    and batched multi-sequence (default 512 tokens).
// 2. Strict execution ceiling per chunk (default 2000 ms, leaving an 8000 ms safety margin
//    under the 10000 ms kernel watchdog threshold).
// 3. Exact roofline FLOP and byte memory-traffic calculations based on model geometry.
// 4. Adaptive chunk down-scaling for deep context positions where attention quadratic cost
//    would otherwise violate the 2000 ms execution ceiling.
// 5. Interleaved synchronization fences (YieldPoints) to allow driver ISR servicing.
// 6. Progress reporting, context cancellation, and execution simulation hooks.

const (
	// DefaultAMDGPUWatchdogTimeoutMs is the Linux kernel amdgpu driver default lockup timeout (10 seconds).
	DefaultAMDGPUWatchdogTimeoutMs float64 = 10000.0

	// DefaultMaxExecutionCeilingMs is the calibrated safe execution ceiling per chunk (2 seconds),
	// providing an 8-second safety margin below the 10-second driver timeout.
	DefaultMaxExecutionCeilingMs float64 = 2000.0

	// DefaultSingleSequenceChunkTokens is the standard chunk size for single-sequence deep prefill.
	DefaultSingleSequenceChunkTokens int = 1024

	// DefaultBatchedMultiSequenceChunkTokens is the standard chunk size for batched multi-sequence prefill.
	DefaultBatchedMultiSequenceChunkTokens int = 512

	// AbsoluteMinChunkTokens is the lower bound floor for adaptive chunk downsizing.
	AbsoluteMinChunkTokens int = 64

	// AbsoluteMaxChunkTokens is the upper bound ceiling for prefill chunk size.
	AbsoluteMaxChunkTokens int = 4096

	// DefaultFlashAttentionQueryTileSize is the default query tile dimension in Flash Attention.
	DefaultFlashAttentionQueryTileSize int = 128
)

// Standard sentinel errors.
var (
	// ErrWatchdogSafetyViolation indicates that a chunk exceeds the watchdog execution threshold.
	ErrWatchdogSafetyViolation = errors.New("watchdog: chunk estimated execution time exceeds safe ceiling")

	// ErrInvalidPromptTokens indicates a non-positive prompt token count.
	ErrInvalidPromptTokens = errors.New("watchdog: prompt token count must be greater than zero")

	// ErrInvalidChunkingProfile indicates malformed parameters in ChunkingProfile.
	ErrInvalidChunkingProfile = errors.New("watchdog: invalid chunking profile configuration")

	// ErrInvalidModelGeometry indicates malformed parameters in ModelGeometry.
	ErrInvalidModelGeometry = errors.New("watchdog: invalid model geometry configuration")

	// ErrInvalidHardwareProfile indicates malformed parameters in APUHardwareProfile.
	ErrInvalidHardwareProfile = errors.New("watchdog: invalid APU hardware profile configuration")

	// ErrEmptySchedule indicates an attempt to execute an empty prefill schedule.
	ErrEmptySchedule = errors.New("watchdog: prefill schedule contains no chunks")

	// ErrExecutionCancelled indicates that prefill was aborted due to context cancellation.
	ErrExecutionCancelled = errors.New("watchdog: prefill execution cancelled")
)

// WatchdogSafetyViolationError provides detailed context when a chunk exceeds the safety ceiling.
type WatchdogSafetyViolationError struct {
	ChunkIndex       int
	StartToken       int
	TokenCount       int
	EstimatedTimeMs  float64
	CeilingTimeMs    float64
	WatchdogLimitMs  float64
	IsMemoryBound    bool
	DetailMessage    string
}

func (e *WatchdogSafetyViolationError) Error() string {
	return fmt.Sprintf("watchdog safety violation: chunk %d [tokens %d..%d] estimated duration %.2f ms exceeds ceiling %.2f ms (watchdog limit: %.2f ms, mem_bound=%v): %s",
		e.ChunkIndex, e.StartToken, e.StartToken+e.TokenCount, e.EstimatedTimeMs, e.CeilingTimeMs, e.WatchdogLimitMs, e.IsMemoryBound, e.DetailMessage)
}

func (e *WatchdogSafetyViolationError) Unwrap() error {
	return ErrWatchdogSafetyViolation
}

// ChunkingProfileType identifies the prefill scheduling workload profile.
type ChunkingProfileType string

const (
	// ProfileSingleSequenceDeepContext is optimized for deep-context single sequences (>128k–262k tokens).
	ProfileSingleSequenceDeepContext ChunkingProfileType = "single_sequence_deep_context"

	// ProfileBatchedMultiSequence is optimized for concurrent or batched multi-sequence prefill.
	ProfileBatchedMultiSequence ChunkingProfileType = "batched_multi_sequence"

	// ProfileCustom allows user-defined chunk limits and timing thresholds.
	ProfileCustom ChunkingProfileType = "custom"
)

// ChunkingProfile defines chunk size bounds, time budgets, and watchdog thresholds.
type ChunkingProfile struct {
	Type                  ChunkingProfileType `json:"type"`
	DefaultChunkTokens    int                 `json:"default_chunk_tokens"`
	MaxChunkTokens        int                 `json:"max_chunk_tokens"`
	MinChunkTokens        int                 `json:"min_chunk_tokens"`
	MaxExecutionCeilingMs float64             `json:"max_execution_ceiling_ms"`
	WatchdogTimeoutMs     float64             `json:"watchdog_timeout_ms"`
	SafetyMarginRatio     float64             `json:"safety_margin_ratio"`
	BatchSize             int                 `json:"batch_size"`
	EnableAdaptivePacing  bool                `json:"enable_adaptive_pacing"`
}

// Validate checks that the chunking profile configuration is logically sound.
func (p ChunkingProfile) Validate() error {
	if p.DefaultChunkTokens <= 0 {
		return fmt.Errorf("%w: default chunk tokens (%d) must be positive", ErrInvalidChunkingProfile, p.DefaultChunkTokens)
	}
	if p.MinChunkTokens <= 0 {
		return fmt.Errorf("%w: min chunk tokens (%d) must be positive", ErrInvalidChunkingProfile, p.MinChunkTokens)
	}
	if p.MaxChunkTokens < p.DefaultChunkTokens {
		return fmt.Errorf("%w: max chunk tokens (%d) cannot be less than default (%d)", ErrInvalidChunkingProfile, p.MaxChunkTokens, p.DefaultChunkTokens)
	}
	if p.MinChunkTokens > p.DefaultChunkTokens {
		return fmt.Errorf("%w: min chunk tokens (%d) cannot be greater than default (%d)", ErrInvalidChunkingProfile, p.MinChunkTokens, p.DefaultChunkTokens)
	}
	if p.MaxExecutionCeilingMs <= 0 {
		return fmt.Errorf("%w: max execution ceiling ms (%.2f) must be positive", ErrInvalidChunkingProfile, p.MaxExecutionCeilingMs)
	}
	if p.WatchdogTimeoutMs <= 0 {
		return fmt.Errorf("%w: watchdog timeout ms (%.2f) must be positive", ErrInvalidChunkingProfile, p.WatchdogTimeoutMs)
	}
	if p.MaxExecutionCeilingMs > p.WatchdogTimeoutMs {
		return fmt.Errorf("%w: max execution ceiling ms (%.2f) exceeds watchdog timeout (%.2f)", ErrInvalidChunkingProfile, p.MaxExecutionCeilingMs, p.WatchdogTimeoutMs)
	}
	if p.BatchSize <= 0 {
		return fmt.Errorf("%w: batch size (%d) must be positive", ErrInvalidChunkingProfile, p.BatchSize)
	}
	return nil
}

// DefaultSingleSequenceProfile returns the standard chunking profile for single-sequence deep context.
func DefaultSingleSequenceProfile() ChunkingProfile {
	return ChunkingProfile{
		Type:                  ProfileSingleSequenceDeepContext,
		DefaultChunkTokens:    DefaultSingleSequenceChunkTokens,
		MaxChunkTokens:        2048,
		MinChunkTokens:        AbsoluteMinChunkTokens,
		MaxExecutionCeilingMs: DefaultMaxExecutionCeilingMs,
		WatchdogTimeoutMs:     DefaultAMDGPUWatchdogTimeoutMs,
		SafetyMarginRatio:     0.20,
		BatchSize:             1,
		EnableAdaptivePacing:  true,
	}
}

// DefaultBatchedMultiSequenceProfile returns the standard chunking profile for batched prefill.
func DefaultBatchedMultiSequenceProfile(batchSize int) ChunkingProfile {
	if batchSize < 1 {
		batchSize = 1
	}
	return ChunkingProfile{
		Type:                  ProfileBatchedMultiSequence,
		DefaultChunkTokens:    DefaultBatchedMultiSequenceChunkTokens,
		MaxChunkTokens:        1024,
		MinChunkTokens:        AbsoluteMinChunkTokens,
		MaxExecutionCeilingMs: DefaultMaxExecutionCeilingMs,
		WatchdogTimeoutMs:     DefaultAMDGPUWatchdogTimeoutMs,
		SafetyMarginRatio:     0.20,
		BatchSize:             batchSize,
		EnableAdaptivePacing:  true,
	}
}

// NewCustomChunkingProfile constructs and validates a customized chunking profile.
func NewCustomChunkingProfile(
	defaultChunk, maxChunk, minChunk int,
	ceilingMs, timeoutMs float64,
	batchSize int,
	adaptive bool,
) (ChunkingProfile, error) {
	if batchSize < 1 {
		batchSize = 1
	}
	if timeoutMs <= 0 {
		timeoutMs = DefaultAMDGPUWatchdogTimeoutMs
	}
	if ceilingMs <= 0 {
		ceilingMs = DefaultMaxExecutionCeilingMs
	}
	margin := ceilingMs / timeoutMs

	p := ChunkingProfile{
		Type:                  ProfileCustom,
		DefaultChunkTokens:    defaultChunk,
		MaxChunkTokens:        maxChunk,
		MinChunkTokens:        minChunk,
		MaxExecutionCeilingMs: ceilingMs,
		WatchdogTimeoutMs:     timeoutMs,
		SafetyMarginRatio:     margin,
		BatchSize:             batchSize,
		EnableAdaptivePacing:  adaptive,
	}
	if err := p.Validate(); err != nil {
		return ChunkingProfile{}, err
	}
	return p, nil
}

// ModelGeometry encapsulates the structural dimensions of a transformer model
// required for computing FLOPs and memory traffic during prefill pacing.
type ModelGeometry struct {
	Name                   string  `json:"name"`
	Layers                 int     `json:"layers"`                   // Number of transformer layers (e.g. 34 for 27B)
	HiddenDim              int     `json:"hidden_dim"`               // Hidden dimension (e.g. 5120)
	NumHeads               int     `json:"num_heads"`                // Query heads (e.g. 40)
	NumKVHeads             int     `json:"num_kv_heads"`             // Key/Value heads for GQA (e.g. 8)
	HeadDim                int     `json:"head_dim"`                 // Head dimension (e.g. 128)
	IntermediateDim        int     `json:"intermediate_dim"`         // MLP intermediate dimension (e.g. 14336)
	VocabSize              int     `json:"vocab_size"`               // Vocabulary size (e.g. 151936)
	WeightBytesPerParam    float64 `json:"weight_bytes_per_param"`   // e.g. 0.5625 for Q4_K, 2.0 for FP16, 1.0 for Q8
	KVBytesPerElement      int     `json:"kv_bytes_per_element"`     // e.g. 2 for FP16, 1 for Q8
	ActivationBytesElement int     `json:"activation_bytes_element"` // e.g. 2 for FP16/BF16
	ContiguizationPass     bool    `json:"contiguization_pass"`      // Whether F16 KV contiguization pass is active
	QueryTileSize          int     `json:"query_tile_size"`          // Flash Attention query tile size (e.g. 128)
}

// Validate verifies that the model geometry dimensions are valid.
func (g ModelGeometry) Validate() error {
	if g.Layers <= 0 {
		return fmt.Errorf("%w: layers (%d) must be positive", ErrInvalidModelGeometry, g.Layers)
	}
	if g.HiddenDim <= 0 {
		return fmt.Errorf("%w: hidden dim (%d) must be positive", ErrInvalidModelGeometry, g.HiddenDim)
	}
	if g.NumHeads <= 0 {
		return fmt.Errorf("%w: num heads (%d) must be positive", ErrInvalidModelGeometry, g.NumHeads)
	}
	if g.NumKVHeads <= 0 {
		return fmt.Errorf("%w: num kv heads (%d) must be positive", ErrInvalidModelGeometry, g.NumKVHeads)
	}
	if g.HeadDim <= 0 {
		return fmt.Errorf("%w: head dim (%d) must be positive", ErrInvalidModelGeometry, g.HeadDim)
	}
	if g.IntermediateDim <= 0 {
		return fmt.Errorf("%w: intermediate dim (%d) must be positive", ErrInvalidModelGeometry, g.IntermediateDim)
	}
	if g.WeightBytesPerParam <= 0 {
		return fmt.Errorf("%w: weight bytes per param (%.4f) must be positive", ErrInvalidModelGeometry, g.WeightBytesPerParam)
	}
	if g.KVBytesPerElement <= 0 {
		return fmt.Errorf("%w: kv bytes per element (%d) must be positive", ErrInvalidModelGeometry, g.KVBytesPerElement)
	}
	if g.ActivationBytesElement <= 0 {
		return fmt.Errorf("%w: activation bytes element (%d) must be positive", ErrInvalidModelGeometry, g.ActivationBytesElement)
	}
	if g.QueryTileSize <= 0 {
		return fmt.Errorf("%w: query tile size (%d) must be positive", ErrInvalidModelGeometry, g.QueryTileSize)
	}
	return nil
}

// TotalActiveParameters computes the total weight parameter count for the model geometry.
func (g ModelGeometry) TotalActiveParameters() int64 {
	// Q, K, V, Out linear projections per layer
	qParams := int64(g.HiddenDim) * int64(g.NumHeads*g.HeadDim)
	kParams := int64(g.HiddenDim) * int64(g.NumKVHeads*g.HeadDim)
	vParams := int64(g.HiddenDim) * int64(g.NumKVHeads*g.HeadDim)
	oParams := int64(g.NumHeads*g.HeadDim) * int64(g.HiddenDim)
	attnParams := qParams + kParams + vParams + oParams

	// MLP SwiGLU: Gate, Up, Down projections per layer
	ffnParams := 3 * int64(g.HiddenDim) * int64(g.IntermediateDim)

	layerParams := attnParams + ffnParams
	totalTransformerParams := int64(g.Layers) * layerParams

	// Embeddings and final LM head
	var vocabParams int64
	if g.VocabSize > 0 {
		vocabParams = 2 * int64(g.VocabSize) * int64(g.HiddenDim)
	}
	return totalTransformerParams + vocabParams
}

// WeightSizeBytes returns the estimated total resident weight footprint in bytes.
func (g ModelGeometry) WeightSizeBytes() int64 {
	params := g.TotalActiveParameters()
	return int64(math.Ceil(float64(params) * g.WeightBytesPerParam))
}

// KVBytesPerToken returns the memory footprint in bytes required to store 1 token of KV cache across all layers.
func (g ModelGeometry) KVBytesPerToken() int64 {
	// 2 for K and V
	return int64(2 * g.Layers * g.NumKVHeads * g.HeadDim * g.KVBytesPerElement)
}

// ModelGeometry27B returns the authoritative 27B model geometry (layers: 34, headDim: 128, hiddenDim: 5120).
func ModelGeometry27B() ModelGeometry {
	return ModelGeometry{
		Name:                   "27B-Dense",
		Layers:                 34,
		HiddenDim:              5120,
		NumHeads:               40,
		NumKVHeads:             8,
		HeadDim:                128,
		IntermediateDim:        14336,
		VocabSize:              151936,
		WeightBytesPerParam:    0.5625, // Q4_K (~4.5 bits/weight)
		KVBytesPerElement:      2,      // FP16
		ActivationBytesElement: 2,      // FP16
		ContiguizationPass:     true,
		QueryTileSize:          DefaultFlashAttentionQueryTileSize,
	}
}

// ModelGeometry7B returns the geometry for a standard 7B-8B class model.
func ModelGeometry7B() ModelGeometry {
	return ModelGeometry{
		Name:                   "7B-Class",
		Layers:                 32,
		HiddenDim:              4096,
		NumHeads:               32,
		NumKVHeads:             8,
		HeadDim:                128,
		IntermediateDim:        11008,
		VocabSize:              32000,
		WeightBytesPerParam:    0.5625,
		KVBytesPerElement:      2,
		ActivationBytesElement: 2,
		ContiguizationPass:     true,
		QueryTileSize:          DefaultFlashAttentionQueryTileSize,
	}
}

// ModelGeometry14B returns the geometry for a standard 14B class model.
func ModelGeometry14B() ModelGeometry {
	return ModelGeometry{
		Name:                   "14B-Class",
		Layers:                 48,
		HiddenDim:              5120,
		NumHeads:               40,
		NumKVHeads:             8,
		HeadDim:                128,
		IntermediateDim:        13824,
		VocabSize:              151936,
		WeightBytesPerParam:    0.5625,
		KVBytesPerElement:      2,
		ActivationBytesElement: 2,
		ContiguizationPass:     true,
		QueryTileSize:          DefaultFlashAttentionQueryTileSize,
	}
}

// ModelGeometry70B returns the geometry for a standard 70B class model.
func ModelGeometry70B() ModelGeometry {
	return ModelGeometry{
		Name:                   "70B-Class",
		Layers:                 80,
		HiddenDim:              8192,
		NumHeads:               64,
		NumKVHeads:             8,
		HeadDim:                128,
		IntermediateDim:        28672,
		VocabSize:              151936,
		WeightBytesPerParam:    0.5625,
		KVBytesPerElement:      2,
		ActivationBytesElement: 2,
		ContiguizationPass:     true,
		QueryTileSize:          DefaultFlashAttentionQueryTileSize,
	}
}

// APUHardwareProfile details the compute and unified memory specifications of an AMD APU.
type APUHardwareProfile struct {
	Architecture           string  `json:"architecture"`
	Codename               string  `json:"codename"`
	ComputeUnits           int     `json:"compute_units"`
	PeakTFLOPS             float64 `json:"peak_tflops"`               // Peak FP16 compute TFLOPS
	SustainedComputeEff    float64 `json:"sustained_compute_eff"`     // Achievable GEMM efficiency (0.0 to 1.0)
	TheoreticalPeakBWGBps  float64 `json:"theoretical_peak_bw_gbps"`  // Memory bus peak bandwidth in GB/s
	SustainedMemoryEff     float64 `json:"sustained_memory_eff"`      // Sustained UMA memory bandwidth efficiency (0.0 to 1.0)
	KernelLaunchOverheadMs float64 `json:"kernel_launch_overhead_ms"` // Base submission and fence overhead in ms
}

// Validate verifies that the APU hardware profile parameters are valid.
func (h APUHardwareProfile) Validate() error {
	if strings.TrimSpace(h.Architecture) == "" {
		return fmt.Errorf("%w: architecture must not be empty", ErrInvalidHardwareProfile)
	}
	if h.ComputeUnits <= 0 {
		return fmt.Errorf("%w: compute units (%d) must be positive", ErrInvalidHardwareProfile, h.ComputeUnits)
	}
	if h.PeakTFLOPS <= 0 {
		return fmt.Errorf("%w: peak TFLOPS (%.2f) must be positive", ErrInvalidHardwareProfile, h.PeakTFLOPS)
	}
	if h.SustainedComputeEff <= 0 || h.SustainedComputeEff > 1.0 {
		return fmt.Errorf("%w: sustained compute eff (%.2f) must be in (0, 1]", ErrInvalidHardwareProfile, h.SustainedComputeEff)
	}
	if h.TheoreticalPeakBWGBps <= 0 {
		return fmt.Errorf("%w: theoretical peak BW (%.2f GB/s) must be positive", ErrInvalidHardwareProfile, h.TheoreticalPeakBWGBps)
	}
	if h.SustainedMemoryEff <= 0 || h.SustainedMemoryEff > 1.0 {
		return fmt.Errorf("%w: sustained memory eff (%.2f) must be in (0, 1]", ErrInvalidHardwareProfile, h.SustainedMemoryEff)
	}
	if h.KernelLaunchOverheadMs < 0 {
		return fmt.Errorf("%w: kernel launch overhead ms (%.2f) cannot be negative", ErrInvalidHardwareProfile, h.KernelLaunchOverheadMs)
	}
	return nil
}

// SustainedFLOPS returns the sustained floating-point throughput in FLOPS (FLOP/sec).
func (h APUHardwareProfile) SustainedFLOPS() float64 {
	return h.PeakTFLOPS * 1e12 * h.SustainedComputeEff
}

// SustainedBandwidthBytesPerSec returns the sustained memory bandwidth in bytes/second.
func (h APUHardwareProfile) SustainedBandwidthBytesPerSec() float64 {
	return h.TheoreticalPeakBWGBps * 1e9 * h.SustainedMemoryEff
}

// StrixHaloHardwareProfile returns the hardware profile for AMD Strix Halo (Ryzen AI MAX+ 395 / Radeon 8060S / gfx1151).
func StrixHaloHardwareProfile() APUHardwareProfile {
	return APUHardwareProfile{
		Architecture:           "gfx1151",
		Codename:               "Strix Halo",
		ComputeUnits:           40,
		PeakTFLOPS:             60.0,
		SustainedComputeEff:    0.65,
		TheoreticalPeakBWGBps:  273.056,
		SustainedMemoryEff:     0.78,
		KernelLaunchOverheadMs: 0.05,
	}
}

// StrixPointHardwareProfile returns the hardware profile for AMD Strix Point (Ryzen AI 9 HX 370 / Radeon 890M / gfx1150).
func StrixPointHardwareProfile() APUHardwareProfile {
	return APUHardwareProfile{
		Architecture:           "gfx1150",
		Codename:               "Strix Point",
		ComputeUnits:           16,
		PeakTFLOPS:             24.0,
		SustainedComputeEff:    0.60,
		TheoreticalPeakBWGBps:  120.0,
		SustainedMemoryEff:     0.75,
		KernelLaunchOverheadMs: 0.05,
	}
}

// PhoenixHardwareProfile returns the hardware profile for AMD Phoenix / Hawk Point (Radeon 780M / gfx1103).
func PhoenixHardwareProfile() APUHardwareProfile {
	return APUHardwareProfile{
		Architecture:           "gfx1103",
		Codename:               "Phoenix",
		ComputeUnits:           12,
		PeakTFLOPS:             18.0,
		SustainedComputeEff:    0.60,
		TheoreticalPeakBWGBps:  102.4,
		SustainedMemoryEff:     0.72,
		KernelLaunchOverheadMs: 0.05,
	}
}

// DefaultAPUHardwareProfile returns the flagship Strix Halo profile.
func DefaultAPUHardwareProfile() APUHardwareProfile {
	return StrixHaloHardwareProfile()
}

// BarrierFenceType defines the synchronization and interrupt-yielding primitive between chunks.
type BarrierFenceType string

const (
	// FenceTypeNone indicates no fence insertion.
	FenceTypeNone BarrierFenceType = "NONE"

	// FenceTypeExecutionBarrier indicates a GPU command buffer pipeline barrier.
	FenceTypeExecutionBarrier BarrierFenceType = "EXECUTION_BARRIER"

	// FenceTypeSignalFence indicates an explicit completion signal fence submitted to the ring.
	FenceTypeSignalFence BarrierFenceType = "SIGNAL_FENCE"

	// FenceTypeHostInterrupt indicates a host interrupt fence unblocking the driver ISR to reset the watchdog.
	FenceTypeHostInterrupt BarrierFenceType = "HOST_INTERRUPT"

	// FenceTypeFullYield indicates complete ring drain, signal fence, and host CPU cooperative yield.
	FenceTypeFullYield BarrierFenceType = "FULL_YIELD"
)

// YieldPoint describes the hardware fence and synchronization barrier at a chunk boundary.
type YieldPoint struct {
	Type              BarrierFenceType `json:"type"`
	FenceID           uint64           `json:"fence_id"`
	FlushL2Cache      bool             `json:"flush_l2_cache"`
	DrainPipeline     bool             `json:"drain_pipeline"`
	HostInterruptWait bool             `json:"host_interrupt_wait"`
	YieldHostCPU      bool             `json:"yield_host_cpu"`
}

// PrefillChunk describes a single paced prefill command buffer submission unit.
type PrefillChunk struct {
	Index               int        `json:"index"`
	StartToken          int        `json:"start_token"`
	TokenCount          int        `json:"token_count"`
	EstimatedFLOPs      uint64     `json:"estimated_flops"`
	EstimatedBytesRead  uint64     `json:"estimated_bytes_read"`
	EstimatedBytesWrite uint64     `json:"estimated_bytes_write"`
	EstimatedTotalBytes uint64     `json:"estimated_total_bytes"`
	EstimatedDurationMs float64    `json:"estimated_duration_ms"`
	ComputeTimeMs       float64    `json:"compute_time_ms"`
	MemoryTimeMs        float64    `json:"memory_time_ms"`
	IsMemoryBound       bool       `json:"is_memory_bound"`
	Yield               YieldPoint `json:"yield"`
	IsLastChunk         bool       `json:"is_last_chunk"`
	BatchID             int        `json:"batch_id"`
}

// PrefillSchedule contains the complete ordered sequence of prefill chunks for a prompt or batch.
type PrefillSchedule struct {
	TotalTokens         int                `json:"total_tokens"`
	BatchSize           int                `json:"batch_size"`
	Profile             ChunkingProfile    `json:"profile"`
	Geometry            ModelGeometry      `json:"geometry"`
	Hardware            APUHardwareProfile `json:"hardware"`
	Chunks              []PrefillChunk     `json:"chunks"`
	TotalEstimatedFLOPs uint64             `json:"total_estimated_flops"`
	TotalEstimatedBytes uint64             `json:"total_estimated_bytes"`
	TotalEstimatedMs    float64            `json:"total_estimated_ms"`
	MaxChunkDurationMs  float64            `json:"max_chunk_duration_ms"`
	IsWatchdogSafe      bool               `json:"is_watchdog_safe"`
}

// ChunkCount returns the number of submission chunks in the schedule.
func (s *PrefillSchedule) ChunkCount() int {
	if s == nil {
		return 0
	}
	return len(s.Chunks)
}

// Validate verifies that the schedule forms a contiguous, non-overlapping partition
// of the prompt tokens and that no chunk violates the watchdog or ceiling thresholds.
func (s *PrefillSchedule) Validate(maxTimeCeilingMs float64) error {
	if s == nil {
		return ErrEmptySchedule
	}
	if s.TotalTokens <= 0 {
		return ErrInvalidPromptTokens
	}
	if len(s.Chunks) == 0 {
		return ErrEmptySchedule
	}
	if maxTimeCeilingMs <= 0 {
		maxTimeCeilingMs = s.Profile.MaxExecutionCeilingMs
	}
	if maxTimeCeilingMs <= 0 {
		maxTimeCeilingMs = DefaultMaxExecutionCeilingMs
	}

	expectedStart := 0
	for i, chunk := range s.Chunks {
		if chunk.Index != i {
			return fmt.Errorf("chunk index mismatch: chunk %d has index field %d", i, chunk.Index)
		}
		if chunk.StartToken != expectedStart {
			return fmt.Errorf("chunk %d gap/overlap: start %d, expected %d", i, chunk.StartToken, expectedStart)
		}
		if chunk.TokenCount <= 0 {
			return fmt.Errorf("chunk %d has non-positive token count %d", i, chunk.TokenCount)
		}
		if chunk.EstimatedDurationMs > maxTimeCeilingMs {
			return &WatchdogSafetyViolationError{
				ChunkIndex:      i,
				StartToken:      chunk.StartToken,
				TokenCount:      chunk.TokenCount,
				EstimatedTimeMs: chunk.EstimatedDurationMs,
				CeilingTimeMs:   maxTimeCeilingMs,
				WatchdogLimitMs: s.Profile.WatchdogTimeoutMs,
				IsMemoryBound:   chunk.IsMemoryBound,
				DetailMessage:   "chunk estimated duration exceeds safe execution ceiling",
			}
		}
		if chunk.EstimatedDurationMs > s.Profile.WatchdogTimeoutMs {
			return &WatchdogSafetyViolationError{
				ChunkIndex:      i,
				StartToken:      chunk.StartToken,
				TokenCount:      chunk.TokenCount,
				EstimatedTimeMs: chunk.EstimatedDurationMs,
				CeilingTimeMs:   maxTimeCeilingMs,
				WatchdogLimitMs: s.Profile.WatchdogTimeoutMs,
				IsMemoryBound:   chunk.IsMemoryBound,
				DetailMessage:   "CRITICAL: chunk estimated duration exceeds Linux kernel amdgpu watchdog timeout",
			}
		}

		isLast := (i == len(s.Chunks)-1)
		if chunk.IsLastChunk != isLast {
			return fmt.Errorf("chunk %d IsLastChunk mismatch: got %v, want %v", i, chunk.IsLastChunk, isLast)
		}

		expectedStart += chunk.TokenCount
	}

	if expectedStart != s.TotalTokens {
		return fmt.Errorf("schedule token count mismatch: partitioned %d tokens, expected %d", expectedStart, s.TotalTokens)
	}

	return nil
}

// Summary returns a human-readable summary of the prefill schedule.
func (s *PrefillSchedule) Summary() string {
	if s == nil {
		return "PrefillSchedule: <nil>"
	}
	return fmt.Sprintf("PrefillSchedule: %d tokens across %d chunks (max_chunk_ms=%.1f, total_ms=%.1f, total_GFLOPs=%.1f, total_MB=%.1f, safe=%v)",
		s.TotalTokens, len(s.Chunks), s.MaxChunkDurationMs, s.TotalEstimatedMs,
		float64(s.TotalEstimatedFLOPs)/1e9, float64(s.TotalEstimatedBytes)/(1024*1024), s.IsWatchdogSafe)
}

// ProgressReport conveys the runtime state after a chunk execution or yield.
type ProgressReport struct {
	ChunkIndex         int           `json:"chunk_index"`
	TotalChunks        int           `json:"total_chunks"`
	CompletedTokens    int           `json:"completed_tokens"`
	TotalTokens        int           `json:"total_tokens"`
	ElapsedDuration    time.Duration `json:"elapsed_duration"`
	EstimatedRemaining time.Duration `json:"estimated_remaining"`
	PercentComplete    float64       `json:"percent_complete"`
	CurrentChunk       PrefillChunk  `json:"current_chunk"`
}

// InterChunkHook is invoked between chunk submissions to allow driver ISR interleaving,
// host yielding, or watchdog timer reset verification.
type InterChunkHook func(ctx context.Context, chunk PrefillChunk, report ProgressReport) error

// ChunkExecutor executes a single prefill chunk command buffer on hardware or simulator.
type ChunkExecutor func(ctx context.Context, chunk PrefillChunk) error

// ExecutionReceipt records the actual execution telemetry of a prefill schedule.
type ExecutionReceipt struct {
	TotalTokens    int           `json:"total_tokens"`
	ChunksExecuted int           `json:"chunks_executed"`
	TotalElapsed   time.Duration `json:"total_elapsed"`
	CompletedClean bool          `json:"completed_clean"`
	YieldPointsHit int           `json:"yield_points_hit"`
	Cancelled      bool          `json:"cancelled"`
	Error          error         `json:"error,omitempty"`
}

// WatchdogPrefillScheduler generates and executes paced prefill command buffer schedules
// calibrated to prevent Linux kernel AMDGPU 10-second ring watchdog resets.
type WatchdogPrefillScheduler struct {
	profile  ChunkingProfile
	geometry ModelGeometry
	hardware APUHardwareProfile
}

// NewWatchdogPrefillScheduler constructs a scheduler with the specified profile, model geometry, and hardware.
func NewWatchdogPrefillScheduler(
	profile ChunkingProfile,
	geometry ModelGeometry,
	hardware APUHardwareProfile,
) (*WatchdogPrefillScheduler, error) {
	if err := profile.Validate(); err != nil {
		return nil, err
	}
	if err := geometry.Validate(); err != nil {
		return nil, err
	}
	if err := hardware.Validate(); err != nil {
		return nil, err
	}
	return &WatchdogPrefillScheduler{
		profile:  profile,
		geometry: geometry,
		hardware: hardware,
	}, nil
}

// NewDefaultAPUPrefillScheduler constructs a scheduler using the default Strix Halo profile
// and single-sequence deep context profile for the specified model geometry.
func NewDefaultAPUPrefillScheduler(geometry ModelGeometry) (*WatchdogPrefillScheduler, error) {
	profile := DefaultSingleSequenceProfile()
	hardware := DefaultAPUHardwareProfile()
	return NewWatchdogPrefillScheduler(profile, geometry, hardware)
}

// NewDeepContextPrefillScheduler constructs a scheduler for 27B model geometry on Strix Halo APU.
func NewDeepContextPrefillScheduler(profileType ChunkingProfileType) (*WatchdogPrefillScheduler, error) {
	var profile ChunkingProfile
	if profileType == ProfileBatchedMultiSequence {
		profile = DefaultBatchedMultiSequenceProfile(2)
	} else {
		profile = DefaultSingleSequenceProfile()
	}
	geometry := ModelGeometry27B()
	hardware := DefaultAPUHardwareProfile()
	return NewWatchdogPrefillScheduler(profile, geometry, hardware)
}

// EstimateChunkPacing calculates the exact FLOPs, memory bytes read/written, and estimated
// execution duration for a chunk spanning [startToken, startToken+tokenCount) under the model geometry.
func (s *WatchdogPrefillScheduler) EstimateChunkPacing(
	startToken, tokenCount, batchSize int,
) (flops uint64, bytesRead uint64, bytesWrite uint64, durationMs float64, computeMs float64, memoryMs float64, isMemBound bool) {
	if tokenCount <= 0 {
		return 0, 0, 0, 0, 0, 0, false
	}
	if batchSize < 1 {
		batchSize = 1
	}

	g := s.geometry
	h := s.hardware

	// 1. Compute FLOPs calculation:
	// A. Weight matrix projections (Linear layers):
	// Q, K, V, O projections + MLP (Gate, Up, Down):
	// 2 FLOPs per parameter per token.
	transformerParams := int64(g.Layers) * (int64(g.HiddenDim)*int64(g.NumHeads*g.HeadDim) +
		2*int64(g.HiddenDim)*int64(g.NumKVHeads*g.HeadDim) +
		int64(g.NumHeads*g.HeadDim)*int64(g.HiddenDim) +
		3*int64(g.HiddenDim)*int64(g.IntermediateDim))

	linearFLOPsPerToken := 2 * transformerParams
	// SwiGLU activation elementwise FLOPs (~2 per intermediate element)
	swigluFLOPsPerToken := int64(g.Layers) * 2 * int64(g.IntermediateDim)
	totalLinearFLOPs := (linearFLOPsPerToken + swigluFLOPsPerToken) * int64(tokenCount) * int64(batchSize)

	// B. Attention FLOPs (Causal Self-Attention):
	// For each token at position pos = startToken + i (for i = 0 .. tokenCount-1):
	// The token attends to (pos + 1) tokens in the KV cache.
	// Average attended length = startToken + (tokenCount + 1)/2.
	// Total attended token-pairs = tokenCount * (startToken + (tokenCount+1)/2).
	avgAttended := float64(startToken) + float64(tokenCount+1)/2.0
	attendedPairs := float64(tokenCount) * avgAttended

	// QK dot product: 2 * NumHeads * HeadDim per pair per layer
	// Softmax: 3 * NumHeads per pair per layer
	// AV dot product: 2 * NumHeads * HeadDim per pair per layer
	attnFLOPsPerPairPerLayer := float64(4*g.NumHeads*g.HeadDim + 3*g.NumHeads)
	totalAttnFLOPs := float64(g.Layers) * attnFLOPsPerPairPerLayer * attendedPairs * float64(batchSize)

	totalFLOPs := float64(totalLinearFLOPs) + totalAttnFLOPs
	flops = uint64(math.Round(totalFLOPs))

	// 2. Memory Traffic (Bytes) calculation:
	// A. Weights:
	// In each chunk submission, weights for all layers are streamed through the processor.
	// Weight reading is amortized across batch sequences if batched.
	weightBytes := float64(transformerParams) * g.WeightBytesPerParam

	// B. KV Cache Write:
	// For each new token in the chunk, K and V are written to UMA memory across all layers.
	kvWriteBytes := float64(2*g.Layers*tokenCount*g.NumKVHeads*g.HeadDim*g.KVBytesPerElement) * float64(batchSize)

	// C. KV Cache Read:
	// In Flash Attention / blocked attention, query tokens of size QueryTileSize stream
	// the KV cache of previous (startToken + tokenCount) tokens.
	queryTileSize := g.QueryTileSize
	if queryTileSize <= 0 {
		queryTileSize = DefaultFlashAttentionQueryTileSize
	}
	queryPasses := math.Ceil(float64(tokenCount) / float64(queryTileSize))
	if queryPasses < 1.0 {
		queryPasses = 1.0
	}

	// Size of resident KV cache up to current position per layer:
	residentKVTokens := float64(startToken) + float64(tokenCount)/2.0
	residentKVBytesPerLayer := residentKVTokens * float64(2*g.NumKVHeads*g.HeadDim*g.KVBytesPerElement)
	kvReadBytes := float64(g.Layers) * residentKVBytesPerLayer * queryPasses * float64(batchSize)

	// D. Activation read/write between layers:
	// Input activation and residual stream: 2 read + 2 write = 4 transfers per layer
	actBytesPerToken := float64(4 * g.Layers * g.HiddenDim * g.ActivationBytesElement)
	actBytes := actBytesPerToken * float64(tokenCount) * float64(batchSize)

	// E. Zero-FLOP operations (Pre-attention F16 KV contiguization pass):
	// On Strix Halo (gfx1151), when context exceeds ContiguizationMinContext (32768 tokens),
	// a contiguous scratch copy pass runs to eliminate 16-channel camping.
	// This performs read + write of the KV slice with 0 FLOPs.
	var contigBytes float64
	if g.ContiguizationPass && (startToken+tokenCount) >= ContiguizationMinContext {
		contigBytes = 2.0 * float64(g.Layers) * residentKVBytesPerLayer * float64(batchSize)
	}

	totalBytesRead := weightBytes + kvReadBytes + (actBytes / 2.0) + (contigBytes / 2.0)
	totalBytesWrite := kvWriteBytes + (actBytes / 2.0) + (contigBytes / 2.0)
	totalBytes := totalBytesRead + totalBytesWrite

	bytesRead = uint64(math.Round(totalBytesRead))
	bytesWrite = uint64(math.Round(totalBytesWrite))

	// 3. Roofline execution duration estimation (ms):
	sustainedFLOPS := h.SustainedFLOPS()
	sustainedBW := h.SustainedBandwidthBytesPerSec()

	compSec := totalFLOPs / sustainedFLOPS
	computeMs = compSec * 1000.0

	memSec := totalBytes / sustainedBW
	memoryMs = memSec * 1000.0

	// Roofline: execution is bounded by the slower of compute or memory, plus kernel launch overhead
	isMemBound = memoryMs > computeMs
	rawDurationMs := math.Max(computeMs, memoryMs)
	durationMs = rawDurationMs + h.KernelLaunchOverheadMs

	return flops, bytesRead, bytesWrite, durationMs, computeMs, memoryMs, isMemBound
}

// PlanSchedule generates an ordered sequence of paced PrefillChunk descriptors for a single sequence prompt.
func (s *WatchdogPrefillScheduler) PlanSchedule(promptTokens int) (*PrefillSchedule, error) {
	if promptTokens <= 0 {
		return nil, fmt.Errorf("%w: %d", ErrInvalidPromptTokens, promptTokens)
	}

	var chunks []PrefillChunk
	start := 0
	chunkIdx := 0
	var totalFLOPs uint64
	var totalBytes uint64
	var totalMs float64
	var maxChunkMs float64
	batchSize := s.profile.BatchSize
	if batchSize < 1 {
		batchSize = 1
	}

	ceilingMs := s.profile.MaxExecutionCeilingMs
	if ceilingMs <= 0 {
		ceilingMs = DefaultMaxExecutionCeilingMs
	}

	for start < promptTokens {
		remaining := promptTokens - start

		// Start with default chunk size
		candidateTokens := s.profile.DefaultChunkTokens
		if candidateTokens > remaining {
			candidateTokens = remaining
		}
		if candidateTokens > s.profile.MaxChunkTokens {
			candidateTokens = s.profile.MaxChunkTokens
		}

		// Adaptive pacing: if quadratic attention cost at deep context causes the chunk
		// to exceed MaxExecutionCeilingMs, iteratively downsize the chunk until safe.
		if s.profile.EnableAdaptivePacing {
			candidateTokens = s.adaptivelyScaleChunk(start, candidateTokens, remaining, batchSize, ceilingMs)
		}

		// Calculate exact pacing metrics for the selected chunk size
		flops, bRead, bWrite, durMs, compMs, memMs, isMemBound := s.EstimateChunkPacing(start, candidateTokens, batchSize)

		isLast := (start + candidateTokens >= promptTokens)

		// Determine synchronization yield point:
		// Regular chunks get a signal fence to reset the hardware watchdog.
		// The last chunk gets a full yield to return control to the decode engine.
		var yield YieldPoint
		if isLast {
			yield = YieldPoint{
				Type:              FenceTypeFullYield,
				FenceID:           uint64(chunkIdx + 1),
				FlushL2Cache:      true,
				DrainPipeline:     true,
				HostInterruptWait: true,
				YieldHostCPU:      true,
			}
		} else {
			yield = YieldPoint{
				Type:              FenceTypeSignalFence,
				FenceID:           uint64(chunkIdx + 1),
				FlushL2Cache:      (start + candidateTokens) >= ContiguizationMinContext,
				DrainPipeline:     false,
				HostInterruptWait: true,
				YieldHostCPU:      false,
			}
		}

		chunk := PrefillChunk{
			Index:               chunkIdx,
			StartToken:          start,
			TokenCount:          candidateTokens,
			EstimatedFLOPs:      flops,
			EstimatedBytesRead:  bRead,
			EstimatedBytesWrite: bWrite,
			EstimatedTotalBytes: bRead + bWrite,
			EstimatedDurationMs: durMs,
			ComputeTimeMs:       compMs,
			MemoryTimeMs:        memMs,
			IsMemoryBound:       isMemBound,
			Yield:               yield,
			IsLastChunk:         isLast,
			BatchID:             0,
		}

		chunks = append(chunks, chunk)
		totalFLOPs += flops
		totalBytes += (bRead + bWrite)
		totalMs += durMs
		if durMs > maxChunkMs {
			maxChunkMs = durMs
		}

		start += candidateTokens
		chunkIdx++
	}

	schedule := &PrefillSchedule{
		TotalTokens:         promptTokens,
		BatchSize:           batchSize,
		Profile:             s.profile,
		Geometry:            s.geometry,
		Hardware:            s.hardware,
		Chunks:              chunks,
		TotalEstimatedFLOPs: totalFLOPs,
		TotalEstimatedBytes: totalBytes,
		TotalEstimatedMs:    totalMs,
		MaxChunkDurationMs:  maxChunkMs,
		IsWatchdogSafe:      maxChunkMs <= ceilingMs,
	}

	// Validate schedule invariants
	if err := schedule.Validate(ceilingMs); err != nil {
		// Return the schedule alongside the typed validation error so callers can inspect
		return schedule, err
	}

	return schedule, nil
}

// PlanBatchedSchedule generates an interleaved schedule across multiple concurrent sequences.
func (s *WatchdogPrefillScheduler) PlanBatchedSchedule(promptTokens []int) (*PrefillSchedule, error) {
	if len(promptTokens) == 0 {
		return nil, ErrInvalidPromptTokens
	}

	totalTokens := 0
	for i, tokens := range promptTokens {
		if tokens <= 0 {
			return nil, fmt.Errorf("%w: sequence %d has invalid token count %d", ErrInvalidPromptTokens, i, tokens)
		}
		totalTokens += tokens
	}

	batchProfile := s.profile
	if batchProfile.Type != ProfileBatchedMultiSequence {
		batchProfile = DefaultBatchedMultiSequenceProfile(len(promptTokens))
	}
	batchProfile.BatchSize = len(promptTokens)

	scheduler, err := NewWatchdogPrefillScheduler(batchProfile, s.geometry, s.hardware)
	if err != nil {
		return nil, err
	}

	// For batched prefill, chunk the aggregate token stream
	return scheduler.PlanSchedule(totalTokens)
}

// adaptivelyScaleChunk iteratively subdivides candidateTokens so that the chunk duration
// remains strictly within ceilingMs at deep sequence positions.
func (s *WatchdogPrefillScheduler) adaptivelyScaleChunk(
	start, initialTokens, remaining, batchSize int, ceilingMs float64,
) int {
	minTokens := s.profile.MinChunkTokens
	if minTokens <= 0 {
		minTokens = AbsoluteMinChunkTokens
	}
	if minTokens > remaining {
		minTokens = remaining
	}

	tokens := initialTokens
	for tokens > minTokens {
		_, _, _, durMs, _, _, _ := s.EstimateChunkPacing(start, tokens, batchSize)
		if durMs <= ceilingMs {
			return tokens
		}
		// Halve the candidate tokens to quickly descend below the ceiling
		nextTokens := tokens / 2
		if nextTokens < minTokens {
			nextTokens = minTokens
		}
		if nextTokens == tokens {
			break
		}
		tokens = nextTokens
	}
	return tokens
}

// Execute drives the prefill schedule, running each chunk through the provided executor
// and invoking the inter-chunk hook between submissions for driver ISR interleaving.
func (s *WatchdogPrefillScheduler) Execute(
	ctx context.Context,
	schedule *PrefillSchedule,
	executor ChunkExecutor,
	hook InterChunkHook,
) (*ExecutionReceipt, error) {
	if schedule == nil || len(schedule.Chunks) == 0 {
		return nil, ErrEmptySchedule
	}
	if ctx == nil {
		ctx = context.Background()
	}

	receipt := &ExecutionReceipt{
		TotalTokens: schedule.TotalTokens,
	}

	startTime := time.Now()
	completedTokens := 0

	for i, chunk := range schedule.Chunks {
		// Check cancellation context before each chunk submission
		if err := ctx.Err(); err != nil {
			receipt.Cancelled = true
			receipt.CompletedClean = false
			receipt.TotalElapsed = time.Since(startTime)
			receipt.Error = fmt.Errorf("%w: %v", ErrExecutionCancelled, err)
			return receipt, receipt.Error
		}

		// Execute the prefill chunk
		if executor != nil {
			if err := executor(ctx, chunk); err != nil {
				receipt.CompletedClean = false
				receipt.TotalElapsed = time.Since(startTime)
				receipt.Error = fmt.Errorf("prefill chunk %d execution failed: %w", i, err)
				return receipt, receipt.Error
			}
		}

		completedTokens += chunk.TokenCount
		receipt.ChunksExecuted++

		elapsed := time.Since(startTime)
		pct := (float64(completedTokens) / float64(schedule.TotalTokens)) * 100.0

		// Estimate remaining time based on remaining chunks
		remainingMs := 0.0
		for j := i + 1; j < len(schedule.Chunks); j++ {
			remainingMs += schedule.Chunks[j].EstimatedDurationMs
		}
		estRemaining := time.Duration(remainingMs * float64(time.Millisecond))

		report := ProgressReport{
			ChunkIndex:         i,
			TotalChunks:        len(schedule.Chunks),
			CompletedTokens:    completedTokens,
			TotalTokens:        schedule.TotalTokens,
			ElapsedDuration:    elapsed,
			EstimatedRemaining: estRemaining,
			PercentComplete:    pct,
			CurrentChunk:       chunk,
		}

		// Invoke inter-chunk hook (ISR interleaving / progress update)
		if hook != nil {
			if err := hook(ctx, chunk, report); err != nil {
				receipt.CompletedClean = false
				receipt.TotalElapsed = time.Since(startTime)
				receipt.Error = fmt.Errorf("prefill chunk %d inter-chunk hook failed: %w", i, err)
				return receipt, receipt.Error
			}
		}

		if !chunk.IsLastChunk {
			receipt.YieldPointsHit++
		}
	}

	receipt.TotalElapsed = time.Since(startTime)
	receipt.CompletedClean = true
	return receipt, nil
}
