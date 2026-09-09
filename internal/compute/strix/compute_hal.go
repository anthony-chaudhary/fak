// Package strix implements the hardware profile, memory bandwidth physics,
// compute HAL, and unified memory pipeline for AMD Strix Halo (Ryzen AI Max+ 395).
package strix

import (
	"errors"
	"fmt"
	"math"
	"sync"
	"sync/atomic"
	"time"
	"unsafe"
)

// RDNAArch represents the targeted AMD GPU microarchitecture.
type RDNAArch string

const (
	// ArchGFX1150 targets Strix Point / Strix Halo variant (16 CUs, dual-issue SIMD32).
	ArchGFX1150 RDNAArch = "gfx1150"
	// ArchGFX1151 targets AMD Strix Halo top-die (40 CUs RDNA 3.5, Ryzen AI Max+ 395).
	ArchGFX1151 RDNAArch = "gfx1151"
)

// TargetConfig encapsulates the compute, clock, wave, and cache topology for RDNA 3.5 targets.
type TargetConfig struct {
	Arch              RDNAArch `json:"arch"`
	ModelName         string   `json:"model_name"`
	ComputeUnits      int      `json:"compute_units"`
	WaveSize          int      `json:"wave_size"`
	ClockMHz          int      `json:"clock_mhz"`
	SIMDPerCU         int      `json:"simd_per_cu"`
	L3CacheBytes      int64    `json:"l3_cache_bytes"`
	SupportsWMMA      bool     `json:"supports_wmma"`
	DualIssueSIMD     bool     `json:"dual_issue_simd"`
	PeakFP32TFLOPs    float64  `json:"peak_fp32_tflops"`
	PeakBF16TFLOPs    float64  `json:"peak_bf16_tflops"`
	PeakBandwidthGBps float64  `json:"peak_bandwidth_gbps"`
}

// Strix Halo default hardware specifications.
const (
	DefaultStrixHaloClockMHz     = 2900
	DefaultStrixHaloGFX1151CUs   = 40
	DefaultStrixPointGFX1150CUs  = 16
	DefaultStrixHaloL3CacheBytes = 32 * 1024 * 1024 // 32MB MALL / Infinity Cache
	DefaultWaveSize              = 32               // RDNA 3.5 native compute Wave32 mode

	// BusBandwidthGBps is the nominal memory bandwidth of the 256-bit LPDDR5X bus (256 GB/s).
	BusBandwidthGBps float64 = 256.0

	// VRAMAllocationCapBytes is the strict ceiling on VRAM allocation (96 GiB = 128 GiB - 32 GiB).
	VRAMAllocationCapBytes int64 = 96 * 1024 * 1024 * 1024
)

// NewGFX1151Config returns the target profile for Ryzen AI Max+ 395 (40 CUs RDNA 3.5).
func NewGFX1151Config() TargetConfig {
	// 40 CUs * 64 ALUs/CU * 2 (dual issue) * 2.9 GHz * 2 (FMA) ~ 29.7 FP32 TFLOPs
	// Dual-issue WMMA BF16 / FP16 delivers ~59.4 TFLOPs.
	return TargetConfig{
		Arch:              ArchGFX1151,
		ModelName:         "AMD Radeon 8060S / 8050S (Strix Halo GFX1151)",
		ComputeUnits:      DefaultStrixHaloGFX1151CUs,
		WaveSize:          DefaultWaveSize,
		ClockMHz:          DefaultStrixHaloClockMHz,
		SIMDPerCU:         2,
		L3CacheBytes:      DefaultStrixHaloL3CacheBytes,
		SupportsWMMA:      true,
		DualIssueSIMD:     true,
		PeakFP32TFLOPs:    29.7,
		PeakBF16TFLOPs:    59.4,
		PeakBandwidthGBps: BusBandwidthGBps, // 256 GB/s unified
	}
}

// NewGFX1150Config returns the target profile for Strix Point (16 CUs RDNA 3.5).
func NewGFX1150Config() TargetConfig {
	return TargetConfig{
		Arch:              ArchGFX1150,
		ModelName:         "AMD Radeon 890M (Strix Point GFX1150)",
		ComputeUnits:      DefaultStrixPointGFX1150CUs,
		WaveSize:          DefaultWaveSize,
		ClockMHz:          DefaultStrixHaloClockMHz,
		SIMDPerCU:         2,
		L3CacheBytes:      16 * 1024 * 1024, // 16MB L3
		SupportsWMMA:      true,
		DualIssueSIMD:     true,
		PeakFP32TFLOPs:    11.88,
		PeakBF16TFLOPs:    23.76,
		PeakBandwidthGBps: 135.0, // standard LPDDR5X dual-channel
	}
}

// GetTargetConfig resolves configuration by target architecture string.
func GetTargetConfig(arch RDNAArch) (TargetConfig, error) {
	switch arch {
	case ArchGFX1151:
		return NewGFX1151Config(), nil
	case ArchGFX1150:
		return NewGFX1150Config(), nil
	default:
		return TargetConfig{}, fmt.Errorf("unsupported RDNA architecture target: %q (expected gfx1150 or gfx1151)", arch)
	}
}

// Unified buffer allocation flags modeling Linux ROCm and Windows Vulkan APIs.
const (
	// AllocHostMapped models Linux hipHostAllocMapped / hipHostAllocPortable.
	AllocHostMapped uint32 = 1 << 0
	// AllocVulkanHostVisible models Windows Vulkan host-visible, host-coherent buffers.
	AllocVulkanHostVisible uint32 = 1 << 1
	// AllocDeviceLocal flags memory as accessible by the RDNA 3.5 compute engine without staging.
	AllocDeviceLocal uint32 = 1 << 2
	// AllocCoherent indicates hardware-level cache coherency across CPU, GPU, and NPU.
	AllocCoherent uint32 = 1 << 3
	// AllocPinned guarantees memory pages are pinned in physical DRAM.
	AllocPinned uint32 = 1 << 4

	// StandardUMAPlags combines all unified memory attributes for zero-copy APU operations.
	StandardUMAPlags = AllocHostMapped | AllocVulkanHostVisible | AllocDeviceLocal | AllocCoherent | AllocPinned
)

// PageAlignment is the 4096-byte memory boundary required for SIMD execution and zero-copy DMA.
const PageAlignment = 4096

// Errors for HAL and buffer operations.
var (
	ErrBufferFreed           = errors.New("strix/hal: unified buffer has already been freed")
	ErrOutOfMemory           = errors.New("strix/hal: out of unified memory capacity")
	ErrBufferMisaligned      = errors.New("strix/hal: buffer base pointer is not 4096-byte page-aligned")
	ErrInvalidDimensions     = errors.New("strix/hal: invalid matrix or vector dimensions")
	ErrDimensionMismatch     = errors.New("strix/hal: tensor dimension mismatch")
	ErrCosineParityViolation = errors.New("strix/hal: cosine parity check failed (< 0.9999)")
	ErrBusSaturation         = errors.New("strix/hal: memory streaming rate exceeds 256 GB/s bus ceiling")
)

// IsPageAligned checks if a memory pointer is aligned to a 4096-byte page boundary.
func IsPageAligned(ptr uintptr) bool {
	return ptr != 0 && (ptr%PageAlignment) == 0
}

// UnifiedBuffer represents a host-visible, device-local memory buffer in Strix Halo unified memory.
type UnifiedBuffer struct {
	HostPtr       uintptr  `json:"host_ptr"`
	DevicePtr     uintptr  `json:"device_ptr"`
	Size          int64    `json:"size"`
	Capacity      int64    `json:"capacity"`
	AlignedOffset int      `json:"aligned_offset"`
	Flags         uint32   `json:"flags"`
	Arch          RDNAArch `json:"arch"`

	raw    []byte
	mu     sync.RWMutex
	isFree uint32
}

// AlignedBytes returns the byte slice starting strictly at the 4096-byte aligned base address.
func (b *UnifiedBuffer) AlignedBytes() []byte {
	b.mu.RLock()
	defer b.mu.RUnlock()
	if atomic.LoadUint32(&b.isFree) == 1 {
		return nil
	}
	return b.raw[b.AlignedOffset : b.AlignedOffset+int(b.Size)]
}

// Float32Slice returns a float32 interpretation of the 4096-byte aligned buffer.
func (b *UnifiedBuffer) Float32Slice() []float32 {
	bytes := b.AlignedBytes()
	if len(bytes) < 4 {
		return nil
	}
	count := len(bytes) / 4
	return unsafe.Slice((*float32)(unsafe.Pointer(&bytes[0])), count)
}

// Uint16Slice returns a uint16 (BF16 raw bits) interpretation of the 4096-byte aligned buffer.
func (b *UnifiedBuffer) Uint16Slice() []uint16 {
	bytes := b.AlignedBytes()
	if len(bytes) < 2 {
		return nil
	}
	count := len(bytes) / 2
	return unsafe.Slice((*uint16)(unsafe.Pointer(&bytes[0])), count)
}

// IsZeroCopy returns true if the buffer has zero-copy host-visible device-local mapping.
func (b *UnifiedBuffer) IsZeroCopy() bool {
	return (b.Flags&AllocHostMapped != 0 || b.Flags&AllocVulkanHostVisible != 0) &&
		(b.Flags&AllocDeviceLocal != 0) && (b.HostPtr == b.DevicePtr)
}

// IsCoherent returns true if hardware snooping/coherency is active.
func (b *UnifiedBuffer) IsCoherent() bool {
	return b.Flags&AllocCoherent != 0
}

// Free releases the buffer back to the unified pool.
func (b *UnifiedBuffer) Free() error {
	if !atomic.CompareAndSwapUint32(&b.isFree, 0, 1) {
		return ErrBufferFreed
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	b.raw = nil
	b.HostPtr = 0
	b.DevicePtr = 0
	b.Size = 0
	return nil
}

// IsFreed reports whether the buffer has been freed.
func (b *UnifiedBuffer) IsFreed() bool {
	return atomic.LoadUint32(&b.isFree) == 1
}

// UnifiedAllocator manages page-aligned allocations in Strix Halo unified memory.
type UnifiedAllocator struct {
	mu             sync.Mutex
	maxBytes       int64
	allocatedBytes int64
	peakBytes      int64
	arch           RDNAArch
	buffers        map[uintptr]*UnifiedBuffer
}

// NewUnifiedAllocator initializes an allocator bounded by maxBytes (e.g. 128 GiB or 96 GiB VRAM cap).
func NewUnifiedAllocator(maxBytes int64, arch RDNAArch) *UnifiedAllocator {
	if maxBytes <= 0 {
		maxBytes = VRAMAllocationCapBytes
	}
	if arch == "" {
		arch = ArchGFX1151
	}
	return &UnifiedAllocator{
		maxBytes: maxBytes,
		arch:     arch,
		buffers:  make(map[uintptr]*UnifiedBuffer),
	}
}

// Allocate allocates a 4096-byte aligned UnifiedBuffer.
func (a *UnifiedAllocator) Allocate(size int64, flags uint32) (*UnifiedBuffer, error) {
	if size <= 0 {
		return nil, errors.New("strix/hal: allocation size must be positive")
	}

	a.mu.Lock()
	defer a.mu.Unlock()

	if a.allocatedBytes+size > a.maxBytes {
		return nil, fmt.Errorf("%w: requested %d bytes, already allocated %d of %d max bytes",
			ErrOutOfMemory, size, a.allocatedBytes, a.maxBytes)
	}

	// Allocate buffer with padding to ensure 4096-byte boundary alignment.
	totalCap := size + PageAlignment
	raw := make([]byte, totalCap)

	baseAddr := uintptr(unsafe.Pointer(&raw[0]))
	offset := int((PageAlignment - (baseAddr % PageAlignment)) % PageAlignment)
	alignedPtr := baseAddr + uintptr(offset)

	if !IsPageAligned(alignedPtr) {
		return nil, ErrBufferMisaligned
	}

	if flags == 0 {
		flags = StandardUMAPlags
	}

	buf := &UnifiedBuffer{
		HostPtr:       alignedPtr,
		DevicePtr:     alignedPtr, // On coherent UMA, DevicePtr matches HostPtr
		Size:          size,
		Capacity:      totalCap,
		AlignedOffset: offset,
		Flags:         flags,
		Arch:          a.arch,
		raw:           raw,
	}

	a.allocatedBytes += size
	if a.allocatedBytes > a.peakBytes {
		a.peakBytes = a.allocatedBytes
	}
	a.buffers[alignedPtr] = buf

	return buf, nil
}

// Free releases a buffer and updates allocator accounting.
func (a *UnifiedAllocator) Free(buf *UnifiedBuffer) error {
	if buf == nil {
		return nil
	}

	a.mu.Lock()
	defer a.mu.Unlock()

	ptr := buf.HostPtr
	size := buf.Size

	err := buf.Free()
	if err != nil {
		return err
	}

	if _, exists := a.buffers[ptr]; exists {
		delete(a.buffers, ptr)
		a.allocatedBytes -= size
		if a.allocatedBytes < 0 {
			a.allocatedBytes = 0
		}
	}

	return nil
}

// AllocatedBytes returns the current memory in use.
func (a *UnifiedAllocator) AllocatedBytes() int64 {
	a.mu.Lock()
	defer a.mu.Unlock()
	return a.allocatedBytes
}

// PeakBytes returns the peak memory allocated during the lifetime.
func (a *UnifiedAllocator) PeakBytes() int64 {
	a.mu.Lock()
	defer a.mu.Unlock()
	return a.peakBytes
}

// AvailableBytes returns the remaining capacity before hitting the limit.
func (a *UnifiedAllocator) AvailableBytes() int64 {
	a.mu.Lock()
	defer a.mu.Unlock()
	avail := a.maxBytes - a.allocatedBytes
	if avail < 0 {
		return 0
	}
	return avail
}

// Reset frees all tracked buffers.
func (a *UnifiedAllocator) Reset() {
	a.mu.Lock()
	defer a.mu.Unlock()
	for _, buf := range a.buffers {
		_ = buf.Free()
	}
	a.buffers = make(map[uintptr]*UnifiedBuffer)
	a.allocatedBytes = 0
}

// --- BF16 and FP32 Conversion and Numerical Parity ---

// FP32ToBF16 converts an IEEE 754 float32 to brain floating point (BF16) using round-to-nearest-even.
func FP32ToBF16(f float32) uint16 {
	u := math.Float32bits(f)

	// Check for NaN: preserve quiet NaN bits
	if (u&0x7F800000) == 0x7F800000 && (u&0x007FFFFF) != 0 {
		return uint16((u >> 16) | 0x0040)
	}

	// Round-to-nearest-even
	rounding := uint32(0x7FFF) + ((u >> 16) & 1)
	u += rounding
	return uint16(u >> 16)
}

// BF16ToFP32 converts a BF16 16-bit representation to an IEEE 754 float32.
func BF16ToFP32(b uint16) float32 {
	u := uint32(b) << 16
	return math.Float32frombits(u)
}

// FP32SliceToBF16 converts a slice of float32 values into BF16 uint16 values.
func FP32SliceToBF16(src []float32) []uint16 {
	dst := make([]uint16, len(src))
	for i, v := range src {
		dst[i] = FP32ToBF16(v)
	}
	return dst
}

// BF16SliceToFP32 converts a slice of BF16 uint16 values into float32 values.
func BF16SliceToFP32(src []uint16) []float32 {
	dst := make([]float32, len(src))
	for i, v := range src {
		dst[i] = BF16ToFP32(v)
	}
	return dst
}

// CosineSimilarity computes the cosine similarity between two float32 slices:
// Cosine(A, B) = (A . B) / (||A||_2 * ||B||_2).
func CosineSimilarity(a, b []float32) (float64, error) {
	if len(a) != len(b) {
		return 0, fmt.Errorf("%w: slice a has len %d, slice b has len %d", ErrDimensionMismatch, len(a), len(b))
	}
	if len(a) == 0 {
		return 0, ErrInvalidDimensions
	}

	var dot, normA, normB float64
	for i := 0; i < len(a); i++ {
		valA := float64(a[i])
		valB := float64(b[i])
		dot += valA * valB
		normA += valA * valA
		normB += valB * valB
	}

	if normA == 0 || normB == 0 {
		if normA == 0 && normB == 0 {
			return 1.0, nil
		}
		return 0.0, nil
	}

	sim := dot / (math.Sqrt(normA) * math.Sqrt(normB))
	// Clamp numerical artifacts
	if sim > 1.0 {
		sim = 1.0
	} else if sim < -1.0 {
		sim = -1.0
	}
	return sim, nil
}

// ReferenceGEMM computes C = alpha * (A * B) + beta * C in full FP32 precision.
// A has dimensions M x K, B has dimensions K x N, C has dimensions M x N.
func ReferenceGEMM(M, N, K int, alpha float32, A, B []float32, beta float32, C []float32) ([]float32, error) {
	if M <= 0 || N <= 0 || K <= 0 {
		return nil, ErrInvalidDimensions
	}
	if len(A) != M*K || len(B) != K*N {
		return nil, fmt.Errorf("%w: A len %d (want %d), B len %d (want %d)", ErrDimensionMismatch, len(A), M*K, len(B), K*N)
	}

	out := make([]float32, M*N)
	if len(C) == M*N && beta != 0 {
		copy(out, C)
	}

	for i := 0; i < M; i++ {
		for j := 0; j < N; j++ {
			var sum float64
			for k := 0; k < K; k++ {
				sum += float64(A[i*K+k]) * float64(B[k*N+j])
			}
			idx := i*N + j
			if beta != 0 && len(C) == M*N {
				out[idx] = float32(float64(alpha)*sum + float64(beta)*float64(C[idx]))
			} else {
				out[idx] = float32(float64(alpha) * sum)
			}
		}
	}

	return out, nil
}

// ReferenceGEMV computes y = A * x in full FP32 precision.
// A has dimensions M x K (row-major), x has length K, output y has length M.
func ReferenceGEMV(M, K int, A []float32, x []float32) ([]float32, error) {
	if M <= 0 || K <= 0 {
		return nil, ErrInvalidDimensions
	}
	if len(A) != M*K || len(x) != K {
		return nil, fmt.Errorf("%w: A len %d (want %d), x len %d (want %d)", ErrDimensionMismatch, len(A), M*K, len(x), K)
	}

	y := make([]float32, M)
	for i := 0; i < M; i++ {
		var sum float64
		rowOffset := i * K
		for k := 0; k < K; k++ {
			sum += float64(A[rowOffset+k]) * float64(x[k])
		}
		y[i] = float32(sum)
	}
	return y, nil
}

// RDNA35_WMMA_GEMM simulates RDNA 3.5 Wave Matrix Multiply Accumulate (WMMA) execution.
// Operates on BF16 input matrices with FP32 accumulation in Wave32 mode.
// C = alpha * (A_bf16 * B_bf16) + beta * C_f32.
func RDNA35_WMMA_GEMM(M, N, K int, alpha float32, A_bf16, B_bf16 []uint16, beta float32, C_f32 []float32) ([]float32, error) {
	if M <= 0 || N <= 0 || K <= 0 {
		return nil, ErrInvalidDimensions
	}
	if len(A_bf16) != M*K || len(B_bf16) != K*N {
		return nil, fmt.Errorf("%w: A_bf16 len %d, B_bf16 len %d", ErrDimensionMismatch, len(A_bf16), len(B_bf16))
	}

	out := make([]float32, M*N)

	// Simulating dual-issue SIMD32 wave execution with 32-element register tiles
	const waveLane = 32

	for i := 0; i < M; i++ {
		for j := 0; j < N; j++ {
			var accum float32 = 0.0

			// Unroll inner dot-product in 32-lane chunks (Wave32)
			for k := 0; k < K; k += waveLane {
				chunk := waveLane
				if k+chunk > K {
					chunk = K - k
				}

				// WMMA accumulator register maintains FP32 precision
				var waveAccum float32 = 0.0
				for c := 0; c < chunk; c++ {
					idxA := i*K + (k + c)
					idxB := (k+c)*N + j
					valA := BF16ToFP32(A_bf16[idxA])
					valB := BF16ToFP32(B_bf16[idxB])
					waveAccum += valA * valB
				}
				accum += waveAccum
			}

			idx := i*N + j
			if beta != 0 && len(C_f32) == M*N {
				out[idx] = alpha*accum + beta*C_f32[idx]
			} else {
				out[idx] = alpha * accum
			}
		}
	}

	return out, nil
}

// RDNA35_GEMV simulates streaming decode matrix-vector multiplication on RDNA 3.5.
// A_bf16 is M x K, x_f32 is K, returning y = A * x.
func RDNA35_GEMV(M, K int, A_bf16 []uint16, x_f32 []float32) ([]float32, error) {
	if M <= 0 || K <= 0 {
		return nil, ErrInvalidDimensions
	}
	if len(A_bf16) != M*K || len(x_f32) != K {
		return nil, fmt.Errorf("%w: A_bf16 len %d (want %d), x_f32 len %d (want %d)", ErrDimensionMismatch, len(A_bf16), M*K, len(x_f32), K)
	}

	y := make([]float32, M)
	const waveSize = 32

	for i := 0; i < M; i++ {
		var rowAccum float32 = 0.0
		rowBase := i * K

		for k := 0; k < K; k += waveSize {
			chunk := waveSize
			if k+chunk > K {
				chunk = K - k
			}

			var waveSum float32 = 0.0
			for c := 0; c < chunk; c++ {
				wVal := BF16ToFP32(A_bf16[rowBase+k+c])
				waveSum += wVal * x_f32[k+c]
			}
			rowAccum += waveSum
		}
		y[i] = rowAccum
	}

	return y, nil
}

// VerifyNumericalParity checks that candidate matches reference with cosine similarity >= minCosine.
func VerifyNumericalParity(reference []float32, candidate []float32, minCosine float64) (similarity float64, ok bool, err error) {
	if minCosine <= 0 {
		minCosine = 0.9999
	}
	sim, err := CosineSimilarity(reference, candidate)
	if err != nil {
		return 0, false, err
	}
	if sim < minCosine {
		return sim, false, fmt.Errorf("%w: achieved %.6f, minimum required %.6f", ErrCosineParityViolation, sim, minCosine)
	}
	return sim, true, nil
}

// --- Analytical Memory Streaming & Roofline Performance ---

// RooflineReport details theoretical vs achieved memory streaming on Strix Halo's 256 GB/s bus.
type RooflineReport struct {
	Arch                    RDNAArch `json:"arch"`
	ModelName               string   `json:"model_name"`
	ModelParams             int64    `json:"model_params"`
	BitsPerWeight           float64  `json:"bits_per_weight"`
	ActiveWeightBytes       int64    `json:"active_weight_bytes"`
	BatchSize               int      `json:"batch_size"`
	PeakBandwidthGBps       float64  `json:"peak_bandwidth_gbps"`
	PeakTFLOPs              float64  `json:"peak_tflops"`
	RidgePoint              float64  `json:"ridge_point_flops_per_byte"`
	OperationalIntensity    float64  `json:"operational_intensity_flops_per_byte"`
	IsMemoryBandwidthBound  bool     `json:"is_memory_bandwidth_bound"`
	TheoreticalDecodeTokSec float64  `json:"theoretical_decode_tok_per_sec"`
	AchievedBandwidthGBps   float64  `json:"achieved_bandwidth_gbps"`
	BusSaturationPercentage float64  `json:"bus_saturation_percentage"`
}

// CalculateStreamingRoofline analyzes decode/prefill operational intensity against the 256 GB/s bus.
func CalculateStreamingRoofline(target TargetConfig, modelParams int64, bitsPerWeight float64, batchSize int) *RooflineReport {
	if target.PeakBandwidthGBps <= 0 {
		target.PeakBandwidthGBps = BusBandwidthGBps
	}
	if target.PeakBF16TFLOPs <= 0 {
		target.PeakBF16TFLOPs = 59.4
	}
	if batchSize <= 0 {
		batchSize = 1
	}
	if bitsPerWeight <= 0 {
		bitsPerWeight = 4.0 // Default 4-bit quantization (Q4_K_M)
	}

	// Active weights in bytes
	activeWeightBytes := int64(float64(modelParams) * (bitsPerWeight / 8.0))

	// In autoregressive decode:
	// Total FLOPs per token per batch = 2 * modelParams * batchSize
	// Memory traffic per step = activeWeightBytes (weights loaded once per token step)
	flopsPerStep := 2.0 * float64(modelParams) * float64(batchSize)
	trafficBytesPerStep := float64(activeWeightBytes)

	operationalIntensity := flopsPerStep / trafficBytesPerStep // = 2 * batchSize / (bitsPerWeight / 8)

	// Ridge point: Peak FLOPs / Peak Bandwidth (Bytes/s)
	ridgePoint := (target.PeakBF16TFLOPs * 1e12) / (target.PeakBandwidthGBps * 1e9)

	isBandwidthBound := operationalIntensity < ridgePoint

	// Theoretical max decode tokens/s: (Bandwidth in bytes/s) / (activeWeightBytes) * batchSize
	var theoreticalTokSec float64
	if activeWeightBytes > 0 {
		theoreticalTokSec = (target.PeakBandwidthGBps * 1e9 / float64(activeWeightBytes)) * float64(batchSize)
	}

	return &RooflineReport{
		Arch:                    target.Arch,
		ModelName:               target.ModelName,
		ModelParams:             modelParams,
		BitsPerWeight:           bitsPerWeight,
		ActiveWeightBytes:       activeWeightBytes,
		BatchSize:               batchSize,
		PeakBandwidthGBps:       target.PeakBandwidthGBps,
		PeakTFLOPs:              target.PeakBF16TFLOPs,
		RidgePoint:              ridgePoint,
		OperationalIntensity:    operationalIntensity,
		IsMemoryBandwidthBound:  isBandwidthBound,
		TheoreticalDecodeTokSec: theoreticalTokSec,
	}
}

// MeasureStreamingRate computes achieved bandwidth in GB/s from bytes transferred and elapsed duration.
// Returns an error if the measured rate violates the physical 256 GB/s ceiling.
func MeasureStreamingRate(bytesTransferred int64, duration time.Duration) (achievedGBps float64, saturationPct float64, err error) {
	if duration <= 0 || bytesTransferred <= 0 {
		return 0, 0, errors.New("strix/hal: bytes and duration must be strictly positive")
	}

	seconds := duration.Seconds()
	bytesPerSec := float64(bytesTransferred) / seconds
	achievedGBps = bytesPerSec / 1e9

	saturationPct = (achievedGBps / BusBandwidthGBps) * 100.0

	// Margin of 1% allowed for measurement noise, but strictly detect physical impossibilities
	if achievedGBps > BusBandwidthGBps*1.05 {
		return achievedGBps, saturationPct, fmt.Errorf("%w: measured %.2f GB/s exceeds physical 256 GB/s ceiling (%.1f%% saturation)",
			ErrBusSaturation, achievedGBps, saturationPct)
	}

	return achievedGBps, saturationPct, nil
}
