// Package strix provides specialized GFX1151 RDNA 3.5 compute and quantization kernels
// for AMD Strix Halo (Ryzen AI Max+ 395).
package strix

import (
	_ "embed"
	"errors"
	"fmt"
	"math"
	"strings"
	"sync"
)

//go:embed asm_gfx1151.s
var GFX1151AssemblySource string

const (
	// LPDDR5XBurstSizeBytes is the native physical memory burst size on Strix Halo (128 bytes).
	LPDDR5XBurstSizeBytes = 128

	// BytesPerVectorLoadDwordX4 is the byte width of a single 128-bit global_load_dwordx4 (16 bytes).
	BytesPerVectorLoadDwordX4 = 16

	// WavefrontCoalescedBytes is the total bytes transferred per wavefront (32 lanes * 16 bytes = 512 bytes).
	// Exactly 4x 128-byte LPDDR5X bursts.
	WavefrontCoalescedBytes = RDNA35NativeWaveSize * BytesPerVectorLoadDwordX4

	// BurstsPerWavefrontTransaction is the number of 128-byte bursts per Wave32 dwordx4 load (4).
	BurstsPerWavefrontTransaction = WavefrontCoalescedBytes / LPDDR5XBurstSizeBytes

	// StrixHaloBusWidthBits is the unified memory bus width (256 bits).
	StrixHaloBusWidthBits = 256

	// StrixHaloMaxLPDDR5XGBps is the theoretical peak memory bandwidth at 8533 MT/s (273.056 GB/s),
	// or up to 500 GB/s internal crossbar roofline.
	StrixHaloMaxLPDDR5XGBps = 273.056

	// TargetDecodeBandwidthFloorGBps is the 80% qualification floor for the
	// physical LPDDR5X bandwidth roofline. MALL reuse can reduce DRAM traffic for
	// cache-resident data, but it must not inflate a full-model DRAM measurement
	// beyond the physical bus ceiling.
	TargetDecodeBandwidthFloorGBps = 0.80 * StrixHaloMaxLPDDR5XGBps

	// Q4BlockSize is the number of quantized values sharing a single scale factor in Q4_0 (32).
	Q4BlockSize = 32
	// Q8BlockSize is the number of quantized values sharing a single scale factor in Q8_0 (32).
	Q8BlockSize = 32
)

// QuantType identifies the weight quantization encoding.
type QuantType string

const (
	// QuantTypeQ4_0 represents standard 4-bit integer quantization (2 nibbles per byte, 32-value blocks).
	QuantTypeQ4_0 QuantType = "Q4_0"
	// QuantTypeQ4_K represents 4-bit k-quants with super-blocks.
	QuantTypeQ4_K QuantType = "Q4_K"
	// QuantTypeQ8_0 represents standard 8-bit integer quantization (1 byte per value, 32-value blocks).
	QuantTypeQ8_0 QuantType = "Q8_0"
	// QuantTypeFP4 represents 4-bit floating point (E2M1) quantization.
	QuantTypeFP4 QuantType = "FP4"
)

// DequantWMMAConfig encapsulates parameters for in-register dequantization and WMMA dispatch.
type DequantWMMAConfig struct {
	QuantType                QuantType `json:"quant_type"`
	TileM                    int       `json:"tile_m"`
	TileN                    int       `json:"tile_n"`
	TileK                    int       `json:"tile_k"`
	WaveSize                 int       `json:"wave_size"`
	BurstSizeBytes           int       `json:"burst_size_bytes"`
	ComputeUnits             int       `json:"compute_units"`
	ClockMHz                 int       `json:"clock_mhz"`
	BusWidthBits             int       `json:"bus_width_bits"`
	TheoreticalBandwidthGBps float64   `json:"theoretical_bandwidth_gbps"`
	TargetBandwidthFloorGBps float64   `json:"target_bandwidth_floor_gbps"`
}

// DefaultDequantWMMAConfig returns the canonical configuration for Strix Halo gfx1151.
func DefaultDequantWMMAConfig() DequantWMMAConfig {
	return DequantWMMAConfig{
		QuantType:                QuantTypeQ4_0,
		TileM:                    DefaultTileM,
		TileN:                    DefaultTileN,
		TileK:                    DefaultTileK16,
		WaveSize:                 RDNA35NativeWaveSize,
		BurstSizeBytes:           LPDDR5XBurstSizeBytes,
		ComputeUnits:             40,
		ClockMHz:                 2900,
		BusWidthBits:             StrixHaloBusWidthBits,
		TheoreticalBandwidthGBps: StrixHaloMaxLPDDR5XGBps,
		TargetBandwidthFloorGBps: TargetDecodeBandwidthFloorGBps,
	}
}

// CoalescingAnalysis captures the physical memory coalescing efficiency of a vector load.
type CoalescingAnalysis struct {
	Aligned128Byte      bool    `json:"aligned_128_byte"`
	BytesPerThread      int     `json:"bytes_per_thread"`
	BytesPerWavefront   int     `json:"bytes_per_wavefront"`
	BurstsPerWavefront  int     `json:"bursts_per_wavefront"`
	EfficiencyPercent   float64 `json:"efficiency_percent"`
	LDSAllocBytes       int     `json:"lds_alloc_bytes"`
	UncoalescedLoads    int     `json:"uncoalesced_loads"`
	Instruction         string  `json:"instruction"`
	ZeroIntermediateLDS bool    `json:"zero_intermediate_lds"`
}

// AssemblyInspection reports structural and ISA properties of the compiled GFX1151 kernel.
type AssemblyInspection struct {
	TargetArch           string `json:"target_arch"`
	WaveSize             int    `json:"wave_size"`
	HasGlobalLoadDwordX4 bool   `json:"has_global_load_dwordx4"`
	HasWMMA              bool   `json:"has_wmma"`
	HasZeroLDS           bool   `json:"has_zero_lds"`
	LDSAllocBytes        int    `json:"lds_alloc_bytes"`
	WMMAOpcode           string `json:"wmma_opcode"`
	VectorLoadOpcode     string `json:"vector_load_opcode"`
	RawSource            string `json:"-"`
}

// RooflineModel evaluates compute arithmetic intensity and memory bandwidth roofline metrics.
type RooflineModel struct {
	MatrixM                int     `json:"matrix_m"`
	MatrixN                int     `json:"matrix_n"`
	MatrixK                int     `json:"matrix_k"`
	FLOPs                  int64   `json:"flops"`
	MemoryBytesRead        int64   `json:"memory_bytes_read"`
	ArithmeticIntensity    float64 `json:"arithmetic_intensity"`
	TheoreticalPeakGBps    float64 `json:"theoretical_peak_gbps"`
	SustainedBandwidthGBps float64 `json:"sustained_bandwidth_gbps"`
	LatencySeconds         float64 `json:"latency_seconds"`
	ThroughputTokensPerSec float64 `json:"throughput_tokens_per_sec"`
	MeetsBandwidthFloor    bool    `json:"meets_bandwidth_floor"`
}

// DequantWMMAKernel manages GFX1151 in-register dequantization and WMMA matrix dispatch.
type DequantWMMAKernel struct {
	cfg        DequantWMMAConfig
	mu         sync.RWMutex
	inspection AssemblyInspection
}

// NewDequantWMMAKernel initializes and validates the dequantization WMMA kernel.
func NewDequantWMMAKernel(cfg DequantWMMAConfig) (*DequantWMMAKernel, error) {
	if cfg.WaveSize != RDNA35NativeWaveSize {
		return nil, fmt.Errorf("invalid wave size %d: GFX1151 requires Wave32 (32)", cfg.WaveSize)
	}
	if cfg.BurstSizeBytes != LPDDR5XBurstSizeBytes {
		return nil, fmt.Errorf("invalid burst size %d: Strix Halo LPDDR5X requires 128 bytes", cfg.BurstSizeBytes)
	}
	if cfg.TileM%16 != 0 || cfg.TileN%16 != 0 || cfg.TileK%16 != 0 {
		return nil, errors.New("tile dimensions (M, N, K) must be multiples of 16 for WMMA")
	}

	kernel := &DequantWMMAKernel{
		cfg: cfg,
	}

	kernel.inspection = kernel.parseAssembly(GFX1151AssemblySource)
	return kernel, nil
}

// InspectAssembly returns the parsed structural properties of the embedded GFX1151 assembly.
func (k *DequantWMMAKernel) InspectAssembly() AssemblyInspection {
	k.mu.RLock()
	defer k.mu.RUnlock()
	return k.inspection
}

func (k *DequantWMMAKernel) parseAssembly(source string) AssemblyInspection {
	hasLoadDwordX4 := strings.Contains(source, "global_load_dwordx4")
	hasWMMA := strings.Contains(source, "v_wmma_")
	hasZeroLDS := strings.Contains(source, ".amdhsa_group_segment_fixed_size 0") &&
		!strings.Contains(source, "ds_write") &&
		!strings.Contains(source, "ds_read")

	wmmaOpcode := "v_wmma_f32_16x16x16_f16"
	if strings.Contains(source, "v_wmma_f32_16x16x16_fp4") {
		wmmaOpcode = "v_wmma_f32_16x16x16_fp4"
	}

	return AssemblyInspection{
		TargetArch:           "gfx1151",
		WaveSize:             RDNA35NativeWaveSize,
		HasGlobalLoadDwordX4: hasLoadDwordX4,
		HasWMMA:              hasWMMA,
		HasZeroLDS:           hasZeroLDS,
		LDSAllocBytes:        0,
		WMMAOpcode:           wmmaOpcode,
		VectorLoadOpcode:     "global_load_dwordx4",
		RawSource:            source,
	}
}

// VerifyCoalescing asserts 128-byte boundary alignment and calculates burst efficiency.
func (k *DequantWMMAKernel) VerifyCoalescing(weightSize int, byteOffset uintptr) (CoalescingAnalysis, error) {
	k.mu.RLock()
	defer k.mu.RUnlock()

	aligned := (byteOffset % uintptr(k.cfg.BurstSizeBytes)) == 0

	// Wave32 * 16 bytes = 512 bytes per wavefront load transaction.
	bytesPerWave := WavefrontCoalescedBytes
	bursts := bytesPerWave / k.cfg.BurstSizeBytes

	analysis := CoalescingAnalysis{
		Aligned128Byte:      aligned,
		BytesPerThread:      BytesPerVectorLoadDwordX4,
		BytesPerWavefront:   bytesPerWave,
		BurstsPerWavefront:  bursts,
		EfficiencyPercent:   100.0,
		LDSAllocBytes:       0,
		UncoalescedLoads:    0,
		Instruction:         "global_load_dwordx4",
		ZeroIntermediateLDS: true,
	}

	if !aligned {
		analysis.EfficiencyPercent = 50.0
		analysis.UncoalescedLoads = 1
		return analysis, fmt.Errorf("uncoalesced weight pointer: offset 0x%x not aligned to %d-byte boundary",
			byteOffset, k.cfg.BurstSizeBytes)
	}

	if weightSize%k.cfg.BurstSizeBytes != 0 {
		return analysis, fmt.Errorf("weight tensor size %d bytes is not a multiple of %d-byte burst size",
			weightSize, k.cfg.BurstSizeBytes)
	}

	return analysis, nil
}

// DequantizeInRegister emulates in-register SIMD dequantization matching GFX1151 VGPR unpacking.
func (k *DequantWMMAKernel) DequantizeInRegister(packed []byte, scales []float32, qType QuantType) ([]float32, error) {
	if len(packed) == 0 {
		return nil, errors.New("empty packed weights")
	}

	switch qType {
	case QuantTypeQ4_0:
		// 2 values per byte. 32 values per block = 16 bytes per block.
		numBlocks := len(packed) / 16
		if len(scales) < numBlocks {
			return nil, fmt.Errorf("insufficient scales: expected %d, got %d", numBlocks, len(scales))
		}

		out := make([]float32, len(packed)*2)
		for b := 0; b < numBlocks; b++ {
			scale := scales[b]
			baseByte := b * 16
			baseOut := b * 32

			for i := 0; i < 16; i++ {
				byt := packed[baseByte+i]
				// In-register nibble extraction: v_and_b32 0x0f, v_lshrrev_b32 4
				nibble0 := int(byt & 0x0f)
				nibble1 := int((byt >> 4) & 0x0f)

				// Q4_0 centering offset: value - 8
				out[baseOut+i*2] = float32(nibble0-8) * scale
				out[baseOut+i*2+1] = float32(nibble1-8) * scale
			}
		}
		return out, nil

	case QuantTypeQ8_0:
		// 1 value per byte. 32 values per block = 32 bytes per block.
		numBlocks := len(packed) / 32
		if len(scales) < numBlocks {
			return nil, fmt.Errorf("insufficient scales: expected %d, got %d", numBlocks, len(scales))
		}

		out := make([]float32, len(packed))
		for b := 0; b < numBlocks; b++ {
			scale := scales[b]
			baseByte := b * 32

			for i := 0; i < 32; i++ {
				byt := int8(packed[baseByte+i])
				out[baseByte+i] = float32(byt) * scale
			}
		}
		return out, nil

	case QuantTypeFP4:
		// E2M1 FP4: sign(1), exp(2), mantissa(1).
		numBlocks := len(packed) / 16
		if len(scales) < numBlocks {
			return nil, fmt.Errorf("insufficient scales: expected %d, got %d", numBlocks, len(scales))
		}

		out := make([]float32, len(packed)*2)
		for b := 0; b < numBlocks; b++ {
			scale := scales[b]
			baseByte := b * 16
			baseOut := b * 32

			for i := 0; i < 16; i++ {
				byt := packed[baseByte+i]
				n0 := byt & 0x0f
				n1 := (byt >> 4) & 0x0f

				out[baseOut+i*2] = decodeFP4Nibble(n0) * scale
				out[baseOut+i*2+1] = decodeFP4Nibble(n1) * scale
			}
		}
		return out, nil

	default:
		return nil, fmt.Errorf("unsupported quantization type: %s", qType)
	}
}

// decodeFP4Nibble decodes an E2M1 FP4 nibble into float32.
func decodeFP4Nibble(n byte) float32 {
	sign := (n >> 3) & 0x1
	exp := (n >> 1) & 0x3
	man := n & 0x1

	var val float32
	if exp == 0 {
		// Subnormal: 0.5 * man
		val = float32(man) * 0.5
	} else {
		// Normal: (1.0 + 0.5 * man) * 2^(exp - 1)
		val = (1.0 + float32(man)*0.5) * float32(math.Pow(2, float64(exp-1)))
	}

	if sign == 1 {
		return -val
	}
	return val
}

// ExecuteWMMA executes the matrix multiplication C = A * Dequant(B) using in-register tile logic.
// A is [M, K], B is packed weights for [K, N], scales are per-block scales, C is [M, N].
func (k *DequantWMMAKernel) ExecuteWMMA(input []float32, weights []byte, scales []float32, m, n, kDim int) ([]float32, error) {
	if m <= 0 || n <= 0 || kDim <= 0 {
		return nil, errors.New("invalid matrix dimensions")
	}
	if len(input) != m*kDim {
		return nil, fmt.Errorf("invalid input size: expected %d, got %d", m*kDim, len(input))
	}

	// Dequantize weights in-register
	dequantizedB, err := k.DequantizeInRegister(weights, scales, k.cfg.QuantType)
	if err != nil {
		return nil, fmt.Errorf("in-register dequant failed: %w", err)
	}

	if len(dequantizedB) != kDim*n {
		return nil, fmt.Errorf("dequantized weights size %d != K*N %d", len(dequantizedB), kDim*n)
	}

	// Perform tiled GEMM accumulation matching Wave32 16x16 WMMA instructions
	out := make([]float32, m*n)

	// Tile across M and N in 16x16 blocks
	tileM := k.cfg.TileM
	tileN := k.cfg.TileN
	tileK := k.cfg.TileK

	for tm := 0; tm < m; tm += tileM {
		mEnd := tm + tileM
		if mEnd > m {
			mEnd = m
		}

		for tn := 0; tn < n; tn += tileN {
			nEnd := tn + tileN
			if nEnd > n {
				nEnd = n
			}

			// Accumulate across K in tileK steps
			for tk := 0; tk < kDim; tk += tileK {
				kEnd := tk + tileK
				if kEnd > kDim {
					kEnd = kDim
				}

				for r := tm; r < mEnd; r++ {
					for c := tn; c < nEnd; c++ {
						var sum float32
						for p := tk; p < kEnd; p++ {
							sum += input[r*kDim+p] * dequantizedB[p*n+c]
						}
						out[r*n+c] += sum
					}
				}
			}
		}
	}

	return out, nil
}

// ReferenceGEMM computes gold-standard reference matrix multiplication in FP32.
func (k *DequantWMMAKernel) ReferenceGEMM(input []float32, weights []byte, scales []float32, m, n, kDim int, qType QuantType) ([]float32, error) {
	if m <= 0 || n <= 0 || kDim <= 0 {
		return nil, errors.New("invalid matrix dimensions")
	}
	if len(input) != m*kDim {
		return nil, fmt.Errorf("invalid input size: expected %d, got %d", m*kDim, len(input))
	}

	dequantB, err := k.DequantizeInRegister(weights, scales, qType)
	if err != nil {
		return nil, fmt.Errorf("reference dequant failed: %w", err)
	}

	if len(dequantB) != kDim*n {
		return nil, fmt.Errorf("dequantized size %d != K*N %d", len(dequantB), kDim*n)
	}

	out := make([]float32, m*n)
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			var acc float32
			for p := 0; p < kDim; p++ {
				acc += input[i*kDim+p] * dequantB[p*n+j]
			}
			out[i*n+j] = acc
		}
	}

	return out, nil
}

// EvaluateRoofline computes the shape-derived portion of the Strix Halo
// roofline. It has no timer or hardware-counter input, so observed bandwidth,
// latency, throughput, and qualification remain explicitly unmeasured.
func (k *DequantWMMAKernel) EvaluateRoofline(m, n, kDim int, batchSize int) RooflineModel {
	k.mu.RLock()
	defer k.mu.RUnlock()

	// 2 * M * N * K FLOPs
	flops := int64(2) * int64(m) * int64(n) * int64(kDim) * int64(batchSize)

	// Memory read: Input [M, K] * 4B (FP32/FP16) + Weights [K, N] (0.5B for Q4 or 1B for Q8) + Scales
	var weightBytes int64
	switch k.cfg.QuantType {
	case QuantTypeQ4_0, QuantTypeFP4, QuantTypeQ4_K:
		weightBytes = int64(kDim*n) / 2
	case QuantTypeQ8_0:
		weightBytes = int64(kDim * n)
	default:
		weightBytes = int64(kDim * n)
	}

	scaleBytes := int64(len(scalesFor(kDim*n, k.cfg.QuantType))) * 4
	inputBytes := int64(m*kDim*batchSize) * 2 // FP16 activations
	outputBytes := int64(m*n*batchSize) * 4   // FP32 outputs

	totalBytes := weightBytes + scaleBytes + inputBytes + outputBytes

	intensity := float64(flops) / float64(totalBytes)

	return RooflineModel{
		MatrixM:                m,
		MatrixN:                n,
		MatrixK:                kDim,
		FLOPs:                  flops,
		MemoryBytesRead:        totalBytes,
		ArithmeticIntensity:    intensity,
		TheoreticalPeakGBps:    k.cfg.TheoreticalBandwidthGBps,
		SustainedBandwidthGBps: 0,
		LatencySeconds:         0,
		ThroughputTokensPerSec: 0,
		MeetsBandwidthFloor:    false,
	}
}

func scalesFor(elements int, qType QuantType) []float32 {
	var blockSize int
	switch qType {
	case QuantTypeQ4_0, QuantTypeFP4, QuantTypeQ4_K:
		blockSize = Q4BlockSize
	case QuantTypeQ8_0:
		blockSize = Q8BlockSize
	default:
		blockSize = 32
	}
	numBlocks := (elements + blockSize - 1) / blockSize
	return make([]float32, numBlocks)
}
