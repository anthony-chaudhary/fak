package compute

import (
	"errors"
	"fmt"
	"math"
)

// CoopMatPrecision defines the element precision for cooperative matrix arithmetic.
type CoopMatPrecision string

const (
	CoopMatFP16 CoopMatPrecision = "fp16"
	CoopMatBF16 CoopMatPrecision = "bf16"
	CoopMatINT8 CoopMatPrecision = "int8"
	CoopMatFP8  CoopMatPrecision = "fp8"
	CoopMatFP32 CoopMatPrecision = "fp32"
)

// CoopMatScope defines the execution scope for cooperative matrix operations.
type CoopMatScope string

const (
	CoopMatScopeSubgroup  CoopMatScope = "subgroup"
	CoopMatScopeWorkgroup CoopMatScope = "workgroup"
)

// CoopMatMatrixUse specifies the role of a cooperative matrix in GEMM.
type CoopMatMatrixUse string

const (
	CoopMatMatrixA           CoopMatMatrixUse = "matrix_a"
	CoopMatMatrixB           CoopMatMatrixUse = "matrix_b"
	CoopMatMatrixAccumulator CoopMatMatrixUse = "matrix_accumulator"
)

// CoopMatAttentionTile configures tile dimensions and memory layout for
// cooperative matrix-accelerated FlashAttention computation.
type CoopMatAttentionTile struct {
	Br            int              `json:"br"`              // Query row tile size
	Bc            int              `json:"bc"`              // Key/Value column tile size
	Bk            int              `json:"bk"`              // Head dimension reduction step size
	SubgroupSize  int              `json:"subgroup_size"`   // SIMD width (e.g. 32 for Wave32, 64 for Wave64)
	Precision     CoopMatPrecision `json:"precision"`       // Tile arithmetic precision
	LDSBytes      int              `json:"lds_bytes"`       // Total shared memory requirement
	PaddedStride  int              `json:"padded_stride"`   // Bank-conflict-free padded LDS stride
	WavesPerGroup int              `json:"waves_per_group"` // Number of wavefronts / subgroups per workgroup
}

// TuneCoopMatAttention calculates the optimal cooperative matrix tile configuration
// for FlashAttention given the head dimension and hardware shared memory (SRAM) budget.
func TuneCoopMatAttention(headDim int, sramBytes int, subgroupSize int) (CoopMatAttentionTile, error) {
	if headDim <= 0 {
		return CoopMatAttentionTile{}, errors.New("compute: headDim must be positive")
	}
	if sramBytes <= 0 {
		sramBytes = 65536 // default 64 KiB LDS
	}
	if subgroupSize <= 0 {
		subgroupSize = 32 // default Wave32
	}

	// Cooperative matrix WMMA hardware requires tile dimensions to be multiples of 16.
	// We select Br and Bc to balance register pressure and shared memory limits.
	br := 64
	bc := 64
	bk := 16
	if headDim%32 == 0 {
		bk = 32
	}

	// Ensure query tile (Br x headDim) + key/value tile (Bc x headDim) fits in SRAM with Pad-2 alignment.
	paddedStride := LDSBankPad2Stride(headDim)
	ldsReq := (br*paddedStride + 2*bc*paddedStride) * 4

	for ldsReq > sramBytes && bc > 16 {
		bc /= 2
		ldsReq = (br*paddedStride + 2*bc*paddedStride) * 4
	}
	for ldsReq > sramBytes && br > 16 {
		br /= 2
		ldsReq = (br*paddedStride + 2*bc*paddedStride) * 4
	}

	if ldsReq > sramBytes {
		return CoopMatAttentionTile{}, fmt.Errorf("compute: unable to fit cooperative matrix tiles in SRAM (need %d bytes, limit %d)", ldsReq, sramBytes)
	}

	wavesPerGroup := (br / 16) * (bc / 16)
	if wavesPerGroup < 1 {
		wavesPerGroup = 1
	}
	if wavesPerGroup > 16 {
		wavesPerGroup = 16
	}

	return CoopMatAttentionTile{
		Br:            br,
		Bc:            bc,
		Bk:            bk,
		SubgroupSize:  subgroupSize,
		Precision:     CoopMatFP16,
		LDSBytes:      ldsReq,
		PaddedStride:  paddedStride,
		WavesPerGroup: wavesPerGroup,
	}, nil
}

// SimulateCoopMatGEMM simulates cooperative matrix tile multiplication C = A * B
// using 16x16 sub-matrix primitives with accumulator registers.
func SimulateCoopMatGEMM(A, B []float32, M, N, K int) []float32 {
	C := make([]float32, M*N)
	const tileM, tileN, tileK = 16, 16, 16

	for i0 := 0; i0 < M; i0 += tileM {
		for j0 := 0; j0 < N; j0 += tileN {
			// Subgroup cooperative accumulator tile
			acc := make([]float32, tileM*tileN)

			for k0 := 0; k0 < K; k0 += tileK {
				for r := 0; r < tileM; r++ {
					for c := 0; c < tileN; c++ {
						var dot float32
						for kk := 0; kk < tileK; kk++ {
							gRow := i0 + r
							gCol := j0 + c
							gK := k0 + kk
							var aVal, bVal float32
							if gRow < M && gK < K {
								aVal = A[gRow*K+gK]
							}
							if gK < K && gCol < N {
								bVal = B[gK*N+gCol]
							}
							dot += aVal * bVal
						}
						acc[r*tileN+c] += dot
					}
				}
			}

			// Store cooperative matrix accumulator to output
			for r := 0; r < tileM; r++ {
				for c := 0; c < tileN; c++ {
					gRow := i0 + r
					gCol := j0 + c
					if gRow < M && gCol < N {
						C[gRow*N+gCol] = acc[r*tileN+c]
					}
				}
			}
		}
	}
	return C
}

// SimulateCoopMatAttentionTile simulates cooperative matrix attention block multiplication
// computing S_tile = Q_tile * K_tile^T * scale, applying online softmax normalization,
// and accumulating into V_tile.
func SimulateCoopMatAttentionTile(qTile, kTile, vTile []float32, Br, Bc, headDim int, scale float32) []float32 {
	out := make([]float32, Br*headDim)

	for r := 0; r < Br; r++ {
		qRow := qTile[r*headDim : (r+1)*headDim]

		// 1. Cooperative score computation S = Q * K^T * scale
		scores := make([]float32, Bc)
		maxScore := float32(-math.MaxFloat32)
		for c := 0; c < Bc; c++ {
			kRow := kTile[c*headDim : (c+1)*headDim]
			var dot float32
			for d := 0; d < headDim; d++ {
				dot += qRow[d] * kRow[d]
			}
			s := dot * scale
			scores[c] = s
			if s > maxScore {
				maxScore = s
			}
		}

		// 2. Online softmax weight scaling
		var sumExp float32
		for c := 0; c < Bc; c++ {
			p := float32(math.Exp(float64(scores[c] - maxScore)))
			scores[c] = p
			sumExp += p
		}

		invSum := float32(0.0)
		if sumExp > 0 {
			invSum = 1.0 / sumExp
		}

		// 3. Accumulate weighted values into output tile
		oRow := out[r*headDim : (r+1)*headDim]
		for c := 0; c < Bc; c++ {
			w := scores[c] * invSum
			vRow := vTile[c*headDim : (c+1)*headDim]
			for d := 0; d < headDim; d++ {
				oRow[d] += w * vRow[d]
			}
		}
	}

	return out
}
