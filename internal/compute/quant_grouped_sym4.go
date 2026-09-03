package compute

import (
	"encoding/binary"
	"fmt"
	"math"
)

// GroupedSym4CodecConfig defines parameters for per-channel grouped symmetric 4-bit quantization.
type GroupedSym4CodecConfig struct {
	GroupSize int `json:"group_size"` // 8, 16, or 32
}

// GroupedSym4Receipt records quantization fidelity and compression metrics.
type GroupedSym4Receipt struct {
	Channels         int     `json:"channels"`
	InDim            int     `json:"in_dim"`
	GroupSize        int     `json:"group_size"`
	OriginalBytes    int     `json:"original_bytes"`
	CompressedBytes  int     `json:"compressed_bytes"`
	CompressionRatio float64 `json:"compression_ratio"`
	MaxAbsoluteError float32 `json:"max_absolute_error"`
	MeanSquareError  float64 `json:"mean_square_error"`
	CosineSimilarity float64 `json:"cosine_similarity"`
	SQNRdB           float64 `json:"sqnr_db"`
}

// GroupedSym4Tensor stores compressed weights in per-channel grouped symmetric 4-bit format.
type GroupedSym4Tensor struct {
	Channels  int    `json:"channels"`
	InDim     int    `json:"in_dim"`
	GroupSize int    `json:"group_size"`
	Data      []byte `json:"data"` // packed: [channels][groups_per_row](2 bytes fp16 scale + groupSize/2 bytes data)
}

// BytesPerGroup returns the packed storage required for one group.
func (c GroupedSym4CodecConfig) BytesPerGroup() int {
	return 2 + c.GroupSize/2
}

// float32ToFP16 converts float32 to IEEE 754 binary16 bits.
func float32ToFP16(f float32) uint16 {
	bits := math.Float32bits(f)
	sign := uint16((bits >> 31) & 0x1)
	exp := int((bits >> 23) & 0xFF)
	mant := bits & 0x7FFFFF

	if exp == 255 {
		if mant != 0 {
			return (sign << 15) | 0x7E00 // NaN
		}
		return (sign << 15) | 0x7C00 // Inf
	}

	exp16 := exp - 127 + 15
	if exp16 >= 31 {
		return (sign << 15) | 0x7C00 // overflow to Inf
	}
	if exp16 <= 0 {
		if exp16 < -10 {
			return sign << 15 // underflow to zero
		}
		mant |= 0x800000
		shift := uint(14 - exp16)
		mant16 := uint16(mant >> shift)
		return (sign << 15) | mant16
	}

	mant16 := uint16(mant >> 13)
	return (sign << 15) | (uint16(exp16) << 10) | mant16
}

// fp16ToFloat32 converts IEEE 754 binary16 bits to float32.
func fp16ToFloat32(h uint16) float32 {
	sign := uint32((h >> 15) & 0x1)
	exp16 := int((h >> 10) & 0x1F)
	mant16 := uint32(h & 0x3FF)

	if exp16 == 31 {
		if mant16 != 0 {
			return float32(math.NaN())
		}
		if sign != 0 {
			return float32(math.Inf(-1))
		}
		return float32(math.Inf(1))
	}
	if exp16 == 0 {
		if mant16 == 0 {
			if sign != 0 {
				return float32(math.Copysign(0, -1))
			}
			return 0.0
		}
		// subnormal
		f := float32(mant16) / 1024.0 * float32(math.Pow(2, -14))
		if sign != 0 {
			return -f
		}
		return f
	}

	exp32 := uint32(exp16 - 15 + 127)
	mant32 := mant16 << 13
	bits := (sign << 31) | (exp32 << 23) | mant32
	return math.Float32frombits(bits)
}

// QuantizeGroupedSym4 encodes a row-major [channels, inDim] float32 weight matrix into grouped symmetric 4-bit.
func QuantizeGroupedSym4(weights []float32, channels, inDim int, cfg GroupedSym4CodecConfig) (GroupedSym4Tensor, error) {
	if cfg.GroupSize != 8 && cfg.GroupSize != 16 && cfg.GroupSize != 32 {
		return GroupedSym4Tensor{}, fmt.Errorf("group size must be 8, 16, or 32, got %d", cfg.GroupSize)
	}
	if channels <= 0 || inDim <= 0 {
		return GroupedSym4Tensor{}, fmt.Errorf("dimensions must be positive: channels=%d, inDim=%d", channels, inDim)
	}
	if inDim%cfg.GroupSize != 0 {
		return GroupedSym4Tensor{}, fmt.Errorf("inDim %d must be divisible by group size %d", inDim, cfg.GroupSize)
	}
	if len(weights) != channels*inDim {
		return GroupedSym4Tensor{}, fmt.Errorf("weights length %d != channels*inDim %d", len(weights), channels*inDim)
	}

	groupsPerRow := inDim / cfg.GroupSize
	bytesPerGroup := cfg.BytesPerGroup()
	totalBytes := channels * groupsPerRow * bytesPerGroup
	data := make([]byte, totalBytes)

	for c := 0; c < channels; c++ {
		rowOffset := c * inDim
		rowBytesOffset := c * groupsPerRow * bytesPerGroup

		for g := 0; g < groupsPerRow; g++ {
			groupElemOffset := rowOffset + g*cfg.GroupSize
			groupByteOffset := rowBytesOffset + g*bytesPerGroup

			// Find max absolute value in group
			var maxAbs float32
			for i := 0; i < cfg.GroupSize; i++ {
				v := weights[groupElemOffset+i]
				if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
					return GroupedSym4Tensor{}, fmt.Errorf("non-finite weight at channel %d, index %d", c, g*cfg.GroupSize+i)
				}
				absV := float32(math.Abs(float64(v)))
				if absV > maxAbs {
					maxAbs = absV
				}
			}

			// Symmetric 4-bit range is [-7, 7] (or [-8, 7]), using scale = maxAbs / 7.0
			var scale float32 = 1.0
			if maxAbs > 0 {
				scale = maxAbs / 7.0
			}

			// Store fp16 scale
			scaleFP16 := float32ToFP16(scale)
			binary.LittleEndian.PutUint16(data[groupByteOffset:groupByteOffset+2], scaleFP16)
			reconScale := fp16ToFloat32(scaleFP16)
			if reconScale == 0 {
				reconScale = 1e-7
			}

			invScale := 1.0 / reconScale
			nibbleOffset := groupByteOffset + 2

			// Quantize and pack pairs of 4-bit values
			for i := 0; i < cfg.GroupSize; i += 2 {
				v0 := weights[groupElemOffset+i]
				v1 := weights[groupElemOffset+i+1]

				q0 := int(math.Round(float64(v0 * invScale)))
				if q0 < -8 {
					q0 = -8
				} else if q0 > 7 {
					q0 = 7
				}

				q1 := int(math.Round(float64(v1 * invScale)))
				if q1 < -8 {
					q1 = -8
				} else if q1 > 7 {
					q1 = 7
				}

				// Pack two signed 4-bit integers into one byte
				u0 := byte(q0 & 0x0F)
				u1 := byte(q1 & 0x0F)
				data[nibbleOffset+i/2] = u0 | (u1 << 4)
			}
		}
	}

	return GroupedSym4Tensor{
		Channels:  channels,
		InDim:     inDim,
		GroupSize: cfg.GroupSize,
		Data:      data,
	}, nil
}

// Dequantize unpacks the grouped symmetric 4-bit tensor back into float32.
func (t GroupedSym4Tensor) Dequantize() ([]float32, error) {
	totalElems := t.Channels * t.InDim
	out := make([]float32, totalElems)
	groupsPerRow := t.InDim / t.GroupSize
	bytesPerGroup := 2 + t.GroupSize/2

	for c := 0; c < t.Channels; c++ {
		rowOffset := c * t.InDim
		rowBytesOffset := c * groupsPerRow * bytesPerGroup

		for g := 0; g < groupsPerRow; g++ {
			groupElemOffset := rowOffset + g*t.GroupSize
			groupByteOffset := rowBytesOffset + g*bytesPerGroup

			scaleFP16 := binary.LittleEndian.Uint16(t.Data[groupByteOffset : groupByteOffset+2])
			scale := fp16ToFloat32(scaleFP16)
			nibbleOffset := groupByteOffset + 2

			for i := 0; i < t.GroupSize; i += 2 {
				b := t.Data[nibbleOffset+i/2]
				u0 := b & 0x0F
				u1 := (b >> 4) & 0x0F

				// Sign extend 4-bit signed int
				var q0, q1 int
				if (u0 & 0x08) != 0 {
					q0 = int(u0) - 16
				} else {
					q0 = int(u0)
				}
				if (u1 & 0x08) != 0 {
					q1 = int(u1) - 16
				} else {
					q1 = int(u1)
				}

				out[groupElemOffset+i] = float32(q0) * scale
				out[groupElemOffset+i+1] = float32(q1) * scale
			}
		}
	}

	return out, nil
}

// MatVec computes y = W * x directly from the packed grouped 4-bit representation in FP32 accumulation.
func (t GroupedSym4Tensor) MatVec(x []float32) ([]float32, error) {
	if len(x) != t.InDim {
		return nil, fmt.Errorf("vector length %d != inDim %d", len(x), t.InDim)
	}

	y := make([]float32, t.Channels)
	groupsPerRow := t.InDim / t.GroupSize
	bytesPerGroup := 2 + t.GroupSize/2

	for c := 0; c < t.Channels; c++ {
		var rowSum float64
		rowBytesOffset := c * groupsPerRow * bytesPerGroup

		for g := 0; g < groupsPerRow; g++ {
			elemBase := g * t.GroupSize
			groupByteOffset := rowBytesOffset + g*bytesPerGroup

			scaleFP16 := binary.LittleEndian.Uint16(t.Data[groupByteOffset : groupByteOffset+2])
			scale := float64(fp16ToFloat32(scaleFP16))
			nibbleOffset := groupByteOffset + 2

			var groupDot float64
			for i := 0; i < t.GroupSize; i += 2 {
				b := t.Data[nibbleOffset+i/2]
				u0 := b & 0x0F
				u1 := (b >> 4) & 0x0F

				var q0, q1 int
				if (u0 & 0x08) != 0 {
					q0 = int(u0) - 16
				} else {
					q0 = int(u0)
				}
				if (u1 & 0x08) != 0 {
					q1 = int(u1) - 16
				} else {
					q1 = int(u1)
				}

				groupDot += float64(q0) * float64(x[elemBase+i])
				groupDot += float64(q1) * float64(x[elemBase+i+1])
			}

			rowSum += groupDot * scale
		}
		y[c] = float32(rowSum)
	}

	return y, nil
}

// EvaluateCodecQuality computes error and fidelity statistics between original and dequantized weights.
func EvaluateCodecQuality(original, dequantized []float32, channels, inDim, groupSize int) GroupedSym4Receipt {
	var maxAbsErr float32
	var sse, sumOrigSq float64
	var dot, normOrig, normDequant float64

	for i := range original {
		diff := float32(math.Abs(float64(original[i] - dequantized[i])))
		if diff > maxAbsErr {
			maxAbsErr = diff
		}
		d64 := float64(diff)
		sse += d64 * d64

		o64 := float64(original[i])
		q64 := float64(dequantized[i])
		sumOrigSq += o64 * o64

		dot += o64 * q64
		normOrig += o64 * o64
		normDequant += q64 * q64
	}

	mse := sse / float64(len(original))
	cosine := 0.0
	if normOrig > 0 && normDequant > 0 {
		cosine = dot / (math.Sqrt(normOrig) * math.Sqrt(normDequant))
	}

	sqnr := 0.0
	if sse > 0 && sumOrigSq > 0 {
		sqnr = 10.0 * math.Log10(sumOrigSq/sse)
	}

	origBytes := len(original) * 4
	bytesPerGroup := 2 + groupSize/2
	groupsTotal := (channels * inDim) / groupSize
	compBytes := groupsTotal * bytesPerGroup

	return GroupedSym4Receipt{
		Channels:         channels,
		InDim:            inDim,
		GroupSize:        groupSize,
		OriginalBytes:    origBytes,
		CompressedBytes:  compBytes,
		CompressionRatio: float64(origBytes) / float64(compBytes),
		MaxAbsoluteError: maxAbsErr,
		MeanSquareError:  mse,
		CosineSimilarity: cosine,
		SQNRdB:           sqnr,
	}
}
