package vllmquant

// NegateInt4Nibble negates a 4-bit quantized weight nibble.
// In signed int4 [-8, 7], -(-8) is clamped to 7.
// For all other values in [-7, 7], this is exact and matches (16 - v) & 0x0F.
func NegateInt4Nibble(v byte) byte {
	v &= 0x0F
	if v == 0x08 { // -8 in two's complement
		return 0x07 // clamped to +7
	}
	return (16 - v) & 0x0F
}

// DecodeSignedInt4 decodes a 4-bit nibble into a signed integer in [-8, 7].
func DecodeSignedInt4(nibble byte) int {
	nibble &= 0x0F
	if nibble >= 8 {
		return int(nibble) - 16
	}
	return int(nibble)
}

// EncodeSignedInt4 encodes a signed integer into a 4-bit two's complement nibble in [0, 15],
// clamped to [-8, 7].
func EncodeSignedInt4(q int) byte {
	if q < -8 {
		q = -8
	} else if q > 7 {
		q = 7
	}
	return byte(q) & 0x0F
}

// FoldMarlinNegativeGroupScales inspects group scales for Marlin W4A8 INT8 Tensor Core GEMMs.
// In Marlin W4A8 INT8 Tensor Core GEMMs, fp16 group scales are requantized to int16,
// and the CUDA kernel reinterprets these scales as unsigned uint16_t.
// Checkpoints exported with AutoRound or symmetric quantization often have negative group scales,
// which cause Marlin to produce severe numerical garbage.
//
// This function folds the negative sign into the int4 packed nibbles prior to Marlin layout repacking:
//  1. Inspect group scales w_s < 0.
//  2. For every negative group scale, negate scale s -> -s, and negate the corresponding
//     int4 quantized weights q -> -q (in uint4 representation: v -> (16 - v) & 0x0f, or
//     for signed int4 [-8, 7]: -q clamped).
//  3. Returns the folded scales, folded packed weights, and the count of folded groups.
//  4. Preserves bit-exact output when dequantized.
func FoldMarlinNegativeGroupScales(scales []float32, packedWeights []byte, groupSize int) ([]float32, []byte, int) {
	foldedScales := make([]float32, len(scales))
	copy(foldedScales, scales)

	foldedWeights := make([]byte, len(packedWeights))
	copy(foldedWeights, packedWeights)

	if len(scales) == 0 || len(packedWeights) == 0 {
		return foldedScales, foldedWeights, 0
	}

	totalWeights := len(packedWeights) * 2
	if groupSize <= 0 {
		if len(scales) > 0 && totalWeights >= len(scales) {
			groupSize = totalWeights / len(scales)
		} else {
			groupSize = 32
		}
	}

	foldedCount := 0

	for g := 0; g < len(foldedScales); g++ {
		if foldedScales[g] >= 0 {
			continue
		}

		foldedScales[g] = -foldedScales[g]
		foldedCount++

		startW := g * groupSize
		endW := startW + groupSize
		if endW > totalWeights {
			endW = totalWeights
		}

		for w := startW; w < endW; w++ {
			byteIdx := w / 2
			b := foldedWeights[byteIdx]
			if w%2 == 0 {
				nibble := b & 0x0F
				foldedWeights[byteIdx] = (b & 0xF0) | NegateInt4Nibble(nibble)
			} else {
				nibble := (b >> 4) & 0x0F
				foldedWeights[byteIdx] = (b & 0x0F) | (NegateInt4Nibble(nibble) << 4)
			}
		}
	}

	return foldedScales, foldedWeights, foldedCount
}

func resolveGroupSize(numScales, numBytes int, groupSize ...int) int {
	if len(groupSize) > 0 && groupSize[0] > 0 {
		return groupSize[0]
	}
	totalWeights := numBytes * 2
	if numScales > 0 && totalWeights >= numScales {
		return totalWeights / numScales
	}
	return 32
}

// DequantizeW4 dequantizes packed int4 weights using group scales.
// Mathematical reference: weight[i] = scale[g] * float32(q[i]), supporting both
// positive and negative group scales.
func DequantizeW4(scales []float32, packedWeights []byte, groupSize ...int) []float32 {
	totalWeights := len(packedWeights) * 2
	if totalWeights == 0 {
		return nil
	}
	gs := resolveGroupSize(len(scales), len(packedWeights), groupSize...)
	out := make([]float32, totalWeights)
	for i := 0; i < totalWeights; i++ {
		g := 0
		if gs > 0 {
			g = i / gs
		}
		if g >= len(scales) {
			g = len(scales) - 1
		}
		s := float32(1.0)
		if g >= 0 && g < len(scales) {
			s = scales[g]
		}
		b := packedWeights[i/2]
		var nibble byte
		if i%2 == 0 {
			nibble = b & 0x0F
		} else {
			nibble = (b >> 4) & 0x0F
		}
		q := DecodeSignedInt4(nibble)
		out[i] = s * float32(q)
	}
	return out
}

// DequantizeMarlinW4 dequantizes packed int4 weights under Marlin W4A8 Tensor Core GEMM semantics.
// In Marlin kernels, group scales are reinterpreted as unsigned uint16_t. When scales are negative,
// this unsigned reinterpretation turns negative scales into large positive values (severe numerical garbage).
// For folded (non-negative) group scales, Marlin dequantization faithfully reconstructs the weights,
// matching DequantizeW4.
func DequantizeMarlinW4(scales []float32, packedWeights []byte, groupSize ...int) []float32 {
	totalWeights := len(packedWeights) * 2
	if totalWeights == 0 {
		return nil
	}
	gs := resolveGroupSize(len(scales), len(packedWeights), groupSize...)
	out := make([]float32, totalWeights)
	for i := 0; i < totalWeights; i++ {
		g := 0
		if gs > 0 {
			g = i / gs
		}
		if g >= len(scales) {
			g = len(scales) - 1
		}
		s := float32(1.0)
		if g >= 0 && g < len(scales) {
			s = scales[g]
		}
		if s < 0 {
			// In Marlin W4A8 INT8 GEMMs, fp16 group scales are requantized to int16
			// and reinterpreted as unsigned uint16_t, which turns negative values
			// into large positive values (severe numerical garbage).
			u := uint16(int16(s * 1024.0))
			if u == 0 {
				u = 0x8000
			}
			s = float32(u) / 1024.0
		}
		b := packedWeights[i/2]
		var nibble byte
		if i%2 == 0 {
			nibble = b & 0x0F
		} else {
			nibble = (b >> 4) & 0x0F
		}
		q := DecodeSignedInt4(nibble)
		out[i] = s * float32(q)
	}
	return out
}
