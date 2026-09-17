package compute

import (
	"fmt"
	"math"
)

// ---- RDNA (gfx11xx) W4A16 skinny GEMM with packed-4-bit zero-points --------------
//
// This file adds an RDNA-family W4A16 (4-bit weight, 16-bit float activation, FP32
// accumulation) skinny-GEMM decode primitive whose zero-points are stored packed as
// uint4 inside int32 words rather than as one int32 per group. The packed layout cuts
// zero-point traffic eightfold (4 bits per row vs 32) and avoids a per-element VGPR
// blowup on the memory-bound
// gfx1151 decode path: instead of holding one register per zero-point, a row's 4-bit
// zero-point lives at word[n/8] bits 4*(n%8) and is extracted on the fly with a
// shift/mask.
//
// The layout mirrors the vLLM RDNA hybrid W4A16 path
// (vllm .../rdna_hybrid_w4a16.py:80,105-112 and
// csrc/rocm/skinny_gemms_int4.cu:110-117, zp_nibble), adapted into a fak-native,
// host-side reference kernel. It is a pure-Go arithmetic primitive: it holds no device
// handles and makes no throughput claim. The existing per-group
// QuantSpec.ZeroPoint []int32 mode is unchanged; this packed layout is additive and is
// selected only by an explicit opt-in.
//
// Invariant: the packed-uint4 layout requires both the row count (M) and the output
// column count (N) to be divisible by 8, so that every 4-bit zero-point fits inside a
// full 32-bit word and no partial word is implied. When the invariant does not hold the
// caller must fall back to the unpacked per-group reference; W4A16PackedMatVec returns
// ErrW4A16PackedShapeHeld rather than mis-computing.

// ErrW4A16PackedShapeHeld is returned when a shape violates the packed-uint4
// divisibility-by-8 invariant. It is a typed refusal: the caller falls back to the
// existing per-group []int32 reference instead of reading a malformed layout.
type ErrW4A16PackedShapeHeld struct {
	M int
	N int
}

func (e ErrW4A16PackedShapeHeld) Error() string {
	return fmt.Sprintf(
		"w4a16 packed zero-point layout requires M and N divisible by 8, got M=%d N=%d; fall back to the per-group []int32 reference",
		e.M, e.N)
}

// W4A16PackedZeroPointZP holds per-output-row zero-points packed as uint4 nibbles in
// int32 words. Row n occupies bits 4*(n%8) of Words[n/8]; the top nibble of every word
// (bits 28..31) belongs to row n where n%8 == 7. Nibbles are signed 4-bit values in
// [-8, 7], matching the int4 weight range.
type W4A16PackedZeroPointZP struct {
	Words []uint32
}

// PackW4A16ZeroPoints packs one signed 4-bit zero-point per output row into the
// word[n/8]-bits-4*(n%8) layout. It returns ErrW4A16PackedShapeHeld when the row count
// is not divisible by 8 (the packed word would otherwise be underspecified).
//
// Each value is masked to 4 bits (two's complement), so a caller passing -1 stores the
// nibble 0xF and later reads it back as -1.
func PackW4A16ZeroPoints(zeroPoints []int32) (W4A16PackedZeroPointZP, error) {
	n := len(zeroPoints)
	if n == 0 {
		return W4A16PackedZeroPointZP{}, nil
	}
	if n%8 != 0 {
		return W4A16PackedZeroPointZP{}, ErrW4A16PackedShapeHeld{M: n, N: 1}
	}
	words := make([]uint32, n/8)
	for i, zp := range zeroPoints {
		words[i/8] |= (uint32(zp) & 0xF) << (4 * uint(i%8))
	}
	return W4A16PackedZeroPointZP{Words: words}, nil
}

// zpNibble extracts the signed 4-bit zero-point for output row n. It is the Go analogue
// of vLLM's zp_nibble: shift the owning word down to bit 0, mask to 4 bits, then sign
// extend. It performs no bounds check beyond the row's owning word (the caller is
// responsible for the divisibility invariant).
func (z W4A16PackedZeroPointZP) zpNibble(n int) int32 {
	word := z.Words[n/8]
	nib := (word >> uint(4*(n%8))) & 0xF
	if nib&0x8 != 0 {
		return int32(nib) - 16
	}
	return int32(nib)
}

// ZPNibble is the exported, bounds-safe extractor for the packed layout. It validates
// row n against the packed word count and the divisibility invariant before reading, so
// it is safe to call directly from a dispatch decision.
func (z W4A16PackedZeroPointZP) ZPNibble(n int, rows int) (int32, error) {
	if rows%8 != 0 {
		return 0, ErrW4A16PackedShapeHeld{M: rows, N: 1}
	}
	if n < 0 || n >= rows {
		return 0, fmt.Errorf("w4a16 zero-point row %d out of range [0,%d)", n, rows)
	}
	return z.zpNibble(n), nil
}

// W4A16PackedWeights holds a W4A16 weight matrix in the packed-nibble layout used here:
// two signed 4-bit weights per byte, low nibble first, row-major [channels][inDim/2].
// Scales are per output group of scaleGroupSize input elements, mirroring the existing
// grouped 4-bit codec so the two references share a decoding convention.
type W4A16PackedWeights struct {
	Channels       int
	InDim          int
	ScaleGroupSize int
	Scales         []float32 // len == Channels * (InDim/ScaleGroupSize)
	Data           []byte    // packed nibbles, len == Channels * InDim/2
	ZeroPoints     W4A16PackedZeroPointZP
}

// W4A16PackedConfig configures the packed W4A16 weight build.
type W4A16PackedConfig struct {
	ScaleGroupSize int // input elements per scale
}

// BuildW4A16PackedWeights quantizes a row-major [channels, inDim] float32 weight matrix
// into packed int4 weights with a per-group fp32 scale and a per-row signed 4-bit
// zero-point packed into the uint4 layout. It is the asymmetric analogue of
// QuantizeGroupedSym4: the reconstructed value is
//
//	scale[g] * q + zeroPoint[c]
//
// rather than scale[g] * q, so a nonzero per-row zero-point is representable.
func BuildW4A16PackedWeights(weights []float32, channels, inDim int, cfg W4A16PackedConfig) (W4A16PackedWeights, error) {
	if channels <= 0 || inDim <= 0 {
		return W4A16PackedWeights{}, fmt.Errorf("dimensions must be positive: channels=%d, inDim=%d", channels, inDim)
	}
	if len(weights) != channels*inDim {
		return W4A16PackedWeights{}, fmt.Errorf("weights length %d != channels*inDim %d", len(weights), channels*inDim)
	}
	if inDim%2 != 0 {
		return W4A16PackedWeights{}, fmt.Errorf("inDim %d must be even for 4-bit packing", inDim)
	}
	if cfg.ScaleGroupSize <= 0 || cfg.ScaleGroupSize%2 != 0 || inDim%cfg.ScaleGroupSize != 0 {
		return W4A16PackedWeights{}, fmt.Errorf("scale group size %d must be a positive even divisor of inDim %d", cfg.ScaleGroupSize, inDim)
	}
	for _, w := range weights {
		if math.IsNaN(float64(w)) || math.IsInf(float64(w), 0) {
			return W4A16PackedWeights{}, fmt.Errorf("non-finite weight in input matrix")
		}
	}

	groupsPerRow := inDim / cfg.ScaleGroupSize
	out := W4A16PackedWeights{
		Channels:       channels,
		InDim:          inDim,
		ScaleGroupSize: cfg.ScaleGroupSize,
		Scales:         make([]float32, channels*groupsPerRow),
		Data:           make([]byte, channels*inDim/2),
	}

	zps := make([]int32, channels)
	for c := 0; c < channels; c++ {
		rowOffset := c * inDim
		rowByteOffset := c * inDim / 2

		// One zero-point per output row, chosen so the row's minimum maps to the
		// signed int4 floor (-8). This keeps the q range centered on the row's
		// observed weight distribution, which is the motivating asymmetry.
		var rowMin float32
		for i := 0; i < inDim; i++ {
			v := weights[rowOffset+i]
			if i == 0 || v < rowMin {
				rowMin = v
			}
		}
		zp := int32(0)
		if rowMin < 0 {
			if scaled := math.Round(float64(rowMin)); scaled < -8 {
				zp = -8
			} else {
				zp = int32(scaled)
			}
		}
		zps[c] = zp

		for g := 0; g < groupsPerRow; g++ {
			groupOffset := rowOffset + g*cfg.ScaleGroupSize
			var maxAbs float32
			for i := 0; i < cfg.ScaleGroupSize; i++ {
				v := weights[groupOffset+i] - float32(zp)
				if a := float32(math.Abs(float64(v))); a > maxAbs {
					maxAbs = a
				}
			}
			scale := float32(1.0)
			if maxAbs > 0 {
				scale = maxAbs / 7.0
			}
			out.Scales[c*groupsPerRow+g] = scale
			inv := 1.0 / scale

			for i := 0; i < cfg.ScaleGroupSize; i += 2 {
				q0 := clampInt4(int(math.Round(float64((weights[groupOffset+i] - float32(zp)) * inv))))
				q1 := clampInt4(int(math.Round(float64((weights[groupOffset+i+1] - float32(zp)) * inv))))
				packed := byte(q0&0xF) | byte((q1&0xF)<<4)
				out.Data[rowByteOffset+(g*cfg.ScaleGroupSize+i)/2] = packed
			}
		}
	}

	packed, err := PackW4A16ZeroPoints(zps)
	if err != nil {
		return W4A16PackedWeights{}, err
	}
	out.ZeroPoints = packed
	return out, nil
}

func clampInt4(v int) int {
	if v < -8 {
		return -8
	}
	if v > 7 {
		return 7
	}
	return v
}

// unpackW4A16Nibble returns the signed 4-bit weight at a byte, low nibble when hi is
// false and high nibble when hi is true.
func unpackW4A16Nibble(b byte, hi bool) int {
	var nib byte
	if hi {
		nib = (b >> 4) & 0xF
	} else {
		nib = b & 0xF
	}
	if nib&0x8 != 0 {
		return int(nib) - 16
	}
	return int(nib)
}

// W4A16PackedMatVec computes y = W * x for a packed W4A16 matrix, reading each output
// row's signed 4-bit zero-point from the packed uint4 layout via a shift/mask. The
// accumulation is float64 for a deterministic host reference, and the reconstruction is
// scale[g]*q + zeroPoint[c].
//
// It returns ErrW4A16PackedShapeHeld when either the input dimension or the output row
// count is not divisible by 8; the caller falls back to the per-group []int32 path.
func (t W4A16PackedWeights) W4A16PackedMatVec(x []float32) ([]float32, error) {
	if t.Channels%8 != 0 || t.InDim%8 != 0 {
		return nil, ErrW4A16PackedShapeHeld{M: t.Channels, N: t.InDim}
	}
	if len(x) != t.InDim {
		return nil, fmt.Errorf("vector length %d != inDim %d", len(x), t.InDim)
	}
	if len(t.ZeroPoints.Words) != t.Channels/8 {
		return nil, fmt.Errorf("packed zero-point word count %d != channels/8 %d", len(t.ZeroPoints.Words), t.Channels/8)
	}

	groupsPerRow := t.InDim / t.ScaleGroupSize
	var xSum float64
	for _, v := range x {
		xSum += float64(v)
	}

	y := make([]float32, t.Channels)
	for c := 0; c < t.Channels; c++ {
		zp := t.ZeroPoints.zpNibble(c)
		rowByteOffset := c * t.InDim / 2
		var rowSum float64

		for g := 0; g < groupsPerRow; g++ {
			scale := float64(t.Scales[c*groupsPerRow+g])
			elemBase := g * t.ScaleGroupSize
			var groupSum float64
			for i := 0; i < t.ScaleGroupSize; i += 2 {
				b := t.Data[rowByteOffset+(elemBase+i)/2]
				groupSum += float64(unpackW4A16Nibble(b, false)) * float64(x[elemBase+i])
				groupSum += float64(unpackW4A16Nibble(b, true)) * float64(x[elemBase+i+1])
			}
			rowSum += groupSum * scale
		}
		// The reconstructed value is scale*q + zp, so the zero-point contributes
		// zp * sum(x) once per output row.
		y[c] = float32(rowSum + float64(zp)*xSum)
	}
	return y, nil
}

// W4A16PackedReferenceMatVec is the unpacked per-group reference that the packed path
// must match. It reads the same packed nibbles and the same per-row zero-points, but
// obtains the zero-point through the per-group []int32 representation rather than the
// packed uint4 words, so the parity test compares two independent readings of the same
// weights.
func (t W4A16PackedWeights) W4A16PackedReferenceMatVec(x []float32, zeroPoints []int32) ([]float32, error) {
	if len(x) != t.InDim {
		return nil, fmt.Errorf("vector length %d != inDim %d", len(x), t.InDim)
	}
	if len(zeroPoints) != t.Channels {
		return nil, fmt.Errorf("zero-point length %d != channels %d", len(zeroPoints), t.Channels)
	}
	groupsPerRow := t.InDim / t.ScaleGroupSize
	var xSum float64
	for _, v := range x {
		xSum += float64(v)
	}

	y := make([]float32, t.Channels)
	for c := 0; c < t.Channels; c++ {
		zp := float64(zeroPoints[c])
		rowByteOffset := c * t.InDim / 2
		var rowSum float64
		for g := 0; g < groupsPerRow; g++ {
			scale := float64(t.Scales[c*groupsPerRow+g])
			elemBase := g * t.ScaleGroupSize
			var groupSum float64
			for i := 0; i < t.ScaleGroupSize; i += 2 {
				b := t.Data[rowByteOffset+(elemBase+i)/2]
				groupSum += float64(unpackW4A16Nibble(b, false)) * float64(x[elemBase+i])
				groupSum += float64(unpackW4A16Nibble(b, true)) * float64(x[elemBase+i+1])
			}
			rowSum += groupSum * scale
		}
		y[c] = float32(rowSum + zp*xSum)
	}
	return y, nil
}

// W4A16ZeroPointByteTraffic reports the per-row zero-point storage cost of the packed
// layout against the unpacked []int32 layout for a given row count. It is a pure byte
// accounting helper, not a measured throughput claim: it exists so a caller can record
// the 8x zero-point traffic reduction the packed layout implies (4 bits per row in
// uint4 words vs one 32-bit int32 per row).
func W4A16ZeroPointByteTraffic(rows int) (packedBytes, unpackedBytes int, ok bool) {
	if rows <= 0 || rows%8 != 0 {
		return 0, 0, false
	}
	return rows / 2, rows * 4, true
}
