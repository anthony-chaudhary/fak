package model

import (
	"encoding/binary"
	"fmt"
	"math"
)

// rocmfp4.go — ROCmFP4 hardware-aligned block quantization layout and reference codec (#10730).
//
// Worldview & Origin:
// Borrowed from julianmb/q38rocm (Qwen 3.8 27B on AMD Strix Halo / RDNA 3.5 / gfx1151).
// In modern unified-memory APUs (such as Ryzen AI Max+ 395 with 128 GB LPDDR5X), memory bandwidth
// during token generation is the primary bottleneck. ROCmFP4 achieves 4.26 bits per weight (effective
// model footprint: 13.55 GB for 27B params) while matching RDNA 3/3.5 vector register strides
// (32 elements per half-wave in Wave64, or full wave in Wave32).
//
// Hardware Alignment Invariant:
// RDNA 3 / 3.5 compute units execute SIMD instructions across 32-lane vector register strides.
// Each block holds exactly 32 FP4 (E2M1) elements and a single 16-bit float (IEEE 754 binary16)
// scale factor. By packing 32 4-bit elements (16 bytes) with an FP16 scale (2 bytes), the total
// block size is 18 bytes.
//
// Benefits:
//  1. Single broadcast scale per 32 lanes: All 32 ALUs in the half-wave share one scale factor loaded
//     into a scalar or broadcast register, eliminating cross-lane shuffle stalls.
//  2. Zero dequantization serialization: 4-bit weights unpack directly into ALUs/WMMA inputs in a
//     single pass.
//  3. Raw block storage: 18 bytes * 8 bits / 32 elements = 4.50 bpw. Net model deployment with
//     unquantized embeddings/norms/heads reaches 4.26 bpw.

const (
	// ROCmFP4BlockSize is the number of elements per quantization block (32 elements),
	// aligning 1:1 with RDNA 3 / 3.5 vector register half-wave strides.
	ROCmFP4BlockSize = 32

	// ROCmFP4BlockBytes is the packed storage size of one block:
	// 2 bytes (FP16 scale factor) + 16 bytes (32 4-bit packed elements) = 18 bytes.
	ROCmFP4BlockBytes = 18

	// ROCmFP4RawBlockBPW is the raw bits per weight of a block (18*8 / 32 = 4.5 bits).
	ROCmFP4RawBlockBPW = 4.5

	// ROCmFP4BitsPerWeight is the effective end-to-end model bits per weight in Qwen 3.8 27B
	// deployment (4.26 bpw, ~13.55 GB total payload).
	ROCmFP4BitsPerWeight = 4.26

	// Q4_0_ROCMFP4_FAST is the canonical layout identifier matching q38rocm / ROCmFPX.
	Q4_0_ROCMFP4_FAST = "Q4_0_ROCMFP4_FAST"

	// ROCmFP4LayoutName is an alias for the format identifier.
	ROCmFP4LayoutName = Q4_0_ROCMFP4_FAST
)

// rocmfp4E2M1Values maps each 4-bit code (0..15) to its OCP / IEEE FP4 E2M1 value:
// 1 sign bit, 2 exponent bits (bias 1), 1 mantissa bit.
// Positive codes 0..7: 0, 0.5, 1, 1.5, 2, 3, 4, 6.
// Negative codes 8..15: -0, -0.5, -1, -1.5, -2, -3, -4, -6.
var rocmfp4E2M1Values = [16]float32{
	0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
	float32(math.Copysign(0, -1)), -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
}

// ROCmFP4Block represents one packed 32-element quantization block.
type ROCmFP4Block struct {
	Scale uint16   // IEEE 754 binary16 scale factor (little-endian)
	Data  [16]byte // 32 packed 4-bit E2M1 elements (low nibble: even index, high nibble: odd index)
}

// ROCmFP4Tensor holds a 2D matrix or 1D vector quantized in ROCmFP4 format.
type ROCmFP4Tensor struct {
	Rows        int
	Cols        int
	NumElements int
	Blocks      []ROCmFP4Block
}

// Float32ToFP16 converts a float32 to IEEE 754 binary16 bits using
// standard Round-to-Nearest, Ties-to-Even (RTNE).
func Float32ToFP16(f float32) uint16 {
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
		mantVal := mant | 0x800000
		shift := uint(14 - exp16)
		mant16 := uint16(mantVal >> shift)
		rem := mantVal & ((uint32(1) << shift) - 1)
		half := uint32(1) << (shift - 1)
		if rem > half || (rem == half && (mant16&1 != 0)) {
			mant16++
		}
		if mant16 == 0x400 {
			return (sign << 15) | (1 << 10) // rounded into normal range: exp=1, mant=0
		}
		return (sign << 15) | mant16
	}

	rem := mant & 0x1fff
	mant16 := uint16(mant >> 13)
	if rem > 0x1000 || (rem == 0x1000 && (mant16&1 != 0)) {
		mant16++
		if mant16 == 0x400 {
			exp16++
			mant16 = 0
		}
	}
	if exp16 >= 31 {
		return (sign << 15) | 0x7C00 // overflow to Inf
	}
	return (sign << 15) | (uint16(exp16) << 10) | mant16
}

// FP16ToFloat32 converts IEEE 754 binary16 bits to float32.
func FP16ToFloat32(h uint16) float32 {
	return math.Float32frombits(F16BitsToF32Bits(h))
}

// quantizeE2M1 finds the closest 4-bit E2M1 code (0..15) for normalized value v = x / scale.
func quantizeE2M1(v float32) byte {
	var sign byte
	target := v
	if math.Signbit(float64(v)) {
		sign = 0x8
		target = -v
	}

	var code byte
	switch {
	case target < 0.25:
		code = 0 // 0.0
	case target < 0.75:
		code = 1 // 0.5
	case target < 1.25:
		code = 2 // 1.0
	case target < 1.75:
		code = 3 // 1.5
	case target < 2.5:
		code = 4 // 2.0
	case target < 3.5:
		code = 5 // 3.0
	case target < 5.0:
		code = 6 // 4.0
	default:
		code = 7 // 6.0
	}
	return sign | code
}

// QuantizeROCmFP4Block quantizes exactly 32 float32 elements into a single ROCmFP4Block.
func QuantizeROCmFP4Block(src []float32) ROCmFP4Block {
	if len(src) != ROCmFP4BlockSize {
		panic(fmt.Sprintf("rocmfp4: block requires %d elements, got %d", ROCmFP4BlockSize, len(src)))
	}

	var maxAbs float32
	for _, v := range src {
		a := v
		if a < 0 {
			a = -a
		}
		if a > maxAbs {
			maxAbs = a
		}
	}

	if maxAbs == 0 {
		return ROCmFP4Block{}
	}

	// Max E2M1 magnitude is 6.0.
	d := maxAbs / 6.0
	scaleFP16 := Float32ToFP16(d)
	scale := FP16ToFloat32(scaleFP16)
	if scale == 0 {
		return ROCmFP4Block{}
	}

	invScale := 1.0 / scale
	var blk ROCmFP4Block
	blk.Scale = scaleFP16

	for j := 0; j < 16; j++ {
		c0 := quantizeE2M1(src[2*j] * invScale)
		c1 := quantizeE2M1(src[2*j+1] * invScale)
		blk.Data[j] = c0 | (c1 << 4)
	}

	return blk
}

// DequantizeROCmFP4Block dequantizes one ROCmFP4Block into exactly 32 float32 values in dst.
func DequantizeROCmFP4Block(b ROCmFP4Block, dst []float32) {
	if len(dst) < ROCmFP4BlockSize {
		panic(fmt.Sprintf("rocmfp4: destination slice must have at least %d elements, got %d", ROCmFP4BlockSize, len(dst)))
	}
	scale := FP16ToFloat32(b.Scale)
	for j := 0; j < 16; j++ {
		packed := b.Data[j]
		c0 := packed & 0x0f
		c1 := packed >> 4
		dst[2*j] = scale * rocmfp4E2M1Values[c0]
		dst[2*j+1] = scale * rocmfp4E2M1Values[c1]
	}
}

// UnpackROCmFP4Block unpacks a block into its decoded scale, the 32 4-bit codes, and the 32 float values.
func UnpackROCmFP4Block(b ROCmFP4Block) (scale float32, codes [32]byte, values [32]float32) {
	scale = FP16ToFloat32(b.Scale)
	for j := 0; j < 16; j++ {
		packed := b.Data[j]
		c0 := packed & 0x0f
		c1 := packed >> 4
		codes[2*j] = c0
		codes[2*j+1] = c1
		values[2*j] = scale * rocmfp4E2M1Values[c0]
		values[2*j+1] = scale * rocmfp4E2M1Values[c1]
	}
	return scale, codes, values
}

// UnpackROCmFP4Bytes unpacks a raw 18-byte block slice into scale, codes, and values.
func UnpackROCmFP4Bytes(blockBytes []byte) (scale float32, codes [32]byte, values [32]float32, err error) {
	if len(blockBytes) != ROCmFP4BlockBytes {
		return 0, codes, values, fmt.Errorf("rocmfp4: block requires %d bytes, got %d", ROCmFP4BlockBytes, len(blockBytes))
	}
	var b ROCmFP4Block
	b.Scale = binary.LittleEndian.Uint16(blockBytes[0:2])
	copy(b.Data[:], blockBytes[2:18])
	scale, codes, values = UnpackROCmFP4Block(b)
	return scale, codes, values, nil
}

// ValidateROCmFP4Dimensions checks that matrix rows and columns are positive and that
// columns are a multiple of 32 (matching RDNA 3/3.5 vector register strides).
func ValidateROCmFP4Dimensions(rows, cols int) error {
	if rows <= 0 || cols <= 0 {
		return fmt.Errorf("rocmfp4: invalid dimensions [%d, %d]: both dimensions must be positive", rows, cols)
	}
	if cols%ROCmFP4BlockSize != 0 {
		return fmt.Errorf("rocmfp4: column dimension %d is not a multiple of block size %d (violates RDNA 3/3.5 half-wave vector stride)",
			cols, ROCmFP4BlockSize)
	}
	return nil
}

// ValidateROCmFP4Layout checks that numElements is a multiple of 32, matches rawBytes length,
// and that all block scales are finite valid numbers.
func ValidateROCmFP4Layout(numElements int, rawBytes []byte) error {
	if numElements <= 0 {
		return fmt.Errorf("rocmfp4: numElements must be positive, got %d", numElements)
	}
	if numElements%ROCmFP4BlockSize != 0 {
		return fmt.Errorf("rocmfp4: numElements %d is not a multiple of block size %d", numElements, ROCmFP4BlockSize)
	}

	numBlocks := numElements / ROCmFP4BlockSize
	wantBytes := numBlocks * ROCmFP4BlockBytes
	if len(rawBytes) != wantBytes {
		return fmt.Errorf("rocmfp4: byte length mismatch: got %d bytes, expected %d bytes for %d elements (%d blocks)",
			len(rawBytes), wantBytes, numElements, numBlocks)
	}

	// Validate scales
	for i := 0; i < numBlocks; i++ {
		scaleBits := binary.LittleEndian.Uint16(rawBytes[i*ROCmFP4BlockBytes : i*ROCmFP4BlockBytes+2])
		scale := FP16ToFloat32(scaleBits)
		if math.IsNaN(float64(scale)) || math.IsInf(float64(scale), 0) {
			return fmt.Errorf("rocmfp4: block %d carries invalid scale factor (bits 0x%04x)", i, scaleBits)
		}
	}
	return nil
}

// QuantizeROCmFP4 quantizes an f32 row-major matrix [rows, cols] into an ROCmFP4Tensor.
// Columns must be a multiple of ROCmFP4BlockSize (32).
func QuantizeROCmFP4(src []float32, rows, cols int) (*ROCmFP4Tensor, error) {
	if err := ValidateROCmFP4Dimensions(rows, cols); err != nil {
		return nil, err
	}
	total := rows * cols
	if len(src) != total {
		return nil, fmt.Errorf("rocmfp4: source length %d does not match rows*cols=%d", len(src), total)
	}

	numBlocks := total / ROCmFP4BlockSize
	blocks := make([]ROCmFP4Block, numBlocks)
	for i := 0; i < numBlocks; i++ {
		offset := i * ROCmFP4BlockSize
		blocks[i] = QuantizeROCmFP4Block(src[offset : offset+ROCmFP4BlockSize])
	}

	return &ROCmFP4Tensor{
		Rows:        rows,
		Cols:        cols,
		NumElements: total,
		Blocks:      blocks,
	}, nil
}

// QuantizeROCmFP4Slice quantizes a 1D slice whose length is a multiple of 32.
func QuantizeROCmFP4Slice(src []float32) (*ROCmFP4Tensor, error) {
	if len(src) == 0 || len(src)%ROCmFP4BlockSize != 0 {
		return nil, fmt.Errorf("rocmfp4: slice length %d must be non-zero and a multiple of %d", len(src), ROCmFP4BlockSize)
	}
	return QuantizeROCmFP4(src, 1, len(src))
}

// DequantizeROCmFP4 reconstructs all float32 values from the tensor.
func DequantizeROCmFP4(t *ROCmFP4Tensor) []float32 {
	if t == nil || len(t.Blocks) == 0 {
		return nil
	}
	out := make([]float32, len(t.Blocks)*ROCmFP4BlockSize)
	for i, blk := range t.Blocks {
		DequantizeROCmFP4Block(blk, out[i*ROCmFP4BlockSize:(i+1)*ROCmFP4BlockSize])
	}
	if t.NumElements > 0 && len(out) > t.NumElements {
		out = out[:t.NumElements]
	}
	return out
}

// DequantizeROCmFP4Bytes dequantizes a raw contiguous byte buffer of ROCmFP4 blocks.
func DequantizeROCmFP4Bytes(raw []byte) ([]float32, error) {
	if len(raw) == 0 || len(raw)%ROCmFP4BlockBytes != 0 {
		return nil, fmt.Errorf("rocmfp4: byte length %d must be non-zero and a multiple of block bytes %d",
			len(raw), ROCmFP4BlockBytes)
	}
	numBlocks := len(raw) / ROCmFP4BlockBytes
	out := make([]float32, numBlocks*ROCmFP4BlockSize)

	for i := 0; i < numBlocks; i++ {
		offset := i * ROCmFP4BlockBytes
		var blk ROCmFP4Block
		blk.Scale = binary.LittleEndian.Uint16(raw[offset : offset+2])
		copy(blk.Data[:], raw[offset+2:offset+ROCmFP4BlockBytes])
		DequantizeROCmFP4Block(blk, out[i*ROCmFP4BlockSize:(i+1)*ROCmFP4BlockSize])
	}
	return out, nil
}

// Bytes serializes the tensor's blocks into a contiguous byte slice.
func (t *ROCmFP4Tensor) Bytes() []byte {
	if t == nil || len(t.Blocks) == 0 {
		return nil
	}
	raw := make([]byte, len(t.Blocks)*ROCmFP4BlockBytes)
	for i, blk := range t.Blocks {
		off := i * ROCmFP4BlockBytes
		binary.LittleEndian.PutUint16(raw[off:off+2], blk.Scale)
		copy(raw[off+2:off+ROCmFP4BlockBytes], blk.Data[:])
	}
	return raw
}

// ROCmFP4FromBytes reconstructs an ROCmFP4Tensor from serialized raw bytes.
func ROCmFP4FromBytes(raw []byte, rows, cols int) (*ROCmFP4Tensor, error) {
	if err := ValidateROCmFP4Dimensions(rows, cols); err != nil {
		return nil, err
	}
	total := rows * cols
	if err := ValidateROCmFP4Layout(total, raw); err != nil {
		return nil, err
	}

	numBlocks := total / ROCmFP4BlockSize
	blocks := make([]ROCmFP4Block, numBlocks)
	for i := 0; i < numBlocks; i++ {
		off := i * ROCmFP4BlockBytes
		blocks[i].Scale = binary.LittleEndian.Uint16(raw[off : off+2])
		copy(blocks[i].Data[:], raw[off+2:off+ROCmFP4BlockBytes])
	}

	return &ROCmFP4Tensor{
		Rows:        rows,
		Cols:        cols,
		NumElements: total,
		Blocks:      blocks,
	}, nil
}

// ByteSize returns the total packed byte size of this tensor.
func (t *ROCmFP4Tensor) ByteSize() int64 {
	if t == nil {
		return 0
	}
	return int64(len(t.Blocks) * ROCmFP4BlockBytes)
}

// ROCmFP4MatVec computes matrix-vector multiplication y = A * x,
// where A is an [rows, cols] ROCmFP4Tensor and x is an f32 vector of length cols.
// It returns an f32 vector y of length rows.
func ROCmFP4MatVec(t *ROCmFP4Tensor, x []float32) ([]float32, error) {
	if t == nil {
		return nil, fmt.Errorf("rocmfp4: tensor is nil")
	}
	if len(x) != t.Cols {
		return nil, fmt.Errorf("rocmfp4: vector length %d does not match cols %d", len(x), t.Cols)
	}
	blocksPerRow := t.Cols / ROCmFP4BlockSize
	y := make([]float32, t.Rows)
	var blkBuf [ROCmFP4BlockSize]float32

	for r := 0; r < t.Rows; r++ {
		var sum float32
		rowBlockOffset := r * blocksPerRow
		for b := 0; b < blocksPerRow; b++ {
			blk := t.Blocks[rowBlockOffset+b]
			DequantizeROCmFP4Block(blk, blkBuf[:])
			xOffset := b * ROCmFP4BlockSize
			for i := 0; i < ROCmFP4BlockSize; i++ {
				sum += blkBuf[i] * x[xOffset+i]
			}
		}
		y[r] = sum
	}
	return y, nil
}

// ROCmFP4Metadata returns the self-describing FP4Metadata preset for this format.
func ROCmFP4Metadata() FP4Metadata {
	return ROCmFP4MetadataPreset()
}
