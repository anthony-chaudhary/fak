package model

import (
	"encoding/binary"
	"fmt"
	"math"
)

// MXFP8 (OCP MX microscaling, the layout NVIDIA ModelOpt emits for
// nvidia/Kimi-K3.1-Pro-NVFP4's attention and shared-expert tensors):
// the data elements are float8_e4m3fn bytes (fak dtype F8_E4M3) and the shared
// scale is one E8M0 byte per 32-element block along the LAST (contiguous) dim:
//
//	out[i] = e4m3(data[i]) * 2^(scale[i/32] - 127)
//
// The E4M3 element decode is the package's existing one (fp8E4M3ToF32 in
// fp8_quantize.go, golden-pinned in fp8_blockscale_test.go) — no second decode
// table lives here. The E8M0 scale is a pure biased exponent: no sign, no
// mantissa, bias 127, so scale = 2^(byte-127) with byte ∈ [0,254]; byte 255 is
// the E8M0 NaN and is refused on decode, matching the package's existing
// F8_E8M0 posture (v41_expert_faultin.go, v41_dspark_reference_rows.go). The
// same 2^(byte-127) convention already decodes MXFP4 blocks (decodeMXFP4Blocks
// in safetensors.go) and the V4.1 32-wide E4M3+E8M0 rows
// (v41_dspark_reference_rows.go); this file is the standalone tensor-shaped
// decode/encode pair for MXFP8, not a new scale dialect.
//
// This is decode + round-trip encode only. Wiring the pairing into the
// safetensors load loop is the fenced follow-on (fak#14365).

// mxfp8Block is the OCP MX block width: one E8M0 scale byte per 32 data
// elements, tiling the last (contiguous) dimension of the tensor.
const mxfp8Block = 32

// mxfp8E4M3Max is the largest finite float8_e4m3fn magnitude (S.1111.110 =
// 448); the encode picks the per-block scale so the block's largest magnitude
// maps onto at most this value.
const mxfp8E4M3Max = 448.0

// mxfp8ScaleBias is the E8M0 exponent bias: scale byte b carries scale
// 2^(b-127), so byte 127 is scale 1.0 and the representable scale range is
// [2^-127, 2^127] (bytes 0..254).
const mxfp8ScaleBias = 127

// mxfp8ScaleNaN is the E8M0 byte whose exponent field is all ones: per OCP MX
// it is NaN, not a scale. Decode refuses it rather than multiplying by a NaN;
// encode never emits it (it computes scale bytes, never accepts them).
const mxfp8ScaleNaN = 255

// mxfp8ScaleShape returns the scale tensor shape for a data shape of any rank:
// every dim except the last is unchanged and the last dim becomes
// ceil(last/mxfp8Block) — one scale per 32-element block along the contiguous
// axis.
func mxfp8ScaleShape(shape []int) ([]int, error) {
	if len(shape) == 0 {
		return nil, fmt.Errorf("mxfp8: empty shape")
	}
	for _, d := range shape {
		if d <= 0 {
			return nil, fmt.Errorf("mxfp8: shape %v has a non-positive dimension", shape)
		}
	}
	out := append([]int(nil), shape...)
	out[len(out)-1] = (shape[len(shape)-1] + mxfp8Block - 1) / mxfp8Block
	return out, nil
}

// mxfp8ScaleElems validates shape and returns (elems, scaleElems): the dense
// element count and the block count prod(shape[:-1]) * ceil(last/32).
func mxfp8ScaleElems(name string, shape []int) (int, int, error) {
	scaleShape, err := mxfp8ScaleShape(shape)
	if err != nil {
		return 0, 0, fmt.Errorf("mxfp8 %s: %w", name, err)
	}
	elems, ok := checkedShapeProduct(shape...)
	if !ok {
		return 0, 0, fmt.Errorf("mxfp8 %s: shape %v overflows element count", name, shape)
	}
	scaleElems, ok := checkedShapeProduct(scaleShape...)
	if !ok {
		return 0, 0, fmt.Errorf("mxfp8 %s: scale shape %v overflows element count", name, scaleShape)
	}
	return elems, scaleElems, nil
}

// mxfp8BlockIndex maps a flat dense index i (row-major, last dim = last) to its
// scale index: row*(blocks per row) + col/32. When last is a multiple of 32
// this equals i/32; with a short final block it does not, because each row owns
// ceil(last/32) scales while the data stays dense (see the tail rule on
// DecodeMXFP8ToF32).
func mxfp8BlockIndex(i, last, blocksPerRow int) int {
	return (i/last)*blocksPerRow + (i%last)/mxfp8Block
}

// DecodeMXFP8ToF32 decodes an MXFP8 tensor to a fresh f32 slice. data holds the
// dense E4M3 bytes (exactly prod(shape) of them, row-major) and scales holds
// one E8M0 byte per 32-element block along the last dim — one byte per block of
// EVERY row, so scales has prod(shape[:-1]) * ceil(last/32) entries, not
// ceil(prod(shape)/32).
//
// Tail rule (OCP MX): when last is not a multiple of 32 the final block of each
// row is zero-padded to 32 for block purposes. The pad contributes nothing to a
// block's magnitude (zeros never raise the amax, so the scale of a short block
// is computed over the real elements only) and the pad elements are not stored:
// data stays dense at prod(shape) bytes and the scale count is ceil(last/32)
// per row. A decoded pad position would be 0*scale = 0, but no such position
// exists in the dense tensor.
//
// Out-of-range decode is best-effort IEEE-754, not saturation: the f32 multiply
// can overflow to ±Inf (possible once e4m3's 448 meets scale bytes above ~119)
// or underflow to ±0/subnormal. The encode side guarantees block maxima fit by
// construction, so an out-of-range decode points at a corrupt or adversarial
// producer; this helper surfaces the IEEE result rather than guessing a clamp.
// A scale byte of 255 (E8M0 NaN) is refused, never multiplied.
func DecodeMXFP8ToF32(name string, shape []int, data, scales []byte) ([]float32, error) {
	elems, scaleElems, err := mxfp8ScaleElems(name, shape)
	if err != nil {
		return nil, err
	}
	if len(data) != elems {
		return nil, fmt.Errorf("mxfp8 %s: data has %d bytes, shape %v implies %d", name, len(data), shape, elems)
	}
	if len(scales) != scaleElems {
		return nil, fmt.Errorf("mxfp8 %s: scales has %d bytes, shape %v (blocks of %d) implies %d", name, len(scales), shape, mxfp8Block, scaleElems)
	}
	for _, b := range scales {
		if b == mxfp8ScaleNaN {
			return nil, fmt.Errorf("mxfp8 %s: scale byte is E8M0 NaN (0xff)", name)
		}
	}
	last := shape[len(shape)-1]
	blocksPerRow := (last + mxfp8Block - 1) / mxfp8Block
	out := make([]float32, elems)
	for i := range out {
		// Ldexp(1, e) is the exact power of two 2^e, so the scale multiply
		// itself never rounds; the only precision loss is E4M3's own.
		out[i] = fp8E4M3ToF32(data[i]) * float32(math.Ldexp(1, int(scales[mxfp8BlockIndex(i, last, blocksPerRow)])-mxfp8ScaleBias))
	}
	return out, nil
}

// DecodeMXFP8ToBytes is the safetensors-load adapter over DecodeMXFP8ToF32: it
// returns the decoded tensor as little-endian f32 bytes plus its shape — the
// byte form the load loop appends into the resident f32 buffer, exactly like
// decodeFP8BlockScaleTensor and decodeMXFP4Blocks.
func DecodeMXFP8ToBytes(name string, shape []int, data, scales []byte) ([]byte, []int, error) {
	f32, err := DecodeMXFP8ToF32(name, shape, data, scales)
	if err != nil {
		return nil, nil, err
	}
	out := make([]byte, len(f32)*4)
	for i, v := range f32 {
		binary.LittleEndian.PutUint32(out[i*4:], math.Float32bits(v))
	}
	return out, append([]int(nil), shape...), nil
}

// DecodeMXFP8ToBF16Bytes decodes an MXFP8 tensor to little-endian bf16 bytes.
// Each element is rounded to nearest even on the f32 product first (the
// RNE-bit-truncate bf16 narrowing the package's resident path uses), then
// narrowed to the top 16 bits of the f32. Overflow to Inf survives the
// narrowing as bf16 Inf — same best-effort rule as DecodeMXFP8ToF32.
func DecodeMXFP8ToBF16Bytes(name string, shape []int, data, scales []byte) ([]byte, []int, error) {
	f32, err := DecodeMXFP8ToF32(name, shape, data, scales)
	if err != nil {
		return nil, nil, err
	}
	out := make([]byte, len(f32)*2)
	for i, v := range f32 {
		binary.LittleEndian.PutUint16(out[i*2:], mxfp8F32ToBF16(v))
	}
	return out, append([]int(nil), shape...), nil
}

// mxfp8F32ToBF16 narrows one f32 to bf16 by round-to-nearest-even on the low 16
// bits: add the half-ulp bias with the tie broken toward an even low bit, then
// truncate. NaN stays NaN (quiet bit set), Inf/±0 pass through.
func mxfp8F32ToBF16(v float32) uint16 {
	bits := math.Float32bits(v)
	low := bits & 0xffff
	if bits&0x7fffffff > 0x7f800000 { // NaN: keep it NaN, force the quiet bit
		return uint16((bits >> 16) | 0x0040)
	}
	rounded := bits >> 16
	if low > 0x8000 || (low == 0x8000 && rounded&1 == 1) {
		rounded++
	}
	return uint16(rounded)
}

// QuantizeMXFP8ToBytes encodes a dense row-major f32 tensor (values, exactly
// prod(shape) of them) to MXFP8: per-block E8M0 scale bytes plus per-element
// E4M3 data bytes, returning (data, scales, scaleShape).
//
// Scale rule (OCP MX): each block's scale is the smallest power of two s with
// amax/s <= 448, i.e. s = 2^ceil(log2(amax/448)), so the block max always fits
// and no element can overflow E4M3 after scaling. It is computed exactly from
// amax's f32 exponent (no 1/448 rounding), then clamped to the E8M0 range
// [2^-127, 2^127]. An all-zero block takes byte 127 (scale 1.0): any byte would
// decode those zeros identically, and 127 is the neutral choice a re-quantize
// of decoded zeros reproduces. A block max below 448*2^-127 clamps to byte 0
// (scale 2^-127); elements whose scaled magnitude falls below E4M3's minimum
// subnormal (2^-9 * s) encode as zero, and the (2^-9*s, 2^-6*s) band encodes as
// E4M3 subnormals with reduced precision — the decode reads both back.
//
// Data rule: each scaled value x/s is rounded to nearest even among E4M3's 254
// finite codes; the two NaN codes are never emitted, so anything above 448
// saturates to code 0x7e/0xfe (448) — unreachable here by the scale rule, kept
// as the encoder's saturation contract. Non-finite input values are a fail-closed
// error, matching the package's other quantize references.
func QuantizeMXFP8ToBytes(name string, shape []int, values []float32) ([]byte, []byte, []int, error) {
	elems, scaleElems, err := mxfp8ScaleElems(name, shape)
	if err != nil {
		return nil, nil, nil, err
	}
	if len(values) != elems {
		return nil, nil, nil, fmt.Errorf("mxfp8 %s: values has %d entries, shape %v implies %d", name, len(values), shape, elems)
	}
	for _, v := range values {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			return nil, nil, nil, fmt.Errorf("mxfp8 %s: non-finite value at input", name)
		}
	}
	last := shape[len(shape)-1]
	blocksPerRow := (last + mxfp8Block - 1) / mxfp8Block
	rows := elems / last
	data := make([]byte, elems)
	scales := make([]byte, scaleElems)
	// Blocks tile each row's last dim independently, so the loop is row-major:
	// a flat 32-step would mix row r's tail with row r+1's head whenever last
	// is not a multiple of 32.
	for row := 0; row < rows; row++ {
		rowBase := row * last
		for b := 0; b < blocksPerRow; b++ {
			col := b * mxfp8Block
			width := mxfp8Block
			if col+width > last {
				width = last - col // short final block: real elements only
			}
			block := values[rowBase+col : rowBase+col+width]
			si := row*blocksPerRow + b
			scales[si] = mxfp8ScaleByte(block)
			inv := float32(math.Ldexp(1, -int(scales[si])+mxfp8ScaleBias))
			for j, v := range block {
				data[rowBase+col+j] = mxfp8EncodeE4M3(v * inv)
			}
		}
	}
	scaleShape := append([]int(nil), shape[:len(shape)-1]...)
	scaleShape = append(scaleShape, blocksPerRow)
	return data, scales, scaleShape, nil
}

// mxfp8ScaleByte picks one block's E8M0 byte: the smallest power of two s =
// 2^e with amax/s <= 448, clamped to the E8M0 range. amax is the max |value| of
// the (possibly short) block; a zero block returns byte 127 (scale 1.0).
func mxfp8ScaleByte(block []float32) byte {
	amax := float32(0)
	for _, v := range block {
		if a := float32(math.Abs(float64(v))); a > amax {
			amax = a
		}
	}
	if amax == 0 {
		return mxfp8ScaleBias // zero block: neutral scale 1.0
	}
	bits := math.Float32bits(amax)
	biased := int(bits>>23) & 0xff
	if biased == 0 { // subnormal amax: ideal exponent is far below the E8M0 floor
		return 0
	}
	e := biased - mxfp8ScaleBias - 8
	if amax > float32(1.75*math.Ldexp(1, biased-mxfp8ScaleBias)) {
		e++
	}
	switch {
	case e < -mxfp8ScaleBias:
		return 0
	case e > mxfp8ScaleBias:
		return 2 * mxfp8ScaleBias
	default:
		return byte(e + mxfp8ScaleBias)
	}
}

// mxfp8EncodeE4M3 rounds one pre-scaled f32 (x = value / blockScale) to the
// nearest E4M3FN byte, ties-to-even, among the 254 finite codes. The two NaN
// codes are never emitted: values above 448 saturate to 0x7e/0xfe. Subnormal
// E4M3 (codes 0x01..0x07) is produced for scaled magnitudes in (2^-10, 2^-6);
// magnitudes at or below 2^-10 round to zero. The decode side is the package's
// existing fp8E4M3ToF32 lookup, so encoder and decoder cannot disagree about
// what a code means.
func mxfp8EncodeE4M3(x float32) byte {
	sign := byte(0)
	v := x
	if math.Signbit(float64(v)) {
		sign = 0x80
		v = -v
	}
	if v == 0 {
		return sign // exact zero (incl. -0) encodes verbatim
	}
	bits := math.Float32bits(v)
	biased := int(bits>>23) & 0xff
	frac := bits & 0x7fffff
	if biased == 0 || biased < 121 { // below 2^-6: E4M3 subnormal/zero band
		// Units of the minimum subnormal 2^-9: round v*2^9 to the nearest
		// integer, ties-to-even (exact in f64; v < 2^-6 keeps the product < 8).
		s := float64(v) * 512.0
		fl := math.Floor(s)
		q := int64(fl)
		if d := s - fl; d > 0.5 || (d == 0.5 && q%2 == 1) {
			q++
		}
		return sign | byte(q) // q ∈ [0,8]; q == 8 is the 2^-6 normal code
	}
	if biased > 127+8 || v > mxfp8E4M3Max { // beyond 448: saturate, NaN excluded
		return sign | 0x7e
	}
	// Normal band 2^-6..448: round the 24-bit significand to 4 bits RTNE.
	e4 := biased - 127 + 7
	sig := int64(1<<23) | int64(frac)
	q := sig >> 20
	rem := sig & ((1 << 20) - 1)
	if rem > 1<<19 || (rem == 1<<19 && q&1 == 1) {
		q++
	}
	if q == 16 { // mantissa carry bumps the exponent
		q, e4 = 8, e4+1
	}
	if e4 > 15 || (e4 == 15 && q == 15) { // overflow or the NaN slot: 448
		return sign | 0x7e
	}
	return sign | byte(e4<<3) | byte(q&7)
}

// QuantizeMXFP8FromBF16 is QuantizeMXFP8ToBytes over little-endian bf16 source
// bytes: each element is widened losslessly (bf16 is the top 16 f32 bits) and
// then encoded by the same per-block scale + RTNE E4M3 path.
func QuantizeMXFP8FromBF16(name string, shape []int, bf16 []byte) ([]byte, []byte, []int, error) {
	elems, _, err := mxfp8ScaleElems(name, shape)
	if err != nil {
		return nil, nil, nil, err
	}
	if len(bf16) != 2*elems {
		return nil, nil, nil, fmt.Errorf("mxfp8 %s: bf16 source has %d bytes, shape %v implies %d", name, len(bf16), shape, 2*elems)
	}
	values := make([]float32, elems)
	for i := range values {
		values[i] = math.Float32frombits(uint32(binary.LittleEndian.Uint16(bf16[i*2:])) << 16)
	}
	return QuantizeMXFP8ToBytes(name, shape, values)
}
