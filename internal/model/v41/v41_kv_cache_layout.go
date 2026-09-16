// Copyright (c) 2023 DeepSeek
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
//
// Adapted from deepseek-ai/DeepSeek-V4.1-Flash attention, transcribed from
// deepseek-ai/FlashMLA tests/quant.py at
// ba89a3466e9470ad08ab39738d4e7bb66989e1e7 (MIT).

package v41

import (
	"errors"
	"fmt"
	"math"

	model "github.com/anthony-chaudhary/fak/internal/model"
)

// v41_kv_cache_layout.go - the typed paged, quantized KV-cache byte layout for
// the DeepSeek V4 / V4.1 native decode path (fak-private#1729, epic #12640).
//
// FlashMLA tests/quant.py (`KVCacheLayout`) is the authoritative public layout
// for the DeepSeek paged KV cache. Every attention layer KV row for one token
// is a fixed byte record: a NoPE part quantized in tiles, a RoPE part (fp8 in
// the same stream for V4.1), and a scale row. `get_meta()` returns
// (d, d_nope, d_rope, tile_size, num_tiles) and `get_bytes_per_token()` returns
// the record size:
//
//	V41 fp8: 448 (NoPE) + 64 (RoPE) + 16 (e8m0 scales) = 528 B/token, tile 32
//	V41 fp4: 448 e2m1 + 64 e2m1 packed 2-per-byte, e4m3 (amax/6) scales
//	         = 512/2 + 32 = 288 B/token, tile 16
//
// This file owns ONLY the layout contract and a host-side reference
// quantize/dequantize that reproduces the published geometry. It runs no
// attention, no projection, and no kernel; the compressed-window construct in
// v4_flash_kv_layout.go is a separate host-side reference, not a byte layout.

// V41KVQuantFormat is the closed set of quantized KV record formats this
// package can size or address. Anything else fails closed.
type V41KVQuantFormat int

const (
	// V41KVUnspecified is the zero value: no format was selected. It is never a
	// valid layout, so a zero-value V41KVCacheLayout fails closed on use.
	V41KVUnspecified V41KVQuantFormat = iota
	// V41KVFP8 is the V4.1 fp8 record: 448 fp8 NoPE + 64 fp8 RoPE + 16 e8m0
	// scale bytes, tile size 32 -> 528 B/token.
	V41KVFP8
	// V41KVFP4 is the V4.1 fp4 record: 448 e2m1 NoPE + 64 e2m1 RoPE packed
	// two-per-byte (512 bytes packed / 2 = 256) + 32 e4m3 (amax/6) scale bytes,
	// tile size 16 -> 288 B/token.
	V41KVFP4
)

// String renders the format for witnesses and error messages.
func (f V41KVQuantFormat) String() string {
	switch f {
	case V41KVFP8:
		return "v41-fp8"
	case V41KVFP4:
		return "v41-fp4"
	default:
		return fmt.Sprintf("v41-kv-format(%d)", int(f))
	}
}

// V41KVScaleFormat is the closed set of scale encodings a V4.1 KV record may
// carry. It is separate from the payload format so a caller cannot silently
// pair an fp8 payload with an fp4 scale stride.
type V41KVScaleFormat int

const (
	// V41KVScaleUnspecified is the zero value and never valid.
	V41KVScaleUnspecified V41KVScaleFormat = iota
	// V41KVScaleE8M0 is the fp8 exponent-only scale: byte 127 is 2^0, and the
	// emitted byte is `amax/448` rounded up to a power of two (ue8m0).
	V41KVScaleE8M0
	// V41KVScaleE4M3 is the fp4 mantissa-bearing scale: a float8_e4m3 carrying
	// `amax/6`.
	V41KVScaleE4M3
)

// String renders the scale format for witnesses and error messages.
func (f V41KVScaleFormat) String() string {
	switch f {
	case V41KVScaleE8M0:
		return "e8m0"
	case V41KVScaleE4M3:
		return "e4m3"
	default:
		return fmt.Sprintf("v41-kv-scale(%d)", int(f))
	}
}

// V41KVCacheLayout is the typed byte geometry of one V4.1 paged KV record for a
// single token. Every field is derived from the published FlashMLA
// `KVCacheLayout.get_meta()` / `get_bytes_per_token()` and is validated before
// the layout can size or address a cache.
type V41KVCacheLayout struct {
	Format V41KVQuantFormat
	// DNoPE is the NoPE latent width quantized in tile-sized groups (448).
	DNoPE int
	// DRoPE is the RoPE width quantized in the same stream (64).
	DRoPE int
	// TileSize is the NoPE quantization tile edge in elements (32 for fp8, 16
	// for fp4). Each tile carries exactly one scale element.
	TileSize int
	// NumTiles is the published total tile count over d = DNoPE + DRoPE (16 for
	// fp8 at tile 32, 32 for fp4 at tile 16). It is the full count returned by
	// `get_meta()`, so NoPE tiles = DNoPE/TileSize and the remaining
	// NumTiles - DNoPE/TileSize tiles cover the RoPE part.
	NumTiles int
	// ScaleFormat is the scale encoding for both the NoPE and RoPE parts.
	ScaleFormat V41KVScaleFormat
	// BytesPerToken is the published full record size (528 fp8, 288 fp4).
	BytesPerToken int
}

// v41KVCacheLayouts is the closed set of published V4.1 KV record layouts. The
// table is the single source of truth for both selection and validation.
var v41KVCacheLayouts = map[V41KVQuantFormat]V41KVCacheLayout{
	V41KVFP8: {
		Format:        V41KVFP8,
		DNoPE:         448,
		DRoPE:         64,
		TileSize:      32,
		NumTiles:      16,
		ScaleFormat:   V41KVScaleE8M0,
		BytesPerToken: 528,
	},
	V41KVFP4: {
		Format:        V41KVFP4,
		DNoPE:         448,
		DRoPE:         64,
		TileSize:      16,
		NumTiles:      32,
		ScaleFormat:   V41KVScaleE4M3,
		BytesPerToken: 288,
	},
}

// V41KVCacheLayoutFor returns the published layout for one V4.1 record format.
// It fails closed (ok=false) for the zero value and for any format outside the
// closed set; the returned layout is a copy of the table entry.
func V41KVCacheLayoutFor(format V41KVQuantFormat) (V41KVCacheLayout, bool) {
	l, ok := v41KVCacheLayouts[format]
	return l, ok
}

// V41KVCacheLayouts returns every published layout, fp8 first, for census and
// witness use.
func V41KVCacheLayouts() []V41KVCacheLayout {
	return []V41KVCacheLayout{v41KVCacheLayouts[V41KVFP8], v41KVCacheLayouts[V41KVFP4]}
}

// Validate re-derives the record size from its parts and refuses a layout whose
// declared BytesPerToken disagrees with the published arithmetic. A layout is
// the cache addressing contract, so a mismatch must fail closed rather than
// silently size a cache that the checkpoint and kernel will address differently.
func (l V41KVCacheLayout) Validate() error {
	if l.Format == V41KVUnspecified {
		return fmt.Errorf("%w: format is unspecified", ErrV41KVCacheLayout)
	}
	published, ok := v41KVCacheLayouts[l.Format]
	if !ok {
		return fmt.Errorf("%w: format %s is outside the published set", ErrV41KVCacheLayout, l.Format)
	}
	if l != published {
		return fmt.Errorf("%w: layout %+v does not match the published %s layout %+v", ErrV41KVCacheLayout, l, l.Format, published)
	}
	if l.DNoPE <= 0 || l.DRoPE <= 0 || l.TileSize <= 0 || l.NumTiles <= 0 {
		return fmt.Errorf("%w: %s has a non-positive geometry field", ErrV41KVCacheLayout, l.Format)
	}
	// The published tile count covers the full d = DNoPE + DRoPE width, and the
	// NoPE part must divide evenly into tiles. A layout whose tile count or NoPE
	// width disagrees with that arithmetic is refused.
	if l.DNoPE%l.TileSize != 0 {
		return fmt.Errorf("%w: %s DNoPE %d is not a multiple of tile %d", ErrV41KVCacheLayout, l.Format, l.DNoPE, l.TileSize)
	}
	if l.DNoPE+l.DRoPE != l.TileSize*l.NumTiles {
		return fmt.Errorf("%w: %s d %d != tile %d x %d tiles", ErrV41KVCacheLayout, l.Format, l.DNoPE+l.DRoPE, l.TileSize, l.NumTiles)
	}
	want, err := l.recordBytes()
	if err != nil {
		return err
	}
	if l.BytesPerToken != want {
		return fmt.Errorf("%w: %s declares %d B/token but its parts derive %d", ErrV41KVCacheLayout, l.Format, l.BytesPerToken, want)
	}
	return nil
}

// recordBytes re-derives the per-token record size from the geometry. fp8 stores
// DNoPE+DRoPE one byte per value; fp4 packs two e2m1 values per byte along both
// parts, so both are halved. The scale row is one element per tile over the NoPE
// part (NumTiles) plus RoPEScaleElements for the RoPE part.
func (l V41KVCacheLayout) recordBytes() (int, error) {
	scaleElems := l.NumTiles
	switch l.Format {
	case V41KVFP8:
		return l.DNoPE + l.DRoPE + scaleElems, nil
	case V41KVFP4:
		if (l.DNoPE+l.DRoPE)%2 != 0 {
			return 0, fmt.Errorf("%w: fp4 record has an odd value count", ErrV41KVCacheLayout)
		}
		return (l.DNoPE+l.DRoPE)/2 + scaleElems, nil
	default:
		return 0, fmt.Errorf("%w: format %s has no record-size rule", ErrV41KVCacheLayout, l.Format)
	}
}

// V41KVCacheRow is one layer decoded KV row for a single token: the NoPE and
// RoPE f32 values, plus the scale elements that were applied, so a witness can
// check the decode as well as the payload.
type V41KVCacheRow struct {
	NoPE       []float32
	RoPE       []float32
	NoPEScales []float32
	RoPEScales []float32
}

// V41KVQuantizeRow packs one already-projected KV row into the layout byte
// record. nope must be exactly DNoPE wide and rope exactly DRoPE; anything else
// fails closed. fp8 encodes each value as e4m3 with one e8m0 scale per tile
// (`amax/448` raised to the next power of two); fp4 encodes each value as an
// e2m1 nibble packed two-per-byte with one e4m3 (`amax/6`) scale per tile.
//
// The record order matches the published layout: NoPE payload, RoPE payload,
// NoPE scales, then RoPE scales.
func (l V41KVCacheLayout) V41KVQuantizeRow(nope, rope []float32) ([]byte, error) {
	if err := l.Validate(); err != nil {
		return nil, err
	}
	if len(nope) != l.DNoPE {
		return nil, fmt.Errorf("%w: NoPE row width %d, want %d", ErrV41KVCacheLayout, len(nope), l.DNoPE)
	}
	if len(rope) != l.DRoPE {
		return nil, fmt.Errorf("%w: RoPE row width %d, want %d", ErrV41KVCacheLayout, len(rope), l.DRoPE)
	}
	if err := finiteV41KVCacheValues(nope, rope); err != nil {
		return nil, err
	}

	out := make([]byte, 0, l.BytesPerToken)
	switch l.Format {
	case V41KVFP8:
		payload, scales := quantizeV41KVFP8(nope, rope, l.TileSize)
		out = append(out, payload...)
		out = append(out, scales...)
	case V41KVFP4:
		payload, scales := quantizeV41KVFP4(nope, rope, l.TileSize)
		out = append(out, payload...)
		out = append(out, scales...)
	default:
		return nil, fmt.Errorf("%w: format %s has no quantizer", ErrV41KVCacheLayout, l.Format)
	}
	if len(out) != l.BytesPerToken {
		return nil, fmt.Errorf("%w: %s quantizer produced %d bytes, want %d", ErrV41KVCacheLayout, l.Format, len(out), l.BytesPerToken)
	}
	return out, nil
}

// V41KVDequantizeRow decodes a record produced by V41KVQuantizeRow back to f32.
// The record length must equal BytesPerToken; a short or long record fails
// closed rather than being truncated or padded.
func (l V41KVCacheLayout) V41KVDequantizeRow(record []byte) (*V41KVCacheRow, error) {
	if err := l.Validate(); err != nil {
		return nil, err
	}
	if len(record) != l.BytesPerToken {
		return nil, fmt.Errorf("%w: %s record has %d bytes, want %d", ErrV41KVCacheLayout, l.Format, len(record), l.BytesPerToken)
	}
	row := &V41KVCacheRow{}
	nopeTiles := l.DNoPE / l.TileSize
	switch l.Format {
	case V41KVFP8:
		payload, scales := record[:l.DNoPE+l.DRoPE], record[l.DNoPE+l.DRoPE:]
		row.NoPE = dequantizeV41KVFP8(payload[:l.DNoPE], scales[:nopeTiles], l.TileSize)
		row.RoPE = dequantizeV41KVFP8(payload[l.DNoPE:], scales[nopeTiles:], l.TileSize)
		row.NoPEScales = v41KVScaleValuesE8M0(scales[:nopeTiles])
		row.RoPEScales = v41KVScaleValuesE8M0(scales[nopeTiles:])
	case V41KVFP4:
		payload := record[:l.DNoPE/2+l.DRoPE/2]
		scales := record[l.DNoPE/2+l.DRoPE/2:]
		row.NoPE = dequantizeV41KVFP4(payload[:l.DNoPE/2], scales[:nopeTiles], l.TileSize)
		row.RoPE = dequantizeV41KVFP4(payload[l.DNoPE/2:], scales[nopeTiles:], l.TileSize)
		row.NoPEScales = v41KVScaleValuesE4M3(scales[:nopeTiles])
		row.RoPEScales = v41KVScaleValuesE4M3(scales[nopeTiles:])
	default:
		return nil, fmt.Errorf("%w: format %s has no dequantizer", ErrV41KVCacheLayout, l.Format)
	}
	return row, nil
}

// finiteV41KVCacheValues refuses a non-finite NoPE or RoPE value before any
// quantization, so a NaN never becomes a plausible byte code.
func finiteV41KVCacheValues(nope, rope []float32) error {
	for i, v := range nope {
		if !model.Finite32(v) {
			return fmt.Errorf("%w: NoPE[%d] is non-finite", ErrV41KVCacheLayout, i)
		}
	}
	for i, v := range rope {
		if !model.Finite32(v) {
			return fmt.Errorf("%w: RoPE[%d] is non-finite", ErrV41KVCacheLayout, i)
		}
	}
	return nil
}

// quantizeV41KVFP8 encodes the NoPE and RoPE values as e4m3 with one e8m0 scale
// per tile, returning the concatenated payload then the concatenated scales.
func quantizeV41KVFP8(nope, rope []float32, tile int) ([]byte, []byte) {
	payload := make([]byte, 0, len(nope)+len(rope))
	scales := make([]byte, 0, len(nope)/tile+len(rope)/tile)
	for _, part := range [][]float32{nope, rope} {
		for start := 0; start < len(part); start += tile {
			scale := e8m0ScaleFor(part[start : start+tile])
			scales = append(scales, scale)
			inv := e8m0ScaleToF32(scale)
			for _, v := range part[start : start+tile] {
				payload = append(payload, encodeE4M3Value(v/inv))
			}
		}
	}
	return payload, scales
}

// dequantizeV41KVFP8 decodes e4m3 payload values with one e8m0 scale per tile.
func dequantizeV41KVFP8(payload, scales []byte, tile int) []float32 {
	out := make([]float32, len(payload))
	for start, s := 0, 0; start < len(payload); start, s = start+tile, s+1 {
		inv := e8m0ScaleToF32(scales[s])
		for i := start; i < start+tile && i < len(payload); i++ {
			out[i] = model.FP8E4M3ToF32(payload[i]) * inv
		}
	}
	return out
}

// quantizeV41KVFP4 encodes the NoPE and RoPE values as e2m1 nibbles packed two
// per byte (low nibble first) with one e4m3 (`amax/6`) scale per tile, returning
// the concatenated packed payload then the concatenated scales.
func quantizeV41KVFP4(nope, rope []float32, tile int) ([]byte, []byte) {
	payload := make([]byte, 0, (len(nope)+len(rope))/2)
	scales := make([]byte, 0, len(nope)/tile+len(rope)/tile)
	for _, part := range [][]float32{nope, rope} {
		for start := 0; start < len(part); start += tile {
			scales = append(scales, encodeE4M3Scale(part[start:start+tile]))
			inv := e4m3ScaleToF32(scales[len(scales)-1])
			for i := start; i < start+tile; i += 2 {
				lo := e2m1Code(part[i] / inv)
				hi := e2m1Code(part[i+1] / inv)
				payload = append(payload, lo|hi<<4)
			}
		}
	}
	return payload, scales
}

// dequantizeV41KVFP4 unpacks e2m1 nibbles (low nibble first) and applies one
// e4m3 (`amax/6`) scale per tile.
func dequantizeV41KVFP4(packed, scales []byte, tile int) []float32 {
	out := make([]float32, 0, len(packed)*2)
	half := tile / 2
	for start, s := 0, 0; start < len(packed); start, s = start+half, s+1 {
		inv := e4m3ScaleToF32(scales[s])
		for i := start; i < start+half && i < len(packed); i++ {
			out = append(out, e2m1NibbleValue(int(packed[i]&0x0f))*inv)
			out = append(out, e2m1NibbleValue(int(packed[i]>>4))*inv)
		}
	}
	return out
}

// e8m0ScaleFor returns the fp8 tile scale byte: the smallest power of two whose
// magnitude bounds amax/448, matching `_cast_scale_inv_to_ue8m0`. A zero tile
// uses byte 127 (2^0) so the decode is the identity.
func e8m0ScaleFor(tile []float32) byte {
	var amax float64
	for _, v := range tile {
		if a := math.Abs(float64(v)); a > amax {
			amax = a
		}
	}
	if amax == 0 {
		return 127
	}
	exp := int(math.Ceil(math.Log2(amax / v41KVE4M3Max)))
	return byte(clampE8M0Exponent(exp))
}

// e8m0ScaleToF32 decodes one e8m0 byte to its power of two. Byte 0xff is the
// e8m0 NaN code.
func e8m0ScaleToF32(b byte) float32 {
	if b == 0xff {
		return float32(math.NaN())
	}
	return float32(math.Ldexp(1, int(b)-127))
}

// v41KVScaleValuesE8M0 widens e8m0 scale bytes to their f32 powers for a
// witness. A NaN byte decodes to NaN so a caller can detect it.
func v41KVScaleValuesE8M0(scales []byte) []float32 {
	out := make([]float32, len(scales))
	for i, b := range scales {
		out[i] = e8m0ScaleToF32(b)
	}
	return out
}

// v41KVScaleValuesE4M3 widens the e4m3 scale bytes to f32.
func v41KVScaleValuesE4M3(scales []byte) []float32 {
	out := make([]float32, len(scales))
	for i, b := range scales {
		out[i] = e4m3ScaleToF32(b)
	}
	return out
}

// encodeE4M3Scale chooses the e4m3 scale byte for a tile: the smallest
// representable e4m3 value at least amax/6, matching the fp4 `amax/6` scale. A
// zero tile uses the e4m3 zero code.
func encodeE4M3Scale(tile []float32) byte {
	var amax float64
	for _, v := range tile {
		if a := math.Abs(float64(v)); a > amax {
			amax = a
		}
	}
	if amax == 0 {
		return 0
	}
	target := amax / v41KVE2M1Max
	best := byte(0)
	bestVal := math.Inf(1)
	for b := 0; b < 256; b++ {
		if b&0x7f == 0x7f {
			continue
		}
		v := math.Abs(float64(e4m3ScaleToF32(byte(b))))
		if v >= target && v < bestVal {
			best, bestVal = byte(b), v
		}
	}
	return best
}

// encodeE4M3Value encodes one already-scaled value as an e4m3 byte, rounding to
// the nearest finite code. It reuses the production decoder for its value table.
func encodeE4M3Value(v float32) byte {
	x := float64(v)
	sign := byte(0)
	if math.Signbit(x) {
		sign = 0x80
		x = -x
	}
	best, bestErr := 0, math.Inf(1)
	for code := 0; code < 128; code++ {
		if code == 0x7f {
			continue
		}
		err := math.Abs(float64(e4m3ScaleToF32(byte(code))) - x)
		if err < bestErr {
			best, bestErr = code, err
		}
	}
	return sign | byte(best)
}

// e4m3ScaleToF32 decodes one e4m3 byte via the shared production decoder.
func e4m3ScaleToF32(b byte) float32 { return model.FP8E4M3ToF32(b) }

// e2m1Code encodes one already-scaled value as an e2m1 nibble, rounding to the
// nearest representable magnitude and saturating at the format maximum.
func e2m1Code(v float32) byte {
	x := float64(v)
	sign := byte(0)
	if math.Signbit(x) {
		sign = 0x8
		x = -x
	}
	best, bestErr := 0, math.Inf(1)
	for code := 0; code < 8; code++ {
		mag := float64(e2m1Value(code))
		err := math.Abs(mag - x)
		if err < bestErr {
			best, bestErr = code, err
		}
	}
	return sign | byte(best)
}

// e2m1NibbleValue decodes one 4-bit e2m1 nibble including its sign bit. Bit 3
// is the sign; the low three bits index the magnitude table. A sign-set zero is
// returned as -0, matching the float4_e2m1fn_x2 encoding.
func e2m1NibbleValue(nibble int) float32 {
	mag := e2m1Value(nibble & 0x7)
	if nibble&0x8 != 0 {
		return -mag
	}
	return mag
}

// e2m1Value decodes one e2m1 code (0..7 magnitude) to its f32 value.
func e2m1Value(code int) float32 {
	exponent := code >> 1 & 3
	mantissa := code & 1
	if exponent == 0 {
		return float32(float64(mantissa) / 2)
	}
	return float32(math.Ldexp(1+float64(mantissa)/2, exponent-1))
}

// clampE8M0Exponent clamps an exponent to the e8m0 byte range, keeping the NaN
// byte 0xff out of the emitted set.
func clampE8M0Exponent(exp int) int {
	biased := exp + 127
	if biased < 0 {
		return 0
	}
	if biased > 254 {
		return 254
	}
	return biased
}

const (
	// v41KVE4M3Max is the largest finite e4m3 magnitude (448), the divisor the
	// fp8 `_cast_scale_inv_to_ue8m0` scale uses.
	v41KVE4M3Max = 448.0
	// v41KVE2M1Max is the largest finite e2m1 magnitude (6), the divisor the fp4
	// `amax/6` scale uses.
	v41KVE2M1Max = 6.0
)

// ErrV41KVCacheLayout is the typed fail-closed refusal for a paged KV record
// whose format, geometry, width, or byte length does not match the published
// FlashMLA layout. A layout is the cache's addressing contract, so an
// unrecognized or inconsistent request is refused rather than guessed.
var ErrV41KVCacheLayout = errors.New("model: DeepSeek V4.1 paged KV cache layout is invalid")
