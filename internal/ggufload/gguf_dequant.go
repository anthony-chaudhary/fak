package ggufload

import (
	"encoding/binary"
	"fmt"
	"io"
	"math"
)

func alignment(meta map[string]Value) (uint64, error) {
	align := uint64(defaultAlign)
	if v, ok := meta["general.alignment"]; ok {
		got, ok := valueUint64(v)
		if !ok {
			return 0, fmt.Errorf("gguf: general.alignment is not an unsigned integer")
		}
		align = got
	}
	if align == 0 || align%8 != 0 {
		return 0, fmt.Errorf("gguf: invalid alignment %d", align)
	}
	return align, nil
}

func alignOffset(off, align uint64) uint64 {
	return off + (align-(off%align))%align
}

// tensorOnDiskBytes is the best-effort on-disk payload size of a tensor for load-progress
// accounting: tensorPayloadBytes, or 0 if its shape/type is not byte-sizable. It never
// errors — a 0 from an exotic tensor only understates the running GB, not the percentage.
func tensorOnDiskBytes(t TensorInfo) int64 {
	n, err := tensorPayloadBytes(t)
	if err != nil {
		return 0
	}
	return int64(n)
}

func tensorPayloadBytes(t TensorInfo) (uint64, error) {
	elems, err := tensorElems(t)
	if err != nil {
		return 0, err
	}
	switch t.Type {
	case TensorF32:
		return elems * 4, nil
	case TensorF16, TensorBF16:
		return elems * 2, nil
	case TensorQ4_0:
		if elems%qk4 != 0 {
			return 0, fmt.Errorf("gguf: tensor %s Q4_0 element count %d is not a multiple of %d", t.Name, elems, qk4)
		}
		return elems / qk4 * blockQ4_0Bytes, nil
	case TensorQ4_1:
		if elems%qk4 != 0 {
			return 0, fmt.Errorf("gguf: tensor %s Q4_1 element count %d is not a multiple of %d", t.Name, elems, qk4)
		}
		return elems / qk4 * blockQ4_1Bytes, nil
	case TensorQ5_0:
		if elems%qk5 != 0 {
			return 0, fmt.Errorf("gguf: tensor %s Q5_0 element count %d is not a multiple of %d", t.Name, elems, qk5)
		}
		return elems / qk5 * blockQ5_0Bytes, nil
	case TensorQ5_1:
		if elems%qk5 != 0 {
			return 0, fmt.Errorf("gguf: tensor %s Q5_1 element count %d is not a multiple of %d", t.Name, elems, qk5)
		}
		return elems / qk5 * blockQ5_1Bytes, nil
	case TensorQ8_0:
		if elems%qk8_0 != 0 {
			return 0, fmt.Errorf("gguf: tensor %s Q8_0 element count %d is not a multiple of %d", t.Name, elems, qk8_0)
		}
		return elems / qk8_0 * blockQ8_0Bytes, nil
	case TensorQ2_K:
		if elems%qkK != 0 {
			return 0, fmt.Errorf("gguf: tensor %s Q2_K element count %d is not a multiple of %d", t.Name, elems, qkK)
		}
		return elems / qkK * blockQ2KBytes, nil
	case TensorQ3_K:
		if elems%qkK != 0 {
			return 0, fmt.Errorf("gguf: tensor %s Q3_K element count %d is not a multiple of %d", t.Name, elems, qkK)
		}
		return elems / qkK * blockQ3KBytes, nil
	case TensorQ4_K:
		if elems%qkK != 0 {
			return 0, fmt.Errorf("gguf: tensor %s Q4_K element count %d is not a multiple of %d", t.Name, elems, qkK)
		}
		return elems / qkK * blockQ4KBytes, nil
	case TensorQ5_K:
		if elems%qkK != 0 {
			return 0, fmt.Errorf("gguf: tensor %s Q5_K element count %d is not a multiple of %d", t.Name, elems, qkK)
		}
		return elems / qkK * blockQ5KBytes, nil
	case TensorQ6_K:
		if elems%qkK != 0 {
			return 0, fmt.Errorf("gguf: tensor %s Q6_K element count %d is not a multiple of %d", t.Name, elems, qkK)
		}
		return elems / qkK * blockQ6KBytes, nil
	case TensorMXFP4:
		if elems%qkMXFP4 != 0 {
			return 0, fmt.Errorf("gguf: tensor %s MXFP4 element count %d is not a multiple of %d", t.Name, elems, qkMXFP4)
		}
		return elems / qkMXFP4 * blockMXFP4Bytes, nil
	default:
		return 0, fmt.Errorf("gguf: tensor %s type %d does not have a simple f32 payload", t.Name, t.Type)
	}
}

func tensorElems(t TensorInfo) (uint64, error) {
	if len(t.Dims) == 0 {
		return 0, fmt.Errorf("gguf: tensor %s has no dimensions", t.Name)
	}
	n := uint64(1)
	for _, d := range t.Dims {
		if d == 0 {
			return 0, fmt.Errorf("gguf: tensor %s has zero dimension", t.Name)
		}
		if n > math.MaxUint64/d {
			return 0, fmt.Errorf("gguf: tensor %s element count overflows uint64", t.Name)
		}
		n *= d
	}
	return n, nil
}

// reuseF32 returns a length-n float32 slice backed by buf when buf's capacity allows, else
// a fresh allocation. The caller overwrites every returned element, so the reused tail is
// not zeroed — and never leaks into the result, whose length is exactly n.
func reuseF32(buf []float32, n int) []float32 {
	if cap(buf) >= n {
		return buf[:n]
	}
	return make([]float32, n)
}

// dequantF32 decodes a GGUF tensor's raw payload into a freshly-allocated f32 slice.
func dequantF32(t TensorInfo, raw []byte) ([]float32, error) {
	return dequantF32Into(nil, t, raw)
}

// dequantF32Into decodes a GGUF tensor's raw payload to f32, writing into scratch when it
// has the capacity (else allocating). The dequant writes every returned element for every
// supported type, so the reused buffer's prior contents never leak. The returned slice
// aliases scratch's backing array on reuse, so a caller recycling one buffer across many
// tensors MUST finish consuming the result before the next dequantF32Into overwrites it.
// Passing nil always allocates — the historical dequantF32 behavior every other caller keeps.
//
// This is the GGUF->Q8 quant-on-load page-churn fix (#440): the quant-on-load path
// dequantizes each tensor only long enough to re-quantize it, so a 27B checkpoint's 800+
// throwaway elems*4 f32 buffers — each faulting in fresh zeroed pages the GC then unmaps —
// collapse to one reused arena grown to the largest tensor.
func dequantF32Into(scratch []float32, t TensorInfo, raw []byte) ([]float32, error) {
	elems, err := tensorElems(t)
	if err != nil {
		return nil, err
	}
	if elems > uint64(math.MaxInt) {
		return nil, fmt.Errorf("gguf: tensor %s element count overflows int", t.Name)
	}
	out := reuseF32(scratch, int(elems))
	switch t.Type {
	case TensorF32:
		if len(raw) != len(out)*4 {
			return nil, fmt.Errorf("gguf: tensor %s f32 payload has %d bytes, want %d", t.Name, len(raw), len(out)*4)
		}
		for i := range out {
			out[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[i*4:]))
		}
	case TensorF16:
		if len(raw) != len(out)*2 {
			return nil, fmt.Errorf("gguf: tensor %s f16 payload has %d bytes, want %d", t.Name, len(raw), len(out)*2)
		}
		for i := range out {
			out[i] = math.Float32frombits(f16bitsToF32bits(binary.LittleEndian.Uint16(raw[i*2:])))
		}
	case TensorBF16:
		if len(raw) != len(out)*2 {
			return nil, fmt.Errorf("gguf: tensor %s bf16 payload has %d bytes, want %d", t.Name, len(raw), len(out)*2)
		}
		for i := range out {
			out[i] = math.Float32frombits(uint32(binary.LittleEndian.Uint16(raw[i*2:])) << 16)
		}
	case TensorQ4_0:
		if elems%qk4 != 0 {
			return nil, fmt.Errorf("gguf: tensor %s Q4_0 element count %d is not a multiple of %d", t.Name, elems, qk4)
		}
		want := int(elems / qk4 * blockQ4_0Bytes)
		if len(raw) != want {
			return nil, fmt.Errorf("gguf: tensor %s Q4_0 payload has %d bytes, want %d", t.Name, len(raw), want)
		}
		dequantQ4_0(out, raw)
	case TensorQ4_1:
		if elems%qk4 != 0 {
			return nil, fmt.Errorf("gguf: tensor %s Q4_1 element count %d is not a multiple of %d", t.Name, elems, qk4)
		}
		want := int(elems / qk4 * blockQ4_1Bytes)
		if len(raw) != want {
			return nil, fmt.Errorf("gguf: tensor %s Q4_1 payload has %d bytes, want %d", t.Name, len(raw), want)
		}
		dequantQ4_1(out, raw)
	case TensorQ5_0:
		if elems%qk5 != 0 {
			return nil, fmt.Errorf("gguf: tensor %s Q5_0 element count %d is not a multiple of %d", t.Name, elems, qk5)
		}
		want := int(elems / qk5 * blockQ5_0Bytes)
		if len(raw) != want {
			return nil, fmt.Errorf("gguf: tensor %s Q5_0 payload has %d bytes, want %d", t.Name, len(raw), want)
		}
		dequantQ5_0(out, raw)
	case TensorQ5_1:
		if elems%qk5 != 0 {
			return nil, fmt.Errorf("gguf: tensor %s Q5_1 element count %d is not a multiple of %d", t.Name, elems, qk5)
		}
		want := int(elems / qk5 * blockQ5_1Bytes)
		if len(raw) != want {
			return nil, fmt.Errorf("gguf: tensor %s Q5_1 payload has %d bytes, want %d", t.Name, len(raw), want)
		}
		dequantQ5_1(out, raw)
	case TensorQ8_0:
		if elems%qk8_0 != 0 {
			return nil, fmt.Errorf("gguf: tensor %s Q8_0 element count %d is not a multiple of %d", t.Name, elems, qk8_0)
		}
		want := int(elems / qk8_0 * blockQ8_0Bytes)
		if len(raw) != want {
			return nil, fmt.Errorf("gguf: tensor %s Q8_0 payload has %d bytes, want %d", t.Name, len(raw), want)
		}
		for block := 0; block < int(elems)/qk8_0; block++ {
			base := block * blockQ8_0Bytes
			d := math.Float32frombits(f16bitsToF32bits(binary.LittleEndian.Uint16(raw[base:])))
			for j := 0; j < qk8_0; j++ {
				out[block*qk8_0+j] = float32(int8(raw[base+2+j])) * d
			}
		}
	case TensorQ2_K:
		if elems%qkK != 0 {
			return nil, fmt.Errorf("gguf: tensor %s Q2_K element count %d is not a multiple of %d", t.Name, elems, qkK)
		}
		want := int(elems / qkK * blockQ2KBytes)
		if len(raw) != want {
			return nil, fmt.Errorf("gguf: tensor %s Q2_K payload has %d bytes, want %d", t.Name, len(raw), want)
		}
		dequantQ2K(out, raw)
	case TensorQ3_K:
		if elems%qkK != 0 {
			return nil, fmt.Errorf("gguf: tensor %s Q3_K element count %d is not a multiple of %d", t.Name, elems, qkK)
		}
		want := int(elems / qkK * blockQ3KBytes)
		if len(raw) != want {
			return nil, fmt.Errorf("gguf: tensor %s Q3_K payload has %d bytes, want %d", t.Name, len(raw), want)
		}
		dequantQ3K(out, raw)
	case TensorQ4_K:
		if elems%qkK != 0 {
			return nil, fmt.Errorf("gguf: tensor %s Q4_K element count %d is not a multiple of %d", t.Name, elems, qkK)
		}
		want := int(elems / qkK * blockQ4KBytes)
		if len(raw) != want {
			return nil, fmt.Errorf("gguf: tensor %s Q4_K payload has %d bytes, want %d", t.Name, len(raw), want)
		}
		dequantQ4K(out, raw)
	case TensorQ5_K:
		if elems%qkK != 0 {
			return nil, fmt.Errorf("gguf: tensor %s Q5_K element count %d is not a multiple of %d", t.Name, elems, qkK)
		}
		want := int(elems / qkK * blockQ5KBytes)
		if len(raw) != want {
			return nil, fmt.Errorf("gguf: tensor %s Q5_K payload has %d bytes, want %d", t.Name, len(raw), want)
		}
		dequantQ5K(out, raw)
	case TensorQ6_K:
		if elems%qkK != 0 {
			return nil, fmt.Errorf("gguf: tensor %s Q6_K element count %d is not a multiple of %d", t.Name, elems, qkK)
		}
		want := int(elems / qkK * blockQ6KBytes)
		if len(raw) != want {
			return nil, fmt.Errorf("gguf: tensor %s Q6_K payload has %d bytes, want %d", t.Name, len(raw), want)
		}
		dequantQ6K(out, raw)
	case TensorMXFP4:
		if elems%qkMXFP4 != 0 {
			return nil, fmt.Errorf("gguf: tensor %s MXFP4 element count %d is not a multiple of %d", t.Name, elems, qkMXFP4)
		}
		want := int(elems / qkMXFP4 * blockMXFP4Bytes)
		if len(raw) != want {
			return nil, fmt.Errorf("gguf: tensor %s MXFP4 payload has %d bytes, want %d", t.Name, len(raw), want)
		}
		dequantMXFP4(out, raw)
	default:
		return nil, fmt.Errorf("gguf: tensor %s type %d cannot dequantize to f32 yet", t.Name, t.Type)
	}
	return out, nil
}

// dequantQ4_0 expands the legacy GGML Q4_0 32-element block. Each block is a
// little-endian f16 scale d followed by qk4/2 bytes of packed 4-bit codes (two
// nibbles per byte). The GGML layout (dequantize_row_q4_0) is interleaved: the low
// nibble of byte j is element j, the high nibble is element j+qk4/2, and each code is
// re-centered by -8 before scaling: y = (nibble-8)*d. This is the 4-bit sibling of
// dequantQ5_0 with no 5th high bit.
func dequantQ4_0(out []float32, raw []byte) {
	for block := 0; block < len(out)/qk4; block++ {
		base := block * blockQ4_0Bytes
		d := math.Float32frombits(f16bitsToF32bits(binary.LittleEndian.Uint16(raw[base:])))
		qs := raw[base+2 : base+blockQ4_0Bytes]
		yi := block * qk4
		for j := 0; j < qk4/2; j++ {
			x0 := int(qs[j]&0x0f) - 8
			x1 := int(qs[j]>>4) - 8
			out[yi+j] = float32(x0) * d
			out[yi+j+qk4/2] = float32(x1) * d
		}
	}
}

// kvaluesMXFP4 maps a 4-bit E2M1 (FP4) code to its value, stored as 2x the real
// FP4 magnitude so the table is exact integers; the ×0.5 that restores the true
// E2M1 values {0,.5,1,1.5,2,3,4,6} is folded into the E8M0 scale by e8m0ToF32Half
// (which yields 2^(e-128) rather than 2^(e-127)). This matches GGML's
// kvalues_mxfp4 + GGML_E8M0_TO_FP32_HALF pairing for gpt-oss weights.
var kvaluesMXFP4 = [16]float32{0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12}

// e8m0ToF32Half decodes an E8M0 shared-exponent scale byte to 2^(e-128) — the
// half-scaled power that pairs with the doubled kvaluesMXFP4 table so that
// kvaluesMXFP4[code] * e8m0ToF32Half(e) == fp4(code) * 2^(e-127).
func e8m0ToF32Half(e uint8) float32 {
	return float32(math.Ldexp(1, int(e)-128))
}

// dequantMXFP4 expands the MXFP4 (gpt-oss) 32-element block: a 1-byte E8M0 shared
// scale followed by qkMXFP4/2 bytes of packed 4-bit E2M1 codes. The GGML layout
// (dequantize_row_mxfp4) interleaves like Q4_0 — the low nibble of byte j is
// element j, the high nibble is element j+qkMXFP4/2 — and each code indexes the
// E2M1 value table scaled by the block's half-scaled E8M0 exponent.
func dequantMXFP4(out []float32, raw []byte) {
	for block := 0; block < len(out)/qkMXFP4; block++ {
		base := block * blockMXFP4Bytes
		d := e8m0ToF32Half(raw[base])
		qs := raw[base+1 : base+blockMXFP4Bytes]
		yi := block * qkMXFP4
		for j := 0; j < qkMXFP4/2; j++ {
			out[yi+j] = kvaluesMXFP4[qs[j]&0x0f] * d
			out[yi+j+qkMXFP4/2] = kvaluesMXFP4[qs[j]>>4] * d
		}
	}
}

// dequantQ4_1 expands the legacy GGML Q4_1 32-element block: a little-endian f16
// scale d, then a little-endian f16 min m, then qk4/2 bytes of packed 4-bit codes.
// The GGML layout (dequantize_row_q4_1) keeps the same low/high-nibble interleave as
// Q4_0 but the codes are NOT re-centered — they carry an affine min: y = nibble*d + m.
func dequantQ4_1(out []float32, raw []byte) {
	for block := 0; block < len(out)/qk4; block++ {
		base := block * blockQ4_1Bytes
		d := math.Float32frombits(f16bitsToF32bits(binary.LittleEndian.Uint16(raw[base:])))
		m := math.Float32frombits(f16bitsToF32bits(binary.LittleEndian.Uint16(raw[base+2:])))
		qs := raw[base+4 : base+blockQ4_1Bytes]
		yi := block * qk4
		for j := 0; j < qk4/2; j++ {
			x0 := int(qs[j] & 0x0f)
			x1 := int(qs[j] >> 4)
			out[yi+j] = float32(x0)*d + m
			out[yi+j+qk4/2] = float32(x1)*d + m
		}
	}
}

func dequantQ5_0(out []float32, raw []byte) {
	for block := 0; block < len(out)/qk5; block++ {
		base := block * blockQ5_0Bytes
		d := math.Float32frombits(f16bitsToF32bits(binary.LittleEndian.Uint16(raw[base:])))
		qh := binary.LittleEndian.Uint32(raw[base+2:])
		qs := raw[base+6 : base+blockQ5_0Bytes]
		yi := block * qk5
		for j := 0; j < qk5/2; j++ {
			xh0 := byte(((qh >> uint(j)) << 4) & 0x10)
			xh1 := byte((qh >> uint(j+12)) & 0x10)
			x0 := int((qs[j]&0x0f)|xh0) - 16
			x1 := int((qs[j]>>4)|xh1) - 16
			out[yi+j] = float32(x0) * d
			out[yi+j+qk5/2] = float32(x1) * d
		}
	}
}

func dequantQ5_1(out []float32, raw []byte) {
	for block := 0; block < len(out)/qk5; block++ {
		base := block * blockQ5_1Bytes
		d := math.Float32frombits(f16bitsToF32bits(binary.LittleEndian.Uint16(raw[base:])))
		m := math.Float32frombits(f16bitsToF32bits(binary.LittleEndian.Uint16(raw[base+2:])))
		qh := binary.LittleEndian.Uint32(raw[base+4:])
		qs := raw[base+8 : base+blockQ5_1Bytes]
		yi := block * qk5
		for j := 0; j < qk5/2; j++ {
			xh0 := byte(((qh >> uint(j)) << 4) & 0x10)
			xh1 := byte((qh >> uint(j+12)) & 0x10)
			x0 := int((qs[j] & 0x0f) | xh0)
			x1 := int((qs[j] >> 4) | xh1)
			out[yi+j] = float32(x0)*d + m
			out[yi+j+qk5/2] = float32(x1)*d + m
		}
	}
}

func dequantQ2K(out []float32, raw []byte) {
	for block := 0; block < len(out)/qkK; block++ {
		base := block * blockQ2KBytes
		scales := raw[base : base+qkK/16]
		q := raw[base+qkK/16 : base+qkK/16+qkK/4]
		dm := base + qkK/16 + qkK/4
		d := math.Float32frombits(f16bitsToF32bits(binary.LittleEndian.Uint16(raw[dm:])))
		min := math.Float32frombits(f16bitsToF32bits(binary.LittleEndian.Uint16(raw[dm+2:])))
		yi := block * qkK
		qi := 0
		is := 0
		for n := 0; n < qkK; n += 128 {
			shift := uint(0)
			for j := 0; j < 4; j++ {
				sc := scales[is]
				is++
				dl, ml := d*float32(sc&0x0f), min*float32(sc>>4)
				for l := 0; l < 16; l++ {
					out[yi+n+j*32+l] = dl*float32((q[qi+l]>>shift)&3) - ml
				}

				sc = scales[is]
				is++
				dl, ml = d*float32(sc&0x0f), min*float32(sc>>4)
				for l := 0; l < 16; l++ {
					out[yi+n+j*32+16+l] = dl*float32((q[qi+16+l]>>shift)&3) - ml
				}
				shift += 2
			}
			qi += 32
		}
	}
}

func dequantQ3K(out []float32, raw []byte) {
	for block := 0; block < len(out)/qkK; block++ {
		base := block * blockQ3KBytes
		hmask := raw[base : base+qkK/8]
		q := raw[base+qkK/8 : base+qkK/8+qkK/4]
		scales := unpackQ3KScales(raw[base+qkK/8+qkK/4 : base+qkK/8+qkK/4+kScaleSize])
		d := math.Float32frombits(f16bitsToF32bits(binary.LittleEndian.Uint16(raw[base+blockQ3KBytes-2:])))
		yi := block * qkK
		qi := 0
		is := 0
		mask := byte(1)
		for n := 0; n < qkK; n += 128 {
			shift := uint(0)
			for j := 0; j < 4; j++ {
				dl := d * float32(scales[is]-32)
				is++
				for l := 0; l < 16; l++ {
					code := int8((q[qi+l] >> shift) & 3)
					if hmask[l]&mask == 0 {
						code -= 4
					}
					out[yi+n+j*32+l] = dl * float32(code)
				}

				dl = d * float32(scales[is]-32)
				is++
				for l := 0; l < 16; l++ {
					code := int8((q[qi+16+l] >> shift) & 3)
					if hmask[16+l]&mask == 0 {
						code -= 4
					}
					out[yi+n+j*32+16+l] = dl * float32(code)
				}
				shift += 2
				mask <<= 1
			}
			qi += 32
		}
	}
}

func unpackQ3KScales(raw []byte) [16]int8 {
	const (
		kmask1 = uint32(0x03030303)
		kmask2 = uint32(0x0f0f0f0f)
	)
	aux0 := binary.LittleEndian.Uint32(raw[0:4])
	aux1 := binary.LittleEndian.Uint32(raw[4:8])
	aux2 := binary.LittleEndian.Uint32(raw[8:12])
	tmp := aux2
	words := [4]uint32{
		(aux0 & kmask2) | (((tmp >> 0) & kmask1) << 4),
		(aux1 & kmask2) | (((tmp >> 2) & kmask1) << 4),
		((aux0 >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4),
		((aux1 >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4),
	}
	var scales [16]int8
	for i, word := range words {
		for j := 0; j < 4; j++ {
			scales[i*4+j] = int8(byte(word >> (8 * j)))
		}
	}
	return scales
}

func dequantQ4K(out []float32, raw []byte) {
	for block := 0; block < len(out)/qkK; block++ {
		base := block * blockQ4KBytes
		d := math.Float32frombits(f16bitsToF32bits(binary.LittleEndian.Uint16(raw[base:])))
		min := math.Float32frombits(f16bitsToF32bits(binary.LittleEndian.Uint16(raw[base+2:])))
		scales := raw[base+4 : base+4+kScaleSize]
		q := raw[base+4+kScaleSize : base+blockQ4KBytes]
		qi := 0
		is := 0
		yi := block * qkK
		for j := 0; j < qkK; j += 64 {
			sc, m := getScaleMinK4(is, scales)
			d1, m1 := d*float32(sc), min*float32(m)
			sc, m = getScaleMinK4(is+1, scales)
			d2, m2 := d*float32(sc), min*float32(m)
			for l := 0; l < 32; l++ {
				out[yi+j+l] = d1*float32(q[qi+l]&0x0f) - m1
			}
			for l := 0; l < 32; l++ {
				out[yi+j+32+l] = d2*float32(q[qi+l]>>4) - m2
			}
			qi += 32
			is += 2
		}
	}
}

func getScaleMinK4(j int, q []byte) (scale, min uint8) {
	if j < 4 {
		return q[j] & 63, q[j+4] & 63
	}
	return (q[j+4] & 0x0f) | ((q[j-4] >> 6) << 4), (q[j+4] >> 4) | ((q[j] >> 6) << 4)
}

func dequantQ5K(out []float32, raw []byte) {
	for block := 0; block < len(out)/qkK; block++ {
		base := block * blockQ5KBytes
		d := math.Float32frombits(f16bitsToF32bits(binary.LittleEndian.Uint16(raw[base:])))
		min := math.Float32frombits(f16bitsToF32bits(binary.LittleEndian.Uint16(raw[base+2:])))
		scales := raw[base+4 : base+4+kScaleSize]
		qh := raw[base+4+kScaleSize : base+4+kScaleSize+qkK/8]
		ql := raw[base+4+kScaleSize+qkK/8 : base+blockQ5KBytes]
		qi := 0
		is := 0
		u1, u2 := byte(1), byte(2)
		yi := block * qkK
		for j := 0; j < qkK; j += 64 {
			sc, m := getScaleMinK4(is, scales)
			d1, m1 := d*float32(sc), min*float32(m)
			sc, m = getScaleMinK4(is+1, scales)
			d2, m2 := d*float32(sc), min*float32(m)
			for l := 0; l < 32; l++ {
				hi := byte(0)
				if qh[l]&u1 != 0 {
					hi = 16
				}
				out[yi+j+l] = d1*float32((ql[qi+l]&0x0f)+hi) - m1
			}
			for l := 0; l < 32; l++ {
				hi := byte(0)
				if qh[l]&u2 != 0 {
					hi = 16
				}
				out[yi+j+32+l] = d2*float32((ql[qi+l]>>4)+hi) - m2
			}
			qi += 32
			is += 2
			u1 <<= 2
			u2 <<= 2
		}
	}
}

func dequantQ6K(out []float32, raw []byte) {
	for block := 0; block < len(out)/qkK; block++ {
		base := block * blockQ6KBytes
		ql := raw[base : base+qkK/2]
		qh := raw[base+qkK/2 : base+qkK/2+qkK/4]
		scales := raw[base+qkK/2+qkK/4 : base+qkK/2+qkK/4+qkK/16]
		d := math.Float32frombits(f16bitsToF32bits(binary.LittleEndian.Uint16(raw[base+blockQ6KBytes-2:])))
		yi := block * qkK
		qlOff, qhOff, scOff := 0, 0, 0
		for n := 0; n < qkK; n += 128 {
			for l := 0; l < 32; l++ {
				is := l / 16
				q1 := int8((ql[qlOff+l+0]&0x0f)|(((qh[qhOff+l]>>0)&3)<<4)) - 32
				q2 := int8((ql[qlOff+l+32]&0x0f)|(((qh[qhOff+l]>>2)&3)<<4)) - 32
				q3 := int8((ql[qlOff+l+0]>>4)|(((qh[qhOff+l]>>4)&3)<<4)) - 32
				q4 := int8((ql[qlOff+l+32]>>4)|(((qh[qhOff+l]>>6)&3)<<4)) - 32
				out[yi+n+l+0] = d * float32(int8(scales[scOff+is+0])) * float32(q1)
				out[yi+n+l+32] = d * float32(int8(scales[scOff+is+2])) * float32(q2)
				out[yi+n+l+64] = d * float32(int8(scales[scOff+is+4])) * float32(q3)
				out[yi+n+l+96] = d * float32(int8(scales[scOff+is+6])) * float32(q4)
			}
			qlOff += 64
			qhOff += 32
			scOff += 8
		}
	}
}

func f16bitsToF32bits(h uint16) uint32 {
	sign := uint32(h&0x8000) << 16
	exp := int((h >> 10) & 0x1f)
	frac := uint32(h & 0x03ff)
	switch exp {
	case 0:
		if frac == 0 {
			return sign
		}
		exp = -14
		for frac&0x0400 == 0 {
			frac <<= 1
			exp--
		}
		frac &= 0x03ff
		return sign | uint32(exp+127)<<23 | frac<<13
	case 0x1f:
		return sign | 0x7f800000 | frac<<13
	default:
		return sign | uint32(exp-15+127)<<23 | frac<<13
	}
}

type countingReader struct {
	r io.Reader
	n int64
}

func (r *countingReader) readFull(b []byte) error {
	if _, err := io.ReadFull(r.r, b); err != nil {
		return err
	}
	r.n += int64(len(b))
	return nil
}

func (r *countingReader) u32() (uint32, error) {
	var b [4]byte
	if err := r.readFull(b[:]); err != nil {
		return 0, err
	}
	return binary.LittleEndian.Uint32(b[:]), nil
}

func (r *countingReader) u64() (uint64, error) {
	var b [8]byte
	if err := r.readFull(b[:]); err != nil {
		return 0, err
	}
	return binary.LittleEndian.Uint64(b[:]), nil
}

func (r *countingReader) str() (string, error) {
	n, err := r.u64()
	if err != nil {
		return "", err
	}
	if n > maxStringBytes {
		return "", fmt.Errorf("string too large: %d bytes", n)
	}
	b := make([]byte, int(n))
	if err := r.readFull(b); err != nil {
		return "", err
	}
	return string(b), nil
}

func (r *countingReader) valueType() (ValueType, error) {
	u, err := r.u32()
	return ValueType(u), err
}

func (r *countingReader) value(typ ValueType) (Value, error) {
	switch typ {
	case TypeUint8:
		var b [1]byte
		if err := r.readFull(b[:]); err != nil {
			return Value{}, err
		}
		return Value{Type: typ, Value: b[0]}, nil
	case TypeInt8:
		var b [1]byte
		if err := r.readFull(b[:]); err != nil {
			return Value{}, err
		}
		return Value{Type: typ, Value: int8(b[0])}, nil
	case TypeUint16:
		var b [2]byte
		if err := r.readFull(b[:]); err != nil {
			return Value{}, err
		}
		return Value{Type: typ, Value: binary.LittleEndian.Uint16(b[:])}, nil
	case TypeInt16:
		var b [2]byte
		if err := r.readFull(b[:]); err != nil {
			return Value{}, err
		}
		return Value{Type: typ, Value: int16(binary.LittleEndian.Uint16(b[:]))}, nil
	case TypeUint32:
		v, err := r.u32()
		return Value{Type: typ, Value: v}, err
	case TypeInt32:
		v, err := r.u32()
		return Value{Type: typ, Value: int32(v)}, err
	case TypeFloat32:
		v, err := r.u32()
		return Value{Type: typ, Value: math.Float32frombits(v)}, err
	case TypeBool:
		var b [1]byte
		if err := r.readFull(b[:]); err != nil {
			return Value{}, err
		}
		if b[0] > 1 {
			return Value{}, fmt.Errorf("invalid bool byte %d", b[0])
		}
		return Value{Type: typ, Value: b[0] == 1}, nil
	case TypeString:
		s, err := r.str()
		return Value{Type: typ, Value: s}, err
	case TypeArray:
		elem, err := r.valueType()
		if err != nil {
			return Value{}, err
		}
		n, err := r.u64()
		if err != nil {
			return Value{}, err
		}
		if n > uint64(math.MaxInt) {
			return Value{}, fmt.Errorf("array too large: %d elements", n)
		}
		items := make([]Value, int(n))
		for i := range items {
			items[i], err = r.value(elem)
			if err != nil {
				return Value{}, fmt.Errorf("array element %d: %w", i, err)
			}
		}
		return Value{Type: typ, Value: items}, nil
	case TypeUint64:
		v, err := r.u64()
		return Value{Type: typ, Value: v}, err
	case TypeInt64:
		v, err := r.u64()
		return Value{Type: typ, Value: int64(v)}, err
	case TypeFloat64:
		v, err := r.u64()
		return Value{Type: typ, Value: math.Float64frombits(v)}, err
	default:
		return Value{}, fmt.Errorf("unsupported value type %d", typ)
	}
}
