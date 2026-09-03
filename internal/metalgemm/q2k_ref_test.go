// q2k_ref_test.go — CPU reference Q2_K dequantizer and GEMV for test oracles.
// Carries NO build tag: both the Apple-Silicon parity tests and the stub-build witness
// tests use this exact reference.

package metalgemm

import (
	"encoding/binary"
	"math"
)

func q2kF16Bits(h uint16) uint32 {
	sign := uint32(h&0x8000) << 16
	exp := (h >> 10) & 0x1f
	frac := uint32(h & 0x03ff)
	switch exp {
	case 0:
		if frac == 0 {
			return sign
		}
		e := int32(-14)
		for frac&0x0400 == 0 {
			frac <<= 1
			e--
		}
		frac &= 0x03ff
		return sign | uint32(e+127)<<23 | frac<<13
	case 0x1f:
		return sign | 0x7f800000 | frac<<13
	default:
		return sign | uint32(int32(exp)-15+127)<<23 | frac<<13
	}
}

// q2kDequantBlock dequantizes one 84-byte Q2_K super-block into 256 float32 elements.
// Matches the GGML_TYPE_Q2_K arithmetic in ggufload.dequantQ2KScalar byte-for-byte.
func q2kDequantBlock(dst []float32, blk []byte) {
	scales := blk[:16]
	q := blk[16:80]
	d := math.Float32frombits(q2kF16Bits(binary.LittleEndian.Uint16(blk[80:82])))
	min := math.Float32frombits(q2kF16Bits(binary.LittleEndian.Uint16(blk[82:84])))
	qi := 0
	is := 0
	for n := 0; n < Q2KBlockWeights; n += 128 {
		shift := uint(0)
		for j := 0; j < 4; j++ {
			sc0 := scales[is]
			is++
			dl0, ml0 := d*float32(sc0&0x0f), min*float32(sc0>>4)
			for l := 0; l < 16; l++ {
				dst[n+j*32+l] = dl0*float32((q[qi+l]>>shift)&3) - ml0
			}

			sc1 := scales[is]
			is++
			dl1, ml1 := d*float32(sc1&0x0f), min*float32(sc1>>4)
			for l := 0; l < 16; l++ {
				dst[n+j*32+16+l] = dl1*float32((q[qi+16+l]>>shift)&3) - ml1
			}
			shift += 2
		}
		qi += 32
	}
}

// q2kReference computes the exact matrix-vector product y[out] = W[out, in] * x[in]
// by dequantizing each block and computing the inner product.
func q2kReference(raw []byte, out, in int, x []float32) []float32 {
	nblk := in / Q2KBlockWeights
	y := make([]float32, out)
	blkWeights := make([]float32, Q2KBlockWeights)
	for o := 0; o < out; o++ {
		var acc float32
		for b := 0; b < nblk; b++ {
			base := (o*nblk + b) * Q2KBlockBytes
			q2kDequantBlock(blkWeights, raw[base:base+Q2KBlockBytes])
			xb := x[b*Q2KBlockWeights : (b+1)*Q2KBlockWeights]
			for i := 0; i < Q2KBlockWeights; i++ {
				acc += blkWeights[i] * xb[i]
			}
		}
		y[o] = acc
	}
	return y
}

// q2kTestRaw generates a deterministic valid Q2_K byte payload of shape [out, in].
// Sets d and dmin to modest non-zero f16 values to ensure finite, non-zero weights.
func q2kTestRaw(out, in int, seed uint64) []byte {
	if in%Q2KBlockWeights != 0 {
		panic("q2kTestRaw: in must be a multiple of 256")
	}
	nblk := in / Q2KBlockWeights
	raw := make([]byte, out*nblk*Q2KBlockBytes)
	state := seed
	for i := range raw {
		state ^= state << 13
		state ^= state >> 7
		state ^= state << 17
		raw[i] = byte(state >> 56)
	}
	for base := 0; base < len(raw); base += Q2KBlockBytes {
		// Valid normal half-precision floats: exponent in [0x04..0x1c], modest magnitudes (~0.01 - 1.0)
		raw[base+80] = byte(0x10 | (raw[base+80] & 0x0f))
		raw[base+81] = byte(0x38 | (raw[base+81] & 0x03)) // ~0.5
		raw[base+82] = byte(0x10 | (raw[base+82] & 0x0f))
		raw[base+83] = byte(0x34 | (raw[base+83] & 0x03)) // ~0.25
	}
	return raw
}
