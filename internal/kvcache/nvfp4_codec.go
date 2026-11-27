package kvcache

import (
	"fmt"
	"math"
)

var e2m1Magnitudes = [8]float32{0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0}

// e4m3fnPosValues maps positive finite OCP E4M3FN byte codes 0x00..0x7E to float32.
var e4m3fnPosValues = func() [127]float32 {
	var tbl [127]float32
	for c := 1; c <= 0x7E; c++ {
		e := (c >> 3) & 0xF
		m := c & 0x7
		if e == 0 {
			tbl[c] = float32(m) / 512.0
		} else {
			tbl[c] = float32(math.Ldexp(1.0+float64(m)/8.0, e-7))
		}
	}
	return tbl
}()

// EncodeE2M1Scalar quantizes a float32 to a 4-bit OCP E2M1 nibble using
// saturating round-to-nearest-even, matching cvt.rn.satfinite.e2m1x2.f32.
func EncodeE2M1Scalar(x float32) byte {
	bits := math.Float32bits(x)
	sign := byte((bits >> 28) & 0x8)
	ax := float32(math.Abs(float64(x)))
	var mag byte
	switch {
	case ax <= 0.25:
		mag = 0
	case ax < 0.75:
		mag = 1
	case ax <= 1.25:
		mag = 2
	case ax < 1.75:
		mag = 3
	case ax <= 2.5:
		mag = 4
	case ax < 3.5:
		mag = 5
	case ax <= 5.0:
		mag = 6
	default:
		mag = 7
	}
	return sign | mag
}

// DecodeE2M1Scalar decodes a 4-bit OCP E2M1 nibble to float32.
func DecodeE2M1Scalar(nib byte) float32 {
	v := e2m1Magnitudes[nib&0x7]
	if nib&0x8 != 0 {
		return -v
	}
	return v
}

// DecodeE4M3FN decodes a positive finite OCP E4M3FN scale byte (0x00..0x7E).
// Negative or NaN codes (0x7F or >= 0x80) decode to 0.
func DecodeE4M3FN(code byte) float32 {
	if int(code) >= len(e4m3fnPosValues) {
		return 0
	}
	return e4m3fnPosValues[code]
}

// EncodeE4M3FN encodes a non-negative float32 to the nearest positive finite
// OCP E4M3FN byte (0x00..0x7E) using round-to-nearest-even. Returns overflow=true
// if x is non-finite, negative, or exceeds NVFP4MaxE4M3FN (448.0).
func EncodeE4M3FN(x float32) (code byte, overflow bool) {
	if math.IsNaN(float64(x)) || math.IsInf(float64(x), 0) || x < 0 {
		return 0, true
	}
	if x == 0 {
		return 0, false
	}
	if x > NVFP4MaxE4M3FN {
		return 0x7E, true
	}
	// Binary search over the monotonic 127-entry table [0x00..0x7E].
	lo, hi := 0, len(e4m3fnPosValues)-1
	for lo < hi {
		mid := (lo + hi) / 2
		if e4m3fnPosValues[mid] < x {
			lo = mid + 1
		} else {
			hi = mid
		}
	}
	if lo == 0 {
		return 0, false
	}
	prev := e4m3fnPosValues[lo-1]
	curr := e4m3fnPosValues[lo]
	dPrev := x - prev
	dCurr := curr - x
	if dPrev < dCurr || (dPrev == dCurr && ((lo-1)&1) == 0) {
		return byte(lo - 1), false
	}
	return byte(lo), false
}

// QuantizeBlock16NVFP4 encodes a contiguous [tokens][kvHeads][headDim] float32
// tensor into packed block-16 NVFP4 codes ([tokens][kvHeads][headDim/2]) and
// OCP E4M3FN scales ([tokens][kvHeads][headDim/16]).
func QuantizeBlock16NVFP4(src []float32, kvHeads, headDim, tokens int) (codes, scales []byte, overflow bool, err error) {
	if tokens < 0 || kvHeads <= 0 || headDim <= 0 || headDim%NVFP4BlockSize != 0 {
		return nil, nil, false, fmt.Errorf("%w: invalid shape tokens=%d kvHeads=%d headDim=%d", ErrInvalidLayout, tokens, kvHeads, headDim)
	}
	wantElems := tokens * kvHeads * headDim
	if len(src) != wantElems {
		return nil, nil, false, fmt.Errorf("%w: got %d float32 elements, want %d", ErrInvalidLayout, len(src), wantElems)
	}
	blocksPerHead := headDim / NVFP4BlockSize
	totalBlocks := tokens * kvHeads * blocksPerHead
	codes = make([]byte, totalBlocks*(NVFP4BlockSize/2))
	scales = make([]byte, totalBlocks)

	for b := 0; b < totalBlocks; b++ {
		base := b * NVFP4BlockSize
		var amax float32
		nonFinite := false
		for i := 0; i < NVFP4BlockSize; i++ {
			v := src[base+i]
			if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
				nonFinite = true
				break
			}
			av := float32(math.Abs(float64(v)))
			if av > amax {
				amax = av
			}
		}
		if nonFinite || amax > NVFP4MaxBlockAbsmax {
			overflow = true
			continue
		}
		if amax < NVFP4MinAbsmax {
			scales[b] = 0
			continue
		}
		scCode, scOverflow := EncodeE4M3FN(amax / NVFP4MaxE2M1)
		if scOverflow {
			overflow = true
		}
		scales[b] = scCode
		effScale := DecodeE4M3FN(scCode)
		if effScale == 0 {
			continue
		}
		invScale := 1.0 / effScale
		codeBase := b * (NVFP4BlockSize / 2)
		for i := 0; i < NVFP4BlockSize/2; i++ {
			loNib := EncodeE2M1Scalar(src[base+2*i] * invScale)
			hiNib := EncodeE2M1Scalar(src[base+2*i+1] * invScale)
			codes[codeBase+i] = loNib | (hiNib << 4)
		}
	}
	return codes, scales, overflow, nil
}

// DequantizeBlock16NVFP4 decodes packed block-16 NVFP4 codes and E4M3FN scales
// back to a contiguous [tokens][kvHeads][headDim] float32 slice.
func DequantizeBlock16NVFP4(codes, scales []byte, kvHeads, headDim, tokens int) ([]float32, error) {
	if tokens < 0 || kvHeads <= 0 || headDim <= 0 || headDim%NVFP4BlockSize != 0 {
		return nil, fmt.Errorf("%w: invalid shape tokens=%d kvHeads=%d headDim=%d", ErrInvalidLayout, tokens, kvHeads, headDim)
	}
	blocksPerHead := headDim / NVFP4BlockSize
	totalBlocks := tokens * kvHeads * blocksPerHead
	wantCodes := totalBlocks * (NVFP4BlockSize / 2)
	if len(codes) != wantCodes || len(scales) != totalBlocks {
		return nil, fmt.Errorf("%w: got codes=%d scales=%d, want codes=%d scales=%d", ErrInvalidLayout, len(codes), len(scales), wantCodes, totalBlocks)
	}
	out := make([]float32, tokens*kvHeads*headDim)
	for b := 0; b < totalBlocks; b++ {
		sc := DecodeE4M3FN(scales[b])
		base := b * NVFP4BlockSize
		codeBase := b * (NVFP4BlockSize / 2)
		for i := 0; i < NVFP4BlockSize/2; i++ {
			packed := codes[codeBase+i]
			out[base+2*i] = DecodeE2M1Scalar(packed&0x0F) * sc
			out[base+2*i+1] = DecodeE2M1Scalar(packed>>4) * sc
		}
	}
	return out, nil
}
