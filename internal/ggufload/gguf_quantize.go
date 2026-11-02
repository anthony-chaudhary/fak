package ggufload

// Modified Go translation of the reference Q4_K/Q5_K/Q6_K encoders and their
// coherent scale-search helpers in ggml-org/llama.cpp, ggml/src/ggml-quants.c at
// 83078fec0db82d6b5a00d9599062c38c39145755 (blob 1ebc50a763f16db909de37090da38cc8c0fdde94).
// Local adaptation adds bounded finite arithmetic and explicit wire serialization.
// The Q6 symmetric helper specializes the source's rmse_type=1, no-weight route.
// Weighted/imatrix and other helper modes are intentionally not public capability.
//
// MIT License
//
// Copyright (c) 2023-2026 The ggml authors
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
// The pinned C reference's half conversion derives from Maratyszcza/FP16; its
// full additional MIT notice is retained in docs/notes/GGUF-KQUANT-CONVERSION.md.
// kQuantHalf below uses integer IEEE rounding, without changing decode arithmetic.

import (
	"encoding/binary"
	"fmt"
	"math"
)

const kQuantGroupEpsilon float32 = 1e-15

// Explicit float32 conversion boundaries are intentional: Go permits fused
// arithmetic unless the intermediate rounding is observable. These operations
// preserve C evaluation order with contraction/fast-math disabled, including in
// discarded search candidates. No invalid intermediate is silently ignored.
type kQuantMath struct{ err error }

func (m *kQuantMath) checked(v float32) float32 {
	if math.Float32bits(v)&0x7f800000 == 0x7f800000 && m.err == nil {
		m.err = fmt.Errorf("nonfinite K-quant arithmetic; unsupported numerical envelope")
	}
	return v
}
func (m *kQuantMath) add(a, b float32) float32 { return m.checked(float32(a + b)) }
func (m *kQuantMath) sub(a, b float32) float32 { return m.checked(float32(a - b)) }
func (m *kQuantMath) mul(a, b float32) float32 { return m.checked(float32(a * b)) }
func (m *kQuantMath) div(a, b float32) float32 { return m.checked(float32(a / b)) }
func kQuantAbs(x float32) float32              { return math.Float32frombits(math.Float32bits(x) & 0x7fffffff) }

func (m *kQuantMath) nearest(v float32) int {
	if kQuantAbs(v) > 4194303 || math.Float32bits(v)&0x7f800000 == 0x7f800000 {
		if m.err == nil {
			m.err = fmt.Errorf("K-quant rounding argument outside reference envelope")
		}
		return 0
	}
	f := float32(v + 12582912)
	return int(math.Float32bits(f)&0x007fffff) - 0x00400000
}

// kQuantHalf is round-to-nearest, ties-to-even IEEE binary32 -> binary16.
// It preserves signed zero/subnormal rounding; scales are separately refused
// when this representation would be nonfinite.
func kQuantHalf(x float32) uint16 {
	b := math.Float32bits(x)
	sign := uint16(b>>16) & 0x8000
	exp := (b >> 23) & 255
	mant := b & 0x7fffff
	if exp == 255 {
		if mant != 0 {
			return sign | 0x7e00
		}
		return sign | 0x7c00
	}
	if exp > 142 {
		return sign | 0x7c00
	}
	if exp < 113 {
		if exp < 102 {
			return sign
		}
		v := mant | 0x800000
		shift := uint(126 - exp)
		q := v >> shift
		rem := v & ((1 << shift) - 1)
		half := uint32(1) << (shift - 1)
		if rem > half || (rem == half && q&1 != 0) {
			q++
		}
		return sign | uint16(q)
	}
	q := ((exp - 112) << 10) | (mant >> 13)
	rem := mant & 0x1fff
	if rem > 0x1000 || (rem == 0x1000 && q&1 != 0) {
		q++
	}
	return sign | uint16(q)
}

func (m *kQuantMath) half(x float32) uint16 {
	m.checked(x)
	h := kQuantHalf(x)
	if h&0x7c00 == 0x7c00 && m.err == nil {
		m.err = fmt.Errorf("K-quant scale is not representable as finite float16")
	}
	return h
}

func quantizeKRow(row []float32, typ TensorType) ([]byte, error) {
	if len(row) == 0 || len(row)%256 != 0 {
		return nil, fmt.Errorf("K-quant row must contain complete 256-value blocks")
	}
	size := 0
	switch typ {
	case TensorQ4_K:
		size = 144
	case TensorQ5_K:
		size = 176
	case TensorQ6_K:
		size = 210
	default:
		return nil, fmt.Errorf("unsupported K-quant target %s", typ)
	}
	for _, v := range row {
		a := kQuantAbs(v)
		if math.Float32bits(v)&0x7f800000 == 0x7f800000 || a > 65504 || (a != 0 && a < 1e-30) {
			return nil, fmt.Errorf("K-quant input outside finite envelope: zero or 1e-30 <= abs(x) <= 65504")
		}
	}
	out := make([]byte, len(row)/256*size)
	m := &kQuantMath{}
	for b := 0; b < len(row)/256; b++ {
		x := row[b*256 : (b+1)*256]
		dst := out[b*size : (b+1)*size]
		if typ == TensorQ6_K {
			m.q6(x, dst)
		} else {
			m.q45(x, dst, typ == TensorQ5_K)
		}
		if m.err != nil {
			return nil, m.err
		}
	}
	return out, nil
}

// qx is the source make_qx_quants(16,32,...,rmse_type=1,qw=NULL) route.
func (m *kQuantMath) qx(x []float32, codes []uint8) float32 {
	var amax, maxValue float32
	for _, v := range x {
		if a := kQuantAbs(v); a > amax {
			amax = a
			maxValue = v
		}
	}
	if amax < kQuantGroupEpsilon {
		clear(codes)
		return 0
	}
	iscale := m.div(-32, maxValue)
	var sumlx, suml2 float32
	for i, v := range x {
		l := max(-32, min(31, m.nearest(m.mul(iscale, v))))
		codes[i] = uint8(l + 32)
		w := m.mul(v, v)
		sumlx = m.add(sumlx, m.mul(m.mul(w, v), float32(l)))
		suml2 = m.add(suml2, m.mul(m.mul(w, float32(l)), float32(l)))
	}
	var scale float32
	if suml2 != 0 {
		scale = m.div(sumlx, suml2)
	}
	best := m.mul(scale, sumlx)
	for is := -9; is <= 9; is++ {
		if is == 0 {
			continue
		}
		iscale = m.div(-m.add(32, m.mul(0.1, float32(is))), maxValue)
		sumlx = 0
		suml2 = 0
		for _, v := range x {
			l := max(-32, min(31, m.nearest(m.mul(iscale, v))))
			w := m.mul(v, v)
			sumlx = m.add(sumlx, m.mul(m.mul(w, v), float32(l)))
			suml2 = m.add(suml2, m.mul(m.mul(w, float32(l)), float32(l)))
		}
		if suml2 > 0 && m.mul(sumlx, sumlx) > m.mul(best, suml2) {
			for i, v := range x {
				codes[i] = uint8(32 + max(-32, min(31, m.nearest(m.mul(iscale, v)))))
			}
			scale = m.div(sumlx, suml2)
			best = m.mul(scale, sumlx)
		}
	}
	return scale
}

// qkx2 retains the source weighted least-squares search with use_mad=false.
// Here weights are exactly those constructed by the unweighted row reference,
// not an importance matrix and not a synthesized substitute for another route.
func (m *kQuantMath) qkx2(x, weights []float32, codes []uint8, nmax int, rmin float32, nstep int) (float32, float32) {
	lo, hi := x[0], x[0]
	sumW := weights[0]
	sumX := m.mul(sumW, x[0])
	for i := 1; i < len(x); i++ {
		if x[i] < lo {
			lo = x[i]
		}
		if x[i] > hi {
			hi = x[i]
		}
		sumW = m.add(sumW, weights[i])
		sumX = m.add(sumX, m.mul(weights[i], x[i]))
	}
	if lo > 0 {
		lo = 0
	}
	if hi == lo {
		clear(codes)
		return 0, -lo
	}
	iscale := m.div(float32(nmax), m.sub(hi, lo))
	scale := m.div(1, iscale)
	var bestError float32
	for i, v := range x {
		l := max(0, min(nmax, m.nearest(m.mul(iscale, m.sub(v, lo)))))
		codes[i] = uint8(l)
		diff := m.sub(m.add(m.mul(scale, float32(l)), lo), v)
		bestError = m.add(bestError, m.mul(weights[i], m.mul(diff, diff)))
	}
	var aux [32]uint8
	for is := 0; is <= nstep; is++ {
		iscale = m.div(m.add(m.add(rmin, m.mul(0.1, float32(is))), float32(nmax)), m.sub(hi, lo))
		var sumL, sumL2, sumXL float32
		for i, v := range x {
			l := max(0, min(nmax, m.nearest(m.mul(iscale, m.sub(v, lo)))))
			aux[i] = uint8(l)
			w := weights[i]
			sumL = m.add(sumL, m.mul(w, float32(l)))
			sumL2 = m.add(sumL2, m.mul(m.mul(w, float32(l)), float32(l)))
			sumXL = m.add(sumXL, m.mul(m.mul(w, float32(l)), v))
		}
		determinant := m.sub(m.mul(sumW, sumL2), m.mul(sumL, sumL))
		if determinant > 0 {
			thisScale := m.div(m.sub(m.mul(sumW, sumXL), m.mul(sumX, sumL)), determinant)
			thisMin := m.div(m.sub(m.mul(sumL2, sumX), m.mul(sumL, sumXL)), determinant)
			if thisMin > 0 {
				thisMin = 0
				thisScale = m.div(sumXL, sumL2)
			}
			var currentError float32
			for i, v := range x {
				diff := m.sub(m.add(m.mul(thisScale, float32(aux[i])), thisMin), v)
				currentError = m.add(currentError, m.mul(weights[i], m.mul(diff, diff)))
			}
			if currentError < bestError {
				copy(codes, aux[:])
				bestError = currentError
				scale = thisScale
				lo = thisMin
			}
		}
	}
	return scale, -lo
}

func (m *kQuantMath) q45(x []float32, out []byte, q5 bool) {
	var codes [256]uint8
	var scales, mins [8]float32
	var weights [32]float32
	nmax, nstep := 15, 20
	rmin := float32(-1)
	if q5 {
		nmax = 31
		nstep = 15
		rmin = -0.5
	}
	var maxScale, maxMin float32
	for j := 0; j < 8; j++ {
		var sumX2 float32
		for _, v := range x[j*32 : (j+1)*32] {
			sumX2 = m.add(sumX2, m.mul(v, v))
		}
		av := m.checked(float32(math.Sqrt(float64(m.div(sumX2, 32)))))
		for l := range weights {
			weights[l] = m.add(av, kQuantAbs(x[j*32+l]))
		}
		scales[j], mins[j] = m.qkx2(x[j*32:(j+1)*32], weights[:], codes[j*32:(j+1)*32], nmax, rmin, nstep)
		if scales[j] > maxScale {
			maxScale = scales[j]
		}
		if mins[j] > maxMin {
			maxMin = mins[j]
		}
	}
	var invScale, invMin float32
	if maxScale > 0 {
		invScale = m.div(63, maxScale)
	}
	if maxMin > 0 {
		invMin = m.div(63, maxMin)
	}
	packed := out[4:16]
	for j := 0; j < 8; j++ {
		ls := min(uint8(63), uint8(m.nearest(m.mul(invScale, scales[j]))))
		lm := min(uint8(63), uint8(m.nearest(m.mul(invMin, mins[j]))))
		if j < 4 {
			packed[j] = ls
			packed[j+4] = lm
		} else {
			packed[j+4] = (ls & 15) | ((lm & 15) << 4)
			packed[j-4] |= (ls >> 4) << 6
			packed[j] |= (lm >> 4) << 6
		}
	}
	binary.LittleEndian.PutUint16(out, m.half(m.div(maxScale, 63)))
	binary.LittleEndian.PutUint16(out[2:], m.half(m.div(maxMin, 63)))
	for j := 0; j < 8; j++ {
		sc, mn := kQuantScaleMin(j, packed)
		d := m.mul(f16At(out, 0), float32(sc))
		if d == 0 {
			continue
		}
		dm := m.mul(f16At(out, 2), float32(mn))
		for i := 0; i < 32; i++ {
			codes[j*32+i] = uint8(max(0, min(nmax, m.nearest(m.div(m.add(x[j*32+i], dm), d)))))
		}
	}
	if !q5 {
		for j := 0; j < 4; j++ {
			for l := 0; l < 32; l++ {
				out[16+j*32+l] = codes[j*64+l] | (codes[j*64+l+32] << 4)
			}
		}
		return
	}
	for j := 0; j < 4; j++ {
		for l := 0; l < 32; l++ {
			l1, l2 := codes[j*64+l], codes[j*64+l+32]
			if l1 > 15 {
				l1 -= 16
				out[16+l] |= 1 << uint(j*2)
			}
			if l2 > 15 {
				l2 -= 16
				out[16+l] |= 2 << uint(j*2)
			}
			out[48+j*32+l] = l1 | (l2 << 4)
		}
	}
}

func kQuantScaleMin(j int, q []byte) (uint8, uint8) {
	if j < 4 {
		return q[j] & 63, q[j+4] & 63
	}
	return (q[j+4] & 15) | ((q[j-4] >> 6) << 4), (q[j+4] >> 4) | ((q[j] >> 6) << 4)
}

func (m *kQuantMath) q6(x []float32, out []byte) {
	var codes [256]uint8
	var scales [16]float32
	var maxScale, maxAbsScale float32
	for j := range scales {
		scales[j] = m.qx(x[j*16:(j+1)*16], codes[j*16:(j+1)*16])
		a := kQuantAbs(scales[j])
		if a > maxAbsScale {
			maxAbsScale = a
			maxScale = scales[j]
		}
	}
	if maxAbsScale < kQuantGroupEpsilon {
		clear(out)
		return
	}
	iscale := m.div(-128, maxScale)
	binary.LittleEndian.PutUint16(out[208:], m.half(m.div(1, iscale)))
	for j, s := range scales {
		out[192+j] = uint8(int8(min(127, m.nearest(m.mul(iscale, s)))))
	}
	for j := range scales {
		d := m.mul(f16At(out, 208), float32(int8(out[192+j])))
		if d == 0 {
			continue
		}
		for i := 0; i < 16; i++ {
			codes[j*16+i] = uint8(max(-32, min(31, m.nearest(m.div(x[j*16+i], d)))) + 32)
		}
	}
	for j := 0; j < 2; j++ {
		for l := 0; l < 32; l++ {
			a, b, c, d := codes[j*128+l], codes[j*128+l+32], codes[j*128+l+64], codes[j*128+l+96]
			out[j*64+l] = (a & 15) | ((c & 15) << 4)
			out[j*64+l+32] = (b & 15) | ((d & 15) << 4)
			out[128+j*32+l] = (a >> 4) | ((b >> 4) << 2) | ((c >> 4) << 4) | ((d >> 4) << 6)
		}
	}
}
