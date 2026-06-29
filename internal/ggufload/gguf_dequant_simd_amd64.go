//go:build amd64

package ggufload

import (
	"encoding/binary"
	"os"
)

const (
	ggufLoadTierScalar = iota
	ggufLoadTierAVX2
)

var ggufLoadDequantTier = resolveGGUFLoadDequantTier()

func resolveGGUFLoadDequantTier() int {
	if os.Getenv("FAK_QKERNEL") == "scalar" {
		return ggufLoadTierScalar
	}
	if ggufDetectAVX2() {
		return ggufLoadTierAVX2
	}
	return ggufLoadTierScalar
}

func ggufLoadDequantAVX2() bool {
	return ggufLoadDequantTier >= ggufLoadTierAVX2
}

//go:noescape
func ggufCPUID(eaxArg, ecxArg uint32) (eax, ebx, ecx, edx uint32)

//go:noescape
func ggufXGETBV() (eax, edx uint32)

func ggufDetectAVX2() bool {
	_, _, ecx1, _ := ggufCPUID(1, 0)
	const osxsave = 1 << 27
	const avx = 1 << 28
	if ecx1&(osxsave|avx) != osxsave|avx {
		return false
	}
	xcr0, _ := ggufXGETBV()
	if xcr0&0x6 != 0x6 {
		return false
	}
	_, ebx7, _, _ := ggufCPUID(7, 0)
	const avx2 = 1 << 5
	return ebx7&avx2 != 0
}

//go:noescape
func ggufDequantQ4KLoAVX2(dst *float32, q *byte, scale, min float32)

//go:noescape
func ggufDequantQ4KHiAVX2(dst *float32, q *byte, scale, min float32)

//go:noescape
func ggufDequantQ5KLoAVX2(dst *float32, ql, qh *byte, scale, min float32, highMask uint32)

//go:noescape
func ggufDequantQ5KHiAVX2(dst *float32, ql, qh *byte, scale, min float32, highMask uint32)

//go:noescape
func ggufDequantQ6KPos0AVX2(dst *float32, ql, qh *byte, scale float32)

//go:noescape
func ggufDequantQ6KPos1AVX2(dst *float32, ql, qh *byte, scale float32)

//go:noescape
func ggufDequantQ6KPos2AVX2(dst *float32, ql, qh *byte, scale float32)

//go:noescape
func ggufDequantQ6KPos3AVX2(dst *float32, ql, qh *byte, scale float32)

//go:noescape
func ggufDequantIQ3XXSGroupAVX2(dst *float32, packed uint64, signMask *uint32, scale float32)

func dequantQ4KArch(out []float32, raw []byte) bool {
	if !ggufLoadDequantAVX2() || len(out) == 0 {
		return false
	}
	for block := 0; block < len(out)/qkK; block++ {
		base := block * blockQ4KBytes
		d := f16At(raw, base)
		min := f16At(raw, base+2)
		scales := raw[base+4 : base+4+kScaleSize]
		q := raw[base+4+kScaleSize : base+blockQ4KBytes]
		qi := 0
		is := 0
		yi := block * qkK
		for j := 0; j < qkK; j += 64 {
			d1, m1, d2, m2 := scaleMinPairK4(d, min, is, scales)
			ggufDequantQ4KLoAVX2(&out[yi+j], &q[qi], d1, m1)
			ggufDequantQ4KHiAVX2(&out[yi+j+32], &q[qi], d2, m2)
			qi += 32
			is += 2
		}
	}
	return true
}

func dequantQ5KArch(out []float32, raw []byte) bool {
	if !ggufLoadDequantAVX2() || len(out) == 0 {
		return false
	}
	for block := 0; block < len(out)/qkK; block++ {
		base := block * blockQ5KBytes
		d := f16At(raw, base)
		min := f16At(raw, base+2)
		scales := raw[base+4 : base+4+kScaleSize]
		qh := raw[base+4+kScaleSize : base+4+kScaleSize+qkK/8]
		ql := raw[base+4+kScaleSize+qkK/8 : base+blockQ5KBytes]
		qi := 0
		is := 0
		u1, u2 := byte(1), byte(2)
		yi := block * qkK
		for j := 0; j < qkK; j += 64 {
			d1, m1, d2, m2 := scaleMinPairK4(d, min, is, scales)
			ggufDequantQ5KLoAVX2(&out[yi+j], &ql[qi], &qh[0], d1, m1, splatByte32(u1))
			ggufDequantQ5KHiAVX2(&out[yi+j+32], &ql[qi], &qh[0], d2, m2, splatByte32(u2))
			qi += 32
			is += 2
			u1 <<= 2
			u2 <<= 2
		}
	}
	return true
}

func dequantQ6KArch(out []float32, raw []byte) bool {
	if !ggufLoadDequantAVX2() || len(out) == 0 {
		return false
	}
	for block := 0; block < len(out)/qkK; block++ {
		base := block * blockQ6KBytes
		ql := raw[base : base+qkK/2]
		qh := raw[base+qkK/2 : base+qkK/2+qkK/4]
		scales := raw[base+qkK/2+qkK/4 : base+qkK/2+qkK/4+qkK/16]
		d := f16At(raw, base+blockQ6KBytes-2)
		yi := block * qkK
		qlOff, qhOff, scOff := 0, 0, 0
		for n := 0; n < qkK; n += 128 {
			for is := 0; is < 2; is++ {
				ggufDequantQ6KPos0AVX2(&out[yi+n+is*16], &ql[qlOff+is*16], &qh[qhOff+is*16], d*float32(int8(scales[scOff+is+0])))
				ggufDequantQ6KPos1AVX2(&out[yi+n+32+is*16], &ql[qlOff+32+is*16], &qh[qhOff+is*16], d*float32(int8(scales[scOff+is+2])))
				ggufDequantQ6KPos2AVX2(&out[yi+n+64+is*16], &ql[qlOff+is*16], &qh[qhOff+is*16], d*float32(int8(scales[scOff+is+4])))
				ggufDequantQ6KPos3AVX2(&out[yi+n+96+is*16], &ql[qlOff+32+is*16], &qh[qhOff+is*16], d*float32(int8(scales[scOff+is+6])))
			}
			qlOff += 64
			qhOff += 32
			scOff += 8
		}
	}
	return true
}

var iq3xxsAVX2SignMasks = initIQ3XXSAVX2SignMasks()

func initIQ3XXSAVX2SignMasks() [256][8]uint32 {
	var masks [256][8]uint32
	for signs := range masks {
		for j := 0; j < 8; j++ {
			if signs&(1<<uint(j)) != 0 {
				masks[signs][j] = 0x80000000
			}
		}
	}
	return masks
}

func dequantIQ3XXSArch(out []float32, raw []byte) bool {
	if !ggufLoadDequantAVX2() || len(out) == 0 {
		return false
	}
	for block := 0; block < len(out)/qkK; block++ {
		base := block * blockIQ3XXSBytes
		d := f16At(raw, base)
		qs := raw[base+2 : base+2+qkK/4]
		sas := raw[base+2+qkK/4 : base+blockIQ3XXSBytes]
		dst := out[block*qkK:]
		for ib32 := 0; ib32 < qkK/32; ib32++ {
			aux32 := binary.LittleEndian.Uint32(sas[4*ib32:])
			db := d * (0.5 + float32(aux32>>28)) * 0.5
			gi := ib32 * 8
			off := ib32 * 32
			for l := 0; l < 4; l++ {
				signs := ksignsIQ2XS[(aux32>>(7*uint(l)))&127]
				g1 := iq3xxsGrid[qs[gi+2*l+0]]
				g2 := iq3xxsGrid[qs[gi+2*l+1]]
				packed := uint64(g1) | uint64(g2)<<32
				ggufDequantIQ3XXSGroupAVX2(&dst[off+l*8], packed, &iq3xxsAVX2SignMasks[signs][0], db)
			}
		}
	}
	return true
}

func splatByte32(b byte) uint32 {
	v := uint32(b)
	return v | v<<8 | v<<16 | v<<24
}
