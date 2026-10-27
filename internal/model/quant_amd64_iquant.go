//go:build amd64

package model

import (
	"encoding/binary"
	"math"
)

// quant_amd64_iquant.go — the AVX2 port of the resident IQ3_XXS decode dequant. The scalar
// twin (iq3xxsDequantSuperBlock, quant_iquant.go) stays as the reference and the non-AVX2
// fallback; this drives the same math through the 8-lane iq3xxsGroupAVX2 kernel when the
// resolved tier has AVX2. It is a faithful port of internal/ggufload's dequantIQ3XXSArch (the
// load-path kernel); model cannot import ggufload (import cycle on the residentMatRows path,
// per quant_iquant.go), so the kernel and its sign-mask precompute are duplicated here.
// FAK_QKERNEL=scalar pins the scalar path (qtier resolves to tierScalar) for A/B measurement.

//go:noescape
func iq3xxsGroupAVX2(dst *float32, packed uint64, signMask *uint32, scale float32)

// iq3xxsAVX2SignMasks[signs][lane] is 0x80000000 exactly when the scalar path would multiply
// that lane by -1 (signs bit `lane` set), else 0 — so a single VXORPS applies the same sign
// flip the scalar s1/s2 factors do. signs is a ksignsIQ2XS value: lanes 0..3 carry the g1
// bytes' signs (scalar j), lanes 4..7 the g2 bytes' signs (scalar j+4).
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

// iq3xxsDequantSuperBlockArch dequantizes one IQ3_XXS super-block through the AVX2 kernel,
// bit-identical to iq3xxsDequantSuperBlock. It returns false (so kQuantDequantSuperBlock falls
// back to the scalar reference) when the resolved tier lacks AVX2.
func iq3xxsDequantSuperBlockArch(dst []float32, blk []byte) bool {
	if qtier < tierAVX2 {
		return false
	}
	d := math.Float32frombits(F16BitsToF32Bits(binary.LittleEndian.Uint16(blk[0:])))
	qs := blk[2 : 2+qkK/4]
	sas := blk[2+qkK/4 : iq3xxsBlockBytes]
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
			iq3xxsGroupAVX2(&dst[off+l*8], packed, &iq3xxsAVX2SignMasks[signs][0], db)
		}
	}
	return true
}

func init() {
	iq2DequantSuperBlockArch = iq2DequantSuperBlockAVX2
}

// iq2DequantSuperBlockAVX2 reuses the existing eight-byte magnitude/scale/sign
// kernel for the IQ2 family. Block indices/scales mirror the in-tree scalar
// decoders in quant_kquant_iqtables.go (llama.cpp 6fe74980162af0ed5e559870d5deccafaa034e7c,
// MIT; full original notice and unchanged codebooks are retained in that file).
//
// For a finite f16 scale, every scaled magnitude remains finite in float32.
// Moving a lane's sign flip after that multiplication preserves every bit,
// including signed zero. Nonfinite scales retain the scalar path so NaN payload
// and sign behavior is not changed by the SIMD multiply-then-XOR formulation.
func iq2DequantSuperBlockAVX2(dst []float32, blk []byte, kind kQuantKind) bool {
	if qtier < tierAVX2 || len(dst) != qkK {
		return false
	}
	switch kind {
	case kindIQ2XXS, kindIQ2XS, kindIQ2S:
	default:
		return false
	}
	if len(blk) != kind.blockBytes() {
		return false
	}
	half := binary.LittleEndian.Uint16(blk)
	if half&0x7c00 == 0x7c00 {
		return false
	}
	d := math.Float32frombits(F16BitsToF32Bits(half))
	for ib32 := 0; ib32 < qkK/32; ib32++ {
		var db [2]float32
		var aux0, aux1 uint32
		if kind == kindIQ2XXS {
			aux0 = binary.LittleEndian.Uint32(blk[2+8*ib32:])
			aux1 = binary.LittleEndian.Uint32(blk[2+8*ib32+4:])
			db[0] = d * (0.5 + float32(aux1>>28)) * 0.25
			db[1] = db[0]
		} else {
			scale := blk[len(blk)-qkK/32+ib32]
			db[0] = d * (0.5 + float32(scale&0x0f)) * 0.25
			db[1] = d * (0.5 + float32(scale>>4)) * 0.25
		}
		for l := 0; l < 4; l++ {
			group := 4*ib32 + l
			var grid uint64
			var signs byte
			switch kind {
			case kindIQ2XXS:
				grid = iq2XXSGrid[byte(aux0>>uint(8*l))]
				signs = ksignsIQ2XS[(aux1>>uint(7*l))&127]
			case kindIQ2XS:
				q := binary.LittleEndian.Uint16(blk[2+2*group:])
				grid = iq2XSGrid[q&511]
				signs = ksignsIQ2XS[q>>9]
			case kindIQ2S:
				index := uint16(blk[2+group]) | (uint16(blk[2+qkK/4+ib32])<<uint(8-2*l))&0x300
				grid = iq2SGrid[index]
				signs = blk[2+qkK/8+group]
			}
			iq3xxsGroupAVX2(&dst[group*8], grid, &iq3xxsAVX2SignMasks[signs][0], db[l/2])
		}
	}
	return true
}
