package compute

import (
	"encoding/binary"
	"math"

	"github.com/anthony-chaudhary/fak/internal/kquantbits"
)

// quant_rawiq.go — native-format llama.cpp i-quant weights for the device matvec path
// (ticket halo-qwen-default-perf 05f).
//
// A raw i-quant tensor is the verbatim GGUF super-block stream (row-major: row o at
// raw[o*nblk*blockBytes:]) carried in the HostBuffer.I8() view exactly like Q2_K/Q4_K. Each
// format registers its block geometry and a scalar CPU dequant here; the CPU reference
// backend multiplies through that dequant and a device backend either uploads the bytes
// verbatim for its native kernel or, when that kernel is unavailable, expands them to Q8_0
// through the same dequant (the pre-05f representation).

// rawIQFormat describes one native i-quant dtype.
type rawIQFormat struct {
	blockBytes int                           // bytes per 256-weight super-block
	dequant    func(dst []float32, b []byte) // scalar dequant of one super-block into dst[:256]
	vulkanID   int                           // fvk IQ format id (vulkan_shim.cpp FVK_IQ_*)
}

const rawIQSuper = 256

var rawIQFormats = map[Dtype]rawIQFormat{
	IQ4_XS:  {blockBytes: iq4xsSuperBlock, dequant: iq4xsDequantSuperBlock, vulkanID: 0},
	IQ3_XXS: {blockBytes: iq3xxsSuperBlock, dequant: iq3xxsDequantSuperBlock, vulkanID: 1},
	IQ2_S:   {blockBytes: iq2sSuperBlock, dequant: iq2sDequantSuperBlock, vulkanID: 2},
	IQ3_S:   {blockBytes: iq3sSuperBlock, dequant: iq3sDequantSuperBlock, vulkanID: 3},
	IQ2_XXS: {blockBytes: iq2xxsSuperBlock, dequant: iq2xxsDequantSuperBlock, vulkanID: 4},
	IQ2_XS:  {blockBytes: iq2xsSuperBlock, dequant: iq2xsDequantSuperBlock, vulkanID: 5},
	IQ1_S:   {blockBytes: iq1sSuperBlock, dequant: iq1sDequantSuperBlock, vulkanID: 6},
}

// IsRawIQ reports whether dt is a registered native i-quant dtype.
func IsRawIQ(dt Dtype) bool {
	_, ok := rawIQFormats[dt]
	return ok
}

func isRawIQ(dt Dtype) bool { return IsRawIQ(dt) }

// RawIQBlockBytes reports the super-block byte size of a registered native i-quant dtype.
func RawIQBlockBytes(dt Dtype) (int, bool) {
	f, ok := rawIQFormats[dt]
	return f.blockBytes, ok
}

// NewRawIQ wraps raw i-quant super-block bytes as a host Tensor of dtype dt with shape
// [out, in] (in a multiple of 256, len(raw) == out*(in/256)*blockBytes).
func NewRawIQ(be Backend, dt Dtype, shape []int, raw []byte) Tensor {
	f, ok := rawIQFormats[dt]
	if !ok {
		panic("compute: NewRawIQ unsupported dtype " + dt.String())
	}
	if len(shape) != 2 || shape[0] <= 0 || shape[1] <= 0 || shape[1]%rawIQSuper != 0 {
		panic("compute: invalid " + dt.String() + " shape")
	}
	blocks := shape[1] / rawIQSuper
	maxInt := int(^uint(0) >> 1)
	if blocks > maxInt/f.blockBytes || shape[0] > maxInt/(blocks*f.blockBytes) || len(raw) != shape[0]*blocks*f.blockBytes {
		panic("compute: invalid " + dt.String() + " byte length")
	}
	return newRawKQuant(be, dt, 4, f.blockBytes, shape, raw)
}

// NewIQ4XS wraps raw IQ4_XS super-block bytes as a host Tensor (see NewRawIQ).
func NewIQ4XS(be Backend, shape []int, raw []byte) Tensor { return NewRawIQ(be, IQ4_XS, shape, raw) }

// DequantRawIQ expands a raw i-quant [out,in] payload to row-major f32.
func DequantRawIQ(dt Dtype, out, in int, raw []byte) []float32 {
	f, ok := rawIQFormats[dt]
	if !ok {
		panic("compute: DequantRawIQ unsupported dtype " + dt.String())
	}
	n := out * in
	if in%rawIQSuper != 0 || len(raw) != n/rawIQSuper*f.blockBytes {
		panic("compute: DequantRawIQ byte length does not match shape")
	}
	dst := make([]float32, n)
	for b := 0; b < n/rawIQSuper; b++ {
		f.dequant(dst[b*rawIQSuper:(b+1)*rawIQSuper], raw[b*f.blockBytes:(b+1)*f.blockBytes])
	}
	return dst
}

// rawIQRowDot computes dot(weight row, x) for one raw i-quant row: dequant each
// super-block and dot it against the matching 256-wide slice of x, in row order.
func rawIQRowDot(f rawIQFormat, raw []byte, x []float32, scratch []float32) float32 {
	var sum float32
	for off, xi := 0, 0; off < len(raw); off, xi = off+f.blockBytes, xi+rawIQSuper {
		f.dequant(scratch, raw[off:off+f.blockBytes])
		xs := x[xi : xi+rawIQSuper]
		for j := 0; j < rawIQSuper; j++ {
			sum += scratch[j] * xs[j]
		}
	}
	return sum
}

func f16At(b []byte) float32 {
	return math.Float32frombits(kquantbits.F16BitsToF32Bits(binary.LittleEndian.Uint16(b)))
}

// ---- IQ4_XS -------------------------------------------------------------------------

// iq4xsSuperBlock: f16 d + u16 scales_h + scales_l[4] + qs[128].
const iq4xsSuperBlock = 2 + 2 + 4 + 128

var kvaluesIQ4NL = [16]float32{-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113}

// iq4xsDequantSuperBlock is llama.cpp dequantize_row_iq4_xs for one super-block
// (byte-for-byte model.iq4xsDequantSuperBlock; compute cannot import model).
func iq4xsDequantSuperBlock(dst []float32, blk []byte) {
	d := f16At(blk[0:])
	scalesH := binary.LittleEndian.Uint16(blk[2:])
	scalesL := blk[4:8]
	qs := blk[8:iq4xsSuperBlock]
	for ib := 0; ib < 8; ib++ {
		lo := int(scalesL[ib/2]>>(4*uint(ib%2))) & 0x0f
		hi := int((scalesH >> (2 * uint(ib))) & 3)
		dl := d * float32((lo|hi<<4)-32)
		sub := qs[ib*16 : ib*16+16]
		for j := 0; j < 16; j++ {
			dst[ib*32+j] = dl * kvaluesIQ4NL[sub[j]&0x0f]
			dst[ib*32+j+16] = dl * kvaluesIQ4NL[sub[j]>>4]
		}
	}
}

// ---- IQ3_XXS ------------------------------------------------------------------------

// iq3xxsSuperBlock: f16 d + qs[64] grid indices + 8 u32 sign/scale words.
const iq3xxsSuperBlock = 2 + 64 + 32

// iq3xxsDequantSuperBlock is llama.cpp dequantize_row_iq3_xxs for one super-block
// (byte-for-byte model.iq3xxsDequantSuperBlock).
func iq3xxsDequantSuperBlock(dst []float32, blk []byte) {
	d := f16At(blk[0:])
	qs := blk[2:66]
	sas := blk[66:iq3xxsSuperBlock]
	for ib32 := 0; ib32 < 8; ib32++ {
		aux32 := binary.LittleEndian.Uint32(sas[4*ib32:])
		db := d * (0.5 + float32(aux32>>28)) * 0.5
		for l := 0; l < 4; l++ {
			signs := ksignsIQ2XS[(aux32>>(7*uint(l)))&127]
			g1 := iq3xxsGrid[qs[8*ib32+2*l]]
			g2 := iq3xxsGrid[qs[8*ib32+2*l+1]]
			for j := 0; j < 4; j++ {
				v1 := float32(byte(g1 >> (8 * uint(j))))
				v2 := float32(byte(g2 >> (8 * uint(j))))
				if signs&(1<<uint(j)) != 0 {
					v1 = -v1
				}
				if signs&(1<<uint(j+4)) != 0 {
					v2 = -v2
				}
				dst[ib32*32+l*8+j] = db * v1
				dst[ib32*32+l*8+j+4] = db * v2
			}
		}
	}
}

// ---- IQ2_S --------------------------------------------------------------------------

// iq2sSuperBlock: f16 d + qs[32] + signs[32] + qh[8] + scales[8].
const iq2sSuperBlock = 2 + 32 + 32 + 8 + 8

// iq2sDequantSuperBlock is llama.cpp dequantize_row_iq2_s for one super-block
// (byte-for-byte model.dequantIQ2SScalar).
func iq2sDequantSuperBlock(dst []float32, blk []byte) {
	d := f16At(blk[0:])
	qs := blk[2:34]
	signs := blk[34:66]
	qh := blk[66:74]
	scales := blk[74:iq2sSuperBlock]
	for ib32 := 0; ib32 < 8; ib32++ {
		db := [2]float32{
			d * (0.5 + float32(scales[ib32]&0x0f)) * 0.25,
			d * (0.5 + float32(scales[ib32]>>4)) * 0.25,
		}
		for l := 0; l < 4; l++ {
			index := uint16(qs[4*ib32+l]) | (uint16(qh[ib32])<<uint(8-2*l))&0x300
			grid := iq2SGrid[index]
			sign := signs[4*ib32+l]
			for j := 0; j < 8; j++ {
				v := float32(byte(grid >> uint(8*j)))
				if sign&(1<<uint(j)) != 0 {
					v = -v
				}
				dst[ib32*32+l*8+j] = db[l/2] * v
			}
		}
	}
}

// ---- IQ3_S --------------------------------------------------------------------------

// iq3sSuperBlock: f16 d + qs[64] + qh[8] + signs[32] + scales[4].
const iq3sSuperBlock = 2 + 64 + 8 + 32 + 4

// iq3sDequantSuperBlock is llama.cpp dequantize_row_iq3_s for one super-block
// (byte-for-byte model.iq3sDequantSuperBlock).
func iq3sDequantSuperBlock(dst []float32, blk []byte) {
	d := f16At(blk[0:])
	qs := blk[2:66]
	qh := blk[66:74]
	signs := blk[74:106]
	scales := blk[106:iq3sSuperBlock]
	for ib32 := 0; ib32 < 8; ib32++ {
		nibble := scales[ib32/2] & 0x0f
		if ib32%2 != 0 {
			nibble = scales[ib32/2] >> 4
		}
		db := d * float32(1+2*int(nibble))
		high := qh[ib32]
		for l := 0; l < 4; l++ {
			idx1 := uint16(qs[8*ib32+2*l]) | (uint16(high)<<uint(8-2*l))&0x100
			idx2 := uint16(qs[8*ib32+2*l+1]) | (uint16(high)<<uint(7-2*l))&0x100
			g1, g2 := iq3SGrid[idx1], iq3SGrid[idx2]
			sign := signs[4*ib32+l]
			for j := 0; j < 4; j++ {
				v1 := float32(byte(g1 >> uint(8*j)))
				v2 := float32(byte(g2 >> uint(8*j)))
				if sign&(1<<uint(j)) != 0 {
					v1 = -v1
				}
				if sign&(1<<uint(j+4)) != 0 {
					v2 = -v2
				}
				dst[ib32*32+l*8+j] = db * v1
				dst[ib32*32+l*8+j+4] = db * v2
			}
		}
	}
}

// ---- IQ2_XS -------------------------------------------------------------------------

// iq2xsSuperBlock: f16 d + qs u16[32] + scales[8].
const iq2xsSuperBlock = 2 + 64 + 8

// iq2xsDequantSuperBlock is llama.cpp dequantize_row_iq2_xs for one super-block
// (byte-for-byte model.dequantIQ2XSScalar).
func iq2xsDequantSuperBlock(dst []float32, blk []byte) {
	d := f16At(blk[0:])
	qs := blk[2:66]
	scales := blk[66:iq2xsSuperBlock]
	for ib32 := 0; ib32 < 8; ib32++ {
		db := [2]float32{
			d * (0.5 + float32(scales[ib32]&0x0f)) * 0.25,
			d * (0.5 + float32(scales[ib32]>>4)) * 0.25,
		}
		for l := 0; l < 4; l++ {
			q := binary.LittleEndian.Uint16(qs[2*(4*ib32+l):])
			grid := iq2XSGrid[q&511]
			signs := ksignsIQ2XS[q>>9]
			for j := 0; j < 8; j++ {
				v := float32(byte(grid >> uint(8*j)))
				if signs&(1<<uint(j)) != 0 {
					v = -v
				}
				dst[ib32*32+l*8+j] = db[l/2] * v
			}
		}
	}
}

// ---- IQ1_S --------------------------------------------------------------------------

// iq1sSuperBlock: f16 d + qs[32] + qh u16[8].
const iq1sSuperBlock = 2 + 32 + 16

// iq1sDequantSuperBlock is llama.cpp dequantize_row_iq1_s for one super-block
// (byte-for-byte model.dequantIQ1SScalar).
func iq1sDequantSuperBlock(dst []float32, blk []byte) {
	d := f16At(blk[0:])
	qs := blk[2:34]
	qh := blk[34:iq1sSuperBlock]
	for ib32 := 0; ib32 < 8; ib32++ {
		high := binary.LittleEndian.Uint16(qh[2*ib32:])
		dl := d * float32(2*((high>>12)&7)+1)
		delta := float32(0.125)
		if high&0x8000 != 0 {
			delta = -delta
		}
		for l := 0; l < 4; l++ {
			index := uint16(qs[4*ib32+l]) | (((high >> uint(3*l)) & 7) << 8)
			grid := iq1SGrid[index]
			for j := 0; j < 8; j++ {
				dst[ib32*32+l*8+j] = dl * (float32(int8(byte(grid>>uint(8*j)))) + delta)
			}
		}
	}
}
