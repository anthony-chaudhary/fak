package compute

import (
	"encoding/binary"
	"math"

	"github.com/anthony-chaudhary/fak/internal/kquantbits"
)

// q2kSuperBlock is the byte length of one 256-weight Q2_K super-block (matches model.q2kBlockBytes
// and ggufload). 16 (scales) + 64 (quants) + 2 (d f16) + 2 (dmin f16) = 84.
const q2kSuperBlock = 84

// q2kSuper is the Q2_K super-block element count (256). Every Q2_K reduction dim is a multiple of it.
const q2kSuper = 256

// NewQ2K wraps raw Q2_K super-block bytes (the verbatim GGUF byte stream, row-major: row o at
// raw[o*nblk*84:], super-block b within a row at +b*84) as a host Tensor of dtype Q2_K. shape is
// [out, in] with in a multiple of 256; len(raw) must be out*(in/256)*84. The bytes ride in the
// HostBuffer.I8() view (one int8 per byte, value-preserving two's-complement reinterpret) — the
// same layout cpuBackend.MatMul reads and backend Upload copies resident.
func NewQ2K(be Backend, shape []int, raw []byte) Tensor {
	return newRawKQuant(be, Q2_K, 2, q2kSuperBlock, shape, raw)
}

// q2kDequantSuperBlock writes the 256 weights of one 84-byte Q2_K super-block into dst (len >= 256).
// Byte-for-byte ggufload.dequantQ2KScalar factored to one super-block.
func q2kDequantSuperBlock(dst []float32, blk []byte) {
	scales := blk[:q2kSuper/16]
	q := blk[q2kSuper/16 : q2kSuper/16+q2kSuper/4]
	dm := q2kSuper/16 + q2kSuper/4
	d := math.Float32frombits(kquantbits.F16BitsToF32Bits(binary.LittleEndian.Uint16(blk[dm:])))
	min := math.Float32frombits(kquantbits.F16BitsToF32Bits(binary.LittleEndian.Uint16(blk[dm+2:])))
	qi := 0
	is := 0
	for n := 0; n < q2kSuper; n += 128 {
		shift := uint(0)
		for j := 0; j < 4; j++ {
			sc := scales[is]
			is++
			dl, ml := d*float32(sc&0x0f), min*float32(sc>>4)
			table := [4]float32{
				0 - ml,
				dl - ml,
				dl*2 - ml,
				dl*3 - ml,
			}
			for l := 0; l < 16; l++ {
				dst[n+j*32+l] = table[(q[qi+l]>>shift)&3]
			}

			sc = scales[is]
			is++
			dl, ml = d*float32(sc&0x0f), min*float32(sc>>4)
			table = [4]float32{
				0 - ml,
				dl - ml,
				dl*2 - ml,
				dl*3 - ml,
			}
			for l := 0; l < 16; l++ {
				dst[n+j*32+16+l] = table[(q[qi+16+l]>>shift)&3]
			}
			shift += 2
		}
		qi += 32
	}
}

func dequantQ2K(dst []float32, blk []byte) {
	q2kDequantSuperBlock(dst, blk)
}

// q2kRowDot computes one output element y[o] = dot(weight row, x) over a Q2_K weight row.
// raw is the [in/256 * 84]-byte weight row.
func q2kRowDot(raw []byte, x []float32, scratch []float32) float32 {
	var sum float32
	for off, xi := 0, 0; off < len(raw); off, xi = off+q2kSuperBlock, xi+q2kSuper {
		q2kDequantSuperBlock(scratch, raw[off:off+q2kSuperBlock])
		xs := x[xi : xi+q2kSuper]
		for j := 0; j < q2kSuper; j++ {
			sum += scratch[j] * xs[j]
		}
	}
	return sum
}
