package compute

import (
	"encoding/binary"
	"math"

	"github.com/anthony-chaudhary/fak/internal/kquantbits"
)

// quant_iq2_xxs.go — IQ2_XXS host-data entry + the cpu-ref IQ2_XXS matmul reference.
//
// IQ2_XXS is llama.cpp's 2.06-bpw i-quant: a 256-weight super-block is 66 bytes — f16 d
// (0..1) plus 64 codebook bytes (qs). Each group of 32 weights is decoded from two 32-bit
// little-endian words (aux0 = 8 grid indices, aux1 = 7-bit sign codes + a 4-bit scale in
// the high nibble). The magnitude is the iq2xxs_grid codebook byte, the sign comes from
// ksigns_iq2xs, and the per-group scale is d*(0.5+aux1>>28)*0.25.
//
// This file gives the compute seam a real IQ2_XXS host dtype and a CPU reference so the
// calibrated mixed-quant candidate (IQ2_XXS gate/up + Q2_K down) has a Backend-level
// peer on the agent host before any GPU kernel or quality comparison. The arithmetic is
// byte-for-byte model.dequantIQ2XXSScalar (internal/model/quant_kquant_iqtables.go): the
// dequant is duplicated here (compute cannot import model — model imports compute) exactly
// as the Q2_K/Q4_K references are.

// iq2xxsSuperBlock is the byte length of one 256-weight IQ2_XXS super-block (matches
// model.iq2xxsBlockBytes and ggufload). 2 (d f16) + 32*2 (qs) = 66.
const iq2xxsSuperBlock = 2 + 32*2

// iq2xxsSuper is the IQ2_XXS super-block element count (256). Every IQ2_XXS reduction dim
// is a multiple of it.
const iq2xxsSuper = 256

// NewIQ2XXS wraps raw IQ2_XXS super-block bytes (the verbatim GGUF byte stream, row-major:
// row o at raw[o*nblk*66:], super-block b within a row at +b*66) as a host Tensor of dtype
// IQ2_XXS. shape is [out, in] with in a multiple of 256; len(raw) must be
// out*(in/256)*66. The bytes ride in the HostBuffer.I8() view (one int8 per byte,
// value-preserving two's-complement reinterpret) — the same layout cpuBackend.MatMul reads
// and backend Upload copies resident.
func NewIQ2XXS(be Backend, shape []int, raw []byte) Tensor {
	if len(shape) != 2 || shape[0] <= 0 || shape[1] <= 0 || shape[1]%iq2xxsSuper != 0 {
		panic("compute: invalid IQ2_XXS shape")
	}
	// Bound each multiplication before checking storage: wrapped products can
	// otherwise admit an empty buffer for a large, nonempty tensor.
	blocks := shape[1] / iq2xxsSuper
	maxInt := int(^uint(0) >> 1)
	if blocks > maxInt/iq2xxsSuperBlock || shape[0] > maxInt/(blocks*iq2xxsSuperBlock) || len(raw) != shape[0]*blocks*iq2xxsSuperBlock {
		panic("compute: invalid IQ2_XXS byte length")
	}
	return newRawKQuant(be, IQ2_XXS, 2, iq2xxsSuperBlock, shape, raw)
}

// iq2xxsDequantSuperBlock writes the 256 weights of one 66-byte IQ2_XXS super-block into
// dst (len >= 256). Byte-for-byte model.dequantIQ2XXSScalar / llama.cpp
// dequantize_row_iq2_xxs factored to one super-block.
func iq2xxsDequantSuperBlock(dst []float32, blk []byte) {
	if len(blk) < iq2xxsSuperBlock {
		panic("compute: short IQ2_XXS super-block")
	}
	if len(dst) < iq2xxsSuper {
		panic("compute: short destination for IQ2_XXS dequant")
	}
	d := math.Float32frombits(kquantbits.F16BitsToF32Bits(binary.LittleEndian.Uint16(blk[0:])))
	qs := blk[2:iq2xxsSuperBlock]
	for ib32 := 0; ib32 < iq2xxsSuper/32; ib32++ {
		aux0 := binary.LittleEndian.Uint32(qs[8*ib32:])
		aux1 := binary.LittleEndian.Uint32(qs[8*ib32+4:])
		db := d * (0.5 + float32(aux1>>28)) * 0.25
		for l := 0; l < 4; l++ {
			grid := iq2xxsGrid[byte(aux0>>uint(8*l))]
			signs := ksignsIQ2XS[(aux1>>uint(7*l))&127]
			for j := 0; j < 8; j++ {
				value := float32(byte(grid >> uint(8*j)))
				if signs&(1<<uint(j)) != 0 {
					value = -value
				}
				dst[ib32*32+l*8+j] = db * value
			}
		}
	}
}

// iq2xxsRowDot computes one output element y[o] = dot(weight row, x) over an IQ2_XXS
// weight row — the exact per-row reduction of model.kQuantMatRowsRange: dequant each
// super-block, dot it against the matching 256-wide slice of x, summed in row order.
// raw is the [in/256 * 66]-byte weight row.
func iq2xxsRowDot(raw []byte, x []float32, scratch []float32) float32 {
	if len(raw)%iq2xxsSuperBlock != 0 {
		panic("compute: misaligned IQ2_XXS raw weight row")
	}
	blocks := len(raw) / iq2xxsSuperBlock
	if len(x) < blocks*iq2xxsSuper {
		panic("compute: short input vector for IQ2_XXS row dot")
	}
	if len(scratch) < iq2xxsSuper {
		panic("compute: short scratch buffer for IQ2_XXS row dot")
	}
	var sum float32
	for off, xi := 0, 0; off < len(raw); off, xi = off+iq2xxsSuperBlock, xi+iq2xxsSuper {
		iq2xxsDequantSuperBlock(scratch, raw[off:off+iq2xxsSuperBlock])
		xs := x[xi : xi+iq2xxsSuper]
		for j := 0; j < iq2xxsSuper; j++ {
			sum += scratch[j] * xs[j]
		}
	}
	return sum
}
