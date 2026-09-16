package compute

import (
	"encoding/binary"
	"math"

	"github.com/anthony-chaudhary/fak/internal/kquantbits"
)

// quant_q3k.go — Q3_K host-data entry + the cpu-ref Q3_K matmul reference.
//
// Q3_K is llama.cpp's 3-bit k-quant: a 256-weight super-block is 110 bytes — 32 bytes of
// high-bit masks (hmask), 64 bytes of packed 2-bit low codes, 12 bytes of packed 6-bit
// sub-scales, and an f16 super-scale d in the final 2 bytes. It is the encoding the
// published DeepSeek V4.1 Flash Q2_K artifact stores its ffn_down_exps routed-expert slabs
// in, so without a compute tensor kind the R5 streamed routed-expert tier cannot stage those
// slabs and the whole expert set is refused on a one-Halo 62 GiB target (see fak#13149).
//
// The arithmetic is byte-for-byte model.q3kDequantSuperBlock (internal/model/quant_q3k_iq3s.go,
// the layout oracle ggml-org/llama.cpp dequantize_row_q3_K). compute cannot import model
// (model imports compute), so the scalar dequant is duplicated here exactly as the Q4_K/Q2_K
// compute dequant already duplicates the model arithmetic. This is the Reference (max|Δ|=0 vs
// model.DequantQ3K); no device backend serves Q3_K yet, so every one must decline cleanly.

// q3kSuperBlock is the byte length of one 256-weight Q3_K super-block (matches
// model.q3kBlockBytes and ggufload). 32 (hmask) + 64 (low codes) + 12 (scales) + 2 (d f16) = 110.
const q3kSuperBlock = 110

// q3kSuper is the Q3_K super-block element count (256). Every Q3_K reduction dim is a multiple of it.
const q3kSuper = 256

// NewQ3K wraps raw Q3_K super-block bytes (the verbatim GGUF byte stream, row-major: row o at
// raw[o*nblk*110:], super-block b within a row at +b*110) as a host Tensor of dtype Q3_K. shape
// is [out, in] with in a multiple of 256; len(raw) must be out*(in/256)*110. The bytes ride in
// the HostBuffer.I8() view (one int8 per byte, value-preserving two's-complement reinterpret) —
// the same layout cpuBackend.MatMul reads.
func NewQ3K(be Backend, shape []int, raw []byte) Tensor {
	if len(shape) != 2 || shape[0] <= 0 || shape[1] <= 0 || shape[1]%q3kSuper != 0 {
		panic("compute: invalid Q3_K shape")
	}
	// Bound each multiplication before checking storage: wrapped products can
	// otherwise admit an empty buffer for a large, nonempty tensor.
	blocks := shape[1] / q3kSuper
	maxInt := int(^uint(0) >> 1)
	if blocks > maxInt/q3kSuperBlock || shape[0] > maxInt/(blocks*q3kSuperBlock) || len(raw) != shape[0]*blocks*q3kSuperBlock {
		panic("compute: invalid Q3_K byte length")
	}
	return newRawKQuant(be, Q3_K, 3, q3kSuperBlock, shape, raw)
}

// q3kDequantSuperBlock writes the 256 weights of one 110-byte Q3_K super-block into dst
// (len >= 256). Byte-for-byte model.q3kDequantSuperBlock / llama.cpp dequantize_row_q3_K.
func q3kDequantSuperBlock(dst []float32, blk []byte) {
	if len(blk) < q3kSuperBlock {
		panic("compute: short Q3_K super-block")
	}
	if len(dst) < q3kSuper {
		panic("compute: short destination for Q3_K dequant")
	}
	hmask := blk[:q3kSuper/8]
	q := blk[q3kSuper/8 : q3kSuper/8+q3kSuper/4]
	scales := unpackQ3KScales(blk[q3kSuper/8+q3kSuper/4 : q3kSuper/8+q3kSuper/4+12])
	d := math.Float32frombits(kquantbits.F16BitsToF32Bits(binary.LittleEndian.Uint16(blk[q3kSuperBlock-2:])))
	qi := 0
	is := 0
	mask := byte(1)
	for n := 0; n < q3kSuper; n += 128 {
		shift := uint(0)
		for j := 0; j < 4; j++ {
			dl := d * float32(scales[is]-32)
			is++
			for l := 0; l < 16; l++ {
				code := int8((q[qi+l] >> shift) & 3)
				if hmask[l]&mask == 0 {
					code -= 4
				}
				dst[n+j*32+l] = dl * float32(code)
			}

			dl = d * float32(scales[is]-32)
			is++
			for l := 0; l < 16; l++ {
				code := int8((q[qi+16+l] >> shift) & 3)
				if hmask[16+l]&mask == 0 {
					code -= 4
				}
				dst[n+j*32+16+l] = dl * float32(code)
			}
			shift += 2
			mask <<= 1
		}
		qi += 32
	}
}

// unpackQ3KScales expands the 12 packed 6-bit sub-scale bytes into 16 signed 6-bit scales,
// byte-for-byte model.unpackQ3KScales (llama.cpp get_scale_min_k4-style Q3_K unpacking).
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

// DequantQ3K expands complete 256-weight/110-byte GGUF Q3_K blocks into out. It is the
// exported reference surface the R5 streamed routed-expert tier and any independent oracle
// compare against; it is byte-for-byte model.DequantQ3K (the layout oracle is
// ggml-org/llama.cpp dequantize_row_q3_K). Callers validate payload geometry before dispatch.
func DequantQ3K(out []float32, raw []byte) {
	for i := 0; i < len(out); i += q3kSuper {
		q3kDequantSuperBlock(out[i:i+q3kSuper], raw[:q3kSuperBlock])
		raw = raw[q3kSuperBlock:]
	}
}

// q3kRowDot computes one output element y[o] = dot(weight row, x) over a Q3_K weight row.
// raw is the [in/256 * 110]-byte weight row.
func q3kRowDot(raw []byte, x []float32, scratch []float32) float32 {
	if len(raw)%q3kSuperBlock != 0 {
		panic("compute: misaligned Q3_K raw weight row")
	}
	blocks := len(raw) / q3kSuperBlock
	if len(x) < blocks*q3kSuper {
		panic("compute: short input vector for Q3_K row dot")
	}
	if len(scratch) < q3kSuper {
		panic("compute: short scratch buffer for Q3_K row dot")
	}
	var sum float32
	for off, xi := 0, 0; off < len(raw); off, xi = off+q3kSuperBlock, xi+q3kSuper {
		q3kDequantSuperBlock(scratch, raw[off:off+q3kSuperBlock])
		xs := x[xi : xi+q3kSuper]
		for j := 0; j < q3kSuper; j++ {
			sum += scratch[j] * xs[j]
		}
	}
	return sum
}
