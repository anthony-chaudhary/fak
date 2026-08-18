package compute

import (
	"encoding/binary"
	"math"

	"github.com/anthony-chaudhary/fak/internal/kquantbits"
)

// quant_q4k.go — Q4_K host-data entry + the cpu-ref Q4_K matmul reference.
//
// Q4_K is llama.cpp's 4-bit k-quant: a 256-weight super-block is 144 bytes — f16 d (0..1),
// f16 dmin (2..3), 12 packed 6-bit sub-scale bytes (4..15), and 128 nibble-code bytes (16..143).
// The cuda backend already serves it natively (k_q4k_gemm, dequant fused into the GEMM tile,
// #485). This adds the CPU REFERENCE: cpuBackend.MatMul/BatchedMatMul on a Q4_K weight, so the
// quantized GLM-DSA device path (Session.glmDsaWeightHAL → backend.Upload(Q4_K) → backend.MatMul)
// is exercisable on the agent-host, and the cuda Q4_K kernel has a Backend-level Reference peer.
//
// The arithmetic is byte-for-byte model.q4kDequantSuperBlock + model.q4kMatRowsRange (the f32
// dequant path, not the int8-SDOT decode kernel): per super-block dequant (w = d·scale·code −
// dmin·min, getScaleMinK4 6-bit geometry), a four-accumulator (s0+s1)+(s2+s3) dot, summed across
// super-blocks in row order. So cpu-ref Q4_K MatMul is bit-identical to the model's f32 Q4_K GEMV
// — it is the Reference (max|Δ|=0 vs that path), and the cuda lane is its Approx peer
// (cudaQ4KCosineMin). The dequant is duplicated here (compute cannot import model — model imports
// compute) exactly as the cuda kernel duplicates getScaleMinK4_dev.

// q4kSuperBlock is the byte length of one 256-weight Q4_K super-block (matches model.q4kBlockBytes
// and ggufload). 2 (d f16) + 2 (dmin f16) + 12 (scales) + 128 (256 nibbles) = 144.
const q4kSuperBlock = 2 + 2 + 12 + q4kSuper/2

// q4kSuper is the Q4_K super-block element count (256). Every Q4_K reduction dim is a multiple of it.
const q4kSuper = 256

// NewQ4K wraps raw Q4_K super-block bytes (the verbatim GGUF byte stream, row-major: row o at
// raw[o*nblk*144:], super-block b within a row at +b*144) as a host Tensor of dtype Q4_K. shape is
// [out, in] with in a multiple of 256; len(raw) must be out*(in/256)*144. The bytes ride in the
// HostBuffer.I8() view (one int8 per byte, value-preserving two's-complement reinterpret) — the
// same layout cpuBackend.MatMul reads and the cuda backend's Upload(_, Q4_K) copies resident.
func NewQ4K(be Backend, shape []int, raw []byte) Tensor {
	i8 := make([]int8, len(raw))
	for i, b := range raw {
		i8[i] = int8(b)
	}
	q := &QuantSpec{Block: q4kSuper, Axis: 2, Bits: 4, Symmetric: false}
	return makeTensor(be, Q4_K, RowMajor, append([]int(nil), shape...), q, &hostBuf{i8: i8})
}

// NewQ5K and NewQ6K wrap verbatim GGUF k-quant rows without expanding them.
// Both formats contain 256 weights per super-block; their raw block widths are
// 176 and 210 bytes respectively.
func NewQ5K(be Backend, shape []int, raw []byte) Tensor {
	return newRawKQuant(be, Q5_K, 5, 176, shape, raw)
}
func NewQ6K(be Backend, shape []int, raw []byte) Tensor {
	return newRawKQuant(be, Q6_K, 6, 210, shape, raw)
}

func newRawKQuant(be Backend, dt Dtype, bits, blockBytes int, shape []int, raw []byte) Tensor {
	if len(shape) != 2 || shape[1]%q4kSuper != 0 || len(raw) != shape[0]*(shape[1]/q4kSuper)*blockBytes {
		panic("compute: invalid raw k-quant shape or byte length")
	}
	i8 := make([]int8, len(raw))
	for i, b := range raw {
		i8[i] = int8(b)
	}
	q := &QuantSpec{Block: q4kSuper, Axis: 2, Bits: uint8(bits), Symmetric: false}
	return makeTensor(be, dt, RowMajor, append([]int(nil), shape...), q, &hostBuf{i8: i8})
}

// i8AsBytes reinterprets the Q4_K HostBuffer's int8 view back to the raw super-block bytes
// (value-preserving: NewQ4K stored each byte as int8(b), so byte(i8) recovers it).
func rawKRowDot(dt Dtype, raw []byte, x []float32, scratch []float32) float32 {
	bs := 176
	if dt == Q6_K {
		bs = 210
	}
	var sum float32
	for off, xi := 0, 0; off < len(raw); off, xi = off+bs, xi+256 {
		b := raw[off : off+bs]
		if dt == Q5_K {
			dequantQ5K(scratch, b)
		} else {
			dequantQ6K(scratch, b)
		}
		for j := 0; j < 256; j++ {
			sum += scratch[j] * x[xi+j]
		}
	}
	return sum
}
func dequantQ5K(dst []float32, b []byte) {
	d := math.Float32frombits(kquantbits.F16BitsToF32Bits(binary.LittleEndian.Uint16(b)))
	dm := math.Float32frombits(kquantbits.F16BitsToF32Bits(binary.LittleEndian.Uint16(b[2:])))
	sc, qh, ql := b[4:16], b[16:48], b[48:]
	qi, is := 0, 0
	u1, u2 := byte(1), byte(2)
	for j := 0; j < 256; j += 64 {
		s1, m1 := kquantbits.ScaleMinK4(is, sc)
		s2, m2 := kquantbits.ScaleMinK4(is+1, sc)
		for l := 0; l < 32; l++ {
			hi := byte(0)
			if qh[l]&u1 != 0 {
				hi = 16
			}
			dst[j+l] = d*float32(s1)*float32((ql[qi+l]&15)+hi) - dm*float32(m1)
		}
		for l := 0; l < 32; l++ {
			hi := byte(0)
			if qh[l]&u2 != 0 {
				hi = 16
			}
			dst[j+32+l] = d*float32(s2)*float32((ql[qi+l]>>4)+hi) - dm*float32(m2)
		}
		qi += 32
		is += 2
		u1 <<= 2
		u2 <<= 2
	}
}
func dequantQ6K(dst []float32, b []byte) {
	ql, qh, sc := b[:128], b[128:192], b[192:208]
	d := math.Float32frombits(kquantbits.F16BitsToF32Bits(binary.LittleEndian.Uint16(b[208:])))
	qo, ho, so := 0, 0, 0
	for n := 0; n < 256; n += 128 {
		for l := 0; l < 32; l++ {
			is := l / 16
			q1 := int8((ql[qo+l]&15)|(((qh[ho+l]>>0)&3)<<4)) - 32
			q2 := int8((ql[qo+l+32]&15)|(((qh[ho+l]>>2)&3)<<4)) - 32
			q3 := int8((ql[qo+l]>>4)|(((qh[ho+l]>>4)&3)<<4)) - 32
			q4 := int8((ql[qo+l+32]>>4)|(((qh[ho+l]>>6)&3)<<4)) - 32
			dst[n+l] = d * float32(int8(sc[so+is])) * float32(q1)
			dst[n+l+32] = d * float32(int8(sc[so+is+2])) * float32(q2)
			dst[n+l+64] = d * float32(int8(sc[so+is+4])) * float32(q3)
			dst[n+l+96] = d * float32(int8(sc[so+is+6])) * float32(q4)
		}
		qo += 64
		ho += 32
		so += 8
	}
}

func i8AsBytes(s []int8) []byte {
	b := make([]byte, len(s))
	for i, v := range s {
		b[i] = byte(v)
	}
	return b
}

// q4kDequantBlock writes the 256 weights of one 144-byte super-block into dst (len >= 256) — the
// exact arithmetic of model.q4kDequantSuperBlock.
func q4kDequantBlock(dst []float32, blk []byte) {
	d := math.Float32frombits(kquantbits.F16BitsToF32Bits(binary.LittleEndian.Uint16(blk[0:])))
	dmin := math.Float32frombits(kquantbits.F16BitsToF32Bits(binary.LittleEndian.Uint16(blk[2:])))
	scales := blk[4 : 4+12]
	q := blk[4+12 : q4kSuperBlock]
	qi, is := 0, 0
	for j := 0; j < q4kSuper; j += 64 {
		sc, m := kquantbits.ScaleMinK4(is, scales)
		d1, m1 := d*float32(sc), dmin*float32(m)
		sc, m = kquantbits.ScaleMinK4(is+1, scales)
		d2, m2 := d*float32(sc), dmin*float32(m)
		for l := 0; l < 32; l++ {
			dst[j+l] = d1*float32(q[qi+l]&0x0f) - m1
		}
		for l := 0; l < 32; l++ {
			dst[j+32+l] = d2*float32(q[qi+l]>>4) - m2
		}
		qi += 32
		is += 2
	}
}

// q4kRowDot computes one output element y[o] = dot(weight row, x) over a Q4_K weight row — the exact
// per-row reduction of model.q4kMatRowsRange: dequant each super-block, four-accumulator dot, summed
// across super-blocks in row order. raw is the [in/256 * 144]-byte weight row.
func q4kRowDot(raw []byte, x []float32, buf []float32) float32 {
	nblk := len(x) / q4kSuper
	var acc float32
	for b := 0; b < nblk; b++ {
		q4kDequantBlock(buf, raw[b*q4kSuperBlock:(b+1)*q4kSuperBlock])
		xs := x[b*q4kSuper:]
		var s0, s1, s2, s3 float32
		for i := 0; i < q4kSuper; i += 4 {
			s0 += buf[i] * xs[i]
			s1 += buf[i+1] * xs[i+1]
			s2 += buf[i+2] * xs[i+2]
			s3 += buf[i+3] * xs[i+3]
		}
		acc += (s0 + s1) + (s2 + s3)
	}
	return acc
}

func f16bitsToF32(h uint16) uint32 { return kquantbits.F16BitsToF32Bits(h) }
