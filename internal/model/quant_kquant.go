package model

// quant_kquant.go — resident raw expert-quant weight path, the load-time lever for GLM-5.2's
// MIXED-quant MoE experts. unsloth's UD checkpoints can store the routed-expert bulk as Q4_K,
// Q5_K/Q6_K, IQ3_XXS/IQ4_XS, or Q8_0. Q4_K has its own resident store (quant_q4k.go); this file
// covers the remaining host-only expert formats so they load as raw bytes instead of taking the
// f32 dequant→Q8 round-trip.
//
// SCOPE: these resident k-quant tensors are EXPERT weights, which the GLM serve runs on the
// HOST CPU under --cpu-offload-experts (the experts dwarf VRAM). So only a CPU dequant-fused
// GEMV is needed — no GPU kernel — and the dispatch is residentMatRows (the host expert seam).
// Dense Q6_K weights keep their existing dequant→Q8 path (they are small and some route to the
// device HAL, which has no Q5_K/Q6_K kernel).
//
// CORRECTNESS: each per-block dequant is ggufload's matching dequant factored to one block, so
// the resident dequant is arithmetically identical to the f32 reference the loader would otherwise
// have produced; kQuantMatRows uses the SAME fixed-order 4-accumulator dot as q4kMatRowsRange.
// Pinned by TestKQuantMatRowsMatchesDequantRef (bit-exact vs the dequant-then-dot reference).

import (
	"encoding/binary"
	"math"
)

// kQuantKind selects the super-block format of a resident k-quant tensor.
type kQuantKind uint8

const (
	kindQ5K kQuantKind = iota
	kindQ6K
	kindIQ3XXS
	kindIQ4XS
	kindQ8_0
)

// Resident k-quant super-block byte sizes per 256 weights (== ggufload.blockQ{5,6}KBytes):
//
//	Q5_K     = d(f16,2) + min(f16,2) + scales(12) + qh(32) + ql(128) = 176
//	Q6_K     = ql(128) + qh(64) + scales(16) + d(f16,2)             = 210
//	IQ3_XXS  = d(f16,2) + qs(64) + scales/signs(32)                 = 98
//	IQ4_XS   = d(f16,2) + scales_h(2) + scales_l(4) + qs(128)       = 136
//	Q8_0     = d(f16,2) + qs(32)                                    = 34
const (
	q5kBlockBytes    = 2 + 2 + 12 + qkK/8 + qkK/2
	q6kBlockBytes    = qkK/2 + qkK/4 + qkK/16 + 2
	iq3xxsBlockBytes = 2 + 3*qkK/8
	iq4xsBlockBytes  = 2 + 2 + qkK/64 + qkK/2
	q8_0BlockWeights = 32
	q8_0BlockBytes   = 2 + q8_0BlockWeights
)

func (k kQuantKind) blockBytes() int {
	switch k {
	case kindQ6K:
		return q6kBlockBytes
	case kindIQ3XXS:
		return iq3xxsBlockBytes
	case kindIQ4XS:
		return iq4xsBlockBytes
	case kindQ8_0:
		return q8_0BlockBytes
	default:
		return q5kBlockBytes
	}
}

func (k kQuantKind) blockWeights() int {
	if k == kindQ8_0 {
		return q8_0BlockWeights
	}
	return qkK
}

func (k kQuantKind) String() string {
	switch k {
	case kindQ6K:
		return "Q6_K"
	case kindIQ3XXS:
		return "IQ3_XXS"
	case kindIQ4XS:
		return "IQ4_XS"
	case kindQ8_0:
		return "Q8_0"
	default:
		return "Q5_K"
	}
}

// kQuantTensor is a resident raw expert-quant weight matrix [out, in]. raw holds the GGUF
// block bytes verbatim (no f32), row-major: row o occupies raw[o*rowBytes:], where rowBytes =
// nblk*blockBytes. This is the exact byte stream the GGUF stores, so the loader copies a tensor's
// payload in with no transform.
type kQuantTensor struct {
	out, in, nblk int
	kind          kQuantKind
	raw           []byte
}

func (qt *kQuantTensor) rowBytes() int { return qt.nblk * qt.kind.blockBytes() }

// q5kDequantSuperBlock writes the 256 weights of one 176-byte Q5_K super-block into dst
// (len >= 256). Byte-for-byte ggufload.dequantQ5K factored to one super-block.
func q5kDequantSuperBlock(dst []float32, blk []byte) {
	d := math.Float32frombits(F16BitsToF32Bits(binary.LittleEndian.Uint16(blk[0:])))
	min := math.Float32frombits(F16BitsToF32Bits(binary.LittleEndian.Uint16(blk[2:])))
	scales := blk[4 : 4+12]
	qh := blk[4+12 : 4+12+qkK/8]
	ql := blk[4+12+qkK/8 : q5kBlockBytes]
	qi := 0
	is := 0
	u1, u2 := byte(1), byte(2)
	for j := 0; j < qkK; j += 64 {
		sc, m := GetScaleMinK4(is, scales)
		d1, m1 := d*float32(sc), min*float32(m)
		sc, m = GetScaleMinK4(is+1, scales)
		d2, m2 := d*float32(sc), min*float32(m)
		for l := 0; l < 32; l++ {
			hi := byte(0)
			if qh[l]&u1 != 0 {
				hi = 16
			}
			dst[j+l] = d1*float32((ql[qi+l]&0x0f)+hi) - m1
		}
		for l := 0; l < 32; l++ {
			hi := byte(0)
			if qh[l]&u2 != 0 {
				hi = 16
			}
			dst[j+32+l] = d2*float32((ql[qi+l]>>4)+hi) - m2
		}
		qi += 32
		is += 2
		u1 <<= 2
		u2 <<= 2
	}
}

// q6kDequantSuperBlock writes the 256 weights of one 210-byte Q6_K super-block into dst
// (len >= 256). Byte-for-byte ggufload.dequantQ6K factored to one super-block.
func q6kDequantSuperBlock(dst []float32, blk []byte) {
	ql := blk[0 : qkK/2]
	qh := blk[qkK/2 : qkK/2+qkK/4]
	scales := blk[qkK/2+qkK/4 : qkK/2+qkK/4+qkK/16]
	d := math.Float32frombits(F16BitsToF32Bits(binary.LittleEndian.Uint16(blk[q6kBlockBytes-2:])))
	qlOff, qhOff, scOff := 0, 0, 0
	for n := 0; n < qkK; n += 128 {
		for l := 0; l < 32; l++ {
			is := l / 16
			q1 := int8((ql[qlOff+l+0]&0x0f)|(((qh[qhOff+l]>>0)&3)<<4)) - 32
			q2 := int8((ql[qlOff+l+32]&0x0f)|(((qh[qhOff+l]>>2)&3)<<4)) - 32
			q3 := int8((ql[qlOff+l+0]>>4)|(((qh[qhOff+l]>>4)&3)<<4)) - 32
			q4 := int8((ql[qlOff+l+32]>>4)|(((qh[qhOff+l]>>6)&3)<<4)) - 32
			dst[n+l+0] = d * float32(int8(scales[scOff+is+0])) * float32(q1)
			dst[n+l+32] = d * float32(int8(scales[scOff+is+2])) * float32(q2)
			dst[n+l+64] = d * float32(int8(scales[scOff+is+4])) * float32(q3)
			dst[n+l+96] = d * float32(int8(scales[scOff+is+6])) * float32(q4)
		}
		qlOff += 64
		qhOff += 32
		scOff += 8
	}
}

func kQuantDequantSuperBlock(dst []float32, blk []byte, kind kQuantKind) {
	switch kind {
	case kindQ6K:
		q6kDequantSuperBlock(dst, blk)
	case kindIQ3XXS:
		iq3xxsDequantSuperBlock(dst, blk)
	case kindIQ4XS:
		iq4xsDequantSuperBlock(dst, blk)
	case kindQ8_0:
		q8_0DequantBlock(dst, blk)
	default:
		q5kDequantSuperBlock(dst, blk)
	}
}

// kQuantMatRows is the resident Q5_K/Q6_K decode GEMV: y[o] = dot(weight row o, x). Like
// q4kMatRows it dequantizes each super-block into a tiny L1-resident scratch and dots it
// against the matching 256-wide slice of x, with row-parallelism over the output rows.
func kQuantMatRows(qt *kQuantTensor, x []float32) []float32 {
	y := make([]float32, qt.out)
	kQuantMatRowsInto(qt, x, y)
	return y
}

func kQuantMatRowsInto(qt *kQuantTensor, x, y []float32) {
	y = y[:qt.out]
	if kQuantSDOTEnabled(qt.kind) {
		// int8 k-quant decode path: quantize the activation ONCE and reuse it across every output
		// row, so the per-row work is the compact int8 reduction instead of a 256-wide f32
		// dequant+dot — the lever for GLM-5.2's mixed-quant offloaded experts. Q5_K
		// (quant_kquant_int8.go) and Q6_K (quant_kquant_int8_q6k.go) each have a reducer; the f32
		// kQuantMatRowsRange below is untouched + byte-identical (TestKQuantMatRowsMatchesF32).
		qv := quantizeVecQ8(x)
		ranger := q5kMatRowsRangeInt8
		if qt.kind == kindQ6K {
			ranger = q6kMatRowsRangeInt8
		}
		if numWorkers <= 1 || qt.out*qt.in < parThreshold {
			ranger(qt, qv, y, 0, qt.out)
			return
		}
		parFor(qt.out, numWorkers, func(lo, hi int) { ranger(qt, qv, y, lo, hi) })
		return
	}
	if numWorkers <= 1 || qt.out*qt.in < parThreshold {
		kQuantMatRowsRange(qt, x, y, 0, qt.out)
		return
	}
	parFor(qt.out, numWorkers, func(lo, hi int) { kQuantMatRowsRange(qt, x, y, lo, hi) })
}

// kQuantMatRowsRange computes y[lo:hi] with the SAME fixed-order four-accumulator dot and
// super-block order as q4kMatRowsRange, so the resident k-quant GEMV is deterministic and
// arithmetically identical to a dequant-then-dot over the same f32 weights.
func kQuantMatRowsRange(qt *kQuantTensor, x, y []float32, lo, hi int) {
	blockWeights := qt.kind.blockWeights()
	buf := make([]float32, blockWeights) // reused per block; L1/L2-resident
	rowBytes := qt.rowBytes()
	bb := qt.kind.blockBytes()
	for o := lo; o < hi; o++ {
		row := qt.raw[o*rowBytes:]
		var acc float32
		for b := 0; b < qt.nblk; b++ {
			kQuantDequantSuperBlock(buf, row[b*bb:(b+1)*bb], qt.kind)
			xs := x[b*blockWeights:]
			var s0, s1, s2, s3 float32
			for i := 0; i < blockWeights; i += 4 {
				s0 += buf[i] * xs[i]
				s1 += buf[i+1] * xs[i+1]
				s2 += buf[i+2] * xs[i+2]
				s3 += buf[i+3] * xs[i+3]
			}
			acc += (s0 + s1) + (s2 + s3)
		}
		y[o] = acc
	}
}

// quantizeKQuantFromRaw wraps a raw GGUF expert-quant payload as a resident kQuantTensor with NO
// transform — the bytes ARE the GGUF bytes. The raw-byte twin of quantizeQ4KFromRaw for the
// mixed-quant expert bulk.
func quantizeKQuantFromRaw(raw []byte, out, in int, kind kQuantKind) *kQuantTensor {
	blockWeights := kind.blockWeights()
	if in%blockWeights != 0 {
		panic("model: resident expert quant reduction dim not a multiple of block size")
	}
	nblk := in / blockWeights
	want := out * nblk * kind.blockBytes()
	if len(raw) != want {
		panic("model: resident expert quant payload size mismatch")
	}
	return &kQuantTensor{out: out, in: in, nblk: nblk, kind: kind, raw: raw}
}

// hasKQuant reports whether a resident raw expert-quant copy is available for a name.
func (m *Model) hasKQuant(name string) bool { return m.kqw != nil && m.kqw[name] != nil }

// HasKQuant reports whether a resident raw Q5_K/Q6_K/IQ*/Q8_0 expert-quant copy is available
// for a name. It is the exported diagnostic twin of hasKQuant, used by loaders/tests that need to
// prove which expert-band tensors were admitted without reaching into the unexported store.
func (m *Model) HasKQuant(name string) bool { return m.hasKQuant(name) }

// KQuantCount returns how many tensors hold a resident raw expert-quant copy (loader diagnostic).
func (m *Model) KQuantCount() int { return len(m.kqw) }

// KQuantRaw returns a resident raw expert-quant tensor's super-block bytes and true, or nil/false
// if the name is not resident. It is the exported read-back for load witnesses and sharded-load
// tests that must prove not just WHICH band was admitted (HasKQuant) but that the admitted band
// carries the RIGHT expert's bytes — the fidelity a zeroed-payload residency check cannot see. The
// slice aliases the resident store; callers must not mutate it.
func (m *Model) KQuantRaw(name string) ([]byte, bool) {
	if m.kqw == nil {
		return nil, false
	}
	qt := m.kqw[name]
	if qt == nil {
		return nil, false
	}
	return qt.raw, true
}

// ResidentKQuantEligible reports whether a canonical tensor name should be held as resident raw
// expert quant. It is the same identity-normalization gate as ResidentQ4KEligible (a matmul weight
// that normalizeCanonicalTensorData does NOT transform), so a transformed tensor's raw bytes are
// never stored. The loader additionally restricts this to EXPERT weights (the CPU-offloaded bulk),
// since dense non-Q4_K raw formats may route to the device HAL which has no kernel for them.
func ResidentKQuantEligible(cfg Config, canon string) bool {
	return ResidentQ4KEligible(cfg, canon)
}

// AddResident* store a raw expert quant payload as a resident kQuantTensor under the canonical
// name, skipping the f32/Q8 round-trip. shape is the model [out, in] convention. Idempotent for
// non-eligible names (returns nil without storing) so the loader can call them unconditionally on
// a matching-type expert tensor.
func (b *QuantBuilder) AddResidentQ6K(canon string, shape []int, raw []byte) error {
	return b.addResidentKQuant(canon, shape, raw, kindQ6K)
}

func (b *QuantBuilder) AddResidentQ5K(canon string, shape []int, raw []byte) error {
	return b.addResidentKQuant(canon, shape, raw, kindQ5K)
}

func (b *QuantBuilder) AddResidentIQ3XXS(canon string, shape []int, raw []byte) error {
	return b.addResidentKQuant(canon, shape, raw, kindIQ3XXS)
}

func (b *QuantBuilder) AddResidentIQ4XS(canon string, shape []int, raw []byte) error {
	return b.addResidentKQuant(canon, shape, raw, kindIQ4XS)
}

func (b *QuantBuilder) AddResidentQ8_0(canon string, shape []int, raw []byte) error {
	return b.addResidentKQuant(canon, shape, raw, kindQ8_0)
}

func (b *QuantBuilder) addResidentKQuant(canon string, shape []int, raw []byte, kind kQuantKind) error {
	name, ok, err := b.residentQuantTarget(canon, shape)
	if !ok || err != nil {
		return err
	}
	if b.m.kqw == nil {
		b.m.kqw = map[string]*kQuantTensor{}
	}
	b.m.kqw[name] = quantizeKQuantFromRaw(raw, shape[0], shape[1], kind)
	return nil
}
