package model

import (
	"math"
	"runtime"
)

// quant_q4.go — resident int4 (Q4_0-style) weight path: the decode-bandwidth lever for
// the in-kernel Qwen3.6 engine. See QWEN36-NATIVE-PERF-PLAN-2026-06-19.md §diagnosis:
// the GGUF→Q8 path streams ~30 GB/token (Q8_0 ≈ 1.125 B/param), and at ~100 GB/s that
// is a hard ~3 tok/s decode ceiling on the M3 Pro — Q8 physically cannot reach the 7.29
// tok/s llama.cpp Metal bar. The GGUF on disk is q4_k_m (~16 GB/token); a resident int4
// path streams roughly half the bytes of Q8 and so raises the decode ceiling toward the
// bar. This file lands that resident path behind an opt-in flag; the proven f32/Q8 paths
// are byte-for-byte untouched.
//
// Format. A q4Tensor is [out, in] row-major, in = nblk*qBlk4 (qBlk4 = 32). Each 32-wide
// block is one f32 scale `d` followed by 32 signed 4-bit codes packed two-per-byte (low
// nibble first), exactly llama.cpp's Q4_0 code layout with an f32 (not f16) scale. A code
// c (0..15) dequantizes to d*(c-8), so codes cover [-8*d, 7*d]. Per resident weight:
// (4 + 16)/32 = 0.625 bytes — vs Q8_0's ~1.125, the ~1.8× decode-bandwidth reduction.
//
// Correctness discipline. Quantization is lossy; this path carries its own honest gate
// (greedy-continuation agreement with the f32 reference + the round-trip unit test), never
// the f32 bit-exact rungs. The dequant is intentionally the inverse of quantizeQ4Block so
// the round trip is exact on the codes and bounded only by 4-bit rounding error.

const qBlk4 = 32

// q4Tensor is a resident int4 weight matrix [out, in], in == nblk*qBlk4. Scales and codes
// are separate per-row slices so a row's codes prefetch alongside its scales.
type q4Tensor struct {
	out, in, nblk int
	d             []float32 // out*nblk per-block f32 scales
	q             []byte    // out*nblk*16 packed 4-bit codes (2 per byte, low nibble first)
}

// quantizeQ4Block stores the 32 weights of src as one (d, 16 nibble bytes) block at dst,
// returning the per-block scale. d = amax/7 (so the largest magnitude maps to code 7 or 0);
// a zero block stays zero. Codes are round(w/d)+8 clamped to [0,15] (signed range [-8,7]).
func quantizeQ4Block(dst []byte, src []float32) float32 {
	var amax float32
	for _, v := range src {
		a := v
		if a < 0 {
			a = -a
		}
		if a > amax {
			amax = a
		}
	}
	if amax == 0 {
		for i := 0; i < qBlk4/2; i++ {
			dst[i] = 0
		}
		return 0
	}
	d := amax / 7
	inv := 1.0 / float64(d)
	for i := 0; i < qBlk4/2; i++ {
		// Round-to-nearest (int() truncates toward zero and would bias every code by up to
		// half a quantum). This runs once at load, so a float64 round is fine.
		c0 := int(math.Round(float64(src[2*i])*inv)) + 8
		c1 := int(math.Round(float64(src[2*i+1])*inv)) + 8
		if c0 < 0 {
			c0 = 0
		} else if c0 > 15 {
			c0 = 15
		}
		if c1 < 0 {
			c1 = 0
		} else if c1 > 15 {
			c1 = 15
		}
		dst[i] = byte(c0) | byte(c1<<4)
	}
	return d
}

// dequantQ4Block writes the 32 weights of one (d, 16 nibble bytes) block into dst.
func dequantQ4Block(dst []float32, d float32, q []byte) {
	for i := 0; i < qBlk4/2; i++ {
		c0 := int(q[i] & 0x0f)
		c1 := int(q[i] >> 4)
		dst[2*i] = d * float32(c0-8)
		dst[2*i+1] = d * float32(c1-8)
	}
}

// quantizeQ4 builds a resident int4 copy of an [out,in] f32 matrix (in must be a multiple
// of qBlk4). Built once at load; not on the hot path.
func quantizeQ4(w []float32, out, in int) *q4Tensor {
	if in%qBlk4 != 0 {
		// Without this guard a non-multiple-of-32 reduction dim would silently drop the
		// tail; fail loudly like the Q8 lane does. Every Qwen3.6 reduction dim (hidden 5120,
		// intermediate 17408, the linear/full-attn projection widths) is a multiple of 32.
		panic("model: int4 reduction dim not a multiple of 32")
	}
	nblk := in / qBlk4
	qt := &q4Tensor{out: out, in: in, nblk: nblk, d: make([]float32, out*nblk), q: make([]byte, out*nblk*(qBlk4/2))}
	parFor(out, numWorkers, func(lo, hi int) {
		blk := make([]float32, qBlk4)
		for o := lo; o < hi; o++ {
			for b := 0; b < nblk; b++ {
				off := o*in + b*qBlk4
				qt.d[o*nblk+b] = quantizeQ4Block(qt.q[o*nblk*(qBlk4/2)+b*(qBlk4/2):], w[off:off+qBlk4])
			}
			_ = blk
		}
	})
	return qt
}

// q4MatRows is the int4 decode GEMV: y[o] = dot(row o, x). Row-parallel exactly like
// parMatRows/qMatRows — decode is memory-bound, so spreading rows across cores taps
// aggregate bandwidth; with int4 codes each core streams ~1.8× fewer bytes than the Q8
// GEMV. The dequant of each block lands in a tiny L1-resident scratch before the fdot, so
// the bandwidth-dominant stream is the 16 code bytes/block (the f32 scale is 1/16 of that).
func q4MatRows(qt *q4Tensor, x []float32) []float32 {
	y := make([]float32, qt.out)
	q4MatRowsInto(qt, x, y)
	return y
}

func q4MatRowsInto(qt *q4Tensor, x, y []float32) {
	y = y[:qt.out]
	if numWorkers <= 1 || qt.out*qt.in < parThreshold {
		q4MatRowsRange(qt, x, y, 0, qt.out)
		return
	}
	parFor(qt.out, numWorkers, func(lo, hi int) { q4MatRowsRange(qt, x, y, lo, hi) })
}

func q4MatRowsRange(qt *q4Tensor, x, y []float32, lo, hi int) {
	blk := make([]float32, qBlk4)
	nblk := qt.nblk
	half := qBlk4 / 2
	for o := lo; o < hi; o++ {
		qrow := qt.q[o*nblk*half:]
		drow := qt.d[o*nblk:]
		var s0, s1, s2, s3 float32
		for b := 0; b < nblk; b++ {
			dequantQ4Block(blk, drow[b], qrow[b*half:])
			xs := x[b*qBlk4:]
			// fdot over the 32-wide block, 8 accumulators (matches the parallel.go fdot
			// ILP pattern; combined in fixed order so the result is deterministic).
			s0 += blk[0]*xs[0] + blk[1]*xs[1] + blk[2]*xs[2] + blk[3]*xs[3]
			s1 += blk[4]*xs[4] + blk[5]*xs[5] + blk[6]*xs[6] + blk[7]*xs[7]
			s2 += blk[8]*xs[8] + blk[9]*xs[9] + blk[10]*xs[10] + blk[11]*xs[11]
			s3 += blk[12]*xs[12] + blk[13]*xs[13] + blk[14]*xs[14] + blk[15]*xs[15]
			s0 += blk[16]*xs[16] + blk[17]*xs[17] + blk[18]*xs[18] + blk[19]*xs[19]
			s1 += blk[20]*xs[20] + blk[21]*xs[21] + blk[22]*xs[22] + blk[23]*xs[23]
			s2 += blk[24]*xs[24] + blk[25]*xs[25] + blk[26]*xs[26] + blk[27]*xs[27]
			s3 += blk[28]*xs[28] + blk[29]*xs[29] + blk[30]*xs[30] + blk[31]*xs[31]
		}
		y[o] = ((s0 + s1) + (s2 + s3))
	}
}

// quantizeQ4FromQ8 builds the int4 tensor directly from a Q8_0 tensor, block-by-block:
// each 32-wide Q8_0 block dequantizes into a tiny 32-f32 buffer and re-quantizes to one
// int4 block. q8 block size (qBlk = 32) equals qBlk4, so the block grids are identical.
// This never materializes a full f32 row (the lm_head row would be ~5 GB), keeping the
// q4-build peak at "one q8 tensor + its q4 twin + 32 floats", not the whole f32 matrix.
func quantizeQ4FromQ8(q8 *q8Tensor) *q4Tensor {
	out, in, nblk := q8.out, q8.in, q8.nblk
	qt := &q4Tensor{out: out, in: in, nblk: nblk, d: make([]float32, out*nblk), q: make([]byte, out*nblk*(qBlk4/2))}
	half := qBlk4 / 2
	parFor(out, numWorkers, func(lo, hi int) {
		blk := make([]float32, qBlk4)
		for o := lo; o < hi; o++ {
			for b := 0; b < nblk; b++ {
				db := q8.d[o*nblk+b]
				qb := q8.q[o*in+b*qBlk : o*in+b*qBlk+qBlk]
				for i := 0; i < qBlk4; i++ {
					blk[i] = float32(qb[i]) * db
				}
				qt.d[o*nblk+b] = quantizeQ4Block(qt.q[o*nblk*half+b*half:], blk)
			}
		}
	})
	return qt
}

// QuantizeQ4 builds the resident int4 copy of every weight the quantized forward path
// uses (the same set Quantize covers: per-layer projections + the LM head). Idempotent.
// It quantizes from the f32 manifest when the tensor is resident, otherwise from the Q8_0
// resident copy (the lean GGUF path drops f32 for the big weights). When building from a
// Q8_0 tensor it frees that Q8_0 entry as soon as its int4 copy exists and forces GC per
// layer, so a 36 GB box that held the Q8_0 27B resident (~26 GB) can build the int4 copy
// (~15 GB) without ever holding both fully — the int4 path is a lean, q4-only resident
// mode (prefill runs the per-token int4 GEMV via Session.Q4, decode the row-parallel GEMV).
func (m *Model) QuantizeQ4() {
	if m.q4w != nil {
		return
	}
	qm := make(map[string]*q4Tensor)
	add := func(name string) *q4Tensor {
		if meta, ok := m.manifest[name]; ok && len(meta.Shape) == 2 {
			qt := quantizeQ4(m.tensor(name), meta.Shape[0], meta.Shape[1])
			qm[name] = qt
			return qt
		}
		if q8, ok := m.q8w[name]; ok {
			res := quantizeQ4FromQ8(q8)
			qm[name] = res
			delete(m.q8w, name) // free the Q8_0 copy now that its int4 twin exists (lean q4-only mode)
			return res
		}
		panic("model: QuantizeQ4 missing tensor " + name)
	}
	for l := 0; l < m.Cfg.NumLayers; l++ {
		p := func(s string) string { return layerName(l, s) }
		if m.Cfg.isLinearAttnLayer(l) {
			add(p("linear_attn.in_proj_qkv.weight"))
			add(p("linear_attn.in_proj_z.weight"))
			add(p("linear_attn.in_proj_a.weight"))
			add(p("linear_attn.in_proj_b.weight"))
			add(p("linear_attn.out_proj.weight"))
		} else {
			add(p("self_attn.q_proj.weight"))
			add(p("self_attn.k_proj.weight"))
			add(p("self_attn.v_proj.weight"))
			add(p("self_attn.o_proj.weight"))
		}
		if m.has(routerName(l)) {
			add(routerName(l))
			for e := 0; e < m.Cfg.NumExperts; e++ {
				add(expertName(l, e, "gate_proj.weight"))
				add(expertName(l, e, "up_proj.weight"))
				add(expertName(l, e, "down_proj.weight"))
			}
		} else if m.Cfg.DenseMLP {
			add(p("mlp.gate_proj.weight"))
			add(p("mlp.down_proj.weight"))
		} else {
			add(p("mlp.gate_proj.weight"))
			add(p("mlp.up_proj.weight"))
			add(p("mlp.down_proj.weight"))
		}
		// Reclaim the Q8_0 pages freed this layer so peak RSS stays near max(q8w, q4w)
		// (~26 GB → ~15 GB) instead of their sum, which would OOM a 36 GB box at the 27B
		// scale. Per-layer cadence is enough (each layer frees ~0.4 GB of Q8_0).
		runtime.GC()
	}
	// Resolve the head key ONCE, before add() frees the q8w entry headName() may key off
	// (a lean GGUF model has no f32 lm_head, so headName() reads m.q8w["lm_head.weight"];
	// add() then frees it, which would shift a later headName() to the tied-embedding key).
	headKey := m.headName()
	add(headKey)
	runtime.GC()
	m.q4head = qm[headKey] // pinned; headQ4 reads this pointer, never re-resolves the name
	m.q4w = qm
}

// q4 returns the prebuilt int4 tensor for a name (QuantizeQ4 must have run).
func (m *Model) q4(name string) *q4Tensor {
	qt, ok := m.q4w[name]
	if !ok {
		panic("model: int4 tensor not built: " + name + " (call Model.QuantizeQ4)")
	}
	return qt
}

// q4Kernel runs the single-position block in int4: prep is the identity (the GEMV
// dequantizes weight blocks on the fly and consumes the f32 activation directly, so the
// same prepared activation is reused across q/k/v and gate/up exactly as f32Kernel does);
// mul applies the named resident int4 weight. Theactivation quantization of the Q8 path is
// avoided entirely — int4 weights + f32 activation is the llama.cpp Q4 decode regime.
type q4Kernel struct{ m *Model }

func (k q4Kernel) prep(x []float32) any { return x }

func (k q4Kernel) mul(name string, x any, out, in int) []float32 {
	return q4MatRows(k.m.q4(name), x.([]float32))
}

// sessionQ4Kernel is the allocation-light twin reusing the session activation buffer — the
// int4 GEMV takes f32 activations directly so there is no per-call quant scratch to reuse;
// this form exists only to mirror sessionQ8Kernel's shape for the blockStep dispatch.
type sessionQ4Kernel struct{ s *Session }

func (k sessionQ4Kernel) prep(x []float32) any { return x }

func (k sessionQ4Kernel) mul(name string, x any, out, in int) []float32 {
	return q4MatRows(k.s.M.q4(name), x.([]float32))
}

// headQ4 applies the resident int4 LM head to a post-final-norm hidden vector.
func (s *Session) headQ4(xf []float32) []float32 {
	if s.qDecode == nil {
		s.qDecode = &qDecodeBuf{}
	}
	y := grow(s.qDecode.Logits, s.M.Cfg.VocabSize)
	s.qDecode.Logits = y
	t := s.phaseStart()
	qt := s.M.q4head
	if qt == nil {
		qt = s.M.q4(s.M.headName()) // fallback when q4w was built without freeing q8w
	}
	q4MatRowsInto(qt, xf, y)
	logitScaleInPlace(y, s.M.Cfg)
	s.phaseEnd("lm_head_q4", t)
	return y
}
