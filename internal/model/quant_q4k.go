package model

// quant_q4k.go — resident Q4_K (k-quant super-block) weight path: the load+decode+memory
// lever for QWEN36-NATIVE-PERF-PLAN-2026-06-19.md P1. See that plan's §diagnosis:
//
//   - The GGUF→Q8 path streams ~30 GB/token (Q8_0 ≈ 1.125 B/param) and at ~100 GB/s that
//     is a hard ~3 tok/s decode ceiling on the M3 Pro — Q8 physically cannot reach the
//     7.29 tok/s llama.cpp Metal bar.
//   - The Q4_0-from-q8w path (quant_q4.go) re-quants from q8w, so it STILL pays the
//     Q4→f32→Q8 load round-trip AND keeps q8w (~30 GB) resident next to q4w (~17 GB) —
//     measured: ~11 GB of swap on the 36 GB box. It also uses Q4_0, not the GGUF's Q4_K,
//     so it does not remove the Q8 quantization error that drifts the greedy continuation
//     by token 3.
//
// This path instead holds the RAW Q4_K blocks straight from the GGUF: no dequantF32, no
// re-quant, no q8w co-residency. Per the plan that (a) cuts load ~10× (the load profile is
// dominated by gguf_dequant Q4→f32 49% + quant_builder f32→Q8 25% + gguf_normalize 17%),
// (b) drops resident footprint to ~16 GB for 27B so it fits the 36 GB M3 with no swap, and
// (c) streams the exact q4_k_m bytes llama.cpp streams, so the greedy continuation matches
// the llama.cpp q4_k_m artifact (closing token-3 drift, #93) rather than a re-quant of it.
//
// Format (byte-for-byte ggufload.dequantQ4K, so the resident bytes ARE the GGUF bytes):
// one super-block per 256 weights = 144 B = d(f16,2) + min(f16,2) + scales(12) + q(128
// nibbles). 8 sub-blocks of 32 weights; the 8 (scale,min) pairs are packed 6-bit via
// GetScaleMinK4. Per-weight dequant: d*sc*(code) - min*m. Per resident weight:
// 144/256 = 0.5625 B — vs Q8_0's 1.125 and Q4_0's 0.625; fewer bytes than either.
//
// Correctness discipline. Quantization is lossy; this path carries its own gate
// (greedy-continuation agreement with the llama.cpp q4_k_m artifact + first-token id
// parity 248068), never the f32 bit-exact rungs. The inline dequant IS the f32 reference
// dequant factored per super-block, so q4kMatRows is correct-by-construction against the
// same f32 weights the loader would have produced — pinned by TestQ4KMatRowsMatchesF32.

import (
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"strings"
	"unsafe"
)

// qkK is the Q4_K super-block size: 256 weights share one (d, min, scales, q) block. This
// matches ggufload.qkK. Every reduction dim a Q4_K dot runs over must be a multiple of 256
// — for Qwen3.6-27B that holds (hidden 5120, intermediate 17408, vocab 49152, and the
// linear/full-attn projection widths). quantizeQ4KFromRaw panics on a non-multiple rather
// than silently dropping a tail.
const qkK = 256

// q4kBlockBytes is the resident byte cost of one 256-weight Q4_K super-block:
// 2 (d f16) + 2 (min f16) + 12 (scales) + 128 (256 nibbles) = 144.
const q4kBlockBytes = 2 + 2 + 12 + qkK/2

// q4kTensor is a resident Q4_K weight matrix [out, in], in == nblk*qkK. raw holds the GGUF
// super-block bytes verbatim (no f32), row-major: row o occupies raw[o*rowBytes:], where
// rowBytes = nblk*q4kBlockBytes, and super-block b within a row sits at +b*q4kBlockBytes.
// This is the exact byte stream the GGUF stores and llama.cpp reads, so the loader copies a
// tensor's payload in with no transform. The Metal path can wrap these bytes with a no-copy
// buffer when the resident slice is page-aligned; quantizeQ4KFromRaw preserves the payload while
// moving unaligned loader copies into aligned backing storage.
type q4kTensor struct {
	out, in, nblk int
	raw           []byte
}

// q4kRowBytes is the byte length of one resident row (nblk super-blocks).
func (qt *q4kTensor) q4kRowBytes() int { return qt.nblk * q4kBlockBytes }

// requireRawCPU is the #1067 legibility guardrail for the CPU Q4_K matmul entry points. Under
// MetalQ4K with FAK_Q4K_FREE_CPU=1 (single residency), metalQ4KWeight drops qt.raw after a
// successful GPU upload. Decode then takes the Metal GEMV, but the BATCHED PREFILL GEMM
// (prefill_q4k.go's proj → q4kGemm, and the q4kGemmDispatch CPU fallback) is NOT Metal-routed —
// so a multi-thousand-token prompt reaches the CPU q4kGemmRange* and reads the freed nil raw,
// which previously died with a cryptic "slice bounds out of range [N:0]" deep in a parFor worker.
// This turns that into a legible failure that names the misconfig and its two remedies: keep the
// CPU copy (unset FAK_Q4K_FREE_CPU) or route the prefill GEMM through Metal. out>0 distinguishes a
// real-but-freed weight from a degenerate empty tensor (which holds no rows to read either way).
func (qt *q4kTensor) requireRawCPU(op string) {
	if len(qt.raw) == 0 && qt.out > 0 {
		panic(fmt.Sprintf("model: Q4_K %s on a freed CPU weight (out=%d in=%d): the resident raw "+
			"Q4_K bytes were dropped after a Metal upload (FAK_Q4K_FREE_CPU=1) but a CPU Q4_K matmul "+
			"ran — prefill is not GPU-routed for this weight. Unset FAK_Q4K_FREE_CPU to keep the CPU "+
			"copy, or route the prefill GEMM through Metal (#1067).", op, qt.out, qt.in))
	}
}

// q4kDequantSuperBlock writes the 256 weights of one 144-byte Q4_K super-block into dst
// (len >= 256). It is ggufload.dequantQ4K factored to one super-block, so the resident
// dequant is arithmetically identical to the loader's f32 reference path.
func q4kDequantSuperBlock(dst []float32, blk []byte) {
	d := math.Float32frombits(F16BitsToF32Bits(binary.LittleEndian.Uint16(blk[0:])))
	min := math.Float32frombits(F16BitsToF32Bits(binary.LittleEndian.Uint16(blk[2:])))
	scales := blk[4 : 4+12]
	q := blk[4+12 : q4kBlockBytes]
	qi := 0
	is := 0
	for j := 0; j < qkK; j += 64 {
		sc, m := GetScaleMinK4(is, scales)
		d1, m1 := d*float32(sc), min*float32(m)
		sc, m = GetScaleMinK4(is+1, scales)
		d2, m2 := d*float32(sc), min*float32(m)
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

// q4kMatRows is the resident-Q4_K decode GEMV: y[o] = dot(weight row o, x). Row-parallel
// exactly like parMatRows/qMatRows/q4MatRows — decode is memory-bound, so spreading rows
// across cores taps aggregate bandwidth; with raw Q4_K each core streams 0.5625 B/weight
// (fewer than Q8_0's 1.125 or Q4_0's 0.625). Each super-block's 256 weights dequant into a
// tiny L1-resident scratch before the dot, so the bandwidth-dominant stream is the 128 code
// bytes + 16 scale bytes per 256 weights.
func q4kMatRows(qt *q4kTensor, x []float32) []float32 {
	y := make([]float32, qt.out)
	q4kMatRowsInto(qt, x, y)
	return y
}

func q4kMatRowsInto(qt *q4kTensor, x, y []float32) {
	qt.requireRawCPU("decode GEMV")
	y = y[:qt.out]
	if q4kSDOTEnabled() {
		// int8 SDOT decode path (plan P2): the activation is quantized ONCE here (quantizeVecQ8)
		// and reused across every output row — this is a GEMV (one x, many weight rows) — so the
		// per-row work is the compact int8 reduction, not a 256-wide f32 dequant+dot. Q4_K's
		// 32-wide sub-blocks align 1:1 with q8Vec's blocks, so no padding/re-quant. The f32 path
		// below is untouched and stays byte-identical (TestQ4KMatRowsMatchesF32).
		qv := quantizeVecQ8(x)
		if numWorkers <= 1 || qt.out*qt.in < parThreshold {
			q4kMatRowsRangeInt8(qt, qv, y, 0, qt.out)
			return
		}
		parFor(qt.out, numWorkers, func(lo, hi int) { q4kMatRowsRangeInt8(qt, qv, y, lo, hi) })
		return
	}
	if numWorkers <= 1 || qt.out*qt.in < parThreshold {
		q4kMatRowsRange(qt, x, y, 0, qt.out)
		return
	}
	parFor(qt.out, numWorkers, func(lo, hi int) { q4kMatRowsRange(qt, x, y, lo, hi) })
}

// q4kMatRowsRange computes y[lo:hi] by dequanting each super-block inline and dotting it
// against the matching 256-wide slice of x. Four independent float32 accumulators per
// super-block, combined in fixed order, then summed across super-blocks in row order — so
// the result is deterministic and a future NEON SDOT kernel (P2) can be held to it.
func q4kMatRowsRange(qt *q4kTensor, x, y []float32, lo, hi int) {
	buf := make([]float32, qkK) // 256 f32, reused per super-block; L1/L2-resident
	rowBytes := qt.q4kRowBytes()
	for o := lo; o < hi; o++ {
		row := qt.raw[o*rowBytes:]
		var acc float32
		for b := 0; b < qt.nblk; b++ {
			q4kDequantSuperBlock(buf, row[b*q4kBlockBytes:(b+1)*q4kBlockBytes])
			xs := x[b*qkK:]
			var s0, s1, s2, s3 float32
			for i := 0; i < qkK; i += 4 {
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

// q4kGemm is the resident-Q4_K PREFILL GEMM: Y[t*out+o] = dot(weight row o, activation
// row t) for all t in [0,P), o in [0,out). It is the batched twin of q4kMatRows — the
// structural prefill win (each weight super-block is dequantized ONCE and reused across
// all P activation rows, instead of re-streaming+re-dequanting the whole weight matrix P
// times as the per-token GEMV does). Prefill is compute-bound, so amortizing the dequant
// + weight-bandwidth across the P free axis is what closes the gap to llama.cpp-Metal
// prefill on the q4_k_m artifact (QWEN36-NATIVE-PERF-PLAN P3).
//
// Correctness contract. For every (o,t), Y[t*out+o] is BIT-IDENTICAL to
// q4kMatRows(qt, X[t*in:(t+1)*in])[o]: the per-super-block dequant is the same
// q4kDequantSuperBlock, the four-accumulator dot is the same fixed-order reduction, and
// the across-super-block accumulation runs in the same row order (b=0..nblk-1). The ONLY
// change is that each super-block's 256 f32 dequant scratch is reused across P token rows
// before eviction — pure compute amortization, no arithmetic difference. Pinned by
// TestQ4KGemmMatchesMatRows. Output is row-major [P, out] (matching matMulBatch/qGemm8).
func q4kGemm(qt *q4kTensor, X []float32, P int) []float32 {
	Y := make([]float32, P*qt.out)
	q4kGemmInto(qt, X, P, Y)
	return Y
}

// q4kGemmInto is q4kGemm writing into a caller-provided Y (len >= P*out), the buffer-reuse
// form matching qGemm8Into / matMulBatch's caller-alloc convention.
func q4kGemmInto(qt *q4kTensor, X []float32, P int, Y []float32) {
	qt.requireRawCPU("prefill GEMM")
	out := qt.out
	Y = Y[:P*out]
	if q4kSDOTEnabled() {
		if q4kExtractOnceGemmEnabled() {
			// Extract-once int8 Q4_K GEMM (#60): lower the Q4_K nibbles into a temporary
			// Q8-shaped weight tensor once per GEMM, run the register-blocked qGemm8 kernel for
			// the d*scale*nibble term, then subtract the affine min term from precomputed
			// activation block sums. The previous int8 GEMM called q4kReduceRow once per
			// (output row, token), re-reading and unpacking the same Q4_K row P times; this path
			// pays that extract cost once and lets qGemm8 reuse the int8 row across token tiles.
			// Numerics follow qGemm8's deferred-reduction order rather than q4kCombineRow's
			// per-token GEMV order, so this path is covered by the Q4_K int8 tolerance/oracle
			// gates, not bit-identity with q4kMatRowsRangeInt8. The f32 path stays selectable
			// with FAK_QKERNEL=scalar.
			qp := quantizeBatchPanel(X, P, qt.in)
			q4kGemmExtractOnceInt8Into(qt, qp, Y)
			return
		}
		qvs := make([]q8Vec, P)
		for t := 0; t < P; t++ {
			qvs[t] = quantizeVecQ8(X[t*qt.in : (t+1)*qt.in])
		}
		if numWorkers <= 1 || out*qt.in*P < parThreshold {
			q4kGemmRangeInt8(qt, qvs, P, Y, 0, out)
			return
		}
		parFor(out, numWorkers, func(lo, hi int) { q4kGemmRangeInt8(qt, qvs, P, Y, lo, hi) })
		return
	}
	if numWorkers <= 1 || out*qt.in*P < parThreshold {
		q4kGemmRange(qt, X, P, Y, 0, out)
		return
	}
	parFor(out, numWorkers, func(lo, hi int) { q4kGemmRange(qt, X, P, Y, lo, hi) })
}

type q4kExtractedGemm struct {
	qt       *q8Tensor
	minScale []float32
}

// q4kGemmExtractOnceInt8Into is the active int8 Q4_K prefill GEMM. On arm64 the hot path
// streams small extracted row tiles through the Q8 GEMM micro-kernel to avoid materializing a
// full Q8-sized copy of the weight. The portable fallback below represents the positive nibble
// term as a temporary Q8-shaped tensor:
//
//	first[t,o] = sum_s (d_o,s*scale_o,s) * dx_t,s * sum_i nibble_o,s,i*qx_t,s,i
//
// and computes it with qGemm8. Q4_K's affine min term cannot be represented as a Q8 dot, so it is
// applied afterwards from block sums of the already-quantized activation panel:
//
//	min[t,o] = sum_s (min_o,s*m_o,s) * dx_t,s * sum_i qx_t,s,i
func q4kGemmExtractOnceInt8Into(qt *q4kTensor, qp *q8Panel, Y []float32) {
	if q4kGemmExtractOnceInt8IntoArch(qt, qp, Y) {
		return
	}
	ex := q4kExtractGemmWeights(qt)
	qGemm8Into(ex.qt, qp, Y)
	sums := q8PanelBlockSums(qp)
	q4kSubtractGemmMinTerm(ex.minScale, sums, qp.d, qp.P, qt.out, ex.qt.nblk, Y)
}

func q4kExtractGemmWeights(qt *q4kTensor) q4kExtractedGemm {
	nblk := qt.nblk * 8
	out, in := qt.out, qt.in
	q8 := &q8Tensor{
		out:  out,
		in:   in,
		nblk: nblk,
		q:    make([]int8, out*in),
		d:    make([]float32, out*nblk),
	}
	minScale := make([]float32, out*nblk)
	rowBytes := qt.q4kRowBytes()
	body := func(lo, hi int) {
		for o := lo; o < hi; o++ {
			q4kExtractGemmRow(
				qt.raw[o*rowBytes:(o+1)*rowBytes],
				q8.q[o*in:(o+1)*in],
				q8.d[o*nblk:(o+1)*nblk],
				minScale[o*nblk:(o+1)*nblk],
				qt.nblk,
			)
		}
	}
	if out*in < parThreshold {
		body(0, out)
	} else {
		parFor(out, numWorkers, body)
	}
	return q4kExtractedGemm{qt: q8, minScale: minScale}
}

func q4kExtractGemmRow(row []byte, qdst []int8, ddst, mdst []float32, nblk int) {
	for b := 0; b < nblk; b++ {
		blk := row[b*q4kBlockBytes : (b+1)*q4kBlockBytes]
		d := math.Float32frombits(F16BitsToF32Bits(binary.LittleEndian.Uint16(blk[0:])))
		min := math.Float32frombits(F16BitsToF32Bits(binary.LittleEndian.Uint16(blk[2:])))
		scales := blk[4 : 4+12]
		q := blk[4+12 : q4kBlockBytes]
		qBase := b * qkK
		scaleBase := b * 8
		qi := 0
		for k := 0; k < 4; k++ {
			loBlock := 2 * k
			sc, m := GetScaleMinK4(loBlock, scales)
			ddst[scaleBase+loBlock] = d * float32(sc)
			mdst[scaleBase+loBlock] = min * float32(m)
			for l := 0; l < qBlk; l++ {
				qdst[qBase+loBlock*qBlk+l] = int8(q[qi+l] & 0x0f)
			}

			hiBlock := loBlock + 1
			sc, m = GetScaleMinK4(hiBlock, scales)
			ddst[scaleBase+hiBlock] = d * float32(sc)
			mdst[scaleBase+hiBlock] = min * float32(m)
			for l := 0; l < qBlk; l++ {
				qdst[qBase+hiBlock*qBlk+l] = int8(q[qi+l] >> 4)
			}
			qi += qBlk
		}
	}
}

func q8PanelBlockSums(qp *q8Panel) []int32 {
	sums := make([]int32, qp.P*qp.nblk)
	body := func(lo, hi int) {
		for t := lo; t < hi; t++ {
			qrow := qp.q[t*qp.in : (t+1)*qp.in]
			srow := sums[t*qp.nblk : (t+1)*qp.nblk]
			for b := 0; b < qp.nblk; b++ {
				qb := qrow[b*qBlk : b*qBlk+qBlk]
				var s int32
				for i := 0; i < qBlk; i++ {
					s += int32(qb[i])
				}
				srow[b] = s
			}
		}
	}
	if qp.P*qp.in < parThreshold {
		body(0, qp.P)
	} else {
		parFor(qp.P, numWorkers, body)
	}
	return sums
}

func q4kSubtractGemmMinTerm(minScale []float32, sums []int32, dx []float32, P, out, nblk int, Y []float32) {
	body := func(lo, hi int) {
		for o := lo; o < hi; o++ {
			ms := minScale[o*nblk : (o+1)*nblk]
			for t := 0; t < P; t++ {
				Y[t*out+o] -= q4kGemmMinTerm(ms, sums[t*nblk:(t+1)*nblk], dx[t*nblk:(t+1)*nblk], nblk)
			}
		}
	}
	if out*P*nblk < parThreshold {
		body(0, out)
	} else {
		parFor(out, numWorkers, body)
	}
}

func q4kGemmMinTerm(minScale []float32, sums []int32, dx []float32, nblk int) float32 {
	var sub float32
	for b := 0; b < nblk; b++ {
		sub += float32(sums[b]) * minScale[b] * dx[b]
	}
	return sub
}

// q4kGemmRangeInt8 is the legacy int8 SDOT batched-GEMM row loop. It is kept as a focused
// reference/benchmark for the pre-#60 path: for each output row it runs q4kReduceRow against
// every activation row, so it re-reads and unpacks the same Q4_K row P times.
func q4kGemmRangeInt8(qt *q4kTensor, qvs []q8Vec, P int, Y []float32, lo, hi int) {
	nblk := qt.nblk
	out := qt.out
	IS := make([]int32, nblk*8)
	SS := make([]int32, nblk*8)
	rowBytes := qt.q4kRowBytes()
	for o := lo; o < hi; o++ {
		row := qt.raw[o*rowBytes:]
		for t := 0; t < P; t++ {
			q4kReduceRow(row, nblk, qvs[t].q, IS, SS)
			Y[t*out+o] = q4kCombineRow(row, nblk, qvs[t].d, IS, SS)
		}
	}
}

// q4kGemmRange computes Y[t*out+o] for o in [lo,hi), all t in [0,P). One dequant scratch
// buffer per worker-row (L1/L2-resident); each super-block is dequantized once per output
// row and dotted against all P activation rows. The per-(o,t) reduction matches
// q4kMatRowsRange exactly (same dequant, same 4-accumulator dot, same super-block order).
func q4kGemmRange(qt *q4kTensor, X []float32, P int, Y []float32, lo, hi int) {
	buf := make([]float32, qkK) // 256 f32, reused per (o,b); L1/L2-resident
	rowBytes := qt.q4kRowBytes()
	acc := make([]float32, P) // per-token accumulator, reused per output row
	for o := lo; o < hi; o++ {
		row := qt.raw[o*rowBytes:]
		for t := 0; t < P; t++ {
			acc[t] = 0
		}
		for b := 0; b < qt.nblk; b++ {
			q4kDequantSuperBlock(buf, row[b*q4kBlockBytes:(b+1)*q4kBlockBytes])
			for t := 0; t < P; t++ {
				xs := X[t*qt.in+b*qkK:]
				var s0, s1, s2, s3 float32
				for i := 0; i < qkK; i += 4 {
					s0 += buf[i] * xs[i]
					s1 += buf[i+1] * xs[i+1]
					s2 += buf[i+2] * xs[i+2]
					s3 += buf[i+3] * xs[i+3]
				}
				acc[t] += (s0 + s1) + (s2 + s3)
			}
		}
		for t := 0; t < P; t++ {
			Y[t*qt.out+o] = acc[t]
		}
	}
}

// q4kKernel runs the single-position block in resident Q4_K: prep is the identity (the GEMV
// dequantizes weight super-blocks on the fly and consumes the f32 activation directly, so
// the same prepared activation is reused across q/k/v and gate/up exactly as f32Kernel
// does); mul applies the named resident Q4_K weight. Like q4Kernel this avoids the
// activation quantization of the Q8 path — raw Q4_K weights + f32 activation is the
// llama.cpp q4_k_m decode regime.
type q4kKernel struct{ m *Model }

func (k q4kKernel) prep(x []float32) any { return x }

func (k q4kKernel) mul(name string, x any, out, in int) []float32 {
	return q4kMatRows(k.m.q4k(name), x.([]float32))
}

// sessionQ4KKernel is the allocation-light twin mirroring sessionQ4Kernel/sessionQ8Kernel:
// the Q4_K GEMV takes f32 activations directly, so prep is the identity and there is no
// per-call quant scratch to reuse; it exists only to match the blockStep dispatch shape.
type sessionQ4KKernel struct{ s *Session }

func (k sessionQ4KKernel) prep(x []float32) any { return x }

func (k sessionQ4KKernel) mul(name string, x any, out, in int) []float32 {
	xf := x.([]float32)
	if qt := k.s.M.q4kw[name]; qt != nil {
		// Resident raw Q4_K matmul weight (the q4_k_m majority): inline-dequant GEMV on CPU,
		// or the Metal q4_k GEMV under MetalQ4K (q4kMatRowsDispatch).
		return k.s.q4kMatRowsDispatch(name, qt, xf)
	}
	if qt := k.s.M.kqw[name]; qt != nil {
		// Resident Q5_K/Q6_K matmul weight (e.g. the q4_k_m expert down_proj, which loads Q6_K
		// into kqw, not q4kw). Use the resident k-quant GEMV — its int8 reducer quantizes the
		// activation once and is byte-identical to the f32 dequant-then-dot (kQuantMatRows;
		// TestKQuantMatRowsMatchesF32). Without this the weight would miss q4kw and fall to the
		// Q8 dequant-and-requantize path below, the slowest available route for these experts.
		return kQuantMatRows(qt, xf)
	}
	// Quant matmul weights with no resident Q4_K or k-quant copy (qwen3.5 attn_qkv → split
	// q/k/v) fall back to the proven Q8_0 GEMV. The f32 activation is quantized on demand for
	// this minority, so prep staying the identity for the Q4_K majority is fine.
	return qMatRows(k.s.M.q8(name), quantizeVecQ8(xf))
}

// mulGroup applies several weights that share ONE prepared activation xn (same in), returning one
// result per name in order. Under the resident-Q4_K session kernel with MetalQ4K it runs the
// q4_k-resident members of the group in a SINGLE Metal command buffer (q4kGroupDispatch) — the
// decode lever (one command buffer instead of one-per-matmul) — and falls the Q8 minority back to
// the per-call CPU GEMV. Every other kernel (and the pure-Go build) loops mul; the results are
// bit-identical either way (GEMVGroup == per-weight GEMV, pinned by TestMetalQ4KGemvGroupMatchesSingle).
func mulGroup(mat matKernel, names []string, xn any, outs []int, in int) [][]float32 {
	if sk, ok := mat.(sessionQ4KKernel); ok {
		if xf, ok2 := xn.([]float32); ok2 {
			if r := sk.s.q4kGroupDispatch(names, xf, outs); r != nil {
				return r
			}
		}
	}
	r := make([][]float32, len(names))
	for i, n := range names {
		r[i] = mat.mul(n, xn, outs[i], in)
	}
	return r
}

// quantizeQ4KFromRaw wraps a raw GGUF Q4_K payload (row-major, in == nblk*qkK) as a
// resident q4kTensor with NO transform — the bytes ARE the GGUF bytes. This is the
// direct-q4 loader entry the plan calls for: it skips ggufload's dequantF32 (Q4→f32) and
// the f32→Q8 re-quant entirely, which is the ~10× load win and the drop in resident
// footprint (no q8w co-residency). The ggufload side hands each Q4_K tensor's payload
// slice here verbatim.
func quantizeQ4KFromRaw(raw []byte, out, in int) *q4kTensor {
	if in%qkK != 0 {
		// Mirror the Q8/Q4_0 guards: a non-multiple-of-256 reduction dim would otherwise
		// mis-super-block every row; fail loudly. Every Qwen3.6 reduction dim is a
		// multiple of 256, so this only fires on a model that violates the Q4_K precondition.
		panic("model: Q4_K reduction dim not a multiple of 256")
	}
	nblk := in / qkK
	want := out * nblk * q4kBlockBytes
	if len(raw) != want {
		panic("model: Q4_K payload size mismatch")
	}
	raw = pageAlignResidentBytes(raw)
	return &q4kTensor{out: out, in: in, nblk: nblk, raw: raw}
}

func pageAlignResidentBytes(raw []byte) []byte {
	if len(raw) == 0 {
		return raw
	}
	page := os.Getpagesize()
	if page <= 1 {
		return raw
	}
	rounded := pageRoundResidentLen(len(raw), page)
	ptrAligned := uintptr(unsafe.Pointer(&raw[0]))%uintptr(page) == 0
	if ptrAligned && len(raw) == rounded {
		return raw
	}
	backing := make([]byte, rounded+page)
	base := uintptr(unsafe.Pointer(&backing[0]))
	off := int((uintptr(page) - base%uintptr(page)) % uintptr(page))
	out := backing[off : off+len(raw)]
	copy(out, raw)
	return out
}

func pageRoundResidentLen(n, page int) int {
	if n <= 0 || page <= 1 {
		return n
	}
	rem := n % page
	if rem == 0 {
		return n
	}
	pad := page - rem
	if n > math.MaxInt-pad {
		return n
	}
	return n + pad
}

// q4k returns the prebuilt resident Q4_K tensor for a name (QuantizeQ4K must have run).
func (m *Model) q4k(name string) *q4kTensor {
	qt, ok := m.q4kw[name]
	if !ok {
		panic("model: resident Q4_K tensor not built: " + name + " (call Model.QuantizeQ4K)")
	}
	return qt
}

// Q4KCount returns how many tensors hold a resident raw Q4_K copy (diagnostic for the loader).
func (m *Model) Q4KCount() int { return len(m.q4kw) }

// Q4KShape returns the (out, in) shape of one resident Q4_K tensor, or (0,0) if absent.
func (m *Model) Q4KShape(name string) (out, in int) {
	if qt := m.q4kw[name]; qt != nil {
		return qt.out, qt.in
	}
	return 0, 0
}

// ResidentQ4KEligible reports whether a canonical tensor name should be held as resident
// raw Q4_K (so the loader skips the f32 round-trip for it). It must be (a) a matmul weight
// after the qwen35 source-name chain, AND (b) IDENTITY-normalized — i.e. not one of the
// tensors normalizeCanonicalTensorData transforms for qwen35 (self_attn q/k/qkv unpermutes,
// and the linear_attn family reorder). Storing a transformed tensor's raw GGUF bytes would
// feed wrongly-laid-out weights to the forward and produce garbage (the first-token failure
// this predicate existed to fix). Identity matmul weights (FFN gate/up/down, self_attn
// v_proj/o_proj, expert FFN, lm_head) are safe to hold raw. The loader still gates on GGUF
// type == Q4_K, so a Q6_K weight that is name-eligible (attn_qkv, ffn_down, the lm_head) is
// routed to Q8 by the type check, not Q4_K.
func ResidentQ4KEligible(cfg Config, canon string) bool {
	name, keep := quantSourceTensorName(cfg, canon)
	if !keep || !isQuantWeight(name) {
		return false
	}
	// self_attn q/k/qkv are unpermuted (rotary / gated) for qwen35 → not identity.
	switch {
	case strings.HasSuffix(name, suffixQKVProj),
		strings.HasSuffix(name, suffixQProj),
		strings.HasSuffix(name, suffixKProj):
		return false
	}
	// Every linear_attn.* matmul tensor is reordered/unpermuted for qwen35 (nK=16 ≠ nV=48)
	// → not identity. (in_proj_z absorbs q_gate_proj via the source chain.)
	if strings.Contains(name, ".linear_attn.") {
		return false
	}
	return true
}

// residentQuantTarget runs the shared eligibility gate for the resident raw-quant
// AddResident* builders (Q4_K and the k-quant kinds): it refuses a built builder, resolves
// the canonical name through the qwen35 source chain, and reports whether this tensor should
// be stored as a resident raw-quant block (a 2-D quant weight that is not a fused QKV /
// gate_up projection). ok==false with err==nil means "skip, not eligible" (idempotent).
func (b *QuantBuilder) residentQuantTarget(canon string, shape []int) (name string, ok bool, err error) {
	if b.built {
		return "", false, fmt.Errorf("model: QuantBuilder already built")
	}
	name, keep := quantSourceTensorName(b.m.Cfg, canon)
	if !keep || !isQuantWeight(name) || len(shape) != 2 {
		return "", false, nil
	}
	if strings.HasSuffix(name, suffixQKVProj) || strings.HasSuffix(name, suffixGateUpProj) {
		return "", false, nil
	}
	return name, true, nil
}

// AddResidentQ4K stores a raw Q4_K payload as a resident q4kTensor under the canonical name
// resolved through the qwen35 source chain, skipping the f32/Q8 round-trip. shape is the
// model [out, in] convention (in a multiple of 256). Idempotent for non-eligible names
// (returns nil without storing) so the loader can call it unconditionally on Q4_K tensors.
func (b *QuantBuilder) AddResidentQ4K(canon string, shape []int, raw []byte) error {
	name, ok, err := b.residentQuantTarget(canon, shape)
	if !ok || err != nil {
		return err
	}
	if b.m.q4kw == nil {
		b.m.q4kw = map[string]*q4kTensor{}
	}
	b.m.q4kw[name] = quantizeQ4KFromRaw(raw, shape[0], shape[1])
	return nil
}
