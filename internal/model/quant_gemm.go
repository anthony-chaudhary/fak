package model

import (
	"math"
	"os"
	"runtime"
	"strconv"
)

const (
	qgemmModeLegacy = "legacy"
	qgemmModeTile   = "tile"
)

// qgemmMode pins the prefill-GEMM path for A/B measurement: "legacy" forces the old
// per-element qdot8 sweep (qGemm8legacy), anything else uses the register-blocked tile
// kernel. The tile kernel's SIMD tier follows qtier, so FAK_QKERNEL still selects
// AVX-512/AVX2/scalar consistently with decode.
var qgemmMode = initQGemmMode()

func initQGemmMode() string {
	env, ok := os.LookupEnv("FAK_QGEMM")
	return resolveQGemmMode(env, ok, runtime.GOARCH)
}

var qgemmGroup = initQGemmGroup()

func initQGemmGroup() bool {
	switch os.Getenv("FAK_QGEMM_GROUP") {
	case "0", "false", "False", "FALSE", "off", "OFF":
		return false
	default:
		return true
	}
}

var qgemmGroupMaxP = initQGemmGroupMaxP()

func initQGemmGroupMaxP() int {
	if s := os.Getenv("FAK_QGEMM_GROUP_MAXP"); s != "" {
		if n, err := strconv.Atoi(s); err == nil && n >= 1 {
			return n
		}
	}
	// Grouping avoids repeated launch barriers across q/k/v and gate/up. After the batched
	// activation/value loops moved off the serial path, Zen5 measurements favor grouping
	// through B=1024; larger batches do not move the peak.
	return 1024
}

func resolveQGemmMode(env string, envSet bool, goarch string) string {
	if envSet {
		return env
	}
	return qgemmModeTile
}

// quant_gemm.go — the register-blocked Q8_0 prefill GEMM: close the prefill gap to
// llama.cpp Q8_0 on its own terms (same format, same int8 SIMD ISA), the residual
// MODEL-BASELINE-RESULTS.md named as "single-block reduce vs a hand-tuned blocked
// micro-kernel".
//
// Why the old qMatMulBatch left throughput on the table. It computed ONE output element
// per qdot8 call, and qdot8 horizontally reduces the block's int32 lanes to a scalar
// INSIDE every block (≈7 shuffle/add ops per block, all latency-bound) just to fold one
// block into the float accumulator. Prefill is compute-bound, so that per-block reduction
// — issued out·P·nblk times — dominates the useful VPMADDWD work ≈11:1, and nothing in the
// per-element loop is reused across the GEMM's two free axes (output rows, tokens).
//
// What this lane does instead — the textbook GEMM micro-kernel, two changes:
//   1. DEFERRED REDUCTION. Keep the block dot's int32 lanes in a VECTOR float accumulator
//      and fold each block in with a single cvt + FMA (scale·int + acc, one rounding);
//      reduce the lanes to a scalar ONCE per output, at the end. The per-block horizontal
//      shuffle vanishes from the inner loop (it now runs once per output, not once per
//      output·block).
//   2. REGISTER BLOCKING. Compute an MR×NR output tile at once (here 5×4 = 20 accumulators
//      held live in zmm). Each loaded+sign-extended weight block feeds NR tokens and each
//      activation block feeds MR rows before eviction — so the int8 loads and sign-extends
//      are amortised MR/NR-fold, raising register-level arithmetic intensity. This is the
//      same structure llama.cpp's tinyBLAS uses for block_q8_0.
//
// Numerics. This deliberately does NOT match qdot8scalar's per-block-scalar reduction
// order — it reduces later, in vector lanes, and folds each block with a single-rounded FMA
// — so it is NOT bit-identical to the decode kernel, and is in fact MORE accurate to the
// true real-valued GEMM (pairwise lane folding + single-rounded FMA): q8bench measures the
// FMA path's last-logit max|Δ| vs the HF oracle (ground truth) as LOWER on every prompt, and
// argmax stays exact 25/25 vs that oracle. It IS held bit-identical to its OWN scalar
// reference qgemm8cell (whose per-block accumulate also uses math.FMA, matching VFMADD231PS
// via innocuous 53→24 double rounding) by TestQGemm8AsmMatchesScalar. The Q8 path's
// authoritative gate is unchanged — argmax-exact vs the HF oracle; the proven f32 path and
// the decode Q8 path (qdot8) are untouched.

// q8Panel is a contiguous Q8_0-quantized activation panel: P rows (tokens), each `in` int8
// codes + `nblk` per-block f32 scales, row-major by token. Unlike []q8Vec (one
// separately-allocated vector per token), a panel lays every token in two flat buffers, so
// the tile kernel can stride across token-columns with a FIXED stride (== the inner dim)
// instead of chasing NR independent slice pointers — the contiguity register blocking needs.
type q8Panel struct {
	q           []int8    // P*in codes: token t, block b, byte i at q[t*in + b*qBlk + i]
	d           []float32 // P*nblk scales: token t, block b at d[t*nblk + b]
	P, in, nblk int
}

type qgemm8Target struct {
	qt *q8Tensor
	Y  []float32
}

// quantizeBatchPanel quantizes P activation rows of width `width` (row-major X) into one
// contiguous Q8_0 panel, parallel across rows. Identical per-block Q8_0 to quantizeVecQ8;
// only the storage is flattened so the GEMM can stride tokens. Built once per distinct
// activation set and reused across the matmuls that share it (q/k/v share one panel).
func quantizeBatchPanel(X []float32, P, width int) *q8Panel {
	qp := &q8Panel{}
	quantizeBatchPanelInto(qp, X, P, width)
	return qp
}

// quantizeBatchPanelInto quantizes into an EXISTING panel, growing its buffers only when
// they are too small. Prefill builds four activation panels per layer (q/k/v share one,
// then o, gate/up, down) consumed sequentially, so a single reused scratch panel serves all
// 4×NumLayers quantizations — eliminating ~120 large slice allocations (and their GC) per
// prefill. Zero blocks (d==0) must be written explicitly here (not skipped as in the
// allocate-fresh path) so a reused buffer never leaks a prior call's codes.
func quantizeBatchPanelInto(qp *q8Panel, X []float32, P, width int) {
	if width%qBlk != 0 {
		panic("model: Q8_0 activation width not a multiple of 32")
	}
	nblk := width / qBlk
	if cap(qp.q) < P*width {
		qp.q = make([]int8, P*width)
	} else {
		qp.q = qp.q[:P*width]
	}
	if cap(qp.d) < P*nblk {
		qp.d = make([]float32, P*nblk)
	} else {
		qp.d = qp.d[:P*nblk]
	}
	qp.P, qp.in, qp.nblk = P, width, nblk
	parFor(P, numWorkers, func(lo, hi int) {
		for t := lo; t < hi; t++ {
			// quantizeRowQ8 is the AVX-512 kernel (scalar fallback off-512), bit-identical
			// to the per-block math this loop used to inline — see quant_quantize.go.
			quantizeRowQ8(X[t*width:(t+1)*width], qp.q[t*width:(t+1)*width], qp.d[t*nblk:(t+1)*nblk], nblk)
		}
	})
}

// qgemm8cell is the REFERENCE Q8_0 dot in the exact deferred-reduction order the tile
// kernel uses — the bit-identity contract for qgemm8tile512 (lanes=16), and the scalar
// reference for the AVX2/NEON lane geometries. For each block it forms `lanes` int32
// partials with the target ISA's dot-lane semantics, converts them to float (exact:
// |partial| <= 8*127^2 < 2^24), scales by dw[b]*dx[b], and FMA-accumulates into `lanes`
// float lanes — reducing to a scalar only once, via the SAME pairwise tree the asm emits.
// The per-block accumulate is math.FMA, matching the asm's VFMADD231PS bit-for-bit (the
// 53->24 narrowing is innocuous double rounding); the final lane reduction stays plain
// adds, matching the asm's unfused VADDPS.
//
// lanes==16 mirrors AVX-512 (one zmm, 16 int32 from one VPMADDWD); lanes==8 mirrors AVX2
// (one ymm, two VPMADDWD halves summed); lanes==4 mirrors NEON SDOT (one q-reg, four int32
// lanes, lane k = the 4-byte group 4k..4k+3 from each 16-byte half). Any other value is a
// programming error.
func qgemm8cell(qw []int8, dw []float32, qx []int8, dx []float32, nblk, lanes int) float32 {
	if lanes != 16 && lanes != 8 && lanes != 4 {
		panic("model: qgemm8cell lanes must be 16, 8, or 4")
	}
	var acc [16]float32
	for b := 0; b < nblk; b++ {
		wb := qw[b*qBlk : b*qBlk+qBlk]
		xb := qx[b*qBlk : b*qBlk+qBlk]
		var p [16]int32
		if lanes == 16 {
			for k := 0; k < 16; k++ {
				p[k] = int32(wb[2*k])*int32(xb[2*k]) + int32(wb[2*k+1])*int32(xb[2*k+1])
			}
		} else if lanes == 8 { // AVX2 sums the two 16-byte halves into the same lane
			for k := 0; k < 8; k++ {
				p[k] = int32(wb[2*k])*int32(xb[2*k]) + int32(wb[2*k+1])*int32(xb[2*k+1]) +
					int32(wb[16+2*k])*int32(xb[16+2*k]) + int32(wb[16+2*k+1])*int32(xb[16+2*k+1])
			}
		} else { // lanes == 4: NEON SDOT accumulates four byte-pairs per q lane, per half.
			for k := 0; k < 4; k++ {
				var sum int32
				for j := 0; j < 4; j++ {
					sum += int32(wb[4*k+j])*int32(xb[4*k+j]) +
						int32(wb[16+4*k+j])*int32(xb[16+4*k+j])
				}
				p[k] = sum
			}
		}
		s := dw[b] * dx[b]
		for k := 0; k < lanes; k++ {
			// FMA — bit-identical to the asm's VFMADD231PS (round24(float32(p)*s + acc), one
			// rounding). float32(p[k]) is exact (|p[k]| < 2^24) so float64(p[k]) equals it;
			// math.FMA gives round53(p*s+acc) and the 53->24 narrowing is innocuous double
			// rounding (53 >= 2*24+2), so float32(math.FMA(...)) == round24(p*s+acc).
			acc[k] = float32(math.FMA(float64(p[k]), float64(s), float64(acc[k])))
		}
	}
	// Reduce `lanes` accumulators to a scalar in the SAME tree order the asm uses:
	// fold the top half into the bottom (16→8→… ), then a fixed 4→2→1 finish.
	if lanes == 16 {
		for k := 0; k < 8; k++ {
			acc[k] += acc[k+8]
		}
	}
	if lanes != 4 {
		for k := 0; k < 4; k++ {
			acc[k] += acc[k+4]
		}
	}
	acc[0] += acc[2]
	acc[1] += acc[3]
	return acc[0] + acc[1]
}

// qGemm8scalar is the fully-portable batched Q8_0 GEMM: Y[t·out+o] = qgemm8cell(weight row
// o, token t), row-parallel across output rows. It is the non-amd64 path and the test
// oracle the amd64 tile kernel is pinned against (run with the matching lane count). Output
// is row-major [P, out].
func qGemm8scalar(qt *q8Tensor, qp *q8Panel, lanes int) []float32 {
	Y := make([]float32, qp.P*qt.out)
	qGemm8scalarInto(qt, qp, lanes, Y)
	return Y
}

// qGemm8scalarInto is qGemm8scalar writing into a caller-provided Y (len >= P*out), so the
// hot decode loop can reuse one buffer across steps instead of allocating P*out floats per
// GEMM. Identical arithmetic — only the output's backing memory changes.
func qGemm8scalarInto(qt *q8Tensor, qp *q8Panel, lanes int, Y []float32) {
	out, in, nblk, P := qt.out, qt.in, qt.nblk, qp.P
	body := func(lo, hi int) {
		for o := lo; o < hi; o++ {
			qw := qt.q[o*in : o*in+in]
			dw := qt.d[o*nblk : o*nblk+nblk]
			for t := 0; t < P; t++ {
				Y[t*out+o] = qgemm8cell(qw, dw, qp.q[t*in:t*in+in], qp.d[t*nblk:t*nblk+nblk], nblk, lanes)
			}
		}
	}
	if out*in*P < parThreshold {
		body(0, out)
		return
	}
	parFor(out, numWorkers, body)
}

// qGemm8legacy reproduces the pre-tile shipped behaviour (one qdot8 GEMV call per output
// element, reading from the panel), so FAK_QGEMM=legacy can A/B the tile kernel against the
// old path in the SAME process under the SAME machine load. Kept only for measurement.
func qGemm8legacy(qt *q8Tensor, qp *q8Panel) []float32 {
	Y := make([]float32, qp.P*qt.out)
	qGemm8legacyInto(qt, qp, Y)
	return Y
}

// qGemm8legacyInto is qGemm8legacy writing into a caller-provided Y (the buffer-reuse form).
func qGemm8legacyInto(qt *q8Tensor, qp *q8Panel, Y []float32) {
	out, in, nblk, P := qt.out, qt.in, qt.nblk, qp.P
	body := func(lo, hi int) {
		for o := lo; o < hi; o++ {
			qw := qt.q[o*in : o*in+in]
			dw := qt.d[o*nblk : o*nblk+nblk]
			for t := 0; t < P; t++ {
				qv := q8Vec{q: qp.q[t*in : t*in+in], d: qp.d[t*nblk : t*nblk+nblk], nblk: nblk}
				Y[t*out+o] = qdot8(qw, dw, qv, nblk)
			}
		}
	}
	if out*in*P < parThreshold {
		body(0, out)
		return
	}
	parFor(out, numWorkers, body)
}
