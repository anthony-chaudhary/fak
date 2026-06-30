//go:build arm64 && !(fakaccel && darwin && cgo)

package model

// quant_arm64_q4k.go — arm64 dispatch for the resident-Q4_K int8 decode reduction. The NEON
// SDOT kernel (quant_arm64_q4k.s) computes the per-sub-block integer reductions (I_s = Σ
// nibble*qx via SDOT, S_s = Σ qx via SADDLV) for a whole row; the float combine stays in shared
// Go (q4kCombineRow), so asm correctness reduces to "the int32 reductions match the scalar
// reference" (TestQ4KReduceAsmMatchesScalar) — integer SDOT is associative with no overflow on
// these ranges, so any lane order is bit-identical. Falls back to the scalar reference on an
// arm64 part without FEAT_DotProd (neonDot). FAK_QKERNEL=scalar pins the f32 path (q4kSDOTEnabled
// observes neonDot, which observes FAK_QKERNEL).

//go:noescape
func q4kReduceRowAsm(row *byte, nblk int, qx *int8, IS, SS *int32)

// q4kSDOTEnabled reports whether the resident-Q4_K int8 decode path is active: arm64 with
// FEAT_DotProd (SDOT) and FAK_QKERNEL not pinning scalar, unless a test forces it off via
// setQ4KSDOTForTest. When false, q4kMatRowsInto keeps the byte-identical f32 scalar GEMV (the
// path TestQ4KMatRowsMatchesF32 pins).
func q4kSDOTEnabled() bool {
	if q4kSDOTForce != 0 {
		return q4kSDOTForce > 0
	}
	return neonDot
}

func q4kExtractOnceGemmEnabled() bool {
	return neonDot
}

func q4kGemmExtractOnceInt8IntoArch(qt *q4kTensor, qp *q8Panel, Y []float32) bool {
	if !neonDot {
		return false
	}
	q4kGemmExtractOnceInt8IntoArm64(qt, qp, Y)
	return true
}

func q4kGemmExtractOnceInt8IntoArm64(qt *q4kTensor, qp *q8Panel, Y []float32) {
	out, in, nblk, P := qt.out, qt.in, qt.nblk*8, qp.P
	rowBytes := qt.q4kRowBytes()
	sums := q8PanelBlockSums(qp)
	body := func(lo, hi int) {
		qbuf := make([]int8, 2*in)
		dbuf := make([]float32, 2*nblk)
		mbuf := make([]float32, 2*nblk)
		ybuf := make([]float32, 2*P)
		for o := lo; o < hi; {
			rows := 1
			q4kExtractGemmRow(
				qt.raw[o*rowBytes:(o+1)*rowBytes],
				qbuf[:in],
				dbuf[:nblk],
				mbuf[:nblk],
				qt.nblk,
			)
			if o+1 < hi {
				rows = 2
				q4kExtractGemmRow(
					qt.raw[(o+1)*rowBytes:(o+2)*rowBytes],
					qbuf[in:2*in],
					dbuf[nblk:2*nblk],
					mbuf[nblk:2*nblk],
					qt.nblk,
				)
			}
			q4kGemmExtractRowsArm64(qbuf[:rows*in], dbuf[:rows*nblk], mbuf[:rows*nblk], qp, sums, rows, out, o, Y, ybuf[:rows*P])
			o += rows
		}
	}
	if out*in*P < parThreshold {
		body(0, out)
	} else {
		parFor(out, numWorkers, body)
	}
}

func q4kGemmExtractRowsArm64(qbuf []int8, dbuf, mbuf []float32, qp *q8Panel, sums []int32, rows, out, rowBase int, Y, ybuf []float32) {
	in, nblk, P := qp.in, qp.nblk, qp.P
	for i := range ybuf {
		ybuf[i] = 0
	}
	Pmain := P &^ 3
	if rows == 2 {
		for t := 0; t < Pmain; t += 4 {
			qgemm8tile2x4NEON(&qbuf[0], &qp.q[t*in], &dbuf[0], &qp.d[t*nblk], in, nblk, rows, &ybuf[t*rows])
		}
	} else {
		for t := 0; t < Pmain; t += 4 {
			qgemm8row4NEON(&qbuf[0], &qp.q[t*in], &dbuf[0], &qp.d[t*nblk], in, nblk, rows, &ybuf[t*rows])
		}
	}
	for t := Pmain; t < P; t++ {
		qx := qp.q[t*in : (t+1)*in]
		dx := qp.d[t*nblk : (t+1)*nblk]
		for r := 0; r < rows; r++ {
			ybuf[t*rows+r] = qgemm8cell(qbuf[r*in:(r+1)*in], dbuf[r*nblk:(r+1)*nblk], qx, dx, nblk, 4)
		}
	}
	for r := 0; r < rows; r++ {
		ms := mbuf[r*nblk : (r+1)*nblk]
		for t := 0; t < P; t++ {
			sub := q4kGemmMinTerm(ms, sums[t*nblk:(t+1)*nblk], qp.d[t*nblk:(t+1)*nblk], nblk)
			Y[t*out+rowBase+r] = ybuf[t*rows+r] - sub
		}
	}
}

// q4kReduceRow dispatches the integer reduction to the NEON kernel when available, else the
// scalar reference. IS/SS are sized nblk*8 (one I_s/S_s per sub-block across all super-blocks).
func q4kReduceRow(row []byte, nblk int, qx []int8, IS, SS []int32) {
	if neonDot && nblk > 0 {
		q4kReduceRowAsm(&row[0], nblk, &qx[0], &IS[0], &SS[0])
		return
	}
	q4kReduceRowScalar(row, nblk, qx, IS, SS)
}
