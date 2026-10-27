//go:build amd64

package model

// qGemm8SIMDTokenTail replaces serial scalar token-tail arithmetic with the
// existing exact tile kernels. Only the absent token columns are zero-padded;
// the caller's panel and weights are never changed. SIMD lanes within a valid
// output cell do not mix with padded cells, so its FMA/reduction order is intact.
func qGemm8SIMDTokenTail(qt *q8Tensor, qp *q8Panel, Y []float32, first int, wide bool) bool {
	if first == qp.P || qt.out == 0 {
		return true
	}
	mr, nr := qgemmMR256, qgemmNR256
	if wide {
		mr, nr = qgemmMR, 4
	}
	rows := qp.P - first
	in, nblk, out := qt.in, qt.nblk, qt.out
	if first < 0 || rows <= 0 || rows >= nr || in <= 0 || nblk <= 0 ||
		in != nblk*qBlk || qp.in != in || qp.nblk != nblk ||
		(wide && qtier != tierAVX512) || (!wide && qtier < tierAVX2) {
		return false
	}
	qpad := make([]int8, nr*in)
	dpad := make([]float32, nr*nblk)
	copy(qpad, qp.q[first*in:qp.P*in])
	copy(dpad, qp.d[first*nblk:qp.P*nblk])
	tiles := 1 + (out-1)/mr
	parForRange(tiles, out*in*rows, func(lo, hi int) {
		var values [20]float32 // at most 5 rows x 4 tokens, worker-owned
		for tile := lo; tile < hi; tile++ {
			o := tile * mr
			valid := min(mr, out-o)
			if wide {
				if valid == mr {
					qgemm8tile512(&qt.q[o*in], &qpad[0], &qt.d[o*nblk], &dpad[0], in, nblk, mr, &values[0])
				} else {
					for r := 0; r < valid; r++ {
						qgemm8tile512x1(&qt.q[(o+r)*in], &qpad[0], &qt.d[(o+r)*nblk], &dpad[0], in, nblk, mr, &values[r])
					}
				}
			} else {
				qw, dw := qt.q[o*in:], qt.d[o*nblk:]
				if valid != mr {
					// Only the final row tile needs padding; no whole weight copy.
					qw, dw = make([]int8, mr*in), make([]float32, mr*nblk)
					copy(qw, qt.q[o*in:out*in])
					copy(dw, qt.d[o*nblk:out*nblk])
				}
				qgemm8tile256(&qw[0], &qpad[0], &dw[0], &dpad[0], in, nblk, mr, &values[0])
			}
			for t := 0; t < rows; t++ {
				copy(Y[(first+t)*out+o:(first+t)*out+o+valid], values[t*mr:t*mr+valid])
			}
		}
	})
	return true // existing worker barrier completes before scratch is released
}

// qGemm8AVX2RowTail handles only the one/two final weight rows and the even
// token bulk. Padding is local and bounded by three weight rows; caller-owned
// output sees only valid cells. The odd token column belongs to the helper above.
func qGemm8AVX2RowTail(qt *q8Tensor, qp *q8Panel, Y []float32, first, tokens int) bool {
	if first == qt.out || tokens == 0 {
		return true
	}
	in, nblk, out := qt.in, qt.nblk, qt.out
	rows := out - first
	if qtier < tierAVX2 || first < 0 || rows <= 0 || rows >= qgemmMR256 ||
		tokens < 0 || tokens > qp.P || tokens%qgemmNR256 != 0 || in <= 0 || nblk <= 0 ||
		in != nblk*qBlk || qp.in != in || qp.nblk != nblk {
		return false
	}
	qw := make([]int8, qgemmMR256*in)
	dw := make([]float32, qgemmMR256*nblk)
	copy(qw, qt.q[first*in:out*in])
	copy(dw, qt.d[first*nblk:out*nblk])
	var values [qgemmMR256 * qgemmNR256]float32
	for t := 0; t < tokens; t += qgemmNR256 {
		qgemm8tile256(&qw[0], &qp.q[t*in], &dw[0], &qp.d[t*nblk], in, nblk, qgemmMR256, &values[0])
		for c := 0; c < qgemmNR256; c++ {
			copy(Y[(t+c)*out+first:(t+c+1)*out], values[c*qgemmMR256:c*qgemmMR256+rows])
		}
	}
	return true
}
