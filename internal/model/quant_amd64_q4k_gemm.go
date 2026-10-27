//go:build amd64

package model

// q4kGemmExtractOnceInt8IntoAMD64 applies the bounded row extraction used by
// quant_arm64_q4k.go to the existing amd64 Q8 tiles. Each worker owns only MR
// rows of codes, scales and minimum scales, not a full expanded weight matrix.
// Extraction still runs once per output row, outside the token loop. The Q8
// reduction and affine subtraction are unchanged from the full-extraction path.
func q4kGemmExtractOnceInt8IntoAMD64(qt *q4kTensor, qp *q8Panel, Y []float32) {
	out, in, nblk, P := qt.out, qt.in, qt.nblk*8, qp.P
	mr, nr, lanes := qgemmMR, 4, 16
	if qtier == tierAVX2 {
		mr, nr, lanes = qgemmMR256, qgemmNR256, 8
	}
	rowBytes := qt.q4kRowBytes()
	sums := q8PanelBlockSums(qp)
	Pmain := P - P%nr
	nTiles := (out + mr - 1) / mr
	body := func(lo, hi int) {
		if lo >= hi {
			return
		}
		qbuf := make([]int8, mr*in)
		dbuf := make([]float32, mr*nblk)
		mbuf := make([]float32, mr*nblk)
		for tile := lo; tile < hi; tile++ {
			o := tile * mr
			rows := min(mr, out-o)
			for r := 0; r < rows; r++ {
				q4kExtractGemmRow(
					qt.raw[(o+r)*rowBytes:(o+r+1)*rowBytes],
					qbuf[r*in:(r+1)*in],
					dbuf[r*nblk:(r+1)*nblk],
					mbuf[r*nblk:(r+1)*nblk], qt.nblk,
				)
			}

			tail := Pmain
			if rows == mr {
				for t := 0; t < Pmain; t += nr {
					if lanes == 16 {
						qgemm8tile512(&qbuf[0], &qp.q[t*in], &dbuf[0], &qp.d[t*nblk], in, nblk, out, &Y[t*out+o])
					} else {
						qgemm8tile256(&qbuf[0], &qp.q[t*in], &dbuf[0], &qp.d[t*nblk], in, nblk, out, &Y[t*out+o])
					}
				}
			} else if lanes == 16 {
				for r := 0; r < rows; r++ {
					for t := 0; t < Pmain; t += nr {
						qgemm8tile512x1(&qbuf[r*in], &qp.q[t*in], &dbuf[r*nblk], &qp.d[t*nblk], in, nblk, out, &Y[t*out+o+r])
					}
				}
			} else {
				// The AVX2 dispatcher uses the lanes=8 scalar cell for row tails.
				tail = 0
			}
			for r := 0; r < rows; r++ {
				qw := qbuf[r*in : (r+1)*in]
				dw := dbuf[r*nblk : (r+1)*nblk]
				ms := mbuf[r*nblk : (r+1)*nblk]
				for t := tail; t < P; t++ {
					Y[t*out+o+r] = qgemm8cell(qw, dw, qp.q[t*in:(t+1)*in], qp.d[t*nblk:(t+1)*nblk], nblk, lanes)
				}
				for t := 0; t < P; t++ {
					Y[t*out+o+r] -= q4kGemmMinTerm(ms, sums[t*nblk:(t+1)*nblk], qp.d[t*nblk:(t+1)*nblk], nblk)
				}
			}
		}
	}
	if out*in*P < parThreshold {
		body(0, nTiles)
	} else {
		parFor(nTiles, currentWorkerCount(), body)
	}
}
