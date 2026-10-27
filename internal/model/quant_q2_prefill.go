package model

const q2PrefillTileRows = 16

// q2G128MatRowsBatchRange shares the immutable block dequant across tokens while
// retaining q2G128MatRowsRange's per-block four-lane fold and serial block sum.
func q2G128MatRowsBatchRange(qt *q2Tensor, X, Y []float32, rows, lo, hi int) {
	var block [qBlk2G128]float32
	rowBytes := qt.nblk * q2G128BlockBytes
	for o := lo; o < hi; o++ {
		row := qt.raw[o*rowBytes:]
		for first := 0; first < rows; first += q2PrefillTileRows {
			count := min(q2PrefillTileRows, rows-first)
			var acc [q2PrefillTileRows]float32
			for b := 0; b < qt.nblk; b++ {
				dequantQ2G128Block(block[:], row[b*q2G128BlockBytes:(b+1)*q2G128BlockBytes])
				for t := 0; t < count; t++ {
					xs := X[(first+t)*qt.in+b*qBlk2G128:]
					var s0, s1, s2, s3 float32
					for i := 0; i < qBlk2G128; i += 4 {
						s0 += block[i] * xs[i]
						s1 += block[i+1] * xs[i+1]
						s2 += block[i+2] * xs[i+2]
						s3 += block[i+3] * xs[i+3]
					}
					acc[t] += (s0 + s1) + (s2 + s3)
				}
			}
			for t := 0; t < count; t++ {
				Y[(first+t)*qt.out+o] = acc[t]
			}
		}
	}
}

// q2G32MatRowsBatchRange must not use the g128 block fold: q2MatRowsRange's
// prototype keeps four accumulators live across all 32-weight blocks.
func q2G32MatRowsBatchRange(qt *q2Tensor, X, Y []float32, rows, lo, hi int) {
	var block [qBlk2]float32
	const quarter = qBlk2 / 4
	for o := lo; o < hi; o++ {
		qrow := qt.q[o*qt.nblk*quarter:]
		drow := qt.d[o*qt.nblk:]
		for first := 0; first < rows; first += q2PrefillTileRows {
			count := min(q2PrefillTileRows, rows-first)
			var acc [q2PrefillTileRows][4]float32
			for b := 0; b < qt.nblk; b++ {
				dequantQ2Block(block[:], drow[b], qrow[b*quarter:])
				for t := 0; t < count; t++ {
					xs := X[(first+t)*qt.in+b*qBlk2:]
					a := &acc[t]
					a[0] += block[0]*xs[0] + block[1]*xs[1] + block[2]*xs[2] + block[3]*xs[3]
					a[1] += block[4]*xs[4] + block[5]*xs[5] + block[6]*xs[6] + block[7]*xs[7]
					a[2] += block[8]*xs[8] + block[9]*xs[9] + block[10]*xs[10] + block[11]*xs[11]
					a[3] += block[12]*xs[12] + block[13]*xs[13] + block[14]*xs[14] + block[15]*xs[15]
					a[0] += block[16]*xs[16] + block[17]*xs[17] + block[18]*xs[18] + block[19]*xs[19]
					a[1] += block[20]*xs[20] + block[21]*xs[21] + block[22]*xs[22] + block[23]*xs[23]
					a[2] += block[24]*xs[24] + block[25]*xs[25] + block[26]*xs[26] + block[27]*xs[27]
					a[3] += block[28]*xs[28] + block[29]*xs[29] + block[30]*xs[30] + block[31]*xs[31]
				}
			}
			for t := 0; t < count; t++ {
				a := acc[t]
				Y[(first+t)*qt.out+o] = (a[0] + a[1]) + (a[2] + a[3])
			}
		}
	}
}
