//go:build amd64

package model

// q4kBlockDot256x2AVX2 computes two independent exact four-accumulator dots
// against one decoded 256-value block. The output holds the two block sums.
//
//go:noescape
func q4kBlockDot256x2AVX2(weights, x0, x1, dst *float32)

func init() {
	q4kGemmF32RangeArch = q4kGemmF32RangeAVX2
}

// q4kGemmF32RangeAVX2 retains the scalar GEMM's once-per-block dequantization
// across all P tokens. Token pairs share each decoded weight load in the SIMD
// block dot, but never share an accumulator. Each block sum is then added in
// the original order. The existing caller owns parallel output-row partitioning.
func q4kGemmF32RangeAVX2(qt *q4kTensor, X []float32, P int, Y []float32, lo, hi int) bool {
	if qtier < tierAVX2 || qgemmMode == qgemmModeLegacy || qgemmMode == qgemmModeAccel ||
		qt == nil || P <= 1 || qt.out <= 0 || qt.in <= 0 || qt.nblk <= 0 ||
		qt.in%qkK != 0 || qt.nblk != qt.in/qkK ||
		P > len(X)/qt.in || P > len(Y)/qt.out || lo < 0 || hi > qt.out || lo > hi {
		return false
	}
	var buf [qkK]float32
	var dots [2]float32
	acc := make([]float32, P)
	rowBytes := qt.q4kRowBytes()
	paired := P &^ 1
	for o := lo; o < hi; o++ {
		row := qt.raw[o*rowBytes:]
		for t := range acc {
			acc[t] = 0
		}
		for b := 0; b < qt.nblk; b++ {
			q4kDequantSuperBlock(buf[:], row[b*q4kBlockBytes:(b+1)*q4kBlockBytes])
			for t := 0; t < paired; t += 2 {
				x0, x1 := t*qt.in+b*qkK, (t+1)*qt.in+b*qkK
				q4kBlockDot256x2AVX2(&buf[0], &X[x0], &X[x1], &dots[0])
				acc[t] += dots[0]
				acc[t+1] += dots[1]
			}
			if paired < P {
				// An odd token keeps the exact original scalar block dot.
				xs := X[paired*qt.in+b*qkK:]
				var s0, s1, s2, s3 float32
				for i := 0; i < qkK; i += 4 {
					s0 += buf[i] * xs[i]
					s1 += buf[i+1] * xs[i+1]
					s2 += buf[i+2] * xs[i+2]
					s3 += buf[i+3] * xs[i+3]
				}
				acc[paired] += (s0 + s1) + (s2 + s3)
			}
		}
		for t := 0; t < P; t++ {
			Y[t*qt.out+o] = acc[t]
		}
	}
	return true
}
