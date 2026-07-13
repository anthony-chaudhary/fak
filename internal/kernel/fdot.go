package kernel

// FDot is the deterministic 8-accumulator float32 inner product shared by the
// model, the portable compute reference, and kernel microbenchmarks. The fixed
// combine order is load-bearing: fak-vs-fak parity witnesses require bit-identical
// reductions across every caller.
func FDot(r, x []float32) float32 {
	var s0, s1, s2, s3, s4, s5, s6, s7 float32
	n := len(r)
	i := 0
	for ; i+8 <= n; i += 8 {
		s0 += r[i] * x[i]
		s1 += r[i+1] * x[i+1]
		s2 += r[i+2] * x[i+2]
		s3 += r[i+3] * x[i+3]
		s4 += r[i+4] * x[i+4]
		s5 += r[i+5] * x[i+5]
		s6 += r[i+6] * x[i+6]
		s7 += r[i+7] * x[i+7]
	}
	s := ((s0 + s1) + (s2 + s3)) + ((s4 + s5) + (s6 + s7))
	for ; i < n; i++ {
		s += r[i] * x[i]
	}
	return s
}
