//go:build arm64

package model

import (
	"fmt"
	"math"
	"testing"
)

// TestQdot8Unroll4Sane checks the 4-accumulator probe kernel is numerically close to the scalar
// reference (it is NOT bit-identical — different reduction order — so only a tolerance check).
func TestQdot8Unroll4Sane(t *testing.T) {
	if !detectDotProd() {
		t.Skip("no FEAT_DotProd")
	}
	for _, in := range []int{32, 64, 576, 1536, 8960} {
		w := mkVec(in, uint64(in*131+7))
		x := mkVec(in, uint64(in*977+3))
		qt := quantizeQ8(w, 1, in)
		qv := quantizeVecQ8(x)
		got := qdot8unroll4NEON(&qt.q[0], &qv.q[0], &qt.d[0], &qv.d[0], qt.nblk)
		want := qdot8scalar(qt.q, qt.d, qv, qt.nblk)
		if math.Abs(float64(got-want)) > 1e-2*(1+math.Abs(float64(want))) {
			t.Fatalf("in=%d: unroll=%v scalar=%v (too far)", in, got, want)
		}
	}
}

// BenchmarkDotKernelSingleCore measures SINGLE-CORE Q8 dot throughput (no parFor) for the current
// per-cell qdot8asm vs the 4-accumulator/4-temp probe, at decode GEMV shapes. Isolates whether the
// kernel is latency-bound (probe wins) or capped (probe ~= asm).
func BenchmarkDotKernelSingleCore(b *testing.B) {
	shapes := []struct{ in, out int }{{1536, 1536}, {8960, 1536}, {1536, 8960}}
	for _, s := range shapes {
		w := mkVec(s.out*s.in, uint64(s.out*s.in*131+7))
		qt := quantizeQ8(w, s.out, s.in)
		x := mkVec(s.in, uint64(s.in*977+3))
		qv := quantizeVecQ8(x)
		nblk := qt.nblk
		run := func(b *testing.B, unroll bool) {
			var sink float32
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				for o := 0; o < s.out; o++ {
					if unroll {
						sink += qdot8unroll4NEON(&qt.q[o*s.in], &qv.q[0], &qt.d[o*nblk], &qv.d[0], nblk)
					} else {
						sink += qdot8asm(&qt.q[o*s.in], &qv.q[0], &qt.d[o*nblk], &qv.d[0], nblk)
					}
				}
			}
			b.StopTimer()
			if sink == 0 {
				b.Fatal("zero")
			}
			b.ReportMetric(float64(s.out)*float64(s.in)/(float64(b.Elapsed().Nanoseconds())/float64(b.N)), "MAC/ns")
		}
		b.Run(fmt.Sprintf("%dx%d/asm", s.out, s.in), func(b *testing.B) { run(b, false) })
		b.Run(fmt.Sprintf("%dx%d/unroll4", s.out, s.in), func(b *testing.B) { run(b, true) })
	}
}

// TestQGemm8Row4Sane checks the load-reusing row4 kernel matches qgemm8cell(...,4) bit-for-bit.
func TestQGemm8Row4Sane(t *testing.T) {
	if !detectDotProd() {
		t.Skip("no FEAT_DotProd")
	}
	for _, in := range []int{32, 64, 576, 1536, 8960} {
		w := mkVec(in, uint64(in*131+7))
		qt := quantizeQ8(w, 1, in)
		X := mkVec(4*in, uint64(in*977+3))
		qp := quantizeBatchPanel(X, 4, in)
		var y [4]float32
		qgemm8row4NEON(&qt.q[0], &qp.q[0], &qt.d[0], &qp.d[0], in, qt.nblk, 1, &y[0])
		for j := 0; j < 4; j++ {
			want := qgemm8cell(qt.q, qt.d, qp.q[j*in:j*in+in], qp.d[j*qt.nblk:j*qt.nblk+qt.nblk], qt.nblk, 4)
			if math.Float32bits(y[j]) != math.Float32bits(want) {
				t.Fatalf("in=%d j=%d: row4 %08x != cell %08x", in, j, math.Float32bits(y[j]), math.Float32bits(want))
			}
		}
	}
}

// BenchmarkGemmKernelSingleCore A/Bs prefill GEMM kernels single-core: per-cell qdot8unroll4NEON
// (reloads weights per token) vs qgemm8row4NEON (weight block reused across 4 tokens).
func BenchmarkGemmKernelSingleCore(b *testing.B) {
	for _, s := range []struct{ out, in, P int }{{512, 1536, 64}, {512, 8960, 64}} {
		w := mkVec(s.out*s.in, uint64(s.out*s.in*131+7))
		qt := quantizeQ8(w, s.out, s.in)
		X := mkVec(s.P*s.in, uint64(s.P*s.in*977+3))
		qp := quantizeBatchPanel(X, s.P, s.in)
		Y := make([]float32, s.P*s.out)
		nblk := qt.nblk
		run := func(b *testing.B, row4 bool) {
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				for o := 0; o < s.out; o++ {
					if row4 {
						for t := 0; t+4 <= s.P; t += 4 {
							qgemm8row4NEON(&qt.q[o*s.in], &qp.q[t*s.in], &qt.d[o*nblk], &qp.d[t*nblk], s.in, nblk, s.out, &Y[t*s.out+o])
						}
					} else {
						for t := 0; t < s.P; t++ {
							Y[t*s.out+o] = qdot8unroll4NEON(&qt.q[o*s.in], &qp.q[t*s.in], &qt.d[o*nblk], &qp.d[t*nblk], nblk)
						}
					}
				}
			}
			b.StopTimer()
			b.ReportMetric(float64(s.out)*float64(s.in)*float64(s.P)/(float64(b.Elapsed().Nanoseconds())/float64(b.N)), "MAC/ns")
		}
		b.Run(fmt.Sprintf("%dx%d_P%d/percell", s.out, s.in, s.P), func(b *testing.B) { run(b, false) })
		b.Run(fmt.Sprintf("%dx%d_P%d/row4", s.out, s.in, s.P), func(b *testing.B) { run(b, true) })
	}
}

// TestQGemm8Tile2x4Sane checks the 2×4 tile matches qgemm8cell(...,4) bit-for-bit for all 8 cells.
func TestQGemm8Tile2x4Sane(t *testing.T) {
	if !detectDotProd() {
		t.Skip("no FEAT_DotProd")
	}
	for _, in := range []int{32, 64, 576, 1536, 8960} {
		w := mkVec(2*in, uint64(in*131+7))
		qt := quantizeQ8(w, 2, in)
		X := mkVec(4*in, uint64(in*977+3))
		qp := quantizeBatchPanel(X, 4, in)
		const out = 2
		var Y [8]float32 // [P=4, out=2]
		qgemm8tile2x4NEON(&qt.q[0], &qp.q[0], &qt.d[0], &qp.d[0], in, qt.nblk, out, &Y[0])
		for i := 0; i < 2; i++ {
			for j := 0; j < 4; j++ {
				want := qgemm8cell(qt.q[i*in:i*in+in], qt.d[i*qt.nblk:i*qt.nblk+qt.nblk],
					qp.q[j*in:j*in+in], qp.d[j*qt.nblk:j*qt.nblk+qt.nblk], qt.nblk, 4)
				if math.Float32bits(Y[j*out+i]) != math.Float32bits(want) {
					t.Fatalf("in=%d (%d,%d): %08x != %08x", in, i, j, math.Float32bits(Y[j*out+i]), math.Float32bits(want))
				}
			}
		}
	}
}

// BenchmarkGemmTile2x4SingleCore A/Bs row4 (1×4) vs tile2x4 (2×4) single-core at in=1536.
func BenchmarkGemmTile2x4SingleCore(b *testing.B) {
	const out, in, P = 512, 1536, 64
	w := mkVec(out*in, uint64(out*in*131+7))
	qt := quantizeQ8(w, out, in)
	X := mkVec(P*in, uint64(P*in*977+3))
	qp := quantizeBatchPanel(X, P, in)
	Y := make([]float32, P*out)
	nblk := qt.nblk
	b.Run("row4", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			for o := 0; o < out; o++ {
				for t := 0; t+4 <= P; t += 4 {
					qgemm8row4NEON(&qt.q[o*in], &qp.q[t*in], &qt.d[o*nblk], &qp.d[t*nblk], in, nblk, out, &Y[t*out+o])
				}
			}
		}
		b.ReportMetric(float64(out)*float64(in)*float64(P)/(float64(b.Elapsed().Nanoseconds())/float64(b.N)), "MAC/ns")
	})
	b.Run("tile2x4", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			for o := 0; o+2 <= out; o += 2 {
				for t := 0; t+4 <= P; t += 4 {
					qgemm8tile2x4NEON(&qt.q[o*in], &qp.q[t*in], &qt.d[o*nblk], &qp.d[t*nblk], in, nblk, out, &Y[t*out+o])
				}
			}
		}
		b.ReportMetric(float64(out)*float64(in)*float64(P)/(float64(b.Elapsed().Nanoseconds())/float64(b.N)), "MAC/ns")
	})
}
