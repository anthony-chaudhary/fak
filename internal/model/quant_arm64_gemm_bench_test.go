//go:build arm64

package model

import (
	"fmt"
	"testing"
)

// BenchmarkDecodeGEMV A/Bs the decode GEMV at real Qwen2.5-1.5B projection shapes: the new NEON
// deferred-reduction path (qMatRowsInto -> qMatRowsRangeFast -> qmatrows4NEON) vs the old per-row
// qdot8GEMV (qdot8asm, per-block VADDV reduction). Both are row-parallel across the same workers,
// so the ratio isolates the kernel change. Reports MAC/ns.
func benchDecodeGEMV(b *testing.B, out, in int, neon bool) {
	w := mkVec(out*in, uint64(out*in*131+7))
	qt := quantizeQ8(w, out, in)
	x := mkVec(in, uint64(in*977+3))
	qv := quantizeVecQ8(x)
	y := make([]float32, out)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if neon {
			qMatRowsInto(qt, qv, y)
		} else {
			parFor(out, numWorkers, func(lo, hi int) {
				for o := lo; o < hi; o++ {
					y[o] = qdot8GEMV(qt.q[o*in:o*in+in], qt.d[o*qt.nblk:o*qt.nblk+qt.nblk], qv, qt.nblk)
				}
			})
		}
	}
	b.StopTimer()
	macs := float64(out) * float64(in)
	b.ReportMetric(macs/(float64(b.Elapsed().Nanoseconds())/float64(b.N)), "MAC/ns")
}

func BenchmarkDecodeGEMV(b *testing.B) {
	shapes := []struct {
		name    string
		out, in int
	}{
		{"qproj_1536x1536", 1536, 1536},
		{"gateup_8960x1536", 8960, 1536},
		{"down_1536x8960", 1536, 8960},
		{"lmhead_151936x1536", 151936, 1536},
	}
	for _, s := range shapes {
		b.Run(fmt.Sprintf("%s/neon", s.name), func(b *testing.B) { benchDecodeGEMV(b, s.out, s.in, true) })
		b.Run(fmt.Sprintf("%s/old", s.name), func(b *testing.B) { benchDecodeGEMV(b, s.out, s.in, false) })
	}
}
