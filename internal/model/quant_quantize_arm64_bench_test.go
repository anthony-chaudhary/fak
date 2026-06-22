//go:build arm64

package model

import (
	"fmt"
	"testing"
)

// quant_quantize_arm64_bench_test.go — the scalar-vs-NEON activation-quant micro-bench for
// issue #476. The NEON activation quantizer (quantizeRowAsmNEON, quant_quantize_arm64.s) is
// dispatched behind FEAT_DotProd by quantizeRowQ8 (quant_quantize_arm64.go) and feeds both the
// decode quantizeVecQ8 and the prefill quantizeBatchPanelInto; TestQuantizeRowAsmMatchesScalar
// already pins it BIT-IDENTICAL to the scalar reference. What was missing was a measurement of
// the speedup the NEON path buys — #476 asks the implementing change to "benchmark quantizeVecQ8
// scalar vs NEON to size the actual decode-step contribution." This bench supplies exactly that.
//
// Why a direct-call A/B (not two FAK_QKERNEL runs): neonDot is resolved ONCE at package init
// (var neonDot = resolveNeonDot()), so toggling FAK_QKERNEL within a single process cannot flip
// quantizeRowQ8's dispatch. Calling quantizeRowQ8scalar and quantizeRowAsmNEON directly times
// both kernels in ONE process, so the scalar/neon ratio is read straight off the two ns/op lines
// — which is what tools/run_476_acceptance_on_arm64.sh parses into the reported delta.
//
// The neon sub-benchmark skips on an arm64 part without FEAT_DotProd (asimddp), where the NEON
// kernel is never dispatched in production; the scalar sub-benchmark runs everywhere on arm64.
// Widths 576 and 1536 are the real Qwen2.5-1.5B decode reduction dims (q/k/v/o/gate/up over
// hidden 576-class inner dims, and down_proj's 1536), the per-token activation rows the hot
// decode path quantizes.
//
//	go test ./internal/model/ -run '^$' -bench '^BenchmarkQuantizeRowScalarVsNEON$' -benchmem
func BenchmarkQuantizeRowScalarVsNEON(b *testing.B) {
	for _, width := range []int{576, 1536} {
		x := mkVec(width, uint64(width*131+17))
		nblk := width / qBlk
		q := make([]int8, width)
		d := make([]float32, nblk)

		b.Run(fmt.Sprintf("scalar/width%d", width), func(b *testing.B) {
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				quantizeRowQ8scalar(x, q, d, nblk)
			}
			b.StopTimer()
			b.ReportMetric(float64(width)*float64(b.N)/b.Elapsed().Seconds(), "float/s")
		})

		b.Run(fmt.Sprintf("neon/width%d", width), func(b *testing.B) {
			if !detectDotProd() {
				b.Skip("FEAT_DotProd (asimddp) not available — NEON quantizer inactive, scalar path only")
			}
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				quantizeRowAsmNEON(&x[0], &q[0], &d[0], nblk)
			}
			b.StopTimer()
			b.ReportMetric(float64(width)*float64(b.N)/b.Elapsed().Seconds(), "float/s")
		})

		// vecneon: the #476-authored decode kernel (quantizeVecAsmNEON), pinned bit-equal to both
		// scalar and the production row kernel by TestQuantizeVecQ8NEONMatchesScalar. Timing it
		// alongside the other two shows the authored vec kernel carries no speed regression.
		b.Run(fmt.Sprintf("vecneon/width%d", width), func(b *testing.B) {
			if !detectDotProd() {
				b.Skip("FEAT_DotProd (asimddp) not available — NEON quantizer inactive, scalar path only")
			}
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				quantizeVecAsmNEON(&x[0], &q[0], &d[0], nblk)
			}
			b.StopTimer()
			b.ReportMetric(float64(width)*float64(b.N)/b.Elapsed().Seconds(), "float/s")
		})
	}
}
