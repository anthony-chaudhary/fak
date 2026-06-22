//go:build arm64

package model

import (
	"fmt"
	"testing"
)

// BenchmarkPrefillTileVsCell478 is the per-cell-vs-tile prefill A/B for issue #478 (the arm64
// NEON register-blocked Q8_0 GEMM tile). It measures the SHIPPED default arm64 prefill path —
// qGemm8Into, the 2×4 load-reusing sweep + qgemm8cell remainders — against the 4×4
// register-blocked tile qGemm8TileInto (which routes the full MR×NR tiles through qgemm8tileNEON,
// this issue's kernel, with the out%4 / P%4 remainders via qgemm8cell). Both are bit-identical to
// the lane-4 scalar reference (TestQGemm8IntoMatchesScalarNEON), so this isolates the kernel-shape
// delta alone: does the 4×4 tile's wider weight-block reuse (each 32-wide weight block streamed
// once, SDOT'd against all 4 tokens, float reduce deferred to end-of-K) beat the 2×4 default.
//
// Shapes are the Qwen2.5-1.5B Q8_0 projection rectangles (hidden 1536, intermediate 8960) at a
// representative prefill token block P=64 — the same family M3-LLAMACPP-RESULTS.md sec. 3/4 uses
// for the 55.5 tok/s prefill measure this tile targets. Reports aggregate MAC/ns (higher = faster);
// run single-core (-cpu 1) for the raw kernel delta or default for the parFor aggregate. The
// arm64 RUN is the residual: on Apple Silicon (M3 Pro) the shipped default has measured FASTER
// (hence the tile is opt-in via FAK_ARM_TILE=1, see qGemm8Into), so a faithful read here is
// "default ≥ tile on M3"; on non-Apple arm64 the wider tile's amortization may pay off.
func BenchmarkPrefillTileVsCell478(b *testing.B) {
	if !detectDotProd() {
		b.Skip("FEAT_DotProd (asimddp) not available — NEON prefill GEMM inactive")
	}
	shapes := []struct {
		name    string
		out, in int
	}{
		{"qkv_1536x1536", 1536, 1536},
		{"gateup_8960x1536", 8960, 1536},
		{"down_1536x8960", 1536, 8960},
	}
	const P = 64 // prefill token block
	for _, s := range shapes {
		w := mkVec(s.out*s.in, uint64(s.out*s.in*131+7))
		qt := quantizeQ8(w, s.out, s.in)
		X := mkVec(P*s.in, uint64(P*s.in*977+3))
		qp := quantizeBatchPanel(X, P, s.in)
		Y := make([]float32, P*s.out)
		macs := float64(s.out) * float64(s.in) * float64(P)
		run := func(b *testing.B, fn func(*q8Tensor, *q8Panel, []float32)) {
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				fn(qt, qp, Y)
			}
			b.StopTimer()
			b.ReportMetric(macs/(float64(b.Elapsed().Nanoseconds())/float64(b.N)), "MAC/ns")
		}
		// "default" = the shipped per-cell/2×4 sweep (qGemm8Into); "tile4x4" = #478's
		// register-blocked qgemm8tileNEON dispatcher (qGemm8TileInto). Called directly so the
		// A/B is independent of FAK_ARM_TILE.
		b.Run(fmt.Sprintf("%s_P%d/default", s.name, P), func(b *testing.B) { run(b, qGemm8Into) })
		b.Run(fmt.Sprintf("%s_P%d/tile4x4", s.name, P), func(b *testing.B) { run(b, qGemm8TileInto) })
	}
}
