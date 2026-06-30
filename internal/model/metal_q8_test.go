//go:build darwin && cgo && fakmetal

package model

// metal_q8_test.go — correctness + perf gates for the Metal Q8_0 dequant-GEMV/GEMM
// (internal/metalgemm/q8.m). The GPU kernels dequant-dot the SAME int8 codes + block scales the CPU
// Q8 paths read, so the two must agree up to GPU vs CPU reduction-order differences. GEMV is the
// missing primitive for the GPU-resident GDN decode forward (#67); GEMM is the missing primitive for
// Qwen3.6's Q8-minority prefill projections (#1087), which are reordered for qwen35 and kept out of
// raw-q4_k residency.

import (
	"fmt"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/metalgemm"
)

// TestMetalQ8GemvMatchesCPU holds the GPU Q8 GEMV to the CPU qMatRows on the SAME quantized
// activation: the weight (random f32 → quantizeQ8) and the activation (quantizeVecQ8) feed both
// paths, so the only divergence is the GPU's per-row simd_sum order vs the CPU's sequential
// per-block float accumulate — cosine ~1.0. A wiring bug (wrong stride, unsigned code read, scale
// mis-index) diverges O(1).
func TestMetalQ8GemvMatchesCPU(t *testing.T) {
	if !metalgemm.Available() {
		t.Skip("no Metal device available")
	}
	defer metalgemm.ResetQ8()
	cases := []struct{ out, in int }{
		{256, 256}, {512, 1024}, {5120, 5120}, // hidden-size GEMV
	}
	for _, c := range cases {
		wf := randomVecF(c.out*c.in, 42)
		qt := quantizeQ8(wf, c.out, c.in)
		x := randomVecF(c.in, 7)
		qv := quantizeVecQ8(x)
		ref := qMatRows(qt, qv) // CPU Q8 GEMV (int8×int8 → int32, float-scaled)

		w := metalgemm.UploadQ8(qt.q, qt.d, c.out, c.in)
		if w == nil {
			t.Fatalf("UploadQ8(%d,%d) returned nil", c.out, c.in)
		}
		got := make([]float32, c.out)
		w.GEMV(qv.q, qv.d, got)

		cos, maxRel := cosineAndMaxRel(ref, got)
		if cos < 0.9999 || maxRel > 5e-3 {
			t.Errorf("q8 GEMV [%d,%d]: cosine=%.6f maxRel=%.4g (want cos>=0.9999, maxRel<=5e-3)\n  ref[:4]=%v\n  got[:4]=%v",
				c.out, c.in, cos, maxRel, ref[:4], got[:4])
		} else {
			t.Logf("q8 GEMV [%d,%d]: cosine=%.6f maxRel=%.4g OK", c.out, c.in, cos, maxRel)
		}
		metalgemm.ResetQ8()
	}
}

// TestMetalQ8GemvGroupMatchesSingle verifies GEMVGroupQ8 (n weights sharing one activation, one
// command buffer) returns the same result as a single GEMV per weight — the gate for batching the
// GDN in_proj quad into one command buffer. Different out dims exercise the per-weight y-offset
// packing. Same kernel + inputs, so the results are bit-identical.
func TestMetalQ8GemvGroupMatchesSingle(t *testing.T) {
	if !metalgemm.Available() {
		t.Skip("no Metal device available")
	}
	defer metalgemm.ResetQ8()
	in := 1024
	outs := []int{512, 1024, 256, 1536} // different out dims, shared in
	x := randomVecF(in, 9)
	qv := quantizeVecQ8(x)
	ws := make([]*metalgemm.Q8Weight, len(outs))
	singles := make([][]float32, len(outs))
	for i, out := range outs {
		qt := quantizeQ8(randomVecF(out*in, int64(200+i)), out, in)
		w := metalgemm.UploadQ8(qt.q, qt.d, out, in)
		if w == nil {
			t.Fatalf("UploadQ8(%d,%d) returned nil", out, in)
		}
		ws[i] = w
		y := make([]float32, out)
		w.GEMV(qv.q, qv.d, y)
		singles[i] = y
	}
	group := metalgemm.GEMVGroupQ8(ws, qv.q, qv.d)
	if len(group) != len(ws) {
		t.Fatalf("GEMVGroupQ8 returned %d results, want %d", len(group), len(ws))
	}
	for i := range ws {
		if len(group[i]) != outs[i] {
			t.Fatalf("group[%d] len=%d want %d", i, len(group[i]), outs[i])
		}
		for o := 0; o < outs[i]; o++ {
			if d := group[i][o] - singles[i][o]; d > 1e-3 || d < -1e-3 {
				t.Fatalf("group[%d][%d]=%g != single %g", i, o, group[i][o], singles[i][o])
			}
		}
	}
	t.Logf("GEMVGroupQ8 matches single GEMV across %d weights (outs=%v)", len(ws), outs)
}

// TestMetalQ8GemmMatchesCPU holds the batched Metal Q8 prefill GEMM to the CPU Q8 prefill GEMM
// on the same quantized activation panel. This is issue #1087's missing primitive: the Q8-minority
// projections in the resident-Q4_K prefill path can run on the GPU instead of falling back to CPU.
func TestMetalQ8GemmMatchesCPU(t *testing.T) {
	if !metalgemm.Available() {
		t.Skip("no Metal device available")
	}
	defer metalgemm.ResetQ8()
	cases := []struct{ out, in, P int }{
		{256, 256, 5},
		{512, 1024, 22},
		{5120, 5120, 16},
	}
	for _, c := range cases {
		qt := quantizeQ8(randomVecF(c.out*c.in, 31), c.out, c.in)
		qp := quantizeBatchPanel(randomVecF(c.P*c.in, 37), c.P, c.in)
		ref := qGemm8(qt, qp)

		w := metalgemm.UploadQ8(qt.q, qt.d, c.out, c.in)
		if w == nil {
			t.Fatalf("UploadQ8(%d,%d) returned nil", c.out, c.in)
		}
		got := make([]float32, c.P*c.out)
		w.GEMM(qp.q, qp.d, c.P, got)

		cos, maxRel := cosineAndMaxRel(ref, got)
		if cos < 0.999 || maxRel > 2e-2 {
			t.Errorf("q8 GEMM [%d,%d]x%d: cosine=%.6f maxRel=%.4g (want cos>=0.999, maxRel<=2e-2)\n  ref[:4]=%v\n  got[:4]=%v",
				c.out, c.in, c.P, cos, maxRel, ref[:4], got[:4])
		} else {
			t.Logf("q8 GEMM [%d,%d]x%d: cosine=%.6f maxRel=%.4g OK", c.out, c.in, c.P, cos, maxRel)
		}
		metalgemm.ResetQ8()
	}
}

// TestMetalQ8GemmDispatchMatchesCPU proves the model-level dispatcher used by the resident-Q4_K
// prefill path uploads/caches Q8-minority weights and returns the same panel shape as CPU qGemm8.
func TestMetalQ8GemmDispatchMatchesCPU(t *testing.T) {
	if !metalgemm.Available() {
		t.Skip("no Metal device available")
	}
	defer metalgemm.ResetQ8()
	cfg := qwen35HybridQ4KTestCfg()
	m := NewSynthetic(cfg)
	m.Quantize()
	s := m.NewSession()
	s.MetalQ4K = true
	name := layerName(0, "linear_attn.in_proj_b.weight")
	qt := m.q8(name)
	P := 9
	qp := quantizeBatchPanel(randomVecF(P*qt.in, 43), P, qt.in)
	ref := qGemm8(qt, qp)
	got := s.q8GemmDispatch(name, qt, qp)
	if len(got) != len(ref) {
		t.Fatalf("dispatch len=%d want %d", len(got), len(ref))
	}
	cos, maxRel := cosineAndMaxRel(ref, got)
	if cos < 0.999 || maxRel > 2e-2 {
		t.Fatalf("dispatch q8 GEMM: cosine=%.6f maxRel=%.4g (want cos>=0.999, maxRel<=2e-2)", cos, maxRel)
	}
}

// BenchmarkMetalQ8Gemv reports the GPU Q8 GEMV throughput at hidden size, to compare against the
// CPU qMatRows (~23 GB/s int8 SDOT on the M3 Pro) and the q4_k GPU GEMV. A lone GEMV is launch-bound
// (one commit→wait per dispatch), so the win for the GDN projections is the grouped/resident path,
// not this isolated call — but the steady kernel rate here is the headroom the resident forward
// captures.
func BenchmarkMetalQ8Gemv(b *testing.B) {
	if !metalgemm.Available() {
		b.Skip("no Metal device available")
	}
	defer metalgemm.ResetQ8()
	out, in := 5120, 5120
	qt := quantizeQ8(randomVecF(out*in, 1), out, in)
	qv := quantizeVecQ8(randomVecF(in, 2))
	w := metalgemm.UploadQ8(qt.q, qt.d, out, in)
	if w == nil {
		b.Fatal("UploadQ8 returned nil")
	}
	y := make([]float32, out)
	weightBytes := float64(out)*float64(in) + float64(out)*float64(in)/32.0*4.0 // codes + scales
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		w.GEMV(qv.q, qv.d, y)
	}
	b.StopTimer()
	if secs := b.Elapsed().Seconds(); secs > 0 {
		b.ReportMetric(weightBytes*float64(b.N)/secs/1e9, "GB/s")
	}
}

// BenchmarkMetalQ8GemmPSweep is the #1087 perf witness for the Q8-minority prefill path:
// full-attn q/k and Qwen3.6 linear_attn.* now have a batched Metal GEMM instead of CPU qGemm8.
// It uses the hidden-size square shape common to q/k-style projections and sweeps prompt length
// so the Mac run can distinguish tiny-prompt launch cost from agentic-length prefill throughput.
func BenchmarkMetalQ8GemmPSweep(b *testing.B) {
	if !metalgemm.Available() {
		b.Skip("no Metal device available")
	}
	defer metalgemm.ResetQ8()
	out, in := 5120, 5120
	qt := quantizeQ8(randomVecF(out*in, 51), out, in)
	w := metalgemm.UploadQ8(qt.q, qt.d, out, in)
	if w == nil {
		b.Fatal("UploadQ8 returned nil")
	}
	weightBytes := float64(out)*float64(in) + float64(out)*float64(in)/32.0*4.0
	for _, P := range []int{22, 64, 256, 512, 1024} {
		qp := quantizeBatchPanel(randomVecF(P*in, int64(700+P)), P, in)
		Y := make([]float32, P*out)
		b.Run(fmt.Sprintf("P=%d", P), func(b *testing.B) {
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				w.GEMM(qp.q, qp.d, P, Y)
			}
			b.StopTimer()
			if s := b.Elapsed().Seconds(); s > 0 {
				b.ReportMetric(weightBytes*float64(b.N)/s/1e9, "GB/s")
				b.ReportMetric(2*float64(out)*float64(in)*float64(P)*float64(b.N)/s/1e9, "GFLOP/s")
				b.ReportMetric(s/float64(b.N)*1e3, "ms/op")
			}
		})
	}
}
