//go:build darwin && cgo && fakmetal

package model

// metal_q8_test.go — the correctness gate for the Metal Q8_0 dequant-GEMV (internal/metalgemm/q8.m).
// The GPU kernel dequant-dots the SAME int8 codes + block scales the CPU qMatRows reads, so the two
// must agree up to the simd_sum vs sequential reduction order. This is the missing primitive for the
// GPU-resident GDN decode forward (#67): the linear_attn.* projections are Q8 (reordered for qwen35,
// kept out of raw-q4_k residency — see fillQ4KMajority), so moving the GDN token mixer onto the
// device needs a Q8 GEMV the q4_k kernels can't provide.

import (
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
