//go:build darwin && cgo && fakmetal

package model

// metal_q4k_test.go — the correctness gate for the Metal q4_k dequant-GEMV/GEMM
// (internal/metalgemm/q4k.m). The GPU kernel reconstructs each weight row's f32 values from
// the SAME resident q4_k super-block bytes the CPU q4kMatRowsRange reference reads, so the two
// must agree up to GPU float-accumulation order. We hold the GPU to the CPU *f32* path
// (q4kMatRowsRange, not the int8-SDOT decode kernel) because the GPU kernel also dequants to
// f32 and dots in float — same arithmetic, only the reduction order differs.
//
// This is the keystone for throughput parity: the CPU int8 path is compute-bound (~23 GB/s,
// ~1.4 tok/s decode ceiling on the M3 Pro) and cannot reach the llama.cpp-Metal bar (7.29
// decode / 51.55 prefill tok/s). A correct q4_k GPU GEMM is the only resident route that fits
// 27B on 36 GB (q4_k_m ≈ 16 GB) AND has the bandwidth + parallel dequant to hit the bar.

import (
	"math"
	"math/rand"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/metalgemm"
)

// randomQ4KTensor builds an [out,in] resident q4_k tensor from deterministic pseudo-random
// super-block bytes. Any byte pattern is a valid q4_k block (the dequant is total), so the CPU
// reference and the GPU kernel interpret identical bytes — the comparison is pure kernel math,
// not a quantizer round-trip.
func randomQ4KTensor(out, in int, seed int64) *q4kTensor {
	if in%qkK != 0 {
		panic("randomQ4KTensor: in not a multiple of 256")
	}
	nblk := in / qkK
	raw := make([]byte, out*nblk*q4kBlockBytes)
	rng := rand.New(rand.NewSource(seed))
	for i := range raw {
		raw[i] = byte(rng.Intn(256))
	}
	// Keep the f16 super-block scales (d, dmin) in a sane finite range so the dot doesn't
	// overflow to Inf/NaN: a uniformly random 16-bit pattern can be a huge/Inf half. Clamp the
	// exponent of the two halves at the head of every 144-B block to a small magnitude.
	for o := 0; o < out; o++ {
		for b := 0; b < nblk; b++ {
			base := (o*nblk + b) * q4kBlockBytes
			// d and dmin: small positive halves (~[1/64, 1/4)) — exponent bits set modestly.
			raw[base+1] = 0x2C | (raw[base+1] & 0x03) // high byte of half d
			raw[base+3] = 0x2C | (raw[base+3] & 0x03) // high byte of half dmin
		}
	}
	return &q4kTensor{out: out, in: in, nblk: nblk, raw: raw}
}

func randomVecF(n int, seed int64) []float32 {
	rng := rand.New(rand.NewSource(seed))
	x := make([]float32, n)
	for i := range x {
		x[i] = float32(rng.NormFloat64())
	}
	return x
}

// cosineAndMaxRel reports cosine similarity and the max relative error over the larger
// magnitudes (the small-magnitude entries are dominated by float noise and ignored).
func cosineAndMaxRel(a, b []float32) (cos float64, maxRel float64) {
	var dot, na, nb float64
	for i := range a {
		dot += float64(a[i]) * float64(b[i])
		na += float64(a[i]) * float64(a[i])
		nb += float64(b[i]) * float64(b[i])
	}
	if na == 0 || nb == 0 {
		return 0, math.Inf(1)
	}
	cos = dot / (math.Sqrt(na) * math.Sqrt(nb))
	scale := math.Sqrt(na / float64(len(a))) // RMS magnitude
	for i := range a {
		if math.Abs(float64(a[i])) < scale {
			continue
		}
		rel := math.Abs(float64(a[i]-b[i])) / math.Abs(float64(a[i]))
		if rel > maxRel {
			maxRel = rel
		}
	}
	return cos, maxRel
}

func TestMetalQ4KGemvMatchesCPU(t *testing.T) {
	if !metalgemm.Available() {
		t.Skip("no Metal device available")
	}
	defer metalgemm.ResetQ4K()
	cases := []struct{ out, in int }{
		{256, 256}, {512, 1024}, {5120, 5120}, // hidden-size GEMV
	}
	for _, c := range cases {
		qt := randomQ4KTensor(c.out, c.in, 42)
		x := randomVecF(c.in, 7)
		ref := make([]float32, c.out)
		q4kMatRowsRange(qt, x, ref, 0, c.out) // CPU f32 reference

		w := metalgemm.UploadQ4K(qt.raw, c.out, c.in)
		if w == nil {
			t.Fatalf("UploadQ4K(%d,%d) returned nil", c.out, c.in)
		}
		got := make([]float32, c.out)
		w.GEMV(x, got)

		cos, maxRel := cosineAndMaxRel(ref, got)
		if cos < 0.9999 || maxRel > 5e-3 {
			t.Errorf("q4k GEMV [%d,%d]: cosine=%.6f maxRel=%.4g (want cos>=0.9999, maxRel<=5e-3)\n  ref[:4]=%v\n  got[:4]=%v",
				c.out, c.in, cos, maxRel, ref[:4], got[:4])
		} else {
			t.Logf("q4k GEMV [%d,%d]: cosine=%.6f maxRel=%.4g OK", c.out, c.in, cos, maxRel)
		}
		_ = w // ResetQ4K (deferred) frees every uploaded buffer
	}
}

// TestMetalQ4KGemvGroupMatchesSingle verifies that GEMVGroup (n weights sharing one activation,
// one command buffer) returns the same result as a single GEMV per weight — the correctness gate
// for the live decode group batching (q/k/v, gate/up). Weights have different out dims to exercise
// the per-weight y-offset packing.
func TestMetalQ4KGemvGroupMatchesSingle(t *testing.T) {
	if !metalgemm.Available() {
		t.Skip("no Metal device available")
	}
	defer metalgemm.ResetQ4K()
	in := 1024
	outs := []int{512, 1024, 256, 1536} // different out dims, shared in
	x := randomVecF(in, 9)
	ws := make([]*metalgemm.Q4KWeight, len(outs))
	singles := make([][]float32, len(outs))
	for i, out := range outs {
		qt := randomQ4KTensor(out, in, int64(100+i))
		w := metalgemm.UploadQ4K(qt.raw, out, in)
		if w == nil {
			t.Fatalf("UploadQ4K(%d,%d) returned nil", out, in)
		}
		ws[i] = w
		y := make([]float32, out)
		w.GEMV(x, y)
		singles[i] = y
	}
	group := metalgemm.GEMVGroup(ws, x)
	if len(group) != len(ws) {
		t.Fatalf("GEMVGroup returned %d results, want %d", len(group), len(ws))
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
	t.Logf("GEMVGroup matches single GEMV across %d weights (outs=%v)", len(ws), outs)
}

// TestMetalQ4KPrefillMatchesCPU is the end-to-end wiring gate: the resident-Q4_K hybrid
// prefill with MetalQ4K=true (q4_k-majority GEMMs on the GPU) produces the same logits as the
// CPU path (MetalQ4K=false) on the synthetic hybrid model. CPU GEMV is forced to f32
// (setQ4KSDOTForTest(false)) so the comparison is GPU-f32 vs CPU-f32 — the q4_k majority is
// then equivalent up to GPU float-accumulation order; the Q8 minority (q/k + linear_attn.*) is
// identical on both paths. A wiring bug (wrong weight, layout mismatch, the GPU result not
// flowing into the recurrence) diverges O(1) per layer and blows past the bound.
func TestMetalQ4KPrefillMatchesCPU(t *testing.T) {
	if !metalgemm.Available() {
		t.Skip("no Metal device available")
	}
	defer metalgemm.ResetQ4K()
	setQ4KSDOTForTest(false)
	t.Cleanup(func() { setQ4KSDOTForTest(true) })

	cfg := qwen35HybridQ4KTestCfg()
	m := NewSynthetic(cfg)
	m.Quantize()
	fillQ4KMajority(t, m, cfg)
	prompt := []int{3, 7, 11, 5, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61}

	cpu := m.NewSession()
	cpu.Q4K = true
	cpuLogits := cpu.Prefill(prompt)

	gpu := m.NewSession()
	gpu.Q4K = true
	gpu.MetalQ4K = true
	gpuLogits := gpu.Prefill(prompt)

	if len(cpuLogits) != len(gpuLogits) {
		t.Fatalf("logit length mismatch: cpu=%d gpu=%d", len(cpuLogits), len(gpuLogits))
	}
	cos, maxRel := cosineAndMaxRel(cpuLogits, gpuLogits)
	if argmaxF(cpuLogits) != argmaxF(gpuLogits) || cos < 0.999 {
		t.Errorf("metal q4k prefill: cpu argmax=%d gpu argmax=%d cosine=%.6f maxRel=%.4g (want same argmax, cos>=0.999)",
			argmaxF(cpuLogits), argmaxF(gpuLogits), cos, maxRel)
	} else {
		t.Logf("metal q4k prefill: argmax match=%d cosine=%.6f maxRel=%.4g OK", argmaxF(gpuLogits), cos, maxRel)
	}
}

// TestMetalQ4KDecodeMatchesCPU verifies the single-residency GPU decode path: with MetalQ4K the
// decode q4_k GEMVs run on the GPU (q4k_gemv) and the CPU q4_k copy is freed after upload. It
// must produce the same greedy decode tokens as the CPU path. Two separate models are built
// because the GPU run frees its model's raw q4_k bytes (single residency), which would break a
// subsequent CPU run on the same model.
func TestMetalQ4KDecodeMatchesCPU(t *testing.T) {
	if !metalgemm.Available() {
		t.Skip("no Metal device available")
	}
	defer metalgemm.ResetQ4K()
	setQ4KSDOTForTest(false)
	t.Cleanup(func() { setQ4KSDOTForTest(true) })
	cfg := qwen35HybridQ4KTestCfg()
	prompt := []int{3, 7, 11, 5, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61}

	decode := func(metal bool) ([]float32, []int) {
		m := NewSynthetic(cfg)
		m.Quantize()
		fillQ4KMajority(t, m, cfg)
		s := m.NewSession()
		s.Q4K = true
		s.MetalQ4K = metal
		lg := s.Prefill(prompt)
		var seq []int
		for i := 0; i < 4; i++ {
			n := argmaxF(lg)
			seq = append(seq, n)
			lg = s.Step(n)
		}
		return lg, seq
	}

	_, cpuSeq := decode(false)
	gpuLast, gpuSeq := decode(true)
	_ = gpuLast
	for i := range cpuSeq {
		if cpuSeq[i] != gpuSeq[i] {
			t.Fatalf("decode token %d: cpu=%d gpu=%d (cpu seq=%v gpu seq=%v)", i, cpuSeq[i], gpuSeq[i], cpuSeq, gpuSeq)
		}
	}
	t.Logf("metal q4k decode: greedy token sequence matches CPU = %v", gpuSeq)
}

func argmaxF(v []float32) int {
	bi := 0
	best := float32(-3.4e38)
	for i, x := range v {
		if x > best {
			best, bi = x, i
		}
	}
	return bi
}

func TestMetalQ4KGemmMatchesCPU(t *testing.T) {
	if !metalgemm.Available() {
		t.Skip("no Metal device available")
	}
	defer metalgemm.ResetQ4K()
	// Cover: small, the 22-token oracle panel, a non-square MLP-like shape, and P>128 (the
	// tiled kernel's token-tile size) so the multi-tile / single-command-buffer path is exercised.
	cases := []struct{ out, in, P int }{
		{1024, 1024, 6},
		{2048, 1024, 22},
		{1024, 512, 130}, // two token tiles (128 + 2)
	}
	for _, c := range cases {
		qt := randomQ4KTensor(c.out, c.in, 99)
		X := randomVecF(c.P*c.in, 11)
		ref := make([]float32, c.P*c.out)
		for tIdx := 0; tIdx < c.P; tIdx++ {
			row := make([]float32, c.out)
			q4kMatRowsRange(qt, X[tIdx*c.in:(tIdx+1)*c.in], row, 0, c.out)
			copy(ref[tIdx*c.out:(tIdx+1)*c.out], row)
		}
		w := metalgemm.UploadQ4K(qt.raw, c.out, c.in)
		if w == nil {
			t.Fatalf("UploadQ4K(%d,%d) returned nil", c.out, c.in)
		}
		got := make([]float32, c.P*c.out)
		w.GEMM(X, c.P, got)
		cos, maxRel := cosineAndMaxRel(ref, got)
		if cos < 0.9999 || maxRel > 5e-3 {
			t.Errorf("q4k GEMM [%d,%d]x%d: cosine=%.6f maxRel=%.4g (want cos>=0.9999, maxRel<=5e-3)", c.out, c.in, c.P, cos, maxRel)
		} else {
			t.Logf("q4k GEMM [%d,%d]x%d: cosine=%.6f maxRel=%.4g OK", c.out, c.in, c.P, cos, maxRel)
		}
		metalgemm.ResetQ4K()
	}
}

// BenchmarkMetalQ4KGemv reports the GPU q4_k GEMV throughput at hidden size. Compare against
// the CPU BenchmarkQ4KMatRowsInt8 (~23 GB/s at 12 workers): the GPU should clear it and head
// toward the unified-memory bandwidth that the 7.29 tok/s decode bar implies.
func BenchmarkMetalQ4KGemv(b *testing.B) {
	if !metalgemm.Available() {
		b.Skip("no Metal device available")
	}
	defer metalgemm.ResetQ4K()
	out, in := 5120, 5120
	qt := randomQ4KTensor(out, in, 1)
	x := randomVecF(in, 2)
	w := metalgemm.UploadQ4K(qt.raw, out, in)
	if w == nil {
		b.Fatal("UploadQ4K returned nil")
	}
	y := make([]float32, out)
	weightBytes := float64(out) * float64(in) / 256.0 * 144.0
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		w.GEMV(x, y)
	}
	b.StopTimer()
	secs := b.Elapsed().Seconds()
	if secs > 0 {
		b.ReportMetric(weightBytes*float64(b.N)/secs/1e9, "GB/s")
	}
}

// BenchmarkMetalQ4KGemvTiny isolates the per-dispatch (command-buffer commit→wait) overhead:
// a 256x256 GEMV does ~16 KB of work, so its ns/op is dominated by the fixed launch cost. The
// gap between this and BenchmarkMetalQ4KGemv (5120x5120) attributes time to overhead vs work,
// which decides the decode-wiring strategy (how many dispatches/token are affordable).
func BenchmarkMetalQ4KGemvTiny(b *testing.B) {
	if !metalgemm.Available() {
		b.Skip("no Metal device available")
	}
	defer metalgemm.ResetQ4K()
	out, in := 256, 256
	qt := randomQ4KTensor(out, in, 1)
	x := randomVecF(in, 2)
	w := metalgemm.UploadQ4K(qt.raw, out, in)
	if w == nil {
		b.Fatal("UploadQ4K returned nil")
	}
	y := make([]float32, out)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		w.GEMV(x, y)
	}
}

// BenchmarkMetalQ4KGemvBatch runs n decode GEMVs of one 5120x5120 weight in a SINGLE command
// buffer (one commit→wait), to quantify how much of the per-GEMV cost is the CPU↔GPU
// submission/sync round-trip vs the kernel. If per-GEMV here collapses far below the single-GEMV
// BenchmarkMetalQ4KGemv (~457 µs) toward the kernel rate, the decode wall is the per-op command
// buffer and the fix is the one-command-buffer resident forward (issue #67). n=64 ≈ the
// projection/MLP GEMV count in one decoder layer scaled up.
func BenchmarkMetalQ4KGemvBatch(b *testing.B) {
	if !metalgemm.Available() {
		b.Skip("no Metal device available")
	}
	defer metalgemm.ResetQ4K()
	out, in, n := 5120, 5120, 64
	qt := randomQ4KTensor(out, in, 1)
	Xcat := randomVecF(n*in, 2)
	w := metalgemm.UploadQ4K(qt.raw, out, in)
	if w == nil {
		b.Fatal("UploadQ4K returned nil")
	}
	Ycat := make([]float32, n*out)
	// Trust check: batched row 0 must equal a single GEMV of the same activation row.
	w.GEMVBatch(Xcat, n, Ycat)
	single := make([]float32, out)
	w.GEMV(Xcat[:in], single)
	for o := 0; o < out; o++ {
		if d := Ycat[o] - single[o]; d > 1e-3 || d < -1e-3 {
			b.Fatalf("GEMVBatch row0[%d]=%g != GEMV %g (offset binding wrong)", o, Ycat[o], single[o])
		}
	}
	weightBytes := float64(out) * float64(in) / 256.0 * 144.0
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		w.GEMVBatch(Xcat, n, Ycat)
	}
	b.StopTimer()
	secs := b.Elapsed().Seconds()
	if secs > 0 {
		b.ReportMetric(weightBytes*float64(n)*float64(b.N)/secs/1e9, "GB/s")
		b.ReportMetric(secs/float64(b.N)/float64(n)*1e6, "us/gemv")
	}
}

// BenchmarkMetalQ4KFusedMLP isolates the fused on-GPU MLP (gate→silu·up→down in ONE command
// buffer, intermediate resident) against the same three matmuls run as separate command buffers
// with a CPU SwiGLU between — the decode MLP is ~54% of q4_k_m decode, so this is the noise-free
// measure of the per-MLP-call lever the end-to-end wall-clock is too contended to show cleanly.
// Shapes are Qwen3.6-27B's: H=5120, I=17408.
func BenchmarkMetalQ4KFusedMLP(b *testing.B) {
	if !metalgemm.Available() {
		b.Skip("no Metal device available")
	}
	defer metalgemm.ResetQ4K()
	H, I := 5120, 17408
	gate := metalgemm.UploadQ4K(randomQ4KTensor(I, H, 1).raw, I, H)
	up := metalgemm.UploadQ4K(randomQ4KTensor(I, H, 2).raw, I, H)
	down := metalgemm.UploadQ4K(randomQ4KTensor(H, I, 3).raw, H, I)
	if gate == nil || up == nil || down == nil {
		b.Fatal("UploadQ4K returned nil")
	}
	x := randomVecF(H, 4)
	y := make([]float32, H)
	rowBytes := func(out, in int) float64 { return float64(out) * float64(in) / 256.0 * 144.0 }
	mlpBytes := rowBytes(I, H) + rowBytes(I, H) + rowBytes(H, I) // gate + up + down weight bytes
	// Trust check: fused output equals the separate path on decisive margin.
	g0, u0, inter0 := make([]float32, I), make([]float32, I), make([]float32, I)
	gate.GEMV(x, g0)
	up.GEMV(x, u0)
	for j := 0; j < I; j++ {
		inter0[j] = silu(g0[j]) * u0[j]
	}
	ySep := make([]float32, H)
	down.GEMV(inter0, ySep)
	metalgemm.FusedMLP(gate, up, down, x, y)
	for o := 0; o < H; o++ {
		if d := y[o] - ySep[o]; d > 1e-2 || d < -1e-2 {
			b.Fatalf("FusedMLP[%d]=%g != separate %g", o, y[o], ySep[o])
		}
	}
	b.Run("fused", func(b *testing.B) {
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			metalgemm.FusedMLP(gate, up, down, x, y)
		}
		b.StopTimer()
		if s := b.Elapsed().Seconds(); s > 0 {
			b.ReportMetric(mlpBytes*float64(b.N)/s/1e9, "GB/s")
			b.ReportMetric(s/float64(b.N)*1e3, "ms/mlp")
		}
	})
	b.Run("separate", func(b *testing.B) {
		g, u, inter := make([]float32, I), make([]float32, I), make([]float32, I)
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			gate.GEMV(x, g)
			up.GEMV(x, u)
			for j := 0; j < I; j++ {
				inter[j] = silu(g[j]) * u[j]
			}
			down.GEMV(inter, y)
		}
		b.StopTimer()
		if s := b.Elapsed().Seconds(); s > 0 {
			b.ReportMetric(mlpBytes*float64(b.N)/s/1e9, "GB/s")
			b.ReportMetric(s/float64(b.N)*1e3, "ms/mlp")
		}
	})
}

// BenchmarkMetalQ4KGemmSteady measures the kernel's raw dequant+dot throughput when the
// per-command-buffer launch overhead is amortized over a large batch (P tokens in one
// dispatch). The single-GEMV bench above is overhead-bound (one commit→wait per ~0.5 ms of
// work); this one shows what the q4_k MSL kernel actually sustains — the number that says
// whether the GPU route can clear the CPU int8 ceiling once the forward is batched into one
// command buffer (the forward.m pattern), which is the wiring step to the decode/prefill bar.
func BenchmarkMetalQ4KGemmSteady(b *testing.B) {
	if !metalgemm.Available() {
		b.Skip("no Metal device available")
	}
	defer metalgemm.ResetQ4K()
	// Realistic Qwen3.6 MLP prefill shape: gate/up are [17408,5120], the 22-tok oracle panel.
	out, in, P := 17408, 5120, 22
	qt := randomQ4KTensor(out, in, 1)
	X := randomVecF(P*in, 2)
	w := metalgemm.UploadQ4K(qt.raw, out, in)
	if w == nil {
		b.Fatal("UploadQ4K returned nil")
	}
	Y := make([]float32, P*out)
	// The tiled kernel reads each weight ONCE per token-tile, so model-bytes ≈ the weight size
	// (×ceil(P/128) tiles). Report effective model-GB/s = weight bytes moved through the GEMM.
	tiles := float64((P + 127) / 128)
	weightBytes := tiles * float64(out) * float64(in) / 256.0 * 144.0
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		w.GEMM(X, P, Y)
	}
	b.StopTimer()
	secs := b.Elapsed().Seconds()
	if secs > 0 {
		b.ReportMetric(weightBytes*float64(b.N)/secs/1e9, "GB/s")
		// FLOP rate: 2*out*in*P MACs per GEMM — the compute-bound view once weights are read once.
		b.ReportMetric(2*float64(out)*float64(in)*float64(P)*float64(b.N)/secs/1e9, "GFLOP/s")
	}
}
