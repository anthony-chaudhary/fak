package model

import (
	"fmt"
	"os"
	"strconv"
	"strings"
	"testing"
)

// benchmark the pure Q8_0 GEMM kernel in isolation (no attention / quant / norm), at the
// real SmolLM2-135M prefill shapes, so the tile kernel's speedup over the legacy per-row
// qdot8 sweep is measured cleanly. Run via WSL:
//
//	./fak/test.ps1 -bench=QGemmKernel -benchtime=2x -run=^$ ./internal/model/
func benchOneGemm(b *testing.B, out, in, P int, legacy bool) {
	w := mkVec(out*in, uint64(out*in*131+7))
	qt := quantizeQ8(w, out, in)
	X := mkVec(P*in, uint64(P*in*977+3))
	qp := quantizeBatchPanel(X, P, in)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if legacy {
			_ = qGemm8legacy(qt, qp)
		} else {
			_ = qGemm8(qt, qp)
		}
	}
	b.StopTimer()
	macs := float64(out) * float64(in) * float64(P)
	b.ReportMetric(macs/(float64(b.Elapsed().Nanoseconds())/float64(b.N)), "MAC/ns")
}

// BenchmarkPrefillQ256 times the FULL Q8 prefill (GEMM + attention + quant + norm + rope)
// so the GEMM-kernel benchmark's per-shape sum can be subtracted to isolate the non-GEMM
// overhead. Run via WSL (16-thread there):
//
//	./fak/test.ps1 -bench=PrefillQ256 -benchtime=5x -run=^$ ./internal/model/
func BenchmarkPrefillQ256(b *testing.B) {
	if _, err := os.Stat(cacheDir + "/weights.f32"); err != nil {
		b.Skip("no exported weights")
	}
	m, err := Load(cacheDir)
	if err != nil {
		b.Fatalf("load: %v", err)
	}
	m.Quantize()
	ids := make([]int, 256)
	st := uint64(2463534242)
	for i := range ids {
		st = (st*1103515245 + 12345) & 0x7fffffff
		ids[i] = int(st % uint64(m.Cfg.VocabSize))
	}
	// warm
	{
		s := m.NewSession()
		s.Quant = true
		s.Prefill(ids)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		s := m.NewSession()
		s.Quant = true
		s.Prefill(ids)
	}
}

// BenchmarkStepBatchQ times one Q8 multi-user decode step (the throughput-lane primitive)
// with -benchmem, so the per-step allocation/GC the batching doc names as a measured
// gap-to-roofline component is quantified. Run via WSL:
//
//	./fak/test.ps1 -bench=StepBatchQ -benchmem -benchtime=20x -run=^$ ./internal/model/
func BenchmarkStepBatchQ(b *testing.B) {
	if _, err := os.Stat(cacheDir + "/weights.f32"); err != nil {
		b.Skip("no exported weights")
	}
	m, err := Load(cacheDir)
	if err != nil {
		b.Fatalf("load: %v", err)
	}
	m.Quantize()
	prefixLen := benchEnvInt("FAK_BENCH_PREFIX", 256)
	prefix := make([]int, prefixLen)
	st := uint64(2463534242)
	for i := range prefix {
		st = (st*1103515245 + 12345) & 0x7fffffff
		prefix[i] = int(st % uint64(m.Cfg.VocabSize))
	}
	for _, B := range benchEnvInts("FAK_BENCH_BATCHES", []int{8, 16, 32, 128}) {
		b.Run(fmt.Sprintf("B%d", B), func(b *testing.B) {
			base := m.NewSession()
			base.Quant = true
			base.Prefill(prefix)
			bs := m.NewBatchFromPrefix(base.Cache, B)
			bs.SetQuant(true)
			bs.Reserve(b.N + 1)
			step := make([]int, B)
			for i := range step {
				step[i] = i % m.Cfg.VocabSize
			}
			bs.StepBatch(step) // warm
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				bs.StepBatch(step)
			}
		})
	}
}

// BenchmarkPrefillEachRectQResult targets fleetserve's repeated private-result ingestion:
// C agents each append the same-size R-token result rectangle between decode bursts. The
// benchmark warms once so -benchmem reports the steady-state allocation after BatchSession's
// rectangular-prefill buffers have grown.
func BenchmarkPrefillEachRectQResult(b *testing.B) {
	if _, err := os.Stat(cacheDir + "/weights.f32"); err != nil {
		b.Skip("no exported weights")
	}
	m, err := Load(cacheDir)
	if err != nil {
		b.Fatalf("load: %v", err)
	}
	m.Quantize()
	B := benchEnvInt("FAK_BENCH_RECT_B", 8)
	P := benchEnvInt("FAK_BENCH_RECT_P", 48)
	prefixLen := benchEnvInt("FAK_BENCH_PREFIX", 1024)
	prefix := make([]int, prefixLen)
	st := uint64(2463534242)
	for i := range prefix {
		st = (st*1103515245 + 12345) & 0x7fffffff
		prefix[i] = int(st % uint64(m.Cfg.VocabSize))
	}
	prompts := make([][]int, B)
	for user := range prompts {
		prompts[user] = make([]int, P)
		st := uint64(10_000 + user*97)
		for i := range prompts[user] {
			st = (st*1103515245 + 12345) & 0x7fffffff
			prompts[user][i] = int(st % uint64(m.Cfg.VocabSize))
		}
	}
	base := m.NewSession()
	base.Quant = true
	base.Prefill(prefix)
	bs := m.NewBatchFromPrefixReserve(base.Cache, B, P*(b.N+2))
	bs.SetQuant(true)
	bs.PrefillEachNoLogits(prompts) // grow reusable buffers off the timed path

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		bs.PrefillEachNoLogits(prompts)
	}
}

// BenchmarkPrefillEachRectQSynthetic measures the full rectangular Q8 result-ingest path on
// deterministic synthetic weights. It stays small enough for local CI but preserves the
// group-3 GQA shape and shows the cost avoided when fleetserve does not need final logits.
func BenchmarkPrefillEachRectQSynthetic(b *testing.B) {
	B := benchEnvInt("FAK_BENCH_RECT_B", 8)
	P := benchEnvInt("FAK_BENCH_RECT_P", 48)
	prefixLen := benchEnvInt("FAK_BENCH_PREFIX", 128)
	cfg := Config{
		HiddenSize: 96, NumLayers: 2, NumHeads: 6, NumKVHeads: 2, HeadDim: 16,
		IntermediateSize: 192, VocabSize: 257, RMSNormEps: 1e-5, RopeTheta: 10000,
		TieWordEmbeddings: true, EOSTokenID: -1,
	}
	m := NewSynthetic(cfg)
	m.Quantize()
	prefix := make([]int, prefixLen)
	for i := range prefix {
		prefix[i] = (i*37 + 9) % cfg.VocabSize
	}
	prompts := make([][]int, B)
	for user := range prompts {
		prompts[user] = make([]int, P)
		for i := range prompts[user] {
			prompts[user][i] = (user*43 + i*19 + 11) % cfg.VocabSize
		}
	}

	bench := func(b *testing.B, wantLogits bool) {
		base := m.NewSession()
		base.Quant = true
		base.Prefill(prefix)
		bs := m.NewBatchFromPrefixReserve(base.Cache, B, P*(b.N+2))
		bs.SetQuant(true)
		if wantLogits {
			bs.PrefillEach(prompts)
		} else {
			bs.PrefillEachNoLogits(prompts)
		}
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			if wantLogits {
				bs.PrefillEach(prompts)
			} else {
				bs.PrefillEachNoLogits(prompts)
			}
		}
	}
	b.Run("with_logits", func(b *testing.B) { bench(b, true) })
	b.Run("no_logits", func(b *testing.B) { bench(b, false) })
}

func BenchmarkPrefillNoLogitsSynthetic(b *testing.B) {
	P := benchEnvInt("FAK_BENCH_RECT_P", 48)
	prefixLen := benchEnvInt("FAK_BENCH_PREFIX", 128)
	cfg := Config{
		HiddenSize: 96, NumLayers: 2, NumHeads: 6, NumKVHeads: 2, HeadDim: 16,
		IntermediateSize: 192, VocabSize: 257, RMSNormEps: 1e-5, RopeTheta: 10000,
		TieWordEmbeddings: true, EOSTokenID: -1,
	}
	m := NewSynthetic(cfg)
	m.Quantize()
	prefix := make([]int, prefixLen)
	for i := range prefix {
		prefix[i] = (i*37 + 9) % cfg.VocabSize
	}
	prompt := make([]int, P)
	for i := range prompt {
		prompt[i] = (i*19 + 11) % cfg.VocabSize
	}

	bench := func(b *testing.B, wantLogits bool) {
		base := m.NewSession()
		base.Quant = true
		base.Prefill(prefix)
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			s := m.NewSession()
			s.Quant = true
			s.Cache = base.Cache.CloneWithReserve(P)
			if wantLogits {
				s.Prefill(prompt)
			} else {
				s.PrefillNoLogits(prompt)
			}
		}
	}
	b.Run("with_logits", func(b *testing.B) { bench(b, true) })
	b.Run("no_logits", func(b *testing.B) { bench(b, false) })
}

// BenchmarkPrefillMultiAttentionSynthetic isolates the rectangular private-result attention
// kernel without requiring exported model weights. It compares the old per-query-head loop
// against the grouped GQA helper used by the Q8 rectangular path.
func BenchmarkPrefillMultiAttentionSynthetic(b *testing.B) {
	B := benchEnvInt("FAK_BENCH_RECT_B", 8)
	P := benchEnvInt("FAK_BENCH_RECT_P", 48)
	prefixLen := benchEnvInt("FAK_BENCH_PREFIX", 1024)
	cfg := Config{
		HiddenSize: 96, NumLayers: 1, NumHeads: 6, NumKVHeads: 2, HeadDim: 16,
		IntermediateSize: 192, VocabSize: 257, RMSNormEps: 1e-5, RopeTheta: 10000,
		TieWordEmbeddings: true, EOSTokenID: -1,
	}
	hd, nH, grp := cfg.HeadDim, cfg.NumHeads, cfg.GroupSize()
	w := cfg.NumKVHeads * hd
	scale := float32(1.0)
	baseB := make([]int, B)
	caches := make([]*KVCache, B)
	for user := 0; user < B; user++ {
		baseB[user] = prefixLen
		c := NewKVCache(cfg)
		c.K[0] = mkVec((prefixLen+P)*w, uint64(user*1009+17))
		c.V[0] = mkVec((prefixLen+P)*w, uint64(user*2039+29))
		caches[user] = c
	}
	Q := mkVec(B*P*nH*hd, uint64(B*P*nH*hd+41))

	b.Run("per_head", func(b *testing.B) {
		b.ReportAllocs()
		attnOut := make([]float32, B*P*nH*hd)
		var scratch [][]float32
		scratch = attnPrefillMultiInto(attnOut, Q, caches, baseB, 0, P, nH, hd, w, grp, -1, scale, fdot, scratch)
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			clear(attnOut)
			scratch = attnPrefillMultiInto(attnOut, Q, caches, baseB, 0, P, nH, hd, w, grp, -1, scale, fdot, scratch)
		}
	})
	b.Run("gqa_grouped", func(b *testing.B) {
		b.ReportAllocs()
		attnOut := make([]float32, B*P*nH*hd)
		var scratch [][]float32
		scratch = attnPrefillMultiGQAInto(attnOut, Q, caches, baseB, 0, P, nH, hd, w, grp, -1, scale, fdot, fdot3scalar, scratch)
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			clear(attnOut)
			scratch = attnPrefillMultiGQAInto(attnOut, Q, caches, baseB, 0, P, nH, hd, w, grp, -1, scale, fdot, fdot3scalar, scratch)
		}
	})
}

func benchEnvInt(name string, def int) int {
	if s := os.Getenv(name); s != "" {
		if n, err := strconv.Atoi(s); err == nil && n >= 1 {
			return n
		}
	}
	return def
}

func benchEnvInts(name string, def []int) []int {
	s := os.Getenv(name)
	if s == "" {
		return def
	}
	var out []int
	for _, part := range strings.Split(s, ",") {
		n, err := strconv.Atoi(strings.TrimSpace(part))
		if err == nil && n >= 1 {
			out = append(out, n)
		}
	}
	if len(out) == 0 {
		return def
	}
	return out
}

func BenchmarkQGemmKernel(b *testing.B) {
	shapes := []struct {
		name       string
		out, in, P int
	}{
		{"gate_1536x576_P256", 1536, 576, 256},
		{"down_576x1536_P256", 576, 1536, 256},
		{"qproj_576x576_P256", 576, 576, 256},
	}
	for _, s := range shapes {
		b.Run(fmt.Sprintf("%s/tile", s.name), func(b *testing.B) { benchOneGemm(b, s.out, s.in, s.P, false) })
		b.Run(fmt.Sprintf("%s/legacy", s.name), func(b *testing.B) { benchOneGemm(b, s.out, s.in, s.P, true) })
	}
}

// BenchmarkQuantizeVecQ8 isolates the decode activation quantizer called before every Q8
// projection. It does not need exported HF weights; use it to size the scalar-vs-arch-dispatch
// term for issue #45:
//
//	./fak/test.ps1 -bench=QuantizeVecQ8 -benchmem -benchtime=200ms -run=^$ ./internal/model/
func BenchmarkQuantizeVecQ8(b *testing.B) {
	for _, width := range []int{576, 1536} {
		x := mkVec(width, uint64(width*131+17))
		b.Run(fmt.Sprintf("width%d", width), func(b *testing.B) {
			b.ReportAllocs()
			var qv q8Vec
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				qv = quantizeVecQ8(x)
			}
			b.StopTimer()
			if qv.nblk == 0 {
				b.Fatal("empty quantized vector")
			}
			b.ReportMetric(float64(width)*float64(b.N)/b.Elapsed().Seconds(), "float/s")
		})
	}
}

func BenchmarkQuantizeVecQ8Reuse(b *testing.B) {
	for _, width := range []int{576, 1536} {
		x := mkVec(width, uint64(width*131+17))
		b.Run(fmt.Sprintf("width%d", width), func(b *testing.B) {
			b.ReportAllocs()
			var qv q8Vec
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				qv = quantizeVecQ8Into(&qv, x)
			}
			b.StopTimer()
			if qv.nblk == 0 {
				b.Fatal("empty quantized vector")
			}
			b.ReportMetric(float64(width)*float64(b.N)/b.Elapsed().Seconds(), "float/s")
		})
	}
}
