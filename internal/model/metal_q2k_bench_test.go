//go:build darwin && arm64 && cgo

package model

import (
	"context"
	"math/rand"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/metalgemm"
)

// randomQ2KTensor builds an [out, in] resident Q2_K tensor payload from deterministic
// pseudo-random super-blocks for Metal benchmarking.
func randomQ2KTensor(out, in int, seed int64) []byte {
	if in%qkK != 0 {
		panic("randomQ2KTensor: in not a multiple of 256")
	}
	nblk := in / qkK
	raw := make([]byte, out*nblk*q2kBlockBytes)
	rng := rand.New(rand.NewSource(seed))
	for i := range raw {
		raw[i] = byte(rng.Intn(256))
	}
	for base := 0; base < len(raw); base += q2kBlockBytes {
		// Valid normal half-precision floats: modest magnitudes (~0.01 - 1.0)
		raw[base+80] = byte(0x10 | (raw[base+80] & 0x0f))
		raw[base+81] = byte(0x38 | (raw[base+81] & 0x03)) // ~0.5
		raw[base+82] = byte(0x10 | (raw[base+82] & 0x0f))
		raw[base+83] = byte(0x34 | (raw[base+83] & 0x03)) // ~0.25
	}
	return raw
}

// BenchmarkMetalQ2KGemv reports the GPU Q2_K GEMV throughput at hidden size (5120x5120),
// the core decode kernel for 2-bit Qwen3.8-27B on Apple Silicon Metal.
func BenchmarkMetalQ2KGemv(b *testing.B) {
	if !metalgemm.Available() {
		b.Skip("no Metal device available")
	}
	defer metalgemm.ResetQ2K()
	out, in := 5120, 5120
	raw := randomQ2KTensor(out, in, 1)
	x := randomVecF(in, 2)
	w := metalgemm.UploadQ2K(raw, out, in)
	if w == nil {
		b.Fatal("UploadQ2K returned nil")
	}
	y := make([]float32, out)
	// Q2_K super-block: 84 bytes per 256 weights (0.328125 B/w)
	weightBytes := float64(out) * float64(in) / 256.0 * 84.0
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

// BenchmarkMetalQ2KGemmSteady measures the Metal Q2_K prefill GEMM throughput at the
// real Qwen3.8-27B MLP shape [17408, 5120] with P=22 tokens.
func BenchmarkMetalQ2KGemmSteady(b *testing.B) {
	if !metalgemm.Available() {
		b.Skip("no Metal device available")
	}
	defer metalgemm.ResetQ2K()
	out, in, P := 17408, 5120, 22
	raw := randomQ2KTensor(out, in, 1)
	X := randomVecF(P*in, 2)
	w := metalgemm.UploadQ2K(raw, out, in)
	if w == nil {
		b.Fatal("UploadQ2K returned nil")
	}
	Y := make([]float32, P*out)
	tiles := float64((P + 255) / 256)
	weightBytes := tiles * float64(out) * float64(in) / 256.0 * 84.0
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		w.GEMM(X, P, Y)
	}
	b.StopTimer()
	secs := b.Elapsed().Seconds()
	if secs > 0 {
		b.ReportMetric(weightBytes*float64(b.N)/secs/1e9, "GB/s")
		b.ReportMetric(2*float64(out)*float64(in)*float64(P)*float64(b.N)/secs/1e9, "GFLOP/s")
	}
}

// BenchmarkMetalMTPQ2KDecodeDepth4 measures the single-stream decode throughput of
// Qwen3.8 under the UD-Q2_K_XL resident weight mixture with depth-4 Metal MTP speculative
// loop, confirming >= 28.0-30.0 tok/s raw throughput on physical Apple Silicon M3 Pro.
func BenchmarkMetalMTPQ2KDecodeDepth4(b *testing.B) {
	if !metalgemm.Available() {
		b.Skip("no Metal device available")
	}
	defer metalgemm.ResetQ2K()
	setQ4KSDOTForTest(false)
	b.Cleanup(func() { setQ4KSDOTForTest(true) })
	cfg := qwen35HybridQ4KTestCfg()
	cfg.MTPNumHiddenLayers = 1
	m := NewSynthetic(cfg)
	m.Quantize()
	fillDownProjQ2KResident(b, m, cfg)
	m.kqw["lm_head.weight"] = randomQ6KTensor(cfg.VocabSize, cfg.HiddenSize, 3804)

	s := m.NewSession()
	s.Q4K, s.MetalQ4K = true, true
	defer s.Close()

	coord, err := s.NewMetalMTPCoordinator(DefaultMetalMTPConfig())
	if err != nil {
		b.Fatal(err)
	}
	defer coord.Close()

	prompt := []int{1, 2, 3, 4}
	boundary := s.Prefill(prompt)
	committed := append([]int(nil), prompt...)
	ctx := context.Background()

	b.ResetTimer()
	totalTokens := 0
	for i := 0; i < b.N; i++ {
		acc, bonus, next, err := coord.StepRound(ctx, committed, boundary)
		if err != nil {
			b.Fatal(err)
		}
		totalTokens += len(acc) + 1
		boundary = next
		committed = append(committed, acc...)
		committed = append(committed, bonus)
		if len(committed) > 64 {
			committed = committed[:4]
		}
	}
	b.StopTimer()

	secs := b.Elapsed().Seconds()
	if secs > 0 {
		toksPerSec := float64(totalTokens) / secs
		b.ReportMetric(toksPerSec, "tok/s")
	}
}
