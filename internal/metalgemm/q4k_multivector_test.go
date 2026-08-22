//go:build darwin && arm64 && cgo

package metalgemm

import (
	"fmt"
	"math"
	"math/rand"
	"slices"
	"testing"
)

const (
	q4kTestBlockWeights = 256
	q4kTestBlockBytes   = 144
)

func q4kTestRaw(out, in int, seed uint64) []byte {
	if in%q4kTestBlockWeights != 0 {
		panic("q4kTestRaw: input width must be a multiple of 256")
	}
	raw := make([]byte, out*(in/q4kTestBlockWeights)*q4kTestBlockBytes)
	state := seed
	for i := range raw {
		state ^= state << 13
		state ^= state >> 7
		state ^= state << 17
		raw[i] = byte(state >> 56)
	}
	for base := 0; base < len(raw); base += q4kTestBlockBytes {
		// Keep d and dmin finite and modest; every other byte pattern is valid Q4_K data.
		raw[base+1] = 0x2c | raw[base+1]&0x03
		raw[base+3] = 0x2c | raw[base+3]&0x03
	}
	return raw
}

func q4kTestVector(n int, seed int64) []float32 {
	rng := rand.New(rand.NewSource(seed))
	x := make([]float32, n)
	for i := range x {
		x[i] = rng.Float32()*2 - 1
	}
	return x
}

func q4kTestCosineMaxRel(want, got []float32) (float64, float64) {
	var dot, nw, ng float64
	for i := range want {
		w, g := float64(want[i]), float64(got[i])
		dot += w * g
		nw += w * w
		ng += g * g
	}
	if nw == 0 || ng == 0 {
		return 0, math.Inf(1)
	}
	cosine := dot / math.Sqrt(nw*ng)
	rms := math.Sqrt(nw / float64(len(want)))
	maxRel := 0.0
	for i := range want {
		w := float64(want[i])
		if math.Abs(w) < rms {
			continue
		}
		rel := math.Abs(w-float64(got[i])) / math.Abs(w)
		if rel > maxRel {
			maxRel = rel
		}
	}
	return cosine, maxRel
}

func TestQ4KGEMVBatchMultiVectorRouteBoundary(t *testing.T) {
	for _, tc := range []struct {
		out, in, vectors int
		want             bool
	}{
		{5120, 5120, 1, false},
		{5120, 5120, 3, false},
		{5120, 5120, 4, true},
		{5120, 5120, 5, true},
		{5120, 5120, 6, true},
		{5120, 5120, 7, true},
		{17408, 5120, 8, true},
		{5120, 5120, 9, false},
		{5120, 5120, 16, false},
		{512, 5120, 4, false},
		{5120, 512, 8, false},
	} {
		if got := q4kUseMultiVector(tc.out, tc.in, tc.vectors); got != tc.want {
			t.Errorf("q4kUseMultiVector(%d, %d, %d) = %v, want %v", tc.out, tc.in, tc.vectors, got, tc.want)
		}
	}
}

func TestQ4KGEMVBatchFallbackMatchesExistingRoute(t *testing.T) {
	if !Available() {
		t.Skip("Metal unavailable")
	}
	defer ResetQ4K()
	for _, tc := range []struct {
		name       string
		out, in    int
		batchSizes []int
	}{
		{"supported-shape-outside-p", 5120, 5120, []int{1, 3, 9, 16}},
		{"unsupported-shape-inside-p", 512, 512, []int{4, 8}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			w := UploadQ4K(q4kTestRaw(tc.out, tc.in, 0x8326), tc.out, tc.in)
			if w == nil {
				t.Fatal("UploadQ4K returned nil")
			}
			defer w.Release()
			for _, p := range tc.batchSizes {
				x := q4kTestVector(p*tc.in, int64(8326+p))
				got := make([]float32, p*tc.out)
				want := make([]float32, p*tc.out)
				w.GEMVBatch(x, p, got)
				w.gemvBatchRepeated(x, p, want)
				if !slices.Equal(got, want) {
					t.Fatalf("[%d,%d] P=%d fallback differs from existing repeated-GEMV route", tc.out, tc.in, p)
				}
			}
		})
	}
}

func TestQ4KGEMVBatchMultiVectorMatchesRepeatedGEMV(t *testing.T) {
	if !Available() {
		t.Skip("Metal unavailable")
	}
	defer ResetQ4K()

	// Qwen3.8-27B has H=5120 and I=17408. Exercise every specialized pipeline on the
	// square projection, then the largest route on the FFN up projection.
	for _, tc := range []struct {
		name       string
		out, in, p int
	}{
		{"hidden-p4", 5120, 5120, 4},
		{"hidden-p5", 5120, 5120, 5},
		{"hidden-p6", 5120, 5120, 6},
		{"hidden-p7", 5120, 5120, 7},
		{"ffn-up-p8", 17408, 5120, 8},
	} {
		t.Run(tc.name, func(t *testing.T) {
			w := UploadQ4K(q4kTestRaw(tc.out, tc.in, 0x8326), tc.out, tc.in)
			if w == nil {
				t.Fatal("UploadQ4K returned nil")
			}
			defer w.Release()
			x := q4kTestVector(tc.p*tc.in, 8326)
			got := make([]float32, tc.p*tc.out)
			want := make([]float32, len(got))

			w.GEMVBatch(x, tc.p, got)
			for p := 0; p < tc.p; p++ {
				w.GEMV(x[p*tc.in:(p+1)*tc.in], want[p*tc.out:(p+1)*tc.out])
			}
			cosine, maxRel := q4kTestCosineMaxRel(want, got)
			if cosine < 0.9999 || maxRel > 5e-3 {
				t.Fatalf("P=%d [%d,%d]: cosine=%g maxRel=%g, want cosine >= 0.9999 and maxRel <= 5e-3", tc.p, tc.out, tc.in, cosine, maxRel)
			}
			t.Logf("P=%d [%d,%d]: cosine=%.9f maxRel=%g", tc.p, tc.out, tc.in, cosine, maxRel)
		})
	}
}

func BenchmarkQ4KGEMVBatchCrossover(b *testing.B) {
	if !Available() {
		b.Skip("Metal unavailable")
	}
	defer ResetQ4K()
	for _, shape := range []struct {
		name    string
		out, in int
	}{{"hidden", 5120, 5120}, {"ffn-up", 17408, 5120}} {
		w := UploadQ4K(q4kTestRaw(shape.out, shape.in, 0x8326), shape.out, shape.in)
		if w == nil {
			b.Fatal("UploadQ4K returned nil")
		}
		for _, p := range []int{1, 4, 8, 16} {
			x := q4kTestVector(p*shape.in, int64(8326+p))
			y := make([]float32, p*shape.out)
			arms := []struct {
				name string
				run  func()
			}{
				{"repeated-gemv", func() { w.gemvBatchRepeated(x, p, y) }},
				{"routed-batch", func() { w.GEMVBatch(x, p, y) }},
				{"gemm", func() { w.GEMM(x, p, y) }},
			}
			for _, arm := range arms {
				b.Run(fmt.Sprintf("%s/P=%d/%s", shape.name, p, arm.name), func(b *testing.B) {
					b.ReportAllocs()
					b.ResetTimer()
					for i := 0; i < b.N; i++ {
						arm.run()
					}
					if elapsed := b.Elapsed().Seconds(); elapsed > 0 {
						b.ReportMetric(float64(p*b.N)/elapsed, "vectors/s")
					}
				})
			}
		}
		w.Release()
	}
}
