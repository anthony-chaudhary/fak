package model

import (
	"math"
	"math/rand"
	"testing"
)

// generateRandomMatrix generates a rows x cols matrix with deterministic random values.
func generateRandomMatrix(rng *rand.Rand, rows, cols int, scale float32) [][]float32 {
	m := make([][]float32, rows)
	for i := 0; i < rows; i++ {
		row := make([]float32, cols)
		for j := 0; j < cols; j++ {
			row[j] = (rng.Float32()*2 - 1) * scale
		}
		m[i] = row
	}
	return m
}

func maxDiffVectors(a, b []float32) float32 {
	var maxD float32
	for i := range a {
		d := float32(math.Abs(float64(a[i] - b[i])))
		if d > maxD {
			maxD = d
		}
	}
	return maxD
}

func maxDiffMatrices(a, b [][]float32) float32 {
	var maxD float32
	for i := range a {
		for j := range a[i] {
			d := float32(math.Abs(float64(a[i][j] - b[i][j])))
			if d > maxD {
				maxD = d
			}
		}
	}
	return maxD
}

func TestWyfMatchesSequential(t *testing.T) {
	const tolerance = 1e-4

	testCases := []struct {
		name      string
		T         int
		dim       int
		chunkSize int
		useAlpha  bool
		useBeta   bool
		useS0     bool
	}{
		{
			name:      "StandardDelta_T64_Dim32_C16",
			T:         64,
			dim:       32,
			chunkSize: 16,
			useAlpha:  false,
			useBeta:   false,
			useS0:     false,
		},
		{
			name:      "StandardDelta_T64_Dim64_C32",
			T:         64,
			dim:       64,
			chunkSize: 32,
			useAlpha:  false,
			useBeta:   false,
			useS0:     false,
		},
		{
			name:      "GatedDelta_T64_Dim64_C16",
			T:         64,
			dim:       64,
			chunkSize: 16,
			useAlpha:  true,
			useBeta:   true,
			useS0:     true,
		},
		{
			name:      "GatedDelta_T64_Dim64_C32",
			T:         64,
			dim:       64,
			chunkSize: 32,
			useAlpha:  true,
			useBeta:   true,
			useS0:     true,
		},
		{
			name:      "GatedDelta_NonDivisible_T50_Dim64_C16",
			T:         50,
			dim:       64,
			chunkSize: 16,
			useAlpha:  true,
			useBeta:   true,
			useS0:     true,
		},
		{
			name:      "GatedDelta_T128_Dim128_C32",
			T:         128,
			dim:       128,
			chunkSize: 32,
			useAlpha:  true,
			useBeta:   true,
			useS0:     true,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			rng := rand.New(rand.NewSource(42))

			// Normalize keys to prevent explosion in recurrence
			k := generateRandomMatrix(rng, tc.T, tc.dim, 1.0)
			for i := 0; i < tc.T; i++ {
				var norm float32
				for d := 0; d < tc.dim; d++ {
					norm += k[i][d] * k[i][d]
				}
				norm = float32(math.Sqrt(float64(norm)))
				if norm > 0 {
					for d := 0; d < tc.dim; d++ {
						k[i][d] /= norm
					}
				}
			}

			v := generateRandomMatrix(rng, tc.T, tc.dim, 0.5)

			var alpha [][]float32
			if tc.useAlpha {
				alpha = make([][]float32, tc.T)
				for i := 0; i < tc.T; i++ {
					// typical decay values between 0.90 and 0.99
					alpha[i] = []float32{0.90 + rng.Float32()*0.09}
				}
			}

			var beta [][]float32
			if tc.useBeta {
				beta = make([][]float32, tc.T)
				for i := 0; i < tc.T; i++ {
					// typical beta gate between 0.2 and 0.8
					beta[i] = []float32{0.2 + rng.Float32()*0.6}
				}
			}

			var s0 [][]float32
			if tc.useS0 {
				s0 = generateRandomMatrix(rng, tc.dim, tc.dim, 0.1)
			}

			// Run sequential reference
			wantOutputs, wantStates := SequentialGDN(k, v, beta, alpha, s0)

			// Run WYF chunkwise recurrence
			gotOutputs, gotStates := WYFChunkwiseRecurrence(k, v, beta, alpha, s0, tc.chunkSize)

			if len(gotOutputs) != len(wantOutputs) {
				t.Fatalf("output length mismatch: got %d, want %d", len(gotOutputs), len(wantOutputs))
			}
			if len(gotStates) != len(wantStates) {
				t.Fatalf("states length mismatch: got %d, want %d", len(gotStates), len(wantStates))
			}

			var maxOutDiff float32
			for i := 0; i < tc.T; i++ {
				d := maxDiffVectors(gotOutputs[i], wantOutputs[i])
				if d > maxOutDiff {
					maxOutDiff = d
				}
				if d >= tolerance {
					t.Fatalf("token %d output diff %v >= tolerance %v", i, d, tolerance)
				}
			}

			var maxStateDiff float32
			for i := 0; i < tc.T; i++ {
				d := maxDiffMatrices(gotStates[i], wantStates[i])
				if d > maxStateDiff {
					maxStateDiff = d
				}
				if d >= tolerance {
					t.Fatalf("token %d state diff %v >= tolerance %v", i, d, tolerance)
				}
			}

			t.Logf("[%s] PASSED: max output diff = %e, max state diff = %e (< %e)",
				tc.name, maxOutDiff, maxStateDiff, tolerance)
		})
	}
}

func TestWyfEdgeCases(t *testing.T) {
	// 1. Empty sequence
	out, st := WYFChunkwiseRecurrence(nil, nil, nil, nil, nil, 16)
	if out != nil || st != nil {
		t.Errorf("expected nil for empty sequence, got %v, %v", out, st)
	}

	// 2. Single token (T = 1)
	rng := rand.New(rand.NewSource(123))
	k := generateRandomMatrix(rng, 1, 32, 1.0)
	v := generateRandomMatrix(rng, 1, 32, 0.5)
	wantOut, wantSt := SequentialGDN(k, v, nil, nil, nil)
	gotOut, gotSt := WYFChunkwiseRecurrence(k, v, nil, nil, nil, 16)

	const tolerance = 1e-4

	if diff := maxDiffVectors(gotOut[0], wantOut[0]); diff >= tolerance {
		t.Errorf("T=1 output diff %e >= %e", diff, tolerance)
	}
	if diff := maxDiffMatrices(gotSt[0], wantSt[0]); diff >= tolerance {
		t.Errorf("T=1 state diff %e >= %e", diff, tolerance)
	}

	// 3. Chunk size > sequence length
	k = generateRandomMatrix(rng, 5, 32, 1.0)
	v = generateRandomMatrix(rng, 5, 32, 0.5)
	wantOut, wantSt = SequentialGDN(k, v, nil, nil, nil)
	gotOut, gotSt = WYFChunkwiseRecurrence(k, v, nil, nil, nil, 64)
	for i := 0; i < 5; i++ {
		if diff := maxDiffVectors(gotOut[i], wantOut[i]); diff >= tolerance {
			t.Errorf("C>T token %d output diff %e >= %e", i, diff, tolerance)
		}
		if diff := maxDiffMatrices(gotSt[i], wantSt[i]); diff >= tolerance {
			t.Errorf("C>T token %d state diff %e >= %e", i, diff, tolerance)
		}
	}
}

func BenchmarkSequentialGDN(b *testing.B) {
	const T = 256
	const dim = 64
	rng := rand.New(rand.NewSource(999))
	k := generateRandomMatrix(rng, T, dim, 0.1)
	v := generateRandomMatrix(rng, T, dim, 0.1)
	alpha := make([][]float32, T)
	beta := make([][]float32, T)
	for i := 0; i < T; i++ {
		alpha[i] = []float32{0.95}
		beta[i] = []float32{0.5}
	}
	s0 := generateRandomMatrix(rng, dim, dim, 0.01)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		SequentialGDN(k, v, beta, alpha, s0)
	}
}

func BenchmarkWYFChunkwiseRecurrenceC16(b *testing.B) {
	const T = 256
	const dim = 64
	rng := rand.New(rand.NewSource(999))
	k := generateRandomMatrix(rng, T, dim, 0.1)
	v := generateRandomMatrix(rng, T, dim, 0.1)
	alpha := make([][]float32, T)
	beta := make([][]float32, T)
	for i := 0; i < T; i++ {
		alpha[i] = []float32{0.95}
		beta[i] = []float32{0.5}
	}
	s0 := generateRandomMatrix(rng, dim, dim, 0.01)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		WYFChunkwiseRecurrence(k, v, beta, alpha, s0, 16)
	}
}

func BenchmarkWYFChunkwiseRecurrenceC32(b *testing.B) {
	const T = 256
	const dim = 64
	rng := rand.New(rand.NewSource(999))
	k := generateRandomMatrix(rng, T, dim, 0.1)
	v := generateRandomMatrix(rng, T, dim, 0.1)
	alpha := make([][]float32, T)
	beta := make([][]float32, T)
	for i := 0; i < T; i++ {
		alpha[i] = []float32{0.95}
		beta[i] = []float32{0.5}
	}
	s0 := generateRandomMatrix(rng, dim, dim, 0.01)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		WYFChunkwiseRecurrence(k, v, beta, alpha, s0, 32)
	}
}

func TestWyfPrefillMatchesSequential(t *testing.T) {
	const tolerance = 1e-4
	const T = 64
	const dim = 64
	rng := rand.New(rand.NewSource(777))
	k := generateRandomMatrix(rng, T, dim, 0.1)
	v := generateRandomMatrix(rng, T, dim, 0.1)
	alpha := make([][]float32, T)
	beta := make([][]float32, T)
	for i := 0; i < T; i++ {
		alpha[i] = []float32{0.95}
		beta[i] = []float32{0.5}
	}
	s0 := generateRandomMatrix(rng, dim, dim, 0.01)

	wantOut, wantFinalS := SequentialGDNPrefill(k, v, beta, alpha, s0)
	gotOut, gotFinalS := WYFChunkwisePrefill(k, v, beta, alpha, s0, 16)

	for i := 0; i < T; i++ {
		if diff := maxDiffVectors(gotOut[i], wantOut[i]); diff >= tolerance {
			t.Fatalf("prefill token %d diff %e >= tolerance %e", i, diff, tolerance)
		}
	}
	if diff := maxDiffMatrices(gotFinalS, wantFinalS); diff >= tolerance {
		t.Fatalf("prefill final state diff %e >= tolerance %e", diff, tolerance)
	}
}

func BenchmarkSequentialGDNPrefill(b *testing.B) {
	const T = 256
	const dim = 64
	rng := rand.New(rand.NewSource(999))
	k := generateRandomMatrix(rng, T, dim, 0.1)
	v := generateRandomMatrix(rng, T, dim, 0.1)
	alpha := make([][]float32, T)
	beta := make([][]float32, T)
	for i := 0; i < T; i++ {
		alpha[i] = []float32{0.95}
		beta[i] = []float32{0.5}
	}
	s0 := generateRandomMatrix(rng, dim, dim, 0.01)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		SequentialGDNPrefill(k, v, beta, alpha, s0)
	}
}

func BenchmarkWYFChunkwisePrefillC16(b *testing.B) {
	const T = 256
	const dim = 64
	rng := rand.New(rand.NewSource(999))
	k := generateRandomMatrix(rng, T, dim, 0.1)
	v := generateRandomMatrix(rng, T, dim, 0.1)
	alpha := make([][]float32, T)
	beta := make([][]float32, T)
	for i := 0; i < T; i++ {
		alpha[i] = []float32{0.95}
		beta[i] = []float32{0.5}
	}
	s0 := generateRandomMatrix(rng, dim, dim, 0.01)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		WYFChunkwisePrefill(k, v, beta, alpha, s0, 16)
	}
}

func BenchmarkWYFChunkwisePrefillC32(b *testing.B) {
	const T = 256
	const dim = 64
	rng := rand.New(rand.NewSource(999))
	k := generateRandomMatrix(rng, T, dim, 0.1)
	v := generateRandomMatrix(rng, T, dim, 0.1)
	alpha := make([][]float32, T)
	beta := make([][]float32, T)
	for i := 0; i < T; i++ {
		alpha[i] = []float32{0.95}
		beta[i] = []float32{0.5}
	}
	s0 := generateRandomMatrix(rng, dim, dim, 0.01)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		WYFChunkwisePrefill(k, v, beta, alpha, s0, 32)
	}
}

func BenchmarkSequentialGDN_T512_Dim64(b *testing.B) {
	const T = 512
	const dim = 64
	rng := rand.New(rand.NewSource(999))
	k := generateRandomMatrix(rng, T, dim, 0.1)
	v := generateRandomMatrix(rng, T, dim, 0.1)
	alpha := make([][]float32, T)
	beta := make([][]float32, T)
	for i := 0; i < T; i++ {
		alpha[i] = []float32{0.95}
		beta[i] = []float32{0.5}
	}
	s0 := generateRandomMatrix(rng, dim, dim, 0.01)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		SequentialGDN(k, v, beta, alpha, s0)
	}
}

func BenchmarkWYFChunkwiseRecurrenceC16_T512_Dim64(b *testing.B) {
	const T = 512
	const dim = 64
	rng := rand.New(rand.NewSource(999))
	k := generateRandomMatrix(rng, T, dim, 0.1)
	v := generateRandomMatrix(rng, T, dim, 0.1)
	alpha := make([][]float32, T)
	beta := make([][]float32, T)
	for i := 0; i < T; i++ {
		alpha[i] = []float32{0.95}
		beta[i] = []float32{0.5}
	}
	s0 := generateRandomMatrix(rng, dim, dim, 0.01)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		WYFChunkwiseRecurrence(k, v, beta, alpha, s0, 16)
	}
}

func BenchmarkSequentialGDNPrefill_T512_Dim64(b *testing.B) {
	const T = 512
	const dim = 64
	rng := rand.New(rand.NewSource(999))
	k := generateRandomMatrix(rng, T, dim, 0.1)
	v := generateRandomMatrix(rng, T, dim, 0.1)
	alpha := make([][]float32, T)
	beta := make([][]float32, T)
	for i := 0; i < T; i++ {
		alpha[i] = []float32{0.95}
		beta[i] = []float32{0.5}
	}
	s0 := generateRandomMatrix(rng, dim, dim, 0.01)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		SequentialGDNPrefill(k, v, beta, alpha, s0)
	}
}

func BenchmarkWYFChunkwisePrefillC16_T512_Dim64(b *testing.B) {
	const T = 512
	const dim = 64
	rng := rand.New(rand.NewSource(999))
	k := generateRandomMatrix(rng, T, dim, 0.1)
	v := generateRandomMatrix(rng, T, dim, 0.1)
	alpha := make([][]float32, T)
	beta := make([][]float32, T)
	for i := 0; i < T; i++ {
		alpha[i] = []float32{0.95}
		beta[i] = []float32{0.5}
	}
	s0 := generateRandomMatrix(rng, dim, dim, 0.01)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		WYFChunkwisePrefill(k, v, beta, alpha, s0, 16)
	}
}
