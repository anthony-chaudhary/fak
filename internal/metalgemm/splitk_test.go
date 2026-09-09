//go:build darwin && arm64 && cgo

package metalgemm

import (
	"fmt"
	"math"
	"math/rand"
	"testing"
	"time"
)

// generateTestFloats creates a deterministic pseudo-random float32 slice.
func generateTestFloats(n int, seed int64, scale float32) []float32 {
	rng := rand.New(rand.NewSource(seed))
	out := make([]float32, n)
	for i := 0; i < n; i++ {
		out[i] = (rng.Float32() - 0.5) * 2.0 * scale
	}
	return out
}

// computeMaxDiff computes the maximum absolute difference between two float slices.
func computeMaxDiff(a, b []float32) float32 {
	var maxDiff float32
	n := len(a)
	if len(b) < n {
		n = len(b)
	}
	for i := 0; i < n; i++ {
		d := float32(math.Abs(float64(a[i] - b[i])))
		if d > maxDiff {
			maxDiff = d
		}
	}
	return maxDiff
}

// TestSplitKConfig verifies the dynamic Split-K factor calculator across diverse matrix shapes.
func TestSplitKConfig(t *testing.T) {
	tests := []struct {
		name               string
		m, n, k, groupSize int
		wantSplitK         int
		wantTotalTGs       int
		wantKPartition     int
	}{
		{
			name:           "decode_m1_n2048_k4096",
			m:              1,
			n:              2048,
			k:              4096,
			groupSize:      32,
			wantSplitK:     8,
			wantTotalTGs:   512,
			wantKPartition: 512,
		},
		{
			name:           "decode_m1_n8192_k4096",
			m:              1,
			n:              8192,
			k:              4096,
			groupSize:      32,
			wantSplitK:     2,
			wantTotalTGs:   512,
			wantKPartition: 2048,
		},
		{
			name:           "prefill_m32_n2048_k4096",
			m:              32,
			n:              2048,
			k:              4096,
			groupSize:      32,
			wantSplitK:     8,
			wantTotalTGs:   512,
			wantKPartition: 512,
		},
		{
			name:           "large_prefill_m512_n2048_k4096_saturated",
			m:              512,
			n:              2048,
			k:              4096,
			groupSize:      32,
			wantSplitK:     1,
			wantTotalTGs:   1024,
			wantKPartition: 4096,
		},
		{
			name:           "unaligned_k_m1_n2048_k4000",
			m:              1,
			n:              2048,
			k:              4000,
			groupSize:      32,
			wantSplitK:     5,
			wantTotalTGs:   320,
			wantKPartition: 800,
		},
		{
			name:           "edge_k0",
			m:              1,
			n:              2048,
			k:              0,
			groupSize:      32,
			wantSplitK:     1,
			wantTotalTGs:   64,
			wantKPartition: 0,
		},
		{
			name:           "edge_group_size_0",
			m:              1,
			n:              2048,
			k:              4096,
			groupSize:      0,
			wantSplitK:     1,
			wantTotalTGs:   64,
			wantKPartition: 4096,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			plan := CalculateSplitK(tc.m, tc.n, tc.k, tc.groupSize)

			if plan.SplitK != tc.wantSplitK {
				t.Errorf("SplitK: got %d, want %d", plan.SplitK, tc.wantSplitK)
			}
			if plan.TotalTGs != tc.wantTotalTGs {
				t.Errorf("TotalTGs: got %d, want %d", plan.TotalTGs, tc.wantTotalTGs)
			}
			if plan.KPartitionSize != tc.wantKPartition {
				t.Errorf("KPartitionSize: got %d, want %d", plan.KPartitionSize, tc.wantKPartition)
			}

			// Strict divisibility invariant check:
			if tc.k > 0 && tc.groupSize > 0 && plan.SplitK > 1 {
				if tc.k%(plan.SplitK*tc.groupSize) != 0 {
					t.Errorf("Divisibility violated: K (%d) not divisible by SplitK (%d) * groupSize (%d)",
						tc.k, plan.SplitK, tc.groupSize)
				}
				if plan.KPartitionSize%tc.groupSize != 0 {
					t.Errorf("Partition size (%d) not aligned to groupSize (%d)",
						plan.KPartitionSize, tc.groupSize)
				}
			}
		})
	}
}

// TestSplitKNumericalParity verifies numerical parity between Split-K and single-pass reference (max diff < 1e-5).
func TestSplitKNumericalParity(t *testing.T) {
	testCases := []struct {
		name string
		m    int
		n    int
		k    int
	}{
		{"decode_m1_n2048_k4096", 1, 2048, 4096},
		{"batch_m4_n1024_k2048", 4, 1024, 2048},
		{"prefill_m32_n512_k1024", 32, 512, 1024},
	}

	formats := []struct {
		name   string
		format QuantFormat
	}{
		{"Q8_0", QuantFormatQ8_0},
		{"Q4_0", QuantFormatQ4_0},
	}

	for _, fmtCase := range formats {
		for _, tc := range testCases {
			t.Run(fmt.Sprintf("%s_%s", fmtCase.name, tc.name), func(t *testing.T) {
				seed := int64(1337 + tc.m*100 + tc.n + tc.k)
				x := generateTestFloats(tc.m*tc.k, seed, 0.5)
				rawW := generateTestFloats(tc.n*tc.k, seed+1, 0.1)

				var W *QuantizedMatrix
				var err error
				if fmtCase.format == QuantFormatQ8_0 {
					W, err = QuantizeQ8_0(rawW, tc.n, tc.k)
				} else {
					W, err = QuantizeQ4_0(rawW, tc.n, tc.k)
				}
				if err != nil {
					t.Fatalf("Quantize %s failed: %v", fmtCase.name, err)
				}

				// 1. Single-pass reference execution (SplitK = 1)
				singlePlan := SplitKConfig{
					SplitK:         1,
					KPartitionSize: tc.k,
					MTiles:         (tc.m + 31) / 32,
					NTiles:         (tc.n + 31) / 32,
					CurrentTGs:     ((tc.m + 31) / 32) * ((tc.n + 31) / 32),
					TotalTGs:       ((tc.m + 31) / 32) * ((tc.n + 31) / 32),
				}
				ySingle := make([]float32, tc.m*tc.n)
				if err := ExecuteSplitKRefWithConfig(x, W, tc.m, tc.n, tc.k, singlePlan, ySingle); err != nil {
					t.Fatalf("ExecuteSplitKRefWithConfig single-pass failed: %v", err)
				}

				// 2. Split-K multi-slice reference execution
				splitPlan := CalculateSplitK(tc.m, tc.n, tc.k, 32)
				if splitPlan.SplitK < 2 {
					splitPlan.SplitK = 4
					splitPlan.KPartitionSize = tc.k / 4
					splitPlan.TotalTGs = splitPlan.CurrentTGs * 4
				}
				ySplitK := make([]float32, tc.m*tc.n)
				if err := ExecuteSplitKRefWithConfig(x, W, tc.m, tc.n, tc.k, splitPlan, ySplitK); err != nil {
					t.Fatalf("ExecuteSplitKRefWithConfig split-k failed: %v", err)
				}

				// Check CPU parity: Split-K vs Single-Pass
				diffRef := computeMaxDiff(ySingle, ySplitK)
				if diffRef >= 1e-5 {
					t.Fatalf("CPU Parity violation for %s %s: maxDiff = %e (want < 1e-5)",
						fmtCase.name, tc.name, diffRef)
				}
				t.Logf("PASS [CPU Parity %s %s]: splitK=%d, maxDiff=%e (< 1e-5)",
					fmtCase.name, tc.name, splitPlan.SplitK, diffRef)

				// 3. GPU execution test if Metal is available
				if MetalSplitKAvailable() {
					yGPU := make([]float32, tc.m*tc.n)
					stats, err := ExecuteSplitK(x, W, tc.m, tc.n, tc.k, yGPU)
					if err != nil {
						t.Fatalf("ExecuteSplitK GPU failed: %v", err)
					}
					if !stats.GPUDispatched {
						t.Logf("Note: GPU execution skipped/fell back to CPU for %s %s", fmtCase.name, tc.name)
					} else {
						diffGPU := computeMaxDiff(ySingle, yGPU)
						if diffGPU >= 1e-5 {
							t.Fatalf("GPU Parity violation for %s %s: maxDiff = %e (want < 1e-5)",
								fmtCase.name, tc.name, diffGPU)
						}
						t.Logf("PASS [GPU Parity %s %s]: maxDiff=%e (< 1e-5), latency=%v",
							fmtCase.name, tc.name, diffGPU, stats.Duration)
					}
				}
			})
		}
	}
}

// TestSplitKQuantizedGEMM_Throughput benchmarks and witnesses latency and threadgroup saturation.
func TestSplitKQuantizedGEMM_Throughput(t *testing.T) {
	configs := []struct {
		name string
		m    int
		n    int
		k    int
	}{
		{"decode_2k", 1, 2048, 4096},
		{"decode_4k", 1, 4096, 4096},
	}

	for _, cfg := range configs {
		t.Run(cfg.name, func(t *testing.T) {
			m, n, k := cfg.m, cfg.n, cfg.k
			x := generateTestFloats(m*k, 42, 0.5)
			rawW := generateTestFloats(n*k, 43, 0.1)

			W, err := QuantizeQ8_0(rawW, n, k)
			if err != nil {
				t.Fatalf("QuantizeQ8_0 failed: %v", err)
			}

			plan := CalculateSplitK(m, n, k, 32)
			outSplitK := make([]float32, m*n)
			outSingle := make([]float32, m*n)

			singlePlan := SplitKConfig{
				SplitK:         1,
				KPartitionSize: k,
				MTiles:         (m + 31) / 32,
				NTiles:         (n + 31) / 32,
				CurrentTGs:     ((m + 31) / 32) * ((n + 31) / 32),
				TotalTGs:       ((m + 31) / 32) * ((n + 31) / 32),
			}

			// Warmup
			_ = ExecuteSplitKRefWithConfig(x, W, m, n, k, singlePlan, outSingle)
			_ = ExecuteSplitKRefWithConfig(x, W, m, n, k, plan, outSplitK)

			const iters = 5

			// Measure Single-Pass
			startSingle := time.Now()
			for i := 0; i < iters; i++ {
				_ = ExecuteSplitKRefWithConfig(x, W, m, n, k, singlePlan, outSingle)
			}
			singleDur := time.Since(startSingle) / iters

			// Measure Split-K
			startSplit := time.Now()
			for i := 0; i < iters; i++ {
				_ = ExecuteSplitKRefWithConfig(x, W, m, n, k, plan, outSplitK)
			}
			splitDur := time.Since(startSplit) / iters

			// Calculate GFLOPS (2 * M * N * K)
			flops := 2.0 * float64(m) * float64(n) * float64(k)
			singleGFLOPS := flops / (singleDur.Seconds() * 1e9)
			splitGFLOPS := flops / (splitDur.Seconds() * 1e9)

			t.Logf("=== Throughput Witness [%s: M=%d, N=%d, K=%d] ===", cfg.name, m, n, k)
			t.Logf("  Single-Pass: Threadgroups = %4d, Latency = %9v, GFLOPS = %6.2f",
				singlePlan.TotalTGs, singleDur, singleGFLOPS)
			t.Logf("  Split-K:     Threadgroups = %4d, Latency = %9v, GFLOPS = %6.2f",
				plan.TotalTGs, splitDur, splitGFLOPS)
			t.Logf("  Saturation Gain: %dx threadgroups (%d -> %d active threadgroups)",
				plan.TotalTGs/singlePlan.TotalTGs, singlePlan.TotalTGs, plan.TotalTGs)

			// GPU benchmark if available
			if MetalSplitKAvailable() {
				outGPU := make([]float32, m*n)
				// Warmup GPU
				_, _ = ExecuteSplitK(x, W, m, n, k, outGPU)
				startGPU := time.Now()
				for i := 0; i < iters; i++ {
					_, _ = ExecuteSplitK(x, W, m, n, k, outGPU)
				}
				gpuDur := time.Since(startGPU) / iters
				gpuGFLOPS := flops / (gpuDur.Seconds() * 1e9)
				t.Logf("  Metal GPU:   Threadgroups = %4d, Latency = %9v, GFLOPS = %6.2f (GPU Dispatched)",
					plan.TotalTGs, gpuDur, gpuGFLOPS)
			}

			// Parity verification
			diff := computeMaxDiff(outSingle, outSplitK)
			if diff >= 1e-5 {
				t.Errorf("Parity divergence in throughput test: maxDiff = %e (want < 1e-5)", diff)
			}
		})
	}
}

// TestSplitK is the master verification test exercising plan, parity, dequantization, and dispatch.
func TestSplitK(t *testing.T) {
	t.Run("Config", TestSplitKConfig)
	t.Run("NumericalParity", TestSplitKNumericalParity)
	t.Run("ThroughputWitness", TestSplitKQuantizedGEMM_Throughput)

	t.Run("DynamicDispatchThresholds", func(t *testing.T) {
		// Verify large M*N routes to single-pass fallback
		x := generateTestFloats(1024*1024, 10, 1.0)
		w := generateTestFloats(1024*1024, 11, 0.5)
		qw, err := QuantizeQ8_0(w, 1024, 1024)
		if err != nil {
			t.Fatalf("QuantizeQ8_0 failed: %v", err)
		}
		out := make([]float32, 1024*1024)
		stats, err := ExecuteSplitK(x, qw, 1024, 1024, 1024, out)
		if err != nil {
			t.Fatalf("ExecuteSplitK failed: %v", err)
		}
		if !stats.SinglePass {
			t.Errorf("Expected SinglePass=true for large M*N (1024x1024), got %v", stats.SinglePass)
		}
		if stats.SplitK != 1 {
			t.Errorf("Expected SplitK=1 for large M*N, got %d", stats.SplitK)
		}
	})

	t.Run("DequantRoundTrip", func(t *testing.T) {
		n, k := 64, 128
		orig := generateTestFloats(n*k, 99, 1.0)

		// Q8_0 roundtrip
		q8, err := QuantizeQ8_0(orig, n, k)
		if err != nil {
			t.Fatalf("QuantizeQ8_0 failed: %v", err)
		}
		rec8 := make([]float32, n*k)
		if err := DequantizeQ8_0(rec8, q8.Data, n, k); err != nil {
			t.Fatalf("DequantizeQ8_0 failed: %v", err)
		}
		diff8 := computeMaxDiff(orig, rec8)
		if diff8 > 0.05 { // Q8 quantization error should be very small
			t.Errorf("Q8_0 roundtrip error too high: %e", diff8)
		}
		t.Logf("Q8_0 dequant roundtrip maxDiff: %e", diff8)

		// Q4_0 roundtrip
		q4, err := QuantizeQ4_0(orig, n, k)
		if err != nil {
			t.Fatalf("QuantizeQ4_0 failed: %v", err)
		}
		rec4 := make([]float32, n*k)
		if err := DequantizeQ4_0(rec4, q4.Data, n, k); err != nil {
			t.Fatalf("DequantizeQ4_0 failed: %v", err)
		}
		diff4 := computeMaxDiff(orig, rec4)
		if diff4 > 0.20 { // Q4 quantization error bounded
			t.Errorf("Q4_0 roundtrip error too high: %e", diff4)
		}
		t.Logf("Q4_0 dequant roundtrip maxDiff: %e", diff4)
	})
}
