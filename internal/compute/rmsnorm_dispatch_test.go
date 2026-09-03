package compute

import (
	"fmt"
	"math"
	"math/rand"
	"testing"
)

func referencePyTorchRMSNorm(x, weight []float32, rows, width int, eps float32) []float32 {
	total := rows * width
	out := make([]float32, total)
	for r := 0; r < rows; r++ {
		rowOff := r * width
		var ss float64
		for i := 0; i < width; i++ {
			v := float64(x[rowOff+i])
			ss += v * v
		}
		inv := float32(1.0 / math.Sqrt(ss/float64(width)+float64(eps)))
		for i := 0; i < width; i++ {
			out[rowOff+i] = x[rowOff+i] * inv * weight[i]
		}
	}
	return out
}

func TestRMSNormDispatchWitness(t *testing.T) {
	// First witness requirements (#9940):
	// 1. Width/row sweep (including Qwen3.8 widths: 128, 256, 512, 1024, 2048, 4096, 5120 and ragged tails: 17, 33, 100, 127)
	// 2. Exact match against F32 reference oracle
	// 3. Deterministic repeated executions
	// 4. Verification of warp-per-row dispatch for widths <= 1024 and block-per-row for widths > 1024

	testWidths := []struct {
		width    int
		wantKind RMSNormDispatchKind
		isWarp   bool
	}{
		{width: 17, wantKind: RMSNormDispatchWarpPerRow, isWarp: true},
		{width: 32, wantKind: RMSNormDispatchWarpPerRow, isWarp: true},
		{width: 33, wantKind: RMSNormDispatchWarpPerRow, isWarp: true},
		{width: 64, wantKind: RMSNormDispatchWarpPerRow, isWarp: true},
		{width: 100, wantKind: RMSNormDispatchWarpPerRow, isWarp: true},
		{width: 127, wantKind: RMSNormDispatchWarpPerRow, isWarp: true},
		{width: 128, wantKind: RMSNormDispatchWarpPerRow, isWarp: true},
		{width: 256, wantKind: RMSNormDispatchWarpPerRow, isWarp: true},
		{width: 512, wantKind: RMSNormDispatchWarpPerRow, isWarp: true},
		{width: 1024, wantKind: RMSNormDispatchWarpPerRow, isWarp: true}, // crossover boundary
		{width: 2048, wantKind: RMSNormDispatchBlockPerRow, isWarp: false},
		{width: 4096, wantKind: RMSNormDispatchBlockPerRow, isWarp: false},
		{width: 5120, wantKind: RMSNormDispatchBlockPerRow, isWarp: false},
	}

	eps := float32(1e-6)
	rng := rand.New(rand.NewSource(1337))

	for _, tc := range testWidths {
		for _, rows := range []int{1, 4, 16} {
			t.Run(fmt.Sprintf("w%d_rows%d", tc.width, rows), func(t *testing.T) {
				total := rows * tc.width
				x := make([]float32, total)
				weight := make([]float32, tc.width)

				for i := range x {
					x[i] = rng.Float32()*2 - 1
				}
				for i := range weight {
					weight[i] = rng.Float32()*0.5 + 0.5
				}

				want := referencePyTorchRMSNorm(x, weight, rows, tc.width, eps)
				got := make([]float32, total)

				// First run
				receipt, err := RMSNormDispatched(x, weight, got, rows, tc.width, eps)
				if err != nil {
					t.Fatalf("RMSNormDispatched failed: %v", err)
				}

				// Verify dispatch selection
				if receipt.DispatchKind != tc.wantKind {
					t.Fatalf("expected dispatch %v, got %v", tc.wantKind, receipt.DispatchKind)
				}
				if receipt.BarrierFree != tc.isWarp {
					t.Fatalf("expected BarrierFree=%t, got %t", tc.isWarp, receipt.BarrierFree)
				}

				// Verify oracle parity
				for i := range want {
					if math.Abs(float64(got[i]-want[i])) > 1e-6 {
						t.Fatalf("mismatch at %d: got %v, want %v", i, got[i], want[i])
					}
				}

				// Deterministic repeated execution check
				got2 := make([]float32, total)
				receipt2, err := RMSNormDispatched(x, weight, got2, rows, tc.width, eps)
				if err != nil {
					t.Fatalf("repeat run failed: %v", err)
				}
				if receipt != receipt2 {
					t.Fatalf("receipt not deterministic: %+v vs %+v", receipt, receipt2)
				}
				for i := range got {
					if got[i] != got2[i] {
						t.Fatalf("repeated execution bit-drift at index %d", i)
					}
				}
			})
		}
	}
}
