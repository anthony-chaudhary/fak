package compute

import (
	"fmt"
	"math"
	"math/rand"
	"testing"
)

func referenceUnfusedRMSNormResidual(x, residualIn, weight []float32, rows, width int, eps float32) ([]float32, []float32) {
	total := rows * width
	resOut := make([]float32, total)
	for i := 0; i < total; i++ {
		resOut[i] = x[i] + residualIn[i]
	}

	normed := make([]float32, total)
	for r := 0; r < rows; r++ {
		rowOff := r * width
		var ss float64
		for i := 0; i < width; i++ {
			v := float64(resOut[rowOff+i])
			ss += v * v
		}
		inv := float32(1.0 / math.Sqrt(ss/float64(width)+float64(eps)))
		for i := 0; i < width; i++ {
			normed[rowOff+i] = resOut[rowOff+i] * inv * weight[i]
		}
	}
	return resOut, normed
}

func TestFusedRMSNormResidualAddWitness(t *testing.T) {
	// First witness requirements (#9941):
	// 1. Exact ordered reference matching unfused implementation
	// 2. Exact parity across aliasing permutations (out-of-place, in-place on residual, in-place on x)
	// 3. FP32 accumulation
	// 4. Ragged widths & typical model widths
	// 5. Fail-closed on invalid shapes, eps, non-finite values
	// 6. Launch & DRAM reduction receipts

	widths := []int{17, 32, 64, 128, 256, 512, 1024, 2048, 4096}
	eps := float32(1e-6)

	rng := rand.New(rand.NewSource(42))

	for _, width := range widths {
		rows := 4
		total := rows * width

		x := make([]float32, total)
		residualIn := make([]float32, total)
		weight := make([]float32, width)

		for i := range x {
			x[i] = rng.Float32()*2 - 1
			residualIn[i] = rng.Float32()*2 - 1
		}
		for i := range weight {
			weight[i] = rng.Float32()*0.5 + 0.5
		}

		wantRes, wantNormed := referenceUnfusedRMSNormResidual(x, residualIn, weight, rows, width, eps)

		// 1. Out-of-place test
		t.Run(fmt.Sprintf("out_of_place_w%d", width), func(t *testing.T) {
			gotRes := make([]float32, total)
			gotNormed := make([]float32, total)

			receipt, err := FusedRMSNormResidualAdd(x, residualIn, weight, gotRes, gotNormed, rows, width, eps)
			if err != nil {
				t.Fatalf("FusedRMSNormResidualAdd failed: %v", err)
			}

			if receipt.InPlaceResidual {
				t.Fatal("expected InPlaceResidual = false for distinct slices")
			}
			if receipt.SavedLaunches != 1 {
				t.Fatalf("expected SavedLaunches = 1, got %d", receipt.SavedLaunches)
			}
			if receipt.EliminatedDRAMMB <= 0 {
				t.Fatalf("expected EliminatedDRAMMB > 0, got %v", receipt.EliminatedDRAMMB)
			}

			for i := range wantRes {
				if math.Abs(float64(gotRes[i]-wantRes[i])) > 1e-6 {
					t.Fatalf("residual mismatch at %d: got %v, want %v", i, gotRes[i], wantRes[i])
				}
				if math.Abs(float64(gotNormed[i]-wantNormed[i])) > 1e-6 {
					t.Fatalf("normed mismatch at %d: got %v, want %v", i, gotNormed[i], wantNormed[i])
				}
			}
		})

		// 2. In-place aliasing on residualIn (residualIn is updated in-place)
		t.Run(fmt.Sprintf("in_place_residual_w%d", width), func(t *testing.T) {
			inPlaceRes := append([]float32(nil), residualIn...)
			gotNormed := make([]float32, total)

			receipt, err := FusedRMSNormResidualAdd(x, inPlaceRes, weight, inPlaceRes, gotNormed, rows, width, eps)
			if err != nil {
				t.Fatalf("FusedRMSNormResidualAdd failed: %v", err)
			}

			if !receipt.InPlaceResidual {
				t.Fatal("expected InPlaceResidual = true for aliased residualOut == residualIn")
			}

			for i := range wantRes {
				if math.Abs(float64(inPlaceRes[i]-wantRes[i])) > 1e-6 {
					t.Fatalf("aliased residual mismatch at %d: got %v, want %v", i, inPlaceRes[i], wantRes[i])
				}
				if math.Abs(float64(gotNormed[i]-wantNormed[i])) > 1e-6 {
					t.Fatalf("normed mismatch at %d: got %v, want %v", i, gotNormed[i], wantNormed[i])
				}
			}
		})

		// 3. In-place aliasing on x (x is updated in-place)
		t.Run(fmt.Sprintf("in_place_x_w%d", width), func(t *testing.T) {
			inPlaceX := append([]float32(nil), x...)
			gotNormed := make([]float32, total)

			receipt, err := FusedRMSNormResidualAdd(inPlaceX, residualIn, weight, inPlaceX, gotNormed, rows, width, eps)
			if err != nil {
				t.Fatalf("FusedRMSNormResidualAdd failed: %v", err)
			}

			if !receipt.InPlaceResidual {
				t.Fatal("expected InPlaceResidual = true for aliased residualOut == x")
			}

			for i := range wantRes {
				if math.Abs(float64(inPlaceX[i]-wantRes[i])) > 1e-6 {
					t.Fatalf("aliased x mismatch at %d: got %v, want %v", i, inPlaceX[i], wantRes[i])
				}
				if math.Abs(float64(gotNormed[i]-wantNormed[i])) > 1e-6 {
					t.Fatalf("normed mismatch at %d: got %v, want %v", i, gotNormed[i], wantNormed[i])
				}
			}
		})
	}

	// 5. Fail-closed validations
	t.Run("fail_closed_validations", func(t *testing.T) {
		dummy := make([]float32, 16)
		w := make([]float32, 4)

		// Invalid rows/width
		if _, err := FusedRMSNormResidualAdd(dummy, dummy, w, dummy, dummy, 0, 4, 1e-6); err == nil {
			t.Fatal("expected error on rows = 0")
		}

		// Invalid epsilon
		if _, err := FusedRMSNormResidualAdd(dummy, dummy, w, dummy, dummy, 4, 4, 0); err == nil {
			t.Fatal("expected error on eps = 0")
		}
		if _, err := FusedRMSNormResidualAdd(dummy, dummy, w, dummy, dummy, 4, 4, float32(math.NaN())); err == nil {
			t.Fatal("expected error on eps = NaN")
		}

		// Mismatched lengths
		short := make([]float32, 10)
		if _, err := FusedRMSNormResidualAdd(short, dummy, w, dummy, dummy, 4, 4, 1e-6); err == nil {
			t.Fatal("expected error on mismatched length")
		}

		// Non-finite input
		withNaN := append([]float32(nil), dummy...)
		withNaN[3] = float32(math.NaN())
		if _, err := FusedRMSNormResidualAdd(withNaN, dummy, w, dummy, dummy, 4, 4, 1e-6); err == nil {
			t.Fatal("expected error on NaN input")
		}
	})
}
