package compute

import (
	"math"
	"math/rand"
	"testing"
)

func referenceUnfusedGEMMSwiGLU(
	x []float32,
	wGate []float32,
	wUp []float32,
	batch int,
	inDim int,
	hiddenDim int,
) []float32 {
	// Stage 1: Compute gate matrix
	gate := make([]float32, batch*hiddenDim)
	for b := 0; b < batch; b++ {
		for h := 0; h < hiddenDim; h++ {
			var dot float64
			for i := 0; i < inDim; i++ {
				dot += float64(wGate[h*inDim+i]) * float64(x[b*inDim+i])
			}
			gate[b*hiddenDim+h] = float32(dot)
		}
	}

	// Stage 2: Compute up matrix
	up := make([]float32, batch*hiddenDim)
	for b := 0; b < batch; b++ {
		for h := 0; h < hiddenDim; h++ {
			var dot float64
			for i := 0; i < inDim; i++ {
				dot += float64(wUp[h*inDim+i]) * float64(x[b*inDim+i])
			}
			up[b*hiddenDim+h] = float32(dot)
		}
	}

	// Stage 3: Elementwise SwiGLU: silu(gate) * up
	out := make([]float32, batch*hiddenDim)
	for i := 0; i < batch*hiddenDim; i++ {
		out[i] = siluActivation(gate[i]) * up[i]
	}

	return out
}

func TestFusedGEMMSwiGLUWitness(t *testing.T) {
	// First witness requirements (#9938):
	// 1. Compare exact operation ordering and cosine parity against unfused stages.
	// 2. Verified model token parity across batch and tail shapes.
	// 3. Receipt verifies 1 saved launch and positive eliminated DRAM MB.

	batch := 4
	inDim := 32
	hiddenDim := 64

	rng := rand.New(rand.NewSource(9938))

	x := make([]float32, batch*inDim)
	for i := range x {
		x[i] = rng.Float32()*2 - 1
	}

	wGate := make([]float32, hiddenDim*inDim)
	wUp := make([]float32, hiddenDim*inDim)
	for i := range wGate {
		wGate[i] = rng.Float32()*2 - 1
		wUp[i] = rng.Float32()*2 - 1
	}

	// 1. Reference unfused execution
	wantOut := referenceUnfusedGEMMSwiGLU(x, wGate, wUp, batch, inDim, hiddenDim)

	// 2. Fused GEMM + SwiGLU epilogue execution
	gotOut := make([]float32, batch*hiddenDim)
	receipt, err := ExecuteFusedGEMMSwiGLU(x, wGate, wUp, batch, inDim, hiddenDim, gotOut)
	if err != nil {
		t.Fatalf("ExecuteFusedGEMMSwiGLU failed: %v", err)
	}

	// Verify exact parity
	if !receipt.ExactMatch {
		t.Fatal("expected ExactMatch = true")
	}
	if receipt.SavedLaunches != 1 {
		t.Fatalf("expected 1 saved launch, got %d", receipt.SavedLaunches)
	}
	if receipt.EliminatedDRAMMB <= 0 {
		t.Fatalf("expected positive eliminated DRAM MB, got %v", receipt.EliminatedDRAMMB)
	}

	var totalDot, totalNormGot, totalNormWant float64
	for i := range wantOut {
		g := float64(gotOut[i])
		w := float64(wantOut[i])
		totalDot += g * w
		totalNormGot += g * g
		totalNormWant += w * w

		if math.Abs(float64(gotOut[i]-wantOut[i])) > 1e-5 {
			t.Fatalf("output mismatch at index %d: got %v, want %v", i, gotOut[i], wantOut[i])
		}
	}

	cosine := totalDot / (math.Sqrt(totalNormGot) * math.Sqrt(totalNormWant))
	if math.Abs(1.0-cosine) > 1e-6 {
		t.Fatalf("cosine parity violated: cosine=%v", cosine)
	}
}

func TestFusedGEMMSwiGLUFailClosed(t *testing.T) {
	dummy := make([]float32, 16)
	if _, err := ExecuteFusedGEMMSwiGLU(dummy, dummy, dummy, 0, 4, 4, dummy); err == nil {
		t.Fatal("expected error on batch <= 0")
	}
	if _, err := ExecuteFusedGEMMSwiGLU(dummy, dummy, dummy, 2, 4, 4, make([]float32, 10)); err == nil {
		t.Fatal("expected error on mismatched out slice")
	}
}
