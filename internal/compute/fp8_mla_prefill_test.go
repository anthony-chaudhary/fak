package compute

import (
	"math"
	"testing"
)

func referenceF32MLAPrefill(q, k []float32, numTokens, numHeads, headDim int) []float32 {
	stride := numHeads * headDim
	out := make([]float32, numTokens*numTokens*numHeads)

	for i := 0; i < numTokens; i++ {
		for j := 0; j < numTokens; j++ {
			for h := 0; h < numHeads; h++ {
				var dot float64
				for d := 0; d < headDim; d++ {
					dot += float64(q[i*stride+h*headDim+d]) * float64(k[j*stride+h*headDim+d])
				}
				out[(i*numTokens+j)*numHeads+h] = float32(dot)
			}
		}
	}
	return out
}

func TestPerTokenScaleFP8MLAPrefillWitness(t *testing.T) {
	// First witness requirements (#9932):
	// 1. Outlier-heavy tokens (e.g. token 0 has 100x magnitude of token 1).
	// 2. Exact match against reference oracle via per-token dequantization scale.
	// 3. High cosine fidelity (> 0.999) on normal tokens despite presence of massive outliers.
	// 4. Fail-closed on scale corruption (<= 0, NaN, Inf).

	numTokens := 4
	numHeads := 2
	headDim := 8
	stride := numHeads * headDim

	// Token 0 is an extreme outlier (magnitude ~100.0)
	// Tokens 1, 2, 3 are normal magnitude (~1.0)
	qFP8 := make([]int8, numTokens*stride)
	kFP8 := make([]int8, numTokens*stride)
	qScales := []float32{1.0, 0.01, 0.01, 0.01} // token 0 scale is 100x larger!
	kScales := []float32{1.0, 0.01, 0.01, 0.01}

	// Reference unquantized F32 vectors
	qF32 := make([]float32, numTokens*stride)
	kF32 := make([]float32, numTokens*stride)

	for i := 0; i < numTokens; i++ {
		for h := 0; h < numHeads; h++ {
			for d := 0; d < headDim; d++ {
				idx := i*stride + h*headDim + d
				valInt := int8((d + 1) * 10)
				if i == 0 {
					valInt = int8(100) // outlier
				}
				qFP8[idx] = valInt
				kFP8[idx] = valInt

				qF32[idx] = float32(valInt) * qScales[i]
				kF32[idx] = float32(valInt) * kScales[i]
			}
		}
	}

	wantScores := referenceF32MLAPrefill(qF32, kF32, numTokens, numHeads, headDim)

	gotScores := make([]float32, numTokens*numTokens*numHeads)
	receipt, err := ExecutePerTokenScaleFP8MLAPrefill(
		qFP8, kFP8, qScales, kScales, numTokens, numHeads, headDim, gotScores,
	)
	if err != nil {
		t.Fatalf("ExecutePerTokenScaleFP8MLAPrefill failed: %v", err)
	}

	if !receipt.OutlierPreserved {
		t.Fatal("expected OutlierPreserved = true")
	}

	// 2 & 3. Verify exact score match with reference
	var totalDot, normGot, normWant float64
	for i := range wantScores {
		g := float64(gotScores[i])
		w := float64(wantScores[i])
		totalDot += g * w
		normGot += g * g
		normWant += w * w

		if math.Abs(float64(gotScores[i]-wantScores[i])) > 1e-4 {
			t.Fatalf("score mismatch at %d: got %v, want %v", i, gotScores[i], wantScores[i])
		}
	}

	cosine := totalDot / (math.Sqrt(normGot) * math.Sqrt(normWant))
	if math.Abs(1.0-cosine) > 1e-6 {
		t.Fatalf("cosine fidelity %v < 1.0", cosine)
	}

	// 4. Fail-closed on scale corruption
	corruptedScales := []float32{1.0, -0.5, 0.01, 0.01} // negative scale
	if _, err := ExecutePerTokenScaleFP8MLAPrefill(qFP8, kFP8, corruptedScales, kScales, numTokens, numHeads, headDim, gotScores); err == nil {
		t.Fatal("expected error on negative scale")
	}

	nanScales := []float32{1.0, float32(math.NaN()), 0.01, 0.01}
	if _, err := ExecutePerTokenScaleFP8MLAPrefill(qFP8, kFP8, nanScales, kScales, numTokens, numHeads, headDim, gotScores); err == nil {
		t.Fatal("expected error on NaN scale")
	}
}
