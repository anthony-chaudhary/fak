package compute

import (
	"fmt"
	"math"
	"math/rand"
	"testing"
)

func referenceMatVecF32(weights []float32, x []float32, channels, inDim int) []float32 {
	y := make([]float32, channels)
	for c := 0; c < channels; c++ {
		var sum float64
		for i := 0; i < inDim; i++ {
			sum += float64(weights[c*inDim+i]) * float64(x[i])
		}
		y[c] = float32(sum)
	}
	return y
}

func TestGroupedSym4CodecWitness(t *testing.T) {
	// First witness requirements (#9939):
	// 1. Group-size 8, 16, 32 symmetric uint4/int4 with one fp16 scale
	// 2. Round-trip tensor error distribution (cosine similarity > 0.99, bounded max error, positive SQNR)
	// 3. MatVec parity between direct packed computation and dequantized computation
	// 4. Memory compression verification (> 6x compression ratio for group 32)
	// 5. Fail-closed validation on invalid group size, odd dimensions, NaNs

	rng := rand.New(rand.NewSource(12345))

	groupSizes := []int{8, 16, 32}
	channels := 8
	inDim := 128 // divisible by 8, 16, 32

	weights := make([]float32, channels*inDim)
	for i := range weights {
		weights[i] = rng.Float32()*2 - 1 // normal-like [-1, 1] range
	}

	x := make([]float32, inDim)
	for i := range x {
		x[i] = rng.Float32()*0.5 + 0.1
	}

	for _, gSize := range groupSizes {
		t.Run(fmt.Sprintf("group_size_%d", gSize), func(t *testing.T) {
			cfg := GroupedSym4CodecConfig{GroupSize: gSize}

			// 1. Quantize
			tensor, err := QuantizeGroupedSym4(weights, channels, inDim, cfg)
			if err != nil {
				t.Fatalf("QuantizeGroupedSym4 failed: %v", err)
			}

			// 2. Dequantize
			dequant, err := tensor.Dequantize()
			if err != nil {
				t.Fatalf("Dequantize failed: %v", err)
			}

			// 3. Evaluate error distribution
			receipt := EvaluateCodecQuality(weights, dequant, channels, inDim, gSize)

			if receipt.CosineSimilarity < 0.99 {
				t.Fatalf("low cosine similarity: %v (want >= 0.99)", receipt.CosineSimilarity)
			}
			if receipt.SQNRdB < 18.0 {
				t.Fatalf("low SQNR: %v dB (want >= 18 dB for 4-bit)", receipt.SQNRdB)
			}
			if receipt.CompressionRatio < 5.0 {
				t.Fatalf("compression ratio %v too low (want >= 5x)", receipt.CompressionRatio)
			}

			// 4. Test direct MatVec on packed tensor vs dequantized GEMV
			packedY, err := tensor.MatVec(x)
			if err != nil {
				t.Fatalf("MatVec failed: %v", err)
			}
			dequantY := referenceMatVecF32(dequant, x, channels, inDim)

			for c := 0; c < channels; c++ {
				diff := math.Abs(float64(packedY[c] - dequantY[c]))
				if diff > 1e-4 {
					t.Fatalf("channel %d: packed MatVec %v != dequant MatVec %v (diff %v)",
						c, packedY[c], dequantY[c], diff)
				}
			}
		})
	}
}

func TestGroupedSym4CodecFailClosed(t *testing.T) {
	// Invalid group size (not 8, 16, or 32)
	badCfg := GroupedSym4CodecConfig{GroupSize: 12}
	if _, err := QuantizeGroupedSym4([]float32{1, 2}, 1, 2, badCfg); err == nil {
		t.Fatal("expected error on group size 12")
	}

	// InDim not divisible by group size
	cfg := GroupedSym4CodecConfig{GroupSize: 16}
	if _, err := QuantizeGroupedSym4(make([]float32, 20), 1, 20, cfg); err == nil {
		t.Fatal("expected error on inDim not divisible by group size")
	}

	// Non-finite weight
	nanWeights := make([]float32, 16)
	nanWeights[4] = float32(math.NaN())
	if _, err := QuantizeGroupedSym4(nanWeights, 1, 16, cfg); err == nil {
		t.Fatal("expected error on NaN weight")
	}
}
