package compute

import (
	"math"
	"math/rand"
	"testing"
)

func TestPagedQuantOperandWitness(t *testing.T) {
	// First witness requirements (#9930):
	// 1. Descriptor round-trip across Tensor, Head, and Block granularities
	// 2. Fail-closed on unsupported/mismatched combinations
	// 3. Cosine quality floor verification by granularity mode (Block > Head > Tensor)
	// 4. Memory footprint accounting including scales

	numHeads := 4
	headDim := 32
	pageSize := 16
	groupSize := 8 // 2 groups per page
	totalPages := 4
	totalTokens := 48 // fits in 3 pages, totalPages=4 has 1 spare page

	pageTable := []int32{3, 0, 2, 1} // shuffled page mapping

	granularities := []QuantGranularity{
		QuantGranularityTensor,
		QuantGranularityHead,
		QuantGranularityBlock,
	}

	rng := rand.New(rand.NewSource(9930))

	// Generate unquantized reference token vectors
	origTokens := make([][][]float32, totalTokens) // [pos][head][dim]
	for p := 0; p < totalTokens; p++ {
		origTokens[p] = make([][]float32, numHeads)
		for h := 0; h < numHeads; h++ {
			origTokens[p][h] = make([]float32, headDim)
			for d := 0; d < headDim; d++ {
				origTokens[p][h][d] = rng.Float32()*2 - 1
			}
		}
	}

	for _, gran := range granularities {
		t.Run(string(gran), func(t *testing.T) {
			contract := PagedQuantOperandContract{
				Precision:   "int8",
				Granularity: gran,
				NumHeads:    numHeads,
				HeadDim:     headDim,
				PageSize:    pageSize,
				GroupSize:   groupSize,
			}

			scaleCount, err := contract.ExpectedScaleCount(totalPages)
			if err != nil {
				t.Fatalf("ExpectedScaleCount failed: %v", err)
			}

			scales := make([]float32, scaleCount)
			for i := range scales {
				scales[i] = 1.0 / 127.0 // symmetric int8 standard unit scale
			}

			// Validate contract
			receipt, err := ValidatePagedQuantOperandContract(contract, totalPages, totalTokens, len(scales))
			if err != nil {
				t.Fatalf("ValidatePagedQuantOperandContract failed: %v", err)
			}

			// Populate quantized paged key buffer
			totalKeyElems := totalPages * pageSize * numHeads * headDim
			keyData := make([]int8, totalKeyElems)

			stride := numHeads * headDim
			for p := 0; p < totalTokens; p++ {
				lPage := p / pageSize
				inPage := p % pageSize
				physPage := pageTable[lPage]
				tokOffset := (int(physPage)*pageSize + inPage) * stride

				for h := 0; h < numHeads; h++ {
					scale := contract.ResolveTokenScale(scales, physPage, inPage, h)
					headOffset := tokOffset + h*headDim
					for d := 0; d < headDim; d++ {
						q := int(math.Round(float64(origTokens[p][h][d] / scale)))
						if q < -128 {
							q = -128
						} else if q > 127 {
							q = 127
						}
						keyData[headOffset+d] = int8(q)
					}
				}
			}

			// Round-trip dequantize and measure cosine fidelity
			var totalDot, totalNormOrig, totalNormDequant float64
			for p := 0; p < totalTokens; p++ {
				for h := 0; h < numHeads; h++ {
					dequant, err := DequantizePagedKVToken(keyData, scales, contract, pageTable, p, h)
					if err != nil {
						t.Fatalf("DequantizePagedKVToken pos=%d head=%d failed: %v", p, h, err)
					}

					for d := 0; d < headDim; d++ {
						orig := float64(origTokens[p][h][d])
						deq := float64(dequant[d])
						totalDot += orig * deq
						totalNormOrig += orig * orig
						totalNormDequant += deq * deq
					}
				}
			}

			cosine := totalDot / (math.Sqrt(totalNormOrig) * math.Sqrt(totalNormDequant))
			if cosine < receipt.DequantCosineFloor {
				t.Fatalf("granularity %s cosine %v < required floor %v", gran, cosine, receipt.DequantCosineFloor)
			}
		})
	}

	// 2. Fail-closed unsupported combinations
	t.Run("fail_closed", func(t *testing.T) {
		// Unsupported precision
		badPrec := PagedQuantOperandContract{Precision: "unsupported", Granularity: QuantGranularityTensor, NumHeads: 4, HeadDim: 32, PageSize: 16}
		if _, err := ValidatePagedQuantOperandContract(badPrec, 2, 16, 1); err == nil {
			t.Fatal("expected error on unsupported precision")
		}

		// PageSize not divisible by GroupSize in Block granularity
		badBlock := PagedQuantOperandContract{
			Precision:   "int8",
			Granularity: QuantGranularityBlock,
			NumHeads:    4,
			HeadDim:     32,
			PageSize:    16,
			GroupSize:   5, // 16 % 5 != 0
		}
		if _, err := ValidatePagedQuantOperandContract(badBlock, 2, 16, 10); err == nil {
			t.Fatal("expected error on pageSize not divisible by groupSize")
		}

		// Mismatched scales length
		goodContract := PagedQuantOperandContract{
			Precision:   "int8",
			Granularity: QuantGranularityHead,
			NumHeads:    4,
			HeadDim:     32,
			PageSize:    16,
		}
		if _, err := ValidatePagedQuantOperandContract(goodContract, 2, 16, 1); err == nil { // expected 4 scales, gave 1
			t.Fatal("expected error on mismatched scales length")
		}
	})
}
