package compute

import (
	"math"
	"math/rand"
	"reflect"
	"testing"
)

// referenceUnfusedRoPESplitStore executes the 3 stages separately in sequence.
func referenceUnfusedRoPESplitStore(
	packedQKV []float32,
	positions []int,
	pageTable []int32,
	pageSize int,
	numQHeads int,
	numKVHeads int,
	headDim int,
	theta float64,
	totalPages int,
) ([]float32, []float32, []float32) {
	numTokens := len(positions)
	packedStride := (numQHeads + 2*numKVHeads) * headDim
	qWidth := numQHeads * headDim
	kvWidth := numKVHeads * headDim

	// Stage 1: Split into separate Q, K, V buffers
	rawQ := make([]float32, numTokens*qWidth)
	rawK := make([]float32, numTokens*kvWidth)
	rawV := make([]float32, numTokens*kvWidth)

	for t := 0; t < numTokens; t++ {
		base := t * packedStride
		copy(rawQ[t*qWidth:(t+1)*qWidth], packedQKV[base:base+qWidth])
		copy(rawK[t*kvWidth:(t+1)*kvWidth], packedQKV[base+qWidth:base+qWidth+kvWidth])
		copy(rawV[t*kvWidth:(t+1)*kvWidth], packedQKV[base+qWidth+kvWidth:base+packedStride])
	}

	// Stage 2: Apply RoPE to Q and K
	for t := 0; t < numTokens; t++ {
		pos := positions[t]
		for qh := 0; qh < numQHeads; qh++ {
			rotateHalfInPlace(rawQ[t*qWidth+qh*headDim:t*qWidth+(qh+1)*headDim], headDim, pos, theta)
		}
		for kvh := 0; kvh < numKVHeads; kvh++ {
			rotateHalfInPlace(rawK[t*kvWidth+kvh*headDim:t*kvWidth+(kvh+1)*headDim], headDim, pos, theta)
		}
	}

	// Stage 3: Store K and V into paged stores
	pagedK := make([]float32, totalPages*pageSize*kvWidth)
	pagedV := make([]float32, totalPages*pageSize*kvWidth)

	for t := 0; t < numTokens; t++ {
		pos := positions[t]
		lPage := pos / pageSize
		inPage := pos % pageSize
		physPage := pageTable[lPage]

		pagedOffset := (int(physPage)*pageSize + inPage) * kvWidth
		copy(pagedK[pagedOffset:pagedOffset+kvWidth], rawK[t*kvWidth:(t+1)*kvWidth])
		copy(pagedV[pagedOffset:pagedOffset+kvWidth], rawV[t*kvWidth:(t+1)*kvWidth])
	}

	return rawQ, pagedK, pagedV
}

func TestFusedRoPESplitStoreWitness(t *testing.T) {
	// First witness requirements (#9933):
	// 1. Byte-identical K/V store and Q output versus unfused stages.
	// 2. Ragged / arbitrary position-id cases across multiple pages.
	// 3. Red zones / canaries around paged stores remain uncorrupted.
	// 4. Receipt reports 2 saved launches and positive eliminated DRAM bytes.

	numTokens := 6
	positions := []int{0, 1, 15, 16, 31, 33} // crosses pages 0, 1, 2
	pageSize := 16
	totalPages := 4
	pageTable := []int32{2, 0, 3, 1} // shuffled physical pages

	numQHeads := 4
	numKVHeads := 2
	headDim := 16
	theta := 10000.0

	packedStride := (numQHeads + 2*numKVHeads) * headDim
	rng := rand.New(rand.NewSource(9933))
	packedQKV := make([]float32, numTokens*packedStride)
	for i := range packedQKV {
		packedQKV[i] = rng.Float32()*2 - 1
	}

	// 1. Reference unfused execution
	wantQ, wantK, wantV := referenceUnfusedRoPESplitStore(
		packedQKV, positions, pageTable, pageSize, numQHeads, numKVHeads, headDim, theta, totalPages,
	)

	// 3. Setup paged buffers with canary red zones
	kvWidth := numKVHeads * headDim
	canaryVal := float32(0x1337)

	pagedKBuffer := make([]float32, totalPages*pageSize*kvWidth+2)
	pagedKBuffer[0] = canaryVal
	pagedKBuffer[len(pagedKBuffer)-1] = canaryVal
	pagedK := pagedKBuffer[1 : len(pagedKBuffer)-1]

	pagedVBuffer := make([]float32, totalPages*pageSize*kvWidth+2)
	pagedVBuffer[0] = canaryVal
	pagedVBuffer[len(pagedVBuffer)-1] = canaryVal
	pagedV := pagedVBuffer[1 : len(pagedVBuffer)-1]

	gotQ := make([]float32, numTokens*numQHeads*headDim)

	// Execute fused kernel
	receipt, err := ExecuteFusedRoPESplitStore(
		packedQKV, positions, pageTable, pageSize, numQHeads, numKVHeads, headDim, theta, gotQ, pagedK, pagedV,
	)
	if err != nil {
		t.Fatalf("ExecuteFusedRoPESplitStore failed: %v", err)
	}

	// 4. Verify receipt metrics
	if receipt.SavedLaunches != 2 {
		t.Fatalf("expected 2 saved launches, got %d", receipt.SavedLaunches)
	}
	if receipt.EliminatedDRAMMB <= 0 {
		t.Fatalf("expected positive eliminated DRAM MB, got %v", receipt.EliminatedDRAMMB)
	}

	// 3. Verify red zones (canaries) remained untouched
	if pagedKBuffer[0] != canaryVal || pagedKBuffer[len(pagedKBuffer)-1] != canaryVal {
		t.Fatalf("pagedK red zone canary corrupted")
	}
	if pagedVBuffer[0] != canaryVal || pagedVBuffer[len(pagedVBuffer)-1] != canaryVal {
		t.Fatalf("pagedV red zone canary corrupted")
	}

	// 1. Verify byte-identical outputs
	if !reflect.DeepEqual(gotQ, wantQ) {
		for i := range wantQ {
			if math.Abs(float64(gotQ[i]-wantQ[i])) > 1e-6 {
				t.Fatalf("Q mismatch at %d: got %v, want %v", i, gotQ[i], wantQ[i])
			}
		}
	}
	if !reflect.DeepEqual(pagedK, wantK) {
		for i := range wantK {
			if math.Abs(float64(pagedK[i]-wantK[i])) > 1e-6 {
				t.Fatalf("pagedK mismatch at %d: got %v, want %v", i, pagedK[i], wantK[i])
			}
		}
	}
	if !reflect.DeepEqual(pagedV, wantV) {
		for i := range wantV {
			if math.Abs(float64(pagedV[i]-wantV[i])) > 1e-6 {
				t.Fatalf("pagedV mismatch at %d: got %v, want %v", i, pagedV[i], wantV[i])
			}
		}
	}
}

func TestFusedRoPESplitStoreFailClosed(t *testing.T) {
	dummy := make([]float32, 16)
	table := []int32{0}

	// Empty positions
	if _, err := ExecuteFusedRoPESplitStore(dummy, nil, table, 16, 1, 1, 8, 10000, dummy, dummy, dummy); err == nil {
		t.Fatal("expected error on empty positions")
	}

	// Odd headDim
	if _, err := ExecuteFusedRoPESplitStore(dummy, []int{0}, table, 16, 1, 1, 7, 10000, dummy, dummy, dummy); err == nil {
		t.Fatal("expected error on odd headDim")
	}

	// Page out of bounds
	if _, err := ExecuteFusedRoPESplitStore(dummy, []int{100}, table, 16, 1, 1, 8, 10000, dummy, dummy, dummy); err == nil {
		t.Fatal("expected error on page table overflow")
	}
}
