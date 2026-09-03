package model

import (
	"testing"
)

func TestSelectGLM5NextDSATopK(t *testing.T) {
	const numIndexHeads = 2
	const indexHeadDim = 4
	const blockDim = numIndexHeads * indexHeadDim
	const totalBlocks = 8

	qIdx := make([]float32, blockDim)
	for i := range qIdx {
		qIdx[i] = 1.0
	}

	downsampledKeys := make([]float32, totalBlocks*blockDim)
	// Block 0: score low (value 0.1)
	// Block 1: score medium (value 0.5)
	// Block 2: score high (value 0.9)
	// Block 3: currentBlock, score very low (value 0.01)
	// Block 4..7: future blocks, huge score (value 10.0)
	for b := 0; b < totalBlocks; b++ {
		var val float32
		switch b {
		case 0:
			val = 0.1
		case 1:
			val = 0.5
		case 2:
			val = 0.9
		case 3:
			val = 0.01
		default:
			val = 10.0
		}
		for i := 0; i < blockDim; i++ {
			downsampledKeys[b*blockDim+i] = val
		}
	}

	// Current block is 3, topK is 2
	selected := SelectGLM5NextDSATopK(qIdx, downsampledKeys, totalBlocks, 3, numIndexHeads, indexHeadDim, 2)

	// 1. Causal constraint: no block > 3 can ever be selected
	for _, b := range selected {
		if b > 3 {
			t.Fatalf("causal violation: selected future block %d (currentBlock=3)", b)
		}
	}

	// 2. Local block retention: currentBlock (3) must be present even though its score is lowest
	hasCurrent := false
	for _, b := range selected {
		if b == 3 {
			hasCurrent = true
		}
	}
	if !hasCurrent {
		t.Fatalf("local block 3 was not retained: selected=%v", selected)
	}

	// 3. Top-scoring historical blocks: block 2 (score 0.9) and block 1 (score 0.5)
	// Since topK=2, block 2 was selected as top-1, and block 3 was retained as local
	hasBlock2 := false
	for _, b := range selected {
		if b == 2 {
			hasBlock2 = true
		}
	}
	if !hasBlock2 {
		t.Fatalf("highest scoring block 2 was not selected: %v", selected)
	}
}
