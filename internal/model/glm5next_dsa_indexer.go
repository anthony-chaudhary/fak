package model

import (
	"math"
	"sort"
)

// SelectGLM5NextDSATopK scores query index against downsampled key blocks and selects top-K block indices:
// qIdx: [numIndexHeads * indexHeadDim] = [32 * 128]
// downsampledKeys: [totalBlocks * numIndexHeads * indexHeadDim]
// currentBlock: the block index containing the current token (t / blockStride).
// Causal mask: block b > currentBlock is forbidden.
// Local window: currentBlock is always retained.
// topK: max number of blocks to select (e.g. 2048).
// Returns slice of selected block indices sorted ascending.
func SelectGLM5NextDSATopK(
	qIdx []float32,
	downsampledKeys []float32,
	totalBlocks int,
	currentBlock int,
	numIndexHeads, indexHeadDim int,
	topK int,
) []int {
	if totalBlocks <= 0 || currentBlock < 0 || numIndexHeads <= 0 || indexHeadDim <= 0 {
		return nil
	}
	blockDim := numIndexHeads * indexHeadDim
	if len(qIdx) < blockDim || len(downsampledKeys) < totalBlocks*blockDim {
		return nil
	}

	validBlocks := min(currentBlock+1, totalBlocks)
	if validBlocks <= 0 {
		return nil
	}

	scale := float32(1.0 / math.Sqrt(float64(indexHeadDim)))

	type blockScore struct {
		block int
		score float32
	}

	scores := make([]blockScore, validBlocks)
	for b := 0; b < validBlocks; b++ {
		var dot float32
		kOff := b * blockDim
		for i := 0; i < blockDim; i++ {
			dot += qIdx[i] * downsampledKeys[kOff+i]
		}
		scores[b] = blockScore{
			block: b,
			score: dot * scale,
		}
	}

	// Sort descending by score
	sort.Slice(scores, func(i, j int) bool {
		if scores[i].score == scores[j].score {
			return scores[i].block > scores[j].block
		}
		return scores[i].score > scores[j].score
	})

	k := min(topK, validBlocks)
	selected := make(map[int]bool, k)
	for i := 0; i < k; i++ {
		selected[scores[i].block] = true
	}

	// Always retain local block (currentBlock)
	if currentBlock < validBlocks {
		selected[currentBlock] = true
	}

	out := make([]int, 0, len(selected))
	for b := range selected {
		out = append(out, b)
	}
	sort.Ints(out)

	return out
}
