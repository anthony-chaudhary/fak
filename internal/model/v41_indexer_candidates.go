// Copyright (c) 2023 DeepSeek
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
//
// Adapted from deepseek-ai/DeepSeek-V4.1-Flash inference/model.py:561-610 at
// dba1be0a40aa45a94ad051997016db3960a90277 (MIT).

package model

import (
	"fmt"
	"math"
	"sort"
)

type v41RankedScore struct {
	index int
	score float32
}

// V41SelectCandidateBlocks implements the first level of the V4.1 indexer for
// one query. Positions at or beyond compressLen are causal-invisible. Equal
// scores choose the lower block index first; upstream torch.topk leaves ties
// unspecified. The newest visible block is pinned even when it is incomplete.
func V41SelectCandidateBlocks(logits []float32, compressLen, topKBlocks, blockSize int) ([]bool, error) {
	if compressLen < 0 || compressLen > len(logits) {
		return nil, fmt.Errorf("v41 candidate blocks: compress length %d outside [0,%d]", compressLen, len(logits))
	}
	if topKBlocks < 0 {
		return nil, fmt.Errorf("v41 candidate blocks: negative top-k %d", topKBlocks)
	}
	if blockSize <= 0 {
		return nil, fmt.Errorf("v41 candidate blocks: block size must be positive")
	}
	for i, score := range logits {
		if math.IsNaN(float64(score)) {
			return nil, fmt.Errorf("v41 candidate blocks: score %d is NaN", i)
		}
	}

	keep := make([]bool, len(logits))
	if len(logits) == 0 || topKBlocks == 0 || compressLen == 0 {
		return keep, nil
	}
	numBlocks := (len(logits)-1)/blockSize + 1
	ranked := make([]v41RankedScore, numBlocks)
	for block := 0; block < numBlocks; block++ {
		best := float32(math.Inf(-1))
		start := block * blockSize
		end := min(start+blockSize, compressLen)
		for position := start; position < end; position++ {
			best = max(best, logits[position])
		}
		ranked[block] = v41RankedScore{index: block, score: best}
	}
	ranked[(compressLen-1)/blockSize].score = float32(math.Inf(1))
	sortV41Scores(ranked)

	for _, candidate := range ranked[:min(topKBlocks, numBlocks)] {
		if math.IsInf(float64(candidate.score), -1) {
			continue
		}
		start := candidate.index * blockSize
		for position := start; position < min(start+blockSize, len(keep)); position++ {
			keep[position] = true
		}
	}
	return keep, nil
}

// V41SelectIndexRows applies an optional source-candidate mask and implements
// the final top-k for one query. Results are position-sorted and shifted by
// offset. Masked or causal-invisible selections are represented by -1. This
// extends the official prefill guard so a short candidate mask cannot resurrect
// an excluded row. Equal scores choose the lower position first.
func V41SelectIndexRows(logits []float32, compressLen, topK, offset int, candidates []bool) ([]int32, error) {
	if compressLen < 0 || compressLen > len(logits) {
		return nil, fmt.Errorf("v41 index rows: compress length %d outside [0,%d]", compressLen, len(logits))
	}
	if topK < 0 {
		return nil, fmt.Errorf("v41 index rows: negative top-k %d", topK)
	}
	if candidates != nil && len(candidates) != len(logits) {
		return nil, fmt.Errorf("v41 index rows: candidate mask length %d, want %d", len(candidates), len(logits))
	}
	if offset < 0 || int64(offset)+int64(max(0, compressLen-1)) > math.MaxInt32 {
		return nil, fmt.Errorf("v41 index rows: offset %d cannot represent visible rows as int32", offset)
	}

	ranked := make([]v41RankedScore, len(logits))
	for position, score := range logits {
		if math.IsNaN(float64(score)) {
			return nil, fmt.Errorf("v41 index rows: score %d is NaN", position)
		}
		if position >= compressLen || (candidates != nil && !candidates[position]) {
			score = float32(math.Inf(-1))
		}
		ranked[position] = v41RankedScore{index: position, score: score}
	}
	sortV41Scores(ranked)
	ranked = ranked[:min(topK, len(ranked))]
	sort.Slice(ranked, func(i, j int) bool { return ranked[i].index < ranked[j].index })

	rows := make([]int32, len(ranked))
	for i, selected := range ranked {
		if selected.index >= compressLen || math.IsInf(float64(selected.score), -1) {
			rows[i] = -1
			continue
		}
		rows[i] = int32(offset + selected.index)
	}
	return rows, nil
}

func sortV41Scores(scores []v41RankedScore) {
	sort.Slice(scores, func(i, j int) bool {
		if scores[i].score == scores[j].score {
			return scores[i].index < scores[j].index
		}
		return scores[i].score > scores[j].score
	})
}
