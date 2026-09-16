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
// Adapted from deepseek-ai/DeepSeek-V4.1-Flash inference/model.py:488-580 at
// dba1be0a40aa45a94ad051997016db3960a90277 (MIT).

package v41

import (
	"fmt"

	model "github.com/anthony-chaudhary/fak/internal/model"
)

// V41IndexerScore consumes already projected and RoPE-rotated index query
// heads for a single query step together with the normalized index keys of the
// compressed positions they may attend. It transcribes the pinned reference
// exactly: per-head dot product, per-element ReLU, then the learned per-head
// reduction applied to the rectified scores.
//
// q is row-major [nHeads * headDim] for one query position. keys is row-major
// [compressLen * headDim]. weights is the per-head reduction weight
// (weights_proj(x) * softmaxScale * nHeads**-0.5) already folded by the caller,
// matching the reference forward. The result is a fresh row-major
// [compressLen] score vector ready to feed V41SelectCandidateBlocks or
// V41SelectIndexRows. Projection, normalization and rotation are caller
// boundaries; this function begins where the reference's einsum does.
func V41IndexerScore(q, keys, weights []float32, nHeads, headDim, compressLen int) ([]float32, error) {
	if nHeads <= 0 || headDim <= 0 {
		return nil, fmt.Errorf("model: v41 indexer score requires positive nHeads=%d headDim=%d", nHeads, headDim)
	}
	if compressLen < 0 {
		return nil, fmt.Errorf("model: v41 indexer score negative compress length %d", compressLen)
	}
	if len(weights) != nHeads {
		return nil, fmt.Errorf("model: v41 indexer score weight length %d, want %d", len(weights), nHeads)
	}
	if len(q) != nHeads*headDim {
		return nil, fmt.Errorf("model: v41 indexer score query length %d, want %d", len(q), nHeads*headDim)
	}
	if len(keys) != compressLen*headDim {
		return nil, fmt.Errorf("model: v41 indexer score key length %d, want %d", len(keys), compressLen*headDim)
	}
	for i, w := range weights {
		if !model.Finite32(w) {
			return nil, fmt.Errorf("model: v41 indexer score weight %d is non-finite", i)
		}
	}
	for i, v := range q {
		if !model.Finite32(v) {
			return nil, fmt.Errorf("model: v41 indexer score query element %d is non-finite", i)
		}
	}
	for i, v := range keys {
		if !model.Finite32(v) {
			return nil, fmt.Errorf("model: v41 indexer score key element %d is non-finite", i)
		}
	}

	score := make([]float32, compressLen)
	for t := 0; t < compressLen; t++ {
		keyBase := t * headDim
		var acc float32
		for h := 0; h < nHeads; h++ {
			queryBase := h * headDim
			var dot float32
			for d := 0; d < headDim; d++ {
				dot += q[queryBase+d] * keys[keyBase+d]
			}
			// The reference rectifies before the learned head reduction:
			//   index_score = (index_score.relu_() * weights.unsqueeze(-1)).sum(dim=2)
			if dot < 0 {
				dot = 0
			}
			acc += dot * weights[h]
		}
		score[t] = acc
	}
	return score, nil
}

// v41IndexerPublication is the immutable, copied source-layer result a later
// reader layer reuses without recomputing the scoring path. It owns its own
// backing storage so a caller mutating its inputs after publication cannot
// change what a reader observes.
type v41IndexerPublication struct {
	layer       int
	compressLen int
	candidates  []bool
	rows        []int32
}

// v41IndexerPublish scores projected index inputs for a source layer and, when
// candidate blocks are requested, publishes the candidate mask alongside the
// final selected row IDs. The returned publication is a deep copy.
//
// topKBlocks <= 0 means this layer is not the candidate source: the mask is nil
// and only the final top-k rows are published, matching the reference's
// "uses_candidates" side that reads another layer's blocks. topK < 0 is refused;
// topK == 0 publishes an empty result.
func v41IndexerPublish(layer int, q, keys, weights []float32, nHeads, headDim, compressLen, topKBlocks, blockSize, topK, offset int) (*v41IndexerPublication, error) {
	if layer < 0 {
		return nil, fmt.Errorf("model: v41 indexer publication negative layer %d", layer)
	}
	if topK < 0 {
		return nil, fmt.Errorf("model: v41 indexer publication negative top-k %d", topK)
	}
	score, err := V41IndexerScore(q, keys, weights, nHeads, headDim, compressLen)
	if err != nil {
		return nil, err
	}
	var mask []bool
	if topKBlocks > 0 {
		mask, err = V41SelectCandidateBlocks(score, compressLen, topKBlocks, blockSize)
		if err != nil {
			return nil, err
		}
	}
	rows, err := V41SelectIndexRows(score, compressLen, topK, offset, mask)
	if err != nil {
		return nil, err
	}
	pub := &v41IndexerPublication{layer: layer, compressLen: compressLen}
	if mask != nil {
		pub.candidates = make([]bool, len(mask))
		copy(pub.candidates, mask)
	}
	pub.rows = make([]int32, len(rows))
	copy(pub.rows, rows)
	return pub, nil
}

// Candidates returns a copy of the published candidate mask, or nil when the
// source layer did not publish blocks. Readers get their own slice so they
// cannot mutate the publication.
func (p *v41IndexerPublication) Candidates() []bool {
	if p == nil || p.candidates == nil {
		return nil
	}
	out := make([]bool, len(p.candidates))
	copy(out, p.candidates)
	return out
}

// Rows returns a copy of the published final row IDs.
func (p *v41IndexerPublication) Rows() []int32 {
	if p == nil || p.rows == nil {
		return nil
	}
	out := make([]int32, len(p.rows))
	copy(out, p.rows)
	return out
}

// Reuse reproduces the reference's "reader layer without its own keys" path:
// given the same source publication and a reader's own reduction weights, it
// re-scores the reader's query against the source's selected rows only and
// returns fresh row IDs. It never recomputes the source scoring and never
// mutates the publication.
func (p *v41IndexerPublication) Reuse(q, keys, weights []float32, nHeads, headDim, compressLen, topK, offset int) ([]int32, error) {
	if p == nil {
		return nil, fmt.Errorf("model: v41 indexer reuse on nil publication")
	}
	if compressLen != p.compressLen {
		return nil, fmt.Errorf("model: v41 indexer reuse compress length %d, publication %d", compressLen, p.compressLen)
	}
	score, err := V41IndexerScore(q, keys, weights, nHeads, headDim, compressLen)
	if err != nil {
		return nil, err
	}
	return V41SelectIndexRows(score, compressLen, topK, offset, p.Candidates())
}
