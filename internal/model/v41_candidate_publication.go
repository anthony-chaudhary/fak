package model

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
// Selection composition adapted from Indexer.forward/select_candidate_blocks in
// deepseek-ai/DeepSeek-V4.1-Flash inference/model.py at
// dba1be0a40aa45a94ad051997016db3960a90277 (MIT). Existing selectors retain their
// deterministic tie and -1 policies; this is not a numerical parity claim.

import "fmt"

// v41CandidateSource identifies the producer and the shared compressed-key
// stream. queryStart is the absolute token position of scores[0]. Prefill starts
// at zero; decode publishes one query. A candidate publication is distinct from
// the final top-k IDs ordinary attention readers borrow.
type v41CandidateSource struct {
	layer         int
	kvSourceLayer int
	ratio         int
	queryStart    int
	queryEnd      int
}

// v41CandidatePublication is an immutable, session-local batch of candidate
// masks. Keeping a row per query prevents a prefill reader from using the last
// query's mask for every position. The producer owns all mask storage; accessors
// copy it. This primitive is not yet stored by V41AttentionState or wired into
// the serving path, whose candidate-source admission fence must remain closed.
type v41CandidatePublication struct {
	source  v41CandidateSource
	width   int
	visible []int
	masks   [][]bool
}

// v41SelectCandidateSource receives already computed source scores, one row per
// query over the full key extent for this call. It computes both outputs from
// those scores: per-query candidate masks for later indexers, and UNMASKED source
// top-k rows. The reference's source branch and later-consumer branch are mutually
// exclusive. No projections, score kernel, or source rescoring run here.
//
// A nil publication means absent. A present zero-width mask is non-nil and must
// not be converted into absence. On failure neither output is published.
func v41SelectCandidateSource(source v41CandidateSource, scores [][]float32, topKBlocks, blockSize, topK, offset int) (*v41CandidatePublication, [][]int32, error) {
	if source.layer < 0 || source.kvSourceLayer < 0 || source.kvSourceLayer > source.layer || source.ratio <= 0 || source.queryStart < 0 || source.queryEnd < source.queryStart || len(scores) == 0 || source.queryEnd-source.queryStart != len(scores) {
		return nil, nil, fmt.Errorf("model: invalid V41 candidate source or query range")
	}
	if source.queryStart > 0 && len(scores) != 1 {
		return nil, nil, fmt.Errorf("model: V41 candidate publication requires initial prefill or single-query decode")
	}
	if topKBlocks <= 0 || blockSize <= 0 || topK < 0 {
		return nil, nil, fmt.Errorf("model: invalid V41 candidate selection budget")
	}
	width := source.queryEnd / source.ratio
	p := &v41CandidatePublication{
		source: source, width: width,
		visible: make([]int, len(scores)),
		masks:   make([][]bool, len(scores)),
	}
	rows := make([][]int32, len(scores))
	for i, logits := range scores {
		if len(logits) != width {
			return nil, nil, fmt.Errorf("model: V41 candidate query %d score width %d, want %d", source.queryStart+i, len(logits), width)
		}
		visible := (source.queryStart + i + 1) / source.ratio
		mask, err := V41SelectCandidateBlocks(logits, visible, topKBlocks, blockSize)
		if err != nil {
			return nil, nil, err
		}
		// Publishing candidate blocks does not restrict the producer's own top-k.
		selected, err := V41SelectIndexRows(logits, visible, topK, offset, nil)
		if err != nil {
			return nil, nil, err
		}
		p.visible[i], p.masks[i], rows[i] = visible, mask, selected
	}
	return p, rows, nil
}

// mask returns owned data for one absolute query position. false means absent
// or out of range, including an attempt to reuse a decode mask on the next step.
func (p *v41CandidatePublication) mask(position int) ([]bool, bool) {
	if p == nil || position < p.source.queryStart || position-p.source.queryStart >= len(p.masks) {
		return nil, false
	}
	row := p.masks[position-p.source.queryStart]
	out := make([]bool, len(row))
	copy(out, row)
	return out, true
}

// selectRows applies this source's mask to a LATER query-owning indexer's own
// score vector. expected binds the full source/query-range identity, while layer
// and position identify the consumer. Missing or mismatched publication is an
// error, never an implicit unrestricted selection. Ordinary readers do not call
// this method: they reuse the latest index source's already selected IDs.
func (p *v41CandidatePublication) selectRows(expected v41CandidateSource, layer, position int, scores []float32, topK, offset int) ([]int32, error) {
	if p == nil || p.source != expected {
		return nil, fmt.Errorf("model: V41 candidate source publication missing or mismatched")
	}
	if layer <= p.source.layer {
		return nil, fmt.Errorf("model: V41 candidate consumer layer %d is not after source %d", layer, p.source.layer)
	}
	if position < p.source.queryStart || position-p.source.queryStart >= len(p.masks) {
		return nil, fmt.Errorf("model: V41 candidate query %d is outside the published range", position)
	}
	if len(scores) != p.width {
		return nil, fmt.Errorf("model: V41 candidate consumer score width %d, want %d", len(scores), p.width)
	}
	i := position - p.source.queryStart
	return V41SelectIndexRows(scores, p.visible[i], topK, offset, p.masks[i])
}
