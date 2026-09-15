package model

// v41_compress_index.go ports the DeepSeek V4.1 CED/CSA2 compressor pooling
// state machine and the lightning indexer scoring/selection primitives into the
// `model` package (fak#13006). It is a faithful transcription of the reference
// deepseek-ai/DeepSeek-V4.1-Flash inference/model.py (Compressor and the
// lightning indexer) at revision dba1be0a40aa45a94ad051997016db3960a90277 (MIT).
//
// The `internal/model/v41` subpackage already carries these primitives, but
// package `model` cannot import it: the subpackage imports `model`, so importing
// it back would be a cycle. The forward assembly in v41_forward.go therefore
// executes the stages against this in-package port. The subpackage copies remain
// the independent oracle; the two are kept comparable by their witness tests.
//
// Boundaries match the reference exactly: the compressor's input is the fp32
// result of the checkpoint's wkv/wgate projections and its output is the pooled
// latent through the BF16 cast and learned RMSNorm tail; the indexer begins where
// the reference's einsum does (projected, RoPE-rotated index queries and
// normalized index keys) and ends at the selected row IDs.

import (
	"fmt"
	"math"
	"sort"
)

// V41CompressorPool owns the causal partial-group state for one sequence and one
// compressed-attention layer. Its input is deliberately projected: callers
// supply the fp32 results of the checkpoint's wkv and wgate operations. Its
// output is the fp32 pooled latent before the checkpoint's learned RMSNorm, dtype
// conversion, RoPE, quantization, or cache publication.
type V41CompressorPool struct {
	ratio   int
	width   int
	nextPos int
	filled  int
	kv      []float32
	scores  []float32
}

// NewV41CompressorPool builds a pooling state for a ratio and latent width.
func NewV41CompressorPool(ratio, width int) (*V41CompressorPool, error) {
	if ratio < 1 {
		return nil, fmt.Errorf("model: V41 compressor ratio must be positive, got %d", ratio)
	}
	if width < 1 {
		return nil, fmt.Errorf("model: V41 compressor width must be positive, got %d", width)
	}
	if ratio > int(^uint(0)>>1)/width {
		return nil, fmt.Errorf("model: V41 compressor geometry %d*%d overflows", ratio, width)
	}
	return &V41CompressorPool{
		ratio:  ratio,
		width:  width,
		kv:     make([]float32, ratio*width),
		scores: make([]float32, ratio*width),
	}, nil
}

// Width reports the compressor latent width.
func (c *V41CompressorPool) Width() int {
	if c == nil {
		return 0
	}
	return c.width
}

// PushNormalized extends the projected-input pooling boundary through the
// reference compressor's dtype and learned-normalization tail: pooled values are
// cast to BF16, RMSNorm is evaluated in FP32, the learned weight is applied, and
// the result is cast back to BF16. For ratio 1 the projected KV enters this tail
// through the same BF16 boundary; projectedScore is unused.
func (c *V41CompressorPool) PushNormalized(pos int, projectedKV, projectedScore, normWeight []float32, eps float32) ([]float32, bool, error) {
	if c == nil || len(normWeight) != c.width {
		return nil, false, fmt.Errorf("model: V41 compressor norm width %d does not match %d", len(normWeight), c.Width())
	}
	if !Finite32(eps) || eps <= 0 {
		return nil, false, fmt.Errorf("model: V41 compressor norm epsilon must be finite and positive")
	}
	for i, weight := range normWeight {
		if !Finite32(weight) {
			return nil, false, fmt.Errorf("model: V41 compressor norm weight[%d] is non-finite", i)
		}
	}
	pooled, emitted, err := c.Push(pos, projectedKV, projectedScore)
	if err != nil || !emitted {
		return nil, emitted, err
	}
	var meanSquare float32
	for i := range pooled {
		pooled[i] = v41RoundBF16(pooled[i])
		meanSquare += pooled[i] * pooled[i]
	}
	rstd := float32(1 / math.Sqrt(float64(meanSquare/float32(c.width)+eps)))
	for i := range pooled {
		pooled[i] = v41RoundBF16(pooled[i] * rstd * normWeight[i])
	}
	return pooled, true, nil
}

// Push consumes one causal position. For ratio > 1 it emits only after the
// current group has ratio rows, matching the reference decode path. Softmax is
// independent for every latent dimension across the token-group axis.
func (c *V41CompressorPool) Push(pos int, projectedKV, projectedScore []float32) ([]float32, bool, error) {
	if c == nil {
		return nil, false, fmt.Errorf("model: nil V41 compressor")
	}
	if pos != c.nextPos {
		return nil, false, fmt.Errorf("model: V41 compressor position %d is not contiguous; want %d", pos, c.nextPos)
	}
	if len(projectedKV) != c.width {
		return nil, false, fmt.Errorf("model: V41 compressor KV width %d, want %d", len(projectedKV), c.width)
	}
	if c.ratio > 1 && len(projectedScore) != c.width {
		return nil, false, fmt.Errorf("model: V41 compressor score width %d, want %d", len(projectedScore), c.width)
	}
	for i, value := range projectedKV {
		if !Finite32(value) {
			return nil, false, fmt.Errorf("model: V41 compressor KV[%d] is non-finite", i)
		}
	}
	if c.ratio == 1 {
		c.nextPos++
		return append([]float32(nil), projectedKV...), true, nil
	}
	for i, value := range projectedScore {
		if !Finite32(value) {
			return nil, false, fmt.Errorf("model: V41 compressor score[%d] is non-finite", i)
		}
	}

	row := c.filled * c.width
	copy(c.kv[row:row+c.width], projectedKV)
	copy(c.scores[row:row+c.width], projectedScore)
	c.filled++
	c.nextPos++
	if c.filled < c.ratio {
		return nil, false, nil
	}

	out := make([]float32, c.width)
	for dim := 0; dim < c.width; dim++ {
		maxScore := c.scores[dim]
		for token := 1; token < c.ratio; token++ {
			if score := c.scores[token*c.width+dim]; score > maxScore {
				maxScore = score
			}
		}
		var denom float32
		for token := 0; token < c.ratio; token++ {
			weight := float32(math.Exp(float64(c.scores[token*c.width+dim] - maxScore)))
			c.scores[token*c.width+dim] = weight
			denom += weight
		}
		var weighted float32
		for token := 0; token < c.ratio; token++ {
			weight := c.scores[token*c.width+dim] / denom
			weighted += c.kv[token*c.width+dim] * weight
		}
		out[dim] = weighted
	}
	c.filled = 0
	return out, true, nil
}

// v41RoundBF16 returns the FP32 widening of an IEEE BF16 round-to-nearest-even
// cast, matching torch's bfloat16 boundary. Unlike v41BF16 (which leaves NaN/Inf
// bit patterns untouched for the Engram path), this rounds every finite value the
// way the reference's dtype cast does; non-finite inputs are refused upstream.
func v41RoundBF16(value float32) float32 {
	bits := math.Float32bits(value)
	bits += 0x7fff + ((bits >> 16) & 1)
	return math.Float32frombits(bits & 0xffff0000)
}

// V41IndexerScore consumes already projected and RoPE-rotated index query heads
// for a single query step together with the normalized index keys of the
// compressed positions they may attend. It transcribes the pinned reference
// exactly: per-head dot product, per-element ReLU, then the learned per-head
// reduction applied to the rectified scores.
//
// q is row-major [nHeads * headDim] for one query position. keys is row-major
// [compressLen * headDim]. weights is the per-head reduction weight already
// folded by the caller. The result is a fresh row-major [compressLen] score
// vector ready to feed V41SelectCandidateBlocks or V41SelectIndexRows.
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
		if !Finite32(w) {
			return nil, fmt.Errorf("model: v41 indexer score weight %d is non-finite", i)
		}
	}
	for i, v := range q {
		if !Finite32(v) {
			return nil, fmt.Errorf("model: v41 indexer score query element %d is non-finite", i)
		}
	}
	for i, v := range keys {
		if !Finite32(v) {
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

// V41IndexerPublication is the immutable, copied source-layer result a later
// reader layer reuses without recomputing the scoring path. It owns its own
// backing storage so a caller mutating its inputs after publication cannot
// change what a reader observes.
type V41IndexerPublication struct {
	layer       int
	compressLen int
	candidates  []bool
	rows        []int32
}

// NewV41IndexerPublication scores projected index inputs for a source layer and,
// when candidate blocks are requested, publishes the candidate mask alongside the
// final selected row IDs. The returned publication is a deep copy.
//
// topKBlocks <= 0 means this layer is not the candidate source: the mask is nil
// and only the final top-k rows are published. topK < 0 is refused; topK == 0
// publishes an empty result.
func NewV41IndexerPublication(layer int, q, keys, weights []float32, nHeads, headDim, compressLen, topKBlocks, blockSize, topK, offset int) (*V41IndexerPublication, error) {
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
	pub := &V41IndexerPublication{layer: layer, compressLen: compressLen}
	if mask != nil {
		pub.candidates = make([]bool, len(mask))
		copy(pub.candidates, mask)
	}
	pub.rows = make([]int32, len(rows))
	copy(pub.rows, rows)
	return pub, nil
}

// Candidates returns a copy of the published candidate mask, or nil when the
// source layer did not publish blocks. Readers get their own slice so they cannot
// mutate the publication.
func (p *V41IndexerPublication) Candidates() []bool {
	if p == nil || p.candidates == nil {
		return nil
	}
	out := make([]bool, len(p.candidates))
	copy(out, p.candidates)
	return out
}

// Rows returns a copy of the published final row IDs.
func (p *V41IndexerPublication) Rows() []int32 {
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
// returns fresh row IDs. It never recomputes the source scoring and never mutates
// the publication.
func (p *V41IndexerPublication) Reuse(q, keys, weights []float32, nHeads, headDim, compressLen, topK, offset int) ([]int32, error) {
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

type v41RankedScore struct {
	index int
	score float32
}

// V41SelectCandidateBlocks implements the first level of the V4.1 indexer for one
// query. Positions at or beyond compressLen are causal-invisible. Equal scores
// choose the lower block index first. The newest visible block is pinned even
// when it is incomplete.
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

// V41SelectIndexRows applies an optional source-candidate mask and implements the
// final top-k for one query. Results are position-sorted and shifted by offset.
// Masked or causal-invisible selections are represented by -1. Equal scores
// choose the lower position first.
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
