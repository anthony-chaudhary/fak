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
	"cmp"
	"fmt"
	"math"
	"slices"
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

	// normalize and gatedPool are session-bound only on ephemeral producer
	// pools. Retained attention state and prefix snapshots contain data, never
	// these callbacks. gatedPool, when set, owns a completed group's softmax
	// gate and weighted sum (ratio >= 2); it receives the group's kv and score
	// rows [ratio*width] in token order and returns the raw pooled row.
	normalize func(pooled, gain []float32, eps float32) ([]float32, error)
	gatedPool func(kv, scores []float32) ([]float32, error)
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
	return c.pushNormalizedObserved(pos, projectedKV, projectedScore, normWeight, eps, nil)
}

func (c *V41CompressorPool) pushNormalizedObserved(pos int, projectedKV, projectedScore, normWeight []float32, eps float32, observe func(string)) ([]float32, bool, error) {
	if err := c.validateNorm(normWeight, eps); err != nil {
		return nil, false, err
	}
	pooled, emitted, err := c.pushObserved(pos, projectedKV, projectedScore, observe)
	if err != nil || !emitted {
		return nil, emitted, err
	}
	normalized, err := c.normalizeEmittedObserved(pooled, normWeight, eps, observe)
	return normalized, true, err
}

func (c *V41CompressorPool) validateNorm(normWeight []float32, eps float32) error {
	if c == nil || len(normWeight) != c.width {
		return fmt.Errorf("model: V41 compressor norm width %d does not match %d", len(normWeight), c.Width())
	}
	if !Finite32(eps) || eps <= 0 {
		return fmt.Errorf("model: V41 compressor norm epsilon must be finite and positive")
	}
	for i, weight := range normWeight {
		if !Finite32(weight) {
			return fmt.Errorf("model: V41 compressor norm weight[%d] is non-finite", i)
		}
	}
	return nil
}

// normalizePooled is PushNormalized's tail for an emitted raw pooled row; it
// may round pooled in place.
func (c *V41CompressorPool) normalizePooled(pooled, normWeight []float32, eps float32) ([]float32, error) {
	return c.normalizeEmittedObserved(pooled, normWeight, eps, nil)
}

func (c *V41CompressorPool) normalizeEmittedObserved(pooled, normWeight []float32, eps float32, observe func(string)) ([]float32, error) {
	if c.normalize != nil {
		return c.normalize(pooled, normWeight, eps)
	}
	if observe != nil && len(pooled) > 0 {
		observe("compressor-norm")
		observe("bf16-boundary")
	}
	var meanSquare float32
	for i := range pooled {
		pooled[i] = v41RoundBF16(pooled[i])
		meanSquare += float32(pooled[i] * pooled[i])
	}
	rstd := float32(1 / math.Sqrt(float64(meanSquare/float32(c.width)+eps)))
	for i := range pooled {
		pooled[i] = v41RoundBF16(pooled[i] * rstd * normWeight[i])
	}
	return pooled, nil
}

// Push consumes one causal position. For ratio > 1 it emits only after the
// current group has ratio rows, matching the reference decode path. Softmax is
// independent for every latent dimension across the token-group axis.
func (c *V41CompressorPool) Push(pos int, projectedKV, projectedScore []float32) ([]float32, bool, error) {
	return c.pushObserved(pos, projectedKV, projectedScore, nil)
}

func (c *V41CompressorPool) pushObserved(pos int, projectedKV, projectedScore []float32, observe func(string)) ([]float32, bool, error) {
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
	if c.gatedPool != nil {
		c.filled = 0
		out, err := c.gatedPool(c.kv, c.scores)
		if err != nil {
			return nil, false, err
		}
		if len(out) != c.width {
			return nil, false, fmt.Errorf("model: V41 compressor gated pool width %d, want %d", len(out), c.width)
		}
		return out, true, nil
	}

	if observe != nil {
		observe("compressor-gated-pool")
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
			weighted += float32(c.kv[token*c.width+dim] * weight)
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
	if v41HostBF16Witness.Load() {
		v41HostBF16Rounds.Add(1)
	}
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
// Products and running sums are explicitly rounded to float32 in dimension and
// head order. The conversions prevent fused multiply-add from discarding an
// intermediate rounding, preserving the scalar F32 scoring contract.
//
// q is row-major [nHeads * headDim] for one query position. keys is row-major
// [compressLen * headDim]. weights is the per-head reduction weight already
// folded by the caller. The result is a fresh row-major [compressLen] score
// vector ready to feed V41SelectCandidateBlocks or V41SelectIndexRows.
func V41IndexerScore(q, keys, weights []float32, nHeads, headDim, compressLen int) ([]float32, error) {
	return v41IndexerScoreObserved(q, keys, weights, nHeads, headDim, compressLen, nil)
}

func v41IndexerScoreObserved(q, keys, weights []float32, nHeads, headDim, compressLen int, observe func(string)) ([]float32, error) {
	if nHeads <= 0 || headDim <= 0 {
		return nil, fmt.Errorf("model: v41 indexer score requires positive nHeads=%d headDim=%d", nHeads, headDim)
	}
	if compressLen < 0 {
		return nil, fmt.Errorf("model: v41 indexer score negative compress length %d", compressLen)
	}
	// Validate both products before shape checks or allocation. Wrapped zero
	// could otherwise admit an empty query or an impossibly large key panel.
	maxInt := int(^uint(0) >> 1)
	if nHeads > maxInt/headDim {
		return nil, fmt.Errorf("model: v41 indexer score query geometry overflows int: nHeads=%d headDim=%d", nHeads, headDim)
	}
	if compressLen > maxInt/headDim {
		return nil, fmt.Errorf("model: v41 indexer score key geometry overflows int: compressLen=%d headDim=%d", compressLen, headDim)
	}
	queryElements, keyElements := nHeads*headDim, compressLen*headDim
	if len(weights) != nHeads {
		return nil, fmt.Errorf("model: v41 indexer score weight length %d, want %d", len(weights), nHeads)
	}
	if len(q) != queryElements {
		return nil, fmt.Errorf("model: v41 indexer score query length %d, want %d", len(q), queryElements)
	}
	if len(keys) != keyElements {
		return nil, fmt.Errorf("model: v41 indexer score key length %d, want %d", len(keys), keyElements)
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
	if observe != nil && compressLen > 0 {
		observe("attention-indexer-score")
	}
	// Only complete output rows are independent. Keep each row's dimension
	// and head reductions in the original order, using the existing worker
	// budget and work threshold. Saturation is only a scheduling estimate;
	// individually valid query/key shapes must not gain a new overflow refusal.
	work := keyElements
	if work > maxInt/nHeads {
		work = maxInt
	} else {
		work *= nHeads
	}
	parForRange(compressLen, work, func(lo, hi int) {
		// Tile independent key rows, not queries: reuse each query load while
		// retaining every row's exact dimension/head reduction and rounding.
		// Preparation, validation, observation and callback order stay outside
		// this local arithmetic transformation. Scratch is eight scalar F32s.
		t := lo
		for ; hi-t >= 4; t += 4 {
			k0, k1 := t*headDim, (t+1)*headDim
			k2, k3 := (t+2)*headDim, (t+3)*headDim
			var a0, a1, a2, a3 float32
			for h := 0; h < nHeads; h++ {
				queryBase := h * headDim
				var d0, d1, d2, d3 float32
				for d := 0; d < headDim; d++ {
					qv := q[queryBase+d]
					d0 = float32(d0 + float32(qv*keys[k0+d]))
					d1 = float32(d1 + float32(qv*keys[k1+d]))
					d2 = float32(d2 + float32(qv*keys[k2+d]))
					d3 = float32(d3 + float32(qv*keys[k3+d]))
				}
				if d0 < 0 {
					d0 = 0
				}
				if d1 < 0 {
					d1 = 0
				}
				if d2 < 0 {
					d2 = 0
				}
				if d3 < 0 {
					d3 = 0
				}
				w := weights[h]
				a0 = float32(a0 + float32(d0*w))
				a1 = float32(a1 + float32(d1*w))
				a2 = float32(a2 + float32(d2*w))
				a3 = float32(a3 + float32(d3*w))
			}
			score[t], score[t+1], score[t+2], score[t+3] = a0, a1, a2, a3
		}
		for ; t < hi; t++ {
			keyBase := t * headDim
			var acc float32
			for h := 0; h < nHeads; h++ {
				queryBase := h * headDim
				var dot float32
				for d := 0; d < headDim; d++ {
					dot = float32(dot + float32(q[queryBase+d]*keys[keyBase+d]))
				}
				// The reference rectifies before the learned head reduction:
				//   index_score = (index_score.relu_() * weights.unsqueeze(-1)).sum(dim=2)
				if dot < 0 {
					dot = 0
				}
				acc = float32(acc + float32(dot*weights[h]))
			}
			score[t] = acc
		}
	})
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
	return newV41IndexerPublicationObserved(layer, q, keys, weights, nHeads, headDim, compressLen, topKBlocks, blockSize, topK, offset, nil)
}

func newV41IndexerPublicationObserved(layer int, q, keys, weights []float32, nHeads, headDim, compressLen, topKBlocks, blockSize, topK, offset int, observe func(string)) (*V41IndexerPublication, error) {
	if layer < 0 {
		return nil, fmt.Errorf("model: v41 indexer publication negative layer %d", layer)
	}
	if topK < 0 {
		return nil, fmt.Errorf("model: v41 indexer publication negative top-k %d", topK)
	}
	score, err := v41IndexerScoreObserved(q, keys, weights, nHeads, headDim, compressLen, observe)
	if err != nil {
		return nil, err
	}
	var mask []bool
	if topKBlocks > 0 {
		mask, err = v41SelectCandidateBlocksObserved(score, compressLen, topKBlocks, blockSize, observe)
		if err != nil {
			return nil, err
		}
	}
	rows, err := v41SelectIndexRowsObserved(score, compressLen, topK, offset, mask, observe)
	if err != nil {
		return nil, err
	}
	// Both selectors return fresh storage owned only by this constructor.
	// Adopt it here; the public getters still copy for every reader.
	return &V41IndexerPublication{
		layer: layer, compressLen: compressLen, candidates: mask, rows: rows,
	}, nil
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
	return v41SelectCandidateBlocksObserved(logits, compressLen, topKBlocks, blockSize, nil)
}

func v41SelectCandidateBlocksObserved(logits []float32, compressLen, topKBlocks, blockSize int, observe func(string)) ([]bool, error) {
	return v41SelectRankedBlocksObserved(logits, compressLen, topKBlocks, blockSize, observe)
}

func v41SelectRankedBlocksObserved(logits []float32, compressLen, topKBlocks, blockSize int, observe func(string)) ([]bool, error) {
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
	if observe != nil {
		observe("attention-index-list")
	}
	numBlocks := (len(logits)-1)/blockSize + 1
	if topKBlocks >= numBlocks {
		// All blocks fit, so score order cannot affect membership. Preserve
		// empty (-Inf) block exclusion and the newest visible block's pin,
		// including its causally hidden trailing positions.
		newest := (compressLen - 1) / blockSize
		for block := 0; block <= newest; block++ {
			start := block * blockSize
			selected := block == newest
			for position := start; position < min(start+blockSize, compressLen); position++ {
				if !math.IsInf(float64(logits[position]), -1) {
					selected = true
				}
			}
			if selected {
				for position := start; position < min(start+blockSize, len(keep)); position++ {
					keep[position] = true
				}
			}
		}
		return keep, nil
	}
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
	return v41SelectIndexRowsObserved(logits, compressLen, topK, offset, candidates, nil)
}

func v41SelectIndexRowsObserved(logits []float32, compressLen, topK, offset int, candidates []bool, observe func(string)) ([]int32, error) {
	if compressLen < 0 || compressLen > len(logits) {
		return nil, fmt.Errorf("v41 index rows: compress length %d outside [0,%d]", compressLen, len(logits))
	}
	if topK < 0 {
		return nil, fmt.Errorf("v41 index rows: negative top-k %d", topK)
	}
	if candidates != nil && len(candidates) != len(logits) {
		return nil, fmt.Errorf("v41 index rows: candidate mask length %d, want %d", len(candidates), len(logits))
	}
	if offset < 0 || int64(offset) > math.MaxInt32-int64(max(0, compressLen-1)) {
		return nil, fmt.Errorf("v41 index rows: offset %d cannot represent visible rows as int32", offset)
	}

	// Selecting every position makes both score ranking and the subsequent
	// index sort redundant. Still inspect every original score for NaN before
	// returning, including masked and causal-invisible positions.
	if topK >= len(logits) {
		rows := make([]int32, len(logits))
		for position, score := range logits {
			if math.IsNaN(float64(score)) {
				return nil, fmt.Errorf("v41 index rows: score %d is NaN", position)
			}
			if position >= compressLen || (candidates != nil && !candidates[position]) || math.IsInf(float64(score), -1) {
				rows[position] = -1
			} else {
				rows[position] = int32(offset + position)
			}
		}
		if observe != nil && len(rows) > 0 {
			observe("attention-index-list")
		}
		return rows, nil
	}

	// Retain only the requested rows. The root is the worst retained score
	// (largest position on ties), so replacement preserves the full-sort
	// selection without ranking or storing the discarded tail.
	ranked := make([]v41RankedScore, topK)
	for position, score := range logits {
		if math.IsNaN(float64(score)) {
			return nil, fmt.Errorf("v41 index rows: score %d is NaN", position)
		}
		if position >= compressLen || (candidates != nil && !candidates[position]) {
			score = float32(math.Inf(-1))
		}
		candidate := v41RankedScore{index: position, score: score}
		if position < topK {
			ranked[position] = candidate
			if position == topK-1 {
				for parent := len(ranked)/2 - 1; parent >= 0; parent-- {
					v41RankedHeapDown(ranked, parent)
				}
			}
		} else if topK > 0 && compareV41Scores(candidate, ranked[0]) < 0 {
			ranked[0] = candidate
			v41RankedHeapDown(ranked, 0)
		}
	}
	// Preserve the host-arm event for a nonempty input even when topK is zero.
	if observe != nil && len(logits) > 0 {
		observe("attention-index-list")
	}
	slices.SortFunc(ranked, func(a, b v41RankedScore) int { return cmp.Compare(a.index, b.index) })

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
	slices.SortFunc(scores, compareV41Scores)
}

func compareV41Scores(a, b v41RankedScore) int {
	if a.score == b.score {
		return cmp.Compare(a.index, b.index)
	}
	return cmp.Compare(b.score, a.score)
}

// v41RankedHeapDown adapts Go 1.26.0 container/heap.down to typed storage and
// the reverse selection order. Typed storage avoids an escaping heap.Interface
// wrapper; selection still owns only its ranked buffer and result buffer.
// Source: https://github.com/golang/go/blob/go1.26.0/src/container/heap/heap.go
//
// Copyright 2009 The Go Authors. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
//   - Redistributions of source code must retain the above copyright
//
// notice, this list of conditions and the following disclaimer.
//   - Redistributions in binary form must reproduce the above
//
// copyright notice, this list of conditions and the following disclaimer
// in the documentation and/or other materials provided with the
// distribution.
//   - Neither the name of Google LLC nor the names of its
//
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
func v41RankedHeapDown(ranked []v41RankedScore, parent int) {
	for {
		left := 2*parent + 1
		if left >= len(ranked) || left < 0 {
			return
		}
		worst := left
		if right := left + 1; right < len(ranked) && compareV41Scores(ranked[right], ranked[left]) > 0 {
			worst = right
		}
		if compareV41Scores(ranked[worst], ranked[parent]) <= 0 {
			return
		}
		ranked[parent], ranked[worst] = ranked[worst], ranked[parent]
		parent = worst
	}
}
