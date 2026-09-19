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
// Adapted from deepseek-ai/DeepSeek-V4.1-Flash inference/kernel.py:311-405 and
// inference/model.py:765-789 at dba1be0a40aa45a94ad051997016db3960a90277 (MIT).

package v41

import (
	"errors"
	"fmt"
	model "github.com/anthony-chaudhary/fak/internal/model"
	"math"
)

// ErrV41SparseSinkNonFinite is the typed fail-closed refusal for the V4.1
// sparse sink contraction (issue #13290). The contraction validates every
// INPUT is finite but the 512-term score/value accumulations can overflow
// float32 to +Inf while the inputs stay finite; +Inf then poisons the softmax
// denominator (Inf-Inf = NaN) and the weighted value sum. This sentinel lets a
// caller (and the layered forward's output projection guard) attribute a
// downstream "non-finite value" refusal to the TRUE producing stage instead of
// the first downstream tensor that happens to notice it.
var ErrV41SparseSinkNonFinite = errors.New("model: DeepSeek V4.1 sparse sink produced a non-finite value")

// v41SparseSinkStage names the arithmetic stage that first produced a
// non-finite value in V41SparseAttentionSink.
type v41SparseSinkStage string

const (
	// v41SinkStageScore is the scaled dot-product score accumulate
	// (dot += q*kv; dot *= softmax). An overflow here makes maxScore +Inf.
	v41SinkStageScore v41SparseSinkStage = "score accumulate"
	// v41SinkStageSoftmax is the softmax denominator sum of exp terms. An
	// Inf-Inf NaN in an exponent poisons sum without tripping the sum==0 guard.
	v41SinkStageSoftmax v41SparseSinkStage = "softmax denominator"
	// v41SinkStageValue is the weighted value accumulate
	// (o[oBase+d] += weight*kv[...]).
	v41SinkStageValue v41SparseSinkStage = "weighted value accumulate"
)

// V41SparseAttentionSinkOptions names the already-projected tensor geometry
// for one sparse sink attention contraction. B is the batch, M the number of
// query positions, Heads the local query head count, HeadDim the per-head
// width, TopK the index-list width, and N the KV row count. The caller owns
// projections, row selection and position mapping; this component begins where
// the pinned reference's sparse_attn kernel inputs do.
//
// Sink is either nil (no learnable sink: the denominator is the softmax sum
// alone) or a Heads-length per-head bias added to the denominator as
// exp(sink[h]-max), matching the reference.
type V41SparseAttentionSinkOptions struct {
	B       int
	M       int
	Heads   int
	HeadDim int
	TopK    int
	N       int
	Softmax float32
	// Inverse, when non-nil, is the inverse output RoPE applied to the trailing
	// RopeDim dimensions of every output head in place. RopeDim must be even.
	Inverse func(ropeDim int, o []float32) error
	RopeDim int
}

// V41SparseAttentionSink contracts already-projected V4.1 query heads against
// caller-selected KV rows using the official sink denominator. It transcribes
// the pinned reference exactly:
//
//	score[b,m,h,i] = (q[b,m,h,:] . kv[b,idx,:]) * softmax
//	invalid idx (-1) contributes neither score nor value
//	o[b,m,h,:] = sum_i softmax_i * kv[b,idx_i,:], with exp(sink[h]-max)
//	             folded into the softmax denominator
//
// q is row-major [B, M, Heads, HeadDim]; kv is row-major [B, N, HeadDim],
// shared across heads; idx is row-major [B, M, TopK] with -1 marking an empty
// slot. sink, when non-nil, is [Heads]. The result is a fresh row-major
// [B, M, Heads, HeadDim] tensor. A row whose indices are all -1 yields an
// all-zero output, matching the reference's finite lower bound. Every input is
// validated and every non-finite value is refused before any output is
// produced.
func V41SparseAttentionSink(q, kv []float32, sink []float32, idx []int32, opt V41SparseAttentionSinkOptions) ([]float32, error) {
	if opt.B <= 0 || opt.M <= 0 || opt.Heads <= 0 || opt.HeadDim <= 0 || opt.TopK < 0 || opt.N < 0 {
		return nil, fmt.Errorf("model: invalid V4.1 sparse sink geometry b=%d m=%d heads=%d dim=%d topk=%d n=%d", opt.B, opt.M, opt.Heads, opt.HeadDim, opt.TopK, opt.N)
	}
	if !model.Finite32(opt.Softmax) || opt.Softmax == 0 {
		return nil, fmt.Errorf("model: V4.1 sparse sink softmax scale must be finite and non-zero, got %g", opt.Softmax)
	}
	if opt.Inverse != nil && (opt.RopeDim <= 0 || opt.RopeDim > opt.HeadDim || opt.RopeDim%2 != 0) {
		return nil, fmt.Errorf("model: V4.1 sparse sink inverse rotation needs an even rope dim in (0,%d], got %d", opt.HeadDim, opt.RopeDim)
	}
	if len(q) != opt.B*opt.M*opt.Heads*opt.HeadDim {
		return nil, fmt.Errorf("model: V4.1 sparse sink query length %d, want %d", len(q), opt.B*opt.M*opt.Heads*opt.HeadDim)
	}
	if len(kv) != opt.B*opt.N*opt.HeadDim {
		return nil, fmt.Errorf("model: V4.1 sparse sink kv length %d, want %d", len(kv), opt.B*opt.N*opt.HeadDim)
	}
	if len(idx) != opt.B*opt.M*opt.TopK {
		return nil, fmt.Errorf("model: V4.1 sparse sink index length %d, want %d", len(idx), opt.B*opt.M*opt.TopK)
	}
	if sink != nil && len(sink) != opt.Heads {
		return nil, fmt.Errorf("model: V4.1 sparse sink length %d, want %d", len(sink), opt.Heads)
	}
	for i, v := range q {
		if !model.Finite32(v) {
			return nil, fmt.Errorf("model: V4.1 sparse sink non-finite query element %d", i)
		}
	}
	for i, v := range kv {
		if !model.Finite32(v) {
			return nil, fmt.Errorf("model: V4.1 sparse sink non-finite kv element %d", i)
		}
	}
	if sink != nil {
		for i, v := range sink {
			if !model.Finite32(v) {
				return nil, fmt.Errorf("model: V4.1 sparse sink non-finite sink value %d", i)
			}
		}
	}
	for i, v := range idx {
		if v < -1 || int(v) >= opt.N {
			return nil, fmt.Errorf("model: V4.1 sparse sink row index %d out of range at %d", v, i)
		}
	}

	negInf := float32(math.Inf(-1))
	o := make([]float32, opt.B*opt.M*opt.Heads*opt.HeadDim)
	for b := 0; b < opt.B; b++ {
		for m := 0; m < opt.M; m++ {
			idxBase := (b*opt.M + m) * opt.TopK
			for h := 0; h < opt.Heads; h++ {
				qBase := ((b*opt.M+m)*opt.Heads + h) * opt.HeadDim
				// Stage 1: max over valid scaled scores, seeded with the sink.
				maxScore := negInf
				if sink != nil {
					maxScore = sink[h]
				}
				for i := 0; i < opt.TopK; i++ {
					row := int(idx[idxBase+i])
					if row < 0 {
						continue
					}
					kvBase := (b*opt.N + row) * opt.HeadDim
					var dot float32
					for d := 0; d < opt.HeadDim; d++ {
						dot += q[qBase+d] * kv[kvBase+d]
					}
					dot *= opt.Softmax
					if !model.Finite32(dot) {
						return nil, sparseSinkNonFinite(v41SinkStageScore, b, m, h, i, -1, dot)
					}
					if dot > maxScore {
						maxScore = dot
					}
				}
				if maxScore == negInf {
					// No valid row and no sink: the reference's finite bound
					// yields an all-zero output row.
					continue
				}
				// Stage 2: softmax denominator with the sink included.
				var sum float32
				if sink != nil {
					sum = exp32(sink[h] - maxScore)
				}
				for i := 0; i < opt.TopK; i++ {
					row := int(idx[idxBase+i])
					if row < 0 {
						continue
					}
					kvBase := (b*opt.N + row) * opt.HeadDim
					var dot float32
					for d := 0; d < opt.HeadDim; d++ {
						dot += q[qBase+d] * kv[kvBase+d]
					}
					term := exp32(dot*opt.Softmax - maxScore)
					if !model.Finite32(term) {
						return nil, sparseSinkNonFinite(v41SinkStageSoftmax, b, m, h, i, -1, term)
					}
					sum += term
				}
				if !model.Finite32(sum) {
					return nil, sparseSinkNonFinite(v41SinkStageSoftmax, b, m, h, -1, -1, sum)
				}
				if sum == 0 {
					continue
				}
				// Stage 3: weighted value sum, normalized by the denominator.
				oBase := qBase
				for i := 0; i < opt.TopK; i++ {
					row := int(idx[idxBase+i])
					if row < 0 {
						continue
					}
					kvBase := (b*opt.N + row) * opt.HeadDim
					var dot float32
					for d := 0; d < opt.HeadDim; d++ {
						dot += q[qBase+d] * kv[kvBase+d]
					}
					weight := exp32(dot*opt.Softmax-maxScore) / sum
					if !model.Finite32(weight) {
						return nil, sparseSinkNonFinite(v41SinkStageValue, b, m, h, i, -1, weight)
					}
					for d := 0; d < opt.HeadDim; d++ {
						o[oBase+d] += weight * kv[kvBase+d]
						if !model.Finite32(o[oBase+d]) {
							return nil, sparseSinkNonFinite(v41SinkStageValue, b, m, h, i, d, o[oBase+d])
						}
					}
				}
			}
		}
	}
	if opt.Inverse != nil {
		for b := 0; b < opt.B; b++ {
			for m := 0; m < opt.M; m++ {
				for h := 0; h < opt.Heads; h++ {
					headBase := ((b*opt.M+m)*opt.Heads + h) * opt.HeadDim
					tail := o[headBase+opt.HeadDim-opt.RopeDim : headBase+opt.HeadDim]
					if err := opt.Inverse(opt.RopeDim, tail); err != nil {
						return nil, err
					}
				}
			}
		}
	}
	return o, nil
}

// V41GroupedOutputProjection applies the V4.1 attention output projection in
// the reference order: the per-group block-diagonal wo_a over each group's
// heads, then the shared wo_b down-projection.
//
// o is row-major [B, M, Heads, HeadDim]. woA is row-major
// [Groups, OLoRARank, HeadsPerGroup*HeadDim] and woB is row-major
// [Dim, Groups*OLoRARank]. The result is a fresh row-major [B, M, Dim] tensor.
func V41GroupedOutputProjection(o, woA, woB []float32, b, m, heads, headDim, groups, oLoRARank, dim int) ([]float32, error) {
	if b <= 0 || m <= 0 || heads <= 0 || headDim <= 0 || groups <= 0 || oLoRARank <= 0 || dim <= 0 {
		return nil, fmt.Errorf("model: invalid V4.1 grouped output geometry b=%d m=%d heads=%d dim=%d groups=%d rank=%d out=%d", b, m, heads, headDim, groups, oLoRARank, dim)
	}
	if heads%groups != 0 {
		return nil, fmt.Errorf("model: V4.1 grouped output heads %d not divisible by groups %d", heads, groups)
	}
	headsPerGroup := heads / groups
	aRow := headsPerGroup * headDim
	if len(o) != b*m*heads*headDim {
		return nil, fmt.Errorf("model: V4.1 grouped output value length %d, want %d", len(o), b*m*heads*headDim)
	}
	if len(woA) != groups*oLoRARank*aRow {
		return nil, fmt.Errorf("model: V4.1 grouped output wo_a length %d, want %d", len(woA), groups*oLoRARank*aRow)
	}
	if len(woB) != dim*groups*oLoRARank {
		return nil, fmt.Errorf("model: V4.1 grouped output wo_b length %d, want %d", len(woB), dim*groups*oLoRARank)
	}
	for i, v := range o {
		if !model.Finite32(v) {
			return nil, fmt.Errorf("model: V4.1 grouped output non-finite value element %d", i)
		}
	}
	for i, v := range woA {
		if !model.Finite32(v) {
			return nil, fmt.Errorf("model: V4.1 grouped output non-finite wo_a element %d", i)
		}
	}
	for i, v := range woB {
		if !model.Finite32(v) {
			return nil, fmt.Errorf("model: V4.1 grouped output non-finite wo_b element %d", i)
		}
	}

	joined := make([]float32, b*m*groups*oLoRARank)
	for pos := 0; pos < b*m; pos++ {
		oBase := pos * heads * headDim
		for g := 0; g < groups; g++ {
			groupBase := oBase + g*aRow
			for r := 0; r < oLoRARank; r++ {
				aBase := (g*oLoRARank + r) * aRow
				var acc float32
				for k := 0; k < aRow; k++ {
					acc += o[groupBase+k] * woA[aBase+k]
				}
				joined[pos*groups*oLoRARank+g*oLoRARank+r] = acc
			}
		}
	}

	out := make([]float32, b*m*dim)
	for pos := 0; pos < b*m; pos++ {
		joinedBase := pos * groups * oLoRARank
		for d := 0; d < dim; d++ {
			bBase := d * groups * oLoRARank
			var acc float32
			for k := 0; k < groups*oLoRARank; k++ {
				acc += woB[bBase+k] * joined[joinedBase+k]
			}
			out[pos*dim+d] = acc
		}
	}
	return out, nil
}

// sparseSinkNonFinite builds the typed refusal naming the TRUE producing
// stage, batch, position, head, the KV row (i>=0) or -1, the offending output
// element d (>=0) or -1, and the non-finite value. It wraps
// ErrV41SparseSinkNonFinite so errors.Is reaches the closed class; the message
// carries the "sparse sink" and stage tokens a caller can attribute.
func sparseSinkNonFinite(stage v41SparseSinkStage, b, m, h, i, d int, v float32) error {
	where := fmt.Sprintf("sparse sink %s produced a non-finite value b=%d m=%d h=%d", stage, b, m, h)
	if i >= 0 {
		where += fmt.Sprintf(" kvRow=%d", i)
	}
	if d >= 0 {
		where += fmt.Sprintf(" element=%d", d)
	}
	return fmt.Errorf("%w: %s value=%v", ErrV41SparseSinkNonFinite, where, v)
}

func exp32(x float32) float32 { return float32(math.Exp(float64(x))) }
