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

package model

import (
	"errors"
	"fmt"
	"math"
	"sync"
)

// ErrV41SparseSinkNonFinite identifies arithmetic overflow or a non-finite
// inverse-rotation result in V41SparseAttentionSink. No partial output is
// returned; callers retain their existing attention-stage rollback boundary.
var ErrV41SparseSinkNonFinite = errors.New("model: DeepSeek V4.1 sparse sink produced a non-finite value")

type v41SparseSinkStage string

const (
	v41SinkStageScore   v41SparseSinkStage = "score accumulate"
	v41SinkStageSoftmax v41SparseSinkStage = "softmax denominator"
	v41SinkStageValue   v41SparseSinkStage = "weighted value accumulate"
	v41SinkStageInverse v41SparseSinkStage = "inverse rotation"
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
	// TopKLength, when non-nil, is the row-major [B, M] count of valid slots in
	// each query's index list. Slots at position >= TopKLength[b*M+m] are masked
	// out exactly as a -1 index is, before the max and the softmax: the pinned
	// reference overwrites them with -1 (mask = arange(topk) >= topk_length).
	// A nil slice means every query uses all TopK slots. Each value must be in
	// [0, TopK]; a negative or over-long length is refused.
	TopKLength []int32
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
//	a slot at i >= topk_length[b,m] is invalid too, exactly like -1
//	o[b,m,h,:] = sum_i softmax_i * kv[b,idx_i,:], with exp(sink[h]-max)
//	             folded into the softmax denominator
//
// q is row-major [B, M, Heads, HeadDim]; kv is row-major [B, N, HeadDim],
// shared across heads; idx is row-major [B, M, TopK] with -1 marking an empty
// slot. sink, when non-nil, is [Heads]. TopKLength, when non-nil, is the
// row-major [B, M] valid-slot count that masks the trailing padding slots. The result is a fresh row-major
// [B, M, Heads, HeadDim] tensor. A row whose indices are all -1 yields an
// all-zero output, matching the reference's finite lower bound. Every input is
// validated and every non-finite value is refused before any output is
// produced. Products are rounded explicitly so no architecture fuses them into
// FMA: the host result is the same rounded F32 on arm64 and amd64.
func V41SparseAttentionSink(q, kv []float32, sink []float32, idx []int32, opt V41SparseAttentionSinkOptions) ([]float32, error) {
	if opt.B <= 0 || opt.M <= 0 || opt.Heads <= 0 || opt.HeadDim <= 0 || opt.TopK < 0 || opt.N < 0 {
		return nil, fmt.Errorf("model: invalid V4.1 sparse sink geometry b=%d m=%d heads=%d dim=%d topk=%d n=%d", opt.B, opt.M, opt.Heads, opt.HeadDim, opt.TopK, opt.N)
	}
	if !finite32(opt.Softmax) || opt.Softmax == 0 {
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
	if opt.TopKLength != nil && len(opt.TopKLength) != opt.B*opt.M {
		return nil, fmt.Errorf("model: V4.1 sparse sink topk length %d, want %d", len(opt.TopKLength), opt.B*opt.M)
	}
	if opt.TopKLength != nil {
		for i, v := range opt.TopKLength {
			if v < 0 || int(v) > opt.TopK {
				return nil, fmt.Errorf("model: V4.1 sparse sink topk length %d out of range at %d", v, i)
			}
		}
	}
	for i, v := range q {
		if !finite32(v) {
			return nil, fmt.Errorf("model: V4.1 sparse sink non-finite query element %d", i)
		}
	}
	for i, v := range kv {
		if !finite32(v) {
			return nil, fmt.Errorf("model: V4.1 sparse sink non-finite kv element %d", i)
		}
	}
	if sink != nil {
		for i, v := range sink {
			if !finite32(v) {
				return nil, fmt.Errorf("model: V4.1 sparse sink non-finite sink value %d", i)
			}
		}
	}
	for i, v := range idx {
		if v < -1 || int(v) >= opt.N {
			return nil, fmt.Errorf("model: V4.1 sparse sink row index %d out of range at %d", v, i)
		}
	}

	return v41SparseAttentionSinkChecked(q, sink, idx, opt, v41SparseSinkRows{flat: kv})
}

// v41SparseSinkRows is a synchronous read-only view. Flat storage is used only
// after the public primitive has validated it; split rows are used only by the
// combined host caller after it validates every original row and canonical ID.
// No callback runs between that validation and this contraction; no view escapes.
type v41SparseSinkRows struct {
	flat               []float32
	window, compressed [][]float32
}

func (r v41SparseSinkRows) row(batch, index, count, width int) []float32 {
	if r.flat != nil {
		lo := (batch*count + index) * width
		return r.flat[lo : lo+width : lo+width]
	}
	if index < len(r.window) {
		return r.window[index]
	}
	return r.compressed[index-len(r.window)]
}

// v41SparseAttentionSinkChecked contains only the shared numerical contraction.
// Callers must finish the full public or combined validation before constructing
// its local row view. It is not an admission flag or a retained validation cache.
func v41SparseAttentionSinkChecked(q, sink []float32, idx []int32, opt V41SparseAttentionSinkOptions, rows v41SparseSinkRows) ([]float32, error) {
	negInf := float32(math.Inf(-1))
	o := make([]float32, opt.B*opt.M*opt.Heads*opt.HeadDim)
	// Independent heads share only immutable inputs. Saturate the scheduling
	// estimate rather than adding a new geometry refusal to the public API.
	work := opt.HeadDim
	maxInt := int(^uint(0) >> 1)
	for _, factor := range []int{opt.TopK, opt.Heads} {
		if factor > 0 && work > maxInt/factor {
			work = maxInt
		} else {
			work *= factor
		}
	}
	for b := 0; b < opt.B; b++ {
		for m := 0; m < opt.M; m++ {
			idxBase := (b*opt.M + m) * opt.TopK
			// validCount is the number of leading slots that count; slots at
			// index >= validCount are padding and contribute nothing.
			validCount := opt.TopK
			if opt.TopKLength != nil {
				validCount = int(opt.TopKLength[b*opt.M+m])
			}
			var firstErr error
			firstHead := opt.Heads
			var errorMu sync.Mutex
			parForRange(opt.Heads, work, func(lo, hi int) {
				// One score/exp scratch per active chunk, reused across its heads.
				terms := make([]float32, min(opt.TopK, len(idx)))
				contract := func(h int) error {
					qBase := ((b*opt.M+m)*opt.Heads + h) * opt.HeadDim
					// Stage 1: max over valid scaled scores, seeded with the sink.
					maxScore := negInf
					if sink != nil {
						maxScore = sink[h]
					}
					for i := 0; i < validCount; i++ {
						row := int(idx[idxBase+i])
						if row < 0 {
							continue
						}
						values := rows.row(b, row, opt.N, opt.HeadDim)
						var dot float32
						for d := 0; d < opt.HeadDim; d++ {
							dot += float32(q[qBase+d] * values[d])
						}
						dot *= opt.Softmax
						if !finite32(dot) {
							return sparseSinkNonFinite(v41SinkStageScore, b, m, h, row, -1, dot)
						}
						terms[i] = dot
						if dot > maxScore {
							maxScore = dot
						}
					}
					if maxScore == negInf {
						// No valid row and no sink: the reference's finite bound
						// yields an all-zero output row.
						return nil
					}
					// Stage 2: softmax denominator with the sink included.
					var sum float32
					if sink != nil {
						sum = exp32(sink[h] - maxScore)
					}
					for i := 0; i < validCount; i++ {
						row := int(idx[idxBase+i])
						if row < 0 {
							continue
						}
						// Match the rounded float32 score used by the maximum pass.
						// A finite score gap may round to -Inf; exp(-Inf)=0 is valid.
						term := exp32(terms[i] - maxScore)
						if !finite32(term) {
							return sparseSinkNonFinite(v41SinkStageSoftmax, b, m, h, row, -1, term)
						}
						terms[i] = term
						sum += term
					}
					if !finite32(sum) {
						return sparseSinkNonFinite(v41SinkStageSoftmax, b, m, h, -1, -1, sum)
					}
					if sum == 0 {
						return nil
					}
					// Stage 3: weighted value sum, normalized by the denominator.
					oBase := qBase
					for i := 0; i < validCount; i++ {
						row := int(idx[idxBase+i])
						if row < 0 {
							continue
						}
						values := rows.row(b, row, opt.N, opt.HeadDim)
						weight := terms[i] / sum
						if !finite32(weight) {
							return sparseSinkNonFinite(v41SinkStageValue, b, m, h, row, -1, weight)
						}
						for d := 0; d < opt.HeadDim; d++ {
							o[oBase+d] += float32(weight * values[d])
							if !finite32(o[oBase+d]) {
								return sparseSinkNonFinite(v41SinkStageValue, b, m, h, row, d, o[oBase+d])
							}
						}
					}
					return nil
				}
				for h := lo; h < hi; h++ {
					if err := contract(h); err != nil {
						errorMu.Lock()
						if h < firstHead {
							firstHead, firstErr = h, err
						}
						errorMu.Unlock()
						// Later heads in this chunk cannot precede its first failure.
						break
					}
				}
			})
			// All writers drain before error selection or any inverse callback.
			if firstErr != nil {
				return nil, firstErr
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
		// A callback can retain a previously supplied tail. Validate the final
		// tensor after every callback has returned, not only its latest tail.
		for i, value := range o {
			if !finite32(value) {
				head := i / opt.HeadDim
				return nil, sparseSinkNonFinite(v41SinkStageInverse, head/(opt.M*opt.Heads), head/opt.Heads%opt.M, head%opt.Heads, -1, i%opt.HeadDim, value)
			}
		}
	}
	return o, nil
}

func sparseSinkNonFinite(stage v41SparseSinkStage, b, m, h, row, d int, value float32) error {
	where := fmt.Sprintf("sparse sink %s produced a non-finite value b=%d m=%d h=%d", stage, b, m, h)
	if row >= 0 {
		where += fmt.Sprintf(" kvRow=%d", row)
	}
	if d >= 0 {
		where += fmt.Sprintf(" element=%d", d)
	}
	return fmt.Errorf("%w: %s value=%v", ErrV41SparseSinkNonFinite, where, value)
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
		if !finite32(v) {
			return nil, fmt.Errorf("model: V4.1 grouped output non-finite value element %d", i)
		}
	}
	for i, v := range woA {
		if !finite32(v) {
			return nil, fmt.Errorf("model: V4.1 grouped output non-finite wo_a element %d", i)
		}
	}
	for i, v := range woB {
		if !finite32(v) {
			return nil, fmt.Errorf("model: V4.1 grouped output non-finite wo_b element %d", i)
		}
	}

	// Each position consumes its grouped projection before the next position.
	// Reuse one joined row instead of retaining the full prompt intermediate.
	joined := make([]float32, groups*oLoRARank)
	out := make([]float32, b*m*dim)
	for pos := 0; pos < b*m; pos++ {
		oBase := pos * heads * headDim
		for g := 0; g < groups; g++ {
			groupBase := oBase + g*aRow
			for r := 0; r < oLoRARank; r++ {
				aBase := (g*oLoRARank + r) * aRow
				var acc float32
				for k := 0; k < aRow; k++ {
					acc += float32(o[groupBase+k] * woA[aBase+k])
				}
				joined[g*oLoRARank+r] = acc
			}
		}
		for d := 0; d < dim; d++ {
			bBase := d * groups * oLoRARank
			var acc float32
			for k := 0; k < groups*oLoRARank; k++ {
				acc += float32(woB[bBase+k] * joined[k])
			}
			out[pos*dim+d] = acc
		}
	}
	return out, nil
}

func exp32(x float32) float32 { return float32(math.Exp(float64(x))) }
