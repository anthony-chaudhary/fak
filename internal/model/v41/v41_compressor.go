package v41

// This file adapts the token-group pooling state machine from
// deepseek-ai/DeepSeek-V4.1-Flash, inference/model.py (Compressor), revision
// dba1be0a40aa45a94ad051997016db3960a90277.
//
// MIT License
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

import (
	"fmt"
	"math"

	model "github.com/anthony-chaudhary/fak/internal/model"
)

// v41CompressorPool owns the causal partial-group state for one sequence and
// one compressed-attention layer. Its input is deliberately projected: callers
// supply the fp32 results of the checkpoint's wkv and wgate operations. Its
// output is the fp32 pooled latent before the checkpoint's learned RMSNorm,
// dtype conversion, RoPE, quantization, or cache publication.
type v41CompressorPool struct {
	ratio   int
	width   int
	nextPos int
	filled  int
	kv      []float32
	scores  []float32
}

func newV41CompressorPool(ratio, width int) (*v41CompressorPool, error) {
	if ratio < 1 {
		return nil, fmt.Errorf("model: V41 compressor ratio must be positive, got %d", ratio)
	}
	if width < 1 {
		return nil, fmt.Errorf("model: V41 compressor width must be positive, got %d", width)
	}
	if ratio > int(^uint(0)>>1)/width {
		return nil, fmt.Errorf("model: V41 compressor geometry %d*%d overflows", ratio, width)
	}
	return &v41CompressorPool{
		ratio:  ratio,
		width:  width,
		kv:     make([]float32, ratio*width),
		scores: make([]float32, ratio*width),
	}, nil
}

// pushNormalized extends the projected-input pooling boundary through the
// reference compressor's dtype and learned-normalization tail: pooled values
// are cast to BF16, RMSNorm is evaluated in FP32, the learned weight is applied,
// and the result is cast back to BF16. For ratio 1 the projected KV enters this
// tail through the same BF16 boundary; projectedScore is unused.
func (c *v41CompressorPool) pushNormalized(pos int, projectedKV, projectedScore, normWeight []float32, eps float32) ([]float32, bool, error) {
	if c == nil || len(normWeight) != c.width {
		return nil, false, fmt.Errorf("model: V41 compressor norm width %d does not match %d", len(normWeight), compressorWidth(c))
	}
	if !model.Finite32(eps) || eps <= 0 {
		return nil, false, fmt.Errorf("model: V41 compressor norm epsilon must be finite and positive")
	}
	for i, weight := range normWeight {
		if !model.Finite32(weight) {
			return nil, false, fmt.Errorf("model: V41 compressor norm weight[%d] is non-finite", i)
		}
	}
	pooled, emitted, err := c.push(pos, projectedKV, projectedScore)
	if err != nil || !emitted {
		return nil, emitted, err
	}
	var meanSquare float32
	for i := range pooled {
		pooled[i] = roundV41BF16(pooled[i])
		meanSquare += pooled[i] * pooled[i]
	}
	rstd := float32(1 / math.Sqrt(float64(meanSquare/float32(c.width)+eps)))
	for i := range pooled {
		pooled[i] = roundV41BF16(pooled[i] * rstd * normWeight[i])
	}
	return pooled, true, nil
}

func compressorWidth(c *v41CompressorPool) int {
	if c == nil {
		return 0
	}
	return c.width
}

// roundV41BF16 returns the FP32 widening of an IEEE BF16 round-to-nearest-even
// cast, matching torch's bfloat16 boundary while keeping this state primitive
// represented as ordinary Go float32 values.
func roundV41BF16(value float32) float32 {
	bits := math.Float32bits(value)
	bits += 0x7fff + ((bits >> 16) & 1)
	return math.Float32frombits(bits & 0xffff0000)
}

// push consumes one causal position. For ratio > 1 it emits only after the
// current group has ratio rows, matching the reference decode path. Softmax is
// independent for every latent dimension across the token-group axis.
func (c *v41CompressorPool) push(pos int, projectedKV, projectedScore []float32) ([]float32, bool, error) {
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
		if !model.Finite32(value) {
			return nil, false, fmt.Errorf("model: V41 compressor KV[%d] is non-finite", i)
		}
	}
	if c.ratio == 1 {
		c.nextPos++
		return append([]float32(nil), projectedKV...), true, nil
	}
	for i, value := range projectedScore {
		if !model.Finite32(value) {
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
