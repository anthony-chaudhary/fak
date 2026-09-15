package v41

// This file adapts DeepSeek V4.1 Flash's mHC coefficient split, Sinkhorn
// normalization, and pre/post mixing from inference/kernel.py and
// inference/model.py at revision dba1be0a40aa45a94ad051997016db3960a90277.
//
// MIT License
// Copyright (c) 2023 DeepSeek
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

import (
	"fmt"
	model "github.com/anthony-chaudhary/fak/internal/model"
	"math"
)

type v41MHCMix struct {
	pre, post []float32
	comb      []float32 // row-major [source stream, destination stream]
}

// v41MHCSplit consumes the already projected and whole-stream-normalized mix
// vector. Projection is a caller boundary; this function begins at the pinned
// hc_split_sinkhorn kernel's inputs.
func v41MHCSplit(mixes, scale, base []float32, hc, iters int, eps float32) (v41MHCMix, error) {
	n := (2 + hc) * hc
	if hc != 4 || iters < 1 || !model.Finite32(eps) || eps <= 0 || len(mixes) != n || len(base) != n || len(scale) != 3 {
		return v41MHCMix{}, fmt.Errorf("model: invalid V41 mHC geometry hc=%d iters=%d mixes=%d base=%d scale=%d eps=%g", hc, iters, len(mixes), len(base), len(scale), eps)
	}
	for i := range mixes {
		if !model.Finite32(mixes[i]) || !model.Finite32(base[i]) {
			return v41MHCMix{}, fmt.Errorf("model: non-finite V41 mHC coefficient at %d", i)
		}
	}
	for i := range scale {
		if !model.Finite32(scale[i]) {
			return v41MHCMix{}, fmt.Errorf("model: non-finite V41 mHC scale at %d", i)
		}
	}
	out := v41MHCMix{pre: make([]float32, hc), post: make([]float32, hc), comb: make([]float32, hc*hc)}
	for j := 0; j < hc; j++ {
		out.pre[j] = sigmoid32(mixes[j]*scale[0]+base[j]) + eps
		out.post[j] = 2 * sigmoid32(mixes[hc+j]*scale[1]+base[hc+j])
	}
	for i := range out.comb {
		out.comb[i] = mixes[2*hc+i]*scale[2] + base[2*hc+i]
	}
	for row := 0; row < hc; row++ {
		start := row * hc
		maxValue := out.comb[start]
		for col := 1; col < hc; col++ {
			maxValue = max(maxValue, out.comb[start+col])
		}
		var sum float32
		for col := 0; col < hc; col++ {
			out.comb[start+col] = float32(math.Exp(float64(out.comb[start+col] - maxValue)))
			sum += out.comb[start+col]
		}
		for col := 0; col < hc; col++ {
			out.comb[start+col] = out.comb[start+col]/sum + eps
		}
	}
	v41MHCNormalizeColumns(out.comb, hc, eps)
	for iter := 1; iter < iters; iter++ {
		v41MHCNormalizeRows(out.comb, hc, eps)
		v41MHCNormalizeColumns(out.comb, hc, eps)
	}
	return out, nil
}

func v41MHCNormalizeRows(m []float32, hc int, eps float32) {
	for row := 0; row < hc; row++ {
		var sum float32
		for col := 0; col < hc; col++ {
			sum += m[row*hc+col]
		}
		for col := 0; col < hc; col++ {
			m[row*hc+col] /= sum + eps
		}
	}
}

func v41MHCNormalizeColumns(m []float32, hc int, eps float32) {
	for col := 0; col < hc; col++ {
		var sum float32
		for row := 0; row < hc; row++ {
			sum += m[row*hc+col]
		}
		for row := 0; row < hc; row++ {
			m[row*hc+col] /= sum + eps
		}
	}
}

func sigmoid32(x float32) float32 { return float32(1 / (1 + math.Exp(float64(-x)))) }

func v41MHCPre(streams [][]float32, pre []float32) ([]float32, error) {
	if len(streams) != 4 || len(pre) != 4 || len(streams[0]) == 0 {
		return nil, fmt.Errorf("model: invalid V41 mHC pre shape")
	}
	out := make([]float32, len(streams[0]))
	for h := range streams {
		if len(streams[h]) != len(out) {
			return nil, fmt.Errorf("model: inconsistent V41 mHC stream width")
		}
		for d := range out {
			out[d] += pre[h] * streams[h][d]
		}
	}
	return out, nil
}

func v41MHCPost(x []float32, residual [][]float32, post []float32, comb []float32) ([][]float32, error) {
	if len(residual) != 4 || len(post) != 4 || len(comb) != 16 || len(x) == 0 {
		return nil, fmt.Errorf("model: invalid V41 mHC post shape")
	}
	out := make([][]float32, 4)
	for dst := 0; dst < 4; dst++ {
		out[dst] = make([]float32, len(x))
		for src := 0; src < 4; src++ {
			if len(residual[src]) != len(x) {
				return nil, fmt.Errorf("model: inconsistent V41 mHC residual width")
			}
			for d := range x {
				out[dst][d] += comb[src*4+dst] * residual[src][d]
			}
		}
		for d := range x {
			out[dst][d] += post[dst] * x[d]
		}
	}
	return out, nil
}
