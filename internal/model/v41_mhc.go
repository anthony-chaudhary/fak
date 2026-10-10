package model

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
	if hc != 4 || iters < 1 || !finite32(eps) || eps <= 0 || len(mixes) != n || len(base) != n || len(scale) != 3 {
		return v41MHCMix{}, fmt.Errorf("model: invalid V41 mHC geometry hc=%d iters=%d mixes=%d base=%d scale=%d eps=%g", hc, iters, len(mixes), len(base), len(scale), eps)
	}
	for i := range mixes {
		if !finite32(mixes[i]) || !finite32(base[i]) {
			return v41MHCMix{}, fmt.Errorf("model: non-finite V41 mHC coefficient at %d", i)
		}
	}
	for i := range scale {
		if !finite32(scale[i]) {
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

// v41MHCCarry is invocation-local, with one F32 pre-mix row per token. It is
// never KV history or snapshot state. A layer publishes its carry only with its
// completed stream panel, so a refused layer cannot advance the graph cursor.
type v41MHCCarry struct {
	pre       [][]float32
	nextLayer int
}

func newV41MHCCarry(rows int) *v41MHCCarry {
	c := &v41MHCCarry{pre: make([][]float32, rows)}
	for i := range c.pre {
		c.pre[i] = []float32{1, 0, 0, 0}
	}
	return c
}

func (c *v41MHCCarry) validate(layer, rows int) error {
	if c == nil || c.nextLayer != layer || len(c.pre) != rows {
		return v41StageErr(v41StageMHC, layer, fmt.Errorf("%w: missing or out-of-order full mHC carry", ErrV41ForwardStage))
	}
	for _, pre := range c.pre {
		if len(pre) != 4 {
			return v41StageErr(v41StageMHC, layer, errV41ProjectionResult)
		}
		for _, v := range pre {
			if !finite32(v) {
				return v41StageErr(v41StageMHC, layer, errV41ProjectionResult)
			}
		}
	}
	return nil
}

func v41MHCMixLeaf(leaf string) bool {
	return leaf == "mhc.mixes.weight" || leaf == "mhc.ffn_mixes.weight"
}

// The pinned model's residuals, sublayer results and hc_pre/hc_post publications
// are BF16. Keep coefficients and arithmetic F32; own every rounded publication.
func v41MHCBF16(layer int, values []float32) ([]float32, error) {
	out := make([]float32, len(values))
	for i, v := range values {
		out[i] = v41RoundBF16(v)
		if !finite32(v) || !finite32(out[i]) {
			return nil, v41StageErr(v41StageMHC, layer, fmt.Errorf("%w: non-finite full mHC BF16 value %d", ErrV41ForwardStage, i))
		}
	}
	return out, nil
}

func v41FullInitialStreams(x []float32) ([][]float32, error) {
	row, err := v41MHCBF16(-1, x)
	if err != nil {
		return nil, err
	}
	streams := make([][]float32, 4)
	for h := range streams {
		streams[h] = append([]float32(nil), row...)
	}
	return streams, nil
}

func v41MHCPreBF16(layer int, streams [][]float32, pre []float32) ([]float32, error) {
	row, err := v41MHCPre(streams, pre)
	if err != nil {
		return nil, v41StageErr(v41StageMHC, layer, err)
	}
	return v41MHCBF16(layer, row)
}

func v41MHCPostBF16(layer int, x []float32, streams [][]float32, mix v41MHCMix) ([][]float32, error) {
	row, err := v41MHCBF16(layer, x)
	if err != nil {
		return nil, err
	}
	next, err := v41MHCPost(row, streams, mix.post, mix.comb)
	if err != nil {
		return nil, v41StageErr(v41StageMHC, layer, err)
	}
	for h := range next {
		next[h], err = v41MHCBF16(layer, next[h])
		if err != nil {
			return nil, err
		}
	}
	return next, nil
}

func (m *Model) v41FFNMHCProjector(layer int, scratch *v41ProjScratch) (func([]float32) ([]float32, error), error) {
	if err := m.v41AdmitMHCNamed(layer, "mhc.ffn_mixes.weight", "mhc.ffn_base", "mhc.ffn_scale", true); err != nil {
		return nil, err
	}
	_, transposed, _ := m.v41MHCWeightLayoutNamed(layer, "mhc.ffn_mixes.weight")
	return m.v41MHCProjectorNamed(layer, "mhc.ffn_mixes.weight", m.Cfg.HiddenSize, float32(m.Cfg.RMSNormEps), true, transposed, scratch), nil
}

// The attention projector must be dead before this phase starts: projectFFN
// may reuse its materialization buffer. The returned streams and norm row are
// owned; the caller retains incoming streams/carry until all later work succeeds.
func (m *Model) v41FullFFNInput(layer int, streams [][]float32, attnMix v41MHCMix, attnOut []float32, projectFFN func([]float32) ([]float32, error), scratch *v41ProjScratch) ([][]float32, v41MHCMix, []float32, error) {
	updated, err := v41MHCPostBF16(layer, attnOut, streams, attnMix)
	if err != nil {
		return nil, v41MHCMix{}, nil, err
	}
	flat := make([]float32, 0, 4*m.Cfg.HiddenSize)
	for _, row := range updated {
		flat = append(flat, row...)
	}
	projected, err := projectFFN(flat)
	if err != nil {
		return nil, v41MHCMix{}, nil, err
	}
	mix, err := v41MHCSplit(projected, m.tensor(layerName(layer, "mhc.ffn_scale")), m.tensor(layerName(layer, "mhc.ffn_base")), 4, hcItersOrDefault(m.Cfg), hcEpsOrDefault(m.Cfg))
	if err != nil {
		return nil, v41MHCMix{}, nil, v41StageErr(v41StageMHC, layer, err)
	}
	collapsed, err := v41MHCPreBF16(layer, updated, attnMix.pre)
	if err != nil {
		return nil, v41MHCMix{}, nil, err
	}
	normalized, err := m.v41FFNNorm(layer, collapsed, float32(m.Cfg.RMSNormEps), scratch.ffnNorm)
	return updated, mix, normalized, err
}
