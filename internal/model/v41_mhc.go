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
	return v41MHCSplitObserved(mixes, scale, base, hc, iters, eps, nil)
}

func v41MHCSplitObserved(mixes, scale, base []float32, hc, iters int, eps float32, observe func(string)) (v41MHCMix, error) {
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
	if observe != nil {
		observe("mhc-streams")
	}
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
	return v41MHCPreObserved(streams, pre, nil)
}

func v41MHCPreObserved(streams [][]float32, pre []float32, observe func(string)) ([]float32, error) {
	if len(streams) != 4 || len(pre) != 4 || len(streams[0]) == 0 {
		return nil, fmt.Errorf("model: invalid V41 mHC pre shape")
	}
	out := make([]float32, len(streams[0]))
	for h := range streams {
		if len(streams[h]) != len(out) {
			return nil, fmt.Errorf("model: inconsistent V41 mHC stream width")
		}
		if h == 0 && observe != nil {
			observe("mhc-streams")
		}
		for d := range out {
			out[d] += float32(pre[h] * streams[h][d])
		}
	}
	return out, nil
}

func v41MHCPost(x []float32, residual [][]float32, post []float32, comb []float32) ([][]float32, error) {
	return v41MHCPostObserved(x, residual, post, comb, nil)
}

func v41MHCPostObserved(x []float32, residual [][]float32, post []float32, comb []float32, observe func(string)) ([][]float32, error) {
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
			if dst == 0 && src == 0 && observe != nil {
				observe("mhc-streams")
			}
			for d := range x {
				out[dst][d] += float32(comb[src*4+dst] * residual[src][d])
			}
		}
		for d := range x {
			out[dst][d] += float32(post[dst] * x[d])
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
	// device is the forward's selected mHC seam; nil keeps the host reference.
	device *v41MHCDevice
	// resident is the device-resident stream panel when device is resident;
	// the host stream slices are then placeholders, never authoritative.
	resident *v41MHCResident
	// preMixOverlap enables the sglang-style pre-mix hand-off on the role
	// layer-step seam: each role layer's FFN window stages the next role
	// layer's attention mix stats and pre-collapse below, and that layer's
	// attention preparation consumes them instead of recomputing. Off by
	// default; a pure scheduling change, never a numeric one.
	preMixOverlap bool
	// staged* hold at most one pending hand-off: the attention mix stats and
	// pre-collapse computed for stagedLayer during the previous role layer's
	// FFN window, published against the graph cursor as it stood then. A
	// consumer accepts them only when it is that layer and the cursor has
	// moved exactly onto it, so an intervening step or panel mutation falls
	// back to the serial computation.
	staged          bool
	stagedLayer     int
	stagedCursor    int
	stagedMix       v41MHCMix
	stagedCollapsed []float32
}

func newV41MHCCarry(rows int) *v41MHCCarry {
	c := &v41MHCCarry{pre: make([][]float32, rows), stagedLayer: -1, stagedCursor: -1}
	for i := range c.pre {
		c.pre[i] = []float32{1, 0, 0, 0}
	}
	return c
}

// EnablePreMixOverlap turns on the sglang-style pre-mix hand-off for this
// carry's role layer steps. The plain-layer seam stays serial either way.
func (c *v41MHCCarry) EnablePreMixOverlap() {
	if c != nil {
		c.preMixOverlap = true
	}
}

// stageAttention publishes one pending pre-mix hand-off, replacing any older
// entry. The caller passes owned coefficient and collapse rows computed from
// the just-published stream panel; publication tags the graph cursor so a
// consumer can prove no step intervened.
func (c *v41MHCCarry) stageAttention(layer int, mix v41MHCMix, collapsed []float32) {
	if c == nil {
		return
	}
	c.staged = true
	c.stagedLayer = layer
	c.stagedCursor = c.nextLayer
	c.stagedMix = mix
	c.stagedCollapsed = collapsed
}

// takeStagedAttention returns the pending pre-mix hand-off for layer l exactly
// once. It refuses (ok=false, slot untouched) unless l is the staged layer and
// the graph cursor sits exactly where the publication was made, and clears the
// slot on success so a replayed step recomputes serially.
func (c *v41MHCCarry) takeStagedAttention(l int) (v41MHCMix, []float32, bool) {
	if c == nil || !c.staged || c.stagedLayer != l || c.nextLayer != c.stagedCursor ||
		len(c.stagedMix.pre) != 4 || len(c.stagedMix.post) != 4 || len(c.stagedMix.comb) != 16 ||
		len(c.stagedCollapsed) == 0 {
		return v41MHCMix{}, nil, false
	}
	mix, collapsed := c.stagedMix, c.stagedCollapsed
	c.staged = false
	c.stagedLayer = -1
	c.stagedCursor = -1
	c.stagedMix = v41MHCMix{}
	c.stagedCollapsed = nil
	return mix, collapsed, true
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
	return v41MHCBF16Observed(layer, values, nil)
}

func v41MHCBF16Observed(layer int, values []float32, observe func(string)) ([]float32, error) {
	return v41MHCBF16IntoObserved(layer, make([]float32, len(values)), values, observe)
}

// v41MHCBF16Into may round in place only when its caller owns the entire row.
// out must have len(values). The allocating wrapper above retains the
// borrowed-input publication contract.
func v41MHCBF16Into(layer int, out, values []float32) ([]float32, error) {
	return v41MHCBF16IntoObserved(layer, out, values, nil)
}

func v41MHCBF16IntoObserved(layer int, out, values []float32, observe func(string)) ([]float32, error) {
	for i, v := range values {
		if i == 0 && observe != nil {
			observe("bf16-boundary")
		}
		out[i] = v41RoundBF16(v)
		if !finite32(v) || !finite32(out[i]) {
			return nil, v41StageErr(v41StageMHC, layer, fmt.Errorf("%w: non-finite full mHC BF16 value %d", ErrV41ForwardStage, i))
		}
	}
	return out, nil
}

func v41FullInitialStreams(x []float32) ([][]float32, error) {
	return v41FullInitialStreamsObserved(x, nil)
}

func v41FullInitialStreamsObserved(x []float32, observe func(string)) ([][]float32, error) {
	row, err := v41MHCBF16Observed(-1, x, observe)
	if err != nil {
		return nil, err
	}
	streams := make([][]float32, 4)
	if len(row) != 0 {
		if observe != nil {
			observe("mhc-streams")
		}
		streams[0] = row
	}
	for h := 1; h < len(streams); h++ {
		streams[h] = append([]float32(nil), row...)
	}
	return streams, nil
}

func v41MHCPreBF16(layer int, streams [][]float32, pre []float32) ([]float32, error) {
	return v41MHCPreBF16Observed(layer, streams, pre, nil)
}

func v41MHCPreBF16Observed(layer int, streams [][]float32, pre []float32, observe func(string)) ([]float32, error) {
	row, err := v41MHCPreObserved(streams, pre, observe)
	if err != nil {
		return nil, v41StageErr(v41StageMHC, layer, err)
	}
	return v41MHCBF16IntoObserved(layer, row, row, observe)
}

func v41MHCPostBF16(layer int, x []float32, streams [][]float32, mix v41MHCMix) ([][]float32, error) {
	return v41MHCPostBF16Observed(layer, x, streams, mix, nil)
}

func v41MHCPostBF16Observed(layer int, x []float32, streams [][]float32, mix v41MHCMix, observe func(string)) ([][]float32, error) {
	row, err := v41MHCBF16Observed(layer, x, observe)
	if err != nil {
		return nil, err
	}
	next, err := v41MHCPostObserved(row, streams, mix.post, mix.comb, observe)
	if err != nil {
		return nil, v41StageErr(v41StageMHC, layer, err)
	}
	for h := range next {
		next[h], err = v41MHCBF16IntoObserved(layer, next[h], next[h], observe)
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
	var device *v41MHCDevice
	if scratch != nil && scratch.mhcCarry != nil {
		device = scratch.mhcCarry.device
	}
	updated, err := v41MHCPostBF16On(device, layer, attnOut, streams, attnMix)
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
	mix, err := v41MHCSplitOn(device, layer, projected, m.v41LeafTensor(layer, v41LeafMHCFFNScale), m.v41LeafTensor(layer, v41LeafMHCFFNBase), 4, hcItersOrDefault(m.Cfg), hcEpsOrDefault(m.Cfg))
	if err != nil {
		return nil, v41MHCMix{}, nil, err
	}
	collapsed, err := v41MHCPreBF16On(device, layer, updated, attnMix.pre)
	if err != nil {
		return nil, v41MHCMix{}, nil, err
	}
	normalized, err := m.v41FFNNormWithPublish(layer, collapsed, float32(m.Cfg.RMSNormEps), scratch.ffnNorm.normFunc(), scratch.gatherOps.bf16PublishFunc())
	return updated, mix, normalized, err
}

// v41MHCStageNextAttention computes layer next's attention-sublayer mHC mix
// stats (projection + split) and its pre-collapse during layer current's FFN
// window tail, adapting the sglang pre-mix hand-off: the next sublayer's
// coefficients run beside the current sublayer's post-mix instead of opening
// the next layer's window. It mirrors the serial arithmetic of layer next's own
// attention preparation exactly -- same projector, split and pre-collapse, same
// device seam and observer -- so a staged coefficient set is byte-identical to
// the serial control.
//
// Fail-closed and self-skipping: it returns nil without touching the carry when
// the schedule cannot consume the result (overlap disabled, no next layer, a
// plain next layer, or a resident panel), and refuses with the typed
// ErrV41ForwardStage when the next layer's admitted geometry or the panel is
// malformed.
func (m *Model) v41MHCStageNextAttention(current, next int, streams [][]float32, scratch *v41ProjScratch) error {
	cfg := m.Cfg
	full, err := v41ForwardGeometry(cfg)
	if err != nil {
		return err
	}
	if !full || next <= current || next >= cfg.NumLayers {
		return nil
	}
	if scratch == nil || scratch.mhcCarry == nil || !scratch.mhcCarry.preMixOverlap || scratch.mhcCarry.resident != nil {
		return nil
	}
	plan, err := v41AttentionPlanFor(cfg, next, m.v41AttentionRolesCached())
	if err != nil {
		return v41StageErr(v41StageMHC, next, err)
	}
	// Only the role seam consumes a staged hand-off; a plain next layer would
	// recompute serially, so staging for it is wasted work, not overlap.
	if plan.Role == V41AttentionRolePerLayer && plan.Ratio == 0 {
		return nil
	}
	H := cfg.HiddenSize
	if len(streams) != 4 {
		return v41StageErr(v41StageMHC, next, errV41ProjectionResult)
	}
	for s := range streams {
		if len(streams[s]) != H {
			return v41StageErr(v41StageMHC, next, errV41ProjectionResult)
		}
	}
	if len(scratch.mhcCarry.pre) == 0 {
		return v41StageErr(v41StageMHC, next, errV41ProjectionResult)
	}
	if err := m.v41AdmitMHC(next); err != nil {
		return err
	}
	mhcFlat, mhcTransposed, ok := m.v41MHCWeightLayout(next)
	if !ok {
		return v41StageErr(v41StageMHC, next,
			fmt.Errorf("%w: mHC mix weight holds no admitted geometry", ErrV41ForwardStage))
	}
	width, valid := checkedMulInt(4, H)
	if !valid {
		return v41StageErr(v41StageMHC, next, errV41ProjectionResult)
	}
	input := make([]float32, 0, width)
	for _, stream := range streams {
		input = append(input, stream...)
	}
	device := scratch.mhcCarry.device
	observe := scratch.v41StepMHCObserver()
	eps := float32(cfg.RMSNormEps)
	projectMHC := m.v41MHCProjectorObserved(next, H, eps, mhcFlat, mhcTransposed, scratch, observe)
	mixes, err := projectMHC(input)
	if err != nil {
		return err
	}
	mix, err := v41MHCSplitOnObserved(device, next, mixes, m.v41LeafTensor(next, v41LeafMHCScale), m.v41LeafTensor(next, v41LeafMHCBase), 4, hcItersOrDefault(cfg), hcEpsOrDefault(cfg), observe)
	if err != nil {
		return err
	}
	var collapsed []float32
	if device != nil {
		collapsed, err = v41MHCPreBF16OnObserved(device, next, streams, scratch.mhcCarry.pre[0], observe)
	} else {
		collapsed, err = v41MHCPreObserved(streams, scratch.mhcCarry.pre[0], observe)
	}
	if err != nil {
		return v41StageErr(v41StageMHC, next, err)
	}
	scratch.mhcCarry.stageAttention(next, mix, collapsed)
	return nil
}
