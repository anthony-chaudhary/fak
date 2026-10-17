// Copyright (c) 2023 DeepSeek
//
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
//
// Native scalar adaptation of inference/model.py at
// 4dd75a8b448d95ae8e284d5c5a1e4b8808bd583b and inference/kernel.py at
// 517ef625df97ec57aadc91b67506a57c20fdc5bb in
// https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash.
// These reference-code revisions are separate from the artifact revision
// DeepSeekV41FlashRevision. No physical checkpoint completeness is asserted.
package model

import (
	"context"
	"crypto/sha256"
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"sort"
	"strconv"
	"sync"
)

var errV41DSparkReference = errors.New("model: V4.1 DSpark reference refused")

// v41DSparkContextRow is owned target data captured BEFORE the named layer's
// attention, AFTER its Engram injection. Hidden contains the BF16 mean of all
// four residual streams, not a layer output or stream-zero approximation.
// Only a future committed target capture transaction may supply serving rows.
type v41DSparkContextRow struct {
	Position, Token int
	Layers          []int
	Hidden          [][]float32
}

func v41DSparkMeanStreams(streams [][]float32, hidden int) ([]float32, error) {
	if len(streams) != 4 || hidden < 1 {
		return nil, errV41DSparkReference
	}
	out := make([]float32, hidden)
	for _, stream := range streams {
		if len(stream) != hidden {
			return nil, errV41DSparkReference
		}
		for i, value := range stream {
			if !finite32(value) {
				return nil, errV41DSparkReference
			}
			out[i] += value
		}
	}
	for i := range out {
		out[i] *= 0.25
	}
	return v41MHCBF16(-1, out)
}

// Anchor is already chosen by the target. All five Proposed tokens are new;
// verification must cover anchor plus five, not reinterpret block5 as four.
type v41DSparkProposal struct {
	Anchor, AnchorPosition                  int
	Proposed                                [5]int
	Logits                                  [5][]float32
	ConfidenceLogits, ConditionalConfidence [5]float32
	hidden                                  [5][]float32 // pre-norm collapsed rows; retained for scalar witnesses
}

// v41DSparkReference is an internal correctness path, never serving admission.
// It executes all three original-format draft stages using native Go math.
// The target Session remains exclusively caller-owned: Step/Close/restore must
// not run concurrently. Borrowed shards must remain immutable and open.
// AppendCommitted publishes all three rings atomically. Propose is read-only;
// transient noise KV never enters a persistent ring. Exact verification,
// feature-capture integration and accepted-prefix rollback remain separate.
type v41DSparkReference struct {
	mu     sync.Mutex
	w      *v41DSparkRows
	target *Session
	model  *Model
	state  *v41ForwardState
	cfg    Config
	ring   [3][v41WindowSize][]float32
	next   int
	prefix [32]byte
	closed bool
}

func newV41DSparkReference(target *Session, w *v41DSparkRows) (*v41DSparkReference, error) {
	if target == nil || target.M == nil || !target.modelWeightsHeld || target.v41Forward == nil ||
		w == nil || w.inv == nil || !target.M.Cfg.IsDeepSeekV41() || target.Backend != nil ||
		target.executionPolicy == ExecutionPolicyDeviceOnly || target.PrecisionPolicy != nil || target.PrecisionSchedule != nil {
		return nil, fmt.Errorf("%w: exclusive native host target required", errV41DSparkReference)
	}
	cfg := target.M.Cfg
	inv, err := DeepSeekV41DSparkInventory(cfg)
	if err != nil {
		return nil, err
	}
	d := cfg.DeepSeekV41.DSpark
	if cfg.NumLayers > 40 || cfg.HiddenSize > 5120 || cfg.NumHeads > 64 || cfg.HeadDim > 512 || cfg.QLoraRank > 1280 ||
		cfg.OLoraRank > 1024 || cfg.OGroups > 8 || cfg.MoEIntermediateSize > 2304 || cfg.VocabSize > 129280 ||
		d.MarkovRank > 256 || d.NumRoutedExperts != 128 || d.NumExpertsPerTok != 3 ||
		cfg.NSharedExperts != 1 || !cfg.NormTopKProb || cfg.ScoringFunc != "sqrtsoftplus" ||
		!finite32(float32(cfg.RoutedScalingFactor)) || cfg.RoutedScalingFactor <= 0 ||
		!finite32(float32(cfg.SwigluLimit)) || cfg.SwigluLimit < 0 ||
		!finite32(float32(cfg.RMSNormEps)) || float32(cfg.RMSNormEps) <= 0 ||
		hcItersOrDefault(cfg) < 1 || hcItersOrDefault(cfg) > 64 ||
		!finite32(hcEpsOrDefault(cfg)) || hcEpsOrDefault(cfg) <= 0 ||
		cfg.RopeTheta <= 0 || math.IsNaN(cfg.RopeTheta) || math.IsInf(cfg.RopeTheta, 0) ||
		cfg.QKRopeHeadDim < 2 || cfg.QKRopeHeadDim > cfg.HeadDim || cfg.QKRopeHeadDim%2 != 0 ||
		cfg.QLoraRank%32 != 0 || cfg.HeadDim%32 != 0 || (cfg.OGroups*cfg.OLoraRank)%32 != 0 ||
		(cfg.NumHeads*cfg.HeadDim/cfg.OGroups)%32 != 0 {
		return nil, fmt.Errorf("%w: unsupported bounded draft geometry", errV41DSparkReference)
	}
	for i, tap := range d.TargetLayerIDs {
		if i > 0 && tap <= d.TargetLayerIDs[i-1] {
			return nil, fmt.Errorf("%w: target taps must follow forward order", errV41DSparkReference)
		}
	}
	for stage := 0; stage < 3; stage++ {
		if v41CompressRatioAt(cfg, cfg.NumLayers+stage) != 0 {
			return nil, fmt.Errorf("%w: draft compression is unsupported", errV41DSparkReference)
		}
		theta := v41RopeThetaForLayer(cfg, cfg.NumLayers+stage)
		if theta <= 0 || math.IsNaN(theta) || math.IsInf(theta, 0) {
			return nil, fmt.Errorf("%w: invalid draft rotary base", errV41DSparkReference)
		}
	}
	for _, name := range inv.Names() {
		want, _ := inv.Lookup(name)
		have, ok := w.inv.Lookup(name)
		if !ok || w.shards[name] == nil || !sameShape(want.LogicalShape, have.LogicalShape) {
			return nil, fmt.Errorf("%w: draft/target geometry mismatch %s", errV41DSparkReference, name)
		}
	}
	if len(inv.names) != len(w.inv.names) {
		return nil, fmt.Errorf("%w: draft inventory mismatch", errV41DSparkReference)
	}
	// Freeze the metadata used by this reference; no source slice is retained.
	meta, draft := *cfg.DeepSeekV41, *d
	draft.TargetLayerIDs = append([]int(nil), d.TargetLayerIDs...)
	meta.CompressRatios = append([]int(nil), meta.CompressRatios...)
	meta.DSpark = &draft
	cfg.DeepSeekV41 = &meta
	cfg.RopeThetaPerLayer = append([]float64(nil), cfg.RopeThetaPerLayer...)
	if !target.M.holdModelWeights() {
		return nil, fmt.Errorf("%w: target weights closing", errV41DSparkReference)
	}
	return &v41DSparkReference{w: w, target: target, model: target.M, state: target.v41Forward, cfg: cfg, prefix: sha256.Sum256(nil)}, nil
}

func v41DSparkPrefix(ids []int) [32]byte {
	h := sha256.New()
	var b [8]byte
	for _, id := range ids {
		binary.LittleEndian.PutUint64(b[:], uint64(id))
		_, _ = h.Write(b[:])
	}
	var out [32]byte
	copy(out[:], h.Sum(nil))
	return out
}

func (f *v41DSparkReference) ready(ctx context.Context) error {
	if ctx == nil || f.closed || f.target == nil || f.target.M != f.model || !f.target.modelWeightsHeld ||
		f.target.v41Forward != f.state || f.state.activeStepRound != nil ||
		f.next > len(f.state.history) || v41DSparkPrefix(f.state.history[:f.next]) != f.prefix {
		return fmt.Errorf("%w: closed, speculative, or changed target prefix", errV41DSparkReference)
	}
	return ctx.Err()
}

func (f *v41DSparkReference) Close() error {
	if f == nil {
		return nil
	}
	f.mu.Lock()
	defer f.mu.Unlock()
	if !f.closed {
		f.closed = true
		f.model.releaseWeightSession()
		f.ring = [3][v41WindowSize][]float32{}
		f.target, f.model, f.state, f.w = nil, nil, nil, nil
	}
	return nil
}

// AppendCommitted only consumes contiguous rows already in the bound target's
// committed history. Up to128 rows may be supplied per call, including prefill.
// Every projection is staged; cancellation, read failure or panic changes no ring.
func (f *v41DSparkReference) AppendCommitted(ctx context.Context, rows []v41DSparkContextRow) error {
	if f == nil {
		return errV41DSparkReference
	}
	f.mu.Lock()
	defer f.mu.Unlock()
	if err := f.ready(ctx); err != nil {
		return err
	}
	if len(rows) == 0 || len(rows) > v41WindowSize || len(rows) > len(f.state.history)-f.next {
		return fmt.Errorf("%w: committed span", errV41DSparkReference)
	}
	d, H := f.cfg.DeepSeekV41.DSpark, f.cfg.HiddenSize
	for i, row := range rows {
		if row.Position != f.next+i || row.Token < 0 || row.Token >= f.cfg.VocabSize ||
			row.Token != f.state.history[row.Position] || len(row.Layers) != len(d.TargetLayerIDs) || len(row.Hidden) != len(d.TargetLayerIDs) {
			return fmt.Errorf("%w: committed feature lineage", errV41DSparkReference)
		}
		for j, tap := range d.TargetLayerIDs {
			if row.Layers[j] != tap || len(row.Hidden[j]) != H {
				return fmt.Errorf("%w: committed feature tap", errV41DSparkReference)
			}
			for _, v := range row.Hidden[j] {
				if !finite32(v) {
					return errV41DSparkReference
				}
			}
		}
	}
	staged := f.ring
	for _, row := range rows {
		x := make([]float32, 0, H*len(d.TargetLayerIDs))
		for _, tap := range row.Hidden {
			x = append(x, tap...)
		}
		main, err := f.linear(ctx, "mtp.0.main_proj.weight", x, true)
		if err != nil {
			return err
		}
		main, err = f.norm(ctx, "mtp.0.main_norm.weight", main)
		if err != nil {
			return err
		}
		for stage := range staged {
			kv, err := f.kv(ctx, stage, main, row.Position)
			if err != nil {
				return err
			}
			staged[stage][row.Position%v41WindowSize] = kv
		}
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	f.ring = staged
	f.next += len(rows)
	f.prefix = v41DSparkPrefix(f.state.history[:f.next])
	return nil
}

func (f *v41DSparkReference) rotate(x []float32, stage, position int, inverse bool) error {
	cos, sin := v41RopeTableForLayer(f.cfg, f.cfg.NumLayers+stage, position)
	if inverse {
		for i := range sin {
			sin[i] = -sin[i]
		}
	}
	applyRopeTailInterleaved(x, cos, sin, f.cfg.QKRopeHeadDim)
	for i := range x {
		x[i] = v41RoundBF16(x[i])
		if !finite32(x[i]) {
			return errV41DSparkReference
		}
	}
	return nil
}

func (f *v41DSparkReference) kv(ctx context.Context, stage int, x []float32, position int) ([]float32, error) {
	p := "mtp." + strconv.Itoa(stage) + ".attn."
	kv, err := f.linear(ctx, p+"wkv.weight", x, true)
	if err != nil {
		return nil, err
	}
	kv, err = f.norm(ctx, p+"kv_norm.weight", kv)
	if err != nil {
		return nil, err
	}
	if err = f.rotate(kv, stage, position, false); err != nil {
		return nil, err
	}
	return v41DSparkActFP8(kv)
}

func (f *v41DSparkReference) mix(ctx context.Context, stage int, sub string, streams [][]float32) (v41MHCMix, error) {
	flat := make([]float32, 0, 4*f.cfg.HiddenSize)
	var squares float32
	for _, stream := range streams {
		flat = append(flat, stream...)
		for _, v := range stream {
			squares += float32(v * v)
		}
	}
	if !finite32(squares) || len(flat) == 0 {
		return v41MHCMix{}, fmt.Errorf("%w: invalid mHC normalization statistic", errV41DSparkReference)
	}
	p := "mtp." + strconv.Itoa(stage) + ".hc_" + sub + "_"
	projected, err := f.linear(ctx, p+"fn", flat, false)
	if err != nil {
		return v41MHCMix{}, err
	}
	rsqrt := float32(1 / math.Sqrt(float64(float32(squares/float32(len(flat)))+float32(f.cfg.RMSNormEps))))
	for i := range projected {
		projected[i] *= rsqrt
	}
	scale, err := f.w.row(ctx, p+"scale", 0)
	if err != nil {
		return v41MHCMix{}, err
	}
	base, err := f.w.row(ctx, p+"base", 0)
	if err != nil {
		return v41MHCMix{}, err
	}
	return v41MHCSplit(projected, scale, base, 4, hcItersOrDefault(f.cfg), hcEpsOrDefault(f.cfg))
}

func (f *v41DSparkReference) attention(ctx context.Context, stage int, input [5][]float32) ([5][]float32, error) {
	var out [5][]float32
	c := f.cfg
	p := "mtp." + strconv.Itoa(stage) + ".attn."
	q := make([]float32, 0, 5*c.NumHeads*c.HeadDim)
	kv := make([]float32, 0, (v41WindowSize+5)*c.HeadDim)
	contextRows := min(f.next, v41WindowSize)
	for slot := 0; slot < contextRows; slot++ {
		row := f.ring[stage][slot]
		if len(row) != c.HeadDim {
			return out, fmt.Errorf("%w: missing context row", errV41DSparkReference)
		}
		kv = append(kv, row...)
	}
	for i, x := range input {
		latent, err := f.linear(ctx, p+"wq_a.weight", x, true)
		if err != nil {
			return out, err
		}
		latent, err = f.norm(ctx, p+"q_norm.weight", latent)
		if err != nil {
			return out, err
		}
		query, err := f.linear(ctx, p+"wq_b.weight", latent, true)
		if err != nil {
			return out, err
		}
		for h := 0; h < c.NumHeads; h++ {
			if err = f.rotate(query[h*c.HeadDim:(h+1)*c.HeadDim], stage, f.next+i, false); err != nil {
				return out, err
			}
		}
		q = append(q, query...)
		row, err := f.kv(ctx, stage, x, f.next+i)
		if err != nil {
			return out, err
		}
		kv = append(kv, row...)
	}
	// Every query sees the same complete trained block, including future masks.
	n := contextRows + 5
	indices := make([]int32, 5*n)
	for i := range indices {
		indices[i] = int32(i % n)
	}
	sink, err := f.w.row(ctx, p+"attn_sink", 0)
	if err != nil {
		return out, err
	}
	values, err := V41SparseAttentionSink(q, kv, sink, indices, V41SparseAttentionSinkOptions{
		B: 1, M: 5, Heads: c.NumHeads, HeadDim: c.HeadDim, TopK: n, N: n, Softmax: float32(1 / math.Sqrt(float64(c.HeadDim))),
	})
	if err != nil {
		return out, err
	}
	groupIn := c.NumHeads * c.HeadDim / c.OGroups
	for i := range out {
		o := values[i*c.NumHeads*c.HeadDim : (i+1)*c.NumHeads*c.HeadDim]
		// sparse_attn publishes BF16 before inverse RoPE.
		o, err = v41MHCBF16(stage, o)
		if err != nil {
			return out, err
		}
		for h := 0; h < c.NumHeads; h++ {
			if err = f.rotate(o[h*c.HeadDim:(h+1)*c.HeadDim], stage, f.next+i, true); err != nil {
				return out, err
			}
		}
		o, err = v41DSparkActFP8(o)
		if err != nil {
			return out, err
		}
		joined := make([]float32, c.OGroups*c.OLoraRank)
		for row := range joined {
			weight, err := f.w.row(ctx, p+"wo_a.weight", row)
			if err != nil {
				return out, err
			}
			g := row / c.OLoraRank
			var sum float32
			for k, w := range weight {
				sum += float32(v41RoundBF16(w) * o[g*groupIn+k])
			}
			joined[row] = v41RoundBF16(sum)
		}
		out[i], err = f.linear(ctx, p+"wo_b.weight", joined, true)
		if err != nil {
			return out, err
		}
	}
	return out, nil
}

func (f *v41DSparkReference) route(ctx context.Context, stage int, x []float32) ([]routePick, error) {
	p := "mtp." + strconv.Itoa(stage) + ".ffn.gate."
	logits, err := f.linear(ctx, p+"weight", x, false)
	if err != nil {
		return nil, err
	}
	bias, err := f.w.row(ctx, p+"e_score_correction_bias", 0)
	if err != nil {
		return nil, err
	}
	raw, choice := make([]float32, 128), make([]float32, 128)
	for i, z := range logits {
		softplus := float32(math.Max(float64(z), 0) + math.Log1p(math.Exp(-math.Abs(float64(z)))))
		raw[i] = float32(math.Sqrt(float64(softplus)))
		choice[i] = raw[i] + bias[i]
		if !finite32(choice[i]) {
			return nil, errV41DSparkReference
		}
	}
	indices := v4TopKIndices(choice, 3)
	var sum float32
	for _, id := range indices[:3] {
		sum += raw[id]
	}
	if !finite32(sum) {
		return nil, errV41DSparkReference
	}
	picks := make([]routePick, 3)
	for i, id := range indices[:3] {
		weight := float32(raw[id] / float32(sum+float32(1e-20)))
		picks[i] = routePick{expert: id, weight: float32(weight * float32(f.cfg.RoutedScalingFactor))}
	}
	// Reference MoE accumulates expert-index order, not selection-score order.
	sort.Slice(picks, func(i, j int) bool { return picks[i].expert < picks[j].expert })
	return picks, nil
}

func (f *v41DSparkReference) expert(ctx context.Context, stem string, x []float32, routeWeight float32) ([]float32, error) {
	gate, err := f.linear(ctx, stem+".w1.weight", x, true)
	if err != nil {
		return nil, err
	}
	up, err := f.linear(ctx, stem+".w3.weight", x, true)
	if err != nil {
		return nil, err
	}
	clampSwiGLUProjections(gate, up, float32(f.cfg.SwigluLimit))
	hidden := make([]float32, len(gate))
	for i, g := range gate {
		v := float32(g * sigmoid32(g))
		v = float32(v * up[i])
		hidden[i] = v41RoundBF16(float32(routeWeight * v))
	}
	// The routing weight belongs before w2 and its quantization, not after it.
	return f.linear(ctx, stem+".w2.weight", hidden, true)
}

func (f *v41DSparkReference) moe(ctx context.Context, stage int, x []float32) ([]float32, error) {
	picks, err := f.route(ctx, stage, x)
	if err != nil {
		return nil, err
	}
	p := "mtp." + strconv.Itoa(stage) + ".ffn."
	sum := make([]float32, f.cfg.HiddenSize)
	for _, pick := range picks {
		y, err := f.expert(ctx, p+"experts."+strconv.Itoa(pick.expert), x, pick.weight)
		if err != nil {
			return nil, err
		}
		for k, v := range y {
			sum[k] += v
		}
	}
	shared, err := f.expert(ctx, p+"shared_experts", x, 1)
	if err != nil {
		return nil, err
	}
	for k, v := range shared {
		sum[k] += v
	}
	return v41MHCBF16(stage, sum)
}

// Propose executes a fixed five-row block, then applies the learned Markov head
// sequentially. Greedy-only: no sampled proposal q or verifier is fabricated.
func (f *v41DSparkReference) Propose(ctx context.Context, anchor int) (*v41DSparkProposal, error) {
	if f == nil {
		return nil, errV41DSparkReference
	}
	f.mu.Lock()
	defer f.mu.Unlock()
	if err := f.ready(ctx); err != nil {
		return nil, err
	}
	if f.next == 0 || f.next != len(f.state.history) || f.next > int(^uint(0)>>1)-5 || anchor < 0 || anchor >= f.cfg.VocabSize {
		return nil, fmt.Errorf("%w: incomplete context or invalid anchor", errV41DSparkReference)
	}
	d := f.cfg.DeepSeekV41.DSpark
	ids := []int{anchor, d.NoiseTokenID, d.NoiseTokenID, d.NoiseTokenID, d.NoiseTokenID}
	embedded, err := f.model.v41EmbeddingPanel(ids)
	if err != nil {
		return nil, err
	}
	var streams [5][][]float32
	var carry [5][]float32
	for i := range streams {
		streams[i], err = v41FullInitialStreams(embedded[i*f.cfg.HiddenSize : (i+1)*f.cfg.HiddenSize])
		if err != nil {
			return nil, err
		}
		carry[i] = []float32{1, 0, 0, 0}
	}
	for stage := 0; stage < 3; stage++ {
		p := "mtp." + strconv.Itoa(stage) + "."
		var attnMix [5]v41MHCMix
		var input [5][]float32
		for i := range streams {
			attnMix[i], err = f.mix(ctx, stage, "attn", streams[i])
			if err != nil {
				return nil, err
			}
			x, err := v41MHCPreBF16(stage, streams[i], carry[i])
			if err != nil {
				return nil, err
			}
			input[i], err = f.norm(ctx, p+"attn_norm.weight", x)
			if err != nil {
				return nil, err
			}
		}
		attention, err := f.attention(ctx, stage, input)
		if err != nil {
			return nil, err
		}
		for i := range streams {
			updated, err := v41MHCPostBF16(stage, attention[i], streams[i], attnMix[i])
			if err != nil {
				return nil, err
			}
			ffnMix, err := f.mix(ctx, stage, "ffn", updated)
			if err != nil {
				return nil, err
			}
			x, err := v41MHCPreBF16(stage, updated, attnMix[i].pre)
			if err != nil {
				return nil, err
			}
			x, err = f.norm(ctx, p+"ffn_norm.weight", x)
			if err != nil {
				return nil, err
			}
			ffn, err := f.moe(ctx, stage, x)
			if err != nil {
				return nil, err
			}
			streams[i], err = v41MHCPostBF16(stage, ffn, updated, ffnMix)
			if err != nil {
				return nil, err
			}
			carry[i] = ffnMix.pre
		}
	}
	result := &v41DSparkProposal{Anchor: anchor, AnchorPosition: f.next}
	previous := anchor
	for i := range result.Proposed {
		hidden, err := v41MHCPreBF16(2, streams[i], carry[i])
		if err != nil {
			return nil, err
		}
		normalized, err := f.norm(ctx, "mtp.2.norm.weight", hidden)
		if err != nil {
			return nil, err
		}
		logits, err := f.model.v41ProjectHeadInput(normalized, nil, 1)
		if err != nil {
			return nil, err
		}
		if len(logits) != f.cfg.VocabSize {
			return nil, errV41DSparkReference
		}
		markov, err := f.w.row(ctx, "mtp.2.markov_head.embed.weight", previous)
		if err != nil {
			return nil, err
		}
		bias, err := f.linear(ctx, "mtp.2.markov_head.head.weight", markov, false)
		if err != nil {
			return nil, err
		}
		best := 0
		for j := range logits {
			logits[j] += bias[j]
			if !finite32(logits[j]) {
				return nil, errV41DSparkReference
			}
			if logits[j] > logits[best] {
				best = j
			}
		}
		confInput := append(append([]float32(nil), hidden...), markov...)
		confidence, err := f.linear(ctx, "mtp.2.confidence_head.proj.weight", confInput, false)
		if err != nil {
			return nil, err
		}
		result.Proposed[i], result.Logits[i], result.hidden[i] = best, logits, hidden
		result.ConfidenceLogits[i] = confidence[0]
		result.ConditionalConfidence[i] = sigmoid32(confidence[0])
		previous = best
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	return result, nil
}
