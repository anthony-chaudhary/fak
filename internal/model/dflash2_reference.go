package model

// Native reference adaptation of Z Lab's DFlash2 math at commit
// 07ebd93db9f472af339b644bb70221ad8428328a, dflash/model_mlx.py.
// Copyright (c) 2026 Z Lab. MIT license: see dflash2_reference_LICENSE.txt.
// This uses FAK-owned BF16 reads and native Go math, never a Python runtime.

import (
	"context"
	"errors"
	"fmt"
	"math"
)

type dflash2Geometry struct {
	hidden, intermediate, heads, kvHeads, headDim, vocab int
	group, rank, topK, window, mask                      int
	eps, theta                                           float64
	taps                                                 []int
}

func qwen38DFlash2Geometry() dflash2Geometry {
	return dflash2Geometry{5120, 17408, 32, 8, 128, 248320, 16, 256, 16, 2048, 248070, 1e-6, 1e7, []int{5, 19, 33, 47, 61}}
}

type dflash2ContextKV struct {
	keys, values []float32 // row-major [context tokens, KV heads, head dimension]
}

// dflash2Reference executes all five pretrained layers. Persistent K/V contains
// only committed target features; block K/V is ephemeral. AppendCommitted is
// failure-atomic across all five layers, and Propose never changes context K/V.
// One caller owns this reference and its target Session through Close; concurrent
// target Step/Close/restore and reference calls are unsupported.
//
// Draft math is the bounded host-f32 correctness path. An admitted HAL target
// uses explicit host taps and its packed shared head. It streams BF16 weight rows,
// rather than retaining a second expanded checkpoint. Its context K/V is f32
// (up to80MiB for stock geometry), not the40MiB BF16 device representation.
// Draft arithmetic need not reproduce another runtime's BF16 rounding; exact
// output tokens still require the target's greedy verification and transaction.
type dflash2Reference struct {
	w            *dflash2Weights
	target       *Session
	g            dflash2Geometry
	cache        [5]dflash2ContextKV
	nextPosition int
	prefixTokens []uint32 // exact owned lineage, including context dropped from K/V
	closed       bool
	closeErr     error
}

// This internal experimental constructor verifies the stock sidecar and target
// geometry. It is not a serving-admission decision: selected target/tokenizer/
// template digests and a qualified native device implementation remain required.
func newQwen38DFlash2Reference(ctx context.Context, target *Session, path string, firstContextPosition int) (*dflash2Reference, error) {
	if target == nil || target.M == nil || target.M.Cfg.NumLayers != 64 || target.M.Cfg.HiddenSize != 5120 || target.M.Cfg.VocabSize != 248320 || !target.M.Cfg.IsQwen35Hybrid() || target.M.Cfg.IsDeepSeekV41() {
		return nil, errors.New("dflash2: stock Qwen3.8 target geometry required")
	}
	w, err := openQwen38DFlash2Weights(ctx, path)
	if err != nil {
		return nil, err
	}
	f, err := newDFlash2Reference(w, target, qwen38DFlash2Geometry(), firstContextPosition)
	if err != nil {
		return nil, errors.Join(err, w.Close())
	}
	return f, nil
}

// The private geometry seam permits real five-layer arithmetic with tiny BF16
// fixtures. Only the stock constructor above is a production artifact opener.
// On success the reference owns w and holds the target Model's weight lifetime.
func newDFlash2Reference(w *dflash2Weights, target *Session, g dflash2Geometry, first int) (*dflash2Reference, error) {
	if w == nil || first < 0 {
		return nil, errors.New("dflash2: live target and weight owner required")
	}
	if err := validateQwen35TargetFeatureSession(target); err != nil {
		return nil, err
	}
	position, lineage, err := qwen35TargetFeaturePrefix(target)
	if err != nil || first > position {
		return nil, errors.New("dflash2: exact target prefix required")
	}
	if g.hidden < 1 || g.hidden > 5120 || g.intermediate < 1 || g.intermediate > 17408 || g.heads < 1 || g.heads > 32 || g.kvHeads < 1 || g.kvHeads > 8 || g.heads%g.kvHeads != 0 || g.headDim < 2 || g.headDim > 128 || g.headDim%2 != 0 || g.vocab < 1 || g.vocab > 248320 || g.group < 1 || g.hidden%g.group != 0 || g.rank < 1 || g.rank > 256 || g.topK < 1 || g.topK > 16 || g.topK > g.vocab || g.window < 9 || g.window > 2048 || g.mask < 0 || g.mask >= g.vocab || len(g.taps) < 1 || len(g.taps) > 5 || g.eps <= 0 || g.theta <= 0 || math.IsNaN(g.eps) || math.IsInf(g.eps, 0) || math.IsNaN(g.theta) || math.IsInf(g.theta, 0) {
		return nil, errors.New("dflash2: invalid bounded draft geometry")
	}
	if first > max(0, position-(g.window-1)) {
		return nil, errors.New("dflash2: context seed must include the complete available tail")
	}
	if target.M.Cfg.HiddenSize != g.hidden || target.M.Cfg.VocabSize != g.vocab {
		return nil, errors.New("dflash2: shared embedding/head geometry mismatch")
	}
	for i, tap := range g.taps {
		if tap < 0 || tap >= target.M.Cfg.NumLayers || (i > 0 && tap <= g.taps[i-1]) {
			return nil, errors.New("dflash2: invalid target layer taps")
		}
	}
	if !target.M.holdModelWeights() {
		return nil, errors.New("dflash2: target weights are closing")
	}
	g.taps = append([]int(nil), g.taps...)
	return &dflash2Reference{w: w, target: target, g: g, nextPosition: first, prefixTokens: append([]uint32(nil), lineage.ids[:first]...)}, nil
}

func (f *dflash2Reference) ready(ctx context.Context) error {
	if ctx == nil || f == nil || f.closed || f.target == nil || !f.target.modelWeightsHeld {
		return errors.New("dflash2: live exclusive reference and context required")
	}
	if f.target.qwen35TargetFeatureTxn != nil {
		return errors.New("dflash2: target transaction must commit before draft context access")
	}
	position, lineage, err := qwen35TargetFeaturePrefix(f.target)
	if err != nil || f.nextPosition != len(f.prefixTokens) || position < f.nextPosition {
		return errors.New("dflash2: target prefix changed; reseed draft context")
	}
	for i, token := range f.prefixTokens {
		if lineage.ids[i] != token {
			return errors.New("dflash2: target prefix changed; reseed draft context")
		}
	}
	return ctx.Err()
}

// proposalReady requires every retained context position, not just matching
// token lineage. A new or partially seeded reference cannot draft from an empty
// or truncated tail, even when its absolute next position looks plausible.
func (f *dflash2Reference) proposalReady(ctx context.Context) error {
	if err := f.ready(ctx); err != nil {
		return err
	}
	position, _, err := qwen35TargetFeaturePrefix(f.target)
	if err != nil || f.nextPosition != position {
		return errors.New("dflash2: context is not at the target boundary")
	}
	want := min(f.nextPosition, f.g.window-1) * f.g.kvHeads * f.g.headDim
	for _, cache := range f.cache {
		if len(cache.keys) != want || len(cache.values) != want {
			return errors.New("dflash2: complete committed context tail required")
		}
	}
	return nil
}

func (f *dflash2Reference) Close() error {
	if f != nil && !f.closed {
		f.closed = true
		f.closeErr = f.w.Close()
		f.target.M.releaseWeightSession()
		f.cache = [5]dflash2ContextKV{}
		f.prefixTokens = nil
		f.target, f.w = nil, nil
	}
	if f == nil {
		return nil
	}
	return f.closeErr
}

// linear streams each BF16 matrix row once for every input row in the block.
// It never expands a whole matrix; the largest stock row is fc's25,600 values.
func (f *dflash2Reference) linear(ctx context.Context, name string, x []float32, in, out int) ([]float32, error) {
	if in < 1 || out < 1 || len(x)%in != 0 {
		return nil, fmt.Errorf("dflash2: invalid linear input %s", name)
	}
	rows := len(x) / in
	y, weight := make([]float32, rows*out), make([]float32, in)
	for j := 0; j < out; j++ {
		if err := f.w.ReadBF16(ctx, name, int64(j)*int64(in), weight); err != nil {
			return nil, err
		}
		for row := 0; row < rows; row++ {
			var sum float32
			for k, v := range weight {
				sum += v * x[row*in+k]
			}
			y[row*out+j] = sum
		}
	}
	return y, nil
}

func (f *dflash2Reference) norm(ctx context.Context, name string, x []float32, width int) ([]float32, error) {
	weight := make([]float32, width)
	if err := f.w.ReadBF16(ctx, name, 0, weight); err != nil {
		return nil, err
	}
	y := make([]float32, len(x))
	for start := 0; start < len(x); start += width {
		rmsnormInto(y[start:start+width], x[start:start+width], weight, float32(f.g.eps))
	}
	return y, nil
}

// The dynamic kernel is [row, pre/post, tap, group], and base is
// [pre/post, tap, channel]. Causal convolution resets at each proposal block.
func dflash2Convolve(x, dynamic, base []float32, hidden, group, side int) []float32 {
	y := make([]float32, len(x))
	groups := hidden / group
	for row := 0; row < len(x)/hidden; row++ {
		for c := 0; c < hidden; c++ {
			for tap := 0; tap < 2 && tap <= row; tap++ {
				value := x[(row-tap)*hidden+c]
				y[row*hidden+c] += base[(side*2+tap)*hidden+c] * value
				y[row*hidden+c] += dynamic[((row*2+side)*2+tap)*groups+c/group] * value
			}
		}
	}
	return y
}

func (f *dflash2Reference) prepareConv(ctx context.Context, name string, x []float32) (input, dynamic, base []float32, err error) {
	dynamic, err = f.linear(ctx, name+".kernel_projection.weight", x, f.g.hidden, 4*(f.g.hidden/f.g.group))
	if err != nil {
		return nil, nil, nil, err
	}
	base = make([]float32, 4*f.g.hidden)
	if err = f.w.ReadBF16(ctx, name+".base_kernel", 0, base); err != nil {
		return nil, nil, nil, err
	}
	return dflash2Convolve(x, dynamic, base, f.g.hidden, f.g.group, 0), dynamic, base, nil
}

func dflash2RoPE(x []float32, heads, dim, first int, theta float64) {
	stride, half := heads*dim, dim/2
	for row := 0; row < len(x)/stride; row++ {
		for h := 0; h < heads; h++ {
			for j := 0; j < half; j++ {
				angle := float64(first+row) / math.Pow(theta, float64(2*j)/float64(dim))
				c, s := float32(math.Cos(angle)), float32(math.Sin(angle))
				a, b := row*stride+h*dim+j, row*stride+h*dim+j+half
				u, v := x[a], x[b]
				x[a], x[b] = u*c-v*s, u*s+v*c
			}
		}
	}
}

func (f *dflash2Reference) projectKV(ctx context.Context, name string, x []float32, first int) (dflash2ContextKV, error) {
	g := f.g
	k, err := f.linear(ctx, name+".k_proj.weight", x, g.hidden, g.kvHeads*g.headDim)
	if err != nil {
		return dflash2ContextKV{}, err
	}
	k, err = f.norm(ctx, name+".k_norm.weight", k, g.headDim)
	if err != nil {
		return dflash2ContextKV{}, err
	}
	v, err := f.linear(ctx, name+".v_proj.weight", x, g.hidden, g.kvHeads*g.headDim)
	if err != nil {
		return dflash2ContextKV{}, err
	}
	dflash2RoPE(k, g.kvHeads, g.headDim, first, g.theta)
	return dflash2ContextKV{k, v}, nil
}

// AppendCommitted consumes only contiguous rows already committed in the live
// target. A later restore/fork requires a new reference seeded from that prefix.
func (f *dflash2Reference) AppendCommitted(ctx context.Context, rows []qwen35TargetFeatureRow) error {
	if err := f.ready(ctx); err != nil {
		return err
	}
	g := f.g
	position, lineage, prefixErr := qwen35TargetFeaturePrefix(f.target)
	if prefixErr != nil || len(rows) < 1 || len(rows) > g.window-1 || f.nextPosition > position-len(rows) {
		return errors.New("dflash2: invalid committed context span")
	}
	width := len(g.taps) * g.hidden
	x := make([]float32, len(rows)*width)
	for i, row := range rows {
		pos := f.nextPosition + i
		if row.Position != pos || row.Token < 0 || row.Token >= g.vocab || len(row.Layers) != len(g.taps) || len(row.Hidden) != len(g.taps) || pos >= len(lineage.ids) || uint32(row.Token) != lineage.ids[pos] {
			return errors.New("dflash2: committed target feature lineage mismatch")
		}
		for tap, layer := range g.taps {
			if row.Layers[tap] != layer || len(row.Hidden[tap]) != g.hidden {
				return errors.New("dflash2: committed target feature tap mismatch")
			}
			copy(x[i*width+tap*g.hidden:], row.Hidden[tap])
		}
	}
	x, err := f.linear(ctx, "fc.weight", x, width, g.hidden)
	if err != nil {
		return err
	}
	x, err = f.norm(ctx, "hidden_norm.weight", x, g.hidden)
	if err != nil {
		return err
	}
	var staged [5]dflash2ContextKV
	stride := g.kvHeads * g.headDim
	for layer := range staged {
		fresh, err := f.projectKV(ctx, fmt.Sprintf("layers.%d.self_attn", layer), x, f.nextPosition)
		if err != nil {
			return err
		}
		keep := min(len(f.cache[layer].keys)/stride, g.window-1-len(rows))
		old := f.cache[layer]
		staged[layer].keys = append(append([]float32(nil), old.keys[len(old.keys)-keep*stride:]...), fresh.keys...)
		staged[layer].values = append(append([]float32(nil), old.values[len(old.values)-keep*stride:]...), fresh.values...)
	}
	prefix := append(append([]uint32(nil), f.prefixTokens...), lineage.ids[f.nextPosition:f.nextPosition+len(rows)]...)
	if err := ctx.Err(); err != nil {
		return err
	}
	f.cache = staged
	f.nextPosition += len(rows)
	f.prefixTokens = prefix
	return nil
}

func (f *dflash2Reference) attention(ctx context.Context, layer int, x []float32) ([]float32, error) {
	g := f.g
	name := fmt.Sprintf("layers.%d.self_attn", layer)
	q, err := f.linear(ctx, name+".q_proj.weight", x, g.hidden, g.heads*g.headDim)
	if err != nil {
		return nil, err
	}
	q, err = f.norm(ctx, name+".q_norm.weight", q, g.headDim)
	if err != nil {
		return nil, err
	}
	dflash2RoPE(q, g.heads, g.headDim, f.nextPosition, g.theta)
	block, err := f.projectKV(ctx, name, x, f.nextPosition)
	if err != nil {
		return nil, err
	}
	rows, kvStride := len(x)/g.hidden, g.kvHeads*g.headDim
	prior := f.cache[layer]
	contextRows := len(prior.keys) / kvStride
	output, scores := make([]float32, len(q)), make([]float32, contextRows+rows)
	scale := float32(1 / math.Sqrt(float64(g.headDim)))
	for row := 0; row < rows; row++ {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		for h := 0; h < g.heads; h++ {
			kvHead, qStart := h/(g.heads/g.kvHeads), (row*g.heads+h)*g.headDim
			maximum := float32(math.Inf(-1))
			for k := range scores {
				if k < contextRows && contextRows+row-k >= g.window {
					scores[k] = float32(math.Inf(-1))
					continue
				}
				keys, keyRow := prior.keys, k
				if k >= contextRows {
					keys, keyRow = block.keys, k-contextRows
				}
				var sum float32
				for d := 0; d < g.headDim; d++ {
					sum += q[qStart+d] * keys[keyRow*kvStride+kvHead*g.headDim+d]
				}
				scores[k] = sum * scale
				maximum = max(maximum, scores[k])
			}
			var denominator float32
			for k := range scores {
				scores[k] = float32(math.Exp(float64(scores[k] - maximum)))
				denominator += scores[k]
			}
			for k, score := range scores {
				values, keyRow := prior.values, k
				if k >= contextRows {
					values, keyRow = block.values, k-contextRows
				}
				for d := 0; d < g.headDim; d++ {
					output[qStart+d] += (score / denominator) * values[keyRow*kvStride+kvHead*g.headDim+d]
				}
			}
		}
	}
	return f.linear(ctx, name+".o_proj.weight", output, g.heads*g.headDim, g.hidden)
}

func (f *dflash2Reference) hidden(ctx context.Context, anchor, proposals int) ([]float32, error) {
	g := f.g
	x := make([]float32, (proposals+1)*g.hidden)
	for row := 0; row <= proposals; row++ {
		token := g.mask
		if row == 0 {
			token = anchor
		}
		embedding, err := f.target.TokenEmbedding(token)
		if err != nil {
			return nil, err
		}
		scaleEmbedInPlace(embedding, f.target.M.Cfg)
		copy(x[row*g.hidden:], embedding)
	}
	for layer := 0; layer < 5; layer++ {
		name := fmt.Sprintf("layers.%d", layer)
		normalized, err := f.norm(ctx, name+".input_layernorm.weight", x, g.hidden)
		if err != nil {
			return nil, err
		}
		input, dynamic, base, err := f.prepareConv(ctx, name+".attention_conv", normalized)
		if err != nil {
			return nil, err
		}
		attended, err := f.attention(ctx, layer, input)
		if err != nil {
			return nil, err
		}
		attended = dflash2Convolve(attended, dynamic, base, g.hidden, g.group, 1)
		for i := range x {
			x[i] += attended[i]
		}
		normalized, err = f.norm(ctx, name+".post_attention_layernorm.weight", x, g.hidden)
		if err != nil {
			return nil, err
		}
		input, dynamic, base, err = f.prepareConv(ctx, name+".mlp_conv", normalized)
		if err != nil {
			return nil, err
		}
		gate, err := f.linear(ctx, name+".mlp.gate_proj.weight", input, g.hidden, g.intermediate)
		if err != nil {
			return nil, err
		}
		up, err := f.linear(ctx, name+".mlp.up_proj.weight", input, g.hidden, g.intermediate)
		if err != nil {
			return nil, err
		}
		for i := range gate {
			gate[i] = gate[i] / (1 + float32(math.Exp(float64(-gate[i])))) * up[i]
		}
		down, err := f.linear(ctx, name+".mlp.down_proj.weight", gate, g.intermediate, g.hidden)
		if err != nil {
			return nil, err
		}
		down = dflash2Convolve(down, dynamic, base, g.hidden, g.group, 1)
		for i := range x {
			x[i] += down[i]
		}
	}
	return f.norm(ctx, "norm.weight", x[g.hidden:], g.hidden)
}

// Propose returns one to seven tokens after the known anchor. Top-k selection is
// greedy and deterministic (lower token ID wins unary ties). It is only a draft;
// these IDs must never be emitted before exact native target verification.
func (f *dflash2Reference) Propose(ctx context.Context, anchor, proposals int) ([]int, error) {
	if err := f.proposalReady(ctx); err != nil {
		return nil, err
	}
	g := f.g
	if anchor < 0 || anchor >= g.vocab || proposals < 1 || proposals > 7 {
		return nil, errors.New("dflash2: anchor/proposal/context boundary mismatch")
	}
	hidden, err := f.hidden(ctx, anchor, proposals)
	if err != nil {
		return nil, err
	}
	return f.selectCandidates(ctx, anchor, hidden)
}

// The selector is separate from layer execution so its trained Markov edges can
// be checked against an independent oracle without target rejection hiding bugs.
func (f *dflash2Reference) selectCandidates(ctx context.Context, anchor int, hidden []float32) ([]int, error) {
	g := f.g
	proposals := len(hidden) / g.hidden
	if anchor < 0 || anchor >= g.vocab || proposals < 1 || proposals > 7 || len(hidden)%g.hidden != 0 {
		return nil, errors.New("dflash2: invalid selector rows")
	}
	projected, err := f.linear(ctx, "candidate_selector.hidden_projection.weight", hidden, g.hidden, g.rank)
	if err != nil {
		return nil, err
	}
	path := make([]int, proposals)
	predecessor, predecessorCode, successorCode := anchor, make([]float32, g.rank), make([]float32, g.rank)
	for row := range path {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		logits, err := f.target.dflash2TargetHead(ctx, hidden[row*g.hidden:(row+1)*g.hidden])
		if err != nil {
			return nil, err
		}
		if len(logits) != g.vocab {
			return nil, errors.New("dflash2: target head did not return complete vocabulary")
		}
		candidates := make([]int, 0, g.topK)
		for token, value := range logits {
			if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
				return nil, errors.New("dflash2: nonfinite draft logits")
			}
			at := len(candidates)
			for at > 0 && value > logits[candidates[at-1]] {
				at--
			}
			if at < g.topK {
				candidates = append(candidates, token)
				copy(candidates[at+1:], candidates[at:len(candidates)-1])
				candidates[at] = token
				if len(candidates) > g.topK {
					candidates = candidates[:g.topK]
				}
			}
		}
		if err := f.w.ReadBF16(ctx, "candidate_selector.predecessor_codebook", int64(predecessor*g.rank), predecessorCode); err != nil {
			return nil, err
		}
		best, bestScore := -1, float32(math.Inf(-1))
		for _, token := range candidates {
			if err := f.w.ReadBF16(ctx, "candidate_selector.successor_codebook", int64(token*g.rank), successorCode); err != nil {
				return nil, err
			}
			score := logits[token]
			for k := 0; k < g.rank; k++ {
				score += predecessorCode[k] * projected[row*g.rank+k] * successorCode[k]
			}
			if math.IsNaN(float64(score)) || math.IsInf(float64(score), 0) {
				return nil, errors.New("dflash2: nonfinite candidate score")
			}
			if best < 0 || score > bestScore {
				best, bestScore = token, score
			}
		}
		path[row], predecessor = best, best
	}
	return path, nil
}

// GreedyRound is the native reference vertical slice: trained draft math,
// exact target verification, accepted-prefix commit, and draft-context commit.
// Nothing is published before both owners commit. maxTokens bounds returned
// token IDs (1..8); EOS ends the accepted prefix. Streaming stop-string policy
// remains the caller's responsibility. beforeLogits must belong to the exact
// current target boundary; width is checked, provenance is a caller precondition.
// There is no sampling path, serving flag, external runtime, or draft-only output.
func (f *dflash2Reference) GreedyRound(ctx context.Context, beforeLogits []float32, maxTokens int) (tokens []int, nextLogits []float32, err error) {
	if err = f.proposalReady(ctx); err != nil {
		return nil, nil, err
	}
	if len(beforeLogits) != f.g.vocab || maxTokens < 1 || maxTokens > 8 {
		return nil, nil, errors.New("dflash2: invalid greedy boundary or token budget")
	}
	anchor := argmaxF32(beforeLogits)
	proposal := []int{anchor}
	if maxTokens > 1 && !f.target.M.Cfg.IsEOS(anchor) {
		draft, e := f.Propose(ctx, anchor, maxTokens-1)
		if e != nil {
			return nil, nil, e
		}
		proposal = append(proposal, draft...)
	}
	snapshot, err := f.target.PrefixSnapshot()
	if err != nil {
		return nil, nil, err
	}
	oldCache, oldPrefix, oldPosition := f.cache, f.prefixTokens, f.nextPosition
	oldStep, oldWarm := f.target.halStep, f.target.halLogitsWarm
	committed := false
	defer func() {
		if p := recover(); p != nil {
			err = fmt.Errorf("dflash2: greedy round panic: %v", p)
		}
		if !committed || err != nil {
			err = errors.Join(err, snapshot.Restore(f.target), ctx.Err())
			f.target.halStep, f.target.halLogitsWarm = oldStep, oldWarm
			f.cache, f.prefixTokens, f.nextPosition = oldCache, oldPrefix, oldPosition
			tokens, nextLogits = nil, nil
		}
		snapshot.Close()
	}()
	tx, err := beginQwen35TargetFeatureTransaction(f.target, beforeLogits, f.g.taps)
	if err != nil {
		return nil, nil, err
	}
	defer func() { err = errors.Join(err, tx.Abort()) }() // runs before outer restore
	verified, err := tx.Verify(ctx, proposal)
	if err != nil {
		return nil, nil, err
	}
	accepted, current := 0, beforeLogits
	for i, token := range proposal {
		if token != argmaxF32(current) {
			break
		}
		accepted++
		current = verified[i]
		if f.target.M.Cfg.IsEOS(token) {
			break
		}
	}
	nextLogits, rows, err := tx.Commit(ctx, accepted)
	if err != nil {
		return nil, nil, err
	}
	if err = f.AppendCommitted(ctx, rows); err != nil {
		return nil, nil, err
	}
	if err = ctx.Err(); err != nil {
		return nil, nil, err
	}
	tokens = append([]int(nil), proposal[:accepted]...)
	committed = true
	return tokens, nextLogits, nil
}
