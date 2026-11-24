package model

// Host reference of the Inkling-2-Mini DSpark draft (the checkpoint's
// mtp.safetensors), ported from vLLM e72fc9ae5 (Apache-2.0):
// vllm/models/inkling2/nvidia/dspark.py (draft layers, Markov head),
// vllm/models/inkling2/speculator.py (f_rnn and the sequential sampling loop)
// and vllm/v1/worker/gpu/spec_decode/dflash/speculator.py (block layout).
// Activations stay f32; the reference rounds intermediates to bf16, so a
// near-tied draft token can differ, which only moves acceptance.

import (
	"fmt"
	"math"
)

// Inkling2DSparkHParams are the draft's mtp_config scalars plus the target
// vocabulary it shares. Block counts the anchor: Block-1 tokens are drafted.
type Inkling2DSparkHParams struct {
	Hidden, Layers, Heads, KVHeads, HeadDim, Intermediate int
	Block, MarkovRank, SlidingWindow, VocabUnpadded       int
	Taps                                                  int
	LocalLayer                                            []bool
	NormEps                                               float32
	RopeTheta                                             float64
}

type Inkling2DSparkLayerWeights struct {
	InputNorm, PostAttnNorm, DFlareNorm []float32
	QNorm, KNorm, KNormCtx              []float32
	Wq, Wk, Wv, Wo, WkCtx, WvCtx        Inkling2LoopMatrix
	Gate, Up, Down                      Inkling2LoopMatrix
}

// Inkling2DSparkWeights are the draft tensors plus the target's normalized
// token embedding and LM head, which the draft shares.
type Inkling2DSparkWeights struct {
	Layers          []Inkling2DSparkLayerWeights
	Norm, MaskEmbed []float32
	// FuseWeight is [Layers+1][Taps]: one softmaxed mix of the target features
	// per layer, the last row for the Markov head.
	FuseWeight    [][]float32
	MarkovW1      Inkling2LoopEmbedding // [vocab][MarkovRank]
	RnnProj       Inkling2LoopMatrix    // [3*rank][2*rank+2*Hidden]
	RnnBias       []float32
	HiddenOut     Inkling2LoopMatrix // [Hidden][rank]
	HiddenOutBias []float32
	MarkovNorm    []float32
	TokEmbd       Inkling2LoopEmbedding
	TokNorm       []float32
	Output        Inkling2LoopMatrix
}

// Inkling2DSparkDraft holds per-layer context K/V built from the target
// features of committed tokens, and the recurrent head's context of the last
// committed token.
type Inkling2DSparkDraft struct {
	HP           Inkling2DSparkHParams
	W            *Inkling2DSparkWeights
	alpha        [][]float32
	keys, values [][][]float32 // [layer][context][KVHeads*HeadDim]
	positions    []int
	fRNN         [][]float32 // per committed token, so Rewind restores it
	// Shortlist, when positive, restricts each drafted token to the Shortlist best
	// base logits lm_head(h) before the Markov bias is added; negative drops the
	// Markov head and drafts the base argmax. A device draft computes the base logits
	// of every row in one batched head pass and the bias only on the shortlist.
	Shortlist int
}

func NewInkling2DSparkDraft(hp Inkling2DSparkHParams, w *Inkling2DSparkWeights) (*Inkling2DSparkDraft, error) {
	if hp.Block < 2 || hp.Taps < 1 || hp.Heads%hp.KVHeads != 0 || hp.HeadDim%2 != 0 {
		return nil, fmt.Errorf("inkling2 dspark: invalid geometry %+v", hp)
	}
	if len(w.Layers) != hp.Layers || len(hp.LocalLayer) != hp.Layers || len(w.FuseWeight) != hp.Layers+1 {
		return nil, fmt.Errorf("inkling2 dspark: %d layers, %d local flags, %d fuse rows for %d layers",
			len(w.Layers), len(hp.LocalLayer), len(w.FuseWeight), hp.Layers)
	}
	if w.RnnProj.Rows() != 3*hp.MarkovRank || w.RnnProj.Cols() != 2*hp.MarkovRank+2*hp.Hidden {
		return nil, fmt.Errorf("inkling2 dspark: rnn_proj %dx%d for rank %d", w.RnnProj.Rows(), w.RnnProj.Cols(), hp.MarkovRank)
	}
	d := &Inkling2DSparkDraft{HP: hp, W: w}
	for _, row := range w.FuseWeight {
		if len(row) != hp.Taps {
			return nil, fmt.Errorf("inkling2 dspark: fuse row of %d taps, want %d", len(row), hp.Taps)
		}
		d.alpha = append(d.alpha, inkling2Softmax(row))
	}
	d.keys = make([][][]float32, hp.Layers)
	d.values = make([][][]float32, hp.Layers)
	return d, nil
}

// Context is the number of committed tokens.
func (d *Inkling2DSparkDraft) Context() int { return len(d.positions) }

// Commit appends one committed token at pos from its target features, in the
// order of the target's FeatureVisits.
func (d *Inkling2DSparkDraft) Commit(pos int, features [][]float32) error {
	hp := d.HP
	if len(features) != hp.Taps {
		return fmt.Errorf("inkling2 dspark: %d target features, want %d", len(features), hp.Taps)
	}
	for l := range d.W.Layers {
		lw := &d.W.Layers[l]
		ctx := d.mix(features, l, lw.DFlareNorm)
		k := inkling2MatVec(lw.WkCtx, ctx)
		d.headNorm(k, hp.KVHeads, lw.KNormCtx)
		dflash2RoPE(k, hp.KVHeads, hp.HeadDim, pos, hp.RopeTheta)
		d.keys[l] = append(d.keys[l], k)
		d.values[l] = append(d.values[l], inkling2MatVec(lw.WvCtx, ctx))
	}
	d.positions = append(d.positions, pos)
	d.fRNN = append(d.fRNN, d.mix(features, hp.Layers, d.W.MarkovNorm))
	return nil
}

// Rewind drops committed context at positions >= pos (rejected drafts).
func (d *Inkling2DSparkDraft) Rewind(pos int) {
	n := len(d.positions)
	for n > 0 && d.positions[n-1] >= pos {
		n--
	}
	for l := range d.keys {
		d.keys[l], d.values[l] = d.keys[l][:n], d.values[l][:n]
	}
	d.positions, d.fRNN = d.positions[:n], d.fRNN[:n]
}

// Draft proposes Block-1 greedy tokens following anchor, the token at pos,
// which comes right after the committed context.
func (d *Inkling2DSparkDraft) Draft(anchor, pos int) ([]int, error) {
	tokens, _, err := d.draft(anchor, pos)
	return tokens, err
}

func (d *Inkling2DSparkDraft) draft(anchor, pos int) ([]int, [][]float32, error) {
	hp := d.HP
	if len(d.positions) == 0 || d.positions[len(d.positions)-1] >= pos {
		return nil, nil, fmt.Errorf("inkling2 dspark: anchor position %d needs committed context before it", pos)
	}
	hidden := d.backbone(anchor, pos)
	state := make([]float32, hp.MarkovRank)
	emb := make([]float32, hp.MarkovRank)
	fRNN := d.fRNN[len(d.fRNN)-1]
	prev := anchor
	out := make([]int, 0, hp.Block-1)
	logits := make([]float32, d.W.Output.Rows())
	for i := 1; i < hp.Block; i++ {
		h := hidden[i]
		d.W.MarkovW1.Row(prev, emb)
		in := make([]float32, 0, d.W.RnnProj.Cols())
		in = append(append(append(append(in, state...), emb...), h...), fRNN...)
		raw := inkling2MatVec(d.W.RnnProj, in)
		r := hp.MarkovRank
		latent := make([]float32, r)
		for c := 0; c < r; c++ {
			raw[c] += d.W.RnnBias[c]
			raw[r+c] += d.W.RnnBias[r+c]
			raw[2*r+c] += d.W.RnnBias[2*r+c]
			gate := sigmoid32(raw[c])
			cand := float32(math.Tanh(float64(raw[r+c])))
			state[c] = gate*state[c] + (1-gate)*cand
			latent[c] = float32(math.Tanh(float64(raw[2*r+c])))
		}
		// Base logits plus the Markov bias share the LM head:
		// lm_head(h) + lm_head(hidden_out(latent)) = lm_head(h + hidden_out(latent)).
		// The final softcap is monotonic, so greedy skips it.
		y := inkling2MatVec(d.W.HiddenOut, latent)
		for j := range y {
			y[j] += h[j] + d.W.HiddenOutBias[j]
		}
		var best int
		if d.Shortlist != 0 {
			best = d.shortlistArgmax(h, y, logits)
		} else {
			d.W.Output.MatVec(logits, y)
			best = inkling2Argmax(logits[:hp.VocabUnpadded])
		}
		out = append(out, best)
		prev = best
	}
	return out, hidden, nil
}

// backbone runs the dense layers over the block (anchor then masks) at
// positions pos..pos+Block-1 and returns the final-norm hidden of each row.
func (d *Inkling2DSparkDraft) backbone(anchor, pos int) [][]float32 {
	hp := d.HP
	res := make([][]float32, hp.Block)
	emb := make([]float32, hp.Hidden)
	d.W.TokEmbd.Row(anchor, emb)
	res[0] = inkling2RMS(emb, d.W.TokNorm, hp.NormEps)
	for i := 1; i < hp.Block; i++ {
		res[i] = append([]float32(nil), d.W.MaskEmbed...)
	}
	var mlp [][]float32
	for l := range d.W.Layers {
		lw := &d.W.Layers[l]
		x := make([][]float32, hp.Block)
		for i := range res {
			if mlp != nil {
				res[i] = inkling2Add(res[i], mlp[i])
			}
			x[i] = inkling2RMS(res[i], lw.InputNorm, hp.NormEps)
		}
		attn := d.attention(l, x, pos)
		mlp = make([][]float32, hp.Block)
		for i := range res {
			res[i] = inkling2Add(res[i], attn[i])
			h := inkling2RMS(res[i], lw.PostAttnNorm, hp.NormEps)
			g, u := inkling2MatVec(lw.Gate, h), inkling2MatVec(lw.Up, h)
			for j := range g {
				g[j] = g[j] * sigmoid32(g[j]) * u[j]
			}
			mlp[i] = inkling2MatVec(lw.Down, g)
		}
	}
	out := make([][]float32, hp.Block)
	for i := range res {
		out[i] = inkling2RMS(inkling2Add(res[i], mlp[i]), d.W.Norm, hp.NormEps)
	}
	return out
}

// attention is bidirectional over the block and attends to the committed
// context; local layers see only the SlidingWindow positions before pos.
func (d *Inkling2DSparkDraft) attention(l int, x [][]float32, pos int) [][]float32 {
	hp := d.HP
	lw := &d.W.Layers[l]
	dim := hp.HeadDim
	keys := make([][]float32, 0, len(d.keys[l])+hp.Block)
	values := make([][]float32, 0, cap(keys))
	for j, p := range d.positions {
		if hp.LocalLayer[l] && p < pos-hp.SlidingWindow {
			continue
		}
		keys = append(keys, d.keys[l][j])
		values = append(values, d.values[l][j])
	}
	qs := make([][]float32, len(x))
	for i, xi := range x {
		q := inkling2MatVec(lw.Wq, xi)
		d.headNorm(q, hp.Heads, lw.QNorm)
		dflash2RoPE(q, hp.Heads, dim, pos+i, hp.RopeTheta)
		qs[i] = q
		k := inkling2MatVec(lw.Wk, xi)
		d.headNorm(k, hp.KVHeads, lw.KNorm)
		dflash2RoPE(k, hp.KVHeads, dim, pos+i, hp.RopeTheta)
		keys = append(keys, k)
		values = append(values, inkling2MatVec(lw.Wv, xi))
	}
	group := hp.Heads / hp.KVHeads
	scale := 1 / math.Sqrt(float64(dim))
	scores := make([]float64, len(keys))
	out := make([][]float32, len(x))
	for i, q := range qs {
		o := make([]float32, hp.Heads*dim)
		for hd := 0; hd < hp.Heads; hd++ {
			kv := (hd / group) * dim
			qh := q[hd*dim : (hd+1)*dim]
			maxS := math.Inf(-1)
			for j, k := range keys {
				scores[j] = float64(inkling2Dot(qh, k[kv:kv+dim])) * scale
				maxS = math.Max(maxS, scores[j])
			}
			var sum float64
			for j := range scores {
				scores[j] = math.Exp(scores[j] - maxS)
				sum += scores[j]
			}
			oh := o[hd*dim : (hd+1)*dim]
			for j, v := range values {
				w := float32(scores[j] / sum)
				for c := range oh {
					oh[c] += w * v[kv+c]
				}
			}
		}
		out[i] = inkling2MatVec(lw.Wo, o)
	}
	return out
}

func (d *Inkling2DSparkDraft) mix(features [][]float32, row int, norm []float32) []float32 {
	mixed := make([]float32, d.HP.Hidden)
	for t, f := range features {
		a := d.alpha[row][t]
		for j := range mixed {
			mixed[j] += a * f[j]
		}
	}
	return inkling2RMS(mixed, norm, d.HP.NormEps)
}

func (d *Inkling2DSparkDraft) headNorm(x []float32, heads int, w []float32) {
	dim := d.HP.HeadDim
	for h := 0; h < heads; h++ {
		seg := x[h*dim : (h+1)*dim]
		copy(seg, inkling2RMS(seg, w, d.HP.NormEps))
	}
}

func inkling2Softmax(x []float32) []float32 {
	maxV := math.Inf(-1)
	for _, v := range x {
		maxV = math.Max(maxV, float64(v))
	}
	out := make([]float32, len(x))
	var sum float64
	for _, v := range x {
		sum += math.Exp(float64(v) - maxV)
	}
	for i, v := range x {
		out[i] = float32(math.Exp(float64(v)-maxV) / sum)
	}
	return out
}

func inkling2Argmax(v []float32) int {
	best := 0
	for t := 1; t < len(v); t++ {
		if v[t] > v[best] {
			best = t
		}
	}
	return best
}

// shortlistArgmax is the draft token under d.Shortlist: the argmax of lm_head(y) over
// the Shortlist largest lm_head(h), or the argmax of lm_head(h) when negative.
func (d *Inkling2DSparkDraft) shortlistArgmax(h, y, logits []float32) int {
	vocab := d.HP.VocabUnpadded
	d.W.Output.MatVec(logits, h)
	if d.Shortlist < 0 {
		return inkling2Argmax(logits[:vocab])
	}
	top := make([]int, 0, d.Shortlist+1)
	for t := 0; t < vocab; t++ {
		i := len(top)
		for i > 0 && logits[top[i-1]] < logits[t] {
			i--
		}
		if i < d.Shortlist {
			top = append(top, 0)
			copy(top[i+1:], top[i:])
			top[i] = t
			if len(top) > d.Shortlist {
				top = top[:d.Shortlist]
			}
		}
	}
	full := make([]float32, len(logits))
	d.W.Output.MatVec(full, y)
	best := top[0]
	for _, t := range top[1:] {
		if full[t] > full[best] {
			best = t
		}
	}
	return best
}
