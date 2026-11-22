package model

import (
	"fmt"
	"math"
	"runtime"
	"sync"
)

// Inkling2LoopMatrix is a row-major weight W[rows][cols] that only exposes
// dst = W·x, so a provider can keep rows packed and dequantize on the fly.
type Inkling2LoopMatrix interface {
	Rows() int
	Cols() int
	MatVec(dst, x []float32)
}

// Inkling2LoopExperts is a stack of same-shape expert matrices.
type Inkling2LoopExperts interface {
	Experts() int
	Rows() int
	Cols() int
	ExpertMatVec(expert int, dst, x []float32)
}

// Inkling2LoopEmbedding returns one dequantized embedding row.
type Inkling2LoopEmbedding interface {
	Row(token int, dst []float32)
}

// Inkling2LoopHParams are the scalar hyperparameters of the forward, read from
// the checkpoint rather than pinned, so the varloop checkpoint shares the code.
type Inkling2LoopHParams struct {
	Hidden, Layers, Heads, KVHeads, HeadDim int
	VocabUnpadded                           int
	SlidingWindow, RelLength, RelExtent     int
	ConvKernel, AttnResBlock                int
	Experts, ExpertsUsed, ExpertWidth       int
	CoreStart, CoreEnd, Passes, GateRank    int
	LocalLayer                              []bool
	NormEps, QKClip, RelClip                float32
	LogScaleFloor, LogScaleAlpha            float32
	SituBeta, SituLinearBeta                float32
	ExpertWeightsScale, FinalLogitSoftcap   float32
}

// Inkling2LoopLayerWeights are one physical layer. Conv weights are [channel][tap],
// RelBias is [distance][RelLength], QClip is [head][HeadDim] and RelClip is
// [head][RelLength]. AttnResQuery has no rows on layer 0, one on local layers
// and three (Q, K, V) on global layers.
type Inkling2LoopLayerWeights struct {
	Wq, Wk, Wv, Wo, Gate, Rel, SinkProj     Inkling2LoopMatrix
	KConvProj, VConvProj                    Inkling2LoopMatrix
	AttnNorm, QInpNorm, KInpNorm, VInpNorm  []float32
	QClip, RelBias, RelClip                 []float32
	KConv, VConv, AttnPostConv, FFNPostConv []float32
	AttnResQuery                            [][]float32
	FFNResQuery, FFNNorm, ExpProbsBias      []float32
	Router                                  Inkling2LoopMatrix
	GateUp, Down                            Inkling2LoopExperts
	GateUpScale, DownScale                  []float32
}

// Inkling2LoopControllerWeights are the cross-pass controllers of one core layer,
// indexed Q, K, V, FFN.
type Inkling2LoopControllerWeights struct {
	MsgQuery [4][]float32
	Down     [4]Inkling2LoopMatrix
	Gain     [4]Inkling2LoopMatrix
	Forget   [4]Inkling2LoopMatrix
}

type Inkling2LoopWeights struct {
	TokEmbd        Inkling2LoopEmbedding
	TokNorm        []float32
	Output         Inkling2LoopMatrix
	OutputNorm     []float32
	OutputResQuery []float32
	Layers         []Inkling2LoopLayerWeights
	Controllers    []Inkling2LoopControllerWeights // CoreEnd-CoreStart entries
}

const (
	inkling2CtrlQ = iota
	inkling2CtrlK
	inkling2CtrlV
	inkling2CtrlFFN
)

// Inkling2LoopHost is the host reference forward: one token per Step, f32
// activations, every layer visit with its own KV and short-conv state.
type Inkling2LoopHost struct {
	HP  Inkling2LoopHParams
	W   *Inkling2LoopWeights
	pos int
	// per cache slot: keys/values [token][kvHeads*headDim], conv history
	// [channel][kernel-1]
	keys, values [][][]float32
	positions    [][]int
	conv         [][]float32
}

func NewInkling2LoopHost(hp Inkling2LoopHParams, w *Inkling2LoopWeights) (*Inkling2LoopHost, error) {
	if hp.CoreStart <= 0 || hp.CoreEnd <= hp.CoreStart || hp.CoreEnd > hp.Layers || hp.Passes < 1 ||
		hp.CoreStart%hp.AttnResBlock != 0 || hp.CoreEnd%hp.AttnResBlock != 0 {
		return nil, fmt.Errorf("inkling2_loop host: invalid loop core [%d,%d) x%d", hp.CoreStart, hp.CoreEnd, hp.Passes)
	}
	if len(w.Layers) != hp.Layers || len(w.Controllers) != hp.CoreEnd-hp.CoreStart || len(hp.LocalLayer) != hp.Layers {
		return nil, fmt.Errorf("inkling2_loop host: %d layers, %d controllers for %d layers", len(w.Layers), len(w.Controllers), hp.Layers)
	}
	slots := hp.Layers + (hp.Passes-1)*(hp.CoreEnd-hp.CoreStart)
	h := &Inkling2LoopHost{HP: hp, W: w}
	h.keys = make([][][]float32, slots)
	h.values = make([][][]float32, slots)
	h.positions = make([][]int, slots)
	h.conv = make([][]float32, slots)
	channels := 2*hp.KVHeads*hp.HeadDim + 2*hp.Hidden
	for i := range h.conv {
		h.conv[i] = make([]float32, channels*(hp.ConvKernel-1))
	}
	return h, nil
}

// Slots is the number of layer visits per token (cache slots).
func (h *Inkling2LoopHost) Slots() int { return len(h.conv) }

// Step consumes one token at the next position and returns its unpadded logits.
func (h *Inkling2LoopHost) Step(token int) []float32 {
	hp := h.HP
	cur := make([]float32, hp.Hidden)
	h.W.TokEmbd.Row(token, cur)
	cur = inkling2RMS(cur, h.W.TokNorm, hp.NormEps)

	var blocks [][]float32
	for il := 0; il < hp.CoreStart; il++ {
		cur = h.layer(il, il, cur, &blocks, nil)
	}
	prelude := cur
	preludeBlocks := append([][]float32(nil), blocks...)
	core := hp.CoreEnd - hp.CoreStart
	for pass := 0; pass < hp.Passes; pass++ {
		var bank [][]float32
		var bankOut []float32
		if pass > 0 {
			bank = append(bank, blocks[len(preludeBlocks)+1:]...)
			bankOut = cur
			cur = prelude
			blocks = append([][]float32(nil), preludeBlocks...)
		}
		for ip := hp.CoreStart; ip < hp.CoreEnd; ip++ {
			if pass == 0 {
				cur = h.layer(ip, ip, cur, &blocks, nil)
				continue
			}
			slot := hp.Layers + (pass-1)*core + (ip - hp.CoreStart)
			ctrl := &h.W.Controllers[ip-hp.CoreStart]
			msgs := make([][]float32, 4)
			for c := range msgs {
				msgs[c] = inkling2Mix(bank, bankOut, ctrl.MsgQuery[c])
			}
			cur = h.layer(slot, ip, cur, &blocks, &inkling2Messages{ctrl: ctrl, msgs: msgs})
		}
	}
	for il := hp.CoreEnd; il < hp.Layers; il++ {
		cur = h.layer(il, il, cur, &blocks, nil)
	}

	cur = inkling2Mix(blocks, cur, h.W.OutputResQuery)
	cur = inkling2RMS(cur, h.W.OutputNorm, hp.NormEps)
	logits := make([]float32, h.W.Output.Rows())
	h.W.Output.MatVec(logits, cur)
	if c := hp.FinalLogitSoftcap; c != 0 {
		for i, v := range logits {
			logits[i] = c * float32(math.Tanh(float64(v/c)))
		}
	}
	h.pos++
	return logits[:hp.VocabUnpadded]
}

type inkling2Messages struct {
	ctrl *Inkling2LoopControllerWeights
	msgs [][]float32
}

func (h *Inkling2LoopHost) layer(slot, ip int, cur []float32, blocks *[][]float32, m *inkling2Messages) []float32 {
	hp := h.HP
	lw := &h.W.Layers[ip]
	local := hp.LocalLayer[ip]
	kvDim := hp.KVHeads * hp.HeadDim
	inpSA := cur

	inputs := [3][]float32{cur, cur, cur}
	if len(lw.AttnResQuery) > 0 {
		for j := 0; j < 3; j++ {
			q := lw.AttnResQuery[0]
			if !local {
				q = lw.AttnResQuery[j]
			}
			inputs[j] = inkling2Mix(*blocks, cur, q)
		}
	}
	if m != nil {
		for j := 0; j < 3; j++ {
			inputs[j] = inkling2Control(inputs[j], m.msgs[j], m.ctrl, j)
		}
	}
	switch {
	case local && m != nil:
		for j := 0; j < 3; j++ {
			inputs[j] = inkling2RMS(inputs[j], lw.AttnNorm, hp.NormEps)
		}
	case local:
		inputs[0] = inkling2RMS(inputs[0], lw.AttnNorm, hp.NormEps)
		inputs[1], inputs[2] = inputs[0], inputs[0]
	default:
		inputs[0] = inkling2RMS(inputs[0], lw.QInpNorm, hp.NormEps)
		inputs[1] = inkling2RMS(inputs[1], lw.KInpNorm, hp.NormEps)
		inputs[2] = inkling2RMS(inputs[2], lw.VInpNorm, hp.NormEps)
	}

	q := inkling2MatVec(lw.Wq, inputs[0])
	var kDyn, vDyn []float32
	if !local {
		kDyn = inkling2MatVec(lw.KConvProj, inputs[1])
		vDyn = inkling2MatVec(lw.VConvProj, inputs[2])
	}
	k := h.convolve(slot, inkling2MatVec(lw.Wk, inputs[1]), lw.KConv, kDyn, hp.KVHeads, 0)
	v := h.convolve(slot, inkling2MatVec(lw.Wv, inputs[2]), lw.VConv, vDyn, hp.KVHeads, kvDim)
	r := inkling2MatVec(lw.Rel, inputs[0])
	sinks := inkling2MatVec(lw.SinkProj, inputs[0])

	for hd := 0; hd < hp.Heads; hd++ {
		inkling2Clip(q[hd*hp.HeadDim:(hd+1)*hp.HeadDim], lw.QClip[hd*hp.HeadDim:(hd+1)*hp.HeadDim], hp.QKClip)
		inkling2Clip(r[hd*hp.RelLength:(hd+1)*hp.RelLength], lw.RelClip[hd*hp.RelLength:(hd+1)*hp.RelLength], hp.RelClip)
	}
	for hd := 0; hd < hp.KVHeads; hd++ {
		inkling2Clip(k[hd*hp.HeadDim:(hd+1)*hp.HeadDim], nil, hp.QKClip)
		seg := v[hd*hp.HeadDim : (hd+1)*hp.HeadDim]
		copy(seg, inkling2RMS(seg, nil, hp.NormEps))
	}
	if !local {
		tau := float32(1 + float64(hp.LogScaleAlpha)*math.Log(math.Max(1, float64(h.pos+1)/float64(hp.LogScaleFloor))))
		for i := range q {
			q[i] *= tau
		}
		for i := range r {
			r[i] *= tau
		}
	}

	h.keys[slot] = append(h.keys[slot], k)
	h.values[slot] = append(h.values[slot], v)
	h.positions[slot] = append(h.positions[slot], h.pos)

	extent := hp.RelExtent
	if local {
		extent = hp.SlidingWindow
	}
	attn := h.attend(slot, q, r, sinks, lw.RelBias, extent, local)

	gate := inkling2MatVec(lw.Gate, inputs[0])
	for i := range attn {
		attn[i] *= sigmoid32(gate[i])
	}
	out := h.convolve(slot, inkling2MatVec(lw.Wo, attn), lw.AttnPostConv, nil, 0, 2*kvDim)

	if ip%hp.AttnResBlock == 0 {
		*blocks = append(*blocks, inpSA)
		cur = out
	} else {
		cur = inkling2Add(inpSA, out)
	}
	ffnInp := cur

	x := inkling2Mix(*blocks, cur, lw.FFNResQuery)
	if m != nil {
		x = inkling2Control(x, m.msgs[inkling2CtrlFFN], m.ctrl, inkling2CtrlFFN)
	}
	x = inkling2RMS(x, lw.FFNNorm, hp.NormEps)
	x = h.moe(lw, x)
	x = h.convolve(slot, x, lw.FFNPostConv, nil, 0, 2*kvDim+hp.Hidden)
	return inkling2Add(ffnInp, x)
}

// attend is causal attention for the newest token of one slot with the
// relative-position bias table, a per-head sink logit and GQA sharing.
func (h *Inkling2LoopHost) attend(slot int, q, r, sinks, relBias []float32, extent int, local bool) []float32 {
	hp := h.HP
	keys, values, positions := h.keys[slot], h.values[slot], h.positions[slot]
	group := hp.Heads / hp.KVHeads
	scale := float32(1 / math.Sqrt(float64(hp.HeadDim)))
	out := make([]float32, hp.Heads*hp.HeadDim)
	scores := make([]float64, len(keys))
	for hd := 0; hd < hp.Heads; hd++ {
		kvh := hd / group
		qh := q[hd*hp.HeadDim : (hd+1)*hp.HeadDim]
		rh := r[hd*hp.RelLength : (hd+1)*hp.RelLength]
		maxS := math.Inf(-1)
		for j := range keys {
			dist := h.pos - positions[j]
			if dist < 0 || (local && dist >= hp.SlidingWindow) {
				scores[j] = math.Inf(-1)
				continue
			}
			bias := -sinks[hd]
			if dist < extent {
				bias += inkling2Dot(relBias[dist*hp.RelLength:(dist+1)*hp.RelLength], rh)
			}
			s := float64(inkling2Dot(qh, keys[j][kvh*hp.HeadDim:(kvh+1)*hp.HeadDim])*scale + bias)
			scores[j] = s
			if s > maxS {
				maxS = s
			}
		}
		// the zero sink logit joins the softmax denominator only
		if maxS < 0 {
			maxS = 0
		}
		denom := math.Exp(-maxS)
		for j := range scores {
			if !math.IsInf(scores[j], -1) {
				scores[j] = math.Exp(scores[j] - maxS)
				denom += scores[j]
			} else {
				scores[j] = 0
			}
		}
		oh := out[hd*hp.HeadDim : (hd+1)*hp.HeadDim]
		for j, p := range scores {
			if p == 0 {
				continue
			}
			w := float32(p / denom)
			vj := values[j][kvh*hp.HeadDim : (kvh+1)*hp.HeadDim]
			for d := range oh {
				oh[d] += w * vj[d]
			}
		}
	}
	return out
}

// convolve is the causal short convolution with residual over channels
// [offset, offset+len(cur)) of the slot's conv history. dynamic adds per-head
// input-dependent taps laid out [head][tap].
func (h *Inkling2LoopHost) convolve(slot int, cur, weight, dynamic []float32, heads, offset int) []float32 {
	kn := h.HP.ConvKernel
	past := kn - 1
	hist := h.conv[slot]
	out := make([]float32, len(cur))
	perHead := 0
	if dynamic != nil {
		perHead = len(cur) / heads
	}
	norm := float32(1 / math.Sqrt(float64(kn+1)))
	if dynamic != nil {
		norm = float32(1 / math.Sqrt(float64(2*kn+1)))
	}
	for c := range cur {
		hc := hist[(offset+c)*past : (offset+c+1)*past]
		w := weight[c*kn : (c+1)*kn]
		// window x[0..kn): past inputs oldest first, then the current input
		var acc float32
		for t := 0; t < past; t++ {
			acc += hc[t] * w[t]
		}
		acc += cur[c] * w[past]
		if dynamic != nil {
			hd := c / perHead
			for lag := 0; lag < kn; lag++ {
				xv := cur[c]
				if lag > 0 {
					xv = hc[past-lag]
				}
				acc += dynamic[hd*kn+lag] * xv
			}
		}
		out[c] = (cur[c] + acc) * norm
		copy(hc, hc[1:])
		hc[past-1] = cur[c]
	}
	return out
}

func (h *Inkling2LoopHost) moe(lw *Inkling2LoopLayerWeights, x []float32) []float32 {
	hp := h.HP
	logits := inkling2MatVec(lw.Router, x)
	probs := make([]float32, len(logits))
	sel := make([]float32, len(logits))
	for i, l := range logits {
		probs[i] = sigmoid32(l)
		sel[i] = probs[i]
		if lw.ExpProbsBias != nil {
			sel[i] += lw.ExpProbsBias[i]
		}
	}
	top := inkling2TopK(sel, hp.ExpertsUsed)
	weights := make([]float32, len(top))
	var sum float32
	for i, e := range top {
		weights[i] = probs[e]
		sum += weights[i]
	}
	for i := range weights {
		weights[i] = weights[i] / (sum + 1e-20)
		if hp.ExpertWeightsScale != 0 {
			weights[i] *= hp.ExpertWeightsScale
		}
	}
	width := hp.ExpertWidth
	outs := make([][]float32, len(top))
	var wg sync.WaitGroup
	for i, e := range top {
		wg.Add(1)
		go func(i, e int) {
			defer wg.Done()
			gu := make([]float32, 2*width)
			lw.GateUp.ExpertMatVec(e, gu, x)
			if lw.GateUpScale != nil {
				for j := range gu {
					gu[j] *= lw.GateUpScale[e]
				}
			}
			act := make([]float32, width)
			beta, lb := float64(hp.SituBeta), float64(hp.SituLinearBeta)
			for j := 0; j < width; j++ {
				g, u := float64(gu[j]), float64(gu[width+j])
				a := beta * math.Tanh(g/beta) * (1 / (1 + math.Exp(-g)))
				if lb > 0 {
					u = lb * math.Tanh(u/lb)
				}
				act[j] = float32(a * u)
			}
			d := make([]float32, hp.Hidden)
			lw.Down.ExpertMatVec(e, d, act)
			if lw.DownScale != nil {
				for j := range d {
					d[j] *= lw.DownScale[e]
				}
			}
			outs[i] = d
		}(i, e)
	}
	wg.Wait()
	out := make([]float32, hp.Hidden)
	for i, d := range outs {
		for j := range out {
			out[j] += weights[i] * d[j]
		}
	}
	return out
}

// inkling2Mix is depth attention: RMS-normalize each earlier block output and
// cur, score them against query and return the softmax-weighted normalized sum.
func inkling2Mix(blocks [][]float32, cur, query []float32) []float32 {
	states := make([][]float32, 0, len(blocks)+1)
	for _, b := range blocks {
		states = append(states, inkling2RMS(b, nil, 1e-6))
	}
	states = append(states, inkling2RMS(cur, nil, 1e-6))
	inv := 1 / math.Sqrt(float64(len(cur)))
	scores := make([]float64, len(states))
	maxS := math.Inf(-1)
	for i, s := range states {
		scores[i] = float64(inkling2Dot(s, query)) * inv
		maxS = math.Max(maxS, scores[i])
	}
	var sum float64
	for i := range scores {
		scores[i] = math.Exp(scores[i] - maxS)
		sum += scores[i]
	}
	out := make([]float32, len(cur))
	for i, s := range states {
		w := float32(scores[i] / sum)
		for j := range out {
			out[j] += w * s[j]
		}
	}
	return out
}

// inkling2Control mixes a cross-pass message: (1-forget)*cur + gain*message,
// both gates read from the normalized pair.
func inkling2Control(cur, msg []float32, ctrl *Inkling2LoopControllerWeights, c int) []float32 {
	pair := append(inkling2RMS(cur, nil, 1e-6), inkling2RMS(msg, nil, 1e-6)...)
	g := inkling2MatVec(ctrl.Down[c], pair)
	gain := inkling2MatVec(ctrl.Gain[c], g)
	forget := inkling2MatVec(ctrl.Forget[c], g)
	out := make([]float32, len(cur))
	for i := range out {
		f := float32(math.Tanh(float64(forget[i])))
		gn := float32(math.Tanh(float64(gain[i])))
		out[i] = cur[i] - f*cur[i] + gn*msg[i]
	}
	return out
}

// inkling2Clip scales x (after an optional per-channel scale) down to an L2
// norm of at most threshold, in place.
func inkling2Clip(x, scale []float32, threshold float32) {
	if scale != nil {
		for i := range x {
			x[i] *= scale[i]
		}
	}
	var ss float32
	for _, v := range x {
		ss += v * v
	}
	norm := float32(math.Sqrt(float64(ss))) + 1e-6
	f := threshold / norm
	if f > 1 {
		f = 1
	}
	if f < 0 {
		f = 0
	}
	for i := range x {
		x[i] *= f
	}
}

func inkling2RMS(x, w []float32, eps float32) []float32 {
	var ss float64
	for _, v := range x {
		ss += float64(v) * float64(v)
	}
	s := float32(1 / math.Sqrt(ss/float64(len(x))+float64(eps)))
	out := make([]float32, len(x))
	for i, v := range x {
		out[i] = v * s
		if w != nil {
			out[i] *= w[i]
		}
	}
	return out
}

func inkling2MatVec(m Inkling2LoopMatrix, x []float32) []float32 {
	dst := make([]float32, m.Rows())
	m.MatVec(dst, x)
	return dst
}

func inkling2Add(a, b []float32) []float32 {
	out := make([]float32, len(a))
	for i := range a {
		out[i] = a[i] + b[i]
	}
	return out
}

func inkling2Dot(a, b []float32) float32 {
	var s float32
	for i := range a {
		s += a[i] * b[i]
	}
	return s
}

// inkling2TopK returns the k largest indices, ties broken by lower index.
func inkling2TopK(v []float32, k int) []int {
	idx := make([]int, 0, k)
	used := make([]bool, len(v))
	for n := 0; n < k; n++ {
		best := -1
		for i, x := range v {
			if !used[i] && (best < 0 || x > v[best]) {
				best = i
			}
		}
		used[best] = true
		idx = append(idx, best)
	}
	return idx
}

// Inkling2LoopDenseMatrix is an f32 row-major matrix, for tests and the small
// F32/BF16 checkpoint tensors.
type Inkling2LoopDenseMatrix struct {
	R, C int
	Data []float32
}

func (m *Inkling2LoopDenseMatrix) Rows() int { return m.R }
func (m *Inkling2LoopDenseMatrix) Cols() int { return m.C }
func (m *Inkling2LoopDenseMatrix) MatVec(dst, x []float32) {
	inkling2ParallelRows(m.R, func(r int) { dst[r] = inkling2Dot(m.Data[r*m.C:(r+1)*m.C], x) })
}

// inkling2ParallelRows runs fn over [0,n) on all CPUs for large row counts.
func inkling2ParallelRows(n int, fn func(int)) {
	workers := runtime.GOMAXPROCS(0)
	if n < 256 || workers == 1 {
		for r := 0; r < n; r++ {
			fn(r)
		}
		return
	}
	chunk := (n + workers - 1) / workers
	var wg sync.WaitGroup
	for start := 0; start < n; start += chunk {
		end := min(start+chunk, n)
		wg.Add(1)
		go func(a, b int) {
			defer wg.Done()
			for r := a; r < b; r++ {
				fn(r)
			}
		}(start, end)
	}
	wg.Wait()
}

// Inkling2LoopParallelRows exposes the row fan-out to weight providers.
func Inkling2LoopParallelRows(n int, fn func(int)) { inkling2ParallelRows(n, fn) }
