package ggufload

import (
	"fmt"
	"math"
	"os"
	"runtime"
	"time"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/model"
)

// inkling2ResidentCtx is the global-slot KV capacity of a resident session; local
// slots keep a sliding-window ring.
const inkling2ResidentCtx = 4096

// inkling2ArenaFloats bounds the per-token scratch arena. Every intermediate of a
// token takes the next arena slot, so a token's device addresses repeat exactly
// on the next token and a captured graph stays valid.
const inkling2ArenaFloats = 16 << 20

type ink2Ref = compute.Ink2Ref

// ink2Mat is one weight the device multiplies: a resident tensor (any MatMul dtype).
type ink2Mat struct {
	t          compute.Tensor
	rows, cols int
}

type ink2Experts struct {
	stack      compute.Tensor
	views      []compute.Tensor
	rows, padK int
	scale      ink2Ref
}

type ink2Layer struct {
	wq, wk, wv, wo, gate, rel, sink, kconv, vconv, router ink2Mat
	attnNorm, qNorm, kNorm, vNorm                         ink2Ref
	qclip, relBias, relClip                               ink2Ref
	kconvW, vconvW, attnPostConv, ffnPostConv             ink2Ref
	resQ                                                  ink2Ref
	resQRows                                              int
	ffnResQ, ffnNorm, probsBias                           ink2Ref
	gateUp, down                                          ink2Experts
}

type ink2Ctrl struct {
	msgQ         ink2Ref
	down         [4]ink2Ref
	gain, forget [4]ink2Ref
	rank         int
}

// inkling2Resident runs the inkling2_loop decode forward with every activation,
// cache and routing decision resident on a device; per token it uploads the
// embedding row and position and reads back one argmax index.
type inkling2Resident struct {
	hp     model.Inkling2LoopHParams
	w      *model.Inkling2LoopWeights
	be     compute.Backend
	dev    compute.Inkling2Device
	owned  []compute.Tensor
	layers []ink2Layer
	ctrls  []ink2Ctrl

	tokNorm, outNorm, outResQ ink2Ref
	output                    ink2Mat

	in, arena, scores ink2Ref
	argIdx            ink2Ref
	kc, vc, hist      []ink2Ref
	caps              []int
	off               int
	pos               int
	err               error

	q8 map[int]ink2Ref // arena offset of an activation -> its Q8_0 codes this token

	graph       bool
	graphReady  bool
	hostRouting bool // an expert dtype has no indexed kernel: ids are read back per layer
}

func (s *inkling2Resident) fail(err error) {
	if err != nil && s.err == nil {
		s.err = err
	}
}

func (s *inkling2Resident) alloc(n int) ink2Ref {
	t, err := s.dev.Ink2Alloc(n)
	if err != nil {
		s.fail(err)
		return ink2Ref{}
	}
	s.owned = append(s.owned, t)
	return compute.Ink2At(t, 0)
}

func (s *inkling2Resident) upload(v []float32) ink2Ref {
	if v == nil {
		return ink2Ref{}
	}
	r := s.alloc(len(v))
	if s.err == nil {
		s.fail(s.dev.Ink2Write(r, v))
	}
	return r
}

func (s *inkling2Resident) uploadF32Mat(rows, cols int, data []float32) ink2Mat {
	r := s.upload(data)
	if s.err != nil {
		return ink2Mat{}
	}
	t := r.T
	t.Shape = []int{rows, cols}
	return ink2Mat{t: t, rows: rows, cols: cols}
}

// mat makes a device weight of m: resident tensors are used as bound, host-side
// small dense matrices are widened to F32 and uploaded.
func (s *inkling2Resident) mat(m model.Inkling2LoopMatrix) ink2Mat {
	switch v := m.(type) {
	case nil:
		return ink2Mat{}
	case *inkling2DeviceMatrix:
		if v.padK != v.cols {
			s.fail(fmt.Errorf("inkling2 resident: padded dense weight [%d %d] pad %d", v.rows, v.cols, v.padK))
		}
		return ink2Mat{t: v.w, rows: v.rows, cols: v.cols}
	case *inkling2QuantMatrix:
		f := make([]float32, v.rows*v.cols)
		for r := 0; r < v.rows; r++ {
			if err := inkling2DequantRow(v.typ, v.raw[r*v.rowBytes:(r+1)*v.rowBytes], f[r*v.cols:(r+1)*v.cols]); err != nil {
				s.fail(err)
				return ink2Mat{}
			}
		}
		return s.uploadF32Mat(v.rows, v.cols, f)
	case *model.Inkling2LoopDenseMatrix:
		return s.uploadF32Mat(v.R, v.C, v.Data)
	}
	s.fail(fmt.Errorf("inkling2 resident: matrix %T is not device-bindable", m))
	return ink2Mat{}
}

func (s *inkling2Resident) experts(e model.Inkling2LoopExperts, scale []float32) ink2Experts {
	d, ok := e.(*inkling2DeviceExperts)
	if !ok {
		s.fail(fmt.Errorf("inkling2 resident: experts %T are not device-resident", e))
		return ink2Experts{}
	}
	return ink2Experts{stack: d.stack, views: d.w, rows: d.rows, padK: d.padK, scale: s.upload(scale)}
}

func concatRows(rows [][]float32) []float32 {
	var out []float32
	for _, r := range rows {
		out = append(out, r...)
	}
	return out
}

func newInkling2Resident(m *Inkling2LoopModel, be compute.Backend) (*inkling2Resident, error) {
	dev, ok := be.(compute.Inkling2Device)
	if !ok {
		return nil, fmt.Errorf("inkling2 resident: backend %s has no inkling2 device ops", be.Name())
	}
	hp, w := m.HP, m.Weights
	s := &inkling2Resident{hp: hp, w: w, be: be, dev: dev, q8: map[int]ink2Ref{}}
	s.tokNorm, s.outNorm, s.outResQ = s.upload(w.TokNorm), s.upload(w.OutputNorm), s.upload(w.OutputResQuery)
	s.output = s.mat(w.Output)
	s.layers = make([]ink2Layer, hp.Layers)
	for i := range w.Layers {
		lw, l := &w.Layers[i], &s.layers[i]
		l.wq, l.wk, l.wv, l.wo = s.mat(lw.Wq), s.mat(lw.Wk), s.mat(lw.Wv), s.mat(lw.Wo)
		l.gate, l.rel, l.sink = s.mat(lw.Gate), s.mat(lw.Rel), s.mat(lw.SinkProj)
		l.kconv, l.vconv, l.router = s.mat(lw.KConvProj), s.mat(lw.VConvProj), s.mat(lw.Router)
		l.attnNorm, l.qNorm, l.kNorm, l.vNorm = s.upload(lw.AttnNorm), s.upload(lw.QInpNorm), s.upload(lw.KInpNorm), s.upload(lw.VInpNorm)
		l.qclip, l.relBias, l.relClip = s.upload(lw.QClip), s.upload(lw.RelBias), s.upload(lw.RelClip)
		l.kconvW, l.vconvW = s.upload(lw.KConv), s.upload(lw.VConv)
		l.attnPostConv, l.ffnPostConv = s.upload(lw.AttnPostConv), s.upload(lw.FFNPostConv)
		l.resQRows = len(lw.AttnResQuery)
		l.resQ = s.upload(concatRows(lw.AttnResQuery))
		l.ffnResQ, l.ffnNorm, l.probsBias = s.upload(lw.FFNResQuery), s.upload(lw.FFNNorm), s.upload(lw.ExpProbsBias)
		l.gateUp = s.experts(lw.GateUp, lw.GateUpScale)
		l.down = s.experts(lw.Down, lw.DownScale)
		if s.err != nil {
			s.close()
			return nil, fmt.Errorf("inkling2 resident: layer %d: %w", i, s.err)
		}
	}
	s.ctrls = make([]ink2Ctrl, len(w.Controllers))
	for i := range w.Controllers {
		cw, c := &w.Controllers[i], &s.ctrls[i]
		c.msgQ = s.upload(concatRows(cw.MsgQuery[:]))
		for j := 0; j < 4; j++ {
			c.down[j] = s.f32Ref(cw.Down[j])
			c.rank = cw.Down[j].Rows()
			c.gain[j] = s.f32Ref(cw.Gain[j])
			c.forget[j] = s.f32Ref(cw.Forget[j])
		}
	}

	slots := hp.Layers + (hp.Passes-1)*(hp.CoreEnd-hp.CoreStart)
	kvDim := hp.KVHeads * hp.HeadDim
	channels := 2*kvDim + 2*hp.Hidden
	s.kc, s.vc, s.hist, s.caps = make([]ink2Ref, slots), make([]ink2Ref, slots), make([]ink2Ref, slots), make([]int, slots)
	maxCap := hp.SlidingWindow
	for slot := 0; slot < slots; slot++ {
		cap := inkling2ResidentCtx
		if hp.LocalLayer[inkling2SlotLayer(hp, slot)] {
			cap = hp.SlidingWindow
		}
		maxCap = max(maxCap, cap)
		s.caps[slot] = cap
		s.kc[slot], s.vc[slot] = s.alloc(cap*kvDim), s.alloc(cap*kvDim)
		s.hist[slot] = s.alloc(channels * (hp.ConvKernel - 1))
	}
	s.scores = s.alloc(2 * hp.Heads * maxCap)
	s.in = s.alloc(hp.Hidden + 4)
	s.arena = s.alloc(inkling2ArenaFloats)
	s.argIdx = s.alloc(4)
	for i := range s.layers {
		for _, e := range []ink2Experts{s.layers[i].gateUp, s.layers[i].down} {
			if !inkling2IndexedExpert(e.stack.Dtype) {
				s.hostRouting = true
			}
		}
	}
	s.graph = inkling2DefaultOn(be, "FAK_INKLING2_GRAPH", os.LookupEnv) && !s.hostRouting
	if s.err != nil {
		s.close()
		return nil, s.err
	}
	return s, nil
}

func (m ink2Mat) ref() ink2Ref { return compute.Ink2At(m.t, 0) }

// f32Ref is a weight the controller kernel reads directly as dense F32 rows.
func (s *inkling2Resident) f32Ref(m model.Inkling2LoopMatrix) ink2Ref {
	d := s.mat(m)
	if s.err == nil && d.t.Dtype != compute.F32 {
		s.fail(fmt.Errorf("inkling2 resident: controller weight is %s, want F32", d.t.Dtype))
	}
	return d.ref()
}

func inkling2IndexedExpert(dt compute.Dtype) bool {
	switch dt {
	case compute.NVFP4, compute.IQ2_XS, compute.IQ2_S, compute.IQ3_XXS, compute.IQ3_S, compute.IQ4_NL, compute.Q3_K:
		return true
	}
	return false
}

// inkling2SlotLayer maps a cache slot to its physical layer.
func inkling2SlotLayer(hp model.Inkling2LoopHParams, slot int) int {
	if slot < hp.Layers {
		return slot
	}
	core := hp.CoreEnd - hp.CoreStart
	return hp.CoreStart + (slot-hp.Layers)%core
}

func (s *inkling2Resident) close() {
	if s.dev != nil {
		s.dev.Ink2GraphReset()
	}
	for _, t := range s.owned {
		s.be.Free(t)
	}
	s.owned = nil
}

func (s *inkling2Resident) tmp(n int) ink2Ref {
	n = (n + 63) &^ 63
	if s.off+n > inkling2ArenaFloats {
		s.fail(fmt.Errorf("inkling2 resident: arena exhausted at %d floats", s.off+n))
		return s.arena
	}
	r := compute.Ink2At(s.arena.T, s.off)
	s.off += n
	return r
}

func at(r ink2Ref, off int) ink2Ref { return compute.Ink2At(r.T, r.Off+off) }

func (s *inkling2Resident) mv(m ink2Mat, x ink2Ref) ink2Ref {
	y := s.tmp(m.rows)
	if s.err != nil {
		return y
	}
	if m.t.Dtype == compute.Q8_0 && x.T.Buf() == s.arena.T.Buf() {
		// arena slots are never reused within a token, so the offset names the value
		q, seen := s.q8[x.Off]
		if !seen {
			q = s.tmp(m.cols/4 + m.cols/32)
			s.q8[x.Off] = q
		}
		ok, err := s.dev.Ink2MatVecQ8(m.t, x, ink2Ref{}, q, !seen, y)
		if err != nil || ok {
			s.fail(err)
			return y
		}
	}
	s.fail(s.dev.Ink2MatVec(m.t, x, y))
	return y
}

// mvGated is mv(m, x * sigmoid(gate)); on a Q8_0 weight the gate is applied while x
// is quantized, so x itself is left unscaled.
func (s *inkling2Resident) mvGated(m ink2Mat, x, gate ink2Ref, n int) ink2Ref {
	if s.err == nil && m.t.Dtype == compute.Q8_0 {
		y := s.tmp(m.rows)
		ok, err := s.dev.Ink2MatVecQ8(m.t, x, gate, s.tmp(m.cols/4+m.cols/32), true, y)
		if err != nil || ok {
			s.fail(err)
			return y
		}
	}
	if s.err == nil {
		s.fail(s.dev.Ink2SigmoidMul(x, gate, n))
	}
	return s.mv(m, x)
}

func (s *inkling2Resident) rms(x, w ink2Ref, n int, eps float32) ink2Ref {
	y := s.tmp(n)
	if s.err == nil {
		s.fail(s.dev.Ink2RMS(x, w, y, 1, n, eps))
	}
	return y
}

func (s *inkling2Resident) mix(states []ink2Ref, queries ink2Ref, nq int) ink2Ref {
	return s.mixNorm(states, queries, nq, nil)
}

// mixNorm is mix with each query row RMS-normalized by its weight in norms, in the
// same launch.
func (s *inkling2Resident) mixNorm(states []ink2Ref, queries ink2Ref, nq int, norms []ink2Ref) ink2Ref {
	out := s.tmp(nq * s.hp.Hidden)
	if s.err == nil {
		s.fail(s.dev.Ink2Mix(states, queries, nq, norms, s.hp.NormEps, out, s.hp.Hidden))
	}
	return out
}

// control runs controllers js of c on (curs[i], msgs[i]) in one batch and returns the
// outputs as consecutive rows, each RMS-normalized by norms[i].
func (s *inkling2Resident) control(curs, msgs []ink2Ref, c *ink2Ctrl, js []int, norms []ink2Ref) ink2Ref {
	h := s.hp.Hidden
	ctl := compute.Ink2Controls{Cur: curs, Msg: msgs}
	for _, j := range js {
		ctl.Down = append(ctl.Down, c.down[j])
		ctl.Gain = append(ctl.Gain, c.gain[j])
		ctl.Forget = append(ctl.Forget, c.forget[j])
	}
	g := s.tmp(len(js) * c.rank)
	out := s.tmp(len(js) * h)
	if s.err == nil {
		s.fail(s.dev.Ink2Control(ctl, g, out, h, c.rank, norms, s.hp.NormEps))
	}
	return out
}

func (s *inkling2Resident) conv(slot int, x, weight, dyn, resid ink2Ref, n, heads, offset int) ink2Ref {
	kn := s.hp.ConvKernel
	norm := inkling2ConvNorm(kn, dyn)
	out := s.tmp(n)
	if s.err == nil {
		s.fail(s.dev.Ink2Conv(x, weight, dyn, at(s.hist[slot], offset*(kn-1)), out, resid, n, max(heads, 1), kn, norm))
	}
	return out
}

// inkling2ConvNorm is the output scale of a short conv of width kn, with or without
// dynamic taps.
func inkling2ConvNorm(kn int, dyn ink2Ref) float32 {
	if dyn.T.Buf() != nil {
		return float32(1 / math.Sqrt(float64(2*kn+1)))
	}
	return float32(1 / math.Sqrt(float64(kn+1)))
}

func with(blocks []ink2Ref, cur ink2Ref) []ink2Ref {
	return append(append([]ink2Ref(nil), blocks...), cur)
}

func (s *inkling2Resident) layer(slot, ip int, cur ink2Ref, blocks *[]ink2Ref, msgs []ink2Ref, c *ink2Ctrl) ink2Ref {
	hp := s.hp
	l := &s.layers[ip]
	local := hp.LocalLayer[ip]
	h := hp.Hidden
	kvDim := hp.KVHeads * hp.HeadDim
	inpSA := cur

	in := [3]ink2Ref{cur, cur, cur}
	normed := false
	if l.resQRows > 0 && msgs == nil {
		normed = true
		if local {
			m := s.mixNorm(with(*blocks, cur), l.resQ, 1, []ink2Ref{l.attnNorm})
			in = [3]ink2Ref{m, m, m}
		} else {
			m := s.mixNorm(with(*blocks, cur), l.resQ, 3, []ink2Ref{l.qNorm, l.kNorm, l.vNorm})
			in = [3]ink2Ref{m, at(m, h), at(m, 2*h)}
		}
	} else if l.resQRows > 0 {
		if local {
			m := s.mix(with(*blocks, cur), l.resQ, 1)
			in = [3]ink2Ref{m, m, m}
		} else {
			m := s.mix(with(*blocks, cur), l.resQ, 3)
			in = [3]ink2Ref{m, at(m, h), at(m, 2*h)}
		}
	}
	if msgs != nil {
		norms := []ink2Ref{l.qNorm, l.kNorm, l.vNorm}
		if local {
			norms = []ink2Ref{l.attnNorm, l.attnNorm, l.attnNorm}
		}
		o := s.control(in[:], msgs[:3], c, []int{0, 1, 2}, norms)
		in = [3]ink2Ref{o, at(o, h), at(o, 2*h)}
		normed = true
	}
	switch {
	case normed:
	case local:
		n0 := s.rms(in[0], l.attnNorm, h, hp.NormEps)
		in = [3]ink2Ref{n0, n0, n0}
	default:
		in[0] = s.rms(in[0], l.qNorm, h, hp.NormEps)
		in[1] = s.rms(in[1], l.kNorm, h, hp.NormEps)
		in[2] = s.rms(in[2], l.vNorm, h, hp.NormEps)
	}

	q := s.mv(l.wq, in[0])
	var kDyn, vDyn ink2Ref
	if !local {
		kDyn = s.mv(l.kconv, in[1])
		vDyn = s.mv(l.vconv, in[2])
	}
	k, v := s.mv(l.wk, in[1]), s.mv(l.wv, in[2])
	kn := hp.ConvKernel
	cv := &compute.Ink2KVConv{
		KW: l.kconvW, VW: l.vconvW, KDyn: kDyn, VDyn: vDyn, Kn: kn,
		KHist: s.hist[slot], VHist: at(s.hist[slot], kvDim*(kn-1)),
		KNorm: inkling2ConvNorm(kn, kDyn), VNorm: inkling2ConvNorm(kn, vDyn),
	}
	r := s.mv(l.rel, in[0])
	sinks := s.mv(l.sink, in[0])

	p := compute.Ink2AttnParams{
		Heads: hp.Heads, KVHeads: hp.KVHeads, HeadDim: hp.HeadDim, RelLen: hp.RelLength,
		QKClip: hp.QKClip, RelClip: hp.RelClip, Eps: hp.NormEps, Global: !local,
		Alpha: hp.LogScaleAlpha, Floor: hp.LogScaleFloor, Scale: float32(1 / math.Sqrt(float64(hp.HeadDim))),
	}
	meta := at(s.in, h)
	cap := s.caps[slot]
	if s.err == nil {
		s.fail(s.dev.Ink2QKVPost(q, r, k, v, l.qclip, l.relClip, s.kc[slot], s.vc[slot], meta, p, cap, cv))
	}
	extent, window := hp.RelExtent, 0
	if local {
		extent, window = hp.SlidingWindow, hp.SlidingWindow
	}
	attn := s.tmp(hp.Heads * hp.HeadDim)
	if s.err == nil {
		s.fail(s.dev.Ink2Attend(q, r, sinks, l.relBias, s.kc[slot], s.vc[slot], s.scores, attn, meta, p, extent, window, cap))
	}
	gate := s.mv(l.gate, in[0])
	o := s.mvGated(l.wo, attn, gate, hp.Heads*hp.HeadDim)
	if ip%hp.AttnResBlock == 0 {
		*blocks = append(*blocks, inpSA)
		cur = s.conv(slot, o, l.attnPostConv, ink2Ref{}, ink2Ref{}, h, 0, 2*kvDim)
	} else {
		cur = s.conv(slot, o, l.attnPostConv, ink2Ref{}, inpSA, h, 0, 2*kvDim)
	}
	ffnInp := cur

	var x ink2Ref
	if msgs != nil {
		x = s.mix(with(*blocks, cur), l.ffnResQ, 1)
		x = s.control([]ink2Ref{x}, msgs[3:4], c, []int{3}, []ink2Ref{l.ffnNorm})
	} else {
		x = s.mixNorm(with(*blocks, cur), l.ffnResQ, 1, []ink2Ref{l.ffnNorm})
	}
	return s.moeConv(l, x, slot, ffnInp, 2*kvDim+h)
}

// expertMatVec runs k routed slots through e into y (rows per slot), with the
// per-expert scale applied. situ fuses the SITU activation into the input staging.
func (s *inkling2Resident) expertMatVec(e *ink2Experts, ids, x ink2Ref, xStride int, y ink2Ref, rows int, situ *compute.Ink2Situ) {
	hp := s.hp
	if s.err != nil {
		return
	}
	ok, err := s.dev.Ink2ExpertMatVec(e.stack, rows, ids, x, xStride, y, hp.ExpertsUsed, e.scale, situ)
	if err != nil || ok {
		s.fail(err)
		return
	}
	if situ != nil {
		act := s.tmp(hp.ExpertsUsed * e.padK)
		s.fail(s.dev.Ink2Situ(x, act, hp.ExpertsUsed, situ.Width, e.padK, situ.Beta, situ.LinearBeta))
		x, xStride = act, e.padK
	}
	got, err := s.dev.Ink2ReadInts(ids, hp.ExpertsUsed)
	if err != nil {
		s.fail(err)
		return
	}
	for t, id := range got {
		if s.err == nil {
			s.fail(s.dev.Ink2MatVec(e.views[id], at(x, t*xStride), at(y, t*rows)))
		}
	}
	if e.scale.T.Buf() != nil && s.err == nil {
		s.fail(s.dev.Ink2ExpertScale(y, ids, e.scale, hp.ExpertsUsed, rows))
	}
}

func (s *inkling2Resident) moe(l *ink2Layer, x ink2Ref) ink2MoE {
	hp := s.hp
	k, width := hp.ExpertsUsed, hp.ExpertWidth
	logits := s.mv(l.router, x)
	ids, wts := s.tmp(k), s.tmp(k)
	if s.err == nil {
		s.fail(s.dev.Ink2Route(logits, l.probsBias, ids, wts, l.router.rows, k, hp.ExpertWeightsScale))
	}
	gu := s.tmp(k * 2 * width)
	s.expertMatVec(&l.gateUp, ids, x, 0, gu, 2*width, nil)
	// SITU runs once per routed value here; fused into the down GEMV staging it
	// would be recomputed by every block in float64
	padK := l.down.padK
	act := s.tmp(k * padK)
	if s.err == nil {
		s.fail(s.dev.Ink2Situ(gu, act, k, width, padK, float64(hp.SituBeta), float64(hp.SituLinearBeta)))
	}
	down := s.tmp(k * hp.Hidden)
	s.expertMatVec(&l.down, ids, act, padK, down, hp.Hidden, nil)
	return ink2MoE{down: down, wts: wts}
}

// ink2MoE is a MoE output before its combine: the routed slots' down rows and weights.
type ink2MoE struct{ down, wts ink2Ref }

// moeConv runs the FFN's MoE and its post conv with the combine folded into the conv.
func (s *inkling2Resident) moeConv(l *ink2Layer, x ink2Ref, slot int, resid ink2Ref, offset int) ink2Ref {
	hp := s.hp
	e := s.moe(l, x)
	kn := hp.ConvKernel
	out := s.tmp(hp.Hidden)
	if s.err == nil {
		s.fail(s.dev.Ink2CombineConv(e.down, e.wts, hp.ExpertsUsed, l.ffnPostConv, at(s.hist[slot], offset*(kn-1)), out, resid,
			hp.Hidden, kn, float32(1/math.Sqrt(float64(kn+1)))))
	}
	return out
}

// forward enqueues one token's forward from the uploaded input row through the
// argmax; it issues no host fence unless host routing is required.
func (s *inkling2Resident) forward() {
	hp := s.hp
	s.off = 0
	clear(s.q8)
	cur := s.rms(s.in, s.tokNorm, hp.Hidden, hp.NormEps)
	var blocks []ink2Ref
	for il := 0; il < hp.CoreStart; il++ {
		cur = s.layer(il, il, cur, &blocks, nil, nil)
	}
	prelude := cur
	preludeBlocks := append([]ink2Ref(nil), blocks...)
	core := hp.CoreEnd - hp.CoreStart
	for pass := 0; pass < hp.Passes; pass++ {
		var bank []ink2Ref
		var bankOut ink2Ref
		if pass > 0 {
			bank = append(bank, blocks[len(preludeBlocks)+1:]...)
			bankOut = cur
			cur = prelude
			blocks = append([]ink2Ref(nil), preludeBlocks...)
		}
		for ip := hp.CoreStart; ip < hp.CoreEnd; ip++ {
			if pass == 0 {
				cur = s.layer(ip, ip, cur, &blocks, nil, nil)
				continue
			}
			slot := hp.Layers + (pass-1)*core + (ip - hp.CoreStart)
			c := &s.ctrls[ip-hp.CoreStart]
			m := s.mix(with(bank, bankOut), c.msgQ, 4)
			msgs := []ink2Ref{m, at(m, hp.Hidden), at(m, 2*hp.Hidden), at(m, 3*hp.Hidden)}
			cur = s.layer(slot, ip, cur, &blocks, msgs, c)
		}
	}
	for il := hp.CoreEnd; il < hp.Layers; il++ {
		cur = s.layer(il, il, cur, &blocks, nil, nil)
	}
	x := s.mixNorm(with(blocks, cur), s.outResQ, 1, []ink2Ref{s.outNorm})
	logits := s.mv(s.output, x)
	if s.err == nil {
		s.fail(s.dev.Ink2SoftcapArgmax(logits, hp.VocabUnpadded, hp.FinalLogitSoftcap, s.argIdx))
	}
}

// step consumes one token at the next position and returns the greedy next token.
func (s *inkling2Resident) step(token int) (int, error) {
	hp := s.hp
	if s.pos >= inkling2ResidentCtx {
		return 0, fmt.Errorf("inkling2 resident: position %d exceeds the %d-token global cache", s.pos, inkling2ResidentCtx)
	}
	row := make([]float32, hp.Hidden+4)
	s.w.TokEmbd.Row(token, row[:hp.Hidden])
	row[hp.Hidden] = math.Float32frombits(uint32(int32(s.pos)))
	s.fail(s.dev.Ink2Write(s.in, row))
	switch {
	case s.graphReady:
		s.fail(s.dev.Ink2GraphLaunch())
	case s.graph && s.pos > 0:
		// capture after one eager token has grown every lazily sized GEMV scratch
		if err := s.dev.Ink2GraphBegin(); err != nil {
			s.graph = false
			s.forward()
			break
		}
		s.forward()
		if err := s.dev.Ink2GraphEnd(); err != nil || s.err != nil {
			s.fail(err)
			s.graph = false
			break
		}
		s.graphReady = true
		s.fail(s.dev.Ink2GraphLaunch())
	default:
		s.forward()
	}
	if s.err != nil {
		return 0, s.err
	}
	got, err := s.dev.Ink2ReadInts(s.argIdx, 1)
	if err != nil {
		return 0, err
	}
	s.pos++
	if r, ok := s.be.(interface{ Recycle() }); ok {
		// token boundary for the backend's profile and transient bookkeeping
		r.Recycle()
	}
	return int(got[0]), nil
}

// inkling2ResidentGreedy is Inkling2LoopGreedyTimed on the resident device forward.
func inkling2ResidentGreedy(m *Inkling2LoopModel, be compute.Backend, prompt []int, n int) ([]int, Inkling2LoopGreedyStats, error) {
	var st Inkling2LoopGreedyStats
	s, err := newInkling2Resident(m, be)
	if err != nil {
		return nil, st, err
	}
	defer s.close()
	// a CUDA stream capture is bound to the OS thread that opened it
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	start := time.Now()
	next := 0
	for _, t := range prompt {
		if next, err = s.step(t); err != nil {
			return nil, st, err
		}
	}
	st.Prefill = time.Since(start)
	out := make([]int, 0, n)
	start = time.Now()
	for len(out) < n {
		out = append(out, next)
		if len(out) == n {
			break
		}
		if next, err = s.step(next); err != nil {
			return nil, st, err
		}
		st.DecodeSteps++
	}
	st.Decode = time.Since(start)
	return out, st, nil
}
