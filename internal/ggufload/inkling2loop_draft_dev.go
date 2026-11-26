package ggufload

import (
	"fmt"
	"math"
	"os"
	"strconv"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/model"
)

// inkling2DraftSlot is the device graph slot of the draft block forward.
const inkling2DraftSlot = 3

type ink2DLayer struct {
	inNorm, postNorm, dfNorm, qNorm, kNorm, kNormCtx ink2Ref
	wq, wk, wv, wo, wkCtx, wvCtx, gate, up, down     ink2Mat
	kc, vc                                           ink2Ref // context rings [inkling2ResidentCtx][kvDim]
	alpha                                            []float32
}

// ink2DeviceDraft is the DSpark draft resident beside a target session. Its context
// K/V are written by every target forward (prefill chunks and verifies) from the
// tapped target features, one ring row per position, so rows of rejected drafts sit
// past the committed position and are overwritten. Each proposal runs the dense
// block forward over the anchor and Block-1 masks and drafts the base-logit argmax
// of every mask row (the Markov head is not run).
type ink2DeviceDraft struct {
	s      *inkling2Resident
	dd     compute.Inkling2DraftDevice
	hp     model.Inkling2DSparkHParams
	w      *model.Inkling2DSparkWeights
	visits []int
	layers []ink2DLayer
	norm   ink2Ref

	in, pos, idx, logits, scr ink2Ref
	scrFloats, off            int
	warm, ready               bool
	rows                      []float32

	// Markov head: shortlist M of base logits per row (0 = base argmax only), steps
	// drafted per proposal
	shortlist, steps               int
	minMargin                      float32
	w1, anchor, frnn               ink2Ref
	rnnProj, hiddenOut             ink2Mat
	rnnBias, hiddenOutBias, mkNorm ink2Ref
	mkAlpha                        []float32
	state, cand, candV, candI      ink2Ref
}

// q8Mat uploads a host matrix requantized to Q8_0 (32-value blocks, absmax scale),
// the dtype the batched matmul serves fastest.
func (s *inkling2Resident) q8Mat(m model.Inkling2LoopMatrix) ink2Mat {
	if s.err != nil {
		return ink2Mat{}
	}
	bf, ok := m.(*inkling2BF16Matrix)
	if !ok || bf.cols%32 != 0 {
		s.fail(fmt.Errorf("inkling2 draft: matrix %T [%d %d] is not a 32-aligned BF16 matrix", m, m.Rows(), m.Cols()))
		return ink2Mat{}
	}
	rows, cols := bf.rows, bf.cols
	codes := make([]int8, rows*cols)
	scales := make([]float32, rows*cols/32)
	var blk [32]float32
	for b := range scales {
		var amax float32
		for i := range blk {
			o := 2 * (b*32 + i)
			blk[i] = bf16ToF32(uint16(bf.raw[o]) | uint16(bf.raw[o+1])<<8)
			amax = max(amax, float32(math.Abs(float64(blk[i]))))
		}
		d := amax / 127
		scales[b] = d
		if d == 0 {
			continue
		}
		for i, v := range blk {
			codes[b*32+i] = int8(math.Round(float64(v / d)))
		}
	}
	t := s.be.Upload(compute.NewQ8(s.be, []int{rows, cols}, codes, scales, 32), compute.Q8_0)
	s.owned = append(s.owned, t)
	return ink2Mat{t: t, rows: rows, cols: cols}
}

func inkling2SoftmaxF32(x []float32) []float32 {
	mx := math.Inf(-1)
	for _, v := range x {
		mx = math.Max(mx, float64(v))
	}
	var sum float64
	for _, v := range x {
		sum += math.Exp(float64(v) - mx)
	}
	out := make([]float32, len(x))
	for i, v := range x {
		out[i] = float32(math.Exp(float64(v)-mx) / sum)
	}
	return out
}

// attachDraft binds dm on the session's device; every later target forward also
// writes the draft context.
func (s *inkling2Resident) attachDraft(dm *Inkling2DSparkModel) (*ink2DeviceDraft, error) {
	dd, ok := s.be.(compute.Inkling2DraftDevice)
	if !ok {
		return nil, fmt.Errorf("inkling2 draft: backend %s has no draft ops", s.be.Name())
	}
	hp, w := dm.HP, dm.Weights
	if hp.Taps > 4 || hp.Hidden != s.hp.Hidden || hp.HeadDim > 256 {
		return nil, fmt.Errorf("inkling2 draft: unsupported geometry %+v", hp)
	}
	d := &ink2DeviceDraft{s: s, dd: dd, hp: hp, w: w, visits: dm.TargetVisits, steps: hp.Block - 1, minMargin: inkling2SpecMinMargin}
	H := hp.Hidden
	kvDim := hp.KVHeads * hp.HeadDim
	for l := range w.Layers {
		lw := &w.Layers[l]
		d.layers = append(d.layers, ink2DLayer{
			inNorm: s.upload(lw.InputNorm), postNorm: s.upload(lw.PostAttnNorm), dfNorm: s.upload(lw.DFlareNorm),
			qNorm: s.upload(lw.QNorm), kNorm: s.upload(lw.KNorm), kNormCtx: s.upload(lw.KNormCtx),
			wq: s.q8Mat(lw.Wq), wk: s.q8Mat(lw.Wk), wv: s.q8Mat(lw.Wv), wo: s.q8Mat(lw.Wo),
			wkCtx: s.q8Mat(lw.WkCtx), wvCtx: s.q8Mat(lw.WvCtx),
			gate: s.q8Mat(lw.Gate), up: s.q8Mat(lw.Up), down: s.q8Mat(lw.Down),
			kc: s.alloc(s.ctx * kvDim), vc: s.alloc(s.ctx * kvDim),
			alpha: inkling2SoftmaxF32(w.FuseWeight[l]),
		})
	}
	d.norm = s.upload(w.Norm)
	if d.shortlist = inkling2DraftShortlist(); d.shortlist > 0 {
		if s.output.t.Dtype != compute.Q5_K {
			d.shortlist = 0
		}
	}
	if d.shortlist > 0 {
		r := hp.MarkovRank
		bf := w.MarkovW1.(inkling2BF16Embedding).m
		packed := make([]float32, len(bf.raw)/4)
		for i := range packed {
			o := 4 * i
			packed[i] = math.Float32frombits(uint32(bf.raw[o]) | uint32(bf.raw[o+1])<<8 | uint32(bf.raw[o+2])<<16 | uint32(bf.raw[o+3])<<24)
		}
		d.w1 = s.upload(packed)
		d.anchor = s.alloc(4)
		d.frnn = s.alloc(s.ctx * H)
		d.rnnProj, d.hiddenOut = s.q8Mat(w.RnnProj), s.q8Mat(w.HiddenOut)
		d.rnnBias, d.hiddenOutBias, d.mkNorm = s.upload(w.RnnBias), s.upload(w.HiddenOutBias), s.upload(w.MarkovNorm)
		d.mkAlpha = inkling2SoftmaxF32(w.FuseWeight[hp.Layers])
		d.state = s.alloc(r)
		d.cand = s.alloc((hp.Block - 1) * d.shortlist)
		d.candV, d.candI = s.alloc((hp.Block-1)*64*d.shortlist), s.alloc((hp.Block-1)*64*d.shortlist)
	}
	B := hp.Block
	qDim := hp.Heads * hp.HeadDim
	d.in = s.alloc(B * H)
	d.pos, d.idx = s.alloc(4), s.alloc(4*B)
	d.logits = s.alloc((B - 1) * s.output.rows)
	d.scrFloats = 4*B*H + 2*B*qDim + 2*B*kvDim + 3*B*hp.Intermediate + 3*hp.MarkovRank + 2*(2*hp.MarkovRank+2*H) + B*H + 64*24 + 6*B*(hp.Intermediate/4+hp.Intermediate/32+64)
	d.scr = s.alloc(d.scrFloats)
	d.rows = make([]float32, B*H)
	for i := 1; i < B; i++ {
		copy(d.rows[i*H:(i+1)*H], w.MaskEmbed)
	}
	if s.err != nil {
		return nil, s.err
	}
	s.draft = d
	return d, nil
}

func (d *ink2DeviceDraft) tapVisits() []int { return d.visits }

func (d *ink2DeviceDraft) dropCaptures() { d.warm, d.ready = false, false }

func (d *ink2DeviceDraft) tmp(n int) ink2Ref {
	n = (n + 63) &^ 63
	if d.off+n > d.scrFloats {
		d.s.fail(fmt.Errorf("inkling2 draft: scratch exhausted at %d floats", d.off+n))
		return d.scr
	}
	r := at(d.scr, d.off)
	d.off += n
	return r
}

// commitTaps writes the context K/V of the T rows of a target forward from its
// tapped features (taps in TargetVisits order; the last is the final-norm output)
// at positions *s.verifyPos + t.
func (d *ink2DeviceDraft) commitTaps(p *ink2Prefill, taps []ink2Ref) {
	s, hp := d.s, d.hp
	if len(taps) != hp.Taps {
		s.fail(fmt.Errorf("inkling2 draft: %d target taps, want %d", len(taps), hp.Taps))
		return
	}
	kvDim := hp.KVHeads * hp.HeadDim
	for l := range d.layers {
		L := &d.layers[l]
		ctx := p.tmp(p.T * hp.Hidden)
		if s.err == nil {
			s.fail(d.dd.Ink2DFuse(taps, L.alpha, L.dfNorm, ctx, p.T, hp.Hidden, hp.NormEps))
		}
		k, v := p.tmp(p.T*kvDim), p.tmp(p.T*kvDim)
		if s.err == nil {
			s.fail(p.pd.Ink2PMatMul(L.wkCtx.t, ctx, k, p.T))
		}
		if s.err == nil {
			s.fail(p.pd.Ink2PMatMul(L.wvCtx.t, ctx, v, p.T))
		}
		if s.err == nil {
			s.fail(d.dd.Ink2DQK(k, L.kNormCtx, L.kc, s.verifyPos, p.T, hp.KVHeads, hp.HeadDim, d.s.ctx, true, hp.NormEps, hp.RopeTheta))
		}
		if s.err == nil {
			s.fail(d.dd.Ink2DRingStore(v, L.vc, s.verifyPos, p.T, kvDim, d.s.ctx))
		}
	}
	if d.shortlist > 0 && s.err == nil {
		f := p.tmp(p.T * hp.Hidden)
		s.fail(d.dd.Ink2DFuse(taps, d.mkAlpha, d.mkNorm, f, p.T, hp.Hidden, hp.NormEps))
		if s.err == nil {
			s.fail(d.dd.Ink2DRingStore(f, d.frnn, s.verifyPos, p.T, hp.Hidden, d.s.ctx))
		}
	}
}

// forward enqueues the block forward from d.in at anchor position *d.pos and
// leaves draft i's token at d.idx[4i].
func (d *ink2DeviceDraft) forward() {
	s, hp := d.s, d.hp
	pd := s.be.(compute.Inkling2PrefillDevice)
	d.off = 0
	B, H := hp.Block, hp.Hidden
	qDim, kvDim := hp.Heads*hp.HeadDim, hp.KVHeads*hp.HeadDim
	run := func(err error) {
		if s.err == nil {
			s.fail(err)
		}
	}
	vdev := s.be.(compute.Inkling2VerifyDevice)
	// a layer's temporaries are written once, so the weights that read one activation
	// share its Q8_0 codes, keyed by its arena offset
	q8 := map[int]ink2Ref{}
	mm := func(m ink2Mat, x ink2Ref, rows int) ink2Ref {
		y := d.tmp(rows * m.rows)
		if s.err == nil {
			codes := func(x ink2Ref, n int) (ink2Ref, bool) {
				if q, ok := q8[x.Off]; ok {
					return q, false
				}
				q := d.tmp(rows * (n/4 + n/32))
				q8[x.Off] = q
				return q, true
			}
			if ok, err := inkling2RowsMatMul(vdev, m, x, y, rows, codes); err != nil || ok {
				run(err)
				return y
			}
			run(pd.Ink2PMatMul(m.t, x, y, rows))
		}
		return y
	}
	// mmMany is mm for every weight in ms on the one activation x, the Q8_0 ones in a
	// single launch
	mmMany := func(x ink2Ref, rows int, ms ...ink2Mat) []ink2Ref {
		ys := make([]ink2Ref, len(ms))
		done := make([]bool, len(ms))
		if s.err == nil {
			xs := make([]ink2Ref, len(ms))
			for i := range xs {
				xs[i] = x
			}
			var err error
			done, err = inkling2SegsMatMul(vdev, ms, xs, ys, rows, d.tmp)
			run(err)
		}
		for i, m := range ms {
			if !done[i] {
				ys[i] = mm(m, x, rows)
			}
		}
		return ys
	}
	res := d.in
	for l := range d.layers {
		// layer temporaries are dead once res is updated, so every layer reuses them
		d.off = 0
		clear(q8)
		L := &d.layers[l]
		x := d.tmp(B * H)
		run(s.dev.Ink2RMS(res, L.inNorm, x, B, H, hp.NormEps))
		qkv := mmMany(x, B, L.wq, L.wk, L.wv)
		q, k, v := qkv[0], qkv[1], qkv[2]
		run(d.dd.Ink2DQK(q, L.qNorm, q, d.pos, B, hp.Heads, hp.HeadDim, 0, false, hp.NormEps, hp.RopeTheta))
		run(d.dd.Ink2DQK(k, L.kNorm, k, d.pos, B, hp.KVHeads, hp.HeadDim, 0, false, hp.NormEps, hp.RopeTheta))
		window := 0
		if hp.LocalLayer[l] {
			window = hp.SlidingWindow
		}
		o := d.tmp(B * qDim)
		run(d.dd.Ink2DAttend(q, k, v, L.kc, L.vc, o, d.pos, B, hp.Heads, hp.KVHeads, hp.HeadDim, window, d.s.ctx))
		run(s.dev.Ink2Add(res, mm(L.wo, o, B), res, B*H))
		h := d.tmp(B * H)
		run(s.dev.Ink2RMS(res, L.postNorm, h, B, H, hp.NormEps))
		gu := mmMany(h, B, L.gate, L.up)
		g, u := gu[0], gu[1]
		run(d.dd.Ink2DSiluMul(g, u, g, B*hp.Intermediate))
		run(s.dev.Ink2Add(res, mm(L.down, g, B), res, B*H))
		_ = kvDim
	}
	hid := d.tmp(B * H)
	run(s.dev.Ink2RMS(res, d.norm, hid, B, H, hp.NormEps))
	rows := s.output.rows
	vd := s.be.(compute.Inkling2VerifyDevice)
	if ok, err := vd.Ink2VRows(s.output.t, at(hid, H), d.logits, d.steps); err != nil || !ok {
		run(err)
		run(pd.Ink2PMatMul(s.output.t, at(hid, H), d.logits, d.steps))
	}
	if d.shortlist == 0 {
		for i := 0; i < d.steps; i++ {
			run(s.dev.Ink2SoftcapArgmax(at(d.logits, i*rows), hp.VocabUnpadded, 0, at(d.idx, 4*i)))
		}
		return
	}
	// the Markov head: the shortlist of every row in one pass, then the sequential
	// recurrent steps, each scoring only its row's shortlist
	M, r := d.shortlist, hp.MarkovRank
	run(d.dd.Ink2DTopM(d.logits, d.steps, rows, hp.VocabUnpadded, M, d.candV, d.candI, d.cand))
	in := d.tmp(2*r + 2*H)
	raw, latent := d.tmp(3*r), d.tmp(r)
	for i := 0; i < d.steps; i++ {
		run(d.dd.Ink2DMarkovIn(d.state, d.w1, d.anchor, d.idx, i, at(hid, (i+1)*H), d.frnn, d.pos, d.s.ctx, r, H, in))
		run(pd.Ink2PMatMul(d.rnnProj.t, in, raw, 1))
		run(d.dd.Ink2DMarkovGate(raw, d.rnnBias, d.state, latent, r, i))
		z := d.tmp(H)
		run(pd.Ink2PMatMul(d.hiddenOut.t, latent, z, 1))
		run(d.dd.Ink2DMarkovPick(s.output.t, at(d.logits, i*rows), at(d.cand, i*M), M, z, d.hiddenOutBias, d.idx, i))
	}
}

// inkling2SpecMinMargin is the lead over the runner-up, in logits, a Markov draft step
// needs to be proposed. Below it the step is accepted too rarely to pay for its verify
// row, and a rejected step also wastes every row drafted after it.
const inkling2SpecMinMargin = 2.5

// propose drafts up to k tokens after anchor, the token at position pos whose
// context (positions < pos) is already committed. With the Markov head the proposal
// stops before the first step whose margin is under inkling2SpecMinMargin, so it can
// be empty.
func (d *ink2DeviceDraft) propose(anchor, pos, k int) ([]int, error) {
	s, hp := d.s, d.hp
	H := hp.Hidden
	emb := make([]float32, H)
	s.w.TokEmbd.Row(anchor, emb)
	var ss float64
	for _, v := range emb {
		ss += float64(v) * float64(v)
	}
	sc := float32(1 / math.Sqrt(ss/float64(H)+float64(hp.NormEps)))
	for i, v := range emb {
		d.rows[i] = v * sc * s.w.TokNorm[i]
	}
	s.fail(s.dev.Ink2Write(d.in, d.rows))
	s.fail(s.dev.Ink2Write(d.pos, []float32{intBits(pos)}))
	if d.shortlist > 0 {
		s.fail(s.dev.Ink2Write(d.anchor, []float32{intBits(anchor)}))
	}
	vd := s.be.(compute.Inkling2VerifyDevice)
	switch {
	case d.ready:
		s.fail(vd.Ink2GraphLaunchSlot(inkling2DraftSlot))
	case d.warm && s.graph:
		if err := s.dev.Ink2GraphBegin(); err != nil {
			s.fail(err)
			break
		}
		d.forward()
		s.fail(vd.Ink2GraphEndSlot(inkling2DraftSlot))
		d.ready = s.err == nil
		s.fail(vd.Ink2GraphLaunchSlot(inkling2DraftSlot))
	default:
		d.forward()
		d.warm = true
	}
	if s.err != nil {
		return nil, s.err
	}
	k = min(k, d.steps)
	got, err := s.dev.Ink2ReadInts(d.idx, 4*k)
	if err != nil {
		return nil, err
	}
	out := make([]int, 0, k)
	for i := 0; i < k; i++ {
		if d.shortlist > 0 && math.Float32frombits(uint32(got[4*i+1])) < d.minMargin {
			break
		}
		out = append(out, int(got[4*i]))
	}
	return out, nil
}

// inkling2DSparkDrafter adapts the device draft to the speculative loop.
type inkling2DSparkDrafter struct{ d *ink2DeviceDraft }

func (a inkling2DSparkDrafter) Draft(history []int, k int) []int {
	props, err := a.d.propose(history[len(history)-1], a.d.s.pos, k)
	if err != nil {
		a.d.s.fail(err)
		return nil
	}
	return props
}

func (inkling2DSparkDrafter) Observe(*inkling2Resident, []int, int) error { return nil }

// inkling2DraftShortlist is the Markov shortlist width from
// FAK_INKLING2_DSPARK_SHORTLIST (default 32; 0 or negative drafts the base argmax).
func inkling2DraftShortlist() int {
	v := os.Getenv("FAK_INKLING2_DSPARK_SHORTLIST")
	if v == "" {
		return 32
	}
	n, err := strconv.Atoi(v)
	if err != nil || n <= 0 {
		return 0
	}
	return min(n, 64)
}
