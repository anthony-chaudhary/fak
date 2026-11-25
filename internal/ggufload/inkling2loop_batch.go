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

// inkling2BatchCtx is the default global-slot KV capacity of each sequence of a
// batched decode; local slots keep their sliding-window ring.
const inkling2BatchCtx = 512

// inkling2Batch decodes C sequences in lockstep through one resident weight set,
// one row per sequence. Dense, mix, controller, routing and expert ops run the
// row-batched prefill kernels over the C rows, so each weight is streamed once per
// step for every sequence, and the looped core reuses that read on every pass. The
// ops that touch sequence state (k/v conv and cache append, attention, post convs,
// argmax) run against per-sequence rows of strided arenas: each slot owns one
// [C][cap][kvDim] k and v arena and one [C][histStride] conv history, all sequences
// sharing the slot's cap, so a sequence stride replaces any pointer table. With seq
// set those ops are one launch for all C rows; otherwise they loop per row.
type inkling2Batch struct {
	s      *inkling2Resident
	p      ink2Prefill
	C, ctx int
	pos    []int
	seq    compute.Inkling2BatchDevice

	kc, vc, hist []ink2Ref
	caps         []int
	scores       ink2Ref
	histStride   int
	scoreStride  int

	in, meta, argIdx ink2Ref

	graph, graphReady, warm bool
}

// inkling2BatchGraph reports whether the batched step is captured and replayed as a
// device graph (FAK_INKLING2_BATCH_GRAPH=0 runs every step eagerly).
func inkling2BatchGraph() bool { return os.Getenv("FAK_INKLING2_BATCH_GRAPH") != "0" }

// inkling2BatchSeqOps returns the backend's sequence-batched state ops, or nil to loop
// them per sequence (no such ops, or FAK_INKLING2_BATCH_SEQOPS=0 for an A/B).
func inkling2BatchSeqOps(be any, lookup func(string) (string, bool)) compute.Inkling2BatchDevice {
	if v, ok := lookup("FAK_INKLING2_BATCH_SEQOPS"); ok && v == "0" {
		return nil
	}
	d, _ := be.(compute.Inkling2BatchDevice)
	return d
}

func newInkling2Batch(m *Inkling2LoopModel, be compute.Backend, C, ctx int) (*inkling2Batch, error) {
	if C < 1 {
		return nil, fmt.Errorf("inkling2 batch: %d sequences", C)
	}
	pd, ok := be.(compute.Inkling2PrefillDevice)
	if !ok {
		return nil, fmt.Errorf("inkling2 batch: backend %s has no row-batched ops", be.Name())
	}
	if ctx < 1 || ctx > inkling2ResidentCtx {
		return nil, fmt.Errorf("inkling2 batch: context %d outside 1..%d", ctx, inkling2ResidentCtx)
	}
	// the batch forward draws scratch from its own arenas, never the per-token one, and
	// keeps every sequence's state in its own strided arenas below
	s, err := newInkling2ResidentSized(m, be, 0, 64)
	if err != nil {
		return nil, err
	}
	if s.hostRouting {
		s.close()
		return nil, fmt.Errorf("inkling2 batch: an expert dtype has no indexed kernel")
	}
	s.graph = false
	hp := s.hp
	b := &inkling2Batch{s: s, C: C, ctx: ctx, pos: make([]int, C), seq: inkling2BatchSeqOps(be, os.LookupEnv),
		graph: inkling2BatchGraph() && inkling2DefaultOn(be, "FAK_INKLING2_GRAPH", os.LookupEnv)}
	b.p = ink2Prefill{s: s, pd: pd, grouped: inkling2GroupedExperts(), T: C}
	keep, scr := s.prefillArenas(C)
	scr += C*s.output.rows + 64
	b.p.keepFloats, b.p.scrFloats = keep, scr
	b.p.keep, b.p.scr = s.alloc(keep), s.alloc(scr)
	// each argmax takes a 4-int slot: the index, then its 8-byte reduction key
	b.in, b.meta, b.argIdx = s.alloc(C*hp.Hidden), s.alloc(C), s.alloc(4*C)

	b.caps = inkling2BatchCaps(hp, s.caps, ctx)
	slots := len(b.caps)
	kvDim := hp.KVHeads * hp.HeadDim
	b.histStride = (2*kvDim + 2*hp.Hidden) * (hp.ConvKernel - 1)
	b.kc, b.vc, b.hist = make([]ink2Ref, slots), make([]ink2Ref, slots), make([]ink2Ref, slots)
	maxCap := 0
	for slot, cap := range b.caps {
		maxCap = max(maxCap, cap)
		b.kc[slot], b.vc[slot] = s.alloc(C*cap*kvDim), s.alloc(C*cap*kvDim)
		b.hist[slot] = s.alloc(C * b.histStride)
	}
	// float64 scores rows per head per sequence, so sequences never serialize on them
	b.scoreStride = 2 * hp.Heads * maxCap
	b.scores = s.alloc(C * b.scoreStride)
	if s.err != nil {
		s.close()
		return nil, s.err
	}
	return b, nil
}

func (b *inkling2Batch) close() { b.s.close() }

// inkling2BatchCaps is the cache capacity of every slot: ctx for a global slot, the
// resident's ring size for a local one.
func inkling2BatchCaps(hp model.Inkling2LoopHParams, local []int, ctx int) []int {
	caps := make([]int, len(local))
	for slot := range caps {
		caps[slot] = ctx
		if hp.LocalLayer[inkling2SlotLayer(hp, slot)] {
			// a lockstep batch never passes position ctx-1, so a ring past ctx is never addressed
			caps[slot] = min(local[slot], ctx)
		}
	}
	return caps
}

// StateBytes is the device memory of every sequence's caches and conv histories.
func (b *inkling2Batch) StateBytes() int64 {
	hp := b.s.hp
	kvDim := hp.KVHeads * hp.HeadDim
	var n int64
	for _, c := range b.caps {
		n += int64(2*c*kvDim + b.histStride)
	}
	return 4 * n * int64(b.C)
}

// inkling2SeqRow is sequence i's row of a strided operand; an absent operand stays
// absent, as the batched kernels keep a null pointer null.
func inkling2SeqRow(r ink2Ref, stride, i int) ink2Ref {
	if r.T.Buf() == nil {
		return ink2Ref{}
	}
	return at(r, i*stride)
}

func (b *inkling2Batch) layer(slot, ip int, cur ink2Ref, blocks *[]ink2Ref, msgs []ink2Ref, c *ink2Ctrl) ink2Ref {
	p := &b.p
	s := b.s
	hp := s.hp
	l := &s.layers[ip]
	local := hp.LocalLayer[ip]
	h := hp.Hidden
	kvDim := hp.KVHeads * hp.HeadDim
	kn := hp.ConvKernel
	inpSA := cur

	in := [3]ink2Ref{cur, cur, cur}
	if l.resQRows > 0 {
		if local {
			m := p.mix(with(*blocks, cur), l.resQ, 1)
			in = [3]ink2Ref{m[0], m[0], m[0]}
		} else {
			m := p.mix(with(*blocks, cur), l.resQ, 3)
			in = [3]ink2Ref{m[0], m[1], m[2]}
		}
	}
	if msgs != nil {
		for j := 0; j < 3; j++ {
			in[j] = p.control(in[j], msgs[j], c, j)
		}
	}
	switch {
	case local && msgs != nil:
		for j := 0; j < 3; j++ {
			in[j] = p.rms(in[j], l.attnNorm, h, hp.NormEps)
		}
	case local:
		n0 := p.rms(in[0], l.attnNorm, h, hp.NormEps)
		in = [3]ink2Ref{n0, n0, n0}
	default:
		in[0] = p.rms(in[0], l.qNorm, h, hp.NormEps)
		in[1] = p.rms(in[1], l.kNorm, h, hp.NormEps)
		in[2] = p.rms(in[2], l.vNorm, h, hp.NormEps)
	}

	q := p.mm(l.wq, in[0])
	var kDyn, vDyn ink2Ref
	if !local {
		kDyn = p.mm(l.kconv, in[1])
		vDyn = p.mm(l.vconv, in[2])
	}
	k, v := p.mm(l.wk, in[1]), p.mm(l.wv, in[2])
	r := p.mm(l.rel, in[0])
	sinks := p.mm(l.sink, in[0])

	ap := compute.Ink2AttnParams{
		Heads: hp.Heads, KVHeads: hp.KVHeads, HeadDim: hp.HeadDim, RelLen: hp.RelLength,
		QKClip: hp.QKClip, RelClip: hp.RelClip, Eps: hp.NormEps, Global: !local,
		Alpha: hp.LogScaleAlpha, Floor: hp.LogScaleFloor, Scale: float32(1 / math.Sqrt(float64(hp.HeadDim))),
	}
	extent, window := hp.RelExtent, 0
	if local {
		extent, window = hp.SlidingWindow, hp.SlidingWindow
	}
	qd := hp.Heads * hp.HeadDim
	attn := p.tmp(b.C * qd)
	cap := b.caps[slot]
	cv := compute.Ink2KVConv{KW: l.kconvW, VW: l.vconvW, Kn: kn, KHist: b.hist[slot], VHist: at(b.hist[slot], kvDim*(kn-1))}
	if !local {
		cv.KDyn, cv.VDyn = kDyn, vDyn
	}
	cv.KNorm, cv.VNorm = inkling2ConvNorm(kn, cv.KDyn), inkling2ConvNorm(kn, cv.VDyn)
	st := compute.Ink2SeqStrides{Q: l.wq.rows, R: l.rel.rows, K: l.wk.rows, V: l.wv.rows, KDyn: l.kconv.rows, VDyn: l.vconv.rows,
		Sinks: l.sink.rows, KCache: cap * kvDim, VCache: cap * kvDim, Hist: b.histStride, Scores: b.scoreStride, Out: qd, Meta: 1}
	if b.seq != nil {
		if s.err == nil {
			s.fail(b.seq.Ink2BQKVPost(q, r, k, v, l.qclip, l.relClip, b.kc[slot], b.vc[slot], b.meta, ap, cap, &cv, st, b.C))
		}
		if s.err == nil {
			s.fail(b.seq.Ink2BAttend(q, r, sinks, l.relBias, b.kc[slot], b.vc[slot], b.scores, attn, b.meta, ap, extent, window, cap, st, b.C))
		}
	}
	for i := 0; b.seq == nil && i < b.C && s.err == nil; i++ {
		ci := cv
		ci.KDyn, ci.VDyn = inkling2SeqRow(cv.KDyn, st.KDyn, i), inkling2SeqRow(cv.VDyn, st.VDyn, i)
		ci.KHist, ci.VHist = inkling2SeqRow(cv.KHist, st.Hist, i), inkling2SeqRow(cv.VHist, st.Hist, i)
		qi, ri, meta := inkling2SeqRow(q, st.Q, i), inkling2SeqRow(r, st.R, i), inkling2SeqRow(b.meta, st.Meta, i)
		kci, vci := inkling2SeqRow(b.kc[slot], st.KCache, i), inkling2SeqRow(b.vc[slot], st.VCache, i)
		s.fail(s.dev.Ink2QKVPost(qi, ri, inkling2SeqRow(k, st.K, i), inkling2SeqRow(v, st.V, i), l.qclip, l.relClip, kci, vci, meta, ap, cap, &ci))
		if s.err == nil {
			s.fail(s.dev.Ink2Attend(qi, ri, inkling2SeqRow(sinks, st.Sinks, i), l.relBias, kci, vci, inkling2SeqRow(b.scores, st.Scores, i),
				inkling2SeqRow(attn, st.Out, i), meta, ap, extent, window, cap))
		}
	}
	gate := p.mm(l.gate, in[0])
	if s.err == nil {
		s.fail(s.dev.Ink2SigmoidMul(attn, gate, b.C*qd))
	}
	o := p.mm(l.wo, attn)
	var ffnInp ink2Ref
	if ip%hp.AttnResBlock == 0 {
		*blocks = append(*blocks, inpSA)
		ffnInp = b.conv(slot, o, l.attnPostConv, ink2Ref{}, 2*kvDim, false)
	} else {
		ffnInp = b.conv(slot, o, l.attnPostConv, inpSA, 2*kvDim, false)
	}

	x := p.mix(with(*blocks, ffnInp), l.ffnResQ, 1)[0]
	if msgs != nil {
		x = p.control(x, msgs[3], c, 3)
	}
	x = p.rms(x, l.ffnNorm, h, hp.NormEps)
	x = p.moe(l, x)
	return b.conv(slot, x, l.ffnPostConv, ffnInp, 2*kvDim+h, true)
}

// conv runs a hidden-width post conv on every row against its sequence's history at
// channel offset; keep holds the output past the layer visit.
func (b *inkling2Batch) conv(slot int, x, weight, resid ink2Ref, offset int, keep bool) ink2Ref {
	s := b.s
	h, kn := s.hp.Hidden, s.hp.ConvKernel
	norm := inkling2ConvNorm(kn, ink2Ref{})
	var out ink2Ref
	if keep {
		out = b.p.hold(b.C * h)
	} else {
		out = b.p.tmp(b.C * h)
	}
	hist := at(b.hist[slot], offset*(kn-1))
	st := compute.Ink2SeqStrides{X: h, Out: h, Resid: h, Hist: b.histStride}
	if b.seq != nil {
		if s.err == nil {
			s.fail(b.seq.Ink2BConv(x, weight, hist, out, resid, h, kn, norm, st, b.C))
		}
		return out
	}
	for i := 0; i < b.C && s.err == nil; i++ {
		s.fail(s.dev.Ink2Conv(inkling2SeqRow(x, st.X, i), weight, ink2Ref{}, inkling2SeqRow(hist, st.Hist, i), inkling2SeqRow(out, st.Out, i),
			inkling2SeqRow(resid, st.Resid, i), h, 1, kn, norm))
	}
	return out
}

// forward enqueues one lockstep step from the uploaded rows through every row's argmax.
func (b *inkling2Batch) forward() {
	p := &b.p
	s := b.s
	hp := s.hp
	h := hp.Hidden
	p.keepOff, p.scrOff = 0, 0
	cur := p.hold(b.C * h)
	if s.err == nil {
		s.fail(s.dev.Ink2RMS(b.in, s.tokNorm, cur, b.C, h, hp.NormEps))
	}
	visit := func() { p.scrOff = 0 }
	var blocks []ink2Ref
	for il := 0; il < hp.CoreStart; il++ {
		visit()
		cur = b.layer(il, il, cur, &blocks, nil, nil)
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
			visit()
			if pass == 0 {
				cur = b.layer(ip, ip, cur, &blocks, nil, nil)
				continue
			}
			slot := hp.Layers + (pass-1)*core + (ip - hp.CoreStart)
			c := &s.ctrls[ip-hp.CoreStart]
			msgs := p.mix(with(bank, bankOut), c.msgQ, 4)
			cur = b.layer(slot, ip, cur, &blocks, msgs, c)
		}
	}
	for il := hp.CoreEnd; il < hp.Layers; il++ {
		visit()
		cur = b.layer(il, il, cur, &blocks, nil, nil)
	}
	visit()
	x := p.mix(with(blocks, cur), s.outResQ, 1)[0]
	xn := p.rms(x, s.outNorm, h, hp.NormEps)
	logits := p.mm(s.output, xn)
	if b.seq != nil {
		if s.err == nil {
			s.fail(b.seq.Ink2BSoftcapArgmax(logits, hp.VocabUnpadded, s.output.rows, hp.FinalLogitSoftcap, b.argIdx, b.C))
		}
		return
	}
	for i := 0; i < b.C && s.err == nil; i++ {
		s.fail(s.dev.Ink2SoftcapArgmax(at(logits, i*s.output.rows), hp.VocabUnpadded, hp.FinalLogitSoftcap, at(b.argIdx, 4*i)))
	}
}

// step feeds tokens[i] to sequence i at its next position and returns every
// sequence's greedy next token.
func (b *inkling2Batch) step(tokens []int) ([]int, error) {
	s := b.s
	hp := s.hp
	if len(tokens) != b.C {
		return nil, fmt.Errorf("inkling2 batch: %d tokens for %d sequences", len(tokens), b.C)
	}
	rows := make([]float32, b.C*hp.Hidden)
	meta := make([]float32, b.C)
	limit := b.ctx
	for i, t := range tokens {
		if b.pos[i] >= limit {
			return nil, fmt.Errorf("inkling2 batch: sequence %d position %d exceeds the %d-token cache", i, b.pos[i], limit)
		}
		s.w.TokEmbd.Row(t, rows[i*hp.Hidden:(i+1)*hp.Hidden])
		meta[i] = math.Float32frombits(uint32(int32(b.pos[i])))
	}
	s.fail(s.dev.Ink2Write(b.in, rows))
	s.fail(s.dev.Ink2Write(b.meta, meta))
	switch {
	case s.err != nil:
	case b.graphReady:
		s.fail(s.dev.Ink2GraphLaunch())
	case b.graph && b.warm:
		// capture after one eager step has grown every lazily sized kernel scratch
		if err := s.dev.Ink2GraphBegin(); err != nil {
			b.graph = false
			b.forward()
			break
		}
		b.forward()
		if err := s.dev.Ink2GraphEnd(); err != nil || s.err != nil {
			s.fail(err)
			b.graph = false
			break
		}
		b.graphReady = true
		s.fail(s.dev.Ink2GraphLaunch())
	default:
		b.forward()
		b.warm = true
	}
	if s.err != nil {
		return nil, s.err
	}
	got, err := s.dev.Ink2ReadInts(b.argIdx, 4*b.C)
	if err != nil {
		return nil, err
	}
	out := make([]int, b.C)
	for i := range out {
		out[i] = int(got[4*i])
		b.pos[i]++
	}
	if r, ok := s.be.(interface{ Recycle() }); ok {
		r.Recycle()
	}
	return out, nil
}

// Inkling2LoopBatchStats times a lockstep batched greedy decode. Prompt covers the
// steps in which some sequence still feeds a prompt token; Decode covers the
// DecodeSteps steps in which every sequence feeds a generated token, and
// DecodeTokens counts the kept tokens those steps produced.
type Inkling2LoopBatchStats struct {
	Sequences    int
	StateBytes   int64
	Prompt       time.Duration
	Decode       time.Duration
	DecodeSteps  int
	DecodeTokens int
}

// AggregateTokensPerSecond is the generated tokens of every sequence per second of
// the decode phase.
func (st Inkling2LoopBatchStats) AggregateTokensPerSecond() float64 {
	if st.Decode <= 0 {
		return 0
	}
	return float64(st.DecodeTokens) / st.Decode.Seconds()
}

// inkling2BatchGreedy decodes n greedy tokens after each prompt with every sequence
// advancing in lockstep, one step per token: a sequence still inside its prompt
// feeds its next prompt token, and the others feed their last generated token.
func inkling2BatchGreedy(m *Inkling2LoopModel, be compute.Backend, prompts [][]int, n, ctx int) ([][]int, Inkling2LoopBatchStats, error) {
	st := Inkling2LoopBatchStats{Sequences: len(prompts)}
	longest := 0
	for i, pr := range prompts {
		if len(pr) == 0 {
			return nil, st, fmt.Errorf("inkling2 batch: sequence %d has an empty prompt", i)
		}
		longest = max(longest, len(pr))
	}
	b, err := newInkling2Batch(m, be, len(prompts), ctx)
	if err != nil {
		return nil, st, err
	}
	defer b.close()
	st.StateBytes = b.StateBytes()
	// a CUDA stream capture is bound to the OS thread that opened it
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	outs := make([][]int, len(prompts))
	tokens := make([]int, len(prompts))
	start := time.Now()
	for t := 0; ; t++ {
		done := true
		for i, pr := range prompts {
			if t < len(pr) {
				tokens[i] = pr[t]
			} else {
				tokens[i] = outs[i][len(outs[i])-1]
			}
			if len(outs[i]) < n {
				done = false
			}
		}
		if done {
			break
		}
		if t == longest {
			st.Prompt = time.Since(start)
			start = time.Now()
		}
		next, err := b.step(tokens)
		if err != nil {
			return nil, st, err
		}
		if t >= longest {
			st.DecodeSteps++
		}
		for i, pr := range prompts {
			if t >= len(pr)-1 && len(outs[i]) < n {
				outs[i] = append(outs[i], next[i])
				if t >= longest {
					st.DecodeTokens++
				}
			}
		}
	}
	if st.DecodeSteps > 0 {
		st.Decode = time.Since(start)
	} else {
		st.Prompt = time.Since(start)
	}
	return outs, st, nil
}
