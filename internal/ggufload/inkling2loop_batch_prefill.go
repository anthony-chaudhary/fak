package ggufload

import (
	"fmt"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// inkling2CohortPrefill reports whether a cohort runs its prompts through the batched
// prefill forward as one ragged token stream before lockstep decode;
// FAK_INKLING2_COHORT_PREFILL=0 feeds every prompt one position per lockstep step.
func inkling2CohortPrefill(lookup func(string) (string, bool)) bool {
	v, ok := lookup("FAK_INKLING2_COHORT_PREFILL")
	return !ok || v != "0"
}

// ink2PrefillSeg is the run of one sequence's prompt inside a cohort prefill chunk:
// tokens off..off+n of the chunk are that row's positions pos0..pos0+n.
type ink2PrefillSeg struct {
	row, pos0, off, n int
	// last is set on the segment holding the row's final prompt token
	last bool
}

// inkling2PrefillChunks packs the prompts, row after row, into chunks of at most T
// tokens; a prompt longer than the room left in a chunk continues in the next.
func inkling2PrefillChunks(lens []int, T int) [][]ink2PrefillSeg {
	var chunks [][]ink2PrefillSeg
	var cur []ink2PrefillSeg
	used := 0
	for row, n := range lens {
		for done := 0; done < n; {
			if used == T {
				chunks, cur, used = append(chunks, cur), nil, 0
			}
			take := min(T-used, n-done)
			cur = append(cur, ink2PrefillSeg{row: row, pos0: done, off: used, n: take, last: done+take == n})
			used += take
			done += take
		}
	}
	if len(cur) > 0 {
		chunks = append(chunks, cur)
	}
	return chunks
}

// ink2PrefillRows makes a prefill chunk a ragged batch of cohort sequences: the dense
// and expert ops run over every token of the chunk at once, and the ops that touch
// sequence state run per segment against that row of the cohort's strided arenas.
type ink2PrefillRows struct {
	b    *inkling2Batch
	segs []ink2PrefillSeg
}

// conv is ink2Prefill.conv for every segment, each against its row's history.
func (r *ink2PrefillRows) conv(p *ink2Prefill, slot int, x, weight, dyn, resid, out ink2Ref, n, heads, offset int, norm float32) {
	b := r.b
	kn := b.s.hp.ConvKernel
	for _, sg := range r.segs {
		if p.s.err != nil {
			return
		}
		hist := at(b.hist[slot], sg.row*b.histStride+offset*(kn-1))
		p.fail(p.pd.Ink2PConv(at(x, sg.off*n), weight, inkling2SeqRow(dyn, heads*kn, sg.off), hist, at(out, sg.off*n),
			inkling2SeqRow(resid, n, sg.off), n, heads, kn, norm, sg.n))
	}
}

// attend appends every segment's keys and values to its row's caches and attends its
// queries over them.
func (r *ink2PrefillRows) attend(p *ink2Prefill, slot int, l *ink2Layer, q, rel, k, v, sinks, attn ink2Ref,
	ap compute.Ink2AttnParams, extent, window int) {
	b := r.b
	hp := b.s.hp
	kvDim := hp.KVHeads * hp.HeadDim
	qd := hp.Heads * hp.HeadDim
	cap := b.caps[slot]
	for _, sg := range r.segs {
		if p.s.err != nil {
			return
		}
		kc, vc := at(b.kc[slot], sg.row*cap*kvDim), at(b.vc[slot], sg.row*cap*kvDim)
		qs, rs := at(q, sg.off*l.wq.rows), at(rel, sg.off*l.rel.rows)
		p.fail(p.pd.Ink2PQKVPost(qs, rs, at(k, sg.off*kvDim), at(v, sg.off*kvDim), l.qclip, l.relClip, kc, vc, ap, cap, sg.pos0, sg.n))
		if p.s.err == nil {
			p.fail(p.pd.Ink2PAttend(qs, rs, at(sinks, sg.off*l.sink.rows), l.relBias, kc, vc, at(attn, sg.off*qd), ap, extent, window,
				cap, sg.pos0, sg.n))
		}
	}
}

// head runs the output head on the last prompt token of every row that ends in the
// chunk, reading the weight once for all of them, and leaves row i's greedy token at
// argIdx[4i].
func (r *ink2PrefillRows) head(p *ink2Prefill, blocks []ink2Ref, cur ink2Ref) {
	s := p.s
	hp := s.hp
	h := hp.Hidden
	var ends []ink2PrefillSeg
	for _, sg := range r.segs {
		if sg.last {
			ends = append(ends, sg)
		}
	}
	if len(ends) == 0 {
		return
	}
	p.scrOff = 0
	E := len(ends)
	x := p.tmp(E * h)
	states := make([]ink2Ref, len(blocks)+1)
	for j, sg := range ends {
		row := (sg.off + sg.n - 1) * h
		for i, blk := range blocks {
			states[i] = at(blk, row)
		}
		states[len(blocks)] = at(cur, row)
		if s.err == nil {
			p.fail(s.dev.Ink2Mix(states, s.outResQ, 1, nil, 0, at(x, j*h), h))
		}
	}
	xn := p.tmp(E * h)
	if s.err == nil {
		p.fail(s.dev.Ink2RMS(x, s.outNorm, xn, E, h, hp.NormEps))
	}
	logits := p.tmp(E * s.output.rows)
	if s.err == nil {
		p.fail(p.pd.Ink2PMatMul(s.output.t, xn, logits, E))
	}
	for j, sg := range ends {
		if s.err == nil {
			p.fail(s.dev.Ink2SoftcapArgmax(at(logits, j*s.output.rows), hp.VocabUnpadded, hp.FinalLogitSoftcap, at(r.b.argIdx, 4*sg.row)))
		}
	}
}

// prefill runs every row's whole prompt through the batched prefill forward, chunked
// at prefillT tokens across rows, into each row's caches and histories, leaving row i
// at position len(prompts[i]) exactly as feeding the prompt one lockstep step at a
// time would; it returns every row's greedy token after its prompt.
func (b *inkling2Batch) prefill(prompts [][]int) ([]int, error) {
	s := b.s
	h := s.hp.Hidden
	if b.prefillT < 1 {
		return nil, fmt.Errorf("inkling2 batch: state sized without cohort prefill")
	}
	if len(prompts) != b.C {
		return nil, fmt.Errorf("inkling2 batch: %d prompts for %d sequences", len(prompts), b.C)
	}
	lens := make([]int, b.C)
	for i, pr := range prompts {
		if b.pos[i] != 0 || len(pr) == 0 || len(pr) > b.ctx {
			return nil, fmt.Errorf("inkling2 batch: sequence %d at position %d cannot prefill %d tokens into %d", i, b.pos[i], len(pr), b.ctx)
		}
		lens[i] = len(pr)
	}
	p := &ink2Prefill{s: s, pd: b.p.pd, grouped: inkling2GroupedExperts(), in: b.in,
		keep: b.p.keep, scr: b.p.scr, keepFloats: b.p.keepFloats, scrFloats: b.p.scrFloats}
	rows := make([]float32, b.prefillT*h)
	for _, segs := range inkling2PrefillChunks(lens, b.prefillT) {
		n := 0
		for _, sg := range segs {
			for j, tok := range prompts[sg.row][sg.pos0 : sg.pos0+sg.n] {
				s.w.TokEmbd.Row(tok, rows[(sg.off+j)*h:(sg.off+j+1)*h])
			}
			n = sg.off + sg.n
		}
		s.fail(s.dev.Ink2Write(b.in, rows[:n*h]))
		p.T, p.keepOff, p.scrOff = n, 0, 0
		p.cohort = &ink2PrefillRows{b: b, segs: segs}
		p.forward(false)
		if s.err != nil {
			return nil, s.err
		}
	}
	got, err := s.dev.Ink2ReadInts(b.argIdx, 4*b.C)
	if err != nil {
		return nil, err
	}
	out := make([]int, b.C)
	for i := range out {
		out[i] = int(got[4*i])
		b.pos[i] = lens[i]
	}
	if r, ok := s.be.(interface{ Recycle() }); ok {
		r.Recycle()
	}
	return out, nil
}

// park holds a finished row below the cache bound: every step advances every row, and
// a prefilled row starts decoding at its own prompt length, so a finished short-budget
// row would otherwise step past ctx while a longer row still decodes.
func (b *inkling2Batch) park(i int) { b.pos[i] = min(b.pos[i], b.ctx-1) }
