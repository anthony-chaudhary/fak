package ggufload

import (
	"fmt"
	"strconv"
	"strings"

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

// inkling2PrefillNextChunk packs the next chunk of at most T tokens from the rows'
// remaining prompts, row after row, advancing cursor past what it packs. lens holds
// each row's end position, so a row whose cursor has reached its end contributes
// nothing; ending a dropped row at its cursor keeps it out of every later chunk.
func inkling2PrefillNextChunk(lens, cursor []int, T int) []ink2PrefillSeg {
	var segs []ink2PrefillSeg
	used := 0
	for row, n := range lens {
		for cursor[row] < n && used < T {
			done := cursor[row]
			take := min(T-used, n-done)
			segs = append(segs, ink2PrefillSeg{row: row, pos0: done, off: used, n: take, last: done+take == n})
			used += take
			cursor[row] = done + take
		}
	}
	return segs
}

// inkling2PrefillChunks packs the prompts, row after row, into chunks of at most T
// tokens; a prompt longer than the room left in a chunk continues in the next.
func inkling2PrefillChunks(lens []int, T int) [][]ink2PrefillSeg {
	cursor := make([]int, len(lens))
	var chunks [][]ink2PrefillSeg
	for segs := inkling2PrefillNextChunk(lens, cursor, T); len(segs) > 0; segs = inkling2PrefillNextChunk(lens, cursor, T) {
		chunks = append(chunks, segs)
	}
	return chunks
}

// inkling2MixedPrefillPlan plans one mixed chunk of at most T tokens without changing
// cursor. A row is prefilling while its cursor is below its lens entry. Each prefilling
// row first takes up to limit tokens, in row order, so a short prompt is not queued
// behind a long one; the room left then goes to the prefilling rows in row order. Each
// row in decoding adds one decode segment after the prompt segments, and while any row
// decodes the chunk's prompt share is held to limit. Each row gets one contiguous segment.
func inkling2MixedPrefillPlan(lens, cursor, decoding []int, T, limit int) []ink2PrefillSeg {
	budget := T - len(decoding)
	if len(decoding) > 0 {
		budget = min(limit, budget)
	}
	left := max(budget, 0)
	take := make([]int, len(lens))
	for row := range lens {
		if rem := lens[row] - cursor[row]; rem > 0 {
			take[row] = min(rem, limit, left)
			left -= take[row]
		}
	}
	for row := range lens {
		if rem := lens[row] - cursor[row] - take[row]; rem > 0 && left > 0 {
			extra := min(rem, left)
			take[row] += extra
			left -= extra
		}
	}
	var segs []ink2PrefillSeg
	off := 0
	for row, n := range take {
		if n == 0 {
			continue
		}
		segs = append(segs, ink2PrefillSeg{row: row, pos0: cursor[row], off: off, n: n, last: cursor[row]+n == lens[row]})
		off += n
	}
	for _, row := range decoding {
		segs = append(segs, ink2PrefillSeg{row: row, pos0: cursor[row], off: off, n: 1, last: true})
		off++
	}
	return segs
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

// prefill runs every row's prompt through the batched prefill forward, chunked at
// prefillT tokens across rows, into each row's caches and histories, leaving row i at
// position cursor[i] exactly as feeding the prompt one lockstep step at a time would.
// Every prompt must be non-empty. keep is polled before each chunk for every row that
// still has prompt left; a row it refuses has its remaining prompt dropped from every
// later chunk, so a cancelled row stops costing compute at the next chunk. A nil keep
// keeps every row. first is called with each kept row's greedy token as soon as the
// chunk that ends its prompt has run.
func (b *inkling2Batch) prefill(prompts [][]int, keep func(row int) bool, first func(row, token int)) error {
	if b.prefillT < 1 {
		return fmt.Errorf("inkling2 batch: state sized without cohort prefill")
	}
	if len(prompts) != b.C {
		return fmt.Errorf("inkling2 batch: %d prompts for %d sequences", len(prompts), b.C)
	}
	end := make([]int, b.C)
	for i, pr := range prompts {
		if len(pr) == 0 {
			return fmt.Errorf("inkling2 batch: sequence %d has an empty prompt", i)
		}
		if b.pos[i] != 0 || len(pr) > b.ctx {
			return fmt.Errorf("inkling2 batch: sequence %d at position %d cannot prefill %d tokens into %d", i, b.pos[i], len(pr), b.ctx)
		}
		end[i] = len(pr)
	}
	cursor := make([]int, b.C)
	for {
		more := false
		for i := range cursor {
			if cursor[i] < end[i] && keep != nil && !keep(i) {
				end[i] = cursor[i]
			}
			more = more || cursor[i] < end[i]
		}
		if !more {
			break
		}
		segs := inkling2PrefillNextChunk(end, cursor, b.prefillT)
		toks := make([][]int, len(segs))
		for k, sg := range segs {
			toks[k] = prompts[sg.row][sg.pos0 : sg.pos0+sg.n]
		}
		if err := b.prefillChunk(segs, toks, first); err != nil {
			return err
		}
	}
	b.endPrefill(cursor)
	return nil
}

// prefillChunk runs one ragged chunk: toks[k] are the tokens of segs[k], embedded at
// the segment's offset in the chunk, and the forward writes each segment into its own
// row. first is called with the greedy token of every segment that ends its row.
func (b *inkling2Batch) prefillChunk(segs []ink2PrefillSeg, toks [][]int, first func(row, token int)) error {
	s := b.s
	h := s.hp.Hidden
	n := 0
	for _, sg := range segs {
		n = max(n, sg.off+sg.n)
	}
	rows := make([]float32, n*h)
	for k, sg := range segs {
		for j, tok := range toks[k] {
			s.w.TokEmbd.Row(tok, rows[(sg.off+j)*h:(sg.off+j+1)*h])
		}
	}
	s.fail(s.dev.Ink2Write(b.in, rows))
	p := &ink2Prefill{s: s, pd: b.p.pd, grouped: inkling2GroupedExperts(), in: b.in,
		keep: b.p.keep, scr: b.p.scr, keepFloats: b.p.keepFloats, scrFloats: b.p.scrFloats}
	p.T = n
	p.cohort = &ink2PrefillRows{b: b, segs: segs}
	p.forward(false)
	if s.err != nil {
		return s.err
	}
	return b.deliverFirst(segs, first)
}

// endPrefill leaves each row at its cursor for the lockstep decode that follows.
func (b *inkling2Batch) endPrefill(cursor []int) {
	copy(b.pos, cursor)
	if r, ok := b.s.be.(interface{ Recycle() }); ok {
		r.Recycle()
	}
}

// deliverFirst reads the greedy tokens of the rows that ended their prompt in this
// chunk; the head wrote each one to its row's slot of argIdx, which no later chunk
// rewrites.
func (b *inkling2Batch) deliverFirst(segs []ink2PrefillSeg, first func(row, token int)) error {
	ended := false
	for _, sg := range segs {
		ended = ended || sg.last
	}
	if !ended || first == nil {
		return nil
	}
	got, err := b.s.dev.Ink2ReadInts(b.argIdx, 4*b.C)
	if err != nil {
		return err
	}
	for _, sg := range segs {
		if sg.last {
			first(sg.row, int(got[4*sg.row]))
		}
	}
	return nil
}

// park holds a finished row below the cache bound: every step advances every row, and
// a prefilled row starts decoding at its own prompt length, so a finished short-budget
// row would otherwise step past ctx while a longer row still decodes.
func (b *inkling2Batch) park(i int) { b.pos[i] = min(b.pos[i], b.ctx-1) }

// inkling2CohortLongPrefillCap reads FAK_NATIVE_LONG_PREFILL_TOKEN_THRESHOLD, the knob
// of the native scheduler's long-prefill token threshold, so the cohort
// gives a row whose prompt has ended its decode tokens in the chunks of a long prompt
// still prefilling, and caps that prompt's share of each such chunk. Unset, empty,
// zero, negative and unparsable all mean off. It stays off by default until a device
// receipt exists: a mixed chunk decodes through the prefill kernels, which differ
// numerically from the step kernels on device.
func inkling2CohortLongPrefillCap(lookup func(string) (string, bool)) int {
	v, ok := lookup("FAK_NATIVE_LONG_PREFILL_TOKEN_THRESHOLD")
	if !ok {
		return 0
	}
	n, err := strconv.Atoi(strings.TrimSpace(v))
	if err != nil || n < 0 {
		return 0
	}
	return n
}
