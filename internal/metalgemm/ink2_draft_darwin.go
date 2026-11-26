//go:build darwin && arm64 && cgo

package metalgemm

import (
	_ "embed"
	"fmt"
	"math"
)

//go:embed ink2_draft.metal
var ink2DraftSource string

// ink2DFMaxTokens is the block rows one draft launch covers.
const ink2DFMaxTokens = 8

type ink2DFLinearShape struct{ outRows, cols, seg, segs, tokens, shift uint32 }

type ink2DFConvShape struct {
	rows, hidden, group, side, kn, dynStride, baseShift, residual uint32
}

type ink2DFAttendShape struct {
	rows, heads, kvHeads, dim, window, cap uint32
	scale                                  float32
}

type ink2DFTopKShape struct{ vocab, rowStride, k, shards, chunk uint32 }

type ink2DFSelectShape struct {
	rows, k, shards, rank, vocab, predShift, succShift, candidates uint32
	softcap                                                        float32
}

// ink2DFBF16 binds a BF16 weight's bytes and returns the element shift the
// kernel adds past the 4-byte aligned binding.
func ink2DFBF16(w Ink2Weight, what string) (Ink2Arg, uint32, error) {
	const bf16 = 30
	if !w.Buf.Valid() || w.Type != bf16 || w.ByteOff < 0 || w.ByteOff%2 != 0 || w.Rows < 1 || w.Cols < 1 ||
		w.Rows > math.MaxInt32/w.Cols {
		return Ink2Arg{}, 0, fmt.Errorf("%w: %s weight type %d [%d %d] at byte %d", ErrInk2Shape, what, w.Type, w.Rows, w.Cols, w.ByteOff)
	}
	shift := w.ByteOff & 3
	return Ink2BufferAt(w.Buf, w.ByteOff-shift, 2*w.Rows*w.Cols+shift), uint32(shift / 2), nil
}

// DraftLinear encodes y = x W^T for tokens <= 8 rows of a BF16 [outRows][cols]
// weight. x holds 1 to 4 column segments of cols/len(x) columns each, every one
// its own [tokens][seg] matrix, so the fc fuse reads the target taps in place.
// y is [tokens][outRows]. The sums reduce across a SIMD group, so they match a
// serial host dot only to rounding. Under DraftQ8Env the weight is read as
// its per-row int8 copy.
func (d *Ink2Device) DraftLinear(x []Ink2Ref, w Ink2Weight, y Ink2Ref, tokens int) error {
	warg, shift, err := ink2DFBF16(w, "draft linear")
	if err != nil {
		return err
	}
	if len(x) < 1 || len(x) > 4 || w.Cols%len(x) != 0 || tokens < 1 || tokens > ink2DFMaxTokens || !y.Valid() {
		return fmt.Errorf("%w: draft linear of %d segments over %d columns, %d tokens", ErrInk2Shape, len(x), w.Cols, tokens)
	}
	seg := w.Cols / len(x)
	// The vector kernel's 16-byte weight and float4 token loads need every
	// weight row and token row 16-byte aligned.
	vec := w.ByteOff%16 == 0 && seg%8 == 0
	args := []Ink2Arg{warg}
	for i := range 4 {
		if i >= len(x) {
			args = append(args, Ink2Ref{}.Span(0))
			continue
		}
		if !x[i].Valid() {
			return fmt.Errorf("%w: draft linear segment %d absent", ErrInk2Shape, i)
		}
		vec = vec && x[i].Off%4 == 0
		args = append(args, x[i].Span(tokens*seg))
	}
	// The vector kernel covers two weight rows per threadgroup and splits the
	// columns among its SIMD groups; the scalar one gives each SIMD group a row.
	name, simds, perGroup := "ink2_df_linear", 4, 4
	if vec && seg%16 == 0 && d.draftQ8() {
		if q, ok, err := d.draftQ8Weight(w); err != nil || ok {
			if err != nil {
				return err
			}
			return d.draftLinearQ8(args[1:], q, y, seg, len(x), tokens)
		}
	}
	if vec {
		bucket := 8
		for _, b := range []int{1, 2, 4} {
			if tokens <= b {
				bucket = b
				break
			}
		}
		name, simds, perGroup = fmt.Sprintf("ink2_df_linear_t%d", bucket), ink2DFLinearSimds(seg/8), 2
	}
	p, err := d.Pipeline(ink2DraftSource, name)
	if err != nil {
		return err
	}
	if p.ThreadWidth() != 32 || p.MaxThreads() < 32*simds {
		return fmt.Errorf("metalgemm: inkling2 draft linear needs %d SIMD groups of 32", simds)
	}
	args = append(args, y.Span(tokens*w.Rows), Ink2Const(ink2DFLinearShape{
		outRows: uint32(w.Rows), cols: uint32(w.Cols), seg: uint32(seg), segs: uint32(len(x)), tokens: uint32(tokens), shift: shift}))
	return d.Dispatch(p, [3]int{(w.Rows + perGroup - 1) / perGroup, 1, 1}, [3]int{32 * simds, 1, 1}, args...)
}

type ink2DFMultiShape struct{ rows0, rows1, rows2, end0, end1, cols, seg, segs, tokens uint32 }

// ink2DFBucket is the token bucket of the vector linear kernels.
func ink2DFBucket(tokens int) int {
	for _, b := range []int{1, 2, 4} {
		if tokens <= b {
			return b
		}
	}
	return 8
}

// ink2DFVecInputs binds 1 to 4 input segments of seg columns for the vector
// kernels, false when one is not float4 aligned.
func ink2DFVecInputs(x []Ink2Ref, seg, tokens int) ([]Ink2Arg, bool, error) {
	if len(x) < 1 || len(x) > 4 {
		return nil, false, fmt.Errorf("%w: draft linear of %d segments", ErrInk2Shape, len(x))
	}
	args, vec := make([]Ink2Arg, 0, 4), seg%8 == 0
	for i := range 4 {
		if i >= len(x) {
			args = append(args, Ink2Ref{}.Span(0))
			continue
		}
		if !x[i].Valid() {
			return nil, false, fmt.Errorf("%w: draft linear segment %d absent", ErrInk2Shape, i)
		}
		vec = vec && x[i].Off%4 == 0
		args = append(args, x[i].Span(tokens*seg))
	}
	return args, vec, nil
}

// DraftLinearMulti is DraftLinear of up to three BF16 weights of equal
// columns over one input, y[i] = x W[i]^T, in one launch; each y[i] is what
// DraftLinear writes. A weight or input the vector kernel cannot read falls
// back to one DraftLinear per weight.
func (d *Ink2Device) DraftLinearMulti(x []Ink2Ref, w []Ink2Weight, y []Ink2Ref, tokens int) error {
	if len(w) < 1 || len(w) > 3 || len(y) != len(w) || tokens < 1 || tokens > ink2DFMaxTokens {
		return fmt.Errorf("%w: draft linear of %d weights into %d outputs, %d tokens", ErrInk2Shape, len(w), len(y), tokens)
	}
	cols := w[0].Cols
	vec := len(x) > 0 && cols%len(x) == 0
	for i := range w {
		vec = vec && w[i].Cols == cols && w[i].Type == 30 && w[i].ByteOff%16 == 0 && y[i].Valid()
	}
	var xargs []Ink2Arg
	if vec {
		var ok bool
		var err error
		if xargs, ok, err = ink2DFVecInputs(x, cols/len(x), tokens); err != nil {
			return err
		}
		vec = ok
	}
	if !vec {
		for i := range w {
			if err := d.DraftLinear(x, w[i], y[i], tokens); err != nil {
				return err
			}
		}
		return nil
	}
	seg := cols / len(x)
	if seg%16 == 0 {
		if qs, ok, err := d.draftQ8Weights(w); err != nil || ok {
			if err != nil {
				return err
			}
			return d.draftLinearMultiQ8(xargs, qs, y, seg, len(x), tokens)
		}
	}
	var wargs, yargs [3]Ink2Arg
	var rows [3]int
	for i := range 3 {
		j := min(i, len(w)-1)
		warg, _, err := ink2DFBF16(w[j], "draft linear")
		if err != nil {
			return err
		}
		wargs[i], yargs[i], rows[i] = warg, y[j].Span(tokens*w[j].Rows), w[j].Rows
		if i >= len(w) {
			rows[i] = 0
		}
	}
	simds := ink2DFLinearSimds(seg / 8)
	p, err := d.Pipeline(ink2DraftSource, fmt.Sprintf("ink2_df_linear_multi_t%d", ink2DFBucket(tokens)))
	if err != nil {
		return err
	}
	if p.ThreadWidth() != 32 || p.MaxThreads() < 32*simds {
		return fmt.Errorf("metalgemm: inkling2 draft linear needs %d SIMD groups of 32", simds)
	}
	g := func(r int) int { return (r + 1) / 2 }
	end0, end1 := g(rows[0]), g(rows[0])+g(rows[1])
	args := append(wargs[:], xargs...)
	args = append(args, yargs[:]...)
	args = append(args, Ink2Const(ink2DFMultiShape{rows0: uint32(rows[0]), rows1: uint32(rows[1]), rows2: uint32(rows[2]),
		end0: uint32(end0), end1: uint32(end1), cols: uint32(cols), seg: uint32(seg), segs: uint32(len(x)), tokens: uint32(tokens)}))
	return d.Dispatch(p, [3]int{end1 + g(rows[2]), 1, 1}, [3]int{32 * simds, 1, 1}, args...)
}

// DraftLinearSiluMul is y = silu(x Wg^T) * (x Wu^T) for BF16 gate and up
// weights of one shape, equal to DraftLinear twice and DraftSiluMul. A weight
// or input the vector kernel cannot read takes that three-launch path, with
// scratch ([tokens][rows of Wg]) holding the up projection.
func (d *Ink2Device) DraftLinearSiluMul(x Ink2Ref, gate, up Ink2Weight, y, scratch Ink2Ref, tokens int) error {
	if gate.Rows != up.Rows || gate.Cols != up.Cols || !y.Valid() || tokens < 1 || tokens > ink2DFMaxTokens {
		return fmt.Errorf("%w: draft gated linear of [%d %d] and [%d %d], %d tokens", ErrInk2Shape, gate.Rows, gate.Cols, up.Rows, up.Cols, tokens)
	}
	xargs, vec, err := ink2DFVecInputs([]Ink2Ref{x}, gate.Cols, tokens)
	if err != nil {
		return err
	}
	if !vec || gate.Type != 30 || up.Type != 30 || gate.ByteOff%16 != 0 || up.ByteOff%16 != 0 {
		if !scratch.Valid() {
			return fmt.Errorf("%w: draft gated linear fallback without scratch", ErrInk2Shape)
		}
		if err := d.DraftLinear([]Ink2Ref{x}, gate, y, tokens); err != nil {
			return err
		}
		if err := d.DraftLinear([]Ink2Ref{x}, up, scratch, tokens); err != nil {
			return err
		}
		return d.DraftSiluMul(y, scratch, y, tokens*gate.Rows)
	}
	if qs, ok, err := d.draftQ8Weights([]Ink2Weight{gate, up}); err != nil || ok {
		if err != nil {
			return err
		}
		return d.draftLinearSiluQ8(xargs[0], qs[0], qs[1], y, tokens)
	}
	garg, _, err := ink2DFBF16(gate, "draft gate")
	if err != nil {
		return err
	}
	uarg, _, err := ink2DFBF16(up, "draft up")
	if err != nil {
		return err
	}
	simds := ink2DFLinearSimds(gate.Cols / 8)
	p, err := d.PipelinePrecise(ink2DraftSource, fmt.Sprintf("ink2_df_linear_silu_t%d", ink2DFBucket(tokens)))
	if err != nil {
		return err
	}
	if p.ThreadWidth() != 32 || p.MaxThreads() < 32*simds {
		return fmt.Errorf("metalgemm: inkling2 draft gated linear needs %d SIMD groups of 32", simds)
	}
	return d.Dispatch(p, [3]int{(gate.Rows + 1) / 2, 1, 1}, [3]int{32 * simds, 1, 1},
		garg, uarg, xargs[0], y.Span(tokens*gate.Rows),
		Ink2Const(ink2DFLinearShape{outRows: uint32(gate.Rows), cols: uint32(gate.Cols), seg: uint32(gate.Cols), segs: 1, tokens: uint32(tokens)}))
}

// ink2DFLinearSimds splits a segment of units 8-column lane loads among
// SIMD groups so each runs the same number of passes when units allows it.
func ink2DFLinearSimds(units int) int {
	if units%32 == 0 {
		for s := 12; s >= 4; s-- {
			if (units/32)%s == 0 {
				return s
			}
		}
	}
	return 4
}

// DraftConv encodes one side of the DFlash2 dynamic short conv over a block of
// rows x hidden: y[row][c] = sum over taps t <= row of x[row-t][c] times
// (base[side][t][c] + dyn[row][side][t][c/group]), each product added in turn,
// then out = resid + y when resid is set. dyn is the kernel projection output
// ([rows][2][kn][hidden/group]); base the BF16 [2*kn][hidden] base kernel. Side
// 0 prepares the attention or MLP input, side 1 finishes its output. out may
// alias resid but not x.
func (d *Ink2Device) DraftConv(x, dyn Ink2Ref, base Ink2Weight, resid, out Ink2Ref, rows, hidden, group, kn, side int) error {
	barg, shift, err := ink2DFBF16(base, "draft conv base")
	if err != nil {
		return err
	}
	if !x.Valid() || !dyn.Valid() || !out.Valid() || rows < 1 || hidden < 1 || group < 1 || hidden%group != 0 ||
		kn < 1 || side < 0 || side > 1 || rows > math.MaxInt32/hidden || base.Rows != 2*kn || base.Cols != hidden {
		return fmt.Errorf("%w: draft conv of %d rows x %d, group %d, kernel %d, side %d, base [%d %d]",
			ErrInk2Shape, rows, hidden, group, kn, side, base.Rows, base.Cols)
	}
	p, err := d.PipelinePrecise(ink2DraftSource, "ink2_df_conv")
	if err != nil {
		return err
	}
	dynStride := 2 * kn * (hidden / group)
	n := rows * hidden
	return d.DispatchThreads(p, [3]int{n, 1, 1}, [3]int{min(n, 256, p.MaxThreads()), 1, 1},
		x.Span(n), dyn.Span(rows*dynStride), barg, resid.Span(n), out.Span(n),
		Ink2Const(ink2DFConvShape{rows: uint32(rows), hidden: uint32(hidden), group: uint32(group), side: uint32(side),
			kn: uint32(kn), dynStride: uint32(dynStride), baseShift: shift, residual: uint32(btoi(resid.Valid()))}))
}

// ink2DFMaxKeys bounds the keys one query sees: 2047 context positions (a
// 2048 window) plus a block.
const ink2DFMaxKeys = 2047 + ink2DFMaxTokens

// DraftAttend encodes the DFlash2 draft attention of a block of rows queries q
// ([rows][heads][dim]) at positions pos+row, pos the int at posDev. Every row
// sees every block key kb/vb ([rows][kvHeads][dim]) and the context positions
// before pos still in the ring kc/vc ([cap][kvHeads*dim], position p at row
// p%cap) that lie within window-1 of the row (window 0: no limit), at most
// 2047 of them. The single-pass kernel keeps the host's key order for dot
// products, softmax and value sums; the key-split kernel that long rings take
// rounds its sums differently, within the same magnitude bound.
func (d *Ink2Device) DraftAttend(q, kb, vb, kc, vc, out, posDev Ink2Ref, rows, heads, kvHeads, dim, window, cap int) error {
	for _, r := range []Ink2Ref{q, kb, vb, kc, vc, out, posDev} {
		if !r.Valid() {
			return fmt.Errorf("%w: draft attend without an operand", ErrInk2Shape)
		}
	}
	if rows < 1 || rows > ink2DFMaxTokens || heads < 1 || kvHeads < 1 || heads%kvHeads != 0 || dim < 4 || dim > 128 || dim%4 != 0 ||
		window < 0 || cap < 1 || cap > math.MaxInt32/(kvHeads*dim) || DraftAttendSpan(rows, window, cap) > ink2DFMaxKeys {
		return fmt.Errorf("%w: draft attend of %d rows, %d/%d heads x %d, window %d, cache of %d",
			ErrInk2Shape, rows, heads, kvHeads, dim, window, cap)
	}
	kvw := kvHeads * dim
	if d.draftAttendSplit(rows, heads, kvHeads, dim, DraftAttendSpan(rows, window, cap)) {
		return d.draftAttendKeySplit(q, kb, vb, kc, vc, out, posDev, rows, heads, kvHeads, dim, window, cap)
	}
	p, err := d.PipelinePrecise(ink2DraftSource, "ink2_df_attend")
	if err != nil {
		return err
	}
	return d.Dispatch(p, [3]int{heads, rows, 1}, [3]int{min(128, p.MaxThreads()), 1, 1},
		q.Span(rows*heads*dim), kb.Span(rows*kvw), vb.Span(rows*kvw), kc.Span(cap*kvw), vc.Span(cap*kvw),
		out.Span(rows*heads*dim), posDev.Span(1),
		Ink2Const(ink2DFAttendShape{rows: uint32(rows), heads: uint32(heads), kvHeads: uint32(kvHeads), dim: uint32(dim),
			window: uint32(window), cap: uint32(cap), scale: float32(1 / math.Sqrt(float64(dim)))}))
}

// draftAttendSplit reports whether DraftAttend takes the key-split pair of
// launches. The span is the ring's, not the live context's, which the host
// never reads back; splits past the context exit at once.
func (d *Ink2Device) draftAttendSplit(rows, heads, kvHeads, dim, span int) bool {
	// The SIMD-group matrices cover 8 rows of 128-wide heads, a SIMD group
	// per query head of a KV head.
	if rows != 8 || dim != 128 || heads/kvHeads > 8 || (span+ink2DFSplitKeys-1)/ink2DFSplitKeys > ink2DFSplitMax {
		return false
	}
	switch d.dfState().split {
	case 1:
		return true
	case -1:
		return false
	}
	return span > 2*ink2DFSplitKeys
}

// draftAttendKeySplit is DraftAttend as flash decoding: every query of a KV
// head shares one load of each key chunk into a softmax partial, then a merge
// in split order normalizes them. Its sums round differently from the
// single-pass kernel's, within the same magnitude bound.
func (d *Ink2Device) draftAttendKeySplit(q, kb, vb, kc, vc, out, posDev Ink2Ref, rows, heads, kvHeads, dim, window, cap int) error {
	sp, err := d.PipelinePrecise(ink2DraftSource, "ink2_df_attend_split")
	if err != nil {
		return err
	}
	mp, err := d.PipelinePrecise(ink2DraftSource, "ink2_df_attend_merge")
	if err != nil {
		return err
	}
	gsize := heads / kvHeads
	if sp.ThreadWidth() != 32 || sp.MaxThreads() < 32*gsize {
		return fmt.Errorf("metalgemm: inkling2 draft split attend needs %d SIMD groups of 32", gsize)
	}
	splits := (DraftAttendSpan(rows, window, cap) + ink2DFSplitKeys - 1) / ink2DFSplitKeys
	n := rows * heads * splits * (dim + 2)
	part, err := d.draftAttendScratch(n)
	if err != nil {
		return err
	}
	kvw := kvHeads * dim
	shape := Ink2Const(ink2DFAttendSplitShape{rows: uint32(rows), heads: uint32(heads), kvHeads: uint32(kvHeads), dim: uint32(dim),
		window: uint32(window), cap: uint32(cap), scale: float32(1 / math.Sqrt(float64(dim))), splits: uint32(splits)})
	if err := d.Dispatch(sp, [3]int{kvHeads, splits, 1}, [3]int{32 * gsize, 1, 1},
		q.Span(rows*heads*dim), kb.Span(rows*kvw), vb.Span(rows*kvw), kc.Span(cap*kvw), vc.Span(cap*kvw),
		part.Span(n), posDev.Span(1), shape); err != nil {
		return err
	}
	return d.Dispatch(mp, [3]int{heads, rows, 1}, [3]int{min(dim, mp.MaxThreads()), 1, 1},
		part.Span(n), out.Span(rows*heads*dim), posDev.Span(1), shape)
}

// DraftSelect encodes the DFlash2 candidate selector's greedy chain over rows
// draft rows. Each row's candidates are the k <= 16 largest of its first vocab
// logits (row r at logits + r*rowStride; the lower id first among equal
// values). A candidate scores its logit plus sum_i pred[p][i]*proj[r][i]*
// succ[c][i], then softcap*tanh(score/softcap) when softcap > 0; the best (the
// earlier candidate on a tie) is path[r] and the predecessor p of row r+1,
// row 0's being the int at anchorDev. pred and succ are BF16 [vocab][rank]
// codebooks, proj is [rows][rank]. cand, when set, receives each row's k
// candidate ids. A row whose logits or edge scores hold a NaN or infinity
// ends the chain: it and every later row get path (and candidates) -1.
func (d *Ink2Device) DraftSelect(logits Ink2Ref, rows, rowStride, vocab, k int, proj Ink2Ref, pred, succ Ink2Weight,
	anchorDev Ink2Ref, softcap float32, scratch, path, cand Ink2Ref) error {
	parg, pshift, err := ink2DFBF16(pred, "selector predecessor codebook")
	if err != nil {
		return err
	}
	sarg, sshift, err := ink2DFBF16(succ, "selector successor codebook")
	if err != nil {
		return err
	}
	rank := pred.Cols
	if !logits.Valid() || !proj.Valid() || !anchorDev.Valid() || !scratch.Valid() || !path.Valid() ||
		rows < 1 || k < 1 || k > 16 || rank < 1 || rank > 256 || vocab < k || rowStride < vocab || rows > math.MaxInt32/rowStride ||
		pred.Rows < vocab || succ.Rows < vocab || succ.Cols != rank || softcap < 0 || math.IsNaN(float64(softcap)) {
		return fmt.Errorf("%w: draft select of %d rows over %d of %d logits, top %d, codebooks [%d %d] [%d %d]",
			ErrInk2Shape, rows, vocab, rowStride, k, pred.Rows, pred.Cols, succ.Rows, succ.Cols)
	}
	tp, err := d.PipelinePrecise(ink2DraftSource, "ink2_df_topk_partial")
	if err != nil {
		return err
	}
	sp, err := d.PipelinePrecise(ink2DraftSource, "ink2_df_select")
	if err != nil {
		return err
	}
	if tp.ThreadWidth() != 32 || sp.ThreadWidth() != 32 || tp.MaxThreads() < 256 || sp.MaxThreads() < 256 {
		return fmt.Errorf("metalgemm: inkling2 draft select needs 32-wide SIMD groups of 256 threads")
	}
	part := rows * ink2DFShards * k
	vals, ids := scratch, scratch.At(part)
	chunk := (vocab + ink2DFShards - 1) / ink2DFShards
	if err := d.Dispatch(tp, [3]int{ink2DFShards, rows, 1}, [3]int{256, 1, 1},
		logits.Span((rows-1)*rowStride+vocab), vals.Span(part), ids.Span(part),
		Ink2Const(ink2DFTopKShape{vocab: uint32(vocab), rowStride: uint32(rowStride), k: uint32(k), shards: ink2DFShards, chunk: uint32(chunk)})); err != nil {
		return err
	}
	return d.Dispatch(sp, [3]int{1, 1, 1}, [3]int{256, 1, 1},
		vals.Span(part), ids.Span(part), proj.Span(rows*rank), parg, sarg, anchorDev.Span(1), path.Span(rows), cand.Span(rows*k),
		Ink2Const(ink2DFSelectShape{rows: uint32(rows), k: uint32(k), shards: ink2DFShards, rank: uint32(rank), vocab: uint32(vocab),
			predShift: pshift, succShift: sshift, candidates: uint32(btoi(cand.Valid())), softcap: softcap}))
}
