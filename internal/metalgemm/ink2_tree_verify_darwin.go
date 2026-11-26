//go:build darwin && arm64 && cgo

package metalgemm

import (
	_ "embed"
	"fmt"
	"math"
)

//go:embed ink2_tree_verify.metal
var ink2TreeVerifySource string

//go:embed ink2_tree_verify_attend.metal
var ink2TreeVerifyAttendSource string

type ink2TreeQKVShape ink2VerifyQKVShape

type ink2TreeAttendShape ink2VerifyAttendShape

type ink2TreeConvShape ink2VerifyConvShape

type ink2TreeCommitShape struct {
	channels, past, rowLen, fromDevice uint32
	cols                               [ink2VerifyMaxKernel]uint32
}

type ink2TreePathShape struct {
	width, cap, pos0, posFromDevice uint32
	count, fromDevice, buffers, pad uint32
	rows                            [TreeVerifyMaxNodes]uint32
}

type ink2TreeGatherShape struct {
	width, inStride, count, fromDevice uint32
	rows                               [TreeVerifyMaxNodes]uint32
}

func (d *Ink2Device) treeKernel(name string) (Ink2Pipeline, error) {
	return d.PipelinePrecise(ink2TreeVerifySource, name)
}

// TreeVerifyQKVPost is VerifyQKVPost over a T-row token tree whose int32
// parent table is at parent (row 0 the root, -1; every other row's parent an
// earlier row): row t's global log-scale temperature takes position
// pos0+depth(t) and its k and v go to the cache row of pos0+t. A chain tree
// is VerifyQKVPost bit for bit.
func (d *Ink2Device) TreeVerifyQKVPost(q, r, k, v, qclip, rclip, kcache, vcache, posDev, parent Ink2Ref, p Ink2AttnParams, cap, pos0, T int) error {
	const limit = math.MaxInt32
	if p.Heads < 1 || p.KVHeads < 1 || p.HeadDim < 1 || p.RelLen < 0 || cap < 1 || cap > limit || T < 1 || T > cap ||
		T > TreeVerifyMaxNodes || pos0 < 0 || pos0 > limit-T || p.Heads > limit/p.HeadDim/T || p.Heads > limit/max(p.RelLen, 1)/T ||
		cap > limit/(p.KVHeads*p.HeadDim) || p.Heads > limit-2*p.KVHeads {
		return fmt.Errorf("%w: tree qkv post of %d rows, %d heads, %d kv heads, head dim %d, cache of %d",
			ErrInk2Shape, T, p.Heads, p.KVHeads, p.HeadDim, cap)
	}
	ok := parent.Valid() && (p.RelLen == 0 || (r.Valid() && rclip.Valid()))
	for _, ref := range []Ink2Ref{q, k, v, qclip, kcache, vcache} {
		ok = ok && ref.Valid()
	}
	if !ok {
		return fmt.Errorf("%w: tree qkv post without an operand", ErrInk2Shape)
	}
	if p.RelLen == 0 {
		r, rclip = Ink2Ref{}, Ink2Ref{}
	}
	pipe, err := d.treeKernel("ink2t_qkv_post")
	if err != nil {
		return err
	}
	lanes := pipe.ThreadWidth()
	if lanes < 1 || lanes > pipe.MaxThreads() {
		return fmt.Errorf("metalgemm: inkling2 tree qkv post: SIMD groups of %d threads", lanes)
	}
	kvDim := p.KVHeads * p.HeadDim
	return d.Dispatch(pipe, [3]int{p.Heads + 2*p.KVHeads, T, 1}, [3]int{lanes, 1, 1},
		q.Span(T*p.Heads*p.HeadDim), r.Span(T*p.Heads*p.RelLen), k.Span(T*kvDim), v.Span(T*kvDim),
		qclip.Span(p.Heads*p.HeadDim), rclip.Span(p.Heads*p.RelLen), kcache.Span(cap*kvDim), vcache.Span(cap*kvDim), posDev.Span(1),
		parent.Span(T),
		Ink2Const(ink2TreeQKVShape{
			heads: uint32(p.Heads), kvHeads: uint32(p.KVHeads), headDim: uint32(p.HeadDim), relLen: uint32(p.RelLen),
			cap: uint32(cap), lanes: uint32(lanes), global: uint32(btoi(p.Global)), posFromDevice: uint32(btoi(posDev.Valid())),
			pos0: int32(pos0), qkClip: p.QKClip, relClip: p.RelClip, eps: p.Eps, alpha: p.Alpha, floorTok: p.Floor,
		}))
}

// TreeVerifyAttend is VerifyAttend over a T-row token tree (parent as in
// TreeVerifyQKVPost): row t is the query at position pos0+depth(t) over the
// committed keys before pos0 and its own root path's rows, the depth-d
// ancestor a's keys at cache row pos0+a. scores is VerifyAttend's scratch,
// rows of heads*VerifyAttendSpan(devicePos, window, cap, pos0, T) floats. A
// chain tree is VerifyAttend bit for bit. A local layer's ring must hold
// window+T-1 rows so that no tree row overwrites a key still in a window.
func (d *Ink2Device) TreeVerifyAttend(q, r, sinks, relBias, kcache, vcache, scores Ink2Ref, scoreFloats int, out, posDev, parent Ink2Ref,
	p Ink2AttnParams, extent, window, cap, pos0, T int) error {
	const limit = math.MaxInt32
	if p.Heads < 1 || p.KVHeads < 1 || p.Heads%p.KVHeads != 0 || p.HeadDim < 1 || p.RelLen < 0 || extent < 0 || window < 0 ||
		cap < 1 || T < 1 || T > TreeVerifyMaxNodes || pos0 < 0 || pos0 > limit-T || p.Heads > limit/p.HeadDim/T ||
		extent > limit || window > limit {
		return fmt.Errorf("%w: tree attend of %d rows, %d heads over %d kv heads of %d dims, extent %d, window %d, cap %d",
			ErrInk2Shape, T, p.Heads, p.KVHeads, p.HeadDim, extent, window, cap)
	}
	kvDim := p.KVHeads * p.HeadDim
	if cap > limit/p.Heads || cap > limit/kvDim || (p.RelLen > 0 && (p.Heads > limit/p.RelLen/T || extent > limit/p.RelLen)) {
		return fmt.Errorf("%w: tree attend operands exceed 32-bit offsets", ErrInk2Shape)
	}
	if p.RelLen == 0 {
		extent = 0
	}
	if !q.Valid() || !sinks.Valid() || !kcache.Valid() || !vcache.Valid() || !scores.Valid() || !out.Valid() || !parent.Valid() ||
		(extent > 0 && (!r.Valid() || !relBias.Valid())) {
		return fmt.Errorf("%w: tree attend without an operand", ErrInk2Shape)
	}
	span := VerifyAttendSpan(posDev.Valid(), window, cap, pos0, T)
	perLaunch := min(T, scoreFloats/(p.Heads*span))
	if perLaunch < 1 {
		return fmt.Errorf("%w: tree attend scores scratch of %d floats, a row needs %d", ErrInk2Shape, scoreFloats, p.Heads*span)
	}
	pipe, err := d.PipelinePrecise(ink2TreeVerifyAttendSource, "ink2t_attend")
	if err != nil {
		return err
	}
	width := pipe.ThreadWidth()
	if width < 1 {
		return fmt.Errorf("metalgemm: inkling2 tree attend kernel reports no SIMD width")
	}
	threads := min(pipe.MaxThreads(), ink2AttendMaxThreads, ink2AttendMaxSIMD*width) / width * width
	if quads := (p.HeadDim + 3) / 4; quads > threads || quads > ink2AttendQTerms*width {
		return fmt.Errorf("%w: tree attend head of %d dims on %d threads of width %d", ErrInk2Shape, p.HeadDim, threads, width)
	}
	rSpan, relSpan := Ink2Ref{}.Span(0), Ink2Ref{}.Span(0)
	if extent > 0 {
		rSpan, relSpan = r.Span(T*p.Heads*p.RelLen), relBias.Span(extent*p.RelLen)
	}
	shape := ink2TreeAttendShape{
		heads: uint32(p.Heads), perKV: uint32(p.Heads / p.KVHeads), headDim: uint32(p.HeadDim), relLen: uint32(p.RelLen),
		kvDim: uint32(kvDim), extent: uint32(extent), window: uint32(window), cap: uint32(cap),
		threads: uint32(threads), width: uint32(width), span: uint32(span),
		posFromDevice: uint32(btoi(posDev.Valid())), pos0: int32(pos0), scale: p.Scale,
	}
	for tok0 := 0; tok0 < T; tok0 += perLaunch {
		rows := min(perLaunch, T-tok0)
		shape.tok0 = uint32(tok0)
		if err := d.Dispatch(pipe, [3]int{p.Heads, rows, 1}, [3]int{threads, 1, 1},
			q.Span(T*p.Heads*p.HeadDim), rSpan, sinks.Span(T*p.Heads), relSpan,
			kcache.Span(cap*kvDim), vcache.Span(cap*kvDim), scores.Span(rows*p.Heads*span), out.Span(T*p.Heads*p.HeadDim),
			posDev.Span(1), parent.Span(T), Ink2Const(shape)); err != nil {
			return err
		}
	}
	return nil
}

// TreeVerifyConv is VerifyConv over a T-row token tree (parent as in
// TreeVerifyQKVPost): row t convolves its input with its nearest ancestors'
// inputs, then hist's entries. hist is read only; a valid log ([n][kn-1+T],
// VerifyConv's layout) is what TreeVerifyConvCommit rebuilds the history of
// the kept path from. A chain tree gives VerifyConv's outputs and log.
func (d *Ink2Device) TreeVerifyConv(x, w, dyn, hist, log, out, resid, parent Ink2Ref, n, heads, kn int, norm float32, T int) error {
	if !x.Valid() || !w.Valid() || !hist.Valid() || !out.Valid() || !parent.Valid() || n < 1 || kn < 2 || kn > ink2VerifyMaxKernel ||
		heads < 1 || T < 1 || T > TreeVerifyMaxNodes || T > math.MaxInt32/n || n > math.MaxInt32/(kn-1+T) ||
		(dyn.Valid() && n%heads != 0) || ink2OpsOverlap(x, T*n, out, T*n) || ink2OpsOverlap(resid, T*n, out, T*n) {
		return fmt.Errorf("%w: tree conv of %d rows of %d channels, %d heads, kernel %d", ErrInk2Shape, T, n, heads, kn)
	}
	p, err := d.treeKernel("ink2t_conv")
	if err != nil {
		return err
	}
	perHead, taps := n, 0
	if dyn.Valid() {
		perHead, taps = n/heads, T*heads*kn
	}
	logSpan := 0
	if log.Valid() {
		logSpan = n * (kn - 1 + T)
	}
	return d.DispatchThreads(p, [3]int{n, 1, 1}, [3]int{min(n, ink2OpsThreads, p.MaxThreads()), 1, 1},
		x.Span(T*n), w.Span(n*kn), dyn.Span(taps), hist.Span(n*(kn-1)), out.Span(T*n), resid.Span(T*n), log.Span(logSpan),
		parent.Span(T),
		Ink2Const(ink2TreeConvShape{n: uint32(n), perHead: uint32(perHead), kn: uint32(kn), tokens: uint32(T),
			dynamic: uint32(btoi(dyn.Valid())), residual: uint32(btoi(resid.Valid())), logged: uint32(btoi(log.Valid())), norm: norm}))
}

// TreeVerifyConvCommit encodes hist[c][j] = log[c][cols[j]] over channels*past
// entries of log rows of L floats: TreeConvCommitCols of the kept path gives
// the history after it. A valid colsDev supplies the past columns as int32
// instead of cols.
func (d *Ink2Device) TreeVerifyConvCommit(hist, log, colsDev Ink2Ref, channels, past, L int, cols []int) error {
	bad := !hist.Valid() || !log.Valid() || channels < 1 || past < 1 || past >= ink2VerifyMaxKernel || L < past ||
		channels > math.MaxInt32/L || (!colsDev.Valid() && len(cols) != past)
	shape := ink2TreeCommitShape{channels: uint32(channels), past: uint32(past), rowLen: uint32(L), fromDevice: uint32(btoi(colsDev.Valid()))}
	for j, c := range cols {
		if colsDev.Valid() || j >= len(shape.cols) || c < 0 || c >= L {
			bad = true
			break
		}
		shape.cols[j] = uint32(c)
	}
	if bad {
		return fmt.Errorf("%w: tree conv commit of %d channels, past %d, columns %v of rows of %d", ErrInk2Shape, channels, past, cols, L)
	}
	p, err := d.treeKernel("ink2t_conv_commit")
	if err != nil {
		return err
	}
	n := channels * past
	return d.DispatchThreads(p, [3]int{n, 1, 1}, [3]int{min(n, ink2OpsThreads, p.MaxThreads()), 1, 1},
		hist.Span(n), log.Span(channels*L), colsDev.Span(past), Ink2Const(shape))
}

func ink2TreePathRows(path []int, rows []uint32) bool {
	if len(path) > len(rows) {
		return false
	}
	for i, r := range path {
		if (i == 0 && r != 0) || (i > 0 && r <= path[i-1]) {
			return false
		}
		rows[i] = uint32(r)
	}
	return true
}

// TreeVerifyKVCompact moves a kept root path's cache rows to contiguous
// positions in each valid ring of cap rows of width floats: the row of
// position pos0+path[i] to pos0+i (posDev as in VerifyQKVPost). A valid
// pathDev, 1+TreeVerifyMaxNodes ints, supplies the path as int32 [count,
// rows...] instead of path. The path must increase from the root, which
// every kept tree path does.
func (d *Ink2Device) TreeVerifyKVCompact(kcache, vcache, pathDev, posDev Ink2Ref, width, cap, pos0 int, path []int) error {
	shape := ink2TreePathShape{width: uint32(width), cap: uint32(cap), pos0: uint32(pos0), posFromDevice: uint32(btoi(posDev.Valid())),
		count: uint32(len(path)), fromDevice: uint32(btoi(pathDev.Valid())), buffers: 1 + uint32(btoi(vcache.Valid()))}
	if !kcache.Valid() || width < 1 || cap < max(len(path), 1) || cap > math.MaxInt32/width ||
		pos0 < 0 || pos0 > math.MaxInt32-TreeVerifyMaxNodes || (pathDev.Valid() && path != nil) ||
		!ink2TreePathRows(path, shape.rows[:]) || ink2OpsOverlap(kcache, cap*width, vcache, cap*width) {
		return fmt.Errorf("%w: tree kv compact of a %d-row path, rows of %d in a ring of %d", ErrInk2Shape, len(path), width, cap)
	}
	if !pathDev.Valid() && len(path) <= 1 {
		return nil
	}
	p, err := d.treeKernel("ink2t_kv_compact")
	if err != nil {
		return err
	}
	if !vcache.Valid() {
		vcache = kcache
	}
	return d.DispatchThreads(p, [3]int{width, int(shape.buffers), 1}, [3]int{min(width, ink2OpsThreads, p.MaxThreads()), 1, 1},
		kcache.Span(cap*width), vcache.Span(cap*width), pathDev.Span(1+TreeVerifyMaxNodes), posDev.Span(1), Ink2Const(shape))
}

// TreeVerifyGatherRows encodes out[i] = in[path[i]] for the rows of a kept
// path, rows of width floats, in's inStride apart and out's packed; pathDev as
// in TreeVerifyKVCompact. rows bounds the rows in holds.
func (d *Ink2Device) TreeVerifyGatherRows(in, out, pathDev Ink2Ref, width, inStride, rows int, path []int) error {
	shape := ink2TreeGatherShape{width: uint32(width), inStride: uint32(inStride), count: uint32(len(path)), fromDevice: uint32(btoi(pathDev.Valid()))}
	if !in.Valid() || !out.Valid() || width < 1 || inStride < width || rows < 1 || rows > TreeVerifyMaxNodes ||
		rows > math.MaxInt32/inStride || (pathDev.Valid() && path != nil) || !ink2TreePathRows(path, shape.rows[:]) ||
		ink2OpsOverlap(in, (rows-1)*inStride+width, out, rows*width) {
		return fmt.Errorf("%w: tree gather of a %d-row path, rows of %d", ErrInk2Shape, len(path), width)
	}
	for _, r := range path {
		if r >= rows {
			return fmt.Errorf("%w: tree gather row %d of %d", ErrInk2Shape, r, rows)
		}
	}
	p, err := d.treeKernel("ink2t_gather_rows")
	if err != nil {
		return err
	}
	return d.DispatchThreads(p, [3]int{width, rows, 1}, [3]int{min(width, ink2OpsThreads, p.MaxThreads()), 1, 1},
		in.Span((rows-1)*inStride+width), out.Span(rows*width), pathDev.Span(1+TreeVerifyMaxNodes), Ink2Const(shape))
}
