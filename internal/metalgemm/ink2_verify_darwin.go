//go:build darwin && arm64 && cgo

package metalgemm

import (
	_ "embed"
	"fmt"
	"math"
	"sync"
)

//go:embed ink2_verify.metal
var ink2VerifySource string

//go:embed ink2_verify_attend.metal
var ink2VerifyAttendSource string

//go:embed ink2_verify_rows.metal
var ink2VerifyRows string

// The rows kernels instantiate the packed decoders, so they compile with them
// and with the flags the single-row GEMV compiles with.
var ink2VerifyRowsSource = sync.OnceValue(func() string {
	return packedCodebooks + packedDequant + packedGEMV + ink2VerifyRows
})

// Past this many rows the 9-row tiles of ink2v_rows_ spread wider over the GPU.
const ink2VerifyPairMax = 16

const (
	// Activation rows one rows threadgroup covers; INK2V_ROWS_TILE.
	ink2VerifyRowsTile = 9
	// INK2V_MAX_KN.
	ink2VerifyMaxKernel = 8
)

type ink2VerifyRowsShape struct{ tokens, xStride, yStride uint32 }

type ink2VerifyExpertsShape struct{ npairs, xStride, xDiv uint32 }

// INK2V_GROUP_MAX: the pairs one grouping covers.
const ink2VerifyGroupMax = 1024

// VerifyExpertsTable is the int count of VerifyExperts' grouping table.
func VerifyExpertsTable(npairs int) int { return 1 + 5*npairs }

type ink2VerifyMixShape struct {
	n, states, tokens, norm, weighted uint32
	eps, invSqrtN                     float32
}

type ink2VerifyCtrlShape struct{ n, rank, tokens uint32 }

type ink2VerifyConvShape struct {
	n, perHead, kn, tokens    uint32
	dynamic, residual, logged uint32
	norm                      float32
}

type ink2VerifyConvPairShape struct {
	n, perHead, kn, tokens, dynamic uint32
	norm0, norm1                    float32
}

type ink2VerifyCommitShape struct{ channels, past, rowLen, keep, keepFromDevice uint32 }

type ink2VerifyQKVShape struct {
	heads, kvHeads, headDim, relLen uint32
	cap, lanes                      uint32
	global, posFromDevice           uint32
	pos0                            int32
	qkClip, relClip, eps            float32
	alpha, floorTok                 float32
}

type ink2VerifyAttendShape struct {
	heads, perKV, headDim, relLen uint32
	kvDim, extent, window, cap    uint32
	threads, width                uint32
	span, tok0, posFromDevice     uint32
	pos0                          int32
	scale                         float32
}

type ink2VerifyPairShape struct{ npairs, width, div, stride uint32 }

func (d *Ink2Device) verifyKernel(name string, reducing bool) (Ink2Pipeline, error) {
	p, err := d.PipelinePrecise(ink2VerifySource, name)
	if err != nil {
		return Ink2Pipeline{}, err
	}
	if reducing && (p.ThreadWidth() != 32 || p.MaxThreads() < ink2OpsThreads) {
		return Ink2Pipeline{}, fmt.Errorf("metalgemm: inkling2 kernel %s needs 32-wide SIMD groups of %d threads", name, ink2OpsThreads)
	}
	return p, nil
}

// VerifyRows encodes y[t] = W x[t] for rows activation rows in one pass over
// the weight, row t at x + t*xStride and its output at y + t*yStride. A row's
// bits do not depend on rows: every count runs the one tile kernel, and a row
// keeps its own accumulator in MatVec's order.
func (d *Ink2Device) VerifyRows(w Ink2Weight, x Ink2Ref, xStride int, y Ink2Ref, yStride, rows int) error {
	if !w.Buf.Valid() || w.ByteOff < 0 || w.Rows < 1 || w.Rows > math.MaxInt32 || w.Cols < 1 || w.Cols > math.MaxInt32 ||
		w.Experts < 1 || int(w.Type) >= len(ink2VerifyRowsKernel) || ink2VerifyRowsKernel[w.Type] == "" {
		return fmt.Errorf("%w: weight type %d [%d %d] x%d", ErrInk2Shape, w.Type, w.Rows, w.Cols, w.Experts)
	}
	if !x.Valid() || !y.Valid() || rows < 1 || xStride < w.Cols || yStride < w.Rows ||
		rows > math.MaxInt32/max(xStride, yStride) || ink2OpsOverlap(x, (rows-1)*xStride+w.Cols, y, (rows-1)*yStride+w.Rows) {
		return fmt.Errorf("%w: verify rows of %d rows, strides %d and %d", ErrInk2Shape, rows, xStride, yStride)
	}
	block, blockBytes, err := packedBlock(w.Type)
	if err != nil {
		return err
	}
	// The block kernels load the activation as float4 vectors.
	if w.Cols%block != 0 || (block > 1 && (x.Off%4 != 0 || xStride%4 != 0)) {
		return fmt.Errorf("%w: type %d weight of %d columns with the activation at offset %d, stride %d",
			ErrInk2Shape, w.Type, w.Cols, x.Off, xStride)
	}
	// Which kernel runs never changes a row's bits.
	name, grid := ink2VerifyPairRowsKernel[w.Type], [3]int{(w.Rows + 7) / 8, 1, 1}
	if rows > ink2VerifyPairMax {
		name, grid = ink2VerifyRowsKernel[w.Type], [3]int{(w.Rows + 3) / 4, (rows + ink2VerifyRowsTile - 1) / ink2VerifyRowsTile, 1}
	}
	p, err := d.verifyRowsPipeline(name)
	if err != nil {
		return err
	}
	d.profileDetail("%dx%d rows=%d", w.Rows, w.Cols, rows)
	bytes := w.Rows * (w.Cols / block * blockBytes)
	args := append(w.args(x.Span((rows-1)*xStride+w.Cols), y.Span((rows-1)*yStride+w.Rows), bytes),
		Ink2Const(ink2VerifyRowsShape{uint32(rows), uint32(xStride), uint32(yStride)}))
	return d.Dispatch(p, grid, [3]int{128, 1, 1}, args...)
}

func (d *Ink2Device) verifyRowsPipeline(name string) (Ink2Pipeline, error) {
	p, err := d.Pipeline(ink2VerifyRowsSource(), name)
	if err != nil {
		return Ink2Pipeline{}, err
	}
	if p.width != 32 || p.maxThreads < 128 {
		return Ink2Pipeline{}, fmt.Errorf("metalgemm: inkling2 kernel %s needs 32-wide SIMD groups of 128 threads", name)
	}
	return p, nil
}

// VerifyExperts encodes, for npairs routed pairs i, y[i] = W[ids[i]] x[i/xDiv]
// with x row r at x + r*xStride and y[i] at y + i*w.Rows, then the optional
// per-expert scale, reading each routed expert once for all its pairs. Pair i
// is ExpertMatVec's slot i bit for bit, however the chunk routes. It declines
// (false) a weight type without a grouped kernel.
// tab holds VerifyExpertsTable(npairs) ints, the grouping of this call.
func (d *Ink2Device) VerifyExperts(w Ink2Weight, ids, x Ink2Ref, xDiv, xStride int, y Ink2Ref, npairs int, scale, tab Ink2Ref) (bool, error) {
	if int(w.Type) >= len(ink2VerifyExpertsKernel) || ink2VerifyExpertsKernel[w.Type] == "" {
		return false, nil
	}
	if !w.Buf.Valid() || w.ByteOff < 0 || w.Rows < 1 || w.Rows > math.MaxInt32 || w.Cols < 1 || w.Cols > math.MaxInt32 || w.Experts < 1 {
		return false, fmt.Errorf("%w: weight type %d [%d %d] x%d", ErrInk2Shape, w.Type, w.Rows, w.Cols, w.Experts)
	}
	if !ids.Valid() || !x.Valid() || !y.Valid() || !tab.Valid() || npairs < 1 || npairs > ink2VerifyGroupMax || xDiv < 1 ||
		xStride < w.Cols || xStride > math.MaxUint32 || npairs > math.MaxInt32/max(w.Rows, xStride) {
		return false, fmt.Errorf("%w: verify experts of %d pairs, input stride %d every %d pairs", ErrInk2Shape, npairs, xStride, xDiv)
	}
	block, blockBytes, err := packedBlock(w.Type)
	if err != nil {
		return false, err
	}
	if w.Cols%block != 0 || x.Off%4 != 0 || xStride%4 != 0 {
		return false, fmt.Errorf("%w: type %d weight of %d columns with the activation at offset %d, stride %d",
			ErrInk2Shape, w.Type, w.Cols, x.Off, xStride)
	}
	gp, err := d.Pipeline(ink2VerifyRowsSource(), "ink2v_group_pairs")
	if err != nil {
		return false, err
	}
	p, err := d.verifyRowsPipeline(ink2VerifyExpertsKernel[w.Type])
	if err != nil {
		return false, err
	}
	shape := Ink2Const(ink2VerifyExpertsShape{uint32(npairs), uint32(xStride), uint32(xDiv)})
	tabSpan := tab.Span(VerifyExpertsTable(npairs))
	if err := d.Dispatch(gp, [3]int{1, 1, 1}, [3]int{min(npairs, gp.MaxThreads()), 1, 1}, ids.Span(npairs), tabSpan, shape); err != nil {
		return false, err
	}
	d.profileDetail("%dx%d pairs=%d of %d", w.Rows, w.Cols, npairs, w.Experts)
	xRows := (npairs-1)/xDiv + 1
	args := append(w.args(x.Span((xRows-1)*xStride+w.Cols), y.Span(npairs*w.Rows), w.Experts*w.Rows*(w.Cols/block*blockBytes)),
		tabSpan, shape)
	if err := d.Dispatch(p, [3]int{(w.Rows + 7) / 8, npairs, 1}, [3]int{128, 1, 1}, args...); err != nil {
		return false, err
	}
	if !scale.Valid() {
		return true, nil
	}
	return true, d.ExpertScale(y, ids, scale, npairs, w.Rows)
}

// VerifyMix encodes Mix for each of T token rows: states are [T][n] and out is
// query-major [nq][T][n]. Row t is Mix of that row.
func (d *Ink2Device) VerifyMix(states []Ink2Ref, queries Ink2Ref, nq int, norms []Ink2Ref, eps float32, out Ink2Ref, n, T int) error {
	ns := len(states)
	if ns < 1 || ns > ink2OpsMaxStates || nq < 1 || nq > ink2OpsMaxRows || len(norms) > ink2OpsMaxRows ||
		n < 1 || n > ink2OpsMaxWidth || T < 1 || T > math.MaxInt32/(nq*n) || !queries.Valid() || !out.Valid() ||
		ink2OpsOverlap(queries, nq*n, out, nq*T*n) {
		return fmt.Errorf("%w: verify mix of %d states by %d queries, %d rows of %d", ErrInk2Shape, ns, nq, T, n)
	}
	p, err := d.verifyKernel("ink2v_mix", true)
	if err != nil {
		return err
	}
	args := make([]Ink2Arg, 0, ink2OpsMaxStates+ink2OpsMaxRows+3)
	for i := range ink2OpsMaxStates {
		var s Ink2Ref
		if i < ns {
			if s = states[i]; !s.Valid() || ink2OpsOverlap(s, T*n, out, nq*T*n) {
				return fmt.Errorf("%w: verify mix state %d is absent or overlaps the output", ErrInk2Shape, i)
			}
		}
		args = append(args, s.Span(T*n))
	}
	args = append(args, queries.Span(nq*n))
	shape := ink2VerifyMixShape{n: uint32(n), states: uint32(ns), tokens: uint32(T), norm: uint32(btoi(norms != nil)), eps: eps,
		invSqrtN: float32(1 / math.Sqrt(float64(n)))}
	for q := range ink2OpsMaxRows {
		var w Ink2Ref
		if q < nq && q < len(norms) {
			w = norms[q]
		}
		if w.Valid() {
			shape.weighted |= 1 << q
		}
		args = append(args, w.Span(n))
	}
	args = append(args, out.Span(nq*T*n), Ink2Const(shape))
	return d.Dispatch(p, [3]int{nq, T, 1}, [3]int{ink2OpsThreads, 1, 1}, args...)
}

// VerifyControl encodes len(ctl.Cur) loop controllers over T rows, one launch
// per stage: g[j] ([T][rank]) = Down_j [rms(cur_j), rms(msg_j)], then
// out[j] ([T][n]) = cur_j - tanh(F_j g_j)*cur_j + tanh(G_j g_j)*msg_j. Row t of
// controller j is Control of that row without its norm.
func (d *Ink2Device) VerifyControl(ctl Ink2Controls, g, out []Ink2Ref, n, rank, T int) error {
	nj := len(ctl.Cur)
	bad := nj < 1 || nj > ink2OpsMaxRows || n < 1 || n > ink2OpsMaxWidth || rank < 1 || rank > math.MaxInt32/(2*n) ||
		T < 1 || T > math.MaxInt32/max(n, rank)
	for _, set := range [][]Ink2Ref{ctl.Cur, ctl.Msg, ctl.Down, ctl.Gain, ctl.Forget, g, out} {
		bad = bad || len(set) != nj
		for _, r := range set {
			bad = bad || !r.Valid()
		}
	}
	if bad {
		return fmt.Errorf("%w: verify control of %d controllers, %d rows of %d, rank %d", ErrInk2Shape, nj, T, n, rank)
	}
	down, err := d.verifyKernel("ink2v_ctrl_down", true)
	if err != nil {
		return err
	}
	apply, err := d.verifyKernel("ink2v_ctrl_apply", false)
	if err != nil {
		return err
	}
	shape := Ink2Const(ink2VerifyCtrlShape{uint32(n), uint32(rank), uint32(T)})
	args := ink2PBind(ink2PBind(make([]Ink2Arg, 0, 17), ctl.Cur, T*n), ctl.Msg, T*n)
	args = append(ink2PBind(ink2PBind(args, ctl.Down, rank*2*n), g, T*rank), shape)
	if err := d.Dispatch(down, [3]int{rank, T, nj}, [3]int{ink2OpsThreads, 1, 1}, args...); err != nil {
		return err
	}
	args = ink2PBind(ink2PBind(make([]Ink2Arg, 0, 25), ctl.Cur, T*n), ctl.Msg, T*n)
	args = ink2PBind(ink2PBind(args, ctl.Gain, n*rank), ctl.Forget, n*rank)
	args = append(ink2PBind(ink2PBind(args, g, T*rank), out, T*n), shape)
	return d.DispatchThreads(apply, [3]int{n, T, nj}, [3]int{min(n, ink2OpsThreads, apply.MaxThreads()), 1, 1}, args...)
}

// VerifyConv encodes Conv along T token rows in token order and leaves the
// final history in hist; dyn is [T][heads][kn]. A valid log ([n][kn-1+T])
// receives each channel's entry history followed by its T inputs. out must not
// overlap x or resid.
func (d *Ink2Device) VerifyConv(x, w, dyn, hist, log, out, resid Ink2Ref, n, heads, kn int, norm float32, T int) error {
	if !x.Valid() || !w.Valid() || !hist.Valid() || !out.Valid() || n < 1 || kn < 2 || kn > ink2VerifyMaxKernel || heads < 1 ||
		T < 1 || T > math.MaxInt32/n || n > math.MaxInt32/(kn-1+T) || (dyn.Valid() && n%heads != 0) ||
		ink2OpsOverlap(x, T*n, out, T*n) || ink2OpsOverlap(resid, T*n, out, T*n) {
		return fmt.Errorf("%w: verify conv of %d rows of %d channels, %d heads, kernel %d", ErrInk2Shape, T, n, heads, kn)
	}
	p, err := d.verifyKernel("ink2v_conv", false)
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
		Ink2Const(ink2VerifyConvShape{n: uint32(n), perHead: uint32(perHead), kn: uint32(kn), tokens: uint32(T),
			dynamic: uint32(btoi(dyn.Valid())), residual: uint32(btoi(resid.Valid())), logged: uint32(btoi(log.Valid())), norm: norm}))
}

// VerifyConvPair encodes two logged VerifyConv calls of one geometry (no
// residual) in one launch; every slice holds the two convs' operands.
func (d *Ink2Device) VerifyConvPair(x, w, dyn, hist, log, out []Ink2Ref, norm []float32, n, heads, kn, T int) error {
	if len(x) != 2 || len(w) != 2 || len(dyn) != 2 || len(hist) != 2 || len(log) != 2 || len(out) != 2 || len(norm) != 2 ||
		n < 1 || kn < 2 || kn > ink2VerifyMaxKernel || heads < 1 || n%heads != 0 || T < 1 || T > math.MaxInt32/n ||
		n > math.MaxInt32/(kn-1+T) {
		return fmt.Errorf("%w: verify conv pair of %d rows of %d channels, %d heads, kernel %d", ErrInk2Shape, T, n, heads, kn)
	}
	shape := ink2VerifyConvPairShape{n: uint32(n), perHead: uint32(n / heads), kn: uint32(kn), tokens: uint32(T), norm0: norm[0], norm1: norm[1]}
	for j := range 2 {
		if !x[j].Valid() || !w[j].Valid() || !hist[j].Valid() || !log[j].Valid() || !out[j].Valid() || ink2OpsOverlap(x[j], T*n, out[j], T*n) {
			return fmt.Errorf("%w: verify conv pair is missing an operand of conv %d", ErrInk2Shape, j)
		}
		if dyn[j].Valid() {
			shape.dynamic |= 1 << j
		}
	}
	p, err := d.verifyKernel("ink2v_conv_pair", false)
	if err != nil {
		return err
	}
	taps := [2]int{}
	for j := range 2 {
		if dyn[j].Valid() {
			taps[j] = T * heads * kn
		}
	}
	return d.DispatchThreads(p, [3]int{n, 2, 1}, [3]int{min(n, ink2OpsThreads, p.MaxThreads()), 1, 1},
		x[0].Span(T*n), x[1].Span(T*n), w[0].Span(n*kn), w[1].Span(n*kn), dyn[0].Span(taps[0]), dyn[1].Span(taps[1]),
		hist[0].Span(n*(kn-1)), hist[1].Span(n*(kn-1)), out[0].Span(T*n), out[1].Span(T*n),
		log[0].Span(n*(kn-1+T)), log[1].Span(n*(kn-1+T)), Ink2Const(shape))
}

// VerifyConvCommit encodes hist[c][j] = log[c][keep+j] over channels*past
// entries of log rows of L floats: the history after the first keep tokens of
// the logged chunk. A valid keepDev supplies keep as the int32 it holds.
func (d *Ink2Device) VerifyConvCommit(hist, log, keepDev Ink2Ref, channels, past, L, keep int) error {
	if !hist.Valid() || !log.Valid() || channels < 1 || past < 1 || L < past || channels > math.MaxInt32/L ||
		(!keepDev.Valid() && (keep < 0 || keep+past > L)) {
		return fmt.Errorf("%w: conv commit of %d channels, past %d, keep %d of rows of %d", ErrInk2Shape, channels, past, keep, L)
	}
	p, err := d.verifyKernel("ink2v_conv_commit", false)
	if err != nil {
		return err
	}
	n := channels * past
	return d.DispatchThreads(p, [3]int{n, 1, 1}, [3]int{min(n, ink2OpsThreads, p.MaxThreads()), 1, 1},
		hist.Span(n), log.Span(channels*L), keepDev.Span(1),
		Ink2Const(ink2VerifyCommitShape{uint32(channels), uint32(past), uint32(L), uint32(max(keep, 0)), uint32(btoi(keepDev.Valid()))}))
}

// VerifyQKVPost encodes QKVPost (without its folded convs) of T token rows at
// positions pos0+t, or the int32 at a valid posDev plus t, appending k and v
// into the cache rows of those positions. T must not exceed cap.
func (d *Ink2Device) VerifyQKVPost(q, r, k, v, qclip, rclip, kcache, vcache, posDev Ink2Ref, p Ink2AttnParams, cap, pos0, T int) error {
	const limit = math.MaxInt32
	if p.Heads < 1 || p.KVHeads < 1 || p.HeadDim < 1 || p.RelLen < 0 || cap < 1 || cap > limit || T < 1 || T > cap ||
		pos0 < 0 || pos0 > limit-T || p.Heads > limit/p.HeadDim/T || p.Heads > limit/max(p.RelLen, 1)/T ||
		cap > limit/(p.KVHeads*p.HeadDim) || p.Heads > limit-2*p.KVHeads {
		return fmt.Errorf("%w: verify qkv post of %d rows, %d heads, %d kv heads, head dim %d, cache of %d",
			ErrInk2Shape, T, p.Heads, p.KVHeads, p.HeadDim, cap)
	}
	ok := p.RelLen == 0 || (r.Valid() && rclip.Valid())
	for _, ref := range []Ink2Ref{q, k, v, qclip, kcache, vcache} {
		ok = ok && ref.Valid()
	}
	if !ok {
		return fmt.Errorf("%w: verify qkv post without an operand", ErrInk2Shape)
	}
	if p.RelLen == 0 {
		r, rclip = Ink2Ref{}, Ink2Ref{}
	}
	pipe, err := d.verifyKernel("ink2v_qkv_post", false)
	if err != nil {
		return err
	}
	// Each head reduces across one SIMD group, so that is the threadgroup.
	lanes := pipe.ThreadWidth()
	if lanes < 1 || lanes > pipe.MaxThreads() {
		return fmt.Errorf("metalgemm: inkling2 verify qkv post: SIMD groups of %d threads", lanes)
	}
	kvDim := p.KVHeads * p.HeadDim
	return d.Dispatch(pipe, [3]int{p.Heads + 2*p.KVHeads, T, 1}, [3]int{lanes, 1, 1},
		q.Span(T*p.Heads*p.HeadDim), r.Span(T*p.Heads*p.RelLen), k.Span(T*kvDim), v.Span(T*kvDim),
		qclip.Span(p.Heads*p.HeadDim), rclip.Span(p.Heads*p.RelLen), kcache.Span(cap*kvDim), vcache.Span(cap*kvDim), posDev.Span(1),
		Ink2Const(ink2VerifyQKVShape{
			heads: uint32(p.Heads), kvHeads: uint32(p.KVHeads), headDim: uint32(p.HeadDim), relLen: uint32(p.RelLen),
			cap: uint32(cap), lanes: uint32(lanes), global: uint32(btoi(p.Global)), posFromDevice: uint32(btoi(posDev.Valid())),
			pos0: int32(pos0), qkClip: p.QKClip, relClip: p.RelClip, eps: p.Eps, alpha: p.Alpha, floorTok: p.Floor,
		}))
}

// VerifyAttendSpan is the most keys a query of a T-row chunk scores: the row
// length VerifyAttend lays its scores scratch out in. A device position is
// unknown here, so it takes the bound of the cache.
func VerifyAttendSpan(devicePos bool, window, cap, pos0, T int) int {
	span := cap
	if !devicePos {
		span = min(span, pos0+T)
	}
	if window > 0 {
		span = min(span, window)
	}
	return span
}

// VerifyAttend encodes Attend for T queries at positions pos0+t (posDev as in
// VerifyQKVPost) over one KV slot into out ([T][heads][headDim]); q, r and
// sinks are [T] rows. Every query scores the keys of its own and earlier
// positions, which VerifyQKVPost appended for the whole chunk. scores is
// scratch of scoreFloats floats: the chunk runs in token ranges of as many rows
// as it holds, heads*VerifyAttendSpan floats each.
func (d *Ink2Device) VerifyAttend(q, r, sinks, relBias, kcache, vcache, scores Ink2Ref, scoreFloats int, out, posDev Ink2Ref,
	p Ink2AttnParams, extent, window, cap, pos0, T int) error {
	const limit = math.MaxInt32
	if p.Heads < 1 || p.KVHeads < 1 || p.Heads%p.KVHeads != 0 || p.HeadDim < 1 || p.RelLen < 0 || extent < 0 || window < 0 ||
		cap < 1 || T < 1 || pos0 < 0 || pos0 > limit-T || p.Heads > limit/p.HeadDim/T || extent > limit || window > limit {
		return fmt.Errorf("%w: verify attend of %d rows, %d heads over %d kv heads of %d dims, extent %d, window %d, cap %d",
			ErrInk2Shape, T, p.Heads, p.KVHeads, p.HeadDim, extent, window, cap)
	}
	kvDim := p.KVHeads * p.HeadDim
	if cap > limit/p.Heads || cap > limit/kvDim ||
		(p.RelLen > 0 && (p.Heads > limit/p.RelLen/T || extent > limit/p.RelLen)) {
		return fmt.Errorf("%w: verify attend operands exceed 32-bit offsets", ErrInk2Shape)
	}
	if p.RelLen == 0 {
		extent = 0
	}
	if !q.Valid() || !sinks.Valid() || !kcache.Valid() || !vcache.Valid() || !scores.Valid() || !out.Valid() ||
		(extent > 0 && (!r.Valid() || !relBias.Valid())) {
		return fmt.Errorf("%w: verify attend without an operand", ErrInk2Shape)
	}
	span := VerifyAttendSpan(posDev.Valid(), window, cap, pos0, T)
	perLaunch := min(T, scoreFloats/(p.Heads*span))
	if perLaunch < 1 {
		return fmt.Errorf("%w: verify attend scores scratch of %d floats, a row needs %d", ErrInk2Shape, scoreFloats, p.Heads*span)
	}
	// Fast math off: the softmax leans on exp's precise form.
	pipe, err := d.PipelinePrecise(ink2VerifyAttendSource, "ink2v_attend")
	if err != nil {
		return err
	}
	width := pipe.ThreadWidth()
	if width < 1 {
		return fmt.Errorf("metalgemm: inkling2 verify attend kernel reports no SIMD width")
	}
	threads := min(pipe.MaxThreads(), ink2AttendMaxThreads, ink2AttendMaxSIMD*width) / width * width
	if quads := (p.HeadDim + 3) / 4; quads > threads || quads > ink2AttendQTerms*width {
		return fmt.Errorf("%w: verify attend head of %d dims on %d threads of width %d", ErrInk2Shape, p.HeadDim, threads, width)
	}
	rSpan, relSpan := Ink2Ref{}.Span(0), Ink2Ref{}.Span(0)
	if extent > 0 {
		rSpan, relSpan = r.Span(T*p.Heads*p.RelLen), relBias.Span(extent*p.RelLen)
	}
	shape := ink2VerifyAttendShape{
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
			posDev.Span(1), Ink2Const(shape)); err != nil {
			return err
		}
	}
	return nil
}

// VerifyRepeatRows encodes out[t*k+j] = x[t] for rows rows of n floats.
func (d *Ink2Device) VerifyRepeatRows(x, out Ink2Ref, n, rows, k int) error {
	if !x.Valid() || !out.Valid() || n < 1 || rows < 1 || k < 1 || rows > math.MaxInt32/k/n ||
		ink2OpsOverlap(x, rows*n, out, rows*k*n) {
		return fmt.Errorf("%w: repeat of %d rows of %d by %d", ErrInk2Shape, rows, n, k)
	}
	p, err := d.verifyKernel("ink2v_repeat_rows", false)
	if err != nil {
		return err
	}
	return d.DispatchThreads(p, [3]int{n, rows * k, 1}, [3]int{min(n, ink2OpsThreads, p.MaxThreads()), 1, 1},
		x.Span(rows*n), out.Span(rows*k*n), Ink2Const(ink2VerifyPairShape{uint32(rows * k), uint32(n), uint32(k), uint32(n)}))
}
