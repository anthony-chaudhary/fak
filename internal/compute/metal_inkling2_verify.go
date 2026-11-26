//go:build darwin && arm64 && cgo

package compute

import (
	"fmt"
	"math"

	"github.com/anthony-chaudhary/fak/internal/metalgemm"
)

var (
	_ Inkling2VerifyDevice = (*metalBackend)(nil)
	_ Inkling2VerifyMoE    = (*metalBackend)(nil)
)

// A verify row's bits never depend on the width of its chunk: every op below
// runs the decode kernel's arithmetic with one grid coordinate per token row.

// metalInk2VerifyScoreBytes bounds the attention scores scratch; a chunk whose
// rows do not fit is attended in token ranges over it.
const metalInk2VerifyScoreBytes = 64 << 20

// metalInk2VerifyScratch is the attention scores scratch of the verify chunks;
// guarded by metalMu.
var metalInk2VerifyScratch struct {
	buf    metalgemm.Ink2Buf
	floats int
}

// metalInk2VerifyTable is the expert grouping of the verify expert ops, each
// launch regrouping before it reads; guarded by metalMu.
var metalInk2VerifyTable struct {
	buf  metalgemm.Ink2Buf
	ints int
}

// metalInk2VerifyScores returns a scores scratch for a T-row attend of heads
// query heads over span keys, and its size; caller holds metalMu.
func metalInk2VerifyScores(d *metalgemm.Ink2Device, heads, span, T int) (metalgemm.Ink2Ref, int, error) {
	row := heads * span
	if row < 1 || T < 1 {
		return metalgemm.Ink2Ref{}, 0, fmt.Errorf("%w: scores scratch for %d rows of %d heads by %d keys", metalgemm.ErrInk2Shape, T, heads, span)
	}
	need := row * max(1, min(T, metalInk2VerifyScoreBytes/4/row))
	s := &metalInk2VerifyScratch
	if s.floats < need {
		buf, err := d.Alloc(need)
		if err != nil {
			return metalgemm.Ink2Ref{}, 0, err
		}
		// launches already encoded keep the old buffer alive
		d.Free(s.buf)
		s.buf, s.floats = buf, need
	}
	return metalgemm.Ink2Ref{Buf: s.buf}, s.floats, nil
}

// metalInk2VerifyRows runs the tile rows kernel for any weight the decode GEMV
// reads, so a verify row is that GEMV's bits.
func (c *metalBackend) metalInk2VerifyRows(op string, w Tensor, x, y Ink2Ref, rows int) (bool, error) {
	ink2MetalScope(op)
	if len(w.Shape) != 2 {
		return false, nil
	}
	metalMu.Lock()
	defer metalMu.Unlock()
	d, r, err := ink2MetalBind(x, y)
	if err != nil {
		return false, ink2MetalErr(op, err)
	}
	wv, _, err := ink2MetalWeight(w, false)
	if err != nil {
		return false, ink2MetalErr(op, err)
	}
	return true, ink2MetalErr(op, d.VerifyRows(wv, r[0], wv.Cols, r[1], wv.Rows, rows))
}

func (c *metalBackend) Ink2VRows(w Tensor, x, y Ink2Ref, rows int) (bool, error) {
	return c.metalInk2VerifyRows("verify rows", w, x, y, rows)
}

// Ink2VQ8Rows ignores the shared codes: the Metal decode GEMV of a Q8_0 weight
// is the plain MatVec, and the rows kernel is that GEMV per row. A gated x has
// no Metal kernel, so it reports nothing launched.
func (c *metalBackend) Ink2VQ8Rows(w Tensor, x, gate, q, y Ink2Ref, rows int, quantize bool) (bool, error) {
	if w.Dtype != Q8_0 || gate.T.buf != nil {
		return false, nil
	}
	return c.metalInk2VerifyRows("verify q8_0 rows", w, x, y, rows)
}

func (c *metalBackend) Ink2VExperts(stack Tensor, expertRows int, ids, x Ink2Ref, xDiv, xStride int, y Ink2Ref, npairs int, scale Ink2Ref) (bool, error) {
	ink2MetalScope("Ink2VExperts")
	metalMu.Lock()
	defer metalMu.Unlock()
	d, r, err := ink2MetalBind(ids, x, y, scale)
	if err != nil {
		return false, ink2MetalErr("verify experts", err)
	}
	wv, indexed, err := ink2MetalWeight(stack, true)
	if err != nil || !indexed {
		return false, ink2MetalErr("verify experts", err)
	}
	if wv.Experts == 1 && expertRows > 0 && wv.Rows%expertRows == 0 {
		wv.Experts, wv.Rows = wv.Rows/expertRows, expertRows
	}
	if wv.Rows != expertRows {
		return false, fmt.Errorf("metal inkling2 verify experts: %d rows per expert, stack has %d", expertRows, wv.Rows)
	}
	tab := &metalInk2VerifyTable
	if need := metalgemm.VerifyExpertsTable(npairs); tab.ints < need {
		buf, err := d.Alloc(need)
		if err != nil {
			return false, ink2MetalErr("verify experts", err)
		}
		// launches already encoded keep the old buffer alive
		d.Free(tab.buf)
		tab.buf, tab.ints = buf, need
	}
	ok, err := d.VerifyExperts(wv, r[0], r[1], xDiv, xStride, r[2], npairs, r[3], metalgemm.Ink2Ref{Buf: tab.buf})
	return ok, ink2MetalErr("verify experts", err)
}

func (c *metalBackend) Ink2VRoute(logits, bias, ids, wts Ink2Ref, nexp, k int, scale float32, T int) error {
	ink2MetalScope("Ink2VRoute")
	metalMu.Lock()
	defer metalMu.Unlock()
	d, r, err := ink2MetalBind(logits, bias, ids, wts)
	if err != nil {
		return ink2MetalErr("verify route", err)
	}
	return ink2MetalErr("verify route", d.RouteRows(r[0], r[1], r[2], r[3], nexp, k, scale, T))
}

func (c *metalBackend) Ink2VCombine(down, wts, out Ink2Ref, k, n, T int) error {
	ink2MetalScope("Ink2VCombine")
	metalMu.Lock()
	defer metalMu.Unlock()
	d, r, err := ink2MetalBind(down, wts, out)
	if err != nil {
		return ink2MetalErr("verify combine", err)
	}
	return ink2MetalErr("verify combine", d.CombineRows(r[0], r[1], r[2], k, n, T))
}

// Ink2VQ8Segs has no single-launch Metal form; each weight runs as Ink2VQ8Rows.
func (c *metalBackend) Ink2VQ8Segs(ws []Tensor, xs, ys []Ink2Ref, q Ink2Ref, rows int) (bool, error) {
	return false, nil
}

// The combined MoE rows feeding the logged post conv, grown on demand and kept
// for the process like the device it belongs to; guarded by metalMu.
var metalInk2CombineRows struct {
	buf    metalgemm.Ink2Buf
	floats int
}

// Ink2PCombineConvLog combines into scratch rows, then runs the logged conv on
// them without dynamic taps; Metal has no fused form.
func (c *metalBackend) Ink2PCombineConvLog(down, wts Ink2Ref, k int, w, hist, log, out, resid Ink2Ref, n, kn int, norm float32, T int) error {
	const op = "verify combine conv"
	metalMu.Lock()
	defer metalMu.Unlock()
	d, r, err := ink2MetalBind(down, wts, w, hist, log, out, resid)
	if err != nil {
		return ink2MetalErr(op, err)
	}
	if n < 1 || T < 1 || T > math.MaxInt32/n {
		return fmt.Errorf("metal inkling2 %s: %d rows of %d channels", op, T, n)
	}
	s := &metalInk2CombineRows
	if need := T * n; need > s.floats {
		b, err := d.Alloc(need)
		if err != nil {
			return ink2MetalErr(op, err)
		}
		// Launches already encoded keep the old scratch alive.
		d.Free(s.buf)
		s.buf, s.floats = b, need
	}
	x := metalgemm.Ink2Ref{Buf: s.buf}
	if err := d.PCombine(r[0], r[1], x, k, n, T); err != nil {
		return ink2MetalErr(op, err)
	}
	return ink2MetalErr(op, d.VerifyConv(x, r[2], metalgemm.Ink2Ref{}, r[3], r[4], r[5], r[6], n, 1, kn, norm, T))
}

func (c *metalBackend) Ink2VMix(states []Ink2Ref, queries Ink2Ref, nq int, norms []Ink2Ref, eps float32, out Ink2Ref, n, T int) (bool, error) {
	metalMu.Lock()
	defer metalMu.Unlock()
	d, r, err := ink2MetalBind(queries, out)
	if err != nil {
		return false, ink2MetalErr("verify mix", err)
	}
	ss, err := ink2MetalRefs(states)
	if err != nil {
		return false, ink2MetalErr("verify mix states", err)
	}
	ns, err := ink2MetalRefs(norms)
	if err != nil {
		return false, ink2MetalErr("verify mix norms", err)
	}
	return true, ink2MetalErr("verify mix", d.VerifyMix(ss, r[0], nq, ns, eps, r[1], n, T))
}

func (c *metalBackend) Ink2PControlMulti(cur, msg, down, gain, forget, g, out []Ink2Ref, n, rank, T int) error {
	metalMu.Lock()
	defer metalMu.Unlock()
	d, err := metalInk2Device()
	if err != nil {
		return ink2MetalErr("verify control", err)
	}
	var mc metalgemm.Ink2Controls
	var mg, mo []metalgemm.Ink2Ref
	for _, f := range []struct {
		dst *[]metalgemm.Ink2Ref
		src []Ink2Ref
	}{{&mc.Cur, cur}, {&mc.Msg, msg}, {&mc.Down, down}, {&mc.Gain, gain}, {&mc.Forget, forget}, {&mg, g}, {&mo, out}} {
		if *f.dst, err = ink2MetalRefs(f.src); err != nil {
			return ink2MetalErr("verify control", err)
		}
	}
	return ink2MetalErr("verify control", d.VerifyControl(mc, mg, mo, n, rank, T))
}

// Ink2PConvLog also takes an absent log, which runs the same conv unlogged.
func (c *metalBackend) Ink2PConvLog(x, w, dyn, hist, log, out, resid Ink2Ref, n, heads, kn int, norm float32, T int) error {
	metalMu.Lock()
	defer metalMu.Unlock()
	d, r, err := ink2MetalBind(x, w, dyn, hist, log, out, resid)
	if err != nil {
		return ink2MetalErr("verify conv", err)
	}
	return ink2MetalErr("verify conv", d.VerifyConv(r[0], r[1], r[2], r[3], r[4], r[5], r[6], n, heads, kn, norm, T))
}

func (c *metalBackend) Ink2PConvPairLog(x, w, dyn, hist, log, out []Ink2Ref, norm []float32, n, heads, kn, T int) error {
	metalMu.Lock()
	defer metalMu.Unlock()
	d, err := metalInk2Device()
	if err != nil {
		return ink2MetalErr("verify conv pair", err)
	}
	var m [6][]metalgemm.Ink2Ref
	for i, src := range [][]Ink2Ref{x, w, dyn, hist, log, out} {
		if m[i], err = ink2MetalRefs(src); err != nil {
			return ink2MetalErr("verify conv pair", err)
		}
	}
	return ink2MetalErr("verify conv pair", d.VerifyConvPair(m[0], m[1], m[2], m[3], m[4], m[5], norm, n, heads, kn, T))
}

func (c *metalBackend) Ink2PConvCommit(hist, log Ink2Ref, channels, past, L, keep int) error {
	metalMu.Lock()
	defer metalMu.Unlock()
	d, r, err := ink2MetalBind(hist, log)
	if err != nil {
		return ink2MetalErr("conv commit", err)
	}
	return ink2MetalErr("conv commit", d.VerifyConvCommit(r[0], r[1], metalgemm.Ink2Ref{}, channels, past, L, keep))
}

func (c *metalBackend) Ink2PConvCommitDev(hist, log, keepDev Ink2Ref, channels, past, L int) error {
	metalMu.Lock()
	defer metalMu.Unlock()
	d, r, err := ink2MetalBind(hist, log, keepDev)
	if err != nil {
		return ink2MetalErr("conv commit", err)
	}
	if !r[2].Valid() {
		return fmt.Errorf("metal inkling2 conv commit: no device keep count")
	}
	return ink2MetalErr("conv commit", d.VerifyConvCommit(r[0], r[1], r[2], channels, past, L, 0))
}

func (c *metalBackend) Ink2VRepeatRows(x, out Ink2Ref, n, rows, k int) error {
	metalMu.Lock()
	defer metalMu.Unlock()
	d, r, err := ink2MetalBind(x, out)
	if err != nil {
		return ink2MetalErr("repeat rows", err)
	}
	return ink2MetalErr("repeat rows", d.VerifyRepeatRows(r[0], r[1], n, rows, k))
}

func (c *metalBackend) Ink2PQKVPostDev(q, r, k, v, qclip, rclip, kcache, vcache, posDev Ink2Ref, p Ink2AttnParams, cap, T int) error {
	metalMu.Lock()
	defer metalMu.Unlock()
	d, o, err := ink2MetalBind(q, r, k, v, qclip, rclip, kcache, vcache, posDev)
	if err != nil {
		return ink2MetalErr("verify qkv post", err)
	}
	if !o[8].Valid() {
		return fmt.Errorf("metal inkling2 verify qkv post: no device position")
	}
	return ink2MetalErr("verify qkv post",
		d.VerifyQKVPost(o[0], o[1], o[2], o[3], o[4], o[5], o[6], o[7], o[8], metalgemm.Ink2AttnParams(p), cap, 0, T))
}

func (c *metalBackend) Ink2PAttendDev(q, r, sinks, relBias, kcache, vcache, out, posDev Ink2Ref, p Ink2AttnParams, extent, window, cap, T int) error {
	metalMu.Lock()
	defer metalMu.Unlock()
	d, o, err := ink2MetalBind(q, r, sinks, relBias, kcache, vcache, out, posDev)
	if err != nil {
		return ink2MetalErr("verify attend", err)
	}
	if !o[7].Valid() {
		return fmt.Errorf("metal inkling2 verify attend: no device position")
	}
	scores, floats, err := metalInk2VerifyScores(d, p.Heads, metalgemm.VerifyAttendSpan(true, window, cap, 0, T), T)
	if err != nil {
		return ink2MetalErr("verify attend", err)
	}
	return ink2MetalErr("verify attend", d.VerifyAttend(o[0], o[1], o[2], o[3], o[4], o[5], scores, floats, o[6], o[7],
		metalgemm.Ink2AttnParams(p), extent, window, cap, 0, T))
}
