//go:build darwin && arm64 && cgo

package compute

import (
	"fmt"

	"github.com/anthony-chaudhary/fak/internal/metalgemm"
)

var _ Inkling2PrefillDevice = (*metalBackend)(nil)

// ink2MetalAttendBudget bounds the floats of the batched attention's scores
// scratch; a chunk needing more is attended in token ranges.
const ink2MetalAttendBudget = 16 << 20

// The batched attention's scores scratch, grown on demand and kept for the
// process like the device it belongs to; guarded by metalMu.
var metalInk2Scores struct {
	buf    metalgemm.Ink2Buf
	floats int
}

// ink2MetalScores returns the scores scratch for T queries of heads heads over
// span keys each; caller holds metalMu.
func ink2MetalScores(d *metalgemm.Ink2Device, heads, span, T int) (metalgemm.Ink2Ref, int, error) {
	if heads < 1 || span < 1 || T < 1 || heads > ink2MetalAttendBudget/span {
		return metalgemm.Ink2Ref{}, 0, fmt.Errorf("scores scratch for %d rows of %d heads over %d keys", T, heads, span)
	}
	row := heads * span
	need := row * min(T, ink2MetalAttendBudget/row)
	s := &metalInk2Scores
	if need > s.floats {
		b, err := d.Alloc(need)
		if err != nil {
			return metalgemm.Ink2Ref{}, 0, err
		}
		// Launches already encoded keep the old scratch alive.
		d.Free(s.buf)
		s.buf, s.floats = b, need
	}
	return metalgemm.Ink2Ref{Buf: s.buf}, s.floats, nil
}

// MetalInkling2DeviceStats counts the inkling2 device's submissions, launches
// and GPU seconds since it opened, for a caller that attributes them to a span
// of ops.
func MetalInkling2DeviceStats() (metalgemm.Ink2Stats, error) {
	d, err := metalInk2Device()
	if err != nil {
		return metalgemm.Ink2Stats{}, err
	}
	return d.Stats(), nil
}

// ink2MetalStack resolves an expert stack to expertRows-row experts; ok is
// false for a stack a device id cannot index.
func ink2MetalStack(stack Tensor, expertRows int) (metalgemm.Ink2Weight, bool, error) {
	wv, indexed, err := ink2MetalWeight(stack, true)
	if err != nil || !indexed {
		return wv, false, err
	}
	if wv.Experts == 1 && expertRows > 0 && wv.Rows%expertRows == 0 {
		// dense rows carry no expert count of their own
		wv.Experts, wv.Rows = wv.Rows/expertRows, expertRows
	}
	if wv.Rows != expertRows {
		return wv, false, fmt.Errorf("%d rows per expert, stack has %d", expertRows, wv.Rows)
	}
	return wv, true, nil
}

func (c *metalBackend) Ink2PMatMul(w Tensor, x, y Ink2Ref, T int) error {
	metalMu.Lock()
	defer metalMu.Unlock()
	d, r, err := ink2MetalBind(x, y)
	if err != nil {
		return ink2MetalErr("prefill matmul", err)
	}
	wv, _, err := ink2MetalWeight(w, false)
	if err != nil {
		return ink2MetalErr("prefill matmul", err)
	}
	return ink2MetalErr("prefill matmul", d.RowsMatMul(wv, r[0], wv.Cols, r[1], wv.Rows, T))
}

func (c *metalBackend) Ink2PMix(states []Ink2Ref, queries Ink2Ref, nq int, out Ink2Ref, n, T int) error {
	metalMu.Lock()
	defer metalMu.Unlock()
	d, r, err := ink2MetalBind(queries, out)
	if err != nil {
		return ink2MetalErr("prefill mix", err)
	}
	ss, err := ink2MetalRefs(states)
	if err != nil {
		return ink2MetalErr("prefill mix states", err)
	}
	return ink2MetalErr("prefill mix", d.PMix(ss, r[0], nq, r[1], n, T))
}

func (c *metalBackend) Ink2PCtrlDown(cur, msg, w, g Ink2Ref, n, rank, T int) error {
	metalMu.Lock()
	defer metalMu.Unlock()
	d, r, err := ink2MetalBind(cur, msg, w, g)
	if err != nil {
		return ink2MetalErr("prefill ctrl down", err)
	}
	return ink2MetalErr("prefill ctrl down", d.PCtrlDown(r[0:1], r[1:2], r[2:3], r[3:4], n, rank, T))
}

func (c *metalBackend) Ink2PCtrlApply(cur, msg, g, gainW, forgetW, out Ink2Ref, n, rank, T int) error {
	metalMu.Lock()
	defer metalMu.Unlock()
	d, r, err := ink2MetalBind(cur, msg, g, gainW, forgetW, out)
	if err != nil {
		return ink2MetalErr("prefill ctrl apply", err)
	}
	return ink2MetalErr("prefill ctrl apply", d.PCtrlApply(r[0:1], r[1:2], r[2:3], r[3:4], r[4:5], r[5:6], n, rank, T))
}

func (c *metalBackend) Ink2PConv(x, w, dyn, hist, out, resid Ink2Ref, n, heads, kn int, norm float32, T int) error {
	metalMu.Lock()
	defer metalMu.Unlock()
	d, r, err := ink2MetalBind(x, w, dyn, hist, out, resid)
	if err != nil {
		return ink2MetalErr("prefill conv", err)
	}
	return ink2MetalErr("prefill conv", d.PConv(r[0], r[1], r[2], r[3], r[4], r[5], n, heads, kn, norm, T))
}

func (c *metalBackend) Ink2PQKVPost(q, r, k, v, qclip, rclip, kcache, vcache Ink2Ref, p Ink2AttnParams, cap, pos0, T int) error {
	metalMu.Lock()
	defer metalMu.Unlock()
	d, o, err := ink2MetalBind(q, r, k, v, qclip, rclip, kcache, vcache)
	if err != nil {
		return ink2MetalErr("prefill qkv post", err)
	}
	return ink2MetalErr("prefill qkv post", d.PQKVPost(o[0], o[1], o[2], o[3], o[4], o[5], o[6], o[7], metalgemm.Ink2AttnParams(p), cap, pos0, T))
}

func (c *metalBackend) Ink2PAttend(q, r, sinks, relBias, kcache, vcache, out Ink2Ref, p Ink2AttnParams, extent, window, cap, pos0, T int) error {
	metalMu.Lock()
	defer metalMu.Unlock()
	d, o, err := ink2MetalBind(q, r, sinks, relBias, kcache, vcache, out)
	if err != nil {
		return ink2MetalErr("prefill attend", err)
	}
	if window < 0 || cap < 1 || pos0 < 0 {
		return fmt.Errorf("metal inkling2 prefill attend: window %d, cap %d, position %d", window, cap, pos0)
	}
	scores, floats, err := ink2MetalScores(d, p.Heads, metalgemm.PAttendSpan(window, cap, pos0, T), T)
	if err != nil {
		return ink2MetalErr("prefill attend", err)
	}
	return ink2MetalErr("prefill attend", d.PAttend(o[0], o[1], o[2], o[3], o[4], o[5], scores, floats, o[6],
		metalgemm.Ink2AttnParams(p), extent, window, cap, pos0, T))
}

func (c *metalBackend) Ink2PRoute(logits, bias, ids, wts Ink2Ref, nexp, k int, scale float32, T int) error {
	metalMu.Lock()
	defer metalMu.Unlock()
	d, r, err := ink2MetalBind(logits, bias, ids, wts)
	if err != nil {
		return ink2MetalErr("prefill route", err)
	}
	return ink2MetalErr("prefill route", d.PRoute(r[0], r[1], r[2], r[3], nexp, k, scale, T))
}

func (c *metalBackend) Ink2PCombine(down, wts, out Ink2Ref, k, n, T int) error {
	metalMu.Lock()
	defer metalMu.Unlock()
	d, r, err := ink2MetalBind(down, wts, out)
	if err != nil {
		return ink2MetalErr("prefill combine", err)
	}
	return ink2MetalErr("prefill combine", d.PCombine(r[0], r[1], r[2], k, n, T))
}

// Ink2PRMS runs the decode RMS kernel over the chunk's rows; Metal has no
// separate fp32-reduction chunk form.
func (c *metalBackend) Ink2PRMS(x, w, y Ink2Ref, rows, n int, eps float32) error {
	return c.Ink2RMS(x, w, y, rows, n, eps)
}

// Ink2PSitu runs the decode SITU kernel over the chunk's routed slots.
func (c *metalBackend) Ink2PSitu(gu, act Ink2Ref, k, width, actStride int, beta, linearBeta float64) error {
	return c.Ink2Situ(gu, act, k, width, actStride, beta, linearBeta)
}

// ink2PGrouped buckets the ids unless bucketed, then runs the grouped GEMM. It
// uses the counts, offsets and order regions of the Ink2PGroupScratch layout
// and leaves the tile list alone: the kernel finds expert runs itself.
func (c *metalBackend) ink2PGrouped(op string, stack Tensor, expertRows, nexp int, ids, x Ink2Ref, xDiv, xStride int, y Ink2Ref, npairs int, scale, scratch Ink2Ref, bucketed bool) (bool, error) {
	if len(stack.Shape) != 2 {
		return false, nil
	}
	metalMu.Lock()
	defer metalMu.Unlock()
	wv, ok, err := ink2MetalStack(stack, expertRows)
	if err != nil || !ok {
		return false, ink2MetalErr(op, err)
	}
	if wv.Experts != nexp {
		return true, fmt.Errorf("metal inkling2 %s: %d experts, stack has %d", op, nexp, wv.Experts)
	}
	counts := scratch
	offsets := Ink2At(scratch.T, counts.Off+nexp)
	order := Ink2At(scratch.T, offsets.Off+nexp+1)
	d, r, err := ink2MetalBind(ids, x, y, scale, counts, offsets, order)
	if err != nil {
		return true, ink2MetalErr(op, err)
	}
	if !bucketed {
		if err := d.PBucket(r[0], r[4], r[5], r[6], npairs, nexp); err != nil {
			return true, ink2MetalErr(op, err)
		}
	}
	return true, ink2MetalErr(op, d.PExpertGrouped(wv, r[0], r[6], r[1], xDiv, xStride, r[2], npairs, r[3]))
}

// Ink2PExpertGrouped serves every dtype the packed kernels read: the grouped
// kernel is the decode expert GEMV's row arithmetic for all of them.
func (c *metalBackend) Ink2PExpertGrouped(stack Tensor, expertRows, nexp int, ids, x Ink2Ref, xDiv, xStride int, y Ink2Ref, npairs int, scale, scratch Ink2Ref, bucketed bool) (bool, error) {
	return c.ink2PGrouped("prefill expert grouped", stack, expertRows, nexp, ids, x, xDiv, xStride, y, npairs, scale, scratch, bucketed)
}

// Ink2PExpertSorted is the grouped GEMM: it already visits the pairs in expert
// order without a gather and scatter, so the sort staging goes unused.
func (c *metalBackend) Ink2PExpertSorted(stack Tensor, expertRows, nexp int, ids, x Ink2Ref, xDiv, xStride int, y Ink2Ref, npairs int, scale, scratch, sortIds, sortX, sortY Ink2Ref) (bool, error) {
	return c.ink2PGrouped("prefill expert sorted", stack, expertRows, nexp, ids, x, xDiv, xStride, y, npairs, scale, scratch, false)
}
