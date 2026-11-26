//go:build darwin && arm64 && cgo

package compute

import (
	"fmt"
	"math"

	"github.com/anthony-chaudhary/fak/internal/metalgemm"
)

var _ Inkling2TreeVerifyDevice = (*metalBackend)(nil)

func (c *metalBackend) Ink2TConvLog(x, w, dyn, hist, log, out, resid, parent Ink2Ref, n, heads, kn int, norm float32, T int) error {
	metalMu.Lock()
	defer metalMu.Unlock()
	d, r, err := ink2MetalBind(x, w, dyn, hist, log, out, resid, parent)
	if err != nil {
		return ink2MetalErr("tree conv", err)
	}
	return ink2MetalErr("tree conv", d.TreeVerifyConv(r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7], n, heads, kn, norm, T))
}

func (c *metalBackend) Ink2TCombineConvLog(down, wts Ink2Ref, k int, w, hist, log, out, resid, parent Ink2Ref, n, kn int, norm float32, T int) error {
	const op = "tree combine conv"
	metalMu.Lock()
	defer metalMu.Unlock()
	d, r, err := ink2MetalBind(down, wts, w, hist, log, out, resid, parent)
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
	return ink2MetalErr(op, d.TreeVerifyConv(x, r[2], metalgemm.Ink2Ref{}, r[3], r[4], r[5], r[6], r[7], n, 1, kn, norm, T))
}

func (c *metalBackend) Ink2TConvCommit(hist, log, colsDev Ink2Ref, channels, past, L int, cols []int) error {
	metalMu.Lock()
	defer metalMu.Unlock()
	d, r, err := ink2MetalBind(hist, log, colsDev)
	if err != nil {
		return ink2MetalErr("tree conv commit", err)
	}
	return ink2MetalErr("tree conv commit", d.TreeVerifyConvCommit(r[0], r[1], r[2], channels, past, L, cols))
}

func (c *metalBackend) Ink2TQKVPostDev(q, r, k, v, qclip, rclip, kcache, vcache, posDev, parent Ink2Ref, p Ink2AttnParams, cap, T int) error {
	metalMu.Lock()
	defer metalMu.Unlock()
	d, o, err := ink2MetalBind(q, r, k, v, qclip, rclip, kcache, vcache, posDev, parent)
	if err != nil {
		return ink2MetalErr("tree qkv post", err)
	}
	if !o[8].Valid() {
		return fmt.Errorf("metal inkling2 tree qkv post: no device position")
	}
	return ink2MetalErr("tree qkv post",
		d.TreeVerifyQKVPost(o[0], o[1], o[2], o[3], o[4], o[5], o[6], o[7], o[8], o[9], metalgemm.Ink2AttnParams(p), cap, 0, T))
}

func (c *metalBackend) Ink2TAttendDev(q, r, sinks, relBias, kcache, vcache, out, posDev, parent Ink2Ref, p Ink2AttnParams, extent, window, cap, T int) error {
	metalMu.Lock()
	defer metalMu.Unlock()
	d, o, err := ink2MetalBind(q, r, sinks, relBias, kcache, vcache, out, posDev, parent)
	if err != nil {
		return ink2MetalErr("tree attend", err)
	}
	if !o[7].Valid() {
		return fmt.Errorf("metal inkling2 tree attend: no device position")
	}
	scores, floats, err := metalInk2VerifyScores(d, p.Heads, metalgemm.VerifyAttendSpan(true, window, cap, 0, T), T)
	if err != nil {
		return ink2MetalErr("tree attend", err)
	}
	return ink2MetalErr("tree attend", d.TreeVerifyAttend(o[0], o[1], o[2], o[3], o[4], o[5], scores, floats, o[6], o[7], o[8],
		metalgemm.Ink2AttnParams(p), extent, window, cap, 0, T))
}

func (c *metalBackend) Ink2TKVCompact(kcache, vcache, pathDev, posDev Ink2Ref, width, cap int, path []int) error {
	metalMu.Lock()
	defer metalMu.Unlock()
	d, r, err := ink2MetalBind(kcache, vcache, pathDev, posDev)
	if err != nil {
		return ink2MetalErr("tree kv compact", err)
	}
	if !r[3].Valid() {
		return fmt.Errorf("metal inkling2 tree kv compact: no device position")
	}
	return ink2MetalErr("tree kv compact", d.TreeVerifyKVCompact(r[0], r[1], r[2], r[3], width, cap, 0, path))
}

func (c *metalBackend) Ink2TGatherRows(in, out, pathDev Ink2Ref, width, inStride, rows int, path []int) error {
	metalMu.Lock()
	defer metalMu.Unlock()
	d, r, err := ink2MetalBind(in, out, pathDev)
	if err != nil {
		return ink2MetalErr("tree gather rows", err)
	}
	return ink2MetalErr("tree gather rows", d.TreeVerifyGatherRows(r[0], r[1], r[2], width, inStride, rows, path))
}
