//go:build darwin && arm64 && cgo

package compute

import (
	"cmp"
	"fmt"
	"os"
	"slices"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/metalgemm"
)

var (
	_ Inkling2PrefillDevice = (*metalBackend)(nil)
	_ Inkling2PrefillHalfKV = (*metalBackend)(nil)
)

// The tiled GEMM and attention kernels sum in another order than the decode
// ops, so they take only chunks wider than any speculative verify chunk: a
// verify row must equal its decode step whatever the width. FAK_INK2P_GEMM=0
// keeps the decode-exact kernels on every chunk.
const (
	ink2MetalGemmMinRows  = 32
	ink2MetalGemmMinPairs = 256
)

var ink2MetalGemmOn = os.Getenv("FAK_INK2P_GEMM") != "0"

// ink2MetalAttendBudget bounds the floats of the batched attention's scores
// scratch; a chunk needing more is attended in token ranges.
const ink2MetalAttendBudget = 16 << 20

// A device scratch grown on demand and kept for the process like the device
// it belongs to; guarded by metalMu.
type ink2MetalScratch struct {
	buf    metalgemm.Ink2Buf
	floats int
}

// get returns the scratch grown to at least need floats; caller holds metalMu.
func (s *ink2MetalScratch) get(d *metalgemm.Ink2Device, need int) (metalgemm.Ink2Ref, error) {
	if need > s.floats {
		b, err := d.Alloc(need)
		if err != nil {
			return metalgemm.Ink2Ref{}, err
		}
		// Launches already encoded keep the old scratch alive.
		d.Free(s.buf)
		s.buf, s.floats = b, need
	}
	return metalgemm.Ink2Ref{Buf: s.buf}, nil
}

// The batched attention's scores and the GEMM controller's products. Launches
// of one command buffer run in order, so successive ops share one scratch.
var metalInk2Scores, metalInk2Ctrl ink2MetalScratch

// ink2MetalScores returns the scores scratch for T queries of heads heads over
// span keys each; caller holds metalMu.
func ink2MetalScores(d *metalgemm.Ink2Device, heads, span, T int) (metalgemm.Ink2Ref, int, error) {
	if heads < 1 || span < 1 || T < 1 || heads > ink2MetalAttendBudget/span {
		return metalgemm.Ink2Ref{}, 0, fmt.Errorf("scores scratch for %d rows of %d heads over %d keys", T, heads, span)
	}
	row := heads * span
	r, err := metalInk2Scores.get(d, row*min(T, ink2MetalAttendBudget/row))
	return r, metalInk2Scores.floats, err
}

// FAK_INK2P_PROFILE=1 flushes around each batched prefill op to attribute GPU
// seconds to it; the extra submissions make the run itself slower.
var ink2PProfileOn = os.Getenv("FAK_INK2P_PROFILE") == "1"

// The per-op GPU seconds and launch counts; guarded by metalMu.
var ink2PProfile = map[string]*[2]float64{}

func ink2PProfiled(d *metalgemm.Ink2Device, op string, run func() error) error {
	if !ink2PProfileOn {
		return run()
	}
	ink2PProfileSpan(d, "other")
	err := run()
	ink2PProfileSpan(d, op)
	return err
}

// ink2PProfiledUnlocked is ink2PProfiled for an op that takes metalMu itself.
func ink2PProfiledUnlocked(op string, run func() error) error {
	if !ink2PProfileOn {
		return run()
	}
	d, err := metalInk2Device()
	if err != nil {
		return err
	}
	metalMu.Lock()
	ink2PProfileSpan(d, "other")
	metalMu.Unlock()
	err = run()
	metalMu.Lock()
	ink2PProfileSpan(d, op)
	metalMu.Unlock()
	return err
}

func ink2PProfileSpan(d *metalgemm.Ink2Device, op string) {
	before := d.Stats().GPUSeconds
	if d.Flush() != nil {
		return
	}
	e := ink2PProfile[op]
	if e == nil {
		e = new([2]float64)
		ink2PProfile[op] = e
	}
	e[0] += d.Stats().GPUSeconds - before
	e[1]++
}

// MetalInkling2PrefillProfile returns, and resets, the GPU seconds per op that
// FAK_INK2P_PROFILE=1 collected, heaviest first; empty when it is off.
func MetalInkling2PrefillProfile() string {
	metalMu.Lock()
	defer metalMu.Unlock()
	if d, err := metalInk2Device(); err == nil && ink2PProfileOn {
		ink2PProfileSpan(d, "other")
	}
	total := 0.0
	ops := make([]string, 0, len(ink2PProfile))
	for op, e := range ink2PProfile {
		ops = append(ops, op)
		total += e[0]
	}
	slices.SortFunc(ops, func(a, b string) int { return cmp.Compare(ink2PProfile[b][0], ink2PProfile[a][0]) })
	var b strings.Builder
	for _, op := range ops {
		e := ink2PProfile[op]
		fmt.Fprintf(&b, "%s %.1fms %.1f%% x%d; ", op, 1e3*e[0], 100*e[0]/total, int(e[1]))
	}
	clear(ink2PProfile)
	return b.String()
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
	return ink2MetalErr("prefill matmul", ink2PProfiled(d, fmt.Sprintf("mm/t%d/%dx%d", wv.Type, wv.Rows, wv.Cols), func() error {
		if ink2MetalGemmOn && T >= ink2MetalGemmMinRows {
			if ok, err := d.RowsGEMM(wv, r[0], wv.Cols, r[1], wv.Rows, T); err != nil || ok {
				return err
			}
		}
		return d.RowsMatMul(wv, r[0], wv.Cols, r[1], wv.Rows, T)
	}))
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
	return ink2MetalErr("prefill mix", ink2PProfiled(d, "mix", func() error { return d.PMix(ss, r[0], nq, r[1], n, T) }))
}

func (c *metalBackend) Ink2PCtrlDown(cur, msg, w, g Ink2Ref, n, rank, T int) error {
	metalMu.Lock()
	defer metalMu.Unlock()
	d, r, err := ink2MetalBind(cur, msg, w, g)
	if err != nil {
		return ink2MetalErr("prefill ctrl down", err)
	}
	return ink2MetalErr("prefill ctrl down", ink2PProfiled(d, fmt.Sprintf("ctrl down/r%d", rank), func() error {
		if ink2MetalGemmOn && T >= ink2MetalGemmMinRows && metalgemm.CtrlTiles(n, rank) {
			part, err := metalInk2Ctrl.get(d, metalgemm.CtrlDownSplitScratch(n, rank, T))
			if err != nil {
				return err
			}
			if ok, err := d.PCtrlDownSplit(r[0], r[1], r[2], r[3], part, n, rank, T); err != nil || ok {
				return err
			}
		}
		if ink2MetalGemmOn && T >= ink2MetalGemmMinRows && metalgemm.CtrlGEMMTiles(n, rank) {
			xs, err := metalInk2Ctrl.get(d, 2*T*n)
			if err != nil {
				return err
			}
			return d.PCtrlDownGEMM(r[0], r[1], r[2], r[3], xs, n, rank, T)
		}
		return d.PCtrlDown(r[0:1], r[1:2], r[2:3], r[3:4], n, rank, T)
	}))
}

func (c *metalBackend) Ink2PCtrlApply(cur, msg, g, gainW, forgetW, out Ink2Ref, n, rank, T int) error {
	metalMu.Lock()
	defer metalMu.Unlock()
	d, r, err := ink2MetalBind(cur, msg, g, gainW, forgetW, out)
	if err != nil {
		return ink2MetalErr("prefill ctrl apply", err)
	}
	return ink2MetalErr("prefill ctrl apply", ink2PProfiled(d, fmt.Sprintf("ctrl apply/r%d", rank), func() error {
		if ink2MetalGemmOn && T >= ink2MetalGemmMinRows && metalgemm.CtrlTiles(n, rank) {
			if ok, err := d.PCtrlApplyTile(r[0], r[1], r[2], r[3], r[4], r[5], n, rank, T); err != nil || ok {
				return err
			}
		}
		if ink2MetalGemmOn && T >= ink2MetalGemmMinRows && metalgemm.CtrlGEMMTiles(n, rank) {
			gs, err := metalInk2Ctrl.get(d, 2*T*n)
			if err != nil {
				return err
			}
			return d.PCtrlApplyGEMM(r[0], r[1], r[2], r[3], r[4], r[5], gs, n, rank, T)
		}
		return d.PCtrlApply(r[0:1], r[1:2], r[2:3], r[3:4], r[4:5], r[5:6], n, rank, T)
	}))
}

func (c *metalBackend) Ink2PConv(x, w, dyn, hist, out, resid Ink2Ref, n, heads, kn int, norm float32, T int) error {
	metalMu.Lock()
	defer metalMu.Unlock()
	d, r, err := ink2MetalBind(x, w, dyn, hist, out, resid)
	if err != nil {
		return ink2MetalErr("prefill conv", err)
	}
	return ink2MetalErr("prefill conv", ink2PProfiled(d, "conv", func() error {
		if ink2MetalGemmOn && T >= ink2MetalGemmMinRows {
			return d.PConvRows(r[0], r[1], r[2], r[3], r[4], r[5], n, heads, kn, norm, T)
		}
		return d.PConv(r[0], r[1], r[2], r[3], r[4], r[5], n, heads, kn, norm, T)
	}))
}

func (c *metalBackend) Ink2PHalfKVReady() error { return nil }

func (c *metalBackend) Ink2PQKVPost(q, r, k, v, qclip, rclip, kcache, vcache Ink2Ref, p Ink2AttnParams, cap, pos0, T int) error {
	return c.ink2PQKVPost(q, r, k, v, qclip, rclip, kcache, vcache, p, cap, pos0, T, false)
}

func (c *metalBackend) Ink2PQKVPostHalf(q, r, k, v, qclip, rclip, kcache, vcache Ink2Ref, p Ink2AttnParams, cap, pos0, T int) error {
	return c.ink2PQKVPost(q, r, k, v, qclip, rclip, kcache, vcache, p, cap, pos0, T, true)
}

func (c *metalBackend) ink2PQKVPost(q, r, k, v, qclip, rclip, kcache, vcache Ink2Ref, p Ink2AttnParams, cap, pos0, T int, kvHalf bool) error {
	metalMu.Lock()
	defer metalMu.Unlock()
	d, o, err := ink2MetalBind(q, r, k, v, qclip, rclip, kcache, vcache)
	if err != nil {
		return ink2MetalErr("prefill qkv post", err)
	}
	if kvHalf {
		return ink2MetalErr("prefill qkv post f16", ink2PProfiled(d, "qkv post f16", func() error {
			return d.PQKVPostHalf(o[0], o[1], o[2], o[3], o[4], o[5], o[6], o[7], metalgemm.Ink2AttnParams(p), cap, pos0, T)
		}))
	}
	return ink2MetalErr("prefill qkv post", ink2PProfiled(d, "qkv post", func() error {
		return d.PQKVPost(o[0], o[1], o[2], o[3], o[4], o[5], o[6], o[7], metalgemm.Ink2AttnParams(p), cap, pos0, T)
	}))
}

func (c *metalBackend) Ink2PAttend(q, r, sinks, relBias, kcache, vcache, out Ink2Ref, p Ink2AttnParams, extent, window, cap, pos0, T int) error {
	return c.ink2PAttend(q, r, sinks, relBias, kcache, vcache, out, p, extent, window, cap, pos0, T, false)
}

func (c *metalBackend) Ink2PAttendHalf(q, r, sinks, relBias, kcache, vcache, out Ink2Ref, p Ink2AttnParams, extent, window, cap, pos0, T int) error {
	return c.ink2PAttend(q, r, sinks, relBias, kcache, vcache, out, p, extent, window, cap, pos0, T, true)
}

func (c *metalBackend) ink2PAttend(q, r, sinks, relBias, kcache, vcache, out Ink2Ref, p Ink2AttnParams, extent, window, cap, pos0, T int, kvHalf bool) error {
	metalMu.Lock()
	defer metalMu.Unlock()
	d, o, err := ink2MetalBind(q, r, sinks, relBias, kcache, vcache, out)
	if err != nil {
		return ink2MetalErr("prefill attend", err)
	}
	if window < 0 || cap < 1 || pos0 < 0 {
		return fmt.Errorf("metal inkling2 prefill attend: window %d, cap %d, position %d", window, cap, pos0)
	}
	label := fmt.Sprintf("attend/w%d", window)
	if kvHalf {
		label += "/f16"
	}
	return ink2MetalErr("prefill attend", ink2PProfiled(d, label, func() error {
		if !kvHalf && ink2MetalGemmOn && T >= ink2MetalGemmMinRows {
			ok, err := d.PFlashAttend(o[0], o[1], o[2], o[3], o[4], o[5], o[6], metalgemm.Ink2AttnParams(p), extent, window, cap, pos0, T)
			if err != nil || ok {
				return err
			}
		}
		scores, floats, err := ink2MetalScores(d, p.Heads, metalgemm.PAttendSpan(window, cap, pos0, T), T)
		if err != nil {
			return err
		}
		if kvHalf {
			return d.PAttendHalf(o[0], o[1], o[2], o[3], o[4], o[5], scores, floats, o[6], metalgemm.Ink2AttnParams(p), extent, window, cap, pos0, T)
		}
		return d.PAttend(o[0], o[1], o[2], o[3], o[4], o[5], scores, floats, o[6], metalgemm.Ink2AttnParams(p), extent, window, cap, pos0, T)
	}))
}

func (c *metalBackend) Ink2PRoute(logits, bias, ids, wts Ink2Ref, nexp, k int, scale float32, T int) error {
	metalMu.Lock()
	defer metalMu.Unlock()
	d, r, err := ink2MetalBind(logits, bias, ids, wts)
	if err != nil {
		return ink2MetalErr("prefill route", err)
	}
	return ink2MetalErr("prefill route", ink2PProfiled(d, "route", func() error { return d.PRoute(r[0], r[1], r[2], r[3], nexp, k, scale, T) }))
}

func (c *metalBackend) Ink2PCombine(down, wts, out Ink2Ref, k, n, T int) error {
	metalMu.Lock()
	defer metalMu.Unlock()
	d, r, err := ink2MetalBind(down, wts, out)
	if err != nil {
		return ink2MetalErr("prefill combine", err)
	}
	return ink2MetalErr("prefill combine", ink2PProfiled(d, "combine", func() error { return d.PCombine(r[0], r[1], r[2], k, n, T) }))
}

// Ink2PRMS runs the decode RMS kernel over the chunk's rows; Metal has no
// separate fp32-reduction chunk form.
func (c *metalBackend) Ink2PRMS(x, w, y Ink2Ref, rows, n int, eps float32) error {
	return ink2PProfiledUnlocked("rms", func() error { return c.Ink2RMS(x, w, y, rows, n, eps) })
}

// Ink2PSitu runs the decode SITU kernel over the chunk's routed slots.
func (c *metalBackend) Ink2PSitu(gu, act Ink2Ref, k, width, actStride int, beta, linearBeta float64) error {
	return ink2PProfiledUnlocked("situ", func() error { return c.Ink2Situ(gu, act, k, width, actStride, beta, linearBeta) })
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
		if err := ink2PProfiled(d, "bucket", func() error {
			if ink2MetalGemmOn && npairs >= ink2MetalGemmMinPairs {
				return d.PBucketSort(r[0], r[4], r[5], r[6], npairs, nexp)
			}
			return d.PBucket(r[0], r[4], r[5], r[6], npairs, nexp)
		}); err != nil {
			return true, ink2MetalErr(op, err)
		}
	}
	return true, ink2MetalErr(op, ink2PProfiled(d, fmt.Sprintf("experts/t%d/%dx%d", wv.Type, wv.Rows, wv.Cols), func() error {
		if ink2MetalGemmOn && npairs >= ink2MetalGemmMinPairs {
			if ok, err := d.PExpertGEMM(wv, r[0], r[6], r[5], r[1], xDiv, xStride, r[2], npairs, r[3]); err != nil || ok {
				return err
			}
		}
		return d.PExpertGrouped(wv, r[0], r[6], r[1], xDiv, xStride, r[2], npairs, r[3])
	}))
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
