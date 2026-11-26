//go:build darwin && arm64 && cgo

package metalgemm

import (
	_ "embed"
	"fmt"
	"math"
)

//go:embed ink2_conv.metal
var ink2ConvSource string

// ink2ConvThreads is the channels one conv threadgroup covers.
const ink2ConvThreads = 256

type ink2ConvShape struct {
	n, perHead, kn, slots uint32
	dynamic, residual     uint32
	norm                  float32
}

type ink2QKVShape struct {
	heads, kvHeads, headDim, relLen      uint32
	cap, lanes                           uint32
	global, conv, kn, kDynamic, vDynamic uint32
	qkClip, relClip, eps                 float32
	alpha, floorTok, kNorm, vNorm        float32
}

// Conv encodes the causal short conv of n channels: out = (x + taps)*norm
// (+ resid), where the taps weigh the kn-1 past inputs in hist ([n][kn-1],
// oldest first) and x by w ([n][kn]), plus the per-head dynamic taps dyn
// ([heads][kn]) when given. x then shifts into hist.
func (d *Ink2Device) Conv(x, w, dyn, hist, out, resid Ink2Ref, n, heads, kn int, norm float32) error {
	if !x.Valid() {
		return fmt.Errorf("%w: conv without an input", ErrInk2Shape)
	}
	return d.conv(x, Ink2Ref{}, 0, w, dyn, hist, out, resid, n, heads, kn, norm)
}

// CombineConv is Conv without dynamic taps whose input is the combine of the
// k routed rows of down ([k][n]) weighted by wts, summed in routing order.
func (d *Ink2Device) CombineConv(down, wts Ink2Ref, k int, w, hist, out, resid Ink2Ref, n, kn int, norm float32) error {
	if !down.Valid() || !wts.Valid() || k < 1 || k > math.MaxInt32 {
		return fmt.Errorf("%w: combine conv of %d routed slots", ErrInk2Shape, k)
	}
	return d.conv(down, wts, k, w, Ink2Ref{}, hist, out, resid, n, 1, kn, norm)
}

func (d *Ink2Device) conv(x, wts Ink2Ref, slots int, w, dyn, hist, out, resid Ink2Ref, n, heads, kn int, norm float32) error {
	// The dynamic taps index a head as channel / (n / heads).
	if !w.Valid() || !hist.Valid() || !out.Valid() || n < 1 || kn < 2 || heads < 1 || n > math.MaxInt32/kn ||
		(dyn.Valid() && n%heads != 0) {
		return fmt.Errorf("%w: conv of %d channels, %d heads, kernel %d", ErrInk2Shape, n, heads, kn)
	}
	p, err := d.PipelinePrecise(ink2ConvSource, "ink2_conv")
	if err != nil {
		return err
	}
	perHead, taps := n, 0
	if dyn.Valid() {
		perHead, taps = n/heads, heads*kn
	}
	return d.DispatchThreads(p, [3]int{n, 1, 1}, [3]int{min(n, ink2ConvThreads, p.MaxThreads()), 1, 1},
		x.Span(max(slots, 1)*n), wts.Span(slots), w.Span(n*kn), dyn.Span(taps), hist.Span(n*(kn-1)), out.Span(n), resid.Span(n),
		Ink2Const(ink2ConvShape{n: uint32(n), perHead: uint32(perHead), kn: uint32(kn), slots: uint32(slots),
			dynamic: uint32(btoi(dyn.Valid())), residual: uint32(btoi(resid.Valid())), norm: norm}))
}

// QKVPost encodes the post-projection attention prep of one layer visit. It
// clips each head of q ([heads][headDim]) and r ([heads][relLen]) in place
// after their per-channel scales qclip and rclip, times the log-scale tau on a
// global layer; clips each head of k and RMS-normalizes each head of v
// ([kvHeads][headDim]); and writes those into row pos%cap of kcache and vcache
// ([cap][kvHeads*headDim]), pos being the int32 at meta. cv first runs the k
// and v short convs on the raw projections, each over its own history.
func (d *Ink2Device) QKVPost(q, r, k, v, qclip, rclip, kcache, vcache, meta Ink2Ref, p Ink2AttnParams, cap int, cv *Ink2KVConv) error {
	const limit = math.MaxInt32
	if p.Heads < 1 || p.KVHeads < 1 || p.HeadDim < 1 || p.RelLen < 0 || cap < 1 || cap > limit ||
		p.Heads > limit/p.HeadDim || p.Heads > limit/max(p.RelLen, 1) || p.KVHeads > limit/p.HeadDim || p.Heads > limit-2*p.KVHeads {
		return fmt.Errorf("%w: qkv post of %d heads, %d kv heads, head dim %d, rel length %d, cache of %d",
			ErrInk2Shape, p.Heads, p.KVHeads, p.HeadDim, p.RelLen, cap)
	}
	// A layer without relative states has no r to clip, as in Attend.
	rel := p.RelLen == 0 || (r.Valid() && rclip.Valid())
	for _, ref := range []Ink2Ref{q, k, v, qclip, kcache, vcache, meta} {
		rel = rel && ref.Valid()
	}
	if !rel {
		return fmt.Errorf("%w: qkv post without an operand", ErrInk2Shape)
	}
	if p.RelLen == 0 {
		r, rclip = Ink2Ref{}, Ink2Ref{}
	}
	kvDim := p.KVHeads * p.HeadDim
	var c Ink2KVConv
	// An absent conv binds no buffers; its spans only have to be well formed.
	kn := 1
	if cv != nil {
		c, kn = *cv, cv.Kn
		if !c.KW.Valid() || !c.VW.Valid() || !c.KHist.Valid() || !c.VHist.Valid() || kn < 2 || kvDim > limit/kn {
			return fmt.Errorf("%w: qkv post conv of kernel %d", ErrInk2Shape, kn)
		}
	}
	source := ink2ConvSource
	if !ink2TinyFuseLegacy && p.HeadDim == ink2QKVNamedHeadDim && p.RelLen == ink2QKVNamedRelLen && cv != nil && kn == ink2QKVNamedTaps {
		source = ink2QKVNamedSource
	}
	pipe, err := d.PipelinePrecise(source, "ink2_qkv_post")
	if err != nil {
		return err
	}
	// Each head reduces across one SIMD group, so that is the threadgroup.
	lanes := pipe.ThreadWidth()
	if source == ink2QKVNamedSource && lanes != ink2QKVNamedLanes {
		return fmt.Errorf("metalgemm: inkling2 qkv post built for SIMD groups of %d threads runs on %d", ink2QKVNamedLanes, lanes)
	}
	if lanes < 1 || lanes > pipe.MaxThreads() {
		return fmt.Errorf("metalgemm: inkling2 qkv post: SIMD groups of %d threads in threadgroups of at most %d", lanes, pipe.MaxThreads())
	}
	return d.Dispatch(pipe, [3]int{p.Heads + 2*p.KVHeads, 1, 1}, [3]int{lanes, 1, 1},
		q.Span(p.Heads*p.HeadDim), r.Span(p.Heads*p.RelLen), k.Span(kvDim), v.Span(kvDim),
		qclip.Span(p.Heads*p.HeadDim), rclip.Span(p.Heads*p.RelLen), kcache.Span(cap*kvDim), vcache.Span(cap*kvDim), meta.Span(1),
		c.KW.Span(kvDim*kn), c.VW.Span(kvDim*kn), c.KDyn.Span(p.KVHeads*kn), c.VDyn.Span(p.KVHeads*kn),
		c.KHist.Span(kvDim*(kn-1)), c.VHist.Span(kvDim*(kn-1)),
		Ink2Const(ink2QKVShape{
			heads: uint32(p.Heads), kvHeads: uint32(p.KVHeads), headDim: uint32(p.HeadDim), relLen: uint32(p.RelLen),
			cap: uint32(cap), lanes: uint32(lanes),
			global: uint32(btoi(p.Global)), conv: uint32(btoi(cv != nil)), kn: uint32(kn),
			kDynamic: uint32(btoi(c.KDyn.Valid())), vDynamic: uint32(btoi(c.VDyn.Valid())),
			qkClip: p.QKClip, relClip: p.RelClip, eps: p.Eps,
			alpha: p.Alpha, floorTok: p.Floor, kNorm: c.KNorm, vNorm: c.VNorm,
		}))
}
