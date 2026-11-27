//go:build darwin && arm64 && cgo

package metalgemm

import (
	_ "embed"
	"fmt"
	"math"
)

//go:embed ink2_attend.metal
var ink2AttendSource string

// The kernel's threadgroup arrays are sized for these.
const (
	ink2AttendMaxThreads = 512
	ink2AttendMaxSIMD    = 64
	ink2AttendQTerms     = 4
)

type ink2AttendShape struct {
	heads, perKV, headDim, relLen uint32
	kvDim, extent, window, cap    uint32
	threads, width                uint32
	scale                         float32
}

// Attend encodes the attention of the newest token over one KV slot into
// out ([heads][headDim]). kcache and vcache are [cap][kvHeads*headDim] rows
// with position p at row p % cap, and meta[0] is the newest position as an
// int32, so the keys are the last min(pos+1, window, cap) positions (window 0
// is unbounded). scores is scratch: its first heads*cap floats are used.
func (d *Ink2Device) Attend(q, r, sinks, relBias, kcache, vcache, scores, out, meta Ink2Ref, p Ink2AttnParams, extent, window, cap int) error {
	return d.attend(q, r, sinks, relBias, kcache, vcache, scores, out, meta, p, extent, window, cap, false)
}

// AttendHalf is Attend over f16 kcache and vcache (two half elements per float).
func (d *Ink2Device) AttendHalf(q, r, sinks, relBias, kcache, vcache, scores, out, meta Ink2Ref, p Ink2AttnParams, extent, window, cap int) error {
	return d.attend(q, r, sinks, relBias, kcache, vcache, scores, out, meta, p, extent, window, cap, true)
}

func (d *Ink2Device) attend(q, r, sinks, relBias, kcache, vcache, scores, out, meta Ink2Ref, p Ink2AttnParams, extent, window, cap int, kvHalf bool) error {
	if p.Heads < 1 || p.KVHeads < 1 || p.Heads%p.KVHeads != 0 || p.HeadDim < 1 || p.HeadDim > math.MaxInt32/p.Heads || p.RelLen < 0 ||
		extent < 0 || window < 0 || cap < 1 || p.RelLen > math.MaxInt32 || extent > math.MaxInt32 || window > math.MaxInt32 {
		return fmt.Errorf("%w: attend of %d heads over %d kv heads of %d dims, %d rel states, extent %d, window %d, cap %d",
			ErrInk2Shape, p.Heads, p.KVHeads, p.HeadDim, p.RelLen, extent, window, cap)
	}
	kvDim := p.KVHeads * p.HeadDim
	// The kernel indexes every operand with 32-bit offsets.
	if cap > math.MaxInt32/p.Heads || cap > math.MaxInt32/kvDim ||
		(p.RelLen > 0 && (p.Heads > math.MaxInt32/p.RelLen || extent > math.MaxInt32/p.RelLen)) {
		return fmt.Errorf("%w: attend operands exceed 32-bit offsets", ErrInk2Shape)
	}
	if p.RelLen == 0 {
		extent = 0
	}
	if !q.Valid() || !sinks.Valid() || !kcache.Valid() || !vcache.Valid() || !scores.Valid() || !out.Valid() || !meta.Valid() ||
		(extent > 0 && (!r.Valid() || !relBias.Valid())) {
		return fmt.Errorf("%w: attend without an operand", ErrInk2Shape)
	}
	// Fast math off: the softmax leans on exp's precise form.
	source := ink2AttendNamedSource
	if ink2TinyFuseLegacy {
		source = ink2AttendSource
	}
	if kvHalf {
		source = "#define INK2_ATT_HALF 1\n" + source
	}
	pipe, err := d.PipelinePrecise(source, "ink2_attend")
	if err != nil {
		return err
	}
	width := pipe.ThreadWidth()
	if width < 1 {
		return fmt.Errorf("metalgemm: inkling2 attend kernel reports no SIMD width")
	}
	threads := min(pipe.MaxThreads(), ink2AttendMaxThreads, ink2AttendMaxSIMD*width) / width * width
	if quads := (p.HeadDim + 3) / 4; quads > threads || quads > ink2AttendQTerms*width {
		return fmt.Errorf("%w: attend head of %d dims on %d threads of width %d", ErrInk2Shape, p.HeadDim, threads, width)
	}
	rSpan, relSpan := Ink2Ref{}.Span(0), Ink2Ref{}.Span(0)
	if extent > 0 {
		rSpan, relSpan = r.Span(p.Heads*p.RelLen), relBias.Span(extent*p.RelLen)
	}
	kvFloats := cap * kvDim
	if kvHalf {
		kvFloats = (kvFloats + 1) / 2
	}
	shape := ink2AttendShape{
		heads: uint32(p.Heads), perKV: uint32(p.Heads / p.KVHeads), headDim: uint32(p.HeadDim), relLen: uint32(p.RelLen),
		kvDim: uint32(kvDim), extent: uint32(extent), window: uint32(window), cap: uint32(cap),
		threads: uint32(threads), width: uint32(width), scale: p.Scale,
	}
	return d.Dispatch(pipe, [3]int{p.Heads, 1, 1}, [3]int{threads, 1, 1},
		q.Span(p.Heads*p.HeadDim), rSpan, sinks.Span(p.Heads), relSpan,
		kcache.Span(kvFloats), vcache.Span(kvFloats), scores.Span(p.Heads*cap), out.Span(p.Heads*p.HeadDim),
		meta.Span(1), Ink2Const(shape))
}
