//go:build darwin && arm64 && cgo

package metalgemm

import (
	_ "embed"
	"fmt"
	"math"
	"sync"
)

//go:embed ink2_dflash2.metal
var ink2DFlash2Source string

//go:embed ink2_dflash2_head.metal
var ink2DFlash2Head string

var ink2DFlash2HeadSource = sync.OnceValue(func() string { return packedCodebooks + packedDequant + ink2DFlash2Head })

// ink2DFlash2HeadKernel names the draft head kernel of each GGML type it covers.
var ink2DFlash2HeadKernel = map[uint32]string{8: "ink2_df2_head_q8_0", 12: "ink2_df2_head_q4_K", 13: "ink2_df2_head_q5_K", 14: "ink2_df2_head_q6_K"}

type ink2DF2HeadShape struct{ rows, cols, byteOffset, tokens, xStride, yStride uint32 }

type ink2DFRopeShape struct {
	rows, heads, dim, cap, ring, base uint32
	eps, log2Theta                    float32
}

type ink2DFRingShape struct{ rows, width, cap, base uint32 }

// DraftQKRope RMS-normalizes each dim-wide head of x ([rows][heads][dim]) by
// w and rotates its halves (neox RoPE, theta) at position *posDev+base+row,
// writing row order, or ring slot position%cap when ring is set. x may alias
// dst in row order.
func (d *Ink2Device) DraftQKRope(x, w, dst, posDev Ink2Ref, rows, heads, dim, cap, base int, ring bool, eps float32, theta float64) error {
	for _, r := range []Ink2Ref{x, w, dst, posDev} {
		if !r.Valid() {
			return fmt.Errorf("%w: draft qk rope without an operand", ErrInk2Shape)
		}
	}
	if rows < 1 || heads < 1 || dim < 2 || dim%2 != 0 || dim > 512 || base < 0 || (ring && cap < 1) || theta <= 0 {
		return fmt.Errorf("%w: draft qk rope of %d rows, %d heads x %d, ring %v of %d", ErrInk2Shape, rows, heads, dim, ring, cap)
	}
	p, err := d.PipelinePrecise(ink2DFlash2Source, "ink2_df_qk_rope")
	if err != nil {
		return err
	}
	out := rows * heads * dim
	if ring {
		out = cap * heads * dim
	}
	threads := 64
	for threads > dim/2 && threads > 1 {
		threads /= 2
	}
	return d.Dispatch(p, [3]int{heads, rows, 1}, [3]int{min(threads, p.MaxThreads()), 1, 1},
		x.Span(rows*heads*dim), w.Span(dim), dst.Span(out), posDev.Span(1),
		Ink2Const(ink2DFRopeShape{rows: uint32(rows), heads: uint32(heads), dim: uint32(dim), cap: uint32(max(cap, 1)),
			ring: uint32(btoi(ring)), base: uint32(base), eps: eps, log2Theta: float32(math.Log2(theta))}))
}

// DraftRingStore copies src row r ([rows][width]) into ring slot
// (*posDev+base+r)%cap.
func (d *Ink2Device) DraftRingStore(src, ring, posDev Ink2Ref, rows, width, cap, base int) error {
	if !src.Valid() || !ring.Valid() || !posDev.Valid() || rows < 1 || width < 1 || cap < 1 || base < 0 || rows > math.MaxInt32/width {
		return fmt.Errorf("%w: draft ring store of %d rows x %d into %d", ErrInk2Shape, rows, width, cap)
	}
	p, err := d.Pipeline(ink2DFlash2Source, "ink2_df_ring_store")
	if err != nil {
		return err
	}
	n := rows * width
	return d.DispatchThreads(p, [3]int{n, 1, 1}, [3]int{min(n, 256, p.MaxThreads()), 1, 1},
		src.Span(n), ring.Span(cap*width), posDev.Span(1),
		Ink2Const(ink2DFRingShape{rows: uint32(rows), width: uint32(width), cap: uint32(cap), base: uint32(base)}))
}

// DraftSiluMul is out = silu(g) * u over n floats; out may alias g or u.
func (d *Ink2Device) DraftSiluMul(g, u, out Ink2Ref, n int) error {
	if !g.Valid() || !u.Valid() || !out.Valid() || n < 1 || n > math.MaxInt32 {
		return fmt.Errorf("%w: draft silu mul of %d", ErrInk2Shape, n)
	}
	p, err := d.PipelinePrecise(ink2DFlash2Source, "ink2_df_silu_mul")
	if err != nil {
		return err
	}
	return d.DispatchThreads(p, [3]int{n, 1, 1}, [3]int{min(n, 256, p.MaxThreads()), 1, 1},
		g.Span(n), u.Span(n), out.Span(n), Ink2Const(uint32(n)))
}

// DraftHead encodes y[t] = W x[t] for tokens <= 8 activation rows (x row t
// at x + t*xStride, output at y + t*yStride) of a packed weight, with each
// activation tile shared by 32 weight rows through SIMD-group matrices. It
// rounds differently from VerifyRows; ok is false for a type or shape it
// does not cover.
func (d *Ink2Device) DraftHead(w Ink2Weight, x Ink2Ref, xStride int, y Ink2Ref, yStride, tokens int) (bool, error) {
	name, ok := ink2DFlash2HeadKernel[w.Type]
	if !ok || w.Experts != 1 || w.Cols%64 != 0 || w.Rows < 1 || w.ByteOff < 0 {
		return false, nil
	}
	if !x.Valid() || !y.Valid() || tokens < 1 || tokens > ink2DFMaxTokens || xStride < w.Cols || yStride < w.Rows ||
		w.Rows > math.MaxInt32 || tokens*max(xStride, yStride) > math.MaxInt32 {
		return false, fmt.Errorf("%w: draft head of %d rows, strides %d and %d", ErrInk2Shape, tokens, xStride, yStride)
	}
	block, blockBytes, err := packedBlock(w.Type)
	if err != nil {
		return false, err
	}
	p, err := d.Pipeline(ink2DFlash2HeadSource(), name)
	if err != nil {
		return false, err
	}
	if p.ThreadWidth() != 32 || p.MaxThreads() < 128 {
		return false, fmt.Errorf("metalgemm: inkling2 draft head needs 4 SIMD groups of 32")
	}
	bytes := w.Rows * (w.Cols / block * blockBytes)
	return true, d.Dispatch(p, [3]int{(w.Rows + 31) / 32, 1, 1}, [3]int{128, 1, 1},
		Ink2BufferAt(w.Buf, w.ByteOff&^3, bytes+w.ByteOff&3), x.Span((tokens-1)*xStride+w.Cols), y.Span((tokens-1)*yStride+w.Rows),
		Ink2Const(ink2DF2HeadShape{rows: uint32(w.Rows), cols: uint32(w.Cols), byteOffset: uint32(w.ByteOff & 3), tokens: uint32(tokens),
			xStride: uint32(xStride), yStride: uint32(yStride)}))
}
