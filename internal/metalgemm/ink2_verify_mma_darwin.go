//go:build darwin && arm64 && cgo

package metalgemm

import (
	_ "embed"
	"fmt"
	"math"
	"os"
	"strconv"
	"sync"
)

//go:embed ink2_verify_mma.metal
var ink2VerifyMMA string

var ink2VerifyMMASource = sync.OnceValue(func() string {
	return packedCodebooks + packedDequant + packedGEMV + ink2VerifyMMA
})

const (
	// Activation rows one MMA pass covers (INK2VM_ROWS), weight rows an MMA
	// threadgroup owns, and weight rows a scalar threadgroup owns.
	ink2VerifyMMARows       = 16
	ink2VerifyMMATile       = 8
	ink2VerifyScalarTile    = 16
	ink2VerifyMMAThreads    = 128
	ink2VerifyMMAScalarRows = 4
	// The kernels' fixed sum order runs in 64-column groups.
	ink2VerifyMMACols = 64
)

type ink2VerifyMMARowsShape struct{ tokens, xStride, yStride, scalarMax uint32 }

type ink2VerifyMMAExpertsShape struct{ npairs, xStride, xDiv, scalarMax uint32 }

// ink2VerifyMMAScalarMax is the widest pass (activation rows, or pairs of one
// expert) the kernels run in the scalar form rather than on MMAs; both give
// the same bits, so it only moves time. The MMA form measured faster at every
// width on an M3 Pro, so it is 0 unless FAK_INK2VM_SCALAR_MAX (at most 4) sets
// it.
var ink2VerifyMMAScalarMax = sync.OnceValue(func() int {
	if n, err := strconv.Atoi(os.Getenv("FAK_INK2VM_SCALAR_MAX")); err == nil && n >= 0 {
		return min(n, ink2VerifyMMAScalarRows)
	}
	return 0
})

func (d *Ink2Device) verifyMMAPipeline(kernels *[41]string, w Ink2Weight, x Ink2Ref, xStride int) (Ink2Pipeline, int, bool, error) {
	if int(w.Type) >= len(kernels) || kernels[w.Type] == "" || w.Cols%ink2VerifyMMACols != 0 || x.Off%4 != 0 || xStride%4 != 0 {
		return Ink2Pipeline{}, 0, false, nil
	}
	block, blockBytes, err := packedBlock(w.Type)
	if err != nil || w.Cols%block != 0 {
		return Ink2Pipeline{}, 0, false, err
	}
	p, err := d.verifyMMAPipelineNamed(kernels[w.Type])
	return p, w.Cols / block * blockBytes, err == nil, err
}

func (d *Ink2Device) verifyMMAPipelineNamed(name string) (Ink2Pipeline, error) {
	p, err := d.Pipeline(ink2VerifyMMASource(), name)
	if err != nil {
		return Ink2Pipeline{}, err
	}
	if p.width != 32 || p.maxThreads < ink2VerifyMMAThreads {
		return Ink2Pipeline{}, fmt.Errorf("metalgemm: inkling2 kernel %s needs 32-wide SIMD groups of %d threads", name, ink2VerifyMMAThreads)
	}
	return p, nil
}

// VerifyRowsMMA is VerifyRows on simdgroup_matrix products: a row's bits do
// not depend on rows, but they are not the decode GEMV's. It declines (false,
// nothing encoded) a weight type or layout it has no kernel for.
func (d *Ink2Device) VerifyRowsMMA(w Ink2Weight, x Ink2Ref, xStride int, y Ink2Ref, yStride, rows int) (bool, error) {
	if !w.Buf.Valid() || w.ByteOff < 0 || w.Rows < 1 || w.Rows > math.MaxInt32 || w.Cols < 1 || w.Cols > math.MaxInt32 || w.Experts < 1 {
		return false, fmt.Errorf("%w: weight type %d [%d %d] x%d", ErrInk2Shape, w.Type, w.Rows, w.Cols, w.Experts)
	}
	if !x.Valid() || !y.Valid() || rows < 1 || xStride < w.Cols || yStride < w.Rows ||
		rows > math.MaxInt32/max(xStride, yStride) || ink2OpsOverlap(x, (rows-1)*xStride+w.Cols, y, (rows-1)*yStride+w.Rows) {
		return false, fmt.Errorf("%w: verify mma rows of %d rows, strides %d and %d", ErrInk2Shape, rows, xStride, yStride)
	}
	// Both forms give the same bits: the width picks the faster one.
	kernels, grid := &ink2VerifyMMARowsKernel, [3]int{(w.Rows + ink2VerifyMMATile - 1) / ink2VerifyMMATile, (rows + ink2VerifyMMARows - 1) / ink2VerifyMMARows, 1}
	if rows <= ink2VerifyMMAScalarMax() {
		kernels, grid = &ink2VerifyScalarRowsKernel, [3]int{(w.Rows + ink2VerifyScalarTile - 1) / ink2VerifyScalarTile, 1, 1}
	}
	p, rowBytes, ok, err := d.verifyMMAPipeline(kernels, w, x, xStride)
	if err != nil || !ok {
		return false, err
	}
	d.profileDetail("mma %dx%d rows=%d", w.Rows, w.Cols, rows)
	args := append(w.args(x.Span((rows-1)*xStride+w.Cols), y.Span((rows-1)*yStride+w.Rows), w.Rows*rowBytes),
		Ink2Const(ink2VerifyMMARowsShape{uint32(rows), uint32(xStride), uint32(yStride), uint32(ink2VerifyMMAScalarMax())}))
	return true, d.Dispatch(p, grid, [3]int{ink2VerifyMMAThreads, 1, 1}, args...)
}

// VerifyExpertsMMA is VerifyExperts on simdgroup_matrix products: pair i's
// bits do not depend on how the chunk routes or how wide it is, but they are
// not ExpertMatVec's. It declines (false, nothing encoded) a weight type or
// layout it has no kernel for.
func (d *Ink2Device) VerifyExpertsMMA(w Ink2Weight, ids, x Ink2Ref, xDiv, xStride int, y Ink2Ref, npairs int, scale, tab Ink2Ref) (bool, error) {
	if !w.Buf.Valid() || w.ByteOff < 0 || w.Rows < 1 || w.Rows > math.MaxInt32 || w.Cols < 1 || w.Cols > math.MaxInt32 || w.Experts < 1 {
		return false, fmt.Errorf("%w: weight type %d [%d %d] x%d", ErrInk2Shape, w.Type, w.Rows, w.Cols, w.Experts)
	}
	if !ids.Valid() || !x.Valid() || !y.Valid() || !tab.Valid() || npairs < 1 || npairs > ink2VerifyGroupMax || xDiv < 1 ||
		xStride < w.Cols || xStride > math.MaxUint32 || npairs > math.MaxInt32/max(w.Rows, xStride) {
		return false, fmt.Errorf("%w: verify mma experts of %d pairs, input stride %d every %d pairs", ErrInk2Shape, npairs, xStride, xDiv)
	}
	p, rowBytes, ok, err := d.verifyMMAPipeline(&ink2VerifyMMAExpertsKernel, w, x, xStride)
	if err != nil || !ok {
		return false, err
	}
	gp, err := d.Pipeline(ink2VerifyRowsSource(), "ink2v_group_pairs")
	if err != nil {
		return false, err
	}
	shape := Ink2Const(ink2VerifyMMAExpertsShape{uint32(npairs), uint32(xStride), uint32(xDiv), uint32(ink2VerifyMMAScalarMax())})
	tabSpan := tab.Span(VerifyExpertsTable(npairs))
	if err := d.Dispatch(gp, [3]int{1, 1, 1}, [3]int{min(npairs, gp.MaxThreads()), 1, 1}, ids.Span(npairs), tabSpan, shape); err != nil {
		return false, err
	}
	d.profileDetail("mma %dx%d pairs=%d of %d", w.Rows, w.Cols, npairs, w.Experts)
	xRows := (npairs-1)/xDiv + 1
	args := append(w.args(x.Span((xRows-1)*xStride+w.Cols), y.Span(npairs*w.Rows), w.Experts*w.Rows*rowBytes), tabSpan, shape)
	// Each form takes the groups of its widths; both give the same bits.
	if ink2VerifyMMAScalarMax() > 0 {
		sp, err := d.verifyMMAPipelineNamed(ink2VerifyScalarExpertsKernel[w.Type])
		if err != nil {
			return false, err
		}
		grid := [3]int{(w.Rows + ink2VerifyScalarTile - 1) / ink2VerifyScalarTile, npairs, 1}
		if err := d.Dispatch(sp, grid, [3]int{ink2VerifyMMAThreads, 1, 1}, args...); err != nil {
			return false, err
		}
	}
	grid := [3]int{(w.Rows + ink2VerifyMMATile - 1) / ink2VerifyMMATile, npairs, 1}
	if err := d.Dispatch(p, grid, [3]int{ink2VerifyMMAThreads, 1, 1}, args...); err != nil {
		return false, err
	}
	if !scale.Valid() {
		return true, nil
	}
	return true, d.ExpertScale(y, ids, scale, npairs, w.Rows)
}
