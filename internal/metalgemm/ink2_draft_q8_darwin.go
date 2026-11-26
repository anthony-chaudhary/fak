//go:build darwin && arm64 && cgo

package metalgemm

import (
	"fmt"
	"os"
	"sync"
	"unsafe"
)

// DraftQ8Env opts the BF16 draft linears into per-row int8 weights, which
// halves the bandwidth-bound draft's weight bytes at a small accuracy cost.
const DraftQ8Env = "FAK_INKLING2_DFLASH2_Q8"

// ink2DFQ8Rows is the weight rows per int8 linear threadgroup.
const ink2DFQ8Rows = 2

// ink2DFQ8MMA reports whether an int8 linear takes the SIMD-group matrix
// kernels, which cover full 8-row blocks of 64-column stages with 32 weight
// rows per threadgroup of 4 SIMD groups; the vector kernels do the rest.
func ink2DFQ8MMA(seg, tokens int) bool { return tokens == ink2DFMaxTokens && seg%64 == 0 }

const ink2DFQ8MMARows = 32

// ink2DFQ8Simds covers a segment of units 16-column lane loads in one pass
// where the SIMD-group cap allows.
func ink2DFQ8Simds(units int) int { return max(1, min(16, (units+31)/32)) }

type ink2DFQ8Key struct {
	buf             unsafe.Pointer
	off, rows, cols int
}

// ink2DFQ8Weight is an int8 [rows][cols] copy of a BF16 weight at byte 0 of
// buf, with its f32 row scales at byte rows*cols.
type ink2DFQ8Weight struct {
	buf        Ink2Buf
	rows, cols int
}

func (w ink2DFQ8Weight) data() Ink2Arg   { return Ink2BufferAt(w.buf, 0, w.rows*w.cols) }
func (w ink2DFQ8Weight) scales() Ink2Arg { return Ink2BufferAt(w.buf, w.rows*w.cols, 4*w.rows) }

// ink2DFState is the per-device draft state the Ink2Device struct does not
// carry: test overrides, the int8 weight cache and the split attention scratch.
type ink2DFState struct {
	q8, split int8 // 0 follows the default, 1 forces on, -1 off
	cache     map[ink2DFQ8Key]ink2DFQ8Weight
	scratch   Ink2Buf
	scratchN  int
}

var ink2DFStates struct {
	sync.Mutex
	m map[*Ink2Device]*ink2DFState
}

func (d *Ink2Device) dfState() *ink2DFState {
	ink2DFStates.Lock()
	defer ink2DFStates.Unlock()
	if ink2DFStates.m == nil {
		ink2DFStates.m = map[*Ink2Device]*ink2DFState{}
	}
	st := ink2DFStates.m[d]
	if st == nil {
		st = &ink2DFState{cache: map[ink2DFQ8Key]ink2DFQ8Weight{}}
		ink2DFStates.m[d] = st
	}
	return st
}

func (d *Ink2Device) draftQ8() bool {
	switch d.dfState().q8 {
	case 1:
		return true
	case -1:
		return false
	}
	return os.Getenv(DraftQ8Env) == "1"
}

// draftQ8Weight returns the int8 copy of a BF16 weight, quantizing it on the
// device on first use; ok is false for a shape the int8 kernels cannot read.
func (d *Ink2Device) draftQ8Weight(w Ink2Weight) (ink2DFQ8Weight, bool, error) {
	if w.Type != 30 || w.ByteOff%16 != 0 || w.Cols%16 != 0 || !w.Buf.Valid() {
		return ink2DFQ8Weight{}, false, nil
	}
	st := d.dfState()
	key := ink2DFQ8Key{w.Buf.h, w.ByteOff, w.Rows, w.Cols}
	ink2DFStates.Lock()
	q, hit := st.cache[key]
	ink2DFStates.Unlock()
	if hit {
		return q, true, nil
	}
	warg, shift, err := ink2DFBF16(w, "draft int8 weight")
	if err != nil {
		return ink2DFQ8Weight{}, false, err
	}
	buf, err := d.Alloc((w.Rows*w.Cols+3)/4 + w.Rows)
	if err != nil {
		return ink2DFQ8Weight{}, false, err
	}
	q = ink2DFQ8Weight{buf: buf, rows: w.Rows, cols: w.Cols}
	p, err := d.PipelinePrecise(ink2DraftSource, "ink2_df_q8_quantize")
	if err != nil {
		d.Free(buf)
		return ink2DFQ8Weight{}, false, err
	}
	if p.ThreadWidth() != 32 || p.MaxThreads() < 256 {
		d.Free(buf)
		return ink2DFQ8Weight{}, false, fmt.Errorf("metalgemm: inkling2 draft int8 quantize needs 256 threads")
	}
	if err := d.Dispatch(p, [3]int{w.Rows, 1, 1}, [3]int{256, 1, 1}, warg, q.data(), q.scales(),
		Ink2Const(ink2DFLinearShape{outRows: uint32(w.Rows), cols: uint32(w.Cols), shift: shift})); err != nil {
		d.Free(buf)
		return ink2DFQ8Weight{}, false, err
	}
	ink2DFStates.Lock()
	st.cache[key] = q
	ink2DFStates.Unlock()
	return q, true, nil
}

// draftQ8Weights is draftQ8Weight of every weight, ok only when all convert.
func (d *Ink2Device) draftQ8Weights(w []Ink2Weight) ([]ink2DFQ8Weight, bool, error) {
	if !d.draftQ8() {
		return nil, false, nil
	}
	qs := make([]ink2DFQ8Weight, len(w))
	for i := range w {
		q, ok, err := d.draftQ8Weight(w[i])
		if err != nil || !ok {
			return nil, false, err
		}
		qs[i] = q
	}
	return qs, true, nil
}

// draftLinearQ8 is the int8 DraftLinear; xargs are the vector kernel's bound
// input segments.
func (d *Ink2Device) draftLinearQ8(xargs []Ink2Arg, q ink2DFQ8Weight, y Ink2Ref, seg, segs, tokens int) error {
	simds, perGroup, name := ink2DFQ8Simds(seg/16), ink2DFQ8Rows, fmt.Sprintf("ink2_df_linear_q8_t%d", ink2DFBucket(tokens))
	if ink2DFQ8MMA(seg, tokens) {
		simds, perGroup, name = 4, ink2DFQ8MMARows, "ink2_df_linear_q8_mma"
	}
	p, err := d.Pipeline(ink2DraftSource, name)
	if err != nil {
		return err
	}
	if p.ThreadWidth() != 32 || p.MaxThreads() < 32*simds {
		return fmt.Errorf("metalgemm: inkling2 draft linear needs %d SIMD groups of 32", simds)
	}
	args := append([]Ink2Arg{q.data()}, xargs...)
	args = append(args, y.Span(tokens*q.rows), Ink2Const(ink2DFLinearShape{
		outRows: uint32(q.rows), cols: uint32(q.cols), seg: uint32(seg), segs: uint32(segs), tokens: uint32(tokens)}), q.scales())
	return d.Dispatch(p, [3]int{(q.rows + perGroup - 1) / perGroup, 1, 1}, [3]int{32 * simds, 1, 1}, args...)
}

func (d *Ink2Device) draftLinearMultiQ8(xargs []Ink2Arg, q []ink2DFQ8Weight, y []Ink2Ref, seg, segs, tokens int) error {
	var wargs, yargs, sargs [3]Ink2Arg
	var rows [3]int
	for i := range 3 {
		j := min(i, len(q)-1)
		wargs[i], yargs[i], sargs[i], rows[i] = q[j].data(), y[j].Span(tokens*q[j].rows), q[j].scales(), q[j].rows
		if i >= len(q) {
			rows[i] = 0
		}
	}
	simds, perGroup, name := ink2DFQ8Simds(seg/16), ink2DFQ8Rows, fmt.Sprintf("ink2_df_linear_multi_q8_t%d", ink2DFBucket(tokens))
	if ink2DFQ8MMA(seg, tokens) {
		simds, perGroup, name = 4, ink2DFQ8MMARows, "ink2_df_linear_multi_q8_mma"
	}
	p, err := d.Pipeline(ink2DraftSource, name)
	if err != nil {
		return err
	}
	if p.ThreadWidth() != 32 || p.MaxThreads() < 32*simds {
		return fmt.Errorf("metalgemm: inkling2 draft linear needs %d SIMD groups of 32", simds)
	}
	g := func(r int) int { return (r + perGroup - 1) / perGroup }
	end0, end1 := g(rows[0]), g(rows[0])+g(rows[1])
	args := append(wargs[:], xargs...)
	args = append(args, yargs[:]...)
	args = append(args, Ink2Const(ink2DFMultiShape{rows0: uint32(rows[0]), rows1: uint32(rows[1]), rows2: uint32(rows[2]),
		end0: uint32(end0), end1: uint32(end1), cols: uint32(q[0].cols), seg: uint32(seg), segs: uint32(segs), tokens: uint32(tokens)}))
	args = append(args, sargs[:]...)
	return d.Dispatch(p, [3]int{end1 + g(rows[2]), 1, 1}, [3]int{32 * simds, 1, 1}, args...)
}

func (d *Ink2Device) draftLinearSiluQ8(x Ink2Arg, gate, up ink2DFQ8Weight, y Ink2Ref, tokens int) error {
	simds, perGroup, name := ink2DFQ8Simds(gate.cols/16), ink2DFQ8Rows, fmt.Sprintf("ink2_df_linear_silu_q8_t%d", ink2DFBucket(tokens))
	if ink2DFQ8MMA(gate.cols, tokens) {
		simds, perGroup, name = 4, ink2DFQ8MMARows, "ink2_df_linear_silu_q8_mma"
	}
	p, err := d.PipelinePrecise(ink2DraftSource, name)
	if err != nil {
		return err
	}
	if p.ThreadWidth() != 32 || p.MaxThreads() < 32*simds {
		return fmt.Errorf("metalgemm: inkling2 draft gated linear needs %d SIMD groups of 32", simds)
	}
	return d.Dispatch(p, [3]int{(gate.rows + perGroup - 1) / perGroup, 1, 1}, [3]int{32 * simds, 1, 1},
		gate.data(), up.data(), x, y.Span(tokens*gate.rows),
		Ink2Const(ink2DFLinearShape{outRows: uint32(gate.rows), cols: uint32(gate.cols), seg: uint32(gate.cols), segs: 1, tokens: uint32(tokens)}),
		gate.scales(), up.scales())
}

// The key-split attention's keys per threadgroup and most splits per query.
const (
	ink2DFSplitKeys = 64
	ink2DFSplitMax  = 64
)

type ink2DFAttendSplitShape struct {
	rows, heads, kvHeads, dim, window, cap uint32
	scale                                  float32
	splits                                 uint32
}

// draftAttendScratch returns a device scratch of at least n floats, kept per
// device; the serial encoder orders every launch that reuses it.
func (d *Ink2Device) draftAttendScratch(n int) (Ink2Ref, error) {
	st := d.dfState()
	ink2DFStates.Lock()
	defer ink2DFStates.Unlock()
	if st.scratchN < n {
		b, err := d.Alloc(n)
		if err != nil {
			return Ink2Ref{}, err
		}
		d.Free(st.scratch)
		st.scratch, st.scratchN = b, n
	}
	return Ink2Ref{Buf: st.scratch}, nil
}
