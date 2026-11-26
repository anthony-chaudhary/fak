package ggufload

import (
	"errors"
	"fmt"
)

// ink2RowCkpt is one cohort row's state at pos, held on the host in position order
// so it can be restored into a row of any later cohort, whatever that cohort's width,
// context or ring sizes. A global slot keeps every position below pos. A local slot
// keeps only the window below pos: attention never reads a local key older than the
// window, so those rows are all a later token can need. Conv histories are copied
// whole; they hold the last kernel-1 inputs and do not depend on the ring size.
type ink2RowCkpt struct {
	pos  int
	half bool
	// posFloats is the float width of one position's keys (and of its values)
	posFloats int
	slots     []ink2SlotCkpt
}

// ink2SlotCkpt is one slot's saved positions [lo, pos) of keys and values, and its
// conv history.
type ink2SlotCkpt struct {
	lo   int
	k, v []float32
	hist []float32
}

// errInkling2RowCkptShape is a checkpoint that cannot be restored into a row: the
// slot layout, the cache dtype or the row's context differs from the one it was
// taken in.
var errInkling2RowCkptShape = errors.New("inkling2 batch: row checkpoint does not fit this cohort")

// Bytes is the host memory the checkpoint holds.
func (ck *ink2RowCkpt) Bytes() int64 {
	var n int64
	for _, sl := range ck.slots {
		n += int64(len(sl.k) + len(sl.v) + len(sl.hist))
	}
	return 4 * n
}

// inkling2SlotSaveLo is the first position a slot must keep for a row at pos: 0 for
// a global slot, pos-window for a local one.
func inkling2SlotSaveLo(local bool, pos, window int) int {
	if !local {
		return 0
	}
	return max(0, pos-window)
}

// saveRow copies row's caches and histories at its current position to the host. A
// ring too small to still hold the row's window refuses rather than save a
// checkpoint that would restore stale keys.
func (b *inkling2Batch) saveRow(row int) (*ink2RowCkpt, error) {
	if row < 0 || row >= b.C {
		return nil, fmt.Errorf("inkling2 batch: save row %d of %d", row, b.C)
	}
	hp := b.s.hp
	kvDim := hp.KVHeads * hp.HeadDim
	pos := b.pos[row]
	ck := &ink2RowCkpt{pos: pos, half: b.kvHalf, posFloats: inkling2BatchKVFloats(kvDim, b.kvHalf), slots: make([]ink2SlotCkpt, len(b.caps))}
	for slot, cap := range b.caps {
		local := inkling2LocalRing(hp, slot)
		lo := inkling2SlotSaveLo(local, pos, hp.SlidingWindow)
		if pos-lo > cap {
			return nil, fmt.Errorf("%w: slot %d ring of %d rows cannot hold positions %d..%d", errInkling2RowCkptShape, slot, cap, lo, pos)
		}
		rowOff := inkling2CohortKVRow(row, cap, kvDim, b.kvHalf)
		sc := ink2SlotCkpt{lo: lo, k: make([]float32, 0, (pos-lo)*ck.posFloats), v: make([]float32, 0, (pos-lo)*ck.posFloats)}
		for _, sp := range inkling2RingSpans(lo, pos, cap) {
			off, n := rowOff+sp.ring*ck.posFloats, sp.n*ck.posFloats
			k, err := b.s.dev.Ink2Read(at(b.kc[slot], off), n)
			if err != nil {
				return nil, err
			}
			v, err := b.s.dev.Ink2Read(at(b.vc[slot], off), n)
			if err != nil {
				return nil, err
			}
			sc.k, sc.v = append(sc.k, k...), append(sc.v, v...)
		}
		hist, err := b.s.dev.Ink2Read(at(b.hist[slot], row*b.histStride), b.histStride)
		if err != nil {
			return nil, err
		}
		sc.hist = hist
		ck.slots[slot] = sc
	}
	return ck, nil
}

// restoreRow writes ck into row and leaves the row at ck.pos, ready to prefill or
// step from there. Each saved position lands on its ring row in this cohort's rings,
// so a checkpoint taken under one ring size restores under another.
func (b *inkling2Batch) restoreRow(row int, ck *ink2RowCkpt) error {
	if row < 0 || row >= b.C {
		return fmt.Errorf("inkling2 batch: restore row %d of %d", row, b.C)
	}
	hp := b.s.hp
	kvDim := hp.KVHeads * hp.HeadDim
	if ck == nil || ck.half != b.kvHalf || len(ck.slots) != len(b.caps) || ck.pos >= b.ctx ||
		ck.posFloats != inkling2BatchKVFloats(kvDim, b.kvHalf) {
		return errInkling2RowCkptShape
	}
	for slot, cap := range b.caps {
		sc := ck.slots[slot]
		local := inkling2LocalRing(hp, slot)
		if sc.lo != inkling2SlotSaveLo(local, ck.pos, hp.SlidingWindow) || ck.pos-sc.lo > cap || len(sc.hist) != b.histStride {
			return errInkling2RowCkptShape
		}
	}
	for slot, cap := range b.caps {
		sc := ck.slots[slot]
		rowOff := inkling2CohortKVRow(row, cap, kvDim, b.kvHalf)
		for _, sp := range inkling2RingSpans(sc.lo, ck.pos, cap) {
			off, from, n := rowOff+sp.ring*ck.posFloats, sp.off*ck.posFloats, sp.n*ck.posFloats
			if err := b.s.dev.Ink2Write(at(b.kc[slot], off), sc.k[from:from+n]); err != nil {
				return err
			}
			if err := b.s.dev.Ink2Write(at(b.vc[slot], off), sc.v[from:from+n]); err != nil {
				return err
			}
		}
		if err := b.s.dev.Ink2Write(at(b.hist[slot], row*b.histStride), sc.hist); err != nil {
			return err
		}
	}
	b.pos[row] = ck.pos
	return nil
}
