package ggufload

import (
	"errors"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/model"
)

// ckptBatch is a host batch of C rows over a local slot (ring of caps[0] rows, window
// 3) and a global slot, its arenas tracked by the cohort fake.
func ckptBatch(t *testing.T, d *fakeInk2Cohort, C, ctx int, caps []int, half bool) *inkling2Batch {
	t.Helper()
	const kvDim, stride = 4, 3
	hp := model.Inkling2LoopHParams{Layers: 2, LocalLayer: []bool{true, false}, SlidingWindow: 3, KVHeads: 1, HeadDim: kvDim}
	b := &inkling2Batch{s: &inkling2Resident{be: d, dev: d, hp: hp}, C: C, ctx: ctx, pos: make([]int, C), histStride: stride, caps: caps, kvHalf: half}
	alloc := func(n int) ink2Ref {
		tensor, err := d.Ink2Alloc(n)
		if err != nil {
			t.Fatal(err)
		}
		return compute.Ink2At(tensor, 0)
	}
	for _, cap := range caps {
		b.kc = append(b.kc, alloc(inkling2BatchKVFloats(C*cap*kvDim, half)))
		b.vc = append(b.vc, alloc(inkling2BatchKVFloats(C*cap*kvDim, half)))
		b.hist = append(b.hist, alloc(C*stride))
	}
	return b
}

// ckptTag is the value element j of position p's key (v=0) or value (v=1) holds in a
// slot, distinct for every slot, position, element and plane.
func ckptTag(slot, p, j, v int) float32 { return float32(10000*slot+10*p+j) + 0.5*float32(v) }

// feed writes positions 0..pos-1 of row into every slot the way the cache append does,
// position p on ring row p%cap, so a short ring wraps and keeps only its last cap.
func ckptFeed(d *fakeInk2Cohort, b *inkling2Batch, row, pos int) {
	pf := inkling2BatchKVFloats(b.s.hp.KVHeads*b.s.hp.HeadDim, b.kvHalf)
	kvDim := b.s.hp.KVHeads * b.s.hp.HeadDim
	for slot, cap := range b.caps {
		base := inkling2CohortKVRow(row, cap, kvDim, b.kvHalf)
		for p := 0; p < pos; p++ {
			for j := 0; j < pf; j++ {
				d.rows(b.kc[slot])[base+(p%cap)*pf+j] = ckptTag(slot, p, j, 0)
				d.rows(b.vc[slot])[base+(p%cap)*pf+j] = ckptTag(slot, p, j, 1)
			}
		}
		for j := 0; j < b.histStride; j++ {
			d.rows(b.hist[slot])[row*b.histStride+j] = float32(-1000*slot - 10*pos - j)
		}
	}
	b.pos[row] = pos
}

// A row saved from a ring that has wrapped twice restores into another cohort's
// row with a different ring size, context and width: every window position of the
// local slot sits on its ring row there, every global position below pos is copied,
// the histories match, and no other row is touched. Both cache dtypes move whole
// positions (an f16 position is half as many floats).
func TestInkling2BatchRowCkptRestoresAcrossRingSizes(t *testing.T) {
	for _, half := range []bool{false, true} {
		d := newFakeInk2Cohort()
		src := ckptBatch(t, d, 2, 16, []int{5, 16}, half)
		const pos = 11
		ckptFeed(d, src, 1, pos)
		ck, err := src.saveRow(1)
		if err != nil {
			t.Fatalf("half=%v: save: %v", half, err)
		}
		pf := ck.posFloats
		if got := len(ck.slots[0].k); got != 3*pf || ck.slots[0].lo != pos-3 {
			t.Fatalf("half=%v: local slot saved %d floats from %d, want the 3-position window from %d", half, got, ck.slots[0].lo, pos-3)
		}
		if got := len(ck.slots[1].k); got != pos*pf {
			t.Fatalf("half=%v: global slot saved %d floats, want all %d positions", half, got, pos)
		}
		dst := ckptBatch(t, d, 3, 24, []int{7, 24}, half)
		for slot := range dst.caps {
			for _, r := range []ink2Ref{dst.kc[slot], dst.vc[slot], dst.hist[slot]} {
				for j := range d.rows(r) {
					d.rows(r)[j] = -7
				}
			}
		}
		if err := dst.restoreRow(2, ck); err != nil {
			t.Fatalf("half=%v: restore: %v", half, err)
		}
		if dst.pos[2] != pos {
			t.Fatalf("half=%v: restored row at %d, want %d", half, dst.pos[2], pos)
		}
		kvDim := 4
		for slot, cap := range dst.caps {
			lo := 0
			if slot == 0 {
				lo = pos - 3
			}
			base := inkling2CohortKVRow(2, cap, kvDim, half)
			for p := lo; p < pos; p++ {
				for j := 0; j < pf; j++ {
					if got := d.rows(dst.kc[slot])[base+(p%cap)*pf+j]; got != ckptTag(slot, p, j, 0) {
						t.Fatalf("half=%v slot %d position %d key %d = %v, want %v", half, slot, p, j, got, ckptTag(slot, p, j, 0))
					}
					if got := d.rows(dst.vc[slot])[base+(p%cap)*pf+j]; got != ckptTag(slot, p, j, 1) {
						t.Fatalf("half=%v slot %d position %d value %d = %v", half, slot, p, j, got)
					}
				}
			}
			for j := 0; j < dst.histStride; j++ {
				if got, want := d.rows(dst.hist[slot])[2*dst.histStride+j], float32(-1000*slot-10*pos-j); got != want {
					t.Fatalf("half=%v slot %d hist %d = %v, want %v", half, slot, j, got, want)
				}
			}
			rowLen := inkling2CohortKVRow(1, cap, kvDim, half)
			for j, v := range d.rows(dst.kc[slot])[:2*rowLen] {
				if v != -7 {
					t.Fatalf("half=%v slot %d: row %d element %d written by a restore into row 2", half, slot, j/rowLen, j%rowLen)
				}
			}
		}
	}
}

// A checkpoint is refused where it cannot be the row's state: a cache of the other
// dtype, a context the position does not fit below, or a ring too short to hold the
// window it was saved with; a save from a ring shorter than the window is refused too.
func TestInkling2BatchRowCkptRefusesMismatch(t *testing.T) {
	d := newFakeInk2Cohort()
	src := ckptBatch(t, d, 1, 16, []int{5, 16}, false)
	ckptFeed(d, src, 0, 9)
	ck, err := src.saveRow(0)
	if err != nil {
		t.Fatal(err)
	}
	for name, dst := range map[string]*inkling2Batch{
		"f16 cache":   ckptBatch(t, d, 1, 16, []int{5, 16}, true),
		"short ctx":   ckptBatch(t, d, 1, 9, []int{5, 9}, false),
		"short ring":  ckptBatch(t, d, 1, 16, []int{2, 16}, false),
		"other slots": ckptBatch(t, d, 1, 16, []int{5}, false),
	} {
		if err := dst.restoreRow(0, ck); !errors.Is(err, errInkling2RowCkptShape) {
			t.Errorf("%s: restore err %v, want the shape refusal", name, err)
		}
		if dst.pos[0] != 0 {
			t.Errorf("%s: a refused restore moved the row to %d", name, dst.pos[0])
		}
	}
	short := ckptBatch(t, d, 1, 16, []int{2, 16}, false)
	ckptFeed(d, short, 0, 9)
	if _, err := short.saveRow(0); !errors.Is(err, errInkling2RowCkptShape) {
		t.Errorf("save from a 2-row ring under a 3-position window: err %v, want the shape refusal", err)
	}
}
