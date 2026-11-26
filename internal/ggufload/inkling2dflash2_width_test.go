package ggufload

import (
	"slices"
	"testing"
)

// run feeds the chooser a step outcome function and returns the chosen widths.
func runInkling2DFlash2Width(w *inkling2DFlash2Width, steps int, accepted func(i, n int) int) []int {
	var got []int
	for i := 0; i < steps; i++ {
		n := w.choose()
		got = append(got, n)
		if n > 0 {
			w.observe(n, accepted(i, n))
		}
	}
	return got
}

func TestInkling2DFlash2WidthDeterministic(t *testing.T) {
	t.Setenv("FAK_INKLING2_DFLASH2_DEPTH_FIXED", "")
	t.Setenv("FAK_INKLING2_VERIFY_MMA", "")
	acc := func(i, n int) int { return min(n, (i*7)%5) }
	a := runInkling2DFlash2Width(newInkling2DFlash2Width(7), 200, acc)
	b := runInkling2DFlash2Width(newInkling2DFlash2Width(7), 200, acc)
	if !slices.Equal(a, b) {
		t.Fatalf("same outcomes chose %v then %v", a, b)
	}
}

func TestInkling2DFlash2WidthCollapse(t *testing.T) {
	t.Setenv("FAK_INKLING2_DFLASH2_DEPTH_FIXED", "")
	for _, mma := range []string{"", "1"} {
		t.Setenv("FAK_INKLING2_VERIFY_MMA", mma)
		w := newInkling2DFlash2Width(7)
		if n := w.choose(); n == 0 {
			t.Fatalf("mma=%q: a fresh request drafts nothing", mma)
		}
		w.observe(w.pending, 0)
		got := runInkling2DFlash2Width(w, 40, func(int, int) int { return 0 })
		zeros := 0
		for _, n := range got {
			if n == 0 {
				zeros++
			}
		}
		if got[len(got)-1] != 0 && got[len(got)-2] != 0 {
			t.Errorf("mma=%q: zero acceptance still drafts %v", mma, got)
		}
		if zeros < len(got)*3/4 {
			t.Errorf("mma=%q: zero acceptance skipped only %d of %d steps: %v", mma, zeros, len(got), got)
		}
		// a collapse is not permanent: refused steps fade it back to the prior
		recovered := runInkling2DFlash2Width(w, 200, func(_, n int) int { return n })
		if recovered[len(recovered)-1] == 0 {
			t.Errorf("mma=%q: full acceptance never resumed drafting: %v", mma, recovered)
		}
	}
}

func TestInkling2DFlash2WidthDeepensWithAcceptance(t *testing.T) {
	t.Setenv("FAK_INKLING2_DFLASH2_DEPTH_FIXED", "")
	for _, mma := range []string{"", "1"} {
		t.Setenv("FAK_INKLING2_VERIFY_MMA", mma)
		hi := newInkling2DFlash2Width(7)
		gotHi := runInkling2DFlash2Width(hi, 60, func(_, n int) int { return n })
		mid := newInkling2DFlash2Width(7)
		gotMid := runInkling2DFlash2Width(mid, 60, func(_, n int) int { return min(n, 1) })
		// unjudged positions keep the prior, so depth grows a step at a time
		if last := gotHi[len(gotHi)-1]; last < 5 || last <= gotHi[0] {
			t.Errorf("mma=%q: full acceptance settles at depth %d from %d: %v", mma, last, gotHi[0], gotHi)
		}
		if gotMid[len(gotMid)-1] >= gotHi[len(gotHi)-1] {
			t.Errorf("mma=%q: one accept per step settles at %d, full acceptance at %d", mma, gotMid[len(gotMid)-1], gotHi[len(gotHi)-1])
		}
	}
	// MMA rows are cheaper, so the same acceptance drafts at least as deep
	t.Setenv("FAK_INKLING2_VERIFY_MMA", "")
	g := newInkling2DFlash2Width(7)
	t.Setenv("FAK_INKLING2_VERIFY_MMA", "1")
	m := newInkling2DFlash2Width(7)
	if ng, nm := g.choose(), m.choose(); nm < ng {
		t.Errorf("prior depth: MMA %d < GEMV %d", nm, ng)
	}
}

func TestInkling2DFlash2WidthFixed(t *testing.T) {
	t.Setenv("FAK_INKLING2_DFLASH2_DEPTH_FIXED", "4")
	w := newInkling2DFlash2Width(7)
	got := runInkling2DFlash2Width(w, 10, func(int, int) int { return 0 })
	for _, n := range got {
		if n != 4 {
			t.Fatalf("fixed depth 4 chose %v", got)
		}
	}
	t.Setenv("FAK_INKLING2_DFLASH2_DEPTH_FIXED", "9")
	if n := newInkling2DFlash2Width(7).choose(); n != 7 {
		t.Fatalf("fixed depth 9 over max 7 chose %d", n)
	}
}
