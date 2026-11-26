package main

import "testing"

// admitShape is a four-layer slice of the Inkling 2 Mini geometry: three local
// sliding-window layers and one global layer, no revisited core.
func admitShape() inkling2ResidentShape {
	return inkling2ResidentShape{
		Hidden: 256, Heads: 8, KVHeads: 2, HeadDim: 64,
		SlidingWindow: 512, ConvKernel: 4,
		CoreStart: 0, CoreEnd: 0, Passes: 1,
		Local: []bool{true, true, true, false},
	}
}

func TestInkling2AdmittedCtxNoRoomRefuses(t *testing.T) {
	if got := inkling2AdmittedCtx(admitShape(), 0); got != 0 {
		t.Fatalf("zero free bytes admitted ctx %d, want 0", got)
	}
}

// The fit must never admit a context the admission model prices above the memory it was
// given: the admitted session bytes must stay within free.
func TestInkling2AdmittedCtxPricedWithinFree(t *testing.T) {
	s := admitShape()
	prev := 0
	for _, free := range []int64{1 << 30, 2 << 30, 4 << 30, 8 << 30, 12 << 30, 16 << 30} {
		ctx := inkling2AdmittedCtx(s, free)
		if ctx < prev {
			t.Fatalf("free %d admitted ctx %d, below %d for less memory", free, ctx, prev)
		}
		prev = ctx
		if ctx == 0 {
			continue
		}
		if ctx%256 != 0 || ctx > inkling2ResidentCtxMax {
			t.Fatalf("free %d: ctx %d is off the 256 step or above the %d cap", free, ctx, inkling2ResidentCtxMax)
		}
		if priced := inkling2ResidentSessionBytes(s, ctx); priced > free {
			t.Fatalf("free %d: ctx %d prices %d bytes, above free", free, ctx, priced)
		}
	}
	if prev == 0 {
		t.Fatal("16 GiB free admitted no context")
	}
}
