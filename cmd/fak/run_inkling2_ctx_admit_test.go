package main

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/ggufload"
	"github.com/anthony-chaudhary/fak/internal/memgate"
)

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
		if ctx%256 != 0 || ctx > ggufload.Inkling2AutoCtxCeiling {
			t.Fatalf("free %d: ctx %d is off the 256 step or above the 16384 auto ceiling", free, ctx)
		}
		if priced := inkling2ResidentSessionBytes(s, ctx); priced > free {
			t.Fatalf("free %d: ctx %d prices %d bytes, above free", free, ctx, priced)
		}
	}
	if prev == 0 {
		t.Fatal("16 GiB free admitted no context")
	}
}

// A loaded Mac: live available memory is below the mapped weight plan, which made every
// default run refuse with a negative byte count before load.
func TestInkling2ResidentCtxLoadedHostAdmitsFloor(t *testing.T) {
	const gib = int64(1) << 30
	plan := int64(14 * gib)
	mem := memgate.Memory{TotalBytes: 36 * gib, WiredBytes: 24383160320, AvailableBytes: plan - 10878574048}
	ctx, err := inkling2ResidentCtxForHost(admitShape(), mem, plan)
	if err != nil {
		t.Fatalf("loaded host refused: %v", err)
	}
	if want := min(inkling2ResidentCtxFloor, inkling2ResidentCtxMax); ctx < want {
		t.Fatalf("loaded host admitted ctx %d, want >= %d", ctx, want)
	}
}

func TestInkling2ResidentCtxRoomyHostKeepsFit(t *testing.T) {
	const gib = int64(1) << 30
	plan := int64(14 * gib)
	mem := memgate.Memory{TotalBytes: 36 * gib, WiredBytes: 2 * gib, AvailableBytes: 30 * gib}
	ctx, err := inkling2ResidentCtxForHost(admitShape(), mem, plan)
	if err != nil {
		t.Fatal(err)
	}
	if want := inkling2AdmittedCtx(admitShape(), mem.AvailableBytes-plan); ctx != want {
		t.Fatalf("roomy host ctx %d, want live fit %d", ctx, want)
	}
}

// RAM that cannot hold the weights at all still refuses with a clear reason.
func TestInkling2ResidentCtxTinyHostRefuses(t *testing.T) {
	const gib = int64(1) << 30
	mem := memgate.Memory{TotalBytes: 8 * gib, WiredBytes: 2 * gib, AvailableBytes: 1 * gib}
	if ctx, err := inkling2ResidentCtxForHost(admitShape(), mem, 14*gib); err == nil {
		t.Fatalf("8 GiB host admitted ctx %d for a 14 GiB plan", ctx)
	}
}
