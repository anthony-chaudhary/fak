package ggufload

import (
	"errors"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/model"
)

// inkling2MiniKVShape is the Inkling-2-Mini cache geometry: 24 layers with the loop
// over core layers 6..18 run 3 passes (48 slots), global attention at layers
// 5, 11, 17, 23 (8 global slots), 8 KV heads of 128, 24 query heads and a 1024
// sliding window. Hidden and ConvKernel are test stand-ins: they set only the conv
// history term, which the global-slot arithmetic does not depend on.
func inkling2MiniKVShape() model.Inkling2LoopHParams {
	local := make([]bool, 24)
	for i := range local {
		local[i] = true
	}
	for _, g := range []int{5, 11, 17, 23} {
		local[g] = false
	}
	return model.Inkling2LoopHParams{
		Hidden: 2048, Layers: 24, Heads: 24, KVHeads: 8, HeadDim: 128,
		SlidingWindow: 1024, ConvKernel: 4,
		CoreStart: 6, CoreEnd: 18, Passes: 3, LocalLayer: local,
	}
}

// inkling2MiniOther is the non-KV device footprint of the Mini session with no wide
// arenas: the input row and the per-token arena. 4*(2048+4+16Mi+4) bytes.
func inkling2MiniOther(hp model.Inkling2LoopHParams) int64 {
	return inkling2ResidentOtherBytes(hp, inkling2ArenaFloats, 0, 0)
}

func TestInkling2ResidentKVBytesMiniGlobalSlotIs64KiBPerToken(t *testing.T) {
	hp := inkling2MiniKVShape()
	if got := len(inkling2ResidentCaps(hp, 1, false, 0)); got != 48 {
		t.Fatalf("slots = %d, want 48", got)
	}
	// one token of the 8 global slots: K and V, 1024 float32 each, 4 bytes per element
	perGlobalToken := int64(8) * 2 * 1024 * 4
	if perGlobalToken != 64<<10 {
		t.Fatalf("global KV per token = %d bytes, want 64 KiB", perGlobalToken)
	}
	// the marginal token also grows the shared score row (2*Heads float32 per cache position)
	delta := inkling2ResidentKVBytes(hp, 23553, false, 0) - inkling2ResidentKVBytes(hp, 23552, false, 0)
	if want := perGlobalToken + 2*24*4; delta != want {
		t.Fatalf("marginal bytes per token = %d, want %d", delta, want)
	}
}

func TestInkling2ResidentKVBytesMiniExactTotals(t *testing.T) {
	hp := inkling2MiniKVShape()
	// local slots hold the 1024-token window, so the total is 65728*ctx plus a constant:
	// 8 global slots of ctx, 40 local slots of window, 48 conv histories, 48*ctx score bytes.
	cases := map[int]int64{
		23552:  1887109120,
		49152:  3569745920,
		131072: 8954183680,
	}
	for ctx, want := range cases {
		if got := inkling2ResidentKVBytes(hp, ctx, false, 0); got != want {
			t.Errorf("ctx %d: KV bytes = %d, want %d", ctx, got, want)
		}
	}
}

func TestInkling2ResidentKVBytesZeroCtxAllocatesNothing(t *testing.T) {
	if got := inkling2ResidentKVBytes(inkling2MiniKVShape(), 0, true, 1024); got != 0 {
		t.Fatalf("ctx 0 KV bytes = %d, want 0", got)
	}
}

func TestInkling2ResidentOtherBytesFixedAndWide(t *testing.T) {
	hp := inkling2MiniKVShape()
	if got, want := inkling2MiniOther(hp), int64(4*(2048+4+inkling2ArenaFloats+4)); got != want {
		t.Fatalf("fixed bytes = %d, want %d", got, want)
	}
	// 48 visits plus 2 times (T*Hidden+64) floats for keep: 50*(512*2048+64)
	keep, scr := inkling2PrefillArenaFloats(hp, 512, 0)
	if keep != 52432000 {
		t.Fatalf("keep floats at T 512 = %d, want 52432000", keep)
	}
	wide := inkling2ResidentOtherBytes(hp, inkling2ArenaFloats, 512, 0) - inkling2MiniOther(hp)
	if want := 4 * int64(keep+scr+512*hp.Hidden); wide != want {
		t.Fatalf("wide arena bytes = %d, want %d", wide, want)
	}
}

func TestInkling2ResidentTotalBytesIsKVPlusOther(t *testing.T) {
	hp := inkling2MiniKVShape()
	want := int64(1887109120) + inkling2MiniOther(hp)
	if got := inkling2ResidentTotalBytes(hp, 23552, false, 0, inkling2ArenaFloats, 0, 0); got != want {
		t.Fatalf("total = %d, want KV plus fixed %d", got, want)
	}
}

func TestInkling2DeviceHeadroomRule(t *testing.T) {
	const gib = int64(1) << 30
	if got := inkling2DeviceHeadroom(2 * gib); got != 2*gib*15/100 {
		t.Errorf("2 GiB free headroom = %d, want 15%%", got)
	}
	if got := inkling2DeviceHeadroom(16 * gib); got != gib {
		t.Errorf("16 GiB free headroom = %d, want the 1 GiB cap", got)
	}
}

func TestInkling2KVFitCeilingFromFreeBytes(t *testing.T) {
	hp := inkling2MiniKVShape()
	const free = int64(2) << 30
	budget := free - inkling2DeviceHeadroom(free) // 1825361101 bytes
	// KV only: 65728 bytes per token plus the fixed 339083264-byte remainder
	const wantMax = 22612
	if got := inkling2MaxResidentCtx(hp, false, 0, 49152, budget); got != wantMax {
		t.Fatalf("max ctx = %d, want %d", got, wantMax)
	}
	if need := inkling2ResidentKVBytes(hp, wantMax, false, 0); need > budget {
		t.Fatalf("ctx %d needs %d, over budget %d", wantMax, need, budget)
	}
	if need := inkling2ResidentKVBytes(hp, wantMax+1, false, 0); need <= budget {
		t.Fatalf("ctx %d needs %d, which fits budget %d: ceiling is not tight", wantMax+1, need, budget)
	}
}

func TestInkling2KVFitRefusesOverBudgetWithMaxCtx(t *testing.T) {
	hp := inkling2MiniKVShape()
	const free = int64(2) << 30
	other := inkling2MiniOther(hp)
	kv, err := inkling2KVFit(hp, 49152, false, 0, other, free)
	var over *Inkling2KVOverBudgetError
	if !errors.As(err, &over) {
		t.Fatalf("err = %v, want *Inkling2KVOverBudgetError", err)
	}
	if kv != inkling2ResidentKVBytes(hp, 49152, false, 0) {
		t.Fatalf("returned KV %d is not the KV-only figure", kv)
	}
	const wantMax = 21591
	if over.Ctx != 49152 || over.MaxCtx != wantMax || over.FreeBytes != free || over.NeedBytes != kv+other {
		t.Fatalf("error fields = %+v, want ctx 49152, max %d, free %d, need %d", *over, wantMax, free, kv+other)
	}
	if !strings.Contains(err.Error(), "largest context that fits is 21591") {
		t.Fatalf("message does not name the max ctx: %q", err.Error())
	}
}

func TestInkling2KVFitAcceptsCtxAtCeiling(t *testing.T) {
	hp := inkling2MiniKVShape()
	const free = int64(2) << 30
	other := inkling2MiniOther(hp)
	if _, err := inkling2KVFit(hp, 21591, false, 0, other, free); err != nil {
		t.Fatalf("ctx at the ceiling refused: %v", err)
	}
	if _, err := inkling2KVFit(hp, 21592, false, 0, other, free); err == nil {
		t.Fatal("ctx one past the ceiling admitted")
	}
}

func TestInkling2KVFitZeroMaxWhenNothingFits(t *testing.T) {
	hp := inkling2MiniKVShape()
	const free = int64(300) << 20 // below the fixed KV remainder and buffers after headroom
	_, err := inkling2KVFit(hp, 4096, false, 0, inkling2MiniOther(hp), free)
	var over *Inkling2KVOverBudgetError
	if !errors.As(err, &over) || over.MaxCtx != 0 {
		t.Fatalf("err = %v, want over-budget error with MaxCtx 0", err)
	}
}

func TestInkling2KVFitMaxCtxBelowFloorReportsZero(t *testing.T) {
	hp := inkling2MiniKVShape()
	// budget leaves room for 300 tokens of KV after the fixed buffers: below the 512 floor
	const free = int64(501080885)
	_, err := inkling2KVFit(hp, 4096, false, 0, inkling2MiniOther(hp), free)
	var over *Inkling2KVOverBudgetError
	if !errors.As(err, &over) {
		t.Fatalf("err = %v, want over-budget error", err)
	}
	if over.MaxCtx != 0 {
		t.Fatalf("max ctx = %d, want 0 when the fit is below the %d floor", over.MaxCtx, Inkling2ResidentCtxMin)
	}
}

func TestInkling2KVFitUnknownFreeKeepsCtxMaxCeiling(t *testing.T) {
	hp := inkling2MiniKVShape()
	if _, err := inkling2KVFit(hp, Inkling2ResidentCtxMax(inkling2ResidentBackend), false, 0, inkling2MiniOther(hp), compute.FreeUnknown); err != nil {
		t.Fatalf("ctx %d with unknown free refused: %v", Inkling2ResidentCtxMax(inkling2ResidentBackend), err)
	}
	_, err := inkling2KVFit(hp, Inkling2ResidentCtxMax(inkling2ResidentBackend)+1, false, 0, inkling2MiniOther(hp), compute.FreeUnknown)
	var over *Inkling2KVOverBudgetError
	if !errors.As(err, &over) || over.MaxCtx != Inkling2ResidentCtxMax(inkling2ResidentBackend) || over.FreeBytes != compute.FreeUnknown {
		t.Fatalf("err = %v, want over-budget error with MaxCtx %d and unknown free", err, Inkling2ResidentCtxMax(inkling2ResidentBackend))
	}
	if !strings.Contains(err.Error(), "free device memory is unknown") {
		t.Fatalf("message does not say free is unknown: %q", err.Error())
	}
	if !strings.Contains(err.Error(), "is 24576") {
		t.Fatalf("message does not name the 24576 ceiling: %q", err.Error())
	}
}

func TestNoteInkling2KVBytesSkipsCohortBinding(t *testing.T) {
	set := &inkling2DeviceSet{placement: Inkling2DevicePlacement{KVBytes: 7}}
	m := &Inkling2LoopModel{closer: set}
	noteInkling2KVBytes(m, 0, 99)
	if set.placement.KVBytes != 7 {
		t.Fatalf("ctx 0 cohort build overwrote KV bytes: got %d, want 7", set.placement.KVBytes)
	}
	noteInkling2KVBytes(m, 4096, 99)
	if set.placement.KVBytes != 99 {
		t.Fatalf("ctx 4096 session did not record KV bytes: got %d, want 99", set.placement.KVBytes)
	}
}

func TestInkling2ResidentPlacementReportsKVBytes(t *testing.T) {
	p := Inkling2DevicePlacement{DenseBytes: 1 << 30, ResidentBytes: 1 << 30}
	if strings.Contains(p.String(), "kv cache") {
		t.Fatalf("placement before any session reports KV: %q", p.String())
	}
	p.KVBytes = 1 << 30
	if !strings.Contains(p.String(), "kv cache 1.00 GiB") {
		t.Fatalf("placement after a session lacks KV: %q", p.String())
	}
}

// Caps advertises the capacity probe the resident fit reads through
// compute.DeviceMemoryInfo. Without it the fit calls Caps on the fake's nil embedded
// Backend, so every fake that reaches the fit needs it.
func (d *fakeInk2Budget) Caps() compute.Caps {
	return compute.Caps{CapacityProbe: d.capBytes > 0}
}

// fitFreeBackend reports admitFree on its first free probe, the wide admission, and
// fitFree on every later one, the resident KV fit. The device admits the wide arenas
// and then has too little room left for the KV they come with.
type fitFreeBackend struct {
	*fakeInk2Budget
	admitFree, fitFree int64
	probes             int
}

func (b *fitFreeBackend) DeviceMemory() (total, free int64, known bool) {
	b.probes++
	if b.probes == 1 {
		return 1 << 30, b.admitFree, true
	}
	return 1 << 30, b.fitFree, true
}

// ringFitNeeds is the device bytes a resident session of the ring model needs at the
// base chunk and at the wide one: KV plus the non-KV buffers. padK is 0 because the
// fake expert stacks carry no padded rows.
func ringFitNeeds(m *Inkling2LoopModel, ctx int, batched bool) (base, wide int64) {
	base = inkling2ResidentTotalBytes(m.HP, ctx, batched, inkling2PrefillChunkBase, fakeInk2RingArena, 0, 0)
	wide = inkling2ResidentTotalBytes(m.HP, ctx, batched, 1024, fakeInk2RingArena, 1024, 0)
	return base, wide
}

// freeForBudget is a free reading whose fit budget, free minus the headroom rule, is
// the budget: 15% of free stays under the 1 GiB cap at these sizes.
func freeForBudget(budget int64) int64 { return budget*100/85 + 2 }

// The wide chunk's KV fit is classified by the wide build that raised it: the wrapper
// retries at the base chunk, and a refusal of the base build is not retried.
func TestInkling2ChunkRefusalClassifiesKVFitByWidth(t *testing.T) {
	over := &Inkling2KVOverBudgetError{Ctx: 49152, MaxCtx: 21591, NeedBytes: 2, FreeBytes: 1}
	wrapped := inkling2DeviceRefused{over}
	if !inkling2ChunkRefusal(wrapped) {
		t.Fatal("a fit refusal on a wide build is not retryable")
	}
	var got *Inkling2KVOverBudgetError
	if !errors.As(wrapped, &got) || got != over {
		t.Fatalf("errors.As through the wide wrapper lost *Inkling2KVOverBudgetError: %v", got)
	}
	if inkling2ChunkRefusal(over) {
		t.Fatal("a non-wide fit refusal is retryable")
	}
}

// A wide build whose KV does not fit the free memory left after its arenas falls back
// to the base chunk and binds, with the hold-off armed and the refusal logged once.
func TestInkling2WideFitRefusalFallsBackToBaseChunk(t *testing.T) {
	logged := captureInk2Log(t)
	resetInk2ScalarWide(t)
	t.Setenv(inkling2PrefillChunkEnv, "1024")
	m := fakeInk2RingModel()
	batched := inkling2BatchedPrefill() || inkling2SpecDepth() > 0
	base, wide := ringFitNeeds(m, fakeInk2RingCtx, batched)
	budget := (base + wide) / 2
	free := freeForBudget(budget)
	if fit := free - inkling2DeviceHeadroom(free); fit < base || fit >= wide {
		t.Fatalf("fit budget %d does not split base %d from wide %d", fit, base, wide)
	}
	d := &fitFreeBackend{fakeInk2Budget: newFakeInk2Budget(1 << 30), admitFree: 1 << 30, fitFree: free}
	s, err := newInkling2ResidentSized(m, d, fakeInk2RingCtx, fakeInk2RingArena)
	if err != nil {
		t.Fatalf("wide fit refusal did not fall back to the base chunk: %v", err)
	}
	defer s.close()
	if s.chunkMax != inkling2PrefillChunkBase || s.prefillCap != 0 {
		t.Fatalf("fallback bound chunk %d with %d arena tokens, want the base chunk", s.chunkMax, s.prefillCap)
	}
	if trunk := trunkInk2ResidentFloats(m, fakeInk2RingCtx, fakeInk2RingArena, batched); d.live != trunk {
		t.Fatalf("fallback holds %d floats, want the unset session's %d", d.live, trunk)
	}
	if got := inkling2ScalarWideHold.Load(); got != inkling2PrefillReprobeEvery {
		t.Fatalf("hold-off = %d after the wide fit refusal, want %d", got, inkling2PrefillReprobeEvery)
	}
	if ink2LogLines(logged) != 1 {
		t.Fatalf("fallback logged %q, want one line", logged)
	}
}

// A base-chunk fit refusal is the session's own error: nothing retries it and no
// hold-off arms.
func TestInkling2BaseFitRefusalIsNotRetried(t *testing.T) {
	resetInk2ScalarWide(t)
	unsetInk2Chunk(t)
	m := fakeInk2RingModel()
	batched := inkling2BatchedPrefill() || inkling2SpecDepth() > 0
	base, _ := ringFitNeeds(m, fakeInk2RingCtx, batched)
	free := freeForBudget(base / 2)
	d := &fitFreeBackend{fakeInk2Budget: newFakeInk2Budget(1 << 30), admitFree: free, fitFree: free}
	s, err := newInkling2ResidentSized(m, d, fakeInk2RingCtx, fakeInk2RingArena)
	var over *Inkling2KVOverBudgetError
	if s != nil || !errors.As(err, &over) {
		t.Fatalf("base fit refusal: session %v, err %v; want *Inkling2KVOverBudgetError", s, err)
	}
	if inkling2ChunkRefusal(err) {
		t.Fatal("a base-chunk fit refusal is retryable")
	}
	if inkling2ScalarWideHold.Load() != 0 {
		t.Fatal("a base-chunk fit refusal armed the wide hold-off")
	}
	if d.live != 0 || d.frees != d.allocs {
		t.Fatalf("refused base build left %d floats live, %d frees of %d allocs", d.live, d.frees, d.allocs)
	}
}
