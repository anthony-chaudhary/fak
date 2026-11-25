package ggufload

import (
	"bytes"
	"context"
	"errors"
	"log"
	"os"
	"reflect"
	"strconv"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/model"
)

// fakeInk2Budget is fakeInk2Cohort on a device of capBytes that reports its free
// memory and refuses an allocation past it, or any allocation refuse picks.
type fakeInk2Budget struct {
	*fakeInk2Cohort
	capBytes int64 // 0: the device cannot report free memory and never fills
	refuse   func(n int) bool
	refusals int
}

func newFakeInk2Budget(capBytes int64) *fakeInk2Budget {
	return &fakeInk2Budget{fakeInk2Cohort: newFakeInk2Cohort(), capBytes: capBytes}
}

func (d *fakeInk2Budget) DeviceMemory() (total, free int64, known bool) {
	if d.capBytes == 0 {
		return 0, compute.FreeUnknown, false
	}
	return d.capBytes, d.capBytes - 4*int64(d.live), true
}

func (d *fakeInk2Budget) Ink2Alloc(n int) (compute.Tensor, error) {
	if (d.refuse != nil && d.refuse(n)) || (d.capBytes > 0 && 4*int64(d.live+n) > d.capBytes) {
		d.refusals++
		return compute.Tensor{}, &compute.DeviceAllocError{Bytes: 4 * n, Site: "fake"}
	}
	return d.fakeInk2Cohort.Ink2Alloc(n)
}

func (d *fakeInk2Budget) Ink2AllocDeviceOnly(n int) (compute.Tensor, error) { return d.Ink2Alloc(n) }

func (d *fakeInk2Budget) Ink2MatVec(_ compute.Tensor, x, y compute.Ink2Ref) error {
	d.rows(y)[0] = d.rows(x)[0]
	return nil
}

const fakeInk2RingWindow = 96

// fakeInk2RingModel binds like a model with local and global layers and a second
// pass over its core, so ring and cache sizes are exercised; it cannot run a forward.
func fakeInk2RingModel() *Inkling2LoopModel {
	const h = 4
	hp := model.Inkling2LoopHParams{Hidden: h, Layers: 4, CoreStart: 1, CoreEnd: 3, Passes: 2,
		Heads: 2, KVHeads: 2, HeadDim: 4, ConvKernel: 3, ExpertsUsed: 2, ExpertWidth: 8, RelLength: 4,
		VocabUnpadded: 1, SlidingWindow: fakeInk2RingWindow, LocalLayer: []bool{true, false, true, false}}
	layers := make([]model.Inkling2LoopLayerWeights, hp.Layers)
	for i := range layers {
		layers[i].GateUp = &inkling2DeviceExperts{stack: compute.Tensor{Dtype: compute.NVFP4}}
		layers[i].Down = &inkling2DeviceExperts{stack: compute.Tensor{Dtype: compute.NVFP4}}
	}
	return &Inkling2LoopModel{HP: hp, Weights: &model.Inkling2LoopWeights{TokEmbd: fakeInk2Embedding{}, Layers: layers,
		Output: &model.Inkling2LoopDenseMatrix{R: 1, C: h, Data: make([]float32, h)}}}
}

// trunkInk2Caps is the slot capacity rule of a session before the chunk setting, with
// its 512-token chunk written out.
func trunkInk2Caps(hp model.Inkling2LoopHParams, ctx int, batched bool) []int {
	caps := make([]int, hp.Layers+(hp.Passes-1)*(hp.CoreEnd-hp.CoreStart))
	for slot := range caps {
		caps[slot] = ctx
		if hp.LocalLayer[inkling2SlotLayer(hp, slot)] {
			caps[slot] = hp.SlidingWindow
			if batched {
				caps[slot] += 512
			}
		}
	}
	return caps
}

// trunkInk2ResidentFloats is every float a session of the ring model allocated at
// bind before the chunk setting.
func trunkInk2ResidentFloats(m *Inkling2LoopModel, ctx, arenaFloats int, batched bool) int {
	hp := m.HP
	kvDim := hp.KVHeads * hp.HeadDim
	n := hp.Hidden // the output matrix, the model's only uploaded weight
	maxCap := hp.SlidingWindow
	for _, c := range trunkInk2Caps(hp, ctx, batched) {
		maxCap = max(maxCap, c)
		if ctx > 0 {
			n += 2*c*kvDim + (2*kvDim+2*hp.Hidden)*(hp.ConvKernel-1)
		}
	}
	if ctx > 0 {
		n += 2 * hp.Heads * maxCap
	}
	return n + hp.Hidden + 4 + arenaFloats + 4
}

func captureInk2Log(t *testing.T) *bytes.Buffer {
	t.Helper()
	var buf bytes.Buffer
	old, flags := log.Writer(), log.Flags()
	log.SetOutput(&buf)
	log.SetFlags(0)
	t.Cleanup(func() { log.SetOutput(old); log.SetFlags(flags) })
	return &buf
}

func ink2LogLines(buf *bytes.Buffer) int { return strings.Count(buf.String(), "\n") }

const fakeInk2RingCtx, fakeInk2RingArena = 2048, 64

// With FAK_INKLING2_PREFILL_CHUNK unset every buffer is the one the 512-token build
// allocated: the scalar rings, the bind-time allocation, the lazily grown prefill
// arenas, the forward widths, and each cohort's rings, chunk and admission bytes.
func TestInkling2PrefillChunkUnsetMatchesTrunk(t *testing.T) {
	logged := captureInk2Log(t)
	unsetInk2Chunk(t)
	resetInk2ScalarWide(t)
	if got := inkling2PrefillChunkWanted(os.LookupEnv); got != 512 {
		t.Fatalf("unset chunk = %d, want 512", got)
	}
	for _, n := range []int{0, 1, 511, 512, 513, 1024, 1300} {
		if got := inkling2PrefillChunkFor(n, inkling2PrefillChunkWanted(envOf(nil))); got != min(512, n) {
			t.Fatalf("forward over %d tokens = %d, want %d", n, got, min(512, n))
		}
	}
	m := fakeInk2RingModel()
	hp := m.HP
	for _, batched := range []string{"1", "0"} {
		t.Setenv("FAK_INKLING2_BATCHED_PREFILL", batched)
		d := newFakeInk2Budget(1 << 30)
		s, err := newInkling2ResidentSized(m, d, fakeInk2RingCtx, fakeInk2RingArena)
		if err != nil {
			t.Fatal(err)
		}
		want := trunkInk2Caps(hp, fakeInk2RingCtx, batched == "1")
		if !reflect.DeepEqual(s.caps, want) || !reflect.DeepEqual(s.baseCaps, want) || s.chunkMax != 512 {
			t.Fatalf("batched=%s: caps %v base %v chunk %d, want %v at 512", batched, s.caps, s.baseCaps, s.chunkMax, want)
		}
		if floats := trunkInk2ResidentFloats(m, fakeInk2RingCtx, fakeInk2RingArena, batched == "1"); d.live != floats {
			t.Fatalf("batched=%s: bind allocated %d floats, trunk allocates %d", batched, d.live, floats)
		}
		if s.prefillCap != 0 || s.prefillKeep.T.Buf() != nil {
			t.Fatalf("batched=%s: prefill arenas allocated at bind (cap %d)", batched, s.prefillCap)
		}
		bound := d.live
		// the arenas are grown to the forward's own width, as before
		s.ensurePrefillArenas(inkling2PrefillChunkFor(300, s.chunkMax))
		keep, scr := s.prefillArenas(300)
		if s.prefillCap != 300 || d.live != bound+keep+scr+300*hp.Hidden {
			t.Fatalf("batched=%s: arenas of a 300-token prompt hold %d floats at cap %d", batched, d.live-bound, s.prefillCap)
		}
		s.close()
		if d.live != 0 {
			t.Fatalf("batched=%s: %d floats live after close", batched, d.live)
		}
	}
	t.Setenv("FAK_INKLING2_BATCHED_PREFILL", "1")

	d := newFakeInk2Budget(1 << 30)
	bt := NewInkling2LoopCohorts(m, d)
	defer bt.Close()
	local := trunkInk2Caps(hp, 0, true)
	if _, err := bt.acquireFor(2, 8, 1); err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct{ C, ctx, longest int }{{2, 64, 10}, {4, 700, 600}, {8, 2048, 1300}, {16, 2048, 40}} {
		bt.ReleaseArenas()
		bound := d.live
		b, err := bt.acquireFor(tc.C, tc.ctx, tc.longest)
		if err != nil {
			t.Fatal(err)
		}
		if !reflect.DeepEqual(bt.s.caps, local) {
			t.Fatalf("cohort binding rings %v, want %v", bt.s.caps, local)
		}
		prefillT := min(512, tc.C*tc.ctx)
		caps := inkling2BatchCaps(hp, local, tc.ctx, prefillT)
		keep, scr := bt.s.prefillArenas(max(tc.C, prefillT))
		scr += tc.C*bt.s.output.rows + 64
		need := inkling2BatchBytes(hp, caps, tc.C, keep, scr, false)
		if b.prefillT != prefillT || !reflect.DeepEqual(b.caps, caps) || b.need != need {
			t.Fatalf("cohort %+v: chunk %d caps %v need %d, trunk has chunk %d caps %v need %d",
				tc, b.prefillT, b.caps, b.need, prefillT, caps, need)
		}
		if got := 4 * int64(d.live-bound); got != ink2CohortStateBytes(need, tc.C, prefillT, hp.Hidden) {
			t.Fatalf("cohort %+v: state allocated %d bytes, admitted %d", tc, got, need)
		}
	}
	// a cohort the cached state covers without much surplus reuses it
	held, allocs := bt.b, d.allocs
	for _, tc := range []struct{ C, ctx, longest int }{{14, 2048, 40}, {16, 1728, 600}, {16, 2048, 1300}} {
		b, err := bt.acquireFor(tc.C, tc.ctx, tc.longest)
		if err != nil {
			t.Fatal(err)
		}
		// the chunk and the admitted bytes stay those of the state's own shape
		prefillT := min(512, held.capC*held.capCtx)
		caps := inkling2BatchCaps(hp, local, tc.ctx, prefillT)
		keep, scr := bt.s.prefillArenas(max(held.capC, prefillT))
		scr += held.capC*bt.s.output.rows + 64
		need := inkling2BatchBytes(hp, inkling2BatchCaps(hp, local, held.capCtx, prefillT), held.capC, keep, scr, false)
		if b != held || d.allocs != allocs || b.prefillT != prefillT || !reflect.DeepEqual(b.caps, caps) || b.need != need {
			t.Fatalf("reused cohort %+v: chunk %d caps %v need %d allocs %d -> %d, trunk has chunk %d caps %v need %d",
				tc, b.prefillT, b.caps, b.need, allocs, d.allocs, prefillT, caps, need)
		}
	}
	if bt.wideCooldown != 0 || logged.Len() != 0 || inkling2ScalarWideHold.Load() != 0 {
		t.Fatalf("unset chunk took the fallback path: cooldown %d, log %q", bt.wideCooldown, logged)
	}
}

func envOf(kv map[string]string) func(string) (string, bool) {
	return func(k string) (string, bool) {
		v, ok := kv[k]
		return v, ok
	}
}

func TestInkling2PrefillChunkParse(t *testing.T) {
	for _, c := range []struct {
		v    string
		set  bool
		want int
		bad  bool
	}{
		{"", false, 512, false},
		{"", true, 512, false},
		{"auto", true, 1024, false},
		{" AUTO ", true, 1024, false},
		{"512", true, 512, false},
		{"1024", true, 1024, false},
		{"2048", true, 2048, false},
		{"64", true, 64, false},
		{"1000", true, 512, true},
		{"32", true, 512, true},
		{"4096", true, 512, true},
		{"-1024", true, 512, true},
		{"wide", true, 512, true},
	} {
		got, err := inkling2PrefillChunkParse(c.v, c.set)
		if got != c.want || (err != nil) != c.bad {
			t.Fatalf("%q set=%v: got %d err %v, want %d bad=%v", c.v, c.set, got, err, c.want, c.bad)
		}
		if err != nil && !strings.Contains(err.Error(), inkling2PrefillChunkEnv) {
			t.Fatalf("%q: error %q does not name the variable", c.v, err)
		}
	}
}

// A value that does not parse is the unset behaviour, announced once.
func TestInkling2PrefillChunkGarbageIsDefaultWithOneLogLine(t *testing.T) {
	logged := captureInk2Log(t)
	inkling2PrefillChunkWarned.Store(false)
	t.Setenv(inkling2PrefillChunkEnv, "lots")
	m := fakeInk2RingModel()
	d := newFakeInk2Budget(1 << 30)
	for i := 0; i < 3; i++ {
		s, err := newInkling2ResidentSized(m, d, fakeInk2RingCtx, fakeInk2RingArena)
		if err != nil {
			t.Fatal(err)
		}
		if s.chunkMax != 512 || s.chunkWanted != 512 || d.live != trunkInk2ResidentFloats(m, fakeInk2RingCtx, fakeInk2RingArena, true) {
			t.Fatalf("garbage chunk bound chunk %d wanted %d with %d floats", s.chunkMax, s.chunkWanted, d.live)
		}
		s.close()
	}
	if ink2LogLines(logged) != 1 || !strings.Contains(logged.String(), inkling2PrefillChunkEnv) {
		t.Fatalf("garbage chunk logged %q, want one line naming the variable", logged)
	}
}

func TestInkling2CohortChunkWidensOnlyPastBasePrompts(t *testing.T) {
	for _, c := range []struct{ wanted, longest, want int }{
		{512, 3000, 512}, {1024, 512, 512}, {1024, 513, 1024}, {2048, 40, 512}, {2048, 600, 2048},
		{256, 40, 256}, {256, 900, 256},
	} {
		if got := inkling2CohortChunk(c.wanted, c.longest); got != c.want {
			t.Fatalf("wanted %d longest %d: chunk %d, want %d", c.wanted, c.longest, got, c.want)
		}
	}
}

func TestInkling2WideAdmitFailsClosed(t *testing.T) {
	need := int64(1 << 30)
	if err := inkling2WideAdmit(1, need, need+inkling2BatchHeadroom, true); err != nil {
		t.Fatalf("an exact fit with headroom must admit: %v", err)
	}
	for _, c := range []struct {
		free  int64
		known bool
	}{{need + inkling2BatchHeadroom - 1, true}, {compute.FreeUnknown, true}, {0, false}} {
		var refused *Inkling2BatchMemoryError
		if err := inkling2WideAdmit(4, need, c.free, c.known); !errors.As(err, &refused) || refused.Sequences != 4 || !inkling2ChunkRefusal(err) {
			t.Fatalf("free %d known=%v admitted a wide chunk: %v", c.free, c.known, err)
		}
	}
	if inkling2ChunkRefusal(errors.New("inkling2 batch: context 0 outside 1..8")) {
		t.Fatal("a validation error read as a device refusal")
	}
}

// The bytes the wide rings add are per sequence, so a cohort's admission carries them
// once per row.
func TestInkling2WideRingExtraBytesScaleWithCohortWidth(t *testing.T) {
	hp := fakeInk2RingModel().HP
	slots := hp.Layers + (hp.Passes-1)*(hp.CoreEnd-hp.CoreStart)
	if got := inkling2WideRingExtraBytes(hp, slots, 512); got != 0 {
		t.Fatalf("base chunk adds %d bytes", got)
	}
	// local layers 0 and 2, and the second visit of core layer 2: three local slots
	one := int64(3 * 512 * 8 * 2 * 4)
	if got := inkling2WideRingExtraBytes(hp, slots, 1024); got != one {
		t.Fatalf("wide extra = %d, want %d", got, one)
	}
	local := inkling2ResidentCaps(hp, 0, true, 512)
	for _, C := range []int{1, 8, 16} {
		base := inkling2BatchStateBytes(hp, inkling2BatchCaps(hp, local, fakeInk2RingCtx, 512), C, false)
		wide := inkling2BatchStateBytes(hp, inkling2BatchCaps(hp, local, fakeInk2RingCtx, 1024), C, false)
		if wide-base != int64(C)*one {
			t.Fatalf("C=%d: wide state adds %d bytes, want %d per row", C, wide-base, one)
		}
	}
}

// An opted-in cohort sizes its rings from the chunk it forwards: the base when no
// prompt passes 512 tokens or cohort prefill is off, window + chunk otherwise, and a
// base cohort on a state grown wide addresses exactly what a base state would.
func TestInkling2PrefillChunkCohortRingsFollowTheCohort(t *testing.T) {
	m := fakeInk2RingModel()
	hp := m.HP
	const C, ctx = 4, fakeInk2RingCtx
	baseline := func() (need int64, live int, caps []int) {
		d := newFakeInk2Budget(1 << 30)
		bt := NewInkling2LoopCohorts(m, d)
		defer bt.Close()
		b, err := bt.acquireFor(C, ctx, 1300)
		if err != nil {
			t.Fatal(err)
		}
		return b.need, d.live, b.caps
	}
	unsetInk2Chunk(t)
	baseNeed, baseLive, baseCaps := baseline()

	for _, chunk := range []string{"1024", "auto"} {
		t.Setenv(inkling2PrefillChunkEnv, chunk)
		d := newFakeInk2Budget(1 << 30)
		bt := NewInkling2LoopCohorts(m, d)
		short, err := bt.acquireFor(C, ctx, 512)
		if err != nil {
			t.Fatal(err)
		}
		if short.prefillT != 512 || short.need != baseNeed || d.live != baseLive || !reflect.DeepEqual(short.caps, baseCaps) {
			t.Fatalf("chunk=%s short cohort: chunk %d need %d live %d caps %v; base has need %d live %d caps %v",
				chunk, short.prefillT, short.need, d.live, short.caps, baseNeed, baseLive, baseCaps)
		}
		long, err := bt.acquireFor(C, ctx, 513)
		if err != nil {
			t.Fatal(err)
		}
		wantCaps := inkling2BatchCaps(hp, inkling2ResidentCaps(hp, 0, true, 512), ctx, 1024)
		if long.prefillT != 1024 || !reflect.DeepEqual(long.caps, wantCaps) || long.caps[0] != fakeInk2RingWindow+1024 {
			t.Fatalf("chunk=%s long cohort: chunk %d caps %v, want 1024 and %v", chunk, long.prefillT, long.caps, wantCaps)
		}
		if held := 4 * int64(d.live-bindingFloats(d, bt)); long.need <= baseNeed || held != ink2CohortStateBytes(long.need, C, 1024, hp.Hidden) {
			t.Fatalf("chunk=%s long cohort admitted %d bytes for %d allocated (base %d)", chunk, long.need, held, baseNeed)
		}
		allocs := d.allocs
		again, err := bt.acquireFor(C, ctx, 40)
		if err != nil {
			t.Fatal(err)
		}
		if again != long || d.allocs != allocs || again.prefillT != 512 || !reflect.DeepEqual(again.caps, baseCaps) {
			t.Fatalf("chunk=%s short cohort on the wide state: chunk %d caps %v allocs %d -> %d", chunk, again.prefillT, again.caps, allocs, d.allocs)
		}
		bt.Close()
		if d.live != 0 {
			t.Fatalf("chunk=%s: %d floats live after Close", chunk, d.live)
		}
	}

	t.Setenv("FAK_INKLING2_COHORT_PREFILL", "0")
	unsetInk2Chunk(t)
	offNeed, offLive, offCaps := baseline()
	t.Setenv(inkling2PrefillChunkEnv, "1024")
	wideNeed, wideLive, wideCaps := baseline()
	if offNeed != wideNeed || offLive != wideLive || !reflect.DeepEqual(offCaps, wideCaps) {
		t.Fatalf("cohort prefill off: wide chunk changed need %d -> %d, live %d -> %d, caps %v -> %v",
			offNeed, wideNeed, offLive, wideLive, offCaps, wideCaps)
	}
}

// ink2CohortProbe is the admitted bytes of a long-prompt C x ctx cohort under the chunk
// setting on a device with room, and the floats of its weight binding.
func ink2CohortProbe(t *testing.T, m *Inkling2LoopModel, chunk string, C, ctx int) (need int64, binding int) {
	t.Helper()
	t.Setenv(inkling2PrefillChunkEnv, chunk)
	d := newFakeInk2Budget(1 << 30)
	bt := NewInkling2LoopCohorts(m, d)
	defer bt.Close()
	b, err := bt.acquireFor(C, ctx, 1300)
	if err != nil {
		t.Fatal(err)
	}
	return b.need, bindingFloats(d, bt)
}

// ink2CohortStateBytes is what a cohort state allocates for need admitted bytes: the
// admission counts one input row per sequence, the state holds one per chunk token.
func ink2CohortStateBytes(need int64, C, prefillT, hidden int) int64 {
	return need + 4*int64((max(C, prefillT)-C)*hidden)
}

// resetInk2ScalarWide clears the process-wide scalar hold-off and its log latch.
func resetInk2ScalarWide(t *testing.T) {
	t.Helper()
	clear := func() { inkling2ScalarWideHold.Store(0); inkling2ScalarWideLogged.Store(false) }
	clear()
	t.Cleanup(clear)
}

func unsetInk2Chunk(t *testing.T) {
	t.Helper()
	t.Setenv(inkling2PrefillChunkEnv, "")
	os.Unsetenv(inkling2PrefillChunkEnv)
}

func bindingFloats(d *fakeInk2Budget, bt *Inkling2LoopCohorts) int {
	n := 0
	for _, tensor := range bt.s.owned {
		n += len(d.data[tensor.Buf()])
	}
	return n
}

// A wide cohort the device cannot hold is rebuilt at the base chunk with the base
// rings and the base admission, so it is admitted wherever the unset build admits it.
func TestInkling2PrefillChunkCohortRefusalShrinksToBase(t *testing.T) {
	logged := captureInk2Log(t)
	m := fakeInk2RingModel()
	const C, ctx = 4, fakeInk2RingCtx
	hidden := m.HP.Hidden
	unsetInk2Chunk(t)
	baseNeed, binding := ink2CohortProbe(t, m, "", C, ctx)
	wideNeed, _ := ink2CohortProbe(t, m, "1024", C, ctx)
	if wideNeed <= baseNeed || logged.Len() != 0 {
		t.Fatalf("wide need %d, base need %d, log %q", wideNeed, baseNeed, logged)
	}
	// the device holds the base cohort with its headroom and one byte short of the wide one
	fits := 4*int64(binding) + inkling2BatchHeadroom + wideNeed - 1

	for _, admit := range []string{"1", "0"} {
		logged.Reset()
		t.Setenv("FAK_INKLING2_BATCH_ADMIT", admit)
		d := newFakeInk2Budget(fits)
		if admit == "0" {
			// with admission off the refusal comes from the device, mid-allocation
			d.capBytes = 4*int64(binding) + (baseNeed+wideNeed)/2
		}
		bt := NewInkling2LoopCohorts(m, d)
		b, err := bt.acquireFor(C, ctx, 1300)
		if err != nil {
			t.Fatalf("admit=%s: a cohort the base chunk holds was refused: %v", admit, err)
		}
		if b.capT != 512 || b.prefillT != 512 || b.need != baseNeed || 4*int64(d.live-binding) != ink2CohortStateBytes(baseNeed, C, 512, hidden) {
			t.Fatalf("admit=%s: fallback state chunk %d need %d holds %d bytes, want the base %d",
				admit, b.prefillT, b.need, 4*(d.live-binding), baseNeed)
		}
		if (admit == "0") != (d.refusals > 0) {
			t.Fatalf("admit=%s: %d device refusals", admit, d.refusals)
		}
		if ink2LogLines(logged) != 1 || !strings.Contains(logged.String(), "chunk 1024 refused") {
			t.Fatalf("admit=%s: fallback logged %q, want one line", admit, logged)
		}
		bt.Close()
		if d.live != 0 || d.frees != d.allocs {
			t.Fatalf("admit=%s: %d floats live, %d frees of %d allocs", admit, d.live, d.frees, d.allocs)
		}
	}
}

// A refusal holds the wide chunk off for a bounded number of cohorts and then probes
// again; an error that is not a refusal neither falls back nor starts the hold-off.
func TestInkling2PrefillChunkCohortRefusalIsNotPermanent(t *testing.T) {
	logged := captureInk2Log(t)
	m := fakeInk2RingModel()
	const C, ctx = 4, fakeInk2RingCtx
	wideNeed, binding := ink2CohortProbe(t, m, "1024", C, ctx)
	d := newFakeInk2Budget(1 << 30)
	bt := NewInkling2LoopCohorts(m, d)
	defer bt.Close()

	if _, err := bt.acquireFor(C, inkling2ResidentCtx+1, 1300); err == nil || inkling2ChunkRefusal(err) {
		t.Fatalf("an out-of-range context returned %v", err)
	}
	if bt.wideCooldown != 0 || logged.Len() != 0 {
		t.Fatalf("a validation error started the hold-off: cooldown %d, log %q", bt.wideCooldown, logged)
	}

	d.capBytes = 4*int64(binding) + inkling2BatchHeadroom + wideNeed - 1
	b, err := bt.acquireFor(C, ctx, 1300)
	if err != nil || b.capT != 512 || bt.wideCooldown != inkling2PrefillReprobeEvery {
		t.Fatalf("refused wide cohort: err %v, cooldown %d", err, bt.wideCooldown)
	}
	// the memory comes back at once, but the state is not rebuilt on every cohort
	d.capBytes = 1 << 30
	allocs := d.allocs
	if _, err := bt.acquireFor(C, ctx, 40); err != nil || bt.wideCooldown != inkling2PrefillReprobeEvery {
		t.Fatalf("a base cohort spent the hold-off: err %v, cooldown %d", err, bt.wideCooldown)
	}
	for i := 0; i < inkling2PrefillReprobeEvery; i++ {
		held, err := bt.acquireFor(C, ctx, 1300)
		if err != nil || held != b || held.prefillT != 512 || d.allocs != allocs {
			t.Fatalf("cohort %d of the hold-off: err %v chunk %d allocs %d -> %d", i, err, held.prefillT, allocs, d.allocs)
		}
	}
	wide, err := bt.acquireFor(C, ctx, 1300)
	if err != nil || wide.capT != 1024 || wide.prefillT != 1024 || bt.wideCooldown != 0 {
		t.Fatalf("after the hold-off: err %v chunk %d cooldown %d", err, wide.prefillT, bt.wideCooldown)
	}
	if ink2LogLines(logged) != 1 {
		t.Fatalf("one refusal logged %q", logged)
	}
}

// A base cohort the wide state does not cover is built at the base chunk, so a device
// without room for it wide neither refuses it nor holds the wide chunk off for the
// next cohort that wants it.
func TestInkling2PrefillChunkBaseCohortRegrowDoesNotArmHoldOff(t *testing.T) {
	m := fakeInk2RingModel()
	const C, ctx = 4, fakeInk2RingCtx
	unsetInk2Chunk(t)
	baseNeed, binding := ink2CohortProbe(t, m, "", 2*C, ctx)
	wideNeed, _ := ink2CohortProbe(t, m, "1024", 2*C, ctx)
	d := newFakeInk2Budget(1 << 30)
	bt := NewInkling2LoopCohorts(m, d)
	defer bt.Close()
	if b, err := bt.acquireFor(C, ctx, 1300); err != nil || b.capT != 1024 {
		t.Fatalf("wide cohort: %v", err)
	}
	// room for twice the rows at the base chunk only
	d.capBytes = 4*int64(binding) + inkling2BatchHeadroom + (baseNeed+wideNeed)/2
	b, err := bt.acquireFor(2*C, ctx, 40)
	if err != nil || b.capT != 512 || b.prefillT != 512 {
		t.Fatalf("base cohort regrowing the wide state: err %v", err)
	}
	if bt.wideCooldown != 0 {
		t.Fatalf("a base cohort armed the hold-off: %d", bt.wideCooldown)
	}
	d.capBytes = 1 << 30
	if b, err := bt.acquireFor(C, ctx, 1300); err != nil || b.prefillT != 1024 {
		t.Fatalf("the next long cohort ran at the base chunk: err %v", err)
	}
}

// A scalar session that opts in holds the wide rings and arenas from bind, runs the
// wide forward, and never regrows or supersedes an arena.
func TestInkling2PrefillChunkScalarWideBind(t *testing.T) {
	m := fakeInk2RingModel()
	hp := m.HP
	slots := hp.Layers + (hp.Passes-1)*(hp.CoreEnd-hp.CoreStart)
	trunk := trunkInk2ResidentFloats(m, fakeInk2RingCtx, fakeInk2RingArena, true)
	resetInk2ScalarWide(t)
	// the admission counts every float the bind takes past the weights and the base
	// caches, at a context whose scores row the wide ring also lengthens
	for _, ctx := range []int{fakeInk2RingCtx, 256} {
		t.Setenv(inkling2PrefillChunkEnv, "1024")
		d := newFakeInk2Budget(1 << 30)
		s, err := newInkling2ResidentSized(m, d, ctx, fakeInk2RingArena)
		if err != nil || s.chunkMax != 1024 {
			t.Fatalf("ctx %d: wide bind: %v", ctx, err)
		}
		past := d.live - trunkInk2ResidentFloats(m, ctx, fakeInk2RingArena, true) + hp.Hidden + 4 + fakeInk2RingArena + 4
		if got := s.wideBindBytes(ctx, 1024); got != 4*int64(past) {
			t.Fatalf("ctx %d: admitted %d bytes for %d allocated", ctx, got, 4*past)
		}
		s.close()
	}
	for _, c := range []struct {
		env   string
		chunk int
	}{{"1024", 1024}, {"auto", 1024}, {"768", 768}} {
		t.Setenv(inkling2PrefillChunkEnv, c.env)
		d := newFakeInk2Budget(1 << 30)
		s, err := newInkling2ResidentSized(m, d, fakeInk2RingCtx, fakeInk2RingArena)
		if err != nil {
			t.Fatal(err)
		}
		if s.chunkMax != c.chunk || s.prefillCap != c.chunk || s.caps[0] != fakeInk2RingWindow+c.chunk || s.caps[1] != fakeInk2RingCtx {
			t.Fatalf("chunk=%s: bound chunk %d arenas %d caps %v", c.env, s.chunkMax, s.prefillCap, s.caps)
		}
		if !reflect.DeepEqual(s.baseCaps, trunkInk2Caps(hp, fakeInk2RingCtx, true)) {
			t.Fatalf("chunk=%s: cohort base rings %v", c.env, s.baseCaps)
		}
		keep, scr := s.prefillArenas(c.chunk)
		// three local slots, keys and values, eight floats a position
		extra := 3 * (c.chunk - 512) * 8 * 2
		if int64(4*extra) != inkling2WideRingExtraBytes(hp, slots, c.chunk) {
			t.Fatalf("chunk=%s: ring extra %d bytes, want %d", c.env, inkling2WideRingExtraBytes(hp, slots, c.chunk), 4*extra)
		}
		if got := d.live - trunk; got != extra+keep+scr+c.chunk*hp.Hidden {
			t.Fatalf("chunk=%s: wide bind adds %d floats, want rings %d + arenas %d", c.env, got, extra, keep+scr+c.chunk*hp.Hidden)
		}
		allocs := d.allocs
		for _, n := range []int{40, 600, c.chunk, 1300} {
			s.ensurePrefillArenas(inkling2PrefillChunkFor(n, s.chunkMax))
		}
		if d.allocs != allocs || s.prefillCap != c.chunk {
			t.Fatalf("chunk=%s: arenas regrew: allocs %d -> %d", c.env, allocs, d.allocs)
		}
		s.close()
		if d.live != 0 || d.frees != d.allocs {
			t.Fatalf("chunk=%s: %d floats live after close", c.env, d.live)
		}
	}

	// a chunk below the base narrows the rings and is never admitted as wide
	t.Setenv(inkling2PrefillChunkEnv, "256")
	d := newFakeInk2Budget(0)
	s, err := newInkling2ResidentSized(m, d, fakeInk2RingCtx, fakeInk2RingArena)
	if err != nil {
		t.Fatal(err)
	}
	defer s.close()
	if s.chunkMax != 256 || s.caps[0] != fakeInk2RingWindow+256 || s.prefillCap != 0 {
		t.Fatalf("chunk=256: bound chunk %d caps %v arenas %d", s.chunkMax, s.caps, s.prefillCap)
	}
}

// A wide setting the device cannot admit, an explicit one included, binds exactly the
// unset session and says so once; nothing of the wide attempt stays allocated.
func TestInkling2PrefillChunkScalarAdmissionFallsBackToTrunk(t *testing.T) {
	logged := captureInk2Log(t)
	m := fakeInk2RingModel()
	hp := m.HP
	trunk := trunkInk2ResidentFloats(m, fakeInk2RingCtx, fakeInk2RingArena, true)
	for _, c := range []struct {
		env      string
		capBytes int64
	}{
		{"1024", 4*int64(trunk) + inkling2BatchHeadroom}, // room for the base session only
		{"auto", 4*int64(trunk) + inkling2BatchHeadroom},
		{"2048", 4*int64(trunk) + inkling2BatchHeadroom},
		{"1024", 0}, // free memory unknown: the wide chunk fails closed
	} {
		logged.Reset()
		resetInk2ScalarWide(t)
		t.Setenv(inkling2PrefillChunkEnv, c.env)
		d := newFakeInk2Budget(c.capBytes)
		s, err := newInkling2ResidentSized(m, d, fakeInk2RingCtx, fakeInk2RingArena)
		if err != nil {
			t.Fatalf("chunk=%s: a session the base chunk holds was refused: %v", c.env, err)
		}
		if s.chunkMax != 512 || s.prefillCap != 0 || d.live != trunk || d.refusals != 0 ||
			!reflect.DeepEqual(s.caps, trunkInk2Caps(hp, fakeInk2RingCtx, true)) {
			t.Fatalf("chunk=%s: fallback bound chunk %d, %d floats (trunk %d), caps %v", c.env, s.chunkMax, d.live, trunk, s.caps)
		}
		if ink2LogLines(logged) != 1 || !strings.Contains(logged.String(), "refused") {
			t.Fatalf("chunk=%s: fallback logged %q, want one line", c.env, logged)
		}
		s.close()
	}
}

// A device that refuses the wide arenas after admission releases the whole wide
// attempt before the base session is built, so the base allocation meets the memory
// an unset session would, and no error sticks to the session. A refusal of the base
// build is an error of that build alone.
func TestInkling2PrefillChunkScalarAllocRefusalReleasesWideAttempt(t *testing.T) {
	logged := captureInk2Log(t)
	resetInk2ScalarWide(t)
	t.Setenv(inkling2PrefillChunkEnv, "1024")
	m := fakeInk2RingModel()
	hp := m.HP
	trunk := trunkInk2ResidentFloats(m, fakeInk2RingCtx, fakeInk2RingArena, true)
	kvDim := hp.KVHeads * hp.HeadDim
	wideRing := (fakeInk2RingWindow + 1024) * kvDim
	baseRing := (fakeInk2RingWindow + 512) * kvDim

	d := newFakeInk2Budget(1 << 30)
	peak := 0
	d.refuse = func(n int) bool {
		peak = max(peak, d.live)
		// only the wide prefill arenas are larger than the largest cache
		return n > fakeInk2RingCtx*kvDim
	}
	s, err := newInkling2ResidentSized(m, d, fakeInk2RingCtx, fakeInk2RingArena)
	if err != nil {
		t.Fatalf("a session the base chunk holds was refused: %v", err)
	}
	if d.refusals == 0 || peak <= trunk {
		t.Fatalf("wide attempt: %d refusals, peak %d floats (trunk %d)", d.refusals, peak, trunk)
	}
	if s.err != nil || s.chunkMax != 512 || s.prefillCap != 0 || d.live != trunk || s.caps[0] != fakeInk2RingWindow+512 {
		t.Fatalf("fallback session: err %v chunk %d arenas %d, %d floats (trunk %d)", s.err, s.chunkMax, s.prefillCap, d.live, trunk)
	}
	if ink2LogLines(logged) != 1 {
		t.Fatalf("fallback logged %q, want one line", logged)
	}
	// the session is the unset one: its first prefill grows the base arenas
	d.refuse = nil
	s.ensurePrefillArenas(inkling2PrefillChunkFor(1300, s.chunkMax))
	if s.err != nil || s.prefillCap != 512 {
		t.Fatalf("base arenas after the fallback: err %v cap %d", s.err, s.prefillCap)
	}
	s.close()
	if d.live != 0 || d.frees != d.allocs {
		t.Fatalf("%d floats live, %d frees of %d allocs", d.live, d.frees, d.allocs)
	}

	// the per-token arena is allocated last, after the wide rings and arenas are in
	// place; its refusal is still the wide build's, so the base build follows
	resetInk2ScalarWide(t)
	d = newFakeInk2Budget(1 << 30)
	armed := true
	d.refuse = func(n int) bool {
		hit := armed && n == fakeInk2RingArena
		armed = armed && !hit
		return hit
	}
	s, err = newInkling2ResidentSized(m, d, fakeInk2RingCtx, fakeInk2RingArena)
	if err != nil || d.refusals != 1 {
		t.Fatalf("a refused per-token arena under the wide chunk failed the bind: %v (%d refusals)", err, d.refusals)
	}
	if s.err != nil || s.chunkMax != 512 || s.prefillCap != 0 || d.live != trunk {
		t.Fatalf("fallback after a late refusal: err %v chunk %d, %d floats (trunk %d)", s.err, s.chunkMax, d.live, trunk)
	}
	s.close()

	resetInk2ScalarWide(t)
	d = newFakeInk2Budget(1 << 30)
	d.refuse = func(n int) bool { return n == wideRing || n == baseRing }
	if s, err := newInkling2ResidentSized(m, d, fakeInk2RingCtx, fakeInk2RingArena); err == nil || s != nil {
		t.Fatal("a device that refuses the base rings bound a session")
	}
	if d.live != 0 || d.frees != d.allocs {
		t.Fatalf("after two refused builds: %d floats live, %d frees of %d allocs", d.live, d.frees, d.allocs)
	}
}

// After a refused wide bind the next sessions bind at the base chunk without trying
// the wide one, so the weights are uploaded once a session, and the refusal is
// announced once however many sessions meet it; the wide chunk is tried again later.
func TestInkling2PrefillChunkScalarRefusalHoldsOff(t *testing.T) {
	logged := captureInk2Log(t)
	resetInk2ScalarWide(t)
	t.Setenv(inkling2PrefillChunkEnv, "1024")
	m := fakeInk2RingModel()
	trunk := trunkInk2ResidentFloats(m, fakeInk2RingCtx, fakeInk2RingArena, true)
	wideRing := (fakeInk2RingWindow + 1024) * m.HP.KVHeads * m.HP.HeadDim
	d := newFakeInk2Budget(1 << 30)
	d.refuse = func(n int) bool { return n == wideRing }
	bind := func() (refusals, allocs int) {
		t.Helper()
		r, a := d.refusals, d.allocs
		s, err := newInkling2ResidentSized(m, d, fakeInk2RingCtx, fakeInk2RingArena)
		if err != nil || s.chunkMax != 512 || d.live != trunk {
			t.Fatalf("bind on a device that refuses the wide rings: %v", err)
		}
		s.close()
		return d.refusals - r, d.allocs - a
	}
	refused, twice := bind()
	if refused == 0 {
		t.Fatal("the first session did not try the wide chunk")
	}
	for i := 0; i < inkling2PrefillReprobeEvery; i++ {
		if r, once := bind(); r != 0 || once >= twice {
			t.Fatalf("session %d of the hold-off: %d refusals, %d allocations (a refused bind takes %d)", i, r, once, twice)
		}
	}
	if r, _ := bind(); r == 0 {
		t.Fatal("the wide chunk was never tried again")
	}
	if ink2LogLines(logged) != 1 {
		t.Fatalf("refusals logged %q, want one line", logged)
	}
	// the hold-off is not a latch: once the device takes the wide rings a session has them
	d.refuse = nil
	resetInk2ScalarWide(t)
	s, err := newInkling2ResidentSized(m, d, fakeInk2RingCtx, fakeInk2RingArena)
	if err != nil || s.chunkMax != 1024 {
		t.Fatalf("session once the device has room: %v", err)
	}
	s.close()
}

// The scalar prefill forwards min(chunk, n) tokens at a time and reports the widest
// forward it ran, the number the device parity test asserts on.
func TestInkling2PrefillChunkScalarForwardWidth(t *testing.T) {
	if inkling2ResidentCtx < 1300 {
		t.Skip("resident context below the longest prompt")
	}
	resetInk2ScalarWide(t)
	for _, c := range []struct {
		env   string
		chunk int
	}{{"", 512}, {"512", 512}, {"1024", 1024}, {"auto", 1024}} {
		for _, n := range []int{40, 512, 513, 1024, 1025, 1300} {
			t.Setenv(inkling2PrefillChunkEnv, c.env)
			d := newFakeInk2Budget(1 << 30)
			s, err := newInkling2ResidentSized(fakeInk2CohortModel(), d, 2048, 4096)
			if err != nil {
				t.Fatal(err)
			}
			prompt := fakeInk2CohortPrompt(n, 3)
			allocs := d.allocs
			got, err := s.prefill(context.Background(), prompt)
			if err != nil {
				t.Fatal(err)
			}
			if got != prompt[n-1]+1 || s.pos != n || s.prefillWidest != min(n, c.chunk) {
				t.Fatalf("chunk=%q pp%d: next %d pos %d widest forward %d, want %d, %d, %d",
					c.env, n, got, s.pos, s.prefillWidest, prompt[n-1]+1, n, min(n, c.chunk))
			}
			if c.chunk > 512 && d.allocs != allocs {
				t.Fatalf("chunk=%q pp%d: a wide session allocated during prefill", c.env, n)
			}
			s.close()
			if d.live != 0 {
				t.Fatalf("chunk=%q pp%d: %d floats live after close", c.env, n, d.live)
			}
		}
	}
}

// The interleaved cohort prefill packs to the cohort's own chunk: under the wide
// setting a short row still decodes beside a capped share of the long prompt, and
// once it is done the long prompt fills 1024-token chunks.
func TestInkling2PrefillChunkWideCohortInterleavesShortRow(t *testing.T) {
	const capTok, shortLen, shortMax, long = 64, 20, 6, 3 * inkling2PrefillChunk
	t.Setenv(inkling2PrefillChunkEnv, "1024")
	t.Setenv("FAK_INKLING2_COHORT_PREFILL", "1")
	t.Setenv("FAK_NATIVE_LONG_PREFILL_TOKEN_THRESHOLD", strconv.Itoa(capTok))
	d := newFakeInk2Budget(1 << 30)
	bt := NewInkling2LoopCohorts(fakeInk2CohortModel(), d)
	defer bt.Close()
	if _, err := bt.Generate(fakeInk2MixedCohort(nil, &[]ink2Event{}, make([][]int, 2))); err != nil {
		t.Fatal(err)
	}
	if bt.b.prefillT != 1024 {
		t.Fatalf("cohort prefill chunk %d, want 1024", bt.b.prefillT)
	}
	in, meta := bt.b.in.T.Buf(), bt.b.meta.T.Buf()
	var events []ink2Event
	got := make([][]int, 2)
	d.writeLog = nil
	if _, err := bt.Generate(fakeInk2MixedCohort(nil, &events, got)); err != nil {
		t.Fatal(err)
	}
	if want := fakeInk2ScalarTokens(t, fakeInk2CohortPrompt(shortLen, 0), shortMax); !reflect.DeepEqual(got[0], want) {
		t.Fatalf("short row emitted %v, scalar %v", got[0], want)
	}
	if want := fakeInk2ScalarTokens(t, fakeInk2CohortPrompt(long, 1), 2); !reflect.DeepEqual(got[1], want) {
		t.Fatalf("long row emitted %v, scalar %v", got[1], want)
	}
	chunks := fakeInk2ChunkTokens(t, d.fakeInk2Cohort, in, meta, bt.b.s.hp.Hidden)
	total, widest := 0, 0
	for k, n := range chunks {
		if k >= 1 && k < shortMax && n != capTok+1 {
			t.Fatalf("chunk %d carried %d tokens %v, want %d while the short row decodes", k, n, chunks, capTok+1)
		}
		total, widest = total+n, max(widest, n)
	}
	if total != long+shortLen+shortMax-1 || widest != 1024 {
		t.Fatalf("chunks %v carry %d tokens, widest %d; want %d and a 1024-token chunk", chunks, total, widest, long+shortLen+shortMax-1)
	}
}

// A row cancelled during a wide cohort prefill leaves every later 1024-token chunk to
// the rows still live.
func TestInkling2PrefillChunkWideCohortDropsCancelledRow(t *testing.T) {
	const long = 3 * 1024
	t.Setenv(inkling2PrefillChunkEnv, "1024")
	t.Setenv("FAK_INKLING2_COHORT_PREFILL", "1")
	d := newFakeInk2Budget(1 << 30)
	bt := NewInkling2LoopCohorts(fakeInk2CohortModel(), d)
	defer bt.Close()
	cohort := func() []Inkling2LoopBatchRequest {
		emit := func(int) bool { return true }
		return []Inkling2LoopBatchRequest{
			{Prompt: fakeInk2CohortPrompt(20, 0), MaxTokens: 2, Emit: emit},
			{Prompt: fakeInk2CohortPrompt(long, 1), MaxTokens: 2, Emit: emit},
			{Prompt: fakeInk2CohortPrompt(long, 2), MaxTokens: 2, Emit: emit},
		}
	}
	if _, err := bt.Generate(cohort()); err != nil {
		t.Fatal(err)
	}
	if bt.b.prefillT != 1024 {
		t.Fatalf("cohort prefill chunk %d, want 1024", bt.b.prefillT)
	}
	in, meta := bt.b.in.T.Buf(), bt.b.meta.T.Buf()
	ctxB, cancelB := context.WithCancel(context.Background())
	defer cancelB()
	d.writeLog = nil
	d.onWrite = func(dst compute.Ink2Ref) {
		if dst.T.Buf() == in {
			cancelB()
		}
	}
	reqs := cohort()
	reqs[1].Context = ctxB
	var live []int
	reqs[2].Emit = func(tok int) bool { live = append(live, tok); return true }
	results, err := bt.Generate(reqs)
	d.onWrite = nil
	if err != nil {
		t.Fatal(err)
	}
	if !errors.Is(results[1].Err, context.Canceled) || results[1].Generated != 0 || results[0].Generated != 2 {
		t.Fatalf("results %+v", results)
	}
	if want := fakeInk2ScalarTokens(t, fakeInk2CohortPrompt(long, 2), 2); !reflect.DeepEqual(live, want) {
		t.Fatalf("live long row emitted %v, scalar %v", live, want)
	}
	// the short row and the cancelled row share the first chunk; the live long row
	// then has the rest to itself, three full chunks
	chunks := fakeInk2ChunkTokens(t, d.fakeInk2Cohort, in, meta, bt.b.s.hp.Hidden)
	if want := []int{1024, 1024, 1024, 1024}; !reflect.DeepEqual(chunks, want) {
		t.Fatalf("prefill chunks %v, want %v: the cancelled row's prompt was packed", chunks, want)
	}
}
