// fak-test:runtime fast est=2s lane=default
package ggufload

import (
	"errors"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/model"
)

func TestInkling2KVHalfFromParsesSelector(t *testing.T) {
	cases := []struct {
		name, value string
		set         bool
		want        bool
		refused     bool
	}{
		{"unset defaults to fp32", "", false, false, false},
		{"unset ignores lookup placeholder", "junk", false, false, false},
		{"explicit fp32", "fp32", true, false, false},
		{"explicit f32", "f32", true, false, false},
		{"f16", "f16", true, true, false},
		{"fp16 with case and space", " FP16 ", true, true, false},
		{"unknown dtype refused", "bf16", true, false, true},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			got, err := inkling2KVHalfFrom(func(string) (string, bool) { return c.value, c.set })
			if (err != nil) != c.refused {
				t.Fatalf("err = %v, refused want %v", err, c.refused)
			}
			if got != c.want {
				t.Fatalf("half = %v, want %v", got, c.want)
			}
		})
	}
}

func TestInkling2KVDTypeFromMatchesResidentAliases(t *testing.T) {
	for _, c := range []struct {
		value string
		want  Inkling2KVDType
	}{
		{"", Inkling2KVF32}, {"f32", Inkling2KVF32}, {"FP32", Inkling2KVF32},
		{"f16", Inkling2KVF16}, {" FP16 ", Inkling2KVF16},
		{"fp8", Inkling2KVFP8}, {" E4M3 ", Inkling2KVFP8}, {"e4m3fn", Inkling2KVFP8},
	} {
		got, err := Inkling2KVDTypeFrom(func(string) (string, bool) { return c.value, c.value != "" })
		if err != nil || got != c.want {
			t.Errorf("FAK_INKLING2_KV=%q: dtype=%v err=%v, want %v", c.value, got, err, c.want)
		}
	}
	if _, err := Inkling2KVDTypeFrom(func(k string) (string, bool) {
		if k != "FAK_INKLING2_KV" {
			t.Fatalf("parser read unexpected env %q", k)
		}
		return "bf16", true
	}); err == nil {
		t.Fatal("bf16 selector accepted")
	}
}

// inkling2NoHalfBackend is a backend with no f16 ops: a session asking for f16 must
// fall back to fp32 rather than allocate half caches nothing can read.
type inkling2NoHalfBackend struct{ compute.Backend }

func (inkling2NoHalfBackend) Name() string { return "stub" }

func TestInkling2ResidentHalfKVFallsBackWithoutF16Ops(t *testing.T) {
	defer func(v bool) { inkling2KVHalfRequested = v }(inkling2KVHalfRequested)
	inkling2KVHalfRequested = true
	if inkling2ResidentHalfKV(inkling2NoHalfBackend{}) {
		t.Fatal("backend without f16 decode ops kept f16 caches")
	}
	inkling2KVHalfRequested = false
	if inkling2ResidentHalfKV(inkling2NoHalfBackend{}) {
		t.Fatal("unrequested session selected f16 caches")
	}
}

func TestInkling2ResidentKVBytesF16GlobalSlotIs32KiBPerToken(t *testing.T) {
	hp := inkling2MiniKVShape()
	perGlobalToken := int64(8) * 2 * 1024 * 2 // 8 global slots, K and V, 1024 f16 each
	if perGlobalToken != 32<<10 {
		t.Fatalf("global f16 KV per token = %d bytes, want 32 KiB", perGlobalToken)
	}
	delta := inkling2ResidentKVBytesDtype(hp, 23553, false, 0, true) - inkling2ResidentKVBytesDtype(hp, 23552, false, 0, true)
	if want := perGlobalToken + 2*24*4; delta != want {
		t.Fatalf("marginal f16 bytes per token = %d, want %d (score row stays fp32)", delta, want)
	}
	// only the global slots shrink: fp32 minus f16 is half the global KV at ctx
	const ctx = 4096
	saved := inkling2ResidentKVBytes(hp, ctx, false, 0) - inkling2ResidentKVBytesDtype(hp, ctx, false, 0, true)
	if want := int64(8) * 2 * 1024 * 2 * ctx; saved != want {
		t.Fatalf("f16 saves %d bytes at ctx %d, want %d", saved, ctx, want)
	}
}

func TestInkling2KVFitF16AdmitsRoughlyTwiceTheFP32Ctx(t *testing.T) {
	hp := inkling2MiniKVShape()
	const free = int64(2) << 30
	budget := free - inkling2DeviceHeadroom(free)
	fp32Max := inkling2MaxResidentCtx(hp, false, 0, 49152, budget)
	f16Max := inkling2MaxResidentCtxDtype(hp, false, 0, true, 49152, budget)
	if f16Max*10 < fp32Max*19 {
		t.Fatalf("f16 max ctx %d is not about twice the fp32 max %d", f16Max, fp32Max)
	}
	// a ctx the fp32 fit refuses is admitted once the global caches are f16
	const ctx = 40000
	other := inkling2MiniOther(hp)
	if _, err := inkling2KVFit(hp, ctx, false, 0, other, free); err == nil {
		t.Fatalf("fp32 fit admitted ctx %d", ctx)
	}
	kv, err := inkling2KVFitDtype(hp, ctx, false, 0, true, other, free)
	if err != nil {
		t.Fatalf("f16 fit refused ctx %d: %v", ctx, err)
	}
	if kv != inkling2ResidentKVBytesDtype(hp, ctx, false, 0, true) {
		t.Fatalf("f16 fit returned %d KV bytes, want the f16 sizing", kv)
	}
	var over *Inkling2KVOverBudgetError
	if _, err := inkling2KVFitDtype(hp, 49152, false, 0, true, other, free); !errors.As(err, &over) {
		t.Fatalf("f16 fit at 49152 err = %v, want *Inkling2KVOverBudgetError", err)
	}
}

func TestInkling2KVHalfSlotKeepsLocalRingsFP32(t *testing.T) {
	hp := inkling2MiniKVShape()
	s := &inkling2Resident{hp: hp, kvHalf: true}
	global := 0
	for slot := range inkling2ResidentCaps(hp, 1, false, 0) {
		want := !hp.LocalLayer[inkling2SlotLayer(hp, slot)]
		if got := s.kvHalfSlot(slot); got != want {
			t.Fatalf("slot %d half = %v, want %v", slot, got, want)
		}
		if want {
			global++
		}
	}
	if global != 8 {
		t.Fatalf("global f16 slots = %d, want 8", global)
	}
	s.kvHalf = false
	for slot := range inkling2ResidentCaps(hp, 1, false, 0) {
		if s.kvHalfSlot(slot) {
			t.Fatalf("fp32 session reports slot %d half", slot)
		}
	}
}

func TestInkling2ResidentF16KVAllowsCheckpointButRefusesSpec(t *testing.T) {
	s := &inkling2Resident{kvHalf: true}
	if err := s.checkpointable(); err != nil {
		t.Fatalf("f16 session refused an in-place global-KV checkpoint: %v", err)
	}
	if err := s.enableSpec(4); err == nil {
		t.Fatal("f16 session enabled speculative decode, whose verify and draft ops read fp32 caches")
	}
}

func TestInkling2KVHalfFromRefusesUnknownSelectorValue(t *testing.T) {
	if _, err := inkling2KVHalfFrom(func(string) (string, bool) { return "int8", true }); err == nil {
		t.Fatal("unknown FAK_INKLING2_KV accepted")
	}
}

func TestInkling2ResidentKVBytesFP8GlobalSlotAndRTX5080Budgets(t *testing.T) {
	hp := inkling2MiniKVShape()
	// 8 global slots, each K+V = 2*(1024 B E4M3 codes + 8*4 B F32 scales) = 2112 B/tok
	perGlobalToken := int64(8) * 2 * (1024 + 8*4)
	if perGlobalToken != 16896 {
		t.Fatalf("global fp8 KV per token = %d bytes, want 16896", perGlobalToken)
	}
	delta := inkling2ResidentKVBytesFor(hp, 23553, false, 0, Inkling2KVFP8) - inkling2ResidentKVBytesFor(hp, 23552, false, 0, Inkling2KVFP8)
	if want := perGlobalToken + 2*24*4; delta != want {
		t.Fatalf("marginal fp8 bytes per token = %d, want %d (16896 global + 192 scores)", delta, want)
	}
	if got := inkling2ResidentPerPosBytesFor(hp, Inkling2KVFP8); got != delta {
		t.Fatalf("perPos fp8 = %d, want %d", got, delta)
	}

	// Verify exact KV + base-prefill budgets at 32k, 64k, 128k for Inkling-2-Mini
	// (Hidden=2304, T=128 batched prefill) targeting 16 GB RTX 5080.
	mini := hp
	mini.Hidden = 2304
	mini.ExpertsUsed = 8
	mini.ExpertWidth = 768
	const chunk = 128
	other := inkling2ResidentOtherBytes(mini, inkling2ArenaFloats, chunk, 0)
	fixed := inkling2ResidentFixedBytes(mini, inkling2ArenaFloats)
	if other <= fixed {
		t.Fatalf("other bytes %d did not include T=%d base prefill arenas (fixed=%d)", other, chunk, fixed)
	}
	for _, ctx := range []int{32768, 65536, 131072} {
		fp32KV := inkling2ResidentKVBytesFor(mini, ctx, true, chunk, Inkling2KVF32)
		f16KV := inkling2ResidentKVBytesFor(mini, ctx, true, chunk, Inkling2KVF16)
		fp8KV := inkling2ResidentKVBytesFor(mini, ctx, true, chunk, Inkling2KVFP8)
		if !(fp8KV < f16KV && f16KV < fp32KV) {
			t.Fatalf("ctx %d: expected fp8 (%d) < f16 (%d) < f32 (%d)", ctx, fp8KV, f16KV, fp32KV)
		}
		// FP8 saves (4096 - 2112) = 1984 B/tok per global slot vs FP16 across 8 global slots = 15872 B/tok.
		if saved := f16KV - fp8KV; saved != int64(8)*1984*int64(ctx) {
			t.Fatalf("ctx %d: fp16-fp8 saved %d bytes, want %d", ctx, saved, int64(8)*1984*int64(ctx))
		}
		t.Logf("ctx=%d fp32_kv=%.3fGiB f16_kv=%.3fGiB fp8_kv=%.3fGiB other_t128=%.3fGiB fp8_total=%.3fGiB",
			ctx,
			float64(fp32KV)/(1<<30), float64(f16KV)/(1<<30), float64(fp8KV)/(1<<30),
			float64(other)/(1<<30), float64(fp8KV+other)/(1<<30))
	}
}

type fakeInk2AutoCtxBackend struct {
	compute.Backend
	free int64
}

func (b fakeInk2AutoCtxBackend) Name() string { return "cuda" }
func (b fakeInk2AutoCtxBackend) Caps() compute.Caps {
	return compute.Caps{CapacityProbe: true}
}
func (b fakeInk2AutoCtxBackend) DeviceMemory() (total, free int64, known bool) {
	return 16 << 30, b.free, true
}

func TestInkling2ResolveAutoCtxPricesActiveKVDType(t *testing.T) {
	hp := inkling2MiniKVShape()
	// 1 GiB free after weights: after 15% headroom (~154 MiB) + 64 MiB arena + 162 MiB prefill +
	// ~323 MiB local SWA rings, ~321 MiB remains for global KV. F32/F16/FP8 scale inversely with perPos.
	be := fakeInk2AutoCtxBackend{free: 1 << 30}
	other := inkling2ResidentFixedBytes(hp, inkling2ArenaFloats) + (162 << 20)
	c32, err := inkling2ResolveAutoCtxFor(be, hp, false, 0, other, Inkling2KVF32)
	if err != nil {
		t.Fatalf("auto ctx f32: %v", err)
	}
	c16, err := inkling2ResolveAutoCtxFor(be, hp, false, 0, other, Inkling2KVF16)
	if err != nil {
		t.Fatalf("auto ctx f16: %v", err)
	}
	c8, err := inkling2ResolveAutoCtxFor(be, hp, false, 0, other, Inkling2KVFP8)
	if err != nil {
		t.Fatalf("auto ctx fp8: %v", err)
	}
	if !(c32 < c16 && c16 < c8) {
		t.Fatalf("auto ctx did not scale with KV dtype: f32=%d f16=%d fp8=%d", c32, c16, c8)
	}
	if c8 < c32*3 {
		t.Fatalf("fp8 auto ctx %d expected >= 3x f32 auto ctx %d", c8, c32)
	}
	// Regression for RTX 5080 aegis2 (2,594,439,168 B free after weights, batched T=512, kv=f32):
	// auto-ctx must account for local SWA rings + prefill arenas so it always passes inkling2KVFitFor.
	mini := hp
	mini.Hidden = 2304
	mini.ExpertsUsed = 8
	mini.ExpertWidth = 768
	const aegis2Free = int64(2594439168)
	aegis2Be := fakeInk2AutoCtxBackend{free: aegis2Free}
	aegis2Other := inkling2ResidentOtherBytes(mini, inkling2ArenaFloats, 512, 0)
	autoCtx, err := inkling2ResolveAutoCtxFor(aegis2Be, mini, true, 512, aegis2Other, Inkling2KVF32)
	if err != nil {
		t.Fatalf("aegis2 auto ctx f32: %v", err)
	}
	if _, err := inkling2KVFitFor(mini, autoCtx, true, 512, Inkling2KVF32, aegis2Other, aegis2Free); err != nil {
		t.Fatalf("aegis2 auto ctx %d failed its own fit check: %v", autoCtx, err)
	}
}

func TestInkling2KVFP8SlotAndSpecAndCheckpointPolicy(t *testing.T) {
	hp := inkling2MiniKVShape()
	s := &inkling2Resident{hp: hp, kvFP8: true}
	global := 0
	for slot := range inkling2ResidentCaps(hp, 1, false, 0) {
		want := !hp.LocalLayer[inkling2SlotLayer(hp, slot)]
		if got := s.kvFP8Slot(slot); got != want {
			t.Fatalf("slot %d fp8 = %v, want %v", slot, got, want)
		}
		if want {
			global++
		}
	}
	if global != 8 {
		t.Fatalf("global fp8 slots = %d, want 8", global)
	}
	if err := s.checkpointable(); err != nil {
		t.Fatalf("fp8 session refused in-place global-KV checkpoint: %v", err)
	}
	if err := s.enableSpec(4); err == nil {
		t.Fatal("fp8 session enabled speculative decode, whose verify ops read fp32 caches")
	}
}

type fakeInk2HalfOnlyBackend struct {
	inkling2NoHalfBackend
}

func (fakeInk2HalfOnlyBackend) Ink2QKVPostHalf(q, r, k, v, qclip, rclip, kcache, vcache, meta compute.Ink2Ref, p compute.Ink2AttnParams, cap int, cv *compute.Ink2KVConv) error {
	return nil
}
func (fakeInk2HalfOnlyBackend) Ink2AttendHalf(q, r, sinks, relBias, kcache, vcache, scores, out, meta compute.Ink2Ref, p compute.Ink2AttnParams, extent, window, cap int) error {
	return nil
}
func (fakeInk2HalfOnlyBackend) Ink2PHalfKVReady() error { return nil }
func (fakeInk2HalfOnlyBackend) Ink2PQKVPostHalf(q, r, k, v, qclip, rclip, kcache, vcache compute.Ink2Ref, p compute.Ink2AttnParams, cap, pos0, T int) error {
	return nil
}
func (fakeInk2HalfOnlyBackend) Ink2PAttendHalf(q, r, sinks, relBias, kcache, vcache, out compute.Ink2Ref, p compute.Ink2AttnParams, extent, window, cap, pos0, T int) error {
	return nil
}

func TestInkling2ResidentFP8KVFallbackHierarchy(t *testing.T) {
	if got := inkling2ResidentKVDType(fakeInk2HalfOnlyBackend{}, Inkling2KVFP8); got != Inkling2KVF16 {
		t.Fatalf("fp8 on half-only backend = %v, want f16 fallback", got)
	}
	if got := inkling2ResidentKVDType(inkling2NoHalfBackend{}, Inkling2KVFP8); got != Inkling2KVF32 {
		t.Fatalf("fp8 on f32-only backend = %v, want f32 fallback", got)
	}
}

type fakeInk2OverflowBackend struct {
	*fakeInk2Budget
	overflowFlag int32
	fp8Writes    int
	halfWrites   int
}

func (b *fakeInk2OverflowBackend) Ink2ReadInts(src compute.Ink2Ref, n int) ([]int32, error) {
	out := make([]int32, n)
	if n == 1 && b.overflowFlag != 0 {
		out[0] = b.overflowFlag
	}
	return out, nil
}
func (b *fakeInk2OverflowBackend) Ink2Write(dst compute.Ink2Ref, src []float32) error { return nil }
func (b *fakeInk2OverflowBackend) Ink2GraphReset()                                    {}
func (b *fakeInk2OverflowBackend) Ink2PFP8KVReady() error                             { return nil }
func (b *fakeInk2OverflowBackend) Ink2QKVPostFP8(q, r, k, v, qclip, rclip, kcache, vcache, kscale, vscale, overflow, meta compute.Ink2Ref, p compute.Ink2AttnParams, cap int, cv *compute.Ink2KVConv) error {
	b.fp8Writes++
	return nil
}
func (b *fakeInk2OverflowBackend) Ink2AttendFP8(q, r, sinks, relBias, kcache, vcache, kscale, vscale, scores, out, meta compute.Ink2Ref, p compute.Ink2AttnParams, extent, window, cap int) error {
	return nil
}
func (b *fakeInk2OverflowBackend) Ink2PQKVPostFP8(q, r, k, v, qclip, rclip, kcache, vcache, kscale, vscale, overflow compute.Ink2Ref, p compute.Ink2AttnParams, cap, pos0, T int) error {
	b.fp8Writes++
	return nil
}
func (b *fakeInk2OverflowBackend) Ink2PAttendFP8(q, r, sinks, relBias, kcache, vcache, kscale, vscale, out compute.Ink2Ref, p compute.Ink2AttnParams, extent, window, cap, pos0, T int) error {
	return nil
}
func (b *fakeInk2OverflowBackend) Ink2QKVPostHalf(q, r, k, v, qclip, rclip, kcache, vcache, meta compute.Ink2Ref, p compute.Ink2AttnParams, cap int, cv *compute.Ink2KVConv) error {
	b.halfWrites++
	return nil
}
func (b *fakeInk2OverflowBackend) Ink2AttendHalf(q, r, sinks, relBias, kcache, vcache, scores, out, meta compute.Ink2Ref, p compute.Ink2AttnParams, extent, window, cap int) error {
	return nil
}
func (b *fakeInk2OverflowBackend) Ink2PHalfKVReady() error { return nil }
func (b *fakeInk2OverflowBackend) Ink2PQKVPostHalf(q, r, k, v, qclip, rclip, kcache, vcache compute.Ink2Ref, p compute.Ink2AttnParams, cap, pos0, T int) error {
	b.halfWrites++
	return nil
}
func (b *fakeInk2OverflowBackend) Ink2PAttendHalf(q, r, sinks, relBias, kcache, vcache, out compute.Ink2Ref, p compute.Ink2AttnParams, extent, window, cap, pos0, T int) error {
	return nil
}

func TestInkling2FP8ScaleOverflowFallbackToFP16(t *testing.T) {
	hp := model.Inkling2LoopHParams{
		Hidden: 64, Layers: 2, Heads: 4, KVHeads: 2, HeadDim: 16,
		SlidingWindow: 16, ConvKernel: 4, Passes: 1,
		LocalLayer: []bool{true, false},
	}
	be := &fakeInk2OverflowBackend{fakeInk2Budget: newFakeInk2Budget(1 << 30), overflowFlag: 1}
	s := &inkling2Resident{
		hp: hp, be: be, dev: be,
		kvFP8: true, fp8Dev: be,
		caps: []int{16, 64},
		kc:   make([]ink2Ref, 2), vc: make([]ink2Ref, 2),
		ks: make([]ink2Ref, 2), vs: make([]ink2Ref, 2),
		hist: make([]ink2Ref, 2),
		pos:  32,
	}
	// Slot 0 local F32, slot 1 global FP8
	s.kc[0], s.vc[0] = s.alloc(16*32), s.alloc(16*32)
	s.kc[1], s.vc[1] = s.alloc(compute.Ink2FP8KVCodeFloats(64, 2, 16)), s.alloc(compute.Ink2FP8KVCodeFloats(64, 2, 16))
	s.ks[1], s.vs[1] = s.alloc(compute.Ink2FP8KVScaleFloats(64, 2)), s.alloc(compute.Ink2FP8KVScaleFloats(64, 2))
	s.hist[0], s.hist[1] = s.alloc(s.histFloats()), s.alloc(s.histFloats())
	s.kvOverflow = s.alloc(1)

	overflow, err := s.checkFP8Overflow()
	if err != nil || !overflow {
		t.Fatalf("checkFP8Overflow = (%v, %v), want (true, nil)", overflow, err)
	}
	if err := s.fallbackFP8ToFP16(); err != nil {
		t.Fatalf("fallbackFP8ToFP16: %v", err)
	}
	if s.kvFP8 || !s.kvHalf || s.kvDType() != Inkling2KVF16 {
		t.Fatalf("post-fallback dtype = %v (fp8=%v half=%v), want f16", s.kvDType(), s.kvFP8, s.kvHalf)
	}
	if s.pos != 0 {
		t.Fatalf("post-fallback pos = %d, want 0", s.pos)
	}
	if s.ks[1].T.Buf() != nil || s.vs[1].T.Buf() != nil || s.kvOverflow.T.Buf() != nil {
		t.Fatal("post-fallback left FP8 scale or overflow buffers attached")
	}
	s.close()
	if be.live != 0 || be.frees != be.allocs {
		t.Fatalf("after close: live=%d frees=%d allocs=%d", be.live, be.frees, be.allocs)
	}
}

// fakeInk2HalfBudget is a resident test backend that supports both f16 decode and f16
// prefill KV ops so newInkling2ResidentRing exercises the real kvHalf selection path.
type fakeInk2HalfBudget struct {
	*fakeInk2Budget
	fakeInk2HalfPrefill
}

func (fakeInk2HalfBudget) Ink2QKVPostHalf(q, r, k, v, qclip, rclip, kcache, vcache, meta compute.Ink2Ref, p compute.Ink2AttnParams, cap int, cv *compute.Ink2KVConv) error {
	return nil
}

func (fakeInk2HalfBudget) Ink2AttendHalf(q, r, sinks, relBias, kcache, vcache, scores, out, meta compute.Ink2Ref, p compute.Ink2AttnParams, extent, window, cap int) error {
	return nil
}

func TestInkling2ResidentPerPosBytesDtypeAccountsForF16(t *testing.T) {
	hp := inkling2MiniKVShape()
	f32 := inkling2ResidentPerPosBytesDtype(hp, false)
	f16 := inkling2ResidentPerPosBytesDtype(hp, true)
	if f32 != inkling2ResidentPerPosBytes(hp) {
		t.Fatalf("f32 perPos = %d, want inkling2ResidentPerPosBytes %d", f32, inkling2ResidentPerPosBytes(hp))
	}
	wantF16 := int64(8)*2*1024*2 + 2*24*4
	if f16 != wantF16 {
		t.Fatalf("f16 perPos = %d, want %d", f16, wantF16)
	}
	if f32-f16 != int64(8)*2*1024*2 {
		t.Fatalf("f32-f16 delta = %d, want half of global KV (%d)", f32-f16, int64(8)*2*1024*2)
	}

	be := &fakeInk2HalfBudget{fakeInk2Budget: newFakeInk2Budget(512 << 20)}
	ctxF32, err := inkling2ResolveAutoCtx(be, hp, 0, false)
	if err != nil {
		t.Fatalf("resolve auto ctx f32: %v", err)
	}
	ctxF16, err := inkling2ResolveAutoCtx(be, hp, 0, true)
	if err != nil {
		t.Fatalf("resolve auto ctx f16: %v", err)
	}
	if ctxF16 <= ctxF32 || ctxF16*10 < ctxF32*19 {
		t.Fatalf("auto ctx f16=%d, f32=%d; want f16 roughly double f32 under constrained memory", ctxF16, ctxF32)
	}
}

func TestInkling2ResidentAutoCtxSelectsHalfKV(t *testing.T) {
	defer func(v bool) { inkling2KVHalfRequested = v }(inkling2KVHalfRequested)
	inkling2KVHalfRequested = true
	m := fakeInk2RingModel()

	t.Run("known free memory enables kvHalf", func(t *testing.T) {
		be := &fakeInk2HalfBudget{fakeInk2Budget: newFakeInk2Budget(1 << 30)}
		s, err := newInkling2ResidentRing(m, be, inkling2ResidentCtxAuto, fakeInk2RingArena, inkling2PrefillChunkBase, inkling2PrefillChunkBase)
		if err != nil {
			t.Fatalf("newInkling2ResidentRing(auto, known free): %v", err)
		}
		defer s.close()
		if s.ctx <= 0 {
			t.Fatalf("resolved auto ctx = %d, want > 0", s.ctx)
		}
		if !s.kvHalf {
			t.Fatal("auto-context session with FAK_INKLING2_KV=f16 selected fp32 caches (kvHalf=false)")
		}
	})

	t.Run("unknown free memory fallback enables kvHalf", func(t *testing.T) {
		be := &fakeInk2HalfBudget{fakeInk2Budget: newFakeInk2Budget(0)}
		s, err := newInkling2ResidentRing(m, be, inkling2ResidentCtxAuto, fakeInk2RingArena, inkling2PrefillChunkBase, inkling2PrefillChunkBase)
		if err != nil {
			t.Fatalf("newInkling2ResidentRing(auto, unknown free): %v", err)
		}
		defer s.close()
		if s.ctx != Inkling2ResidentCtxDefault {
			t.Fatalf("unknown-free auto ctx = %d, want default %d", s.ctx, Inkling2ResidentCtxDefault)
		}
		if !s.kvHalf {
			t.Fatal("unknown-free auto-context fallback selected fp32 caches (kvHalf=false)")
		}
	})

	t.Run("cohort binding ctx 0 keeps kvHalf false", func(t *testing.T) {
		be := &fakeInk2HalfBudget{fakeInk2Budget: newFakeInk2Budget(1 << 30)}
		s, err := newInkling2ResidentRing(m, be, 0, fakeInk2RingArena, inkling2PrefillChunkBase, inkling2PrefillChunkBase)
		if err != nil {
			t.Fatalf("newInkling2ResidentRing(ctx=0): %v", err)
		}
		defer s.close()
		if s.ctx != 0 {
			t.Fatalf("cohort binding ctx = %d, want 0", s.ctx)
		}
		if s.kvHalf {
			t.Fatal("cohort binding (ctx=0) set kvHalf=true")
		}
	})
}
