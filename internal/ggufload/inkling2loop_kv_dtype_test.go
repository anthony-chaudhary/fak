package ggufload

import (
	"errors"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
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
