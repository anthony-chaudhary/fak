package ggufload

import (
	"math"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/model"
)

type convCommitCall struct {
	hist, log compute.Ink2Ref
	channels  int
	past, L   int
	keep      int
}

type fakeSpecBackend struct {
	compute.Backend
	compute.Inkling2Device
	compute.Inkling2PrefillDevice
	compute.Inkling2VerifyDevice

	keepVal      int
	capturing    bool
	captureCalls []convCommitCall
	slotOps      map[int][]convCommitCall
	executed     []convCommitCall
	launches     []int
}

func newFakeSpecBackend() *fakeSpecBackend {
	return &fakeSpecBackend{slotOps: map[int][]convCommitCall{}}
}

func (*fakeSpecBackend) Name() string { return "fake-spec" }

func (f *fakeSpecBackend) Ink2Write(_ compute.Ink2Ref, data []float32) error {
	if len(data) > 0 {
		f.keepVal = int(int32(math.Float32bits(data[0])))
	}
	return nil
}

func (f *fakeSpecBackend) Ink2GraphBegin() error {
	f.capturing = true
	f.captureCalls = nil
	return nil
}

func (f *fakeSpecBackend) Ink2GraphEndSlot(slot int) error {
	f.capturing = false
	f.slotOps[slot] = append([]convCommitCall(nil), f.captureCalls...)
	f.captureCalls = nil
	return nil
}

func (f *fakeSpecBackend) Ink2GraphLaunchSlot(slot int) error {
	f.launches = append(f.launches, slot)
	for _, op := range f.slotOps[slot] {
		op.keep = f.keepVal
		f.executed = append(f.executed, op)
	}
	return nil
}

func (f *fakeSpecBackend) Ink2PConvCommitDev(hist, log, _ compute.Ink2Ref, channels, past, L int) error {
	call := convCommitCall{hist: hist, log: log, channels: channels, past: past, L: L, keep: f.keepVal}
	if f.capturing {
		f.captureCalls = append(f.captureCalls, call)
		return nil
	}
	f.executed = append(f.executed, call)
	return nil
}

func TestInkling2SpecCommitVerifyRewindInvariants(t *testing.T) {
	fb := newFakeSpecBackend()
	hp := model.Inkling2LoopHParams{
		Hidden:     64,
		KVHeads:    2,
		HeadDim:    16,
		ConvKernel: 4,
	}
	kvDim := hp.KVHeads * hp.HeadDim
	wantChannels := 2*kvDim + 2*hp.Hidden
	wantPast := hp.ConvKernel - 1

	s := &inkling2Resident{
		hp:      hp,
		be:      fb,
		dev:     fb,
		graph:   true,
		specMax: 5,
		pos:     19,
		ctx:     4096,
		hist:    []ink2Ref{{Off: 10}, {Off: 20}},
		convLog: []ink2Ref{{Off: 100}, {Off: 200}},
	}

	// 1. keep == 0 (full rewind): pos unchanged, captures inkling2CommitSlot, executes with keep=0, L=past+5.
	if err := s.commitVerify(5, 0); err != nil {
		t.Fatalf("commitVerify(5, 0): %v", err)
	}
	if s.pos != 19 {
		t.Fatalf("keep=0 moved pos to %d, want 19", s.pos)
	}
	if !s.keepReady {
		t.Fatal("keepReady not set after full-width commit graph capture")
	}
	if len(fb.launches) != 1 || fb.launches[0] != inkling2CommitSlot {
		t.Fatalf("launches = %v, want [%d]", fb.launches, inkling2CommitSlot)
	}
	if len(fb.executed) != 2 {
		t.Fatalf("executed %d conv commits, want 2", len(fb.executed))
	}
	for i, c := range fb.executed {
		if c.channels != wantChannels || c.past != wantPast || c.L != wantPast+5 || c.keep != 0 {
			t.Errorf("slot %d commit = %+v, want channels=%d past=%d L=%d keep=0", i, c, wantChannels, wantPast, wantPast+5)
		}
	}

	// 2. 0 < keep < T (partial accept): replays inkling2CommitSlot with updated keep=2, advances pos by 2.
	fb.executed = nil
	if err := s.commitVerify(5, 2); err != nil {
		t.Fatalf("commitVerify(5, 2): %v", err)
	}
	if s.pos != 21 {
		t.Fatalf("keep=2 moved pos to %d, want 21", s.pos)
	}
	if len(fb.executed) != 2 || fb.executed[0].keep != 2 || fb.executed[1].keep != 2 {
		t.Fatalf("partial commit executed = %+v, want keep=2 on both slots", fb.executed)
	}

	// 3. keep == T (full accept): advances pos by 5, skips conv commit (hist already holds state after all T tokens).
	fb.executed = nil
	if err := s.commitVerify(5, 5); err != nil {
		t.Fatalf("commitVerify(5, 5): %v", err)
	}
	if s.pos != 26 {
		t.Fatalf("keep=5 moved pos to %d, want 26", s.pos)
	}
	if len(fb.executed) != 0 {
		t.Fatalf("full accept executed %d conv commits, want 0", len(fb.executed))
	}

	// 4. Tail width T=3 < specMax: uses its own tail commit slot and L=past+3.
	fb.executed = nil
	_, wantTailCommit := inkling2TailSlots(3)
	if err := s.commitVerify(3, 1); err != nil {
		t.Fatalf("commitVerify(3, 1): %v", err)
	}
	if s.pos != 27 {
		t.Fatalf("tail keep=1 moved pos to %d, want 27", s.pos)
	}
	if !s.tailKeepReady[3] {
		t.Fatal("tailKeepReady[3] not set")
	}
	if got := fb.launches[len(fb.launches)-1]; got != wantTailCommit {
		t.Fatalf("tail commit launched slot %d, want %d", got, wantTailCommit)
	}
	if len(fb.executed) != 2 || fb.executed[0].L != wantPast+3 || fb.executed[0].keep != 1 {
		t.Fatalf("tail commit executed = %+v, want L=%d keep=1", fb.executed, wantPast+3)
	}

	// 5. Invalid keep bounds fail without mutating pos.
	for _, bad := range []int{-1, 6} {
		if err := s.commitVerify(5, bad); err == nil {
			t.Errorf("commitVerify(5, %d) succeeded, want error", bad)
		}
		if s.pos != 27 {
			t.Errorf("invalid keep=%d mutated pos to %d, want 27", bad, s.pos)
		}
	}
}

func TestInkling2TailSlotsDistinctAndInRange(t *testing.T) {
	seen := map[int]bool{
		0:                  true, // decode graph slot
		inkling2VerifySlot: true,
		inkling2CommitSlot: true,
		3:                  true, // draft graph slot
	}
	const cudaGraphSlots = 24
	for width := 1; width <= inkling2SpecMaxDraft; width++ {
		v, c := inkling2TailSlots(width)
		if v < 0 || v >= cudaGraphSlots || c < 0 || c >= cudaGraphSlots {
			t.Fatalf("width %d slots (%d, %d) out of range [0, %d)", width, v, c, cudaGraphSlots)
		}
		if seen[v] {
			t.Fatalf("width %d verify slot %d collides with an existing slot", width, v)
		}
		seen[v] = true
		if seen[c] {
			t.Fatalf("width %d commit slot %d collides with an existing slot", width, c)
		}
		seen[c] = true
	}
}

func TestInkling2SpecVerifyRespectsSessionCtx(t *testing.T) {
	s := &inkling2Resident{
		specMax: 5,
		pos:     510,
		ctx:     512,
	}
	if _, err := s.verify([]int{1, 2, 3}); err == nil || !strings.Contains(err.Error(), "512-token global cache") {
		t.Fatalf("verify past s.ctx=512 err = %v, want 512-token global cache error", err)
	}
}

type fakeHalfSpecBackend struct {
	specCapableBackend
	compute.Inkling2HalfKVDevice
	compute.Inkling2PrefillHalfKV
}

func (fakeHalfSpecBackend) Name() string            { return "fake-half-spec" }
func (fakeHalfSpecBackend) Ink2PHalfKVReady() error { return nil }

func TestInkling2SpecKVHalfAutoDisableAndExplicitRefusal(t *testing.T) {
	defer func(v bool) { inkling2KVHalfRequested = v }(inkling2KVHalfRequested)
	inkling2KVHalfRequested = true

	be := fakeHalfSpecBackend{}
	m := &Inkling2LoopModel{SpecDefault: Inkling2MetalSpecDefault(be)}

	// Unset FAK_INKLING2_SPEC + FAK_INKLING2_KV=f16 auto-disables spec instead of crashing.
	setOrUnset(t, "FAK_INKLING2_SPEC", "")
	if !inkling2SpecAutoDisabledForKV(be, func(string) (string, bool) { return "", false }) {
		t.Fatal("unset FAK_INKLING2_SPEC with f16 KV did not auto-disable spec")
	}
	if got := Inkling2SpecLabel(m, be); got != "spec=off" {
		t.Fatalf("Inkling2SpecLabel with f16 KV and unset FAK_INKLING2_SPEC = %q, want spec=off", got)
	}

	// Blank FAK_INKLING2_SPEC="" behaves like unset (respects m.SpecDefault and auto-disables under f16).
	t.Setenv("FAK_INKLING2_SPEC", "")
	if got := inkling2SpecDepthFor(m); got != m.SpecDefault.Depth {
		t.Fatalf("blank FAK_INKLING2_SPEC depth = %d, want model default %d", got, m.SpecDefault.Depth)
	}
	if got := Inkling2SpecLabel(m, be); got != "spec=off" {
		t.Fatalf("Inkling2SpecLabel with f16 KV and blank FAK_INKLING2_SPEC = %q, want spec=off", got)
	}

	// Explicit FAK_INKLING2_SPEC=4 + f16 KV is NOT auto-disabled; enableSpec refuses with actionable message.
	t.Setenv("FAK_INKLING2_SPEC", "4")
	if inkling2SpecAutoDisabledForKV(be, func(k string) (string, bool) { return "4", true }) {
		t.Fatal("explicit FAK_INKLING2_SPEC=4 was auto-disabled")
	}
	s := &inkling2Resident{kvHalf: true}
	err := s.enableSpec(4)
	if err == nil || !strings.Contains(err.Error(), "FAK_INKLING2_SPEC=0") {
		t.Fatalf("enableSpec on kvHalf err = %v, want error naming FAK_INKLING2_SPEC=0", err)
	}
}
