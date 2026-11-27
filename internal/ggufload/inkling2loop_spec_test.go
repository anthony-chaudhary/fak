package ggufload

import (
	"errors"
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

type fakeVerifyAllKVBackend struct {
	fakeHalfSpecBackend
	compute.Inkling2FP8KVDevice
	compute.Inkling2NVFP4KVDevice
	compute.Inkling2VerifyHalfKVDevice
	compute.Inkling2VerifyFP8KVDevice
	compute.Inkling2VerifyNVFP4KVDevice
	halfReadyErr  error
	fp8ReadyErr   error
	nvfp4ReadyErr error
}

func (f fakeVerifyAllKVBackend) Name() string             { return "fake-verify-all-kv" }
func (f fakeVerifyAllKVBackend) Ink2PHalfKVReady() error  { return f.halfReadyErr }
func (f fakeVerifyAllKVBackend) Ink2PFP8KVReady() error   { return f.fp8ReadyErr }
func (f fakeVerifyAllKVBackend) Ink2PNVFP4KVReady() error { return f.nvfp4ReadyErr }
func (fakeVerifyAllKVBackend) Ink2Alloc(n int) (compute.Tensor, error) {
	return compute.Tensor{Shape: []int{n}}, nil
}

type fakeFP8NoVerifyBackend struct {
	specCapableBackend
	compute.Inkling2FP8KVDevice
}

func (fakeFP8NoVerifyBackend) Name() string           { return "fake-fp8-no-verify" }
func (fakeFP8NoVerifyBackend) Ink2PFP8KVReady() error { return nil }

type fakeNVFP4NoVerifyBackend struct {
	specCapableBackend
	compute.Inkling2NVFP4KVDevice
}

func (fakeNVFP4NoVerifyBackend) Name() string             { return "fake-nvfp4-no-verify" }
func (fakeNVFP4NoVerifyBackend) Ink2PNVFP4KVReady() error { return nil }

func TestInkling2SpecKVVerifySupportAndAutoDisable(t *testing.T) {
	unset := func(string) (string, bool) { return "", false }
	kvEnv := func(v string) func(string) (string, bool) {
		return func(k string) (string, bool) {
			if k == "FAK_INKLING2_KV" {
				return v, true
			}
			return "", false
		}
	}

	all := fakeVerifyAllKVBackend{}
	for _, dtype := range []Inkling2KVDType{Inkling2KVF32, Inkling2KVF16, Inkling2KVFP8, Inkling2KVNVFP4} {
		if !inkling2BackendSupportsSpecKV(all, dtype) {
			t.Fatalf("inkling2BackendSupportsSpecKV(all, %s) = false, want true", dtype)
		}
	}
	if inkling2SpecAutoDisabledForKV(all, kvEnv("f16")) {
		t.Fatal("f16 with VerifyHalfKVDevice should not auto-disable spec")
	}
	if inkling2SpecAutoDisabledForKV(all, kvEnv("fp8")) {
		t.Fatal("fp8 with VerifyFP8KVDevice should not auto-disable spec")
	}
	if inkling2SpecAutoDisabledForKV(all, kvEnv("nvfp4")) {
		t.Fatal("nvfp4 with VerifyNVFP4KVDevice should not auto-disable spec")
	}

	// Without Inkling2Verify*KVDevice, non-fp32 KV auto-disables spec and enableSpec refuses.
	fp8Only := fakeFP8NoVerifyBackend{}
	if inkling2BackendSupportsSpecKV(fp8Only, Inkling2KVFP8) {
		t.Fatal("fp8 without VerifyFP8KVDevice unexpectedly supported")
	}
	if !inkling2SpecAutoDisabledForKV(fp8Only, kvEnv("fp8")) {
		t.Fatal("fp8 without VerifyFP8KVDevice did not auto-disable spec")
	}
	nvfp4Only := fakeNVFP4NoVerifyBackend{}
	if inkling2BackendSupportsSpecKV(nvfp4Only, Inkling2KVNVFP4) {
		t.Fatal("nvfp4 without VerifyNVFP4KVDevice unexpectedly supported")
	}
	if !inkling2SpecAutoDisabledForKV(nvfp4Only, kvEnv("nvfp4")) {
		t.Fatal("nvfp4 without VerifyNVFP4KVDevice did not auto-disable spec")
	}

	// Ready() failure disables spec KV support.
	notReady := fakeVerifyAllKVBackend{
		halfReadyErr:  errors.ErrUnsupported,
		fp8ReadyErr:   errors.ErrUnsupported,
		nvfp4ReadyErr: errors.ErrUnsupported,
	}
	for _, dtype := range []Inkling2KVDType{Inkling2KVF16, Inkling2KVFP8, Inkling2KVNVFP4} {
		if inkling2BackendSupportsSpecKV(notReady, dtype) {
			t.Fatalf("notReady backend should not support %s", dtype)
		}
	}

	// enableSpec keeps kvAutoHalf when VerifyHalfKVDevice is present, and refuses when missing.
	sOK := &inkling2Resident{
		be:         all,
		dev:        all,
		batched:    true,
		kvHalf:     true,
		kvAutoHalf: true,
		hp:         model.Inkling2LoopHParams{ConvKernel: 4, Hidden: 8, KVHeads: 2, HeadDim: 4},
		output:     ink2Mat{rows: 16},
	}
	if err := sOK.enableSpec(2); err != nil {
		t.Fatalf("enableSpec with VerifyHalfKVDevice: %v", err)
	}
	if !sOK.kvHalf {
		t.Fatal("enableSpec downgraded kvAutoHalf even though VerifyHalfKVDevice is supported")
	}
	_ = unset
}

type fakeVerifyDispatchBackend struct {
	compute.Backend
	compute.Inkling2Device
	compute.Inkling2PrefillDevice
	compute.Inkling2VerifyDevice
	compute.Inkling2VerifyHalfKVDevice
	compute.Inkling2VerifyFP8KVDevice
	compute.Inkling2VerifyNVFP4KVDevice
	calls        []string
	verifyIdx    compute.Ink2Ref
	overflowRef  compute.Ink2Ref
	overflowFlag int32
}

func (*fakeVerifyDispatchBackend) Name() string             { return "fake-verify-dispatch" }
func (*fakeVerifyDispatchBackend) Ink2PHalfKVReady() error  { return nil }
func (*fakeVerifyDispatchBackend) Ink2PFP8KVReady() error   { return nil }
func (*fakeVerifyDispatchBackend) Ink2PNVFP4KVReady() error { return nil }
func (*fakeVerifyDispatchBackend) Ink2Alloc(n int) (compute.Tensor, error) {
	return compute.Tensor{Shape: []int{n}}, nil
}
func (*fakeVerifyDispatchBackend) Ink2Write(compute.Ink2Ref, []float32) error { return nil }
func (f *fakeVerifyDispatchBackend) Ink2ReadInts(r compute.Ink2Ref, n int) ([]int32, error) {
	if r.T.Buf() != nil && r.T.Buf() == f.overflowRef.T.Buf() && r.Off == f.overflowRef.Off {
		return []int32{f.overflowFlag}, nil
	}
	return make([]int32, n), nil
}
func (*fakeVerifyDispatchBackend) Ink2GraphBegin() error         { return errors.ErrUnsupported }
func (*fakeVerifyDispatchBackend) Ink2GraphEndSlot(int) error    { return nil }
func (*fakeVerifyDispatchBackend) Ink2GraphLaunchSlot(int) error { return nil }
func (*fakeVerifyDispatchBackend) Ink2PMix([]compute.Ink2Ref, compute.Ink2Ref, int, compute.Ink2Ref, int, int) error {
	return nil
}
func (*fakeVerifyDispatchBackend) Ink2PRMS(compute.Ink2Ref, compute.Ink2Ref, compute.Ink2Ref, int, int, float32) error {
	return nil
}
func (*fakeVerifyDispatchBackend) Ink2RMS(compute.Ink2Ref, compute.Ink2Ref, compute.Ink2Ref, int, int, float32) error {
	return nil
}
func (*fakeVerifyDispatchBackend) Ink2Route(compute.Ink2Ref, compute.Ink2Ref, compute.Ink2Ref, compute.Ink2Ref, int, int, float32) error {
	return nil
}
func (*fakeVerifyDispatchBackend) Ink2Combine(compute.Ink2Ref, compute.Ink2Ref, compute.Ink2Ref, int, int) error {
	return nil
}
func (*fakeVerifyDispatchBackend) Ink2Situ(compute.Ink2Ref, compute.Ink2Ref, int, int, int, float64, float64) error {
	return nil
}
func (*fakeVerifyDispatchBackend) Ink2SigmoidMul(compute.Ink2Ref, compute.Ink2Ref, int) error {
	return nil
}
func (*fakeVerifyDispatchBackend) Ink2VMix([]compute.Ink2Ref, compute.Ink2Ref, int, []compute.Ink2Ref, float32, compute.Ink2Ref, int, int) (bool, error) {
	return true, nil
}
func (*fakeVerifyDispatchBackend) Ink2SoftcapArgmax(compute.Ink2Ref, int, float32, compute.Ink2Ref) error {
	return nil
}
func (*fakeVerifyDispatchBackend) Ink2ExpertMatVec(compute.Tensor, int, compute.Ink2Ref, compute.Ink2Ref, int, compute.Ink2Ref, int, compute.Ink2Ref, *compute.Ink2Situ) (bool, error) {
	return true, nil
}
func (*fakeVerifyDispatchBackend) Ink2PMatMul(compute.Tensor, compute.Ink2Ref, compute.Ink2Ref, int) error {
	return nil
}
func (*fakeVerifyDispatchBackend) Ink2VRows(compute.Tensor, compute.Ink2Ref, compute.Ink2Ref, int) (bool, error) {
	return false, nil
}
func (*fakeVerifyDispatchBackend) Ink2VQ8Rows(compute.Tensor, compute.Ink2Ref, compute.Ink2Ref, compute.Ink2Ref, compute.Ink2Ref, int, bool) (bool, error) {
	return false, nil
}
func (*fakeVerifyDispatchBackend) Ink2VQ8Segs([]compute.Tensor, []compute.Ink2Ref, []compute.Ink2Ref, compute.Ink2Ref, int) (bool, error) {
	return false, nil
}
func (*fakeVerifyDispatchBackend) Ink2PConv(compute.Ink2Ref, compute.Ink2Ref, compute.Ink2Ref, compute.Ink2Ref, compute.Ink2Ref, compute.Ink2Ref, int, int, int, float32, int) error {
	return nil
}
func (*fakeVerifyDispatchBackend) Ink2PConvLog(compute.Ink2Ref, compute.Ink2Ref, compute.Ink2Ref, compute.Ink2Ref, compute.Ink2Ref, compute.Ink2Ref, compute.Ink2Ref, int, int, int, float32, int) error {
	return nil
}
func (*fakeVerifyDispatchBackend) Ink2PConvPairLog([]compute.Ink2Ref, []compute.Ink2Ref, []compute.Ink2Ref, []compute.Ink2Ref, []compute.Ink2Ref, []compute.Ink2Ref, []float32, int, int, int, int) error {
	return nil
}
func (*fakeVerifyDispatchBackend) Ink2PRoute(compute.Ink2Ref, compute.Ink2Ref, compute.Ink2Ref, compute.Ink2Ref, int, int, float32, int) error {
	return nil
}
func (*fakeVerifyDispatchBackend) Ink2PSitu(compute.Ink2Ref, compute.Ink2Ref, int, int, int, float64, float64) error {
	return nil
}
func (*fakeVerifyDispatchBackend) Ink2PExpertGrouped(compute.Tensor, int, int, compute.Ink2Ref, compute.Ink2Ref, int, int, compute.Ink2Ref, int, compute.Ink2Ref, compute.Ink2Ref, bool) (bool, error) {
	return true, nil
}
func (*fakeVerifyDispatchBackend) Ink2PCombineConvLog(compute.Ink2Ref, compute.Ink2Ref, int, compute.Ink2Ref, compute.Ink2Ref, compute.Ink2Ref, compute.Ink2Ref, compute.Ink2Ref, int, int, float32, int) error {
	return nil
}
func (f *fakeVerifyDispatchBackend) Ink2PQKVPostDev(compute.Ink2Ref, compute.Ink2Ref, compute.Ink2Ref, compute.Ink2Ref, compute.Ink2Ref, compute.Ink2Ref, compute.Ink2Ref, compute.Ink2Ref, compute.Ink2Ref, compute.Ink2AttnParams, int, int) error {
	f.calls = append(f.calls, "post_f32")
	return nil
}
func (f *fakeVerifyDispatchBackend) Ink2PAttendDev(compute.Ink2Ref, compute.Ink2Ref, compute.Ink2Ref, compute.Ink2Ref, compute.Ink2Ref, compute.Ink2Ref, compute.Ink2Ref, compute.Ink2Ref, compute.Ink2AttnParams, int, int, int, int) error {
	f.calls = append(f.calls, "attend_f32")
	return nil
}
func (f *fakeVerifyDispatchBackend) Ink2PQKVPostHalfDev(compute.Ink2Ref, compute.Ink2Ref, compute.Ink2Ref, compute.Ink2Ref, compute.Ink2Ref, compute.Ink2Ref, compute.Ink2Ref, compute.Ink2Ref, compute.Ink2Ref, compute.Ink2AttnParams, int, int) error {
	f.calls = append(f.calls, "post_f16")
	return nil
}
func (f *fakeVerifyDispatchBackend) Ink2PAttendHalfDev(compute.Ink2Ref, compute.Ink2Ref, compute.Ink2Ref, compute.Ink2Ref, compute.Ink2Ref, compute.Ink2Ref, compute.Ink2Ref, compute.Ink2Ref, compute.Ink2AttnParams, int, int, int, int) error {
	f.calls = append(f.calls, "attend_f16")
	return nil
}
func (f *fakeVerifyDispatchBackend) Ink2PQKVPostFP8Dev(compute.Ink2Ref, compute.Ink2Ref, compute.Ink2Ref, compute.Ink2Ref, compute.Ink2Ref, compute.Ink2Ref, compute.Ink2Ref, compute.Ink2Ref, compute.Ink2Ref, compute.Ink2Ref, compute.Ink2Ref, compute.Ink2Ref, compute.Ink2AttnParams, int, int) error {
	f.calls = append(f.calls, "post_fp8")
	return nil
}
func (f *fakeVerifyDispatchBackend) Ink2PAttendFP8Dev(compute.Ink2Ref, compute.Ink2Ref, compute.Ink2Ref, compute.Ink2Ref, compute.Ink2Ref, compute.Ink2Ref, compute.Ink2Ref, compute.Ink2Ref, compute.Ink2Ref, compute.Ink2Ref, compute.Ink2AttnParams, int, int, int, int) error {
	f.calls = append(f.calls, "attend_fp8")
	return nil
}
func (f *fakeVerifyDispatchBackend) Ink2PQKVPostNVFP4Dev(compute.Ink2Ref, compute.Ink2Ref, compute.Ink2Ref, compute.Ink2Ref, compute.Ink2Ref, compute.Ink2Ref, compute.Ink2Ref, compute.Ink2Ref, compute.Ink2Ref, compute.Ink2Ref, compute.Ink2Ref, compute.Ink2Ref, compute.Ink2AttnParams, int, int) error {
	f.calls = append(f.calls, "post_nvfp4")
	return nil
}
func (f *fakeVerifyDispatchBackend) Ink2PAttendNVFP4Dev(compute.Ink2Ref, compute.Ink2Ref, compute.Ink2Ref, compute.Ink2Ref, compute.Ink2Ref, compute.Ink2Ref, compute.Ink2Ref, compute.Ink2Ref, compute.Ink2Ref, compute.Ink2Ref, compute.Ink2AttnParams, int, int, int, int) error {
	f.calls = append(f.calls, "attend_nvfp4")
	return nil
}

func TestInkling2SpecVerifyGlobalVsLocalDispatch(t *testing.T) {
	cases := []struct {
		name      string
		kvHalf    bool
		kvFP8     bool
		kvNVFP4   bool
		wantLocal []string
		wantGlob  []string
	}{
		{"f32", false, false, false, []string{"post_f32", "attend_f32"}, []string{"post_f32", "attend_f32"}},
		{"f16", true, false, false, []string{"post_f32", "attend_f32"}, []string{"post_f16", "attend_f16"}},
		{"fp8", false, true, false, []string{"post_f32", "attend_f32"}, []string{"post_fp8", "attend_fp8"}},
		{"nvfp4", false, false, true, []string{"post_f32", "attend_f32"}, []string{"post_nvfp4", "attend_nvfp4"}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			f := &fakeVerifyDispatchBackend{}
			s := &inkling2Resident{
				be:      f,
				dev:     f,
				kvHalf:  tc.kvHalf,
				kvFP8:   tc.kvFP8,
				kvNVFP4: tc.kvNVFP4,
				hp: model.Inkling2LoopHParams{
					Hidden: 8, Heads: 2, KVHeads: 1, HeadDim: 4, RelLength: 4, Layers: 2,
					ConvKernel: 2, Experts: 2, ExpertsUsed: 1, ExpertWidth: 4,
					AttnResBlock: 1, LocalLayer: []bool{true, false},
				},
				layers: []ink2Layer{
					{gateUp: ink2Experts{stack: compute.Tensor{Shape: []int{16, 8}}}, down: ink2Experts{stack: compute.Tensor{Shape: []int{16, 8}}}},
					{gateUp: ink2Experts{stack: compute.Tensor{Shape: []int{16, 8}}}, down: ink2Experts{stack: compute.Tensor{Shape: []int{16, 8}}}},
				},
				caps:    []int{16, 16},
				kc:      make([]ink2Ref, 2),
				vc:      make([]ink2Ref, 2),
				ks:      make([]ink2Ref, 2),
				vs:      make([]ink2Ref, 2),
				hist:    make([]ink2Ref, 2),
				convLog: make([]ink2Ref, 2),
			}
			p := &ink2Prefill{
				s: s, pd: f, vd: f, vhalf: f, vfp8: f, vnvfp4: f,
				T: 2, verify: true, grouped: true,
				scrFloats: 1 << 16, keepFloats: 1 << 16,
			}
			var blocks []ink2Ref
			p.layer(0, 0, ink2Ref{}, &blocks, nil, nil)
			if len(f.calls) != 2 || f.calls[0] != tc.wantLocal[0] || f.calls[1] != tc.wantLocal[1] {
				t.Fatalf("local slot calls = %v, want %v", f.calls, tc.wantLocal)
			}
			f.calls = nil
			p.scrOff = 0
			p.layer(1, 1, ink2Ref{}, &blocks, nil, nil)
			if len(f.calls) != 2 || f.calls[0] != tc.wantGlob[0] || f.calls[1] != tc.wantGlob[1] {
				t.Fatalf("global slot calls = %v, want %v", f.calls, tc.wantGlob)
			}
		})
	}
}

type fakeVerifyEmbedding struct{}

func (fakeVerifyEmbedding) Row(_ int, dst []float32) {
	clear(dst)
}

func TestInkling2SpecVerifyOverflowCheck(t *testing.T) {
	ovfTensor := compute.NewF32(compute.Default(), []int{1}, []float32{0})
	ovfRef := compute.Ink2At(ovfTensor, 0)

	for _, tc := range []struct {
		name    string
		kvNVFP4 bool
		kvFP8   bool
		wantErr error
	}{
		{"nvfp4", true, false, ErrInkling2NVFP4ScaleOverflow},
		{"fp8", false, true, ErrInkling2FP8ScaleOverflow},
	} {
		t.Run(tc.name, func(t *testing.T) {
			f := &fakeVerifyDispatchBackend{overflowRef: ovfRef, overflowFlag: 1}
			s := &inkling2Resident{
				be:          f,
				dev:         f,
				kvNVFP4:     tc.kvNVFP4,
				kvFP8:       tc.kvFP8,
				kvOverflow:  ovfRef,
				specMax:     2,
				ctx:         64,
				hp:          model.Inkling2LoopHParams{Hidden: 4},
				w:           &model.Inkling2LoopWeights{TokEmbd: fakeVerifyEmbedding{}},
				verifyWarm:  true,
				verifyReady: true, // takes Ink2GraphLaunchSlot (no-op) so verify() reaches Ink2ReadInts + checkKVOverflow
			}
			_, err := s.verify([]int{0})
			if err != tc.wantErr {
				t.Fatalf("verify overflow err = %v, want %v", err, tc.wantErr)
			}
		})
	}
}

func TestInkling2Spec128kNVFP4DSparkVRAMBudget(t *testing.T) {
	mini := inkling2MiniKVShape()
	mini.Hidden = 2304
	mini.ExpertsUsed = 8
	mini.ExpertWidth = 768

	const (
		ctx128k             = 131072
		chunk               = 128
		dsparkQ8WeightBytes = 278 << 20 // ~278 MiB Q8_0 DSpark draft weights
		outputVocabRows     = 151936
		shortlist           = 32
	)

	// 1. Bounded DSpark ring sizing prevents 128k OOM and k_ink2d_attend smem overflow.
	if got := inkling2DraftFRNNCap(ctx128k, chunk); got != 1024 {
		t.Fatalf("inkling2DraftFRNNCap(128k, 128) = %d, want 1024 (saves >1.1 GiB vs unbounded ctx)", got)
	}
	if cap, win := inkling2DraftLayerRing(ctx128k, chunk, 1024, true); cap != 1152 || win != 1024 {
		t.Fatalf("local draft ring(128k, 128, 1024) = (%d, %d), want (1152, 1024)", cap, win)
	}
	if cap, win := inkling2DraftLayerRing(ctx128k, chunk, 1024, false); cap != 4224 || win != 4096 {
		t.Fatalf("global draft ring(128k, 128, 1024) = (%d, %d), want (4224, 4096)", cap, win)
	}

	// 2. Total post-weight VRAM budget at 128k NVFP4 + 278 MiB Q8 DSpark draft fits
	// within RTX 5080 (15.92 GiB) ~2.45 GiB free after 24/24 all-resident weights.
	draftHP := model.Inkling2DSparkHParams{
		Hidden: 2304, Layers: 2, Heads: 16, KVHeads: 8, HeadDim: 128,
		Intermediate: 6144, Block: 8, MarkovRank: 256, SlidingWindow: 1024,
		VocabUnpadded: outputVocabRows, Taps: 4, LocalLayer: []bool{true, false},
	}
	targetNVFP4KV := inkling2ResidentKVBytesFor(mini, ctx128k, true, chunk, Inkling2KVNVFP4)
	targetOther := inkling2ResidentOtherBytes(mini, inkling2ArenaFloats, chunk, 0)
	draftState := inkling2DSparkStateBytes(draftHP, ctx128k, chunk, outputVocabRows, shortlist)
	draftTotal := dsparkQ8WeightBytes + draftState
	totalPostWeights := targetNVFP4KV + targetOther + draftTotal

	const rtx5080FreeAfterWeights = int64(245) * (1 << 30) / 100 // 2.45 GiB
	t.Logf("128k NVFP4 + DSpark budget: target_kv=%.3f GiB, target_other=%.3f GiB, dspark_weights=%.3f GiB, dspark_state=%.3f GiB, total_post_weights=%.3f GiB (free=%.2f GiB)",
		float64(targetNVFP4KV)/(1<<30), float64(targetOther)/(1<<30),
		float64(dsparkQ8WeightBytes)/(1<<30), float64(draftState)/(1<<30),
		float64(totalPostWeights)/(1<<30), float64(rtx5080FreeAfterWeights)/(1<<30))

	if totalPostWeights > rtx5080FreeAfterWeights {
		t.Fatalf("128k NVFP4 + DSpark post-weights %.3f GiB exceeds RTX 5080 2.45 GiB free after 24/24 weights",
			float64(totalPostWeights)/(1<<30))
	}
}
