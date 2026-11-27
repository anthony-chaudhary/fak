package ggufload

import (
	"context"
	"slices"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

type fakeInk2GraphResetter struct {
	resets int
}

func (f *fakeInk2GraphResetter) Ink2GraphReset() { f.resets++ }

// fak-test:runtime fast est=1ms
func TestInkling2DeviceOwnerHandoffAndCohortReset(t *testing.T) {
	resetter := &fakeInk2GraphResetter{}
	dev := NewInkling2Device(resetter)

	ready := func(s *inkling2Resident) {
		s.graph, s.graphReady, s.warm = true, true, true
		s.verifyWarm, s.verifyReady, s.keepReady = true, true, true
		s.tailWarm[1], s.tailReady[1], s.tailKeepReady[1] = true, true, true
		d := s.draft.(*ink2DeviceDraft)
		d.warm, d.ready = true, true
	}

	a := &inkling2ResidentSession{s: &inkling2Resident{draft: &ink2DeviceDraft{}}, devOwner: dev}
	b := &inkling2ResidentSession{s: &inkling2Resident{draft: &ink2DeviceDraft{}}, devOwner: dev}

	drops := 0
	ownCount := func(r *inkling2ResidentSession) {
		wasReady := r.s.graphReady
		r.own()
		if wasReady && !r.s.graphReady {
			drops++
		}
	}

	// Initial claim for a, then repeat claim: zero drops.
	a.own()
	if dev.Owner() != a.s {
		t.Fatalf("owner = %v, want session a", dev.Owner())
	}
	ready(a.s)
	ownCount(a)
	if drops != 0 || !a.s.graphReady {
		t.Fatalf("same-owner claim dropped capture: drops=%d graphReady=%v", drops, a.s.graphReady)
	}

	// Record an exec under a, switch to b, then back to a: 1 drop for a.
	keyA := Inkling2GraphExecKey{Geometry: "scalar", Width: 1, Fingerprint: 0x11}
	dev.RecordExec(keyA, "exec-a")
	if dev.ExecCount() != 1 {
		t.Fatalf("ExecCount = %d, want 1", dev.ExecCount())
	}
	b.own()
	if dev.ExecCount() != 0 {
		t.Fatalf("owner switch must clear exec table, got %d", dev.ExecCount())
	}
	ready(b.s)
	ownCount(a)
	if drops != 1 || a.s.graphReady {
		t.Fatalf("switch back to a: drops=%d graphReady=%v, want 1/false", drops, a.s.graphReady)
	}
	ready(a.s)

	// Cohort claims ownership, records an exec, then releases arenas (Reset):
	// owner must be nil, exec table empty, resetter called, and a must drop next.
	bt := &Inkling2LoopCohorts{
		devOwner: dev,
		b:        &inkling2Batch{s: &inkling2Resident{}},
	}
	if !dev.Claim(bt) {
		t.Fatal("cohort claim from scalar a must report switched=true")
	}
	keyC := Inkling2GraphExecKey{Geometry: "cohort", Width: 4, Fingerprint: 0x44}
	dev.RecordExec(keyC, "exec-c4")
	if _, ok := dev.LookupExec(keyC); !ok {
		t.Fatal("cohort exec not found after RecordExec")
	}
	bt.ReleaseArenas()
	if dev.Owner() != nil {
		t.Fatalf("owner after cohort ReleaseArenas = %v, want nil", dev.Owner())
	}
	if dev.ExecCount() != 0 {
		t.Fatalf("ExecCount after cohort Reset = %d, want 0", dev.ExecCount())
	}
	if resetter.resets != 1 {
		t.Fatalf("Ink2GraphReset calls = %d, want 1", resetter.resets)
	}

	// Scalar a runs after cohort reset: must drop its stale capture (drops == 2).
	ownCount(a)
	if drops != 2 || a.s.graphReady || a.s.warm {
		t.Fatalf("a after cohort reset: drops=%d graphReady=%v warm=%v, want 2/false/false", drops, a.s.graphReady, a.s.warm)
	}
}

// fak-test:runtime fast est=1ms
func TestInkling2ResidentSessionPrefillArenaBytesSubsetOfHeldBytes(t *testing.T) {
	keep := compute.Tensor{Dtype: compute.F32, Shape: []int{16}}
	scr := compute.Tensor{Dtype: compute.F32, Shape: []int{32}}
	in := compute.Tensor{Dtype: compute.F32, Shape: []int{8}}
	kv := compute.Tensor{Dtype: compute.F32, Shape: []int{64}}

	r := &inkling2ResidentSession{
		s: &inkling2Resident{
			owned:       []compute.Tensor{kv, keep, scr, in},
			prefillCap:  64,
			prefillKeep: compute.Ink2At(keep, 0),
			prefillScr:  compute.Ink2At(scr, 0),
			prefillIn:   compute.Ink2At(in, 0),
		},
	}
	if got, want := r.PrefillArenaBytes(), int64((16+32+8)*4); got != want {
		t.Fatalf("PrefillArenaBytes = %d, want %d", got, want)
	}
	if got, want := r.HeldBytes(), int64((64+16+32+8)*4); got != want {
		t.Fatalf("HeldBytes = %d, want %d", got, want)
	}
}

type fakeInk2CohortWithCaps struct {
	*fakeInk2Cohort
}

func (fakeInk2CohortWithCaps) Caps() compute.Caps { return compute.Caps{} }
func (d fakeInk2CohortWithCaps) Ink2MatVec(w compute.Tensor, x, out compute.Ink2Ref) error {
	return d.Ink2PMatMul(w, x, out, 1)
}

// fak-test:runtime fast est=20ms
func TestInkling2BatchStreamScalarParityC2C4C8(t *testing.T) {
	t.Setenv("FAK_INKLING2_BATCH_ADMIT", "0")
	m := fakeInk2CohortModel()

	// Build 8 distinct prompts of varying lengths to stress C2, C4, C8 cohorts.
	prompts := make([][]int, 8)
	for i := range prompts {
		p := make([]int, 5+i)
		for j := range p {
			p[j] = (i+1)*3 + j
		}
		prompts[i] = p
	}
	const maxTok = 6

	// Compute scalar reference tokens per prompt via resident session decode.
	refTokens := make([][]int, len(prompts))
	for i, p := range prompts {
		d := fakeInk2CohortWithCaps{newFakeInk2Cohort()}
		res, err := newInkling2ResidentSized(m, d, 64, 1024)
		if err != nil {
			t.Fatalf("scalar resident %d: %v", i, err)
		}
		sess := &inkling2ResidentSession{s: res}
		var got []int
		turn := &Inkling2LoopTurn{
			MaxTokens: maxTok,
			Emit: func(tok int) bool {
				got = append(got, tok)
				return true
			},
		}
		if _, err := Inkling2LoopSessionDecode(context.Background(), sess, p, turn); err != nil {
			_ = sess.Close()
			t.Fatalf("scalar decode %d: %v", i, err)
		}
		_ = sess.Close()
		refTokens[i] = got
	}

	// Verify C2, C4, and C8 cohorts match scalar token-for-token under both
	// mixed cohort prefill and lockstep prompt feeding, and that cohort
	// ReleaseArenas clears the per-device graph owner token.
	for _, width := range []int{2, 4, 8} {
		for _, mode := range []struct {
			name   string
			lookup map[string]string
		}{
			{"mixed-prefill", map[string]string{"FAK_INKLING2_BATCH_ADMIT": "0", "FAK_INKLING2_COHORT_PREFILL": "1", "FAK_NATIVE_LONG_PREFILL_TOKEN_THRESHOLD": "512"}},
			{"lockstep", map[string]string{"FAK_INKLING2_BATCH_ADMIT": "0", "FAK_INKLING2_COHORT_PREFILL": "0"}},
		} {
			d := newFakeInk2Cohort()
			lookup := func(k string) (string, bool) {
				v, ok := mode.lookup[k]
				return v, ok
			}
			bt := NewInkling2LoopCohortsWithLookup(m, d, lookup)
			got := make([][]int, width)
			reqs := make([]Inkling2LoopBatchRequest, width)
			for i := 0; i < width; i++ {
				row := i
				reqs[i] = Inkling2LoopBatchRequest{
					Context:   context.Background(),
					Prompt:    prompts[i],
					MaxTokens: maxTok,
					Emit: func(tok int) bool {
						got[row] = append(got[row], tok)
						return true
					},
				}
			}
			if _, err := bt.Generate(reqs); err != nil {
				bt.Close()
				t.Fatalf("C%d %s Generate: %v", width, mode.name, err)
			}
			if bt.device().Owner() != bt {
				bt.Close()
				t.Fatalf("C%d %s: device owner = %v, want cohort", width, mode.name, bt.device().Owner())
			}
			for i := 0; i < width; i++ {
				if !slices.Equal(got[i], refTokens[i]) {
					bt.Close()
					t.Fatalf("C%d %s row %d: got %v, want scalar %v", width, mode.name, i, got[i], refTokens[i])
				}
			}
			if bt.ArenaBytes() <= 0 {
				bt.Close()
				t.Fatalf("C%d %s: ArenaBytes = %d, want > 0", width, mode.name, bt.ArenaBytes())
			}
			bt.ReleaseArenas()
			if bt.device().Owner() != nil {
				bt.Close()
				t.Fatalf("C%d %s: device owner after ReleaseArenas = %v, want nil", width, mode.name, bt.device().Owner())
			}
			if bt.ArenaBytes() != 0 {
				bt.Close()
				t.Fatalf("C%d %s: ArenaBytes after ReleaseArenas = %d, want 0", width, mode.name, bt.ArenaBytes())
			}
			bt.Close()
		}
	}
}

type fakeInk2CloseDevice struct {
	fakeInk2GraphCohort
	resets int
}

func (d *fakeInk2CloseDevice) Ink2GraphReset() { d.resets++ }

// fak-test:runtime fast est=1ms
func TestInkling2ResidentFailedClosePreservesLiveOwnerGraph(t *testing.T) {
	devBackend := &fakeInk2CloseDevice{fakeInk2GraphCohort: *newFakeInk2GraphCohort()}
	dev := inkling2DeviceFor(devBackend)
	dev.Reset()
	resetsBefore := devBackend.resets

	owner := &inkling2ResidentSession{s: &inkling2Resident{dev: devBackend, be: devBackend}, devOwner: dev}
	owner.own()
	owner.s.graph, owner.s.graphReady, owner.s.warm = true, true, true

	// A failed second-slot build has not claimed the device or captured graphs, so
	// closing it must not destroy the live owner's captured CUDA graph exec.
	failed := &inkling2Resident{dev: devBackend, be: devBackend}
	failed.close()
	if devBackend.resets != resetsBefore {
		t.Fatalf("failed unowned close called Ink2GraphReset %d time(s), want 0", devBackend.resets-resetsBefore)
	}
	if dev.Owner() != owner.s || !owner.s.graphReady {
		t.Fatalf("owner=%v graphReady=%v after failed close, want intact owner", dev.Owner(), owner.s.graphReady)
	}

	// Once a session claims and closes, Reset clears both the graph and owner atomically.
	second := &inkling2ResidentSession{s: &inkling2Resident{dev: devBackend, be: devBackend}, devOwner: dev}
	second.own()
	second.s.close()
	if devBackend.resets != resetsBefore+1 {
		t.Fatalf("claimed close reset count = %d, want %d", devBackend.resets-resetsBefore, 1)
	}
	if dev.Owner() != nil {
		t.Fatalf("dev.Owner() after claimed close = %v, want nil", dev.Owner())
	}
	if !owner.s.graphReady {
		t.Fatal("owner.s.graphReady should still be true before reclaim")
	}
	owner.own()
	if owner.s.graphReady || owner.s.warm {
		t.Fatalf("owner did not drop stale captures after reclaim: ready=%v warm=%v", owner.s.graphReady, owner.s.warm)
	}
}
