package ggufload

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// The device keeps one decode exec per process, so a resident session that another
// session ran after must recapture rather than replay an exec bound to the other
// session's buffers.
func TestInkling2ResidentSessionsRecaptureAfterAnotherSessionRan(t *testing.T) {
	t.Cleanup(func() { inkling2GraphOwner.Store(nil) })
	inkling2GraphOwner.Store(nil)
	ready := func(s *inkling2Resident) {
		s.graph, s.graphReady, s.warm = true, true, true
		s.verifyWarm, s.verifyReady, s.keepReady = true, true, true
		s.tailWarm[1], s.tailReady[1], s.tailKeepReady[1] = true, true, true
		d := s.draft.(*ink2DeviceDraft)
		d.warm, d.ready = true, true
	}
	a := &inkling2ResidentSession{s: &inkling2Resident{draft: &ink2DeviceDraft{}}}
	b := &inkling2ResidentSession{s: &inkling2Resident{draft: &ink2DeviceDraft{}}}

	a.own()
	ready(a.s)
	a.own()
	if !a.s.graphReady || !a.s.warm {
		t.Fatal("the owning session dropped its own capture on its next call")
	}
	b.own()
	ready(b.s)
	a.own()
	if s := a.s; s.graphReady || s.warm || s.verifyWarm || s.verifyReady || s.keepReady || s.tailWarm[1] || s.tailReady[1] || s.tailKeepReady[1] {
		t.Fatalf("session resumed after another ran with captured state %+v", s)
	}
	if d := a.s.draft.(*ink2DeviceDraft); d.ready || d.warm {
		t.Fatal("the draft would launch the slot exec another session captured")
	}
	if !a.s.graph {
		t.Fatal("recapture must stay enabled, only the stale exec is dropped")
	}
	if !b.s.graphReady {
		t.Fatal("taking ownership changed the other session's flags")
	}
	b.own()
	if b.s.graphReady {
		t.Fatal("b replayed its exec after a captured over it")
	}

	// closing a session destroys every exec, so whoever runs next recaptures
	ready(b.s)
	if err := (&inkling2ResidentSession{s: &inkling2Resident{}}).Close(); err != nil {
		t.Fatal(err)
	}
	b.own()
	if b.s.graphReady {
		t.Fatal("session replayed an exec after another session's close reset the graphs")
	}
}

func TestInkling2ResidentSessionHeldBytesCountsOwnedAndCheckpointBuffers(t *testing.T) {
	f32 := func(n ...int) []compute.Tensor {
		var ts []compute.Tensor
		for _, v := range n {
			ts = append(ts, compute.Tensor{Dtype: compute.F32, Shape: []int{v}})
		}
		return ts
	}
	r := &inkling2ResidentSession{s: &inkling2Resident{owned: f32(10, 6)}}
	if got := r.HeldBytes(); got != 64 {
		t.Fatalf("owned only: %d bytes, want 64", got)
	}
	r.ck = &ink2Rewind{base: f32(3), grown: f32(2, 2)}
	if got := r.HeldBytes(); got != 64+28 {
		t.Fatalf("with checkpoint buffers: %d bytes, want %d", got, 64+28)
	}
}
