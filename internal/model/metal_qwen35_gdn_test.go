package model

import (
	"errors"
	"reflect"
	"testing"
)

type residentGDNBackend struct {
	*fakeQwen35GDNSequenceBackend
	seeded   map[Qwen35GDNAuxState]qwen35GDNLayerSnapshot
	failSeed bool
}

func newResidentGDNBackend() *residentGDNBackend {
	return &residentGDNBackend{fakeQwen35GDNSequenceBackend: newFakeQwen35GDNSequenceBackend(), seeded: make(map[Qwen35GDNAuxState]qwen35GDNLayerSnapshot)}
}

func (b *residentGDNBackend) SeedQwen35GDNAuxState(state Qwen35GDNAuxState, conv, recurrent []float32) error {
	if b.failSeed {
		return errors.New("injected seed failure")
	}
	b.seeded[state] = qwen35GDNLayerSnapshot{conv: append([]float32(nil), conv...), recurrent: append([]float32(nil), recurrent...)}
	return nil
}

func residentTestSession(t *testing.T, b Qwen35GDNPreprojectedSequenceBackend) *Session {
	t.Helper()
	s := NewSynthetic(qwen35HybridTestCfg()).NewSession()
	accepted, err := s.initQwen35GDNPreprojectedSequence(b)
	if err != nil || !accepted {
		t.Fatalf("init resident GDN: accepted=%v err=%v", accepted, err)
	}
	return s
}

func residentSnapshots(s *Session) []qwen35GDNLayerSnapshot {
	var out []qwen35GDNLayerSnapshot
	for layer, state := range s.qwen35HAL.sequenceLayers {
		if !state.valid() {
			continue
		}
		out = append(out, qwen35GDNLayerSnapshot{layer: layer, conv: []float32{float32(layer + 1)}, recurrent: []float32{float32(layer + 2)}})
	}
	return out
}

func TestQwen35MetalGDNResidentStateLifecycleAndPathIdentity(t *testing.T) {
	b := newResidentGDNBackend()
	s := residentTestSession(t, b)
	snapshots := residentSnapshots(s)
	used, err := s.promoteQwen35MetalGDNDecode(snapshots)
	if err != nil || !used {
		t.Fatalf("promote: used=%v err=%v", used, err)
	}
	if path, ok := s.Qwen35GDNDecodePath(); !ok || path != Qwen35MetalGDNDecodeForwardPath {
		t.Fatalf("path=(%q,%v), want fak-native %q", path, ok, Qwen35MetalGDNDecodeForwardPath)
	}
	for _, snap := range snapshots {
		got := b.seeded[s.qwen35HAL.sequenceLayers[snap.layer]]
		if !reflect.DeepEqual(got.conv, snap.conv) || !reflect.DeepEqual(got.recurrent, snap.recurrent) {
			t.Fatalf("layer %d seed=%v/%v, want %v/%v", snap.layer, got.conv, got.recurrent, snap.conv, snap.recurrent)
		}
	}
	states := append([]Qwen35GDNAuxState(nil), b.allocated...)
	s.Close()
	assertEachAuxStateFreedOnce(t, b.fakeQwen35GDNSequenceBackend, states)
	s.Close()
	assertEachAuxStateFreedOnce(t, b.fakeQwen35GDNSequenceBackend, states)
}

func TestQwen35MetalGDNUnsupportedSeedDoesNotSelectOrMutateHostState(t *testing.T) {
	b := newResidentGDNBackend()
	b.failSeed = true
	s := residentTestSession(t, b)
	defer s.Close()
	before := s.Cache.linear
	states := append([]Qwen35GDNAuxState(nil), b.allocated...)
	used, err := s.promoteQwen35MetalGDNDecode(residentSnapshots(s))
	if err == nil || used {
		t.Fatalf("seed failure: used=%v err=%v, want false error", used, err)
	}
	if s.Cache.linear != before {
		t.Fatal("seed failure mutated historical host state")
	}
	if _, ok := s.Qwen35GDNDecodePath(); ok {
		t.Fatal("seed failure selected resident path")
	}
	assertEachAuxStateFreedOnce(t, b.fakeQwen35GDNSequenceBackend, states)
}

func TestQwen35MetalGDNLegacyBackendNilCloseReleasesOwners(t *testing.T) {
	b := newResidentGDNBackend()
	s := residentTestSession(t, b)
	states := append([]Qwen35GDNAuxState(nil), b.allocated...)
	s.Backend = nil
	s.Close()
	assertEachAuxStateFreedOnce(t, b.fakeQwen35GDNSequenceBackend, states)
}

func TestQwen35MetalGDNResetAndOperationFailureReleaseOwners(t *testing.T) {
	t.Run("reset", func(t *testing.T) {
		b := newResidentGDNBackend()
		s := residentTestSession(t, b)
		states := append([]Qwen35GDNAuxState(nil), b.allocated...)
		if used, err := s.promoteQwen35MetalGDNDecode(residentSnapshots(s)); err != nil || !used {
			t.Fatalf("promote: used=%v err=%v", used, err)
		}
		s.ResetQwen35MetalGDNDecode()
		if _, ok := s.Qwen35GDNDecodePath(); ok {
			t.Fatal("reset retained native path identity")
		}
		assertEachAuxStateFreedOnce(t, b.fakeQwen35GDNSequenceBackend, states)
		s.Close()
		assertEachAuxStateFreedOnce(t, b.fakeQwen35GDNSequenceBackend, states)
	})
	t.Run("post-submit failure", func(t *testing.T) {
		b := newResidentGDNBackend()
		s := residentTestSession(t, b)
		states := append([]Qwen35GDNAuxState(nil), b.allocated...)
		if used, err := s.promoteQwen35MetalGDNDecode(residentSnapshots(s)); err != nil || !used {
			t.Fatalf("promote: used=%v err=%v", used, err)
		}
		b.failRun = true
		_, used, err := s.tryQwen35MetalGDNDecode(0, nil, nil, nil, nil, nil, nil, nil, nil, 0)
		if !used || err == nil {
			t.Fatalf("failure: used=%v err=%v", used, err)
		}
		if _, ok := s.Qwen35GDNDecodePath(); ok {
			t.Fatal("operation failure retained native path identity")
		}
		assertEachAuxStateFreedOnce(t, b.fakeQwen35GDNSequenceBackend, states)
	})
}
