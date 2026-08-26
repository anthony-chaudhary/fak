package model

import (
	"errors"
	"reflect"
	"testing"
)

type fakeQwen35GDNSequence struct {
	preflightErr error
	allocErr     error
	runErr       error
	preflights   int
	allocs       int
	runs         int
	frees        map[any]int
	seen         [][]Qwen35GDNSequenceState
}

func (f *fakeQwen35GDNSequence) PreflightQwen35GDNSequence(Qwen35GDNSequenceConfig) error {
	f.preflights++
	return f.preflightErr
}
func (f *fakeQwen35GDNSequence) NewQwen35GDNSequenceState(cfg Qwen35GDNSequenceConfig) ([]Qwen35GDNSequenceState, error) {
	f.allocs++
	if f.allocErr != nil {
		return nil, f.allocErr
	}
	states := make([]Qwen35GDNSequenceState, cfg.Layers)
	for i := range states {
		states[i] = Qwen35GDNSequenceState{Layer: i, Value: &struct{ n int }{n: f.allocs*100 + i}}
	}
	return states, nil
}
func (f *fakeQwen35GDNSequence) RunQwen35GDNPreprojectedSequence(states []Qwen35GDNSequenceState, _ Qwen35GDNPreprojectedSequence) error {
	f.runs++
	f.seen = append(f.seen, append([]Qwen35GDNSequenceState(nil), states...))
	return f.runErr
}
func (f *fakeQwen35GDNSequence) FreeQwen35GDNSequenceState(states []Qwen35GDNSequenceState) {
	if f.frees == nil {
		f.frees = make(map[any]int)
	}
	for _, state := range states {
		f.frees[state.Value]++
	}
}

type notQwen35GDNSequence struct{}

func TestQwen35GDNSequenceCapabilityAdmissionPrecedesAllocation(t *testing.T) {
	s := &Session{}
	if err := s.InitQwen35GDNSequence(notQwen35GDNSequence{}, Qwen35GDNSequenceConfig{Layers: 2, StateSize: 4}); !errors.Is(err, ErrQwen35GDNSequenceUnsupported) {
		t.Fatalf("absent capability error = %v", err)
	}
	cap := &fakeQwen35GDNSequence{preflightErr: errors.New("wrong path")}
	if err := s.InitQwen35GDNSequence(cap, Qwen35GDNSequenceConfig{Layers: 2, StateSize: 4}); !errors.Is(err, ErrQwen35GDNSequenceUnsupported) {
		t.Fatalf("refused capability error = %v", err)
	}
	if cap.preflights != 1 || cap.allocs != 0 || cap.runs != 0 || len(cap.frees) != 0 {
		t.Fatalf("mutation before admission: %+v", cap)
	}
}

func TestQwen35GDNSequenceSessionIsolationStableHandlesAndClose(t *testing.T) {
	cap := &fakeQwen35GDNSequence{}
	a, b := &Session{}, &Session{}
	cfg := Qwen35GDNSequenceConfig{Layers: 2, StateSize: 4}
	for _, s := range []*Session{a, b} {
		if s.Backend != nil {
			t.Fatal("backend-nil production path changed")
		}
		if err := s.InitQwen35GDNSequence(cap, cfg); err != nil {
			t.Fatal(err)
		}
		if err := s.RunQwen35GDNPreprojectedSequence(Qwen35GDNPreprojectedSequence{Tokens: 2}); err != nil {
			t.Fatal(err)
		}
	}
	if reflect.DeepEqual(cap.seen[0], cap.seen[1]) || cap.seen[0][0].Value == cap.seen[1][0].Value {
		t.Fatal("two sessions shared auxiliary state")
	}
	for i, states := range cap.seen {
		before := append([]Qwen35GDNSequenceState(nil), states...)
		if err := []*Session{a, b}[i].RunQwen35GDNPreprojectedSequence(Qwen35GDNPreprojectedSequence{Tokens: 1}); err != nil {
			t.Fatal(err)
		}
		if !reflect.DeepEqual(before, cap.seen[i+2]) {
			t.Fatalf("session %d state handles changed", i)
		}
	}
	a.Close()
	a.Close()
	b.Close()
	if len(cap.frees) != 4 {
		t.Fatalf("freed handles = %d, want 4", len(cap.frees))
	}
	for handle, count := range cap.frees {
		if count != 1 {
			t.Fatalf("handle %v freed %d times", handle, count)
		}
	}
}

func TestQwen35GDNSequenceResetAndFailureCleanupNoFallback(t *testing.T) {
	cfg := Qwen35GDNSequenceConfig{Layers: 2, StateSize: 4}
	resetCap := &fakeQwen35GDNSequence{}
	resetSession := &Session{}
	if err := resetSession.InitQwen35GDNSequence(resetCap, cfg); err != nil {
		t.Fatal(err)
	}
	resetSession.ResetQwen35GDNSequence()
	resetSession.ResetQwen35GDNSequence()
	resetSession.Close()
	for handle, count := range resetCap.frees {
		if count != 1 {
			t.Fatalf("reset handle %v freed %d times", handle, count)
		}
	}

	want := errors.New("submitted failure")
	failureCap := &fakeQwen35GDNSequence{runErr: want}
	failureSession := &Session{}
	if err := failureSession.InitQwen35GDNSequence(failureCap, cfg); err != nil {
		t.Fatal(err)
	}
	if err := failureSession.RunQwen35GDNPreprojectedSequence(Qwen35GDNPreprojectedSequence{Tokens: 2}); !errors.Is(err, want) {
		t.Fatalf("run error = %v", err)
	}
	if failureCap.runs != 1 {
		t.Fatalf("submit attempts = %d, want 1", failureCap.runs)
	}
	if err := failureSession.RunQwen35GDNPreprojectedSequence(Qwen35GDNPreprojectedSequence{Tokens: 2}); !errors.Is(err, ErrQwen35GDNSequenceUnsupported) {
		t.Fatalf("post-failure run = %v", err)
	}
	if failureCap.runs != 1 {
		t.Fatalf("post-submit fallback/retry attempts = %d", failureCap.runs)
	}
	failureSession.Close()
	for handle, count := range failureCap.frees {
		if count != 1 {
			t.Fatalf("failure handle %v freed %d times", handle, count)
		}
	}
}

func TestQwen35GDNSequenceLegacySessionUnchanged(t *testing.T) {
	s := &Session{}
	if s.Backend != nil || s.qwen35GDNSequence != nil {
		t.Fatalf("legacy session mutated: %+v", s)
	}
	s.ResetQwen35GDNSequence()
	s.Close()
	s.Close()
}
