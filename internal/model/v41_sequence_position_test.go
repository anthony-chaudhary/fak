package model

import (
	"errors"
	"reflect"
	"testing"
)

// fak-test:justify why=regression when=changed:internal/model/qwen35_backend_contract.go
// fak-test:runtime short est=2s lane=default
func TestV41SequencePositionCommittedHistory(t *testing.T) {
	t.Setenv("FAK_PAGED_KV", "0")
	m := v41DecodeStateModel(t, 2)
	t.Cleanup(func() { _ = m.CloseWeights() })
	m.Cfg.Window = []int{2, 2}
	s := m.NewSession()
	t.Cleanup(s.Close)
	check := func(want int) {
		t.Helper()
		if got := s.SequencePosition(); got != want {
			t.Fatalf("position=%d want=%d committed=%v", got, want, s.v41Forward)
		}
		if s.v41Forward != nil && s.SequencePosition() != len(s.v41Forward.history) {
			t.Fatal("position differs from committed history")
		}
	}
	check(0)
	empty, err := s.PrefixSnapshot()
	if err != nil {
		t.Fatal(err)
	}
	if empty.v41 == nil || empty.v41.hadState || empty.Tokens != 0 || empty.v41ResidentBytes() != 0 {
		t.Fatal("empty V4.1 capture lost absent-state contract or charged payload")
	}
	emptyClone, err := empty.Clone()
	if err != nil {
		t.Fatal(err)
	}
	empty.Close()
	empty = emptyClone
	t.Cleanup(empty.Close)
	if empty.v41 == nil || empty.v41.hadState || empty.Tokens != 0 {
		t.Fatal("empty clone lost independent absent-state ownership")
	}
	s.Prefill([]int{1, 3})
	check(2)
	s.Prefill([]int{5, 7})
	check(4)
	saved, err := s.PrefixSnapshot()
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(saved.Close)
	if saved.Tokens != s.SequencePosition() {
		t.Fatal("snapshot position differs")
	}
	s.Step(2)
	check(5)
	before := append([]int(nil), s.v41Forward.history...)
	func() {
		completed := false
		defer func() {
			if recover() == nil && completed {
				t.Fatal("invalid token accepted")
			}
		}()
		s.Step(m.Cfg.VocabSize)
		completed = true
	}()
	check(5)
	if !reflect.DeepEqual(before, s.v41Forward.history) {
		t.Fatal("failed token changed committed history")
	}
	// The full-prefill transaction appends provisional history before selected
	// embedding work. Observe that work, refuse it, and require rollback.
	fault := errors.New("selected embedding failure after history append")
	calls := 0
	priorGather := s.v41Forward.gatherOps
	s.v41Forward.gatherOps = &v41DeviceGatherOps{owner: s, embedGather: func(ids []int) ([]float32, error) {
		calls++
		if len(s.v41Forward.history) != len(before)+1 || len(ids) != len(before)+1 {
			t.Fatal("selected failure did not reach provisional transaction history")
		}
		return nil, fault
	}}
	_, forwardErr := m.forwardV41([]int{4}, s.v41Forward)
	s.v41Forward.gatherOps = priorGather
	if !errors.Is(forwardErr, fault) || calls != 1 {
		t.Fatalf("selected failure replayed or lost cause: calls=%d err=%v", calls, forwardErr)
	}
	check(5)
	if !reflect.DeepEqual(before, s.v41Forward.history) {
		t.Fatal("selected failure did not restore committed history")
	}
	// Rebinding an already-diverged committed branch is deliberately refused.
	if err := saved.Restore(s); err == nil {
		t.Fatal("diverged restore accepted")
	}
	check(5)
	restored := m.NewSession()
	t.Cleanup(restored.Close)
	if err := saved.Restore(restored); err != nil {
		t.Fatal(err)
	}
	s = restored
	check(4)
	if err := empty.Restore(s); err != nil {
		t.Fatal(err)
	}
	check(0)
	if s.v41Forward != nil || empty.v41 != nil {
		t.Fatal("empty restore failed to clear state or transfer ownership")
	}
	if err := empty.Restore(s); err == nil {
		t.Fatal("consumed empty snapshot restored twice")
	}
	s.Prefill([]int{1})
	check(1)
}

// fak-test:justify why=regression when=changed:internal/model/qwen35_backend_contract.go
// fak-test:runtime short est=1s lane=default
func TestV41SequencePositionNilAndDenseUnchanged(t *testing.T) {
	var absent *Session
	if absent.SequencePosition() != 0 || (&Session{}).SequencePosition() != 0 {
		t.Fatal("empty position changed")
	}
	m := NewSynthetic(Config{ModelType: "llama", HiddenSize: 8, NumLayers: 1, NumHeads: 1, NumKVHeads: 1, HeadDim: 8, IntermediateSize: 16, VocabSize: 16, RMSNormEps: 1e-5, RopeTheta: 10000})
	t.Cleanup(func() { _ = m.CloseWeights() })
	s := m.NewSession()
	t.Cleanup(s.Close)
	denseEmpty, err := s.PrefixSnapshot()
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(denseEmpty.Close)
	if denseEmpty.v41 != nil {
		t.Fatal("non-V4.1 snapshot gained a V4.1 marker")
	}
	s.Prefill([]int{1, 2})
	if s.SequencePosition() != s.Cache.Len() || s.SequencePosition() != 2 {
		t.Fatal("dense cache position changed")
	}
	s.Step(3)
	if s.SequencePosition() != 3 {
		t.Fatal("dense step position changed")
	}
}
