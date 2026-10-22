package model

import (
	"errors"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// TestQwen35IndependentDecodeBatchSimulatedDeclinePreservesSerialFallback is a
// portable backend-decline witness. It does not claim execution on Metal.
// fak-test:runtime fast est=20ms lane=default
func TestQwen35IndependentDecodeBatchSimulatedDeclinePreservesSerialFallback(t *testing.T) {
	m := NewSynthetic(qwen35HybridTestCfg())
	batched := m.NewBatchSession(2)
	want := []*Session{m.NewSession(), m.NewSession()}
	ids := []int{11, 17}

	old := qwen35HybridQ4KBatchStep
	defer func() { qwen35HybridQ4KBatchStep = old }()
	declines := 0
	qwen35HybridQ4KBatchStep = func(*BatchSession, []int) ([][]float32, bool) {
		declines++
		return nil, false
	}
	fallbacks := 0
	batched.ObserveSerialStep = func(reason string, _ time.Duration) {
		if reason != "backend_declined" {
			t.Fatalf("serial reason = %q", reason)
		}
		fallbacks++
	}
	got := batched.StepBatch(ids)
	if declines != 1 || fallbacks != len(ids) || batched.LastStepSharedPanels() != 0 || batched.LastStepMACs() != 0 {
		t.Fatalf("declines=%d fallbacks=%d panels=%d macs=%d", declines, fallbacks, batched.LastStepSharedPanels(), batched.LastStepMACs())
	}
	for i := range got {
		assertFloat32BitsEqual(t, "declined lane "+itoa(i), want[i].Step(ids[i]), got[i])
		assertKVCacheBitsEqual(t, "declined lane "+itoa(i), want[i].Cache, batched.Seqs[i].Cache)
	}
}

// TestQwen35IndependentDecodeBatchSimulatedPostMutationPoisonsEveryLaneWithoutReplay
// exercises the portable model lifecycle contract. It does not prove that the
// Vulkan backend invokes this lifecycle after a native batch failure.
// fak-test:runtime fast est=20ms lane=default
func TestQwen35IndependentDecodeBatchSimulatedPostMutationPoisonsEveryLaneWithoutReplay(t *testing.T) {
	m := NewSynthetic(qwen35HybridTestCfg())
	backend := newFakeQwen35GDNSequenceBackend()
	bs := m.NewBatchSession(3)
	for lane, s := range bs.Seqs {
		if accepted, err := s.initQwen35GDNPreprojectedSequence(backend); !accepted || err != nil {
			t.Fatalf("lane %d attach = accepted %v err %v", lane, accepted, err)
		}
	}
	allocated := append([]Qwen35GDNAuxState(nil), backend.allocated...)
	postMutation := errors.New("injected post-mutation batch failure")
	old := qwen35HybridQ4KBatchStep
	defer func() { qwen35HybridQ4KBatchStep = old }()
	qwen35HybridQ4KBatchStep = func(bs *BatchSession, _ []int) ([][]float32, bool) {
		for _, s := range bs.Seqs {
			s.failQwen35GDNSequence(0, "batch post-mutation", postMutation)
		}
		panic(&compute.Qwen35IndependentDecodeBatchError{Stage: "injected-post-mutation", Mutated: true, Cause: postMutation})
	}
	fallbacks := 0
	bs.ObserveSerialStep = func(string, time.Duration) { fallbacks++ }
	var recovered any
	func() {
		defer func() { recovered = recover() }()
		bs.StepBatch([]int{23, 29, 31})
	}()
	err, ok := recovered.(error)
	if !ok || !compute.Qwen35IndependentDecodeBatchMutated(err) || !errors.Is(err, postMutation) {
		t.Fatalf("panic=%v, want mutated batch failure", recovered)
	}
	if fallbacks != 0 || bs.LastStepSharedPanels() != 0 || bs.LastStepMACs() != 0 {
		t.Fatalf("fallbacks=%d panels=%d macs=%d", fallbacks, bs.LastStepSharedPanels(), bs.LastStepMACs())
	}
	for lane, s := range bs.Seqs {
		beforeRuns := backend.runCalls
		_, accepted, retryErr := s.tryQwen35GDNPreprojectedSequence(Qwen35GDNPreprojectedSequenceRequest{Layer: 0, Tokens: 1})
		if !accepted || !errors.Is(retryErr, postMutation) || backend.runCalls != beforeRuns {
			t.Fatalf("lane %d retried/fell back: accepted=%v err=%v runs=%d->%d", lane, accepted, retryErr, beforeRuns, backend.runCalls)
		}
	}
	assertEachAuxStateFreedOnce(t, backend, allocated)
}
