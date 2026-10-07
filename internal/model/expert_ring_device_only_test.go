package model

import (
	"errors"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/polymodel"
)

// SW-VERIFIED only: recording CPU backend, no hardware qualification.
func deviceOnlyExpertRingPanic(fn func()) (value any) {
	defer func() { value = recover() }()
	fn()
	return nil
}

// fak-test:runtime fast est=1s
func TestDeviceOnlyExpertRingSharedRefusalReleasesOnlyFailedMember(t *testing.T) {
	t.Parallel()
	m := expertRingTestModel(t, 256, 1)
	bytes := expertRingWeightBytes(t, m)
	be := &sharedRingBackend{Backend: compute.Default()}
	sh, err := NewSharedExpertRing(SharedExpertRingConfig{Model: m, Backend: be, BudgetBytes: bytes})
	if err != nil {
		t.Fatal(err)
	}
	peer, failed := sharedRingAgent(m, be), sharedRingAgent(m, be)
	for name, session := range map[string]*Session{"peer": peer, "failed": failed} {
		if err := sh.Attach(session, name); err != nil {
			t.Fatal(err)
		}
	}
	gate := expertName(0, 0, "gate_proj.weight")
	up := expertName(0, 0, "up_proj.weight")
	peer.weightHALQ4K(gate, m.q4kw[gate])
	done := peer.ringEnter(peer.expertRing)
	peer.expertRing.hold("q4k:" + gate)
	done()
	failed.SetExecutionPolicy(ExecutionPolicyDeviceOnly)
	beforeFrees, beforeUploads := be.frees, be.uploads
	type outcome struct {
		payload      any
		outer, inner func()
	}
	refused := make(chan outcome, 1)
	go func() {
		outer := failed.ringEnter(failed.expertRing)
		inner := failed.ringEnter(failed.expertRing)
		payload := deviceOnlyExpertRingPanic(func() { failed.weightHALQ4K(up, m.q4kw[up]) })
		if payload == nil {
			inner()
			outer()
		}
		refused <- outcome{payload, outer, inner}
	}()
	var result outcome
	select {
	case result = <-refused:
	case <-time.After(2 * time.Second):
		t.Fatal("shared refusal did not return from nested member spans")
	}
	if result.payload == nil {
		failed.Close()
		peer.Close()
		sh.Close()
		t.Fatal("shared strict capacity refusal did not raise an operation error")
	}
	operation, ok := result.payload.(error)
	var typed *BackendForwardOperationError
	if !ok || !errors.As(operation, &typed) || !errors.Is(operation, polymodel.ErrPinnedNoRoom) {
		t.Fatalf("shared refusal payload = %T, want operation preserving pinned-capacity cause", result.payload)
	}
	if !failed.BackendSessionClosed() {
		t.Fatal("shared refusal left failed member open")
	}
	if be.frees != beforeFrees+1 || be.uploads != beforeUploads+1 {
		t.Fatal("shared refusal freed a retained peer handle or staged a permanent escape")
	}
	spanReady := make(chan func(), 1)
	go func() { spanReady <- peer.ringEnter(peer.expertRing) }()
	var peerSpan func()
	select {
	case peerSpan = <-spanReady:
	case <-time.After(2 * time.Second):
		t.Fatal("shared refusal stranded the surviving peer's staging span")
	}
	result.inner()
	result.outer()
	stats := make(chan SharedExpertRingStats, 1)
	started := make(chan struct{})
	go func() {
		close(started)
		stats <- sh.Stats()
	}()
	<-started
	select {
	case <-stats:
		peerSpan()
		t.Fatal("stale failed-member callbacks released an active peer span")
	case <-time.After(500 * time.Millisecond):
	}
	peer.weightHALQ4K(gate, m.q4kw[gate])
	peer.expertRing.release("q4k:" + gate)
	peerSpan()
	select {
	case receipt := <-stats:
		if receipt.Agents != 1 || receipt.Ring.ResidentBytes != bytes {
			t.Fatal("shared refusal lost peer membership or retained residency")
		}
	case <-time.After(2 * time.Second):
		t.Fatal("peer span release did not restore shared stats access")
	}
	if be.uploads != beforeUploads+1 || be.frees != beforeFrees+1 {
		t.Fatal("peer could not reuse its surviving handle after refusal")
	}
	failed.Close()
	peer.Close()
	if err := sh.Close(); err != nil {
		t.Fatal(err)
	}
}

// fak-test:runtime fast est=100ms
func TestDeviceOnlyExpertRingRefusesCapacityEscape(t *testing.T) {
	m := expertRingTestModel(t, 256, 1)
	bytes := expertRingWeightBytes(t, m)
	name := expertName(0, 0, "gate_proj.weight")
	for _, tc := range []struct {
		name      string
		budget    int64
		permanent bool
		forward   bool
		cause     error
	}{
		{name: "oversized", budget: bytes - 1, cause: polymodel.ErrTooLarge},
		{name: "held projections", budget: bytes * 2, forward: true, cause: polymodel.ErrPinnedNoRoom},
		{name: "existing permanent weight", budget: bytes - 1, permanent: true, cause: polymodel.ErrTooLarge},
	} {
		t.Run(tc.name, func(t *testing.T) {
			s := expertRingSession(m, 0)
			t.Cleanup(s.Close)
			if tc.permanent {
				s.weightHALQ4K(name, m.q4kw[name])
			}
			s.ExpertRingBytes = tc.budget
			s.SetExecutionPolicy(ExecutionPolicyDeviceOnly)
			payload := deviceOnlyExpertRingPanic(func() {
				if tc.forward {
					expertSwiGLU(m, 0, 0, expertRingTestInput(256), sessionQ4KKernel{s: s})
				} else {
					s.weightHALQ4K(name, m.q4kw[name])
				}
			})
			err, ok := payload.(error)
			var operation *BackendForwardOperationError
			if !ok || !errors.As(err, &operation) {
				t.Fatalf("capacity refusal payload = %T, want *BackendForwardOperationError", payload)
			}
			if !errors.Is(err, tc.cause) {
				t.Fatalf("capacity refusal lost sentinel %v", tc.cause)
			}
			if !s.BackendSessionClosed() {
				t.Fatal("capacity refusal left strict session open")
			}
			for key := range s.halW {
				if isRoutedExpertWeight(key) {
					t.Fatal("capacity refusal retained a permanent routed weight")
				}
			}
		})
	}
}

// fak-test:runtime fast est=100ms
func TestDeviceOnlyExpertRingPolicyCompatibility(t *testing.T) {
	m := expertRingTestModel(t, 256, 1)
	bytes := expertRingWeightBytes(t, m)
	name := expertName(0, 0, "gate_proj.weight")
	for _, tc := range []struct {
		name   string
		policy ExecutionPolicy
		budget int64
	}{
		{"default oversized", ExecutionPolicyPortable, bytes - 1},
		{"default held projections", ExecutionPolicyPortable, bytes * 2},
		{"strict full resident", ExecutionPolicyDeviceOnly, 0},
	} {
		t.Run(tc.name, func(t *testing.T) {
			s := expertRingSession(m, tc.budget)
			t.Cleanup(s.Close)
			s.SetExecutionPolicy(tc.policy)
			expertSwiGLU(m, 0, 0, expertRingTestInput(256), sessionQ4KKernel{s: s})
			key := "q4k:" + name
			if tc.budget == bytes*2 {
				key = "q4k:" + expertName(0, 0, "down_proj.weight")
			}
			if _, ok := s.halW[key]; !ok {
				t.Fatal("compatible policy lost permanent residency fallback")
			}
			if tc.budget == 0 && s.expertRing != nil {
				t.Fatal("zero budget created a ring")
			}
		})
	}
}

// fak-test:runtime fast est=100ms
func TestDeviceOnlyExpertRingSuccessfulStagingReusesBoundedWeights(t *testing.T) {
	m := expertRingTestModel(t, 256, 2)
	s := expertRingSession(m, 3*expertRingWeightBytes(t, m))
	t.Cleanup(s.Close)
	s.SetExecutionPolicy(ExecutionPolicyDeviceOnly)
	be := s.Backend.(*expertHALRecordingBackend)
	for _, expert := range []int{0, 0, 1, 1, 0} {
		expertSwiGLU(m, 0, expert, expertRingTestInput(256), sessionQ4KKernel{s: s})
		st := s.ExpertRing()
		if st.ResidentBytes > st.BudgetBytes || st.PeakBytes > st.BudgetBytes {
			t.Fatal("strict staging exceeded declared resident byte budget")
		}
		for key := range s.halW {
			if isRoutedExpertWeight(key) {
				t.Fatal("strict staged expert escaped into permanent residency")
			}
		}
	}
	st := s.ExpertRing()
	if st.Hits != 6 || st.PageIns != 9 || st.Evictions != 6 || be.uploads[compute.Q4_K] != 9 {
		t.Fatalf("reuse ledger: hits=%d pageins=%d evictions=%d uploads=%d", st.Hits, st.PageIns, st.Evictions, be.uploads[compute.Q4_K])
	}
}
