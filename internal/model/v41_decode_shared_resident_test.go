package model

import (
	"errors"
	"reflect"
	"sync"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// fak-test:runtime fast est=500ms lane=default
func TestV41DecodeSharedResidentOwned(t *testing.T) {
	t.Setenv(v41DeferredBF16PublishEnv, "1")
	m, input := v41ColdSharedFixture(t, compute.Q6_K)
	t.Cleanup(func() { _ = m.CloseWeights() })
	newLane := func() (*Session, *v41ProjScratch, *v41ColdSharedRecorder) {
		b := newV41ColdSharedRecorder()
		s := v41ColdSharedSession(m, b)
		ops := &v41DeviceGatherOps{owner: s, bf16Publish: s.v41BF16PublishFunc()}
		s.v41Forward = &v41ForwardState{callbackOwner: s, gatherOps: ops}
		scratch := &v41ProjScratch{denseProjection: s.v41DenseProjectionFunc(), sharedActivation: s.v41SharedActivationFunc(), gatherOps: ops}
		t.Cleanup(s.Close)
		return s, scratch, b
	}
	owned := func(s *Session, run func()) {
		// The CPU recorder has no advertised device domain. Bind it to an
		// explicitly held private scope, as backendExecutionScope permits.
		scope := &backendExecutionScope{owner: &sync.Mutex{}, live: true}
		scope.owner.Lock()
		scope.bind(s)
		defer scope.finish()
		run()
	}
	control, oldScratch, old := newLane()
	var want []float32
	owned(control, func() {
		var err error
		want, err = m.v41SharedExpertSwiGLUWithPublish(0, input, m.Cfg, oldScratch.denseProjection, oldScratch.sharedActivation, oldScratch.gatherOps.bf16PublishFunc())
		if err != nil {
			t.Fatal(err)
		}
	})
	s, scratch, next := newLane()
	owned(s, func() {
		got, err := m.v41SharedExpertForStep(0, input, m.Cfg, scratch)
		if err != nil || !v41ColdSharedEqualWords(got, want) || !v41ColdSharedEqualMath(next.math, old.math) {
			t.Fatalf("owned resident shared scalar arithmetic differs: %v", err)
		}
		if old.dataReads != 8 || next.dataReads != 1 || next.activationUploads != 1 || next.deferredFences != 4 {
			t.Fatalf("payload reads %d→%d, uploads=%d publication fences=%d", old.dataReads, next.dataReads, next.activationUploads, next.deferredFences)
		}
		if next.activationBytes != 4*len(input) || next.dataBytes != 4*len(input) || !reflect.DeepEqual(next.deferredSites, old.deferredSites) {
			t.Fatal("resident transfer extent or checked-publication order changed")
		}
		v41ColdSharedAssertReleased(t, next, s)
	})
	t.Run("unbound_scope_retains_original_callbacks", func(t *testing.T) {
		_, scratch, b := newLane()
		got, err := m.v41SharedExpertForStep(0, input, m.Cfg, scratch)
		if err != nil || !v41ColdSharedEqualWords(got, want) || b.dataReads != 8 || !v41ColdSharedEqualMath(b.math, old.math) {
			t.Fatalf("unbound decode selected resident work: reads=%d error=%v", b.dataReads, err)
		}
	})
	t.Run("checked_fault_precedes_later_projection_and_cleanup", func(t *testing.T) {
		s, scratch, b := newLane()
		primary := errors.New("first shared checked publication")
		b.deferredFaultAt, b.deferredFault = 1, primary
		b.failMatMul, b.fault, b.cleanupPanic = 2, errors.New("later up projection"), true
		defer func() { b.cleanupPanic = false }()
		owned(s, func() {
			out, err := m.v41SharedExpertForStep(0, input, m.Cfg, scratch)
			if out != nil || !errors.Is(err, primary) || b.matmuls != 1 || b.deferredFences != 1 || !s.BackendSessionClosed() || len(b.deferredPending) != 0 {
				t.Fatalf("shared primary/order/retirement changed: matmuls=%d fences=%d error=%v", b.matmuls, b.deferredFences, err)
			}
		})
	})
}
