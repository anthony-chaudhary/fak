package model

import (
	"errors"
	"math"
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
	cold, _, packedCold := newLane()
	owned(cold, func() {
		got, selected, err := cold.v41ColdSharedExpertChain(0)(input)
		if err != nil || !selected || !v41ColdSharedEqualWords(got, want) || packedCold.dataReads != 1 || packedCold.deferredFences != 1 {
			t.Fatalf("cold resident packed read changed: reads=%d fences=%d error=%v", packedCold.dataReads, packedCold.deferredFences, err)
		}
	})
	s, scratch, next := newLane()
	owned(s, func() {
		got, err := m.v41SharedExpertForStep(0, input, m.Cfg, scratch)
		if err != nil || !v41ColdSharedEqualWords(got, want) || !v41ColdSharedEqualMath(next.math, old.math) {
			t.Fatalf("owned resident shared scalar arithmetic differs: %v", err)
		}
		if old.dataReads != 8 || next.dataReads != 1 || next.activationUploads != 1 || next.deferredFences != 1 {
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
	t.Run("issued_checked_fault_survives_later_recording_and_cleanup", func(t *testing.T) {
		s, scratch, b := newLane()
		primary := errors.New("first shared checked publication")
		b.deferredFaultAt, b.deferredFault = 1, primary
		b.failMatMul, b.fault, b.cleanupPanic = 2, errors.New("later up projection"), true
		defer func() { b.cleanupPanic = false }()
		owned(s, func() {
			out, err := m.v41SharedExpertForStep(0, input, m.Cfg, scratch)
			var boundary *V41BF16BoundaryOperationError
			if out != nil || !errors.Is(err, primary) || !errors.As(err, &boundary) || boundary.Site != "ffn.shared_experts.w1.weight projection" || b.matmuls != 2 || b.deferredFences != 1 || !s.BackendSessionClosed() || len(b.deferredPending) != 0 {
				t.Fatalf("shared primary/order/retirement changed: matmuls=%d fences=%d error=%v", b.matmuls, b.deferredFences, err)
			}
		})
	})
	t.Run("winning_publication_site_is_restored", func(t *testing.T) {
		s, scratch, b := newLane()
		primary := errors.New("up publication verdict")
		b.deferredFaultAt, b.deferredFault = 2, primary
		b.failMatMul, b.fault = 3, &compute.BackendError{Backend: "test", Site: "down projection", Err: errors.New("later down recording")}
		owned(s, func() {
			out, err := m.v41SharedExpertForStep(0, input, m.Cfg, scratch)
			var boundary *V41BF16BoundaryOperationError
			if out != nil || !errors.Is(err, primary) || !errors.As(err, &boundary) || boundary.Site != "ffn.shared_experts.w3.weight projection" || b.matmuls != 3 || b.deferredPublishes != 3 || b.deferredFences != 1 || !s.BackendSessionClosed() || len(b.deferredPending) != 0 {
				t.Fatalf("earlier site's verdict replaced or recording replayed: %v", err)
			}
		})
	})
	t.Run("transport_failure_keeps_numeric_verdict_unknown", func(t *testing.T) {
		s, scratch, b := newLane()
		unobserved, transport := errors.New("unobserved gate numeric fault"), errors.New("packed read transport failure")
		b.deferredFaultAt, b.deferredFault, b.deferredReadErr = 1, unobserved, transport
		owned(s, func() {
			out, err := m.v41SharedExpertForStep(0, input, m.Cfg, scratch)
			if out != nil || !errors.Is(err, transport) || errors.Is(err, unobserved) || b.matmuls != 3 || b.deferredPublishes != 4 || b.deferredFences != 1 || !s.BackendSessionClosed() || len(b.deferredPending) != 0 {
				t.Fatalf("transport exposed unknown numeric verdict or replayed: %v", err)
			}
		})
	})
}

// Both controls use the same F32 add. The recorder checks that neither operand
// was freed and records operand order without counting inspection as readback.
func (b *v41ColdSharedRecorder) AddInPlace(acc, shared compute.Tensor) {
	if b.freed[acc.Buf()] != 0 || b.freed[shared.Buf()] != 0 {
		panic("shared add consumed a released tensor")
	}
	op := v41ColdSharedMath{kind: "shared-add", x: append([]float32(nil), b.Backend.Read(acc)...), second: append([]float32(nil), b.Backend.Read(shared)...)}
	if b.malformed == "shared-add-panic" {
		b.math = append(b.math, op)
		panic(b.fault)
	}
	b.Backend.AddInPlace(acc, shared)
	if b.malformed == "shared-add-nonfinite" {
		b.Backend.Read(acc)[0] = float32(math.Inf(1))
	}
	op.y = append([]float32(nil), b.Backend.Read(acc)...)
	b.math = append(b.math, op)
}

// fak-test:runtime fast est=500ms lane=default
func TestV41DecodeSharedAddResidentOwned(t *testing.T) {
	t.Setenv(v41DeferredBF16PublishEnv, "1")
	m, input := v41ColdSharedFixture(t, compute.F32)
	t.Cleanup(func() { _ = m.CloseWeights() })
	routed := []float32{.000123, -.031234, .12345, -.00123}
	inputBefore, routedBefore := append([]float32(nil), input...), append([]float32(nil), routed...)
	routeCfg := v41DefaultRouterConfig()
	newLane := func() (*Session, *v41ProjScratch, *v41ColdSharedRecorder) {
		b := newV41ColdSharedRecorder()
		s := v41ColdSharedSession(m, b)
		ops := &v41DeviceGatherOps{owner: s, bf16Publish: s.v41BF16PublishFunc()}
		s.v41Forward = &v41ForwardState{callbackOwner: s, gatherOps: ops}
		scratch := &v41ProjScratch{denseProjection: s.v41DenseProjectionFunc(), sharedActivation: s.v41SharedActivationFunc(), sharedAdd: s.v41SharedAddFunc(), gatherOps: ops}
		t.Cleanup(s.Close)
		return s, scratch, b
	}
	owned := func(s *Session, run func()) {
		scope := &backendExecutionScope{owner: &sync.Mutex{}, live: true}
		scope.owner.Lock()
		scope.bind(s)
		defer scope.finish()
		run()
	}
	control, oldScratch, old := newLane()
	var want []float32
	owned(control, func() {
		shared, err := m.v41SharedExpertForStep(0, input, m.Cfg, oldScratch)
		if err != nil {
			t.Fatal(err)
		}
		want, err = v41SharedExpertAddWith(0, routed, shared, routeCfg, oldScratch.sharedAdd)
		if err != nil || v41BF16Widened(want) {
			t.Fatalf("control must contain finite, unrounded F32 sum values: %v", err)
		}
		v41ColdSharedAssertReleased(t, old, control)
	})
	s, scratch, next := newLane()
	owned(s, func() {
		got, err := m.v41SharedExpertAddForStep(0, input, routed, m.Cfg, routeCfg, scratch)
		if err != nil || !v41ColdSharedEqualWords(got, want) || !v41ColdSharedEqualMath(next.math, old.math) {
			t.Fatalf("resident sum changed arithmetic, operand order or F32 bits: %v", err)
		}
		if old.activationUploads != 3 || next.activationUploads != 2 || old.dataReads != 2 || next.dataReads != 1 || old.deferredFences != 1 || next.deferredFences != 1 {
			t.Fatalf("uploads %d→%d, payload reads %d→%d, packed fences %d→%d", old.activationUploads, next.activationUploads, old.dataReads, next.dataReads, old.deferredFences, next.deferredFences)
		}
		if next.activationBytes != 8*len(input) || next.dataBytes != 4*len(input) || !reflect.DeepEqual(next.deferredSites, old.deferredSites) || len(next.deferredPending) != 0 {
			t.Fatal("sum transfer extents or ordered publication records changed")
		}
		if !v41ColdSharedEqualWords(input, inputBefore) || !v41ColdSharedEqualWords(routed, routedBefore) {
			t.Fatal("shared input or routed host row was mutated")
		}
		v41ColdSharedAssertReleased(t, next, s)
	})
	t.Run("unbound_scope_retains_callbacks", func(t *testing.T) {
		_, scratch, b := newLane()
		got, err := m.v41SharedExpertAddForStep(0, input, routed, m.Cfg, routeCfg, scratch)
		if err != nil || !v41ColdSharedEqualWords(got, want) || b.dataReads != 9 {
			t.Fatalf("unbound scope selected resident sum: reads=%d error=%v", b.dataReads, err)
		}
	})
	t.Run("issued_publication_precedes_add_and_cleanup_failure", func(t *testing.T) {
		s, scratch, b := newLane()
		primary := errors.New("earlier up numeric verdict")
		b.deferredFaultAt, b.deferredFault = 2, primary
		b.malformed, b.fault, b.cleanupPanic = "shared-add-panic", &compute.BackendError{Backend: "test", Site: "shared-add", Err: errors.New("later add failure")}, true
		defer func() { b.cleanupPanic = false }()
		owned(s, func() {
			got, err := m.v41SharedExpertAddForStep(0, input, routed, m.Cfg, routeCfg, scratch)
			var boundary *V41BF16BoundaryOperationError
			if got != nil || !errors.Is(err, primary) || !errors.As(err, &boundary) || boundary.Site != "ffn.shared_experts.w3.weight projection" || b.matmuls != 3 || b.deferredPublishes != 4 || b.deferredFences != 1 || len(b.deferredPending) != 0 || !s.BackendSessionClosed() {
				t.Fatalf("earlier numeric primary, site or cleanup order lost: %v", err)
			}
		})
	})
	t.Run("add_failure_reports_shared_add_site", func(t *testing.T) {
		s, scratch, b := newLane()
		primary := &compute.BackendError{Backend: "test", Site: "shared-add", Err: errors.New("add failed")}
		b.malformed, b.fault = "shared-add-panic", primary
		owned(s, func() {
			got, err := m.v41SharedExpertAddForStep(0, input, routed, m.Cfg, routeCfg, scratch)
			var operation *BackendForwardOperationError
			var boundary *V41BF16BoundaryOperationError
			var forward *V41ForwardError
			if got != nil || !errors.Is(err, primary) || !errors.As(err, &operation) || operation.Path != "v41-shared-add" || operation.Stage != "shared-add" || errors.As(err, &boundary) || !errors.As(err, &forward) || forward.Stage != v41StageMoE || b.deferredFences != 1 || len(b.deferredPending) != 0 || !s.BackendSessionClosed() {
				t.Fatalf("add failure site, drainage or retirement changed: %v", err)
			}
		})
	})
	t.Run("incoming_typed_add_failure_is_rethrown", func(t *testing.T) {
		s, scratch, b := newLane()
		primary := &BackendForwardOperationError{Backend: "test", Path: "incoming", Cause: errors.New("already typed add failure")}
		b.malformed, b.fault = "shared-add-panic", primary
		var caught any
		func() {
			defer func() { caught = recover() }()
			owned(s, func() {
				_, _ = m.v41SharedExpertAddForStep(0, input, routed, m.Cfg, routeCfg, scratch)
			})
		}()
		if caught != primary || s.halFailure != primary || b.deferredFences != 1 || len(b.deferredPending) != 0 || !s.BackendSessionClosed() {
			t.Fatalf("typed add failure was wrapped, returned or left owner live: %v", caught)
		}
	})
	t.Run("nonfinite_sum_is_rejected_after_publications", func(t *testing.T) {
		s, scratch, b := newLane()
		b.malformed = "shared-add-nonfinite"
		owned(s, func() {
			got, err := m.v41SharedExpertAddForStep(0, input, routed, m.Cfg, routeCfg, scratch)
			var operation *BackendForwardOperationError
			var route *v4RouteError
			if got != nil || !errors.As(err, &operation) || operation.Path != "v41-shared-add" || operation.Stage != "readback" || !errors.As(err, &route) || route.Field != "shared_add" || b.deferredFences != 1 || !s.BackendSessionClosed() {
				t.Fatalf("non-finite sum accepted or misattributed: %v", err)
			}
		})
	})
	t.Run("transport_failure_has_no_numeric_verdict", func(t *testing.T) {
		s, scratch, b := newLane()
		unknown, transport := errors.New("unobserved gate fault"), errors.New("sum read transport failure")
		b.deferredFaultAt, b.deferredFault, b.deferredReadErr = 1, unknown, transport
		owned(s, func() {
			got, err := m.v41SharedExpertAddForStep(0, input, routed, m.Cfg, routeCfg, scratch)
			var operation *BackendForwardOperationError
			if got != nil || !errors.Is(err, transport) || errors.Is(err, unknown) || !errors.As(err, &operation) || operation.Path != "v41-shared-add" || b.matmuls != 3 || b.deferredPublishes != 4 || b.deferredFences != 1 || len(b.deferredPending) != 0 || !s.BackendSessionClosed() {
				t.Fatalf("transport invented a verdict, replayed or retained records: %v", err)
			}
		})
	})
	t.Run("shared_only_and_cold_control", TestV41DecodeSharedResidentOwned)
}
