package model

import (
	"errors"
	"reflect"
	"slices"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// Ordinary publication requires the outer Session backend identity. Routed
// expert tensors retain their borrowed weights' cpu-ref identity. This is the
// same split used by v41EntryPointCohortWitness for a public Prefill call.
type v41RoutedOutputPrefillWitness struct {
	*v41RoutedOutputOwnerWitness
	ordinary *v41BF16PublishRecorder
}

func (b *v41RoutedOutputPrefillWitness) UploadClass(x compute.Tensor, dt compute.Dtype, class compute.MemoryClass, site string) compute.Tensor {
	if site == "V4.1 BF16 publication input" {
		return b.ordinary.UploadClass(x, dt, class, site)
	}
	return b.v41RoutedOutputOwnerWitness.UploadClass(x, dt, class, site)
}
func (b *v41RoutedOutputPrefillWitness) V41BF16Publish(x compute.Tensor, site uint32) (compute.Tensor, error) {
	if x.Backend() == b {
		return b.ordinary.V41BF16Publish(x, site)
	}
	return b.v41RoutedOutputOwnerWitness.V41BF16Publish(x, site)
}
func (b *v41RoutedOutputPrefillWitness) V41BF16PublishDeferred(x compute.Tensor, site uint32) (compute.Tensor, compute.V41DeferredStatus, error) {
	if x.Backend() == b {
		return b.ordinary.V41BF16PublishDeferred(x, site)
	}
	return b.v41RoutedOutputOwnerWitness.V41BF16PublishDeferred(x, site)
}
func (b *v41RoutedOutputPrefillWitness) V41ReadDeferred(statuses []compute.V41DeferredStatus, rows []compute.Tensor) ([]error, [][]float32, error) {
	if len(statuses) > 0 && statuses[0].Record.Backend() == b {
		return b.ordinary.V41ReadDeferred(statuses, rows)
	}
	return b.v41RoutedOutputOwnerWitness.V41ReadDeferred(statuses, rows)
}

// Exercise the public cold Prefill transaction, with the same checked cohorts
// in both arms. Only routed-sum authority differs. The recorder's storage is
// poisoned on release and cannot be borrowed after Free.
// fak-test:runtime medium est=10s lane=default
func TestV41RoutedOutputPrefill(t *testing.T) {
	t.Setenv(v41DeferredBF16PublishEnv, "1")
	t.Setenv(v41RoutedIDEnv, "0")
	t.Setenv(v41PrefillExpertPanelEnv, "0")
	m := v41PinnedSlateFixture(t, false)
	newSession := func(replace bool) (*Session, *v41RoutedOutputPrefillWitness) {
		b := &v41RoutedOutputPrefillWitness{v41RoutedOutputOwnerWitness: &v41RoutedOutputOwnerWitness{
			v41ScalarCohortWitness: &v41ScalarCohortWitness{v41RoutedLayerTestBackend: newV41RoutedLayerTestBackend(), enabled: true},
			freed:                  make(map[compute.Buffer]bool),
		}}
		b.ordinary = newV41DeferredPublishRecorder()
		b.ordinary.self = b
		s, err := m.NewBackendSessionChecked(b)
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(s.Close)
		if replace {
			st := s.v41State()
			st.setRoutedSum(st.routedSumFunc())
		}
		return s, b
	}
	t.Run("ordinary_replacement_and_canonical_parity", func(t *testing.T) {
		control, before := newSession(true)
		candidate, after := newSession(false)
		prompt := []int{1, 2, 3}
		want, got := control.Prefill(prompt), candidate.Prefill(prompt)
		if !v41SameBits(got, want) || after.borrowedAdds == 0 || before.borrowedAdds != 0 ||
			before.sumUploads-after.sumUploads != after.borrowedAdds || before.fences != after.fences || before.ordinary.deferredFences != after.ordinary.deferredFences ||
			!reflect.DeepEqual(before.matmulInputs, after.matmulInputs) || !reflect.DeepEqual(before.cohortRows, after.cohortRows) {
			t.Fatalf("cold route differs: uploads=%d/%d borrowed=%d/%d fences=%d/%d", before.sumUploads, after.sumUploads, before.borrowedAdds, after.borrowedAdds, before.fences, after.fences)
		}
		if before.freeRows != len(before.routed) || after.freeRows != len(after.routed) || len(after.pending) != 0 || len(before.ordinary.deferredPending) != 0 || len(after.ordinary.deferredPending) != 0 || candidate.BackendSessionClosed() {
			t.Fatal("successful cold forward retained a publication or failed its session")
		}
	})
	t.Run("cleanup_failure_rolls_back_cold_transaction", func(t *testing.T) {
		s, b := newSession(false)
		state := s.v41State()
		priorHistory := append([]int(nil), state.history...)
		priorLayers := append([]*V41AttentionState(nil), state.layers...)
		b.failRetainedCleanup = true
		var logits []float32
		err := recoverError(func() { logits = s.Prefill([]int{1, 2, 3}) })
		var fatal *BackendForwardOperationError
		if !errors.As(err, &fatal) {
			t.Errorf("cold cleanup fault error type: got %T (%v), want BackendForwardOperationError", err, err)
		}
		if logits != nil {
			t.Errorf("cold cleanup fault published logits: len=%d", len(logits))
		}
		if b.borrowedAdds == 0 {
			t.Error("cold cleanup fault never exercised retained routed rows")
		}
		if !s.BackendSessionClosed() {
			t.Error("cold cleanup fault left the backend session open")
		}
		// Rollback restores the logical history; a zero-length retained buffer
		// and a nil pre-call slice represent the same empty history.
		if !slices.Equal(state.history, priorHistory) {
			t.Errorf("cold cleanup fault changed history: got=%v want=%v", state.history, priorHistory)
		}
		if !reflect.DeepEqual(state.layers, priorLayers) {
			t.Errorf("cold cleanup fault changed layer state: got=%#v want=%#v", state.layers, priorLayers)
		}
		if b.freeRows != len(b.routed) {
			t.Errorf("cold cleanup fault leaked routed rows: released=%d owned=%d", b.freeRows, len(b.routed))
		}
		if len(b.pending) != 0 {
			t.Errorf("cold cleanup fault retained routed statuses: pending=%d", len(b.pending))
		}
		if len(b.ordinary.deferredPending) != 0 {
			t.Errorf("cold cleanup fault retained ordinary statuses: pending=%d", len(b.ordinary.deferredPending))
		}
	})
}
