package model

import (
	"context"
	"errors"
	"reflect"
	"sync"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

type v41LayerCohortBackend struct {
	*v41ScalarCohortWitness
	owner           sync.Mutex
	ordinary        *v41BF16PublishRecorder
	expertFences    []int
	outputsPerFence []int
	onCohort        func()
	cohortFailure   error
}

func (b *v41LayerCohortBackend) ForwardExecutionOwner() *sync.Mutex { return &b.owner }

// Ordinary publication requires the outer Session backend identity. Reuse its
// existing checked CPU recorder; expert tensors keep their borrowed weight's
// cpu-ref identity, as required by the scalar expert protocol witness.
func (b *v41LayerCohortBackend) UploadClass(x compute.Tensor, dt compute.Dtype, class compute.MemoryClass, site string) compute.Tensor {
	if site == "V4.1 BF16 publication input" {
		return b.ordinary.UploadClass(x, dt, class, site)
	}
	return b.Backend.Upload(x, dt)
}
func (b *v41LayerCohortBackend) V41BF16Publish(x compute.Tensor, site uint32) (compute.Tensor, error) {
	if x.Backend() == b {
		return b.ordinary.V41BF16Publish(x, site)
	}
	return b.v41ScalarCohortWitness.V41BF16Publish(x, site)
}
func (b *v41LayerCohortBackend) V41BF16PublishDeferred(x compute.Tensor, site uint32) (compute.Tensor, compute.V41DeferredStatus, error) {
	if x.Backend() == b {
		return b.ordinary.V41BF16PublishDeferred(x, site)
	}
	return b.v41ScalarCohortWitness.V41BF16PublishDeferred(x, site)
}

func (b *v41LayerCohortBackend) V41SharedHeadAvailable(dt compute.Dtype, out, in, rows int) bool {
	return dt == compute.F32 && out > 0 && in > 0 && rows >= 1 && rows <= 4
}
func (b *v41LayerCohortBackend) V41ReadDeferred(statuses []compute.V41DeferredStatus, rows []compute.Tensor) ([]error, [][]float32, error) {
	if len(statuses) > 0 && statuses[0].Record.Backend() == b {
		return b.ordinary.V41ReadDeferred(statuses, rows)
	}
	if len(statuses) >= 12 {
		b.expertFences = append(b.expertFences, len(statuses))
		b.outputsPerFence = append(b.outputsPerFence, len(rows))
		if len(statuses) > 12 {
			if b.onCohort != nil {
				f := b.onCohort
				b.onCohort = nil
				f()
			}
			if b.cohortFailure != nil {
				// A deferred read consumes the recorded statuses even on a
				// transport verdict; preserve that contract in this witness.
				_, _, _ = b.v41ScalarCohortWitness.V41ReadDeferred(statuses, rows)
				return nil, nil, b.cohortFailure
			}
		}
	}
	return b.v41ScalarCohortWitness.V41ReadDeferred(statuses, rows)
}
func (b *v41LayerCohortBackend) resetCohort() { b.expertFences = nil; b.outputsPerFence = nil }

// fak-test:runtime medium est=6s lane=default
func TestV41LayerCohortTransactions(t *testing.T) {
	t.Setenv(v41DeferredBF16PublishEnv, "1")
	t.Setenv(v41RoutedIDEnv, "0")
	t.Setenv(v41PrefillExpertPanelEnv, "0")
	// Real block-aligned Q2_K gate/up + Q3_K down arithmetic, not fabricated
	// cohort results. The host-resident provider is no hardware qualification.
	m := v41PinnedSlateFixture(t, false)
	defer m.CloseWeights()
	newFixture := func(t *testing.T) (*v41LayerCohortBackend, []V41BatchLane, []*Session) {
		t.Helper()
		b := &v41LayerCohortBackend{v41ScalarCohortWitness: &v41ScalarCohortWitness{v41RoutedLayerTestBackend: newV41RoutedLayerTestBackend(), enabled: true}}
		b.ordinary = newV41DeferredPublishRecorder()
		b.ordinary.self = b
		var lanes []V41BatchLane
		var refs []*Session
		for i, prefix := range [][]int{{1, 2}, {1, 2, 3}} {
			s, err := m.NewBackendSessionChecked(b)
			if err != nil {
				t.Fatal(err)
			}
			t.Cleanup(s.Close)
			r, err := m.NewBackendSessionChecked(b)
			if err != nil {
				t.Fatal(err)
			}
			t.Cleanup(r.Close)
			s.Prefill(prefix)
			r.Prefill(prefix)
			lanes = append(lanes, V41BatchLane{ID: uint64(91 - i*74), Session: s, Token: 4 + i})
			refs = append(refs, r)
		}
		b.resetCohort()
		return b, lanes, refs
	}
	t.Run("one real layer fence and serial continuation parity", func(t *testing.T) {
		b, lanes, refs := newFixture(t)
		want := make([][]float32, len(refs))
		for i, r := range refs {
			want[i] = append([]float32(nil), r.Step(lanes[i].Token)...)
		}
		if len(b.expertFences) != 2*m.Cfg.NumLayers {
			t.Fatalf("serial expert fences=%v", b.expertFences)
		}
		for i, n := range b.expertFences {
			if n != 12 || b.outputsPerFence[i] != 1 {
				t.Fatalf("serial layer fence rows=%v outputs=%v", b.expertFences, b.outputsPerFence)
			}
		}
		b.resetCohort()
		got, err := m.StepV41Batch(lanes)
		if err != nil || !got.Selected {
			t.Fatalf("batch selected=%v err=%v", got.Selected, err)
		}
		if len(b.expertFences) != m.Cfg.NumLayers {
			t.Fatalf("cohort expert fences=%v", b.expertFences)
		}
		for i, n := range b.expertFences {
			if n != 24 || b.outputsPerFence[i] != 2 {
				t.Fatalf("combined layer fence rows=%v outputs=%v", b.expertFences, b.outputsPerFence)
			}
		}
		for i, result := range got.Lanes {
			if result.ID != lanes[i].ID || !result.Committed || result.Err != nil || !reflect.DeepEqual(result.Logits, want[i]) {
				t.Fatalf("lane%d id=%d commit=%v err=%v logits match=%v", i, result.ID, result.Committed, result.Err, reflect.DeepEqual(result.Logits, want[i]))
			}
			if next, ref := lanes[i].Session.Step(6), refs[i].Step(6); !reflect.DeepEqual(next, ref) {
				t.Fatalf("lane%d continuation diverged", i)
			}
		}
	})
	t.Run("prefill-only panel selection preserves decode cohort", func(t *testing.T) {
		t.Setenv(v41PrefillExpertPanelEnv, "1")
		b, lanes, refs := newFixture(t)
		want := make([][]float32, len(refs))
		for i, r := range refs {
			want[i] = append([]float32(nil), r.Step(lanes[i].Token)...)
		}
		b.resetCohort()
		got, err := m.StepV41Batch(lanes)
		if err != nil || !got.Selected || len(b.expertFences) != m.Cfg.NumLayers {
			t.Fatalf("prefill-only flag disabled decode cohort: selected=%v fences=%v err=%v", got.Selected, b.expertFences, err)
		}
		for i, n := range b.expertFences {
			if n != 24 || b.outputsPerFence[i] != 2 {
				t.Fatalf("prefill-only layer fence rows=%v outputs=%v", b.expertFences, b.outputsPerFence)
			}
		}
		for i, result := range got.Lanes {
			if result.ID != lanes[i].ID || !result.Committed || result.Err != nil || !reflect.DeepEqual(result.Logits, want[i]) {
				t.Fatalf("prefill-only lane%d id=%d commit=%v err=%v", i, result.ID, result.Committed, result.Err)
			}
			if next, ref := lanes[i].Session.Step(6), refs[i].Step(6); !reflect.DeepEqual(next, ref) {
				t.Fatalf("prefill-only lane%d continuation diverged", i)
			}
		}
	})
	t.Run("explicit ID selection retains whole-lane route", func(t *testing.T) {
		b, lanes, refs := newFixture(t)
		t.Setenv(v41RoutedIDEnv, "1")
		want := make([][]float32, 2)
		for i, r := range refs {
			want[i] = append([]float32(nil), r.Step(lanes[i].Token)...)
		}
		b.resetCohort()
		got, err := m.StepV41Batch(lanes)
		if err != nil || !got.Selected || len(b.expertFences) != 2*m.Cfg.NumLayers {
			t.Fatalf("explicit route fences=%v err=%v", b.expertFences, err)
		}
		for i, r := range got.Lanes {
			if !r.Committed || r.Err != nil || !reflect.DeepEqual(r.Logits, want[i]) {
				t.Fatalf("fallback lane%d err=%v", i, r.Err)
			}
		}
	})
	t.Run("cancelled row rolls back while peer commits", func(t *testing.T) {
		b, lanes, refs := newFixture(t)
		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()
		lanes[0].Context = ctx
		before := captureV41ForwardSnapshot(lanes[0].Session.v41State())
		want := append([]float32(nil), refs[1].Step(lanes[1].Token)...)
		b.onCohort = cancel
		b.resetCohort()
		got, err := m.StepV41Batch(lanes)
		if err != nil || !errors.Is(got.Lanes[0].Err, context.Canceled) || got.Lanes[0].Committed || !got.Lanes[1].Committed || !reflect.DeepEqual(got.Lanes[1].Logits, want) {
			t.Fatalf("cancel result=%+v err=%v", got, err)
		}
		if !reflect.DeepEqual(before, captureV41ForwardSnapshot(lanes[0].Session.v41State())) {
			t.Fatal("cancelled lane retained prepared state")
		}
		if len(b.expertFences) == 0 || b.expertFences[0] != 24 {
			t.Fatal("cancellation never traversed selected expert cohort")
		}
	})
	t.Run("selected cohort failure retires and rolls back every lane", func(t *testing.T) {
		b, lanes, _ := newFixture(t)
		before := make([]*v41ForwardSnapshot, len(lanes))
		for i, lane := range lanes {
			before[i] = captureV41ForwardSnapshot(lane.Session.v41State())
		}
		primary := &compute.BackendError{Backend: "cpu-contract", Class: compute.VulkanClassExecutionFailed, Err: errors.New("selected cohort fault")}
		b.cohortFailure = primary
		got, err := m.StepV41Batch(lanes)
		if !got.Selected || !errors.Is(err, primary) {
			t.Fatalf("selected=%v err=%v", got.Selected, err)
		}
		for i, lane := range lanes {
			if got.Lanes[i].ID != lane.ID || got.Lanes[i].Committed || !lane.Session.halClosed || !reflect.DeepEqual(before[i], captureV41ForwardSnapshot(lane.Session.v41State())) {
				t.Fatalf("lane%d was not retired/rolled back", i)
			}
		}
		if len(b.expertFences) != 1 {
			t.Fatalf("selected operation replayed: fences=%v", b.expertFences)
		}
	})
}
