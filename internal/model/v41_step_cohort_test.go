package model

import (
	"context"
	"errors"
	"reflect"
	"sync"
	"testing"
)

type v41CohortTestBackend struct {
	*v41PrefillStepBatchBackend
	endPanic any
}

func (b *v41CohortTestBackend) EndV41StepBatch() error {
	err := b.v41PrefillStepBatchBackend.EndV41StepBatch()
	if b.endPanic != nil {
		panic(b.endPanic)
	}
	return err
}

// This bounded software witness covers scheduling and isolation, not shared
// matrix rows or device throughput. The runtime estimate is unmeasured.
// fak-test:runtime fast est=1s lane=default
func TestV41StepCohortIsolation(t *testing.T) {
	t.Setenv(v41StepBatchEnv, "1")
	m := v41CompressorTestFixtureIndex(t, 128)
	type fixture struct {
		backend *v41CohortTestBackend
		lanes   []v41StepCohortLane
		refs    []*Session
	}
	prepare := func(t *testing.T) fixture {
		t.Helper()
		f := fixture{backend: &v41CohortTestBackend{v41PrefillStepBatchBackend: newV41PrefillStepBatchBackend()}}
		for lane, prefix := range [][]int{{1, 2, 3}, {1, 2, 3, 4}} {
			source := m.NewSession()
			t.Cleanup(source.Close)
			if len(source.Prefill(prefix)) == 0 {
				t.Fatal("prefix produced no logits")
			}
			snapshot := captureV41ForwardSnapshot(source.v41State())
			source.v41State().denseProjection = func(int, string, []float32, int, int, int) ([]float32, v41DenseProjectionOutcome, error) {
				t.Fatal("cohort called a prefix source callback")
				return nil, v41ProjectionDeclined, nil
			}
			s := v41DenseTestSession(t, m, f.backend)
			ref := v41DenseTestSession(t, m, f.backend)
			s.v41Forward, ref.v41Forward = snapshot.clone().restore(), snapshot.clone().restore()
			s.v41State()
			ref.v41State()
			// Seed the next ring slot with real bytes to make rollback of an
			// overwritten slot observable without a long synthetic prefix.
			for _, st := range []*v41ForwardState{s.v41Forward, ref.v41Forward} {
				for _, state := range st.layers {
					slot := state.nextWindowPos % state.windowSize
					state.window[slot] = append([]float32(nil), state.window[0]...)
				}
			}
			f.lanes = append(f.lanes, v41StepCohortLane{session: s, id: 6 + lane})
			f.refs = append(f.refs, ref)
		}
		f.backend.reset()
		return f
	}

	t.Run("layer major parity and commit boundary", func(t *testing.T) {
		f := prepare(t)
		var owner sync.Mutex
		var order [][2]int
		want := make([][]float32, len(f.lanes))
		for lane, input := range f.lanes {
			var err error
			want[lane], _, err = m.forwardV41Step(input.id, f.refs[lane].v41State(), nil)
			if err != nil {
				t.Fatal(err)
			}
			project := input.session.v41State().denseProjection
			input.session.v41State().denseProjection = func(layer int, leaf string, panel []float32, out, in, rows int) ([]float32, v41DenseProjectionOutcome, error) {
				if owner.TryLock() {
					owner.Unlock()
					t.Fatal("operation ran without the cohort device owner")
				}
				if leaf == "attn.wq_a.weight" {
					order = append(order, [2]int{layer, lane})
					if !f.backend.active {
						t.Fatal("layer ran outside its recording window")
					}
				}
				if layer == -1 && f.backend.active {
					t.Fatal("head can commit before the layer window flush")
				}
				return project(layer, leaf, panel, out, in, rows)
			}
		}
		f.backend.reset()
		got, err := m.forwardV41StepCohort(&owner, f.lanes)
		if err != nil {
			t.Fatal(err)
		}
		if !reflect.DeepEqual(order, [][2]int{{0, 0}, {0, 1}, {1, 0}, {1, 1}}) || f.backend.beginCalls != 1 || f.backend.ends != 1 || f.backend.active {
			t.Fatalf("order=%v begin=%d end=%d", order, f.backend.beginCalls, f.backend.ends)
		}
		for lane, result := range got {
			if result.err != nil || !result.stats.Committed || !reflect.DeepEqual(result.logits, want[lane]) || !reflect.DeepEqual(captureV41ForwardSnapshot(f.lanes[lane].session.v41Forward), captureV41ForwardSnapshot(f.refs[lane].v41Forward)) {
				t.Fatalf("lane %d differs from serial: %+v", lane, result.stats)
			}
		}
	})

	t.Run("decline before mutation", func(t *testing.T) {
		f := prepare(t)
		var owner sync.Mutex
		before := captureV41ForwardSnapshot(f.lanes[0].session.v41Forward)
		owner.Lock()
		_, heldErr := m.forwardV41StepCohort(&owner, f.lanes)
		owner.Unlock()
		if !errors.Is(heldErr, errV41StepCohortDeclined) || f.backend.beginCalls != 0 {
			t.Fatalf("already-held device owner was admitted: %v", heldErr)
		}
		f.lanes[1].id = m.Cfg.VocabSize
		if _, err := m.forwardV41StepCohort(&owner, f.lanes); !errors.Is(err, errV41StepCohortDeclined) || f.backend.beginCalls != 0 || !reflect.DeepEqual(captureV41ForwardSnapshot(f.lanes[0].session.v41Forward), before) {
			t.Fatalf("bad token mutated a peer: %v", err)
		}
		f.lanes[1].id = 7
		oldBackend := f.lanes[1].session.Backend
		f.lanes[1].session.Backend = newV41PrefillStepBatchBackend()
		_, err := m.forwardV41StepCohort(&owner, f.lanes)
		f.lanes[1].session.Backend = oldBackend
		if !errors.Is(err, errV41StepCohortDeclined) || f.backend.beginCalls != 0 {
			t.Fatalf("mixed backend admitted: %v", err)
		}
		f.backend.beginOK = false
		if _, err := m.forwardV41StepCohort(&owner, f.lanes); !errors.Is(err, errV41StepCohortDeclined) || f.backend.beginCalls != 1 || f.backend.ends != 0 || !reflect.DeepEqual(captureV41ForwardSnapshot(f.lanes[0].session.v41Forward), before) {
			t.Fatalf("declined recording window mutated a peer: %v", err)
		}
	})

	t.Run("cancel one lane after its first layer", func(t *testing.T) {
		f := prepare(t)
		var owner sync.Mutex
		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()
		f.lanes[1].ctx = ctx
		st := f.lanes[1].session.v41State()
		before, registry := captureV41ForwardSnapshot(st), st.attn
		project := st.denseProjection
		st.denseProjection = func(layer int, leaf string, panel []float32, out, in, rows int) ([]float32, v41DenseProjectionOutcome, error) {
			if layer == 0 && leaf == "attn.wq_a.weight" {
				cancel()
			}
			return project(layer, leaf, panel, out, in, rows)
		}
		got, err := m.forwardV41StepCohort(&owner, f.lanes)
		if err != nil || !got[0].stats.Committed || !errors.Is(got[1].err, context.Canceled) || got[1].stats.Committed || got[1].stats.LayersRolledBack != 1 || got[1].logits != nil {
			t.Fatalf("cancellation results=%+v err=%v", got, err)
		}
		if st.attn != registry || !reflect.DeepEqual(captureV41ForwardSnapshot(st), before) {
			t.Fatal("cancelled lane retained ring or publication mutation")
		}
	})

	t.Run("flush failure rolls back every lane", func(t *testing.T) {
		f := prepare(t)
		var owner sync.Mutex
		before := []*v41ForwardSnapshot{captureV41ForwardSnapshot(f.lanes[0].session.v41Forward), captureV41ForwardSnapshot(f.lanes[1].session.v41Forward)}
		fault := errors.New("cohort final layer flush failed")
		f.backend.endErr = fault
		got, err := m.forwardV41StepCohort(&owner, f.lanes)
		if !errors.Is(err, fault) || f.backend.beginCalls != 1 || f.backend.ends != 1 {
			t.Fatalf("flush error=%v begin=%d end=%d", err, f.backend.beginCalls, f.backend.ends)
		}
		for lane := range f.lanes {
			if got[lane].stats.Committed || got[lane].stats.LayersRolledBack != m.Cfg.NumLayers || !reflect.DeepEqual(captureV41ForwardSnapshot(f.lanes[lane].session.v41Forward), before[lane]) {
				t.Fatalf("flush fault committed lane %d", lane)
			}
		}
	})

	for _, primaryFault := range []bool{false, true} {
		name := "flush panic aborts every lane"
		if primaryFault {
			name = "layer panic survives secondary flush panic"
		}
		t.Run(name, func(t *testing.T) {
			f := prepare(t)
			var owner sync.Mutex
			before := []*v41ForwardSnapshot{captureV41ForwardSnapshot(f.lanes[0].session.v41Forward), captureV41ForwardSnapshot(f.lanes[1].session.v41Forward)}
			secondary := errors.New("cohort End panic")
			var want error = secondary
			f.backend.endPanic = secondary
			if primaryFault {
				primary := &V41ProjectionOperationError{Layer: 1, Leaf: "attn.wq_a.weight", Cause: errors.New("selected layer panic")}
				want = primary
				st := f.lanes[1].session.v41State()
				project := st.denseProjection
				st.denseProjection = func(layer int, leaf string, panel []float32, out, in, rows int) ([]float32, v41DenseProjectionOutcome, error) {
					if layer == 1 && leaf == "attn.wq_a.weight" {
						panic(primary)
					}
					return project(layer, leaf, panel, out, in, rows)
				}
			}
			got := recoverError(func() { _, _ = m.forwardV41StepCohort(&owner, f.lanes) })
			if got != want || f.backend.beginCalls != 1 || f.backend.ends != 1 || f.backend.active {
				t.Fatalf("panic=%v want=%v begin=%d end=%d", got, want, f.backend.beginCalls, f.backend.ends)
			}
			for lane := range f.lanes {
				if !reflect.DeepEqual(captureV41ForwardSnapshot(f.lanes[lane].session.v41Forward), before[lane]) {
					t.Fatalf("panic retained lane %d mutation", lane)
				}
			}
			if !owner.TryLock() {
				t.Fatal("flush panic leaked device ownership")
			}
			owner.Unlock()
		})
	}

	t.Run("selected head fault preserves committed peer", func(t *testing.T) {
		f := prepare(t)
		var owner sync.Mutex
		before := captureV41ForwardSnapshot(f.lanes[1].session.v41Forward)
		registry := f.lanes[1].session.v41Forward.attn
		want, _, err := m.forwardV41Step(f.lanes[0].id, f.refs[0].v41State(), nil)
		if err != nil || len(want) == 0 {
			t.Fatal(err)
		}
		fault := errors.New("selected lane one head fault")
		calls := 0
		f.lanes[1].session.v41State().finalNorm = func([]float32) ([]float32, error) { calls++; return nil, fault }
		f.backend.reset()
		panicked := recoverError(func() { _, _ = m.forwardV41StepCohort(&owner, f.lanes) })
		if !errors.Is(panicked, fault) || calls != 1 || f.backend.beginCalls != 1 || f.backend.ends != 1 {
			t.Fatalf("selected failure was lost or replayed: %v calls=%d", panicked, calls)
		}
		if f.lanes[1].session.v41Forward.attn != registry || !reflect.DeepEqual(captureV41ForwardSnapshot(f.lanes[1].session.v41Forward), before) || !reflect.DeepEqual(captureV41ForwardSnapshot(f.lanes[0].session.v41Forward), captureV41ForwardSnapshot(f.refs[0].v41Forward)) {
			t.Fatal("head fault violated lane-local rollback or peer commit")
		}
		if !owner.TryLock() {
			t.Fatal("head panic leaked device ownership")
		}
		owner.Unlock()
	})
}
