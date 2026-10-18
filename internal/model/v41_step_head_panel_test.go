package model

import (
	"context"
	"errors"
	"reflect"
	"sync"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

type v41SharedHeadRecorder struct {
	*v41PrefillStepBatchBackend
	headWeight               compute.Buffer
	headRows                 []int
	headInputs               [][]float32
	beforeHead, afterHead    func()
	denyShared               bool
	primary, retirementPanic any
	freePanic, readPanic     any
	badRead                  bool
	headTensors, headOutputs map[compute.Buffer]bool
	freeAttempts             int
}

func (b *v41SharedHeadRecorder) V41SharedHeadAvailable(dtype compute.Dtype, out, in, maxRows int) bool {
	// This recorder executes cpu-ref's actual panel primitive. F32/Q2_K/Q6_K
	// use the same per-row dot as scalar MatMul; Q8_0 does not.
	return !b.denyShared && (dtype == compute.F32 || dtype == compute.Q2_K || dtype == compute.Q6_K) && out > 0 && in > 0 && maxRows <= v41SharedHeadMaxLanes
}
func (b *v41SharedHeadRecorder) BeginBatch() {}
func (b *v41SharedHeadRecorder) FlushBatch() {
	if b.retirementPanic != nil {
		panic(b.retirementPanic)
	}
}
func (b *v41SharedHeadRecorder) multiply(w, x compute.Tensor, rows int, batched bool) compute.Tensor {
	if w.Buf() == b.headWeight {
		b.headTensors[x.Buf()] = true
		b.headRows = append(b.headRows, rows)
		b.headInputs = append(b.headInputs, append([]float32(nil), x.Buf().(compute.HostBuffer).F32()...))
		if b.beforeHead != nil {
			b.beforeHead()
		}
		if b.primary != nil {
			panic(b.primary)
		}
	}
	var y compute.Tensor
	if batched {
		y = b.v41PrefillStepBatchBackend.BatchedMatMul(w, x, rows)
	} else {
		y = b.v41PrefillStepBatchBackend.MatMul(w, x)
	}
	if w.Buf() == b.headWeight {
		b.headTensors[y.Buf()], b.headOutputs[y.Buf()] = true, true
		if b.afterHead != nil {
			b.afterHead()
		}
	}
	return y
}
func (b *v41SharedHeadRecorder) MatMul(w, x compute.Tensor) compute.Tensor {
	return b.multiply(w, x, 1, false)
}
func (b *v41SharedHeadRecorder) BatchedMatMul(w, x compute.Tensor, rows int) compute.Tensor {
	return b.multiply(w, x, rows, true)
}

func (b *v41SharedHeadRecorder) Read(x compute.Tensor) []float32 {
	if b.headOutputs[x.Buf()] && b.readPanic != nil {
		panic(b.readPanic)
	}
	values := b.v41PrefillStepBatchBackend.Read(x)
	if b.headOutputs[x.Buf()] && b.badRead {
		return append(append([]float32(nil), values...), 0)
	}
	return values
}
func (b *v41SharedHeadRecorder) Free(x compute.Tensor) {
	head := b.headTensors[x.Buf()]
	if head {
		b.freeAttempts++
		delete(b.headTensors, x.Buf())
		delete(b.headOutputs, x.Buf())
	}
	b.v41PrefillStepBatchBackend.Free(x)
	if head && b.freePanic != nil {
		panic(b.freePanic)
	}
}

type v41SharedHeadFixture struct {
	m             *Model
	b             *v41SharedHeadRecorder
	lanes         []v41StepCohortLane
	refs          []*Session
	before        []*v41ForwardSnapshot
	headHostCalls int
}

func newV41SharedHeadFixture(t *testing.T, m *Model) v41SharedHeadFixture {
	t.Helper()
	f := v41SharedHeadFixture{m: m, b: &v41SharedHeadRecorder{v41PrefillStepBatchBackend: newV41PrefillStepBatchBackend(), headTensors: map[compute.Buffer]bool{}, headOutputs: map[compute.Buffer]bool{}}}
	for lane, prefix := range [][]int{{1, 2, 3}, {1, 2, 3, 4}} {
		source := m.NewSession()
		if len(source.Prefill(prefix)) == 0 {
			t.Fatal("prefix produced no logits")
		}
		snapshot := captureV41ForwardSnapshot(source.v41State())
		source.Close()
		s, ref := v41DenseTestSession(t, m, f.b), v41DenseTestSession(t, m, f.b)
		s.v41Forward, ref.v41Forward = snapshot.clone().restore(), snapshot.clone().restore()
		for _, st := range []*v41ForwardState{s.v41State(), ref.v41State()} {
			for _, state := range st.layers {
				slot := state.nextWindowPos % state.windowSize
				state.window[slot] = append([]float32(nil), state.window[0]...)
			}
		}
		f.lanes = append(f.lanes, v41StepCohortLane{session: s, id: 6 + lane})
		f.refs = append(f.refs, ref)
		f.before = append(f.before, captureV41ForwardSnapshot(s.v41Forward))
	}
	dtype, ok := f.refs[0].v41HeadProjectionDtype(m.headName(), m.Cfg.VocabSize, m.Cfg.HiddenSize)
	if !ok {
		t.Fatal("fixture has no device head")
	}
	f.b.headWeight = f.refs[0].v41HeadWeightHAL(m.headName(), dtype, m.Cfg.VocabSize, m.Cfg.HiddenSize).Buf()
	f.b.reset()
	f.b.ops = nil
	m.v41SetExpertFaultPhase(V41PhaseDecode)
	f.headHostCalls = m.V41ExpertFaultAttribution().Decode.HeadProjectionHostCalls
	return f
}

func (f v41SharedHeadFixture) assertRollback(t *testing.T, closed bool) {
	t.Helper()
	if got := f.m.V41ExpertFaultAttribution().Decode.HeadProjectionHostCalls; got != f.headHostCalls {
		t.Fatalf("shared selected operation replayed on host: before=%d after=%d", f.headHostCalls, got)
	}
	for lane, input := range f.lanes {
		if input.session.BackendSessionClosed() != closed || !reflect.DeepEqual(captureV41ForwardSnapshot(input.session.v41Forward), f.before[lane]) {
			t.Fatalf("lane %d closed=%t or snapshot changed after failure", lane, input.session.BackendSessionClosed())
		}
	}
}

// The recorder executes actual cpu-ref BatchedMatMul, not a panel counter or a
// sequential scalar stand-in. Physical Vulkan and throughput are unqualified.
// fak-test:runtime medium est=5s lane=default
func TestV41StepCohortSharedHead(t *testing.T) {
	t.Setenv(v41StepBatchEnv, "1")
	for _, dtype := range []compute.Dtype{compute.F32, compute.Q2_K, compute.Q6_K} {
		name := dtype.String() + " full BF16 graph"
		if dtype == compute.F32 {
			name = "F32 compressed source reader"
		}
		t.Run(name, func(t *testing.T) {
			var m *Model
			if dtype == compute.F32 {
				m = v41CompressorTestFixtureIndex(t, 128)
			} else {
				m = v41HeadProjFixture(t)
				if dtype == compute.Q6_K {
					m.kqw[m.headName()] = q6kTestTensor(m.Cfg.VocabSize, m.Cfg.HiddenSize, 0x61366)
				}
			}
			f := newV41SharedHeadFixture(t, m)
			var want [][]float32
			for lane, input := range f.lanes {
				logits, _, err := m.forwardV41Step(input.id, f.refs[lane].v41State(), nil)
				if err != nil {
					t.Fatal(err)
				}
				want = append(want, logits)
			}
			if !reflect.DeepEqual(f.b.headRows, []int{1, 1}) {
				t.Fatalf("serial oracle did not execute two scalar heads: %v", f.b.headRows)
			}
			wantInput := append(append([]float32(nil), f.b.headInputs[0]...), f.b.headInputs[1]...)
			f.b.headRows, f.b.headInputs, f.b.ops = nil, nil, nil
			var owner sync.Mutex
			f.b.beforeHead = func() {
				if owner.TryLock() {
					owner.Unlock()
					t.Fatal("shared head ran without device owner")
				}
				if f.b.active || f.b.ends != 1 {
					t.Fatal("shared head ran before final layer flush")
				}
				for lane, input := range f.lanes {
					if !reflect.DeepEqual(input.session.v41Forward.history, f.before[lane].history) {
						t.Fatal("lane committed before shared projection")
					}
				}
			}
			got, err := m.forwardV41StepCohortSharedHead(&owner, f.lanes)
			if err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(f.b.headRows, []int{2}) || !reflect.DeepEqual(f.b.headInputs, [][]float32{wantInput}) || f.b.beginCalls != 1 || f.b.ends != 1 {
				t.Fatalf("shared head rows=%v begin=%d end=%d or serial input boundary changed", f.b.headRows, f.b.beginCalls, f.b.ends)
			}
			actual := 0
			for _, op := range f.b.ops {
				if op.weight == f.b.headWeight {
					actual++
					if !op.batch || op.rows != 2 || op.in != m.Cfg.HiddenSize || op.out != m.Cfg.VocabSize || f.b.uploads[op.input] != 8*m.Cfg.HiddenSize || f.b.reads[op.output] != 8*m.Cfg.VocabSize {
						t.Fatalf("actual shared backend operation=%+v", op)
					}
				}
			}
			if actual != 1 {
				t.Fatalf("actual BatchedMatMul calls through one weight handle=%d", actual)
			}
			for lane, result := range got {
				if result.err != nil || !result.stats.Committed || !reflect.DeepEqual(result.logits, want[lane]) || !reflect.DeepEqual(captureV41ForwardSnapshot(f.lanes[lane].session.v41Forward), captureV41ForwardSnapshot(f.refs[lane].v41Forward)) {
					t.Fatalf("lane %d differs from serial: %+v", lane, result.stats)
				}
			}
		})
	}

	m := v41CompressorTestFixtureIndex(t, 128)
	m.Cfg.LogitScale = 0.5
	t.Run("unsupported capability declines before Begin", func(t *testing.T) {
		f := newV41SharedHeadFixture(t, m)
		f.b.denyShared = true
		var owner sync.Mutex
		if _, err := m.forwardV41StepCohortSharedHead(&owner, f.lanes); !errors.Is(err, errV41StepCohortDeclined) || f.b.beginCalls != 0 || len(f.b.ops) != 0 {
			t.Fatalf("unsupported capability mutated: %v", err)
		}
		f.assertRollback(t, false)
	})
	for _, teardown := range []bool{false, true} {
		name := "selected projection fault retires both"
		if teardown {
			name = "selected panic survives retirement panic"
		}
		t.Run(name, func(t *testing.T) {
			f := newV41SharedHeadFixture(t, m)
			primary := errors.New("selected shared head matmul panic")
			f.b.primary = primary
			if teardown {
				f.b.beforeHead = func() { f.b.retirementPanic = errors.New("secondary Close panic") }
			}
			var owner sync.Mutex
			got := recoverError(func() { _, _ = m.forwardV41StepCohortSharedHead(&owner, f.lanes) })
			f.b.retirementPanic = nil
			if got != primary || !reflect.DeepEqual(f.b.headRows, []int{2}) || f.b.beginCalls != 1 || f.b.ends != 1 {
				t.Fatalf("panic=%v rows=%v begin=%d end=%d", got, f.b.headRows, f.b.beginCalls, f.b.ends)
			}
			f.assertRollback(t, true)
			if !owner.TryLock() {
				t.Fatal("fault leaked device owner")
			}
			owner.Unlock()
		})
	}
	for _, site := range []string{"read panic", "invalid read", "successful operation"} {
		t.Run(site+" with transient Free panic", func(t *testing.T) {
			f := newV41SharedHeadFixture(t, m)
			primary, secondary := errors.New("selected head read"), errors.New("head transient Free")
			f.b.freePanic = secondary
			if site == "read panic" {
				f.b.readPanic = primary
			} else if site == "invalid read" {
				f.b.badRead = true
			}
			var owner sync.Mutex
			got := recoverError(func() { _, _ = m.forwardV41StepCohortSharedHead(&owner, f.lanes) })
			f.b.freePanic = nil
			switch site {
			case "read panic":
				if got != primary {
					t.Fatalf("primary read panic replaced: %v", got)
				}
			case "invalid read":
				if !errors.Is(got, errV41ProjectionResult) {
					t.Fatalf("primary read validation error replaced: %v", got)
				}
			default:
				if got != secondary {
					t.Fatalf("standalone Free panic hidden: %v", got)
				}
			}
			if f.b.freeAttempts != 2 {
				t.Fatalf("transient frees attempted=%d want2", f.b.freeAttempts)
			}
			f.assertRollback(t, true)
		})
	}

	t.Run("selected decline cannot replay on host", func(t *testing.T) {
		f := newV41SharedHeadFixture(t, m)
		st := f.lanes[1].session.v41State()
		norm := st.finalNorm()
		st.setFinalNorm(func(x []float32) ([]float32, error) {
			f.b.deny = true
			if norm != nil {
				return norm(x)
			}
			return m.finalNorm(x), nil
		})
		var owner sync.Mutex
		got := recoverError(func() { _, _ = m.forwardV41StepCohortSharedHead(&owner, f.lanes) })
		var selected *V41ProjectionOperationError
		if !errors.As(got, &selected) || len(f.b.headRows) != 0 {
			t.Fatalf("selected decline error=%v heads=%v", got, f.b.headRows)
		}
		f.assertRollback(t, true)
	})
	t.Run("cancellation after projection preserves peer commit", func(t *testing.T) {
		f := newV41SharedHeadFixture(t, m)
		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()
		f.lanes[1].ctx = ctx
		f.b.afterHead = cancel
		var owner sync.Mutex
		got, err := m.forwardV41StepCohortSharedHead(&owner, f.lanes)
		if err != nil || !got[0].stats.Committed || got[1].stats.Committed || !errors.Is(got[1].err, context.Canceled) || !reflect.DeepEqual(f.b.headRows, []int{2}) {
			t.Fatalf("cancellation results=%+v err=%v rows=%v", got, err, f.b.headRows)
		}
		if !reflect.DeepEqual(captureV41ForwardSnapshot(f.lanes[1].session.v41Forward), f.before[1]) {
			t.Fatal("cancelled lane retained shared-head mutation")
		}
	})
	t.Run("cancel during preparation leaves one scalar survivor", func(t *testing.T) {
		f := newV41SharedHeadFixture(t, m)
		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()
		f.lanes[1].ctx = ctx
		st := f.lanes[1].session.v41State()
		norm := st.finalNorm()
		st.setFinalNorm(func(x []float32) ([]float32, error) {
			cancel()
			if norm != nil {
				return norm(x)
			}
			return m.finalNorm(x), nil
		})
		var owner sync.Mutex
		got, err := m.forwardV41StepCohortSharedHead(&owner, f.lanes)
		if err != nil || !got[0].stats.Committed || got[1].stats.Committed || !errors.Is(got[1].err, context.Canceled) || !reflect.DeepEqual(f.b.headRows, []int{1}) {
			t.Fatalf("singleton cancellation results=%+v err=%v rows=%v", got, err, f.b.headRows)
		}
		if !reflect.DeepEqual(captureV41ForwardSnapshot(f.lanes[1].session.v41Forward), f.before[1]) {
			t.Fatal("cancelled prepared lane retained journal mutation")
		}
	})
	for _, cancelDuringScale := range []bool{false, true} {
		name := "late scale failure preserves committed peer"
		if cancelDuringScale {
			name = "cancellation during scale checked before commit"
		}
		t.Run(name, func(t *testing.T) {
			f := newV41SharedHeadFixture(t, m)
			want, _, err := m.forwardV41Step(f.lanes[0].id, f.refs[0].v41State(), nil)
			if err != nil {
				t.Fatal(err)
			}
			f.b.headRows = nil
			fault := errors.New("lane one scale failed")
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			f.lanes[1].ctx = ctx
			f.lanes[1].session.v41State().setLogitScale(func(values []float32, factor float32) ([]float32, error) {
				if !cancelDuringScale {
					return nil, fault
				}
				cancel()
				out := append([]float32(nil), values...)
				for i := range out {
					out[i] *= factor
				}
				return out, nil
			})
			var owner sync.Mutex
			got, err := m.forwardV41StepCohortSharedHead(&owner, f.lanes)
			if cancelDuringScale {
				if err != nil || !errors.Is(got[1].err, context.Canceled) {
					t.Fatalf("postscale cancellation result=%v err=%v", got[1].err, err)
				}
			} else if !errors.Is(err, fault) || !errors.Is(got[1].err, fault) {
				t.Fatalf("late scale failure=%v lane=%v", err, got[1].err)
			}
			if !got[0].stats.Committed || got[1].stats.Committed || !reflect.DeepEqual(got[0].logits, want) || !reflect.DeepEqual(f.b.headRows, []int{2}) {
				t.Fatalf("late finish changed peer commit: results=%+v rows=%v", got, f.b.headRows)
			}
			if !reflect.DeepEqual(captureV41ForwardSnapshot(f.lanes[0].session.v41Forward), captureV41ForwardSnapshot(f.refs[0].v41Forward)) || !reflect.DeepEqual(captureV41ForwardSnapshot(f.lanes[1].session.v41Forward), f.before[1]) {
				t.Fatal("late finish failure changed committed or aborted lane snapshot")
			}
		})
	}

}
