package model

import (
	"context"
	"errors"
	"reflect"
	"sync"
	"sync/atomic"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

type v41BatchFixture struct {
	m      *Model
	b      *executionOwnerTestBackend
	owner  *sync.Mutex
	lanes  []V41BatchLane
	refs   []*Session
	before []*v41ForwardSnapshot
}

func newV41BatchFixture(t *testing.T) v41BatchFixture {
	t.Helper()
	f := v41BatchFixture{m: v41CompressorTestFixtureIndex(t, 128), owner: &sync.Mutex{}}
	f.m.Cfg.LogitScale = 0.5
	f.b = newExecutionOwnerTestBackend(f.owner)
	for i, prefix := range [][]int{{1, 2, 3}, {1, 2, 3, 4}} {
		source := f.m.NewSession()
		if len(source.Prefill(prefix)) == 0 {
			t.Fatal("prefix produced no logits")
		}
		snapshot := captureV41ForwardSnapshot(source.v41State())
		source.Close()
		s, ref := v41DenseTestSession(t, f.m, f.b), v41DenseTestSession(t, f.m, f.b)
		s.v41Forward, ref.v41Forward = snapshot.clone().restore(), snapshot.clone().restore()
		for _, st := range []*v41ForwardState{s.v41State(), ref.v41State()} {
			for _, state := range st.layers {
				slot := state.nextWindowPos % state.windowSize
				state.window[slot] = append([]float32(nil), state.window[0]...)
			}
		}
		f.lanes = append(f.lanes, V41BatchLane{ID: uint64(91 - i*74), Session: s, Token: 6 + i})
		f.refs = append(f.refs, ref)
		f.before = append(f.before, captureV41ForwardSnapshot(s.v41Forward))
	}
	scope := f.lanes[0].Session.beginBackendExecution()
	func() {
		defer scope.finish()
		dtype, ok := f.lanes[0].Session.v41HeadProjectionDtype(f.m.headName(), f.m.Cfg.VocabSize, f.m.Cfg.HiddenSize)
		if !ok {
			t.Fatal("fixture has no supported head")
		}
		f.b.headWeight = f.lanes[0].Session.v41HeadWeightHAL(f.m.headName(), dtype, f.m.Cfg.VocabSize, f.m.Cfg.HiddenSize).Buf()
	}()
	f.reset()
	return f
}

func (f v41BatchFixture) reset() {
	f.b.reset()
	f.b.ops, f.b.headRows, f.b.headInputs = nil, nil, nil
	f.b.calls.Store(0)
	f.b.panels.Store(0)
}

func (f v41BatchFixture) available(t *testing.T) {
	t.Helper()
	executionOwnerAvailable(t, f.owner, f.lanes[0].Session, f.lanes[1].Session)
}

func (f v41BatchFixture) unchanged(t *testing.T, i int) {
	t.Helper()
	if !reflect.DeepEqual(captureV41ForwardSnapshot(f.lanes[i].Session.v41Forward), f.before[i]) {
		t.Fatalf("lane %d retained mutation", i)
	}
}

func (f v41BatchFixture) receipt(t *testing.T, got V41BatchResult, rows int) {
	t.Helper()
	panels := 0
	sharedRows := 0
	if rows > 1 {
		panels, sharedRows = 1, rows
	}
	if got.SharedHeadPanels != panels || got.SharedHeadRows != sharedRows {
		t.Fatalf("receipt panels=%d rows=%d; expected panels=%d rows=%d", got.SharedHeadPanels, got.SharedHeadRows, panels, sharedRows)
	}
	actual := 0
	for _, op := range f.b.ops {
		if op.weight == f.b.headWeight {
			actual++
			if op.rows != rows || op.batch != (rows > 1) || op.in != f.m.Cfg.HiddenSize || op.out != f.m.Cfg.VocabSize {
				t.Fatalf("actual head operation differs: %+v", op)
			}
		}
	}
	want := 1
	if rows == 0 {
		want = 0
	}
	if actual != want {
		t.Fatalf("actual head operations=%d, want=%d", actual, want)
	}
}

// This is a caller/CPU-reference witness, not Vulkan or serving qualification.
// The runtime estimate is unmeasured until the bounded witness is authorized.
// fak-test:runtime medium est=5s lane=default
func TestV41BatchStep(t *testing.T) {
	t.Setenv(v41StepBatchEnv, "1")
	t.Run("stable identity parity and boundary accounting", func(t *testing.T) {
		f := newV41BatchFixture(t)
		want := make([][]float32, len(f.refs))
		for i, s := range f.refs {
			want[i] = s.Step(f.lanes[i].Token)
		}
		f.reset()
		beforeTokens := f.m.V41ExpertFaultAttribution().Decode.Tokens
		beforeHost := f.m.V41ExpertFaultAttribution().Decode.HeadProjectionHostCalls
		oldKeepAlive, oldProbe := beginSessionGPUKeepAlive, v41IncrementalStepProbe
		begins, ends, probes := 0, 0, 0
		beginSessionGPUKeepAlive = func(s *Session) func() {
			unowned := f.owner.TryLock()
			if unowned {
				f.owner.Unlock()
			}
			if s.backendExecutionScope == nil || unowned {
				panic("batch keepalive began without ownership")
			}
			begins++
			return func() { ends++ }
		}
		v41IncrementalStepProbe = func() { probes++ }
		defer func() { beginSessionGPUKeepAlive, v41IncrementalStepProbe = oldKeepAlive, oldProbe }()
		got, err := f.m.StepV41Batch(f.lanes)
		if err != nil || !got.Selected || begins != 2 || ends != 2 || probes != 2 {
			t.Fatalf("result=%+v err=%v keepalive=%d/%d probes=%d", got, err, begins, ends, probes)
		}
		f.receipt(t, got, 2)
		for i, lane := range got.Lanes {
			if lane.ID != f.lanes[i].ID || lane.Err != nil || !lane.Committed || !reflect.DeepEqual(lane.Logits, want[i]) || !reflect.DeepEqual(captureV41ForwardSnapshot(f.lanes[i].Session.v41Forward), captureV41ForwardSnapshot(f.refs[i].v41Forward)) {
				t.Fatalf("lane %d differs from scalar public Step: %+v", i, lane)
			}
		}
		if f.b.beginCalls != 1 || f.b.ends != 1 || f.m.v41ExpertFaultPhase() != V41PhaseUnknown || f.m.V41ExpertFaultAttribution().Decode.Tokens != beforeTokens+2 || f.m.V41ExpertFaultAttribution().Decode.HeadProjectionHostCalls != beforeHost {
			t.Fatal("recording, phase, token, or no-host-replay accounting differs")
		}
		f.available(t)
	})

	for _, why := range []string{"owner busy", "missing owner", "unsupported head", "duplicate ID", "duplicate session", "invalid token"} {
		t.Run(why+" is not fallback", func(t *testing.T) {
			f := newV41BatchFixture(t)
			input := append([]V41BatchLane(nil), f.lanes...)
			want := ErrV41BatchUnavailable
			switch why {
			case "owner busy":
				f.owner.Lock()
				want = ErrV41BatchBusy
			case "missing owner":
				for _, lane := range input {
					lane.Session.Backend = f.b.v41SharedHeadRecorder
				}
				defer func() {
					for _, lane := range input {
						lane.Session.Backend = f.b
					}
				}()
			case "unsupported head":
				f.b.denyShared = true
			case "duplicate ID":
				input[1].ID = input[0].ID
			case "duplicate session":
				input[1].Session = input[0].Session
			case "invalid token":
				input[1].Token = f.m.Cfg.VocabSize
			}
			got, err := f.m.StepV41Batch(input)
			if why == "owner busy" {
				f.owner.Unlock()
			}
			if !errors.Is(err, want) || got.Selected || f.b.calls.Load() != 0 || f.b.beginCalls != 0 {
				t.Fatalf("refusal touched execution: result=%+v err=%v calls=%d", got, err, f.b.calls.Load())
			}
			f.unchanged(t, 0)
			f.unchanged(t, 1)
			f.available(t)
		})
	}

	for _, site := range []string{"before selection", "during layer", "after projection", "during scale", "all after projection", "all before selection"} {
		t.Run("cancel "+site, func(t *testing.T) {
			f := newV41BatchFixture(t)
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			f.lanes[1].Context = ctx
			rows, committed := 2, 1
			switch site {
			case "before selection":
				cancel()
				rows = 1
			case "during layer":
				project := f.lanes[1].Session.v41State().denseProjection()
				f.lanes[1].Session.v41State().setDenseProjection(func(layer int, leaf string, x []float32, out, in, n int) ([]float32, v41DenseProjectionOutcome, error) {
					if layer == 0 && leaf == "attn.wq_a.weight" {
						cancel()
					}
					return project(layer, leaf, x, out, in, n)
				})
				rows = 1
			case "after projection":
				f.b.afterHead = cancel
			case "during scale":
				f.lanes[1].Session.v41State().logitScale = func(x []float32, factor float32) ([]float32, error) {
					cancel()
					y := append([]float32(nil), x...)
					for i := range y {
						y[i] *= factor
					}
					return y, nil
				}
			case "all after projection":
				f.lanes[0].Context = ctx
				f.b.afterHead = cancel
				committed = 0
			case "all before selection":
				f.lanes[0].Context = ctx
				cancel()
				committed, rows = 0, 0
			}
			beforeTokens := f.m.V41ExpertFaultAttribution().Decode.Tokens
			got, err := f.m.StepV41Batch(f.lanes)
			if err != nil || got.Lanes[1].ID != f.lanes[1].ID || !errors.Is(got.Lanes[1].Err, context.Canceled) || got.Lanes[1].Committed || got.Lanes[1].Logits != nil {
				t.Fatalf("cancellation result=%+v err=%v", got, err)
			}
			if got.Lanes[0].Committed != (committed == 1) || f.m.V41ExpertFaultAttribution().Decode.Tokens != beforeTokens+committed || f.m.v41ExpertFaultPhase() != V41PhaseUnknown {
				t.Fatal("cancellation lost commit or phase accounting")
			}
			f.receipt(t, got, rows)
			f.unchanged(t, 1)
			if committed == 0 {
				f.unchanged(t, 0)
			}
			if site == "all before selection" && (got.Selected || f.b.calls.Load() != 0) {
				t.Fatal("all-canceled batch selected execution")
			}
			f.available(t)
		})
	}

	t.Run("later selected failure preserves committed result", func(t *testing.T) {
		f := newV41BatchFixture(t)
		want := f.refs[0].Step(f.lanes[0].Token)
		f.reset()
		primary := errors.New("second lane logit scaling failed")
		f.lanes[1].Session.v41State().logitScale = func([]float32, float32) ([]float32, error) { return nil, primary }
		beforeTokens := f.m.V41ExpertFaultAttribution().Decode.Tokens
		beforeHost := f.m.V41ExpertFaultAttribution().Decode.HeadProjectionHostCalls
		got, err := f.m.StepV41Batch(f.lanes)
		if !errors.Is(err, primary) || !got.Selected || !got.Lanes[0].Committed || got.Lanes[0].Err != nil || !reflect.DeepEqual(got.Lanes[0].Logits, want) || got.Lanes[1].Committed || !errors.Is(got.Lanes[1].Err, primary) {
			t.Fatalf("partial commit result=%+v err=%v", got, err)
		}
		f.receipt(t, got, 2)
		f.unchanged(t, 1)
		if !reflect.DeepEqual(captureV41ForwardSnapshot(f.lanes[0].Session.v41Forward), captureV41ForwardSnapshot(f.refs[0].v41Forward)) || f.m.V41ExpertFaultAttribution().Decode.Tokens != beforeTokens+1 || f.m.V41ExpertFaultAttribution().Decode.HeadProjectionHostCalls != beforeHost || f.m.v41ExpertFaultPhase() != V41PhaseUnknown {
			t.Fatal("partial commit state, token attribution, or no-replay mismatch")
		}
		f.available(t)
	})

	for _, panics := range []bool{false, true} {
		name := "returned selected indexer error"
		if panics {
			name = "panicked selected indexer error"
		}
		t.Run(name+" retires only faulting lane", func(t *testing.T) {
			f := newV41BatchFixture(t)
			primary := &V41IndexerScoreOperationError{Layer: 0, Cause: errors.New("selected indexer rejected scores")}
			calls := 0
			f.lanes[1].Session.v41State().indexerScore = func(layer int, q, keys, weights []float32, heads, dim, rows int) ([]float32, error) {
				calls++
				if layer != 0 || rows == 0 {
					t.Fatal("selected score fault did not reach real retained source rows")
				}
				if panics {
					panic(primary)
				}
				return nil, primary
			}
			got, err := f.m.StepV41Batch(f.lanes)
			if calls != 1 || !errors.Is(err, primary) || !errors.Is(got.Lanes[1].Err, primary) || got.Lanes[0].Err != ErrV41BatchAborted || f.lanes[0].Session.BackendSessionClosed() || !f.lanes[1].Session.BackendSessionClosed() {
				t.Fatalf("indexer fault attribution result=%+v err=%v", got, err)
			}
			f.unchanged(t, 0)
			f.unchanged(t, 1)
			if f.b.beginCalls != 1 || f.b.ends != 1 || f.m.v41ExpertFaultPhase() != V41PhaseUnknown {
				t.Fatal("selected indexer unwind missed recording or phase cleanup")
			}
			f.available(t)
		})
	}

	t.Run("device-only validation still closes invalid lane", func(t *testing.T) {
		f := newV41BatchFixture(t)
		for _, lane := range f.lanes {
			lane.Session.executionPolicy = ExecutionPolicyDeviceOnly
		}
		got, err := f.m.StepV41Batch(f.lanes)
		var operation *BackendForwardOperationError
		if !got.Selected || !errors.As(err, &operation) || !f.lanes[0].Session.BackendSessionClosed() || f.lanes[1].Session.BackendSessionClosed() || got.Lanes[1].Err != ErrV41BatchAborted || f.b.beginCalls != 0 {
			t.Fatalf("device-only boundary bypassed: result=%+v err=%v", got, err)
		}
		f.unchanged(t, 0)
		f.unchanged(t, 1)
		f.available(t)
	})

	t.Run("shared projection fault is not blamed on one lane", func(t *testing.T) {
		f := newV41BatchFixture(t)
		primary := &compute.BackendError{Backend: "batch-test", Site: "shared head", Err: errors.New("selected panel failed")}
		f.b.primary = primary
		beforeHost := f.m.V41ExpertFaultAttribution().Decode.HeadProjectionHostCalls
		got, err := f.m.StepV41Batch(f.lanes)
		f.b.primary = nil
		if !errors.Is(err, primary) || !got.Selected || got.SharedHeadPanels != 0 || got.SharedHeadRows != 0 || f.m.V41ExpertFaultAttribution().Decode.HeadProjectionHostCalls != beforeHost {
			t.Fatalf("shared failure receipt or no-replay mismatch: %+v %v", got, err)
		}
		for i, lane := range got.Lanes {
			if lane.Err != ErrV41BatchAborted || lane.Committed || !f.lanes[i].Session.BackendSessionClosed() {
				t.Fatalf("shared operation wrongly blamed one lane: %+v", got)
			}
			f.unchanged(t, i)
		}
		f.available(t)
	})

	t.Run("unknown panic keeps identity and unwinds", func(t *testing.T) {
		f := newV41BatchFixture(t)
		primary := &struct{ value string }{"unclassified backend panic"}
		f.b.primary = primary
		var got any
		func() {
			defer func() { got = recover() }()
			_, _ = f.m.StepV41Batch(f.lanes)
		}()
		if got != primary || f.b.beginCalls != 1 || f.b.ends != 1 || f.m.v41ExpertFaultPhase() != V41PhaseUnknown {
			t.Fatalf("unknown panic identity or cleanup changed: %v", got)
		}
		f.b.primary = nil
		f.unchanged(t, 0)
		f.unchanged(t, 1)
		f.available(t)
	})
}

// A public Close on an admitted Session must wait for the actual result-bearing
// entrypoint, including selected teardown; the owner is the backend domain.
// fak-test:runtime medium est=1s lane=default
func TestV41BatchStepCloseWaiter(t *testing.T) {
	t.Setenv(v41StepBatchEnv, "1")
	f := newV41BatchFixture(t)
	entered, release := make(chan struct{}), make(chan struct{})
	var releaseOnce sync.Once
	finish := func() { releaseOnce.Do(func() { close(release) }) }
	t.Cleanup(finish)
	var held atomic.Bool
	primary := errors.New("selected batch fault")
	f.b.hook = func(site string) {
		if site == "matmul" && held.CompareAndSwap(false, true) {
			close(entered)
			<-release
			panic(f.lanes[0].Session.v41CompressorCloseFailure(0, "batch waiter", primary))
		}
	}
	var out V41BatchResult
	var err error
	batch := executionOwnerAsync(func() { out, err = f.m.StepV41Batch(f.lanes) })
	executionOwnerSignal(t, entered)
	f.b.query = make(chan struct{}, 8)
	before := f.b.calls.Load()
	closed := executionOwnerAsync(f.lanes[0].Session.Close)
	executionOwnerSignal(t, f.b.query)
	if f.b.calls.Load() != before {
		t.Fatal("public Close entered the active batch's domain")
	}
	finish()
	if p := executionOwnerWait(t, batch); p != nil {
		t.Fatalf("selected batch panic escaped: %v", p)
	}
	if p := executionOwnerWait(t, closed); p != nil {
		t.Fatalf("waiting Close failed: %v", p)
	}
	f.b.hook = nil
	if !errors.Is(err, primary) || !out.Selected || !errors.Is(out.Lanes[0].Err, primary) || out.Lanes[1].Err != ErrV41BatchAborted || !f.lanes[0].Session.BackendSessionClosed() {
		t.Fatalf("selected teardown lost primary/result: %+v %v", out, err)
	}
	f.unchanged(t, 0)
	f.unchanged(t, 1)
	f.available(t)
}

// fak-test:runtime medium est=1s lane=default
func TestV41BatchStepBusyPublicForward(t *testing.T) {
	t.Setenv(v41StepBatchEnv, "1")
	f := newV41BatchFixture(t)
	other := newExecutionOwnerTestBackend(f.owner)
	serial := v41DenseTestSession(t, f.m, other)
	serial.Prefill([]int{1, 2})
	entered, release := make(chan struct{}), make(chan struct{})
	var releaseOnce sync.Once
	finish := func() { releaseOnce.Do(func() { close(release) }) }
	t.Cleanup(finish)
	var held atomic.Bool
	other.hook = func(site string) {
		if site == "begin" && held.CompareAndSwap(false, true) {
			close(entered)
			<-release
		}
	}
	forward := executionOwnerAsync(func() { serial.Step(5) })
	executionOwnerSignal(t, entered)
	got, err := f.m.StepV41Batch(f.lanes)
	if err != ErrV41BatchBusy || got.Selected || f.b.calls.Load() != 0 || f.b.beginCalls != 0 {
		t.Fatalf("busy public forward triggered fallback: %+v %v", got, err)
	}
	f.unchanged(t, 0)
	f.unchanged(t, 1)
	finish()
	if p := executionOwnerWait(t, forward); p != nil {
		t.Fatalf("public forward failed: %v", p)
	}
	other.hook = nil
	f.available(t)
}
