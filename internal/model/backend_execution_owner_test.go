package model

import (
	"context"
	"errors"
	"reflect"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// This backend executes the CPU reference primitives. Its owner represents one
// shared recording domain across distinct wrapper identities, not device proof.
type executionOwnerTestBackend struct {
	*v41SharedHeadRecorder
	owner  *sync.Mutex
	query  chan struct{}
	hook   func(string)
	calls  atomic.Int64
	panels atomic.Int64
}

func newExecutionOwnerTestBackend(owner *sync.Mutex) *executionOwnerTestBackend {
	return &executionOwnerTestBackend{
		owner: owner,
		v41SharedHeadRecorder: &v41SharedHeadRecorder{
			v41PrefillStepBatchBackend: newV41PrefillStepBatchBackend(),
			headTensors:                map[compute.Buffer]bool{}, headOutputs: map[compute.Buffer]bool{},
		},
	}
}

func (b *executionOwnerTestBackend) ForwardExecutionOwner() *sync.Mutex {
	if b.query != nil {
		select {
		case b.query <- struct{}{}:
		default:
		}
	}
	return b.owner
}

func (b *executionOwnerTestBackend) operation(site string) {
	if b.owner.TryLock() {
		b.owner.Unlock()
		panic("unowned backend operation: " + site)
	}
	b.calls.Add(1)
	if b.hook != nil {
		b.hook(site)
	}
}

func (b *executionOwnerTestBackend) NewKV(cfg compute.KVConfig) compute.KVStore {
	b.operation("new-kv")
	return &executionOwnerTestKV{KVStore: b.Backend.NewKV(cfg), backend: b}
}
func (b *executionOwnerTestBackend) GraphReset() { b.operation("graph-reset") }
func (b *executionOwnerTestBackend) Upload(x compute.Tensor, dtype compute.Dtype) compute.Tensor {
	b.operation("upload")
	return b.v41SharedHeadRecorder.Upload(x, dtype)
}
func (b *executionOwnerTestBackend) MatMul(w, x compute.Tensor) compute.Tensor {
	b.operation("matmul")
	return b.v41SharedHeadRecorder.MatMul(w, x)
}
func (b *executionOwnerTestBackend) BatchedMatMul(w, x compute.Tensor, rows int) compute.Tensor {
	b.operation("panel")
	if rows > 1 {
		b.panels.Add(1)
	}
	return b.v41SharedHeadRecorder.BatchedMatMul(w, x, rows)
}
func (b *executionOwnerTestBackend) Read(x compute.Tensor) []float32 {
	b.operation("read")
	return b.v41SharedHeadRecorder.Read(x)
}
func (b *executionOwnerTestBackend) Free(x compute.Tensor) {
	b.operation("free")
	b.v41SharedHeadRecorder.Free(x)
}
func (b *executionOwnerTestBackend) BeginV41StepBatch() bool {
	b.operation("begin")
	return b.v41SharedHeadRecorder.BeginV41StepBatch()
}
func (b *executionOwnerTestBackend) EndV41StepBatch() error {
	b.operation("end")
	return b.v41SharedHeadRecorder.EndV41StepBatch()
}
func (b *executionOwnerTestBackend) FlushBatch() {
	b.operation("flush")
	b.v41SharedHeadRecorder.FlushBatch()
}

type executionOwnerTestKV struct {
	compute.KVStore
	backend *executionOwnerTestBackend
}

func (k *executionOwnerTestKV) Clone() compute.KVStore {
	k.backend.operation("clone-kv")
	return &executionOwnerTestKV{KVStore: k.KVStore.Clone(), backend: k.backend}
}
func (k *executionOwnerTestKV) Free() {
	k.backend.operation("free-kv")
	k.KVStore.Free()
}
func (k *executionOwnerTestKV) Evict(from, n int) int {
	k.backend.operation("evict-kv")
	return k.KVStore.Evict(from, n)
}

func executionOwnerAsync(run func()) <-chan any {
	done := make(chan any, 1)
	go func() {
		defer func() { done <- recover() }()
		run()
	}()
	return done
}

func executionOwnerWait(t *testing.T, done <-chan any) any {
	t.Helper()
	select {
	case result := <-done:
		return result
	case <-time.After(5 * time.Second):
		t.Fatal("execution owner operation did not finish")
		return nil
	}
}

func executionOwnerSignal(t *testing.T, signal <-chan struct{}) {
	t.Helper()
	select {
	case <-signal:
	case <-time.After(5 * time.Second):
		t.Fatal("execution owner operation did not reach its boundary")
	}
}

func executionOwnerAvailable(t *testing.T, owner *sync.Mutex, sessions ...*Session) {
	t.Helper()
	if !owner.TryLock() {
		t.Fatal("execution owner leaked")
	}
	owner.Unlock()
	for _, s := range sessions {
		if s.backendExecutionScope != nil {
			t.Fatal("expired operation scope remains bound to a session")
		}
	}
}

// fak-test:runtime medium est=3s lane=default
func TestBackendExecutionOwnerLifecycle(t *testing.T) {
	t.Setenv(v41StepBatchEnv, "1")
	for _, entry := range []string{"step", "prefill", "prefill-no-logits", "close", "snapshot", "snapshot-clone", "snapshot-restore", "snapshot-close", "constructor", "evict", "disable-retained-logits"} {
		t.Run(entry, func(t *testing.T) {
			var owner sync.Mutex
			m := v41CompressorTestFixtureIndex(t, 128)
			a, b := newExecutionOwnerTestBackend(&owner), newExecutionOwnerTestBackend(&owner)
			sa, sb := v41DenseTestSession(t, m, a), v41DenseTestSession(t, m, b)
			sa.Prefill([]int{1, 2, 3})
			sb.Prefill([]int{1, 2, 3, 4})
			snapshot, err := sb.PrefixSnapshot()
			if err != nil {
				t.Fatal(err)
			}
			t.Cleanup(snapshot.Close)
			target := v41DenseTestSession(t, m, b)
			if entry == "disable-retained-logits" {
				// Prepare B's owned CPU-reference storage before A starts its
				// paused forward. This is lifetime protocol, not CUDA allocation.
				func() {
					owner.Lock()
					defer owner.Unlock()
					x := compute.NewF32(compute.Default(), []int{m.Cfg.VocabSize}, make([]float32, m.Cfg.VocabSize))
					sb.greedyLogits = b.Upload(x, compute.F32)
				}()
				sb.SetRetainGreedyLogits(true)
			}
			entered, release := make(chan struct{}), make(chan struct{})
			var releaseOnce sync.Once
			releaseFirst := func() { releaseOnce.Do(func() { close(release) }) }
			t.Cleanup(releaseFirst)
			var blocked atomic.Bool
			a.hook = func(site string) {
				if site == "begin" && blocked.CompareAndSwap(false, true) {
					close(entered)
					<-release
				}
			}
			first := executionOwnerAsync(func() { sa.Step(5) })
			executionOwnerSignal(t, entered)
			b.query = make(chan struct{}, 4)
			before := b.calls.Load()
			var frees atomic.Int64
			if entry == "disable-retained-logits" {
				b.hook = func(site string) {
					if site == "free" {
						frees.Add(1)
					}
				}
			}
			second := executionOwnerAsync(func() {
				switch entry {
				case "step":
					sb.Step(6)
				case "prefill":
					sb.Prefill([]int{6, 7})
				case "prefill-no-logits":
					sb.PrefillNoLogits([]int{6, 7})
				case "close":
					sb.Close()
				case "snapshot":
					got, err := sb.PrefixSnapshot()
					if err != nil {
						panic(err)
					}
					got.Close()
				case "snapshot-clone":
					got, err := snapshot.Clone()
					if err != nil {
						panic(err)
					}
					got.Close()
				case "snapshot-restore":
					if err := snapshot.Restore(target); err != nil {
						panic(err)
					}
				case "snapshot-close":
					snapshot.Close()
				case "constructor":
					got, err := m.NewBackendSessionChecked(b)
					if err != nil {
						panic(err)
					}
					got.Close()
				case "evict":
					sb.evictKV(0, 1)
				case "disable-retained-logits":
					sb.SetRetainGreedyLogits(false)
				}
			})
			firstFinished, secondFinished := false, false
			if entry == "disable-retained-logits" {
				t.Cleanup(func() {
					releaseFirst()
					if !firstFinished {
						if r := executionOwnerWait(t, first); r != nil {
							t.Errorf("first forward cleanup panic: %v", r)
						}
					}
					if !secondFinished {
						if r := executionOwnerWait(t, second); r != nil {
							t.Errorf("setter cleanup panic: %v", r)
						}
					}
				})
				select {
				case <-b.query:
				case r := <-second:
					secondFinished = true
					t.Fatalf("disable setter completed while another session owned its forward: %v", r)
				case <-time.After(5 * time.Second):
					t.Fatal("disable setter did not attempt owner acquisition")
				}
				if !sb.retainGreedyLogits || sb.greedyLogits.Buf() == nil {
					t.Fatal("disable setter mutated retained state before acquiring owner")
				}
			} else {
				executionOwnerSignal(t, b.query)
			}
			if got := b.calls.Load(); got != before {
				t.Fatalf("foreign operation entered active window: %d -> %d", before, got)
			}
			releaseFirst()
			firstResult := executionOwnerWait(t, first)
			firstFinished = true
			if r := firstResult; r != nil {
				t.Fatalf("first forward panic: %v", r)
			}
			secondResult := executionOwnerWait(t, second)
			secondFinished = true
			if r := secondResult; r != nil {
				t.Fatalf("second public operation panic: %v", r)
			}
			if entry == "disable-retained-logits" {
				if frees.Load() != 1 || b.calls.Load() != before+1 || sb.retainGreedyLogits || sb.greedyLogits.Buf() != nil {
					t.Fatal("disable setter did not free exactly once and clear retained state")
				}
				queries := len(b.query)
				sb.SetRetainGreedyLogits(false)
				if frees.Load() != 1 || b.calls.Load() != before+1 || len(b.query) != queries {
					t.Fatal("repeated empty disable acquired owner or freed again")
				}
			}
			executionOwnerAvailable(t, &owner, sa, sb, target)
		})
	}
}

// fak-test:runtime medium est=1s lane=default
func TestBackendExecutionOwnerCohortAndSelectedFailure(t *testing.T) {
	t.Setenv(v41StepBatchEnv, "1")
	var owner sync.Mutex
	m := v41CompressorTestFixtureIndex(t, 128)
	b := newExecutionOwnerTestBackend(&owner)
	a, c := v41DenseTestSession(t, m, b), v41DenseTestSession(t, m, b)
	a.Prefill([]int{1, 2, 3})
	c.Prefill([]int{1, 2, 3, 4})
	lanes := []v41StepCohortLane{{session: a, id: 5}, {session: c, id: 6}}
	before := []*v41ForwardSnapshot{captureV41ForwardSnapshot(a.v41Forward), captureV41ForwardSnapshot(c.v41Forward)}
	owner.Lock()
	_, err := m.forwardV41StepCohortSharedHead(&owner, lanes)
	owner.Unlock()
	if !errors.Is(err, errV41StepCohortDeclined) {
		t.Fatalf("busy domain admission: %v", err)
	}
	for i, lane := range lanes {
		if !reflect.DeepEqual(before[i], captureV41ForwardSnapshot(lane.session.v41Forward)) {
			t.Fatal("busy admission mutated a lane")
		}
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	lanes[1].ctx = ctx
	b.panels.Store(0)
	results, err := m.forwardV41StepCohortSharedHead(&owner, lanes)
	if err != nil || !results[0].stats.Committed || !errors.Is(results[1].err, context.Canceled) || b.panels.Load() != 0 {
		t.Fatalf("one-survivor result: %+v, %v", results, err)
	}
	if !reflect.DeepEqual(before[1], captureV41ForwardSnapshot(c.v41Forward)) {
		t.Fatal("canceled lane changed")
	}
	executionOwnerAvailable(t, &owner, a, c)
	lanes[1].ctx = nil
	results, err = m.forwardV41StepCohortSharedHead(&owner, lanes)
	if err != nil || !results[0].stats.Committed || !results[1].stats.Committed || b.panels.Load() != 1 {
		t.Fatalf("actual shared projection: %+v, %v, panels=%d", results, err, b.panels.Load())
	}
	executionOwnerAvailable(t, &owner, a, c)

	// Exercise an otherwise unguarded selected-error close path. The original
	// typed compressor failure must survive a FlushBatch cleanup panic.
	primary, cleanup := errors.New("selected compressor failure"), errors.New("retirement failure")
	b.hook = func(site string) {
		if site == "matmul" {
			panic(a.v41CompressorCloseFailure(0, "owner witness", primary))
		}
		if site == "flush" {
			panic(cleanup)
		}
	}
	r := panicAsError(func() { a.Step(7) })
	if !errors.Is(r, primary) || errors.Is(r, cleanup) || !a.BackendSessionClosed() {
		t.Fatalf("selected primary replaced: %v", r)
	}
	b.hook = nil
	executionOwnerAvailable(t, &owner, a, c)
}

// fak-test:runtime fast est=100ms lane=default
func TestBackendExecutionOwnerRetirementAndNestedRestore(t *testing.T) {
	var owner sync.Mutex
	m := v41CompressorTestFixtureIndex(t, 128)
	b := newExecutionOwnerTestBackend(&owner)
	s := v41DenseTestSession(t, m, b)
	s.Prefill([]int{1, 2, 3})
	snapshot, err := s.PrefixSnapshot()
	if err != nil {
		t.Fatal(err)
	}
	defer snapshot.Close()
	// This is the same private nested restore used by target transaction Abort
	// and partial Commit. Borrowing this exact scope must not re-acquire it.
	scope := s.beginBackendExecution()
	func() {
		defer scope.finish()
		if err := snapshot.restoreOwned(scope, s); err != nil {
			t.Fatal(err)
		}
	}()
	executionOwnerAvailable(t, &owner, s)
	logits := s.Step(4)
	tx, err := beginQwen35MTPTargetTransaction(s, logits)
	if err != nil {
		t.Fatal(err)
	}
	if err := tx.Abort(); err != nil {
		t.Fatal(err)
	}
	tx, err = beginQwen35MTPTargetTransaction(s, logits)
	if err != nil {
		t.Fatal(err)
	}
	tx.draft = []int{5, 6}
	before := len(s.v41Forward.history)
	if got, err := tx.Commit(1); err != nil || len(got) != m.Cfg.VocabSize || len(s.v41Forward.history) != before+1 {
		t.Fatalf("partial transaction commit: logits=%d err=%v history=%d", len(got), err, len(s.v41Forward.history))
	}
	executionOwnerAvailable(t, &owner, s)
	var retired atomic.Bool
	scope = s.beginBackendExecution()
	scope.afterRelease(func() {
		if !owner.TryLock() {
			panic("model retirement ran under old domain")
		}
		owner.Unlock()
		retired.Store(true)
	})
	primary := errors.New("forward failed")
	r := panicAsError(func() { defer scope.finish(); panic(primary) })
	if r != primary || !retired.Load() {
		t.Fatalf("post-release retirement result=%v retired=%t", r, retired.Load())
	}
	executionOwnerAvailable(t, &owner, s)
}

type executionOwnerCloser struct {
	owner  *sync.Mutex
	closed bool
}

func (c *executionOwnerCloser) Close() error {
	if !c.owner.TryLock() {
		panic("checkpoint release ran under session domain")
	}
	c.owner.Unlock()
	c.closed = true
	return nil
}

// fak-test:runtime medium est=1s lane=default
func TestBackendExecutionOwnerLastSessionAndUnsupportedProducers(t *testing.T) {
	var owner sync.Mutex
	m := v41CompressorTestFixtureIndex(t, 128)
	b := newExecutionOwnerTestBackend(&owner)
	s := v41DenseTestSession(t, m, b)
	s.Prefill([]int{1, 2, 3})
	closer := &executionOwnerCloser{owner: &owner}
	m.SetWeightCloser(closer)
	var active *WeightSessionsActiveError
	if err := m.CloseWeights(); !errors.As(err, &active) {
		t.Fatalf("active model close: %v", err)
	}
	s.Close()
	if !closer.closed {
		t.Fatal("last session did not complete checkpoint retirement")
	}
	executionOwnerAvailable(t, &owner, s)
	before := b.calls.Load()
	if _, err := NewMoEDoubleBufferedPipeline(MoEPipelineConfig{Backend: b, NumExperts: 2, NumLayers: 2}); !errors.Is(err, ErrMoEPipelineExecutionOwner) {
		t.Fatalf("asynchronous producer admission: %v", err)
	}
	if _, err := NewBackendCollective(b); err == nil {
		t.Fatal("unowned collective was admitted")
	}
	if b.calls.Load() != before {
		t.Fatal("unsupported producer touched backend")
	}
	mgr := NewQwen38MTPCancellationManager(Qwen38MTPBoundsConfig{})
	if _, _, err := mgr.ExecuteSessionRoundWithContext(context.Background(), &Qwen38MTPPersistentSession{target: s}); !errors.Is(err, ErrMTPExecutionOwnerCancellation) {
		t.Fatalf("unjoined cancellation admission: %v", err)
	}
	if b.calls.Load() != before {
		t.Fatal("unjoined cancellation touched backend before refusal")
	}
	if p, err := NewMoEDoubleBufferedPipeline(MoEPipelineConfig{Backend: compute.Default(), NumExperts: 2, NumLayers: 2}); err != nil || p == nil {
		t.Fatalf("legacy no-owner pipeline changed: %v", err)
	}
}

type executionOwnerAsyncBackend struct{ *executionOwnerTestBackend }

func (b *executionOwnerAsyncBackend) UploadAsync(compute.Tensor, compute.Dtype) (compute.Tensor, compute.Fence) {
	panic("owner-capable ring launched an unjoined asynchronous transfer")
}

// fak-test:runtime fast est=100ms lane=default
func TestBackendExecutionOwnerAsyncAndConstructorFailure(t *testing.T) {
	var owner sync.Mutex
	b := &executionOwnerAsyncBackend{newExecutionOwnerTestBackend(&owner)}
	scope := beginBackendExecution(b)
	func() {
		defer scope.finish()
		src := compute.NewF32(compute.Default(), []int{4}, []float32{1, 2, 3, 4})
		got, fence := uploadStagedAsync(b, src, compute.F32, "owner witness", false)
		if fence != nil {
			t.Fatal("owned ring retained a pending fence")
		}
		b.Free(got)
	}()
	executionOwnerAvailable(t, &owner)
	m := v41CompressorTestFixtureIndex(t, 128)
	primary := errors.New("constructor graph reset failure")
	b.hook = func(site string) {
		if site == "graph-reset" {
			panic(primary)
		}
	}
	r := panicAsError(func() { _, _ = m.NewBackendSessionChecked(b) })
	if r != primary {
		t.Fatalf("constructor primary changed: %v", r)
	}
	b.hook = nil
	executionOwnerAvailable(t, &owner)
	m.weightCloser.mu.Lock()
	holds := m.weightCloser.sessions
	m.weightCloser.mu.Unlock()
	if holds != 0 {
		t.Fatalf("failed constructor retained %d model holds", holds)
	}
}

// fak-test:runtime medium est=1s lane=default
func TestBackendExecutionOwnerRealCohortVersusPublicStep(t *testing.T) {
	t.Setenv(v41StepBatchEnv, "1")
	var owner sync.Mutex
	m := v41CompressorTestFixtureIndex(t, 128)
	b, other := newExecutionOwnerTestBackend(&owner), newExecutionOwnerTestBackend(&owner)
	a, c := v41DenseTestSession(t, m, b), v41DenseTestSession(t, m, b)
	serial := v41DenseTestSession(t, m, other)
	a.Prefill([]int{1, 2, 3})
	c.Prefill([]int{1, 2, 3, 4})
	serial.Prefill([]int{1, 2})
	lanes := []v41StepCohortLane{{session: a, id: 5}, {session: c, id: 6}}
	entered, release := make(chan struct{}), make(chan struct{})
	var releaseOnce sync.Once
	releaseFirst := func() { releaseOnce.Do(func() { close(release) }) }
	t.Cleanup(releaseFirst)
	var held atomic.Bool
	b.hook = func(site string) {
		if site == "begin" && held.CompareAndSwap(false, true) {
			close(entered)
			<-release
		}
	}
	cohort := executionOwnerAsync(func() {
		got, err := m.forwardV41StepCohortSharedHead(&owner, lanes)
		if err != nil {
			panic(err)
		}
		if !got[0].stats.Committed || !got[1].stats.Committed {
			panic("cohort failed to commit")
		}
	})
	executionOwnerSignal(t, entered)
	other.query = make(chan struct{}, 4)
	before := other.calls.Load()
	step := executionOwnerAsync(func() { serial.Step(7) })
	executionOwnerSignal(t, other.query)
	if other.calls.Load() != before {
		t.Fatal("public Step entered active cohort window")
	}
	releaseFirst()
	if r := executionOwnerWait(t, cohort); r != nil {
		t.Fatalf("cohort: %v", r)
	}
	if r := executionOwnerWait(t, step); r != nil {
		t.Fatalf("serial: %v", r)
	}
	b.hook = nil
	executionOwnerAvailable(t, &owner, a, c, serial)
	// Reverse direction: a real legacy public forward owns the domain, so the
	// cohort must decline before Begin instead of joining its recording window.
	entered2, release2 := make(chan struct{}), make(chan struct{})
	var once2 sync.Once
	releaseSecond := func() { once2.Do(func() { close(release2) }) }
	t.Cleanup(releaseSecond)
	var held2 atomic.Bool
	other.hook = func(site string) {
		if site == "begin" && held2.CompareAndSwap(false, true) {
			close(entered2)
			<-release2
		}
	}
	step = executionOwnerAsync(func() { serial.Step(6) })
	executionOwnerSignal(t, entered2)
	starts := b.beginCalls
	_, err := m.forwardV41StepCohortSharedHead(&owner, lanes)
	if !errors.Is(err, errV41StepCohortDeclined) || b.beginCalls != starts {
		t.Fatalf("cohort joined legacy recording: %v", err)
	}
	releaseSecond()
	if r := executionOwnerWait(t, step); r != nil {
		t.Fatalf("legacy: %v", r)
	}
	other.hook = nil
	executionOwnerAvailable(t, &owner, a, c, serial)
}

// fak-test:runtime medium est=1s lane=default
func TestBackendExecutionOwnerSelectedCloseWithExternalWaiter(t *testing.T) {
	t.Setenv(v41StepBatchEnv, "1")
	var owner sync.Mutex
	m := v41CompressorTestFixtureIndex(t, 128)
	b := newExecutionOwnerTestBackend(&owner)
	s := v41DenseTestSession(t, m, b)
	s.Prefill([]int{1, 2, 3})
	entered, release := make(chan struct{}), make(chan struct{})
	var releaseOnce sync.Once
	releaseFirst := func() { releaseOnce.Do(func() { close(release) }) }
	t.Cleanup(releaseFirst)
	primary, cleanup := errors.New("selected compressor failure"), errors.New("cleanup failed")
	var selected atomic.Bool
	b.hook = func(site string) {
		if site == "matmul" && selected.CompareAndSwap(false, true) {
			close(entered)
			<-release
			panic(s.v41CompressorCloseFailure(0, "concurrent close witness", primary))
		}
		if site == "flush" {
			panic(cleanup)
		}
	}
	forward := executionOwnerAsync(func() { s.Step(4) })
	executionOwnerSignal(t, entered)
	b.query = make(chan struct{}, 8)
	closed := executionOwnerAsync(s.Close)
	executionOwnerSignal(t, b.query)
	before := b.calls.Load()
	releaseFirst()
	got := executionOwnerWait(t, forward)
	err, ok := got.(error)
	if !ok || !errors.Is(err, primary) || errors.Is(err, cleanup) {
		t.Fatalf("primary = %v", got)
	}
	if got := executionOwnerWait(t, closed); got != nil {
		t.Fatalf("external Close: %v", got)
	}
	if b.calls.Load() <= before || !s.BackendSessionClosed() {
		t.Fatal("selected owned teardown was not executed")
	}
	b.hook = nil
	executionOwnerAvailable(t, &owner, s)
}
