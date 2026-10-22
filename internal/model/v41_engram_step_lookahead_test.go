package model

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"reflect"
	"runtime"
	"sync"
	"sync/atomic"
	"testing"
)

func v41StepLookAheadFixture(t *testing.T) (*Session, []*v41LookAheadTestReader) {
	t.Helper()
	m, sources := v41IncrementalEngramFixture(t, false)
	readers := make([]*v41LookAheadTestReader, len(sources))
	for i, old := range sources {
		data := old.base.packed
		r := &v41LookAheadTestReader{ReaderAt: bytes.NewReader(data), width: V41EngramPackedRowBytes,
			closing: make(chan struct{}), entered: make(chan struct{}), resume: make(chan struct{}),
			readReturned: make(chan struct{}), released: make(chan struct{}, 8)}
		r.eligible.Store(true)
		r.accept.Store(true)
		shard := V41EngramShard{ID: fmt.Sprintf("step-owned-%d", i), Size: int64(len(data)), Rows: old.base.rows}
		shard.ExpectedID = shard.ID
		binding := V41EngramArtifactBinding{Shard: shard.ID, Size: shard.Size, Rows: shard.Rows, RowBytes: r.width, Digest: v41EngramTestDigest(data)}
		source, err := NewV41VerifiedEngramRowSource(r, shard, binding)
		if err != nil {
			t.Fatal(err)
		}
		m.v41EngramStageFor().caches[i].src = source
		readers[i] = r
	}
	s := m.NewSession()
	t.Cleanup(s.Close)
	t.Cleanup(func() {
		for _, r := range readers {
			r.unblock()
		}
	})
	s.Prefill([]int{1, 3, 2})
	for i, cache := range m.v41EngramStageFor().caches {
		cache.invalidateRows()
		readers[i].active.Store(true)
	}
	return s, readers
}

type v41StepLookAheadOutcome struct {
	logits     []float32
	panicValue any
	returned   bool
}

type v41StepLookAheadCloseObserved struct {
	v41EngramLookAhead
	entered chan struct{}
	once    sync.Once
}

func (f *v41StepLookAheadCloseObserved) close() {
	f.once.Do(func() { close(f.entered) })
	f.v41EngramLookAhead.close()
}

// This is a connected serial Step witness, not a throughput benchmark. Reader
// and compute channels establish overlap and teardown order without sleeps.
// fak-test:justify why=regression when=changed:internal/model/**
// fak-test:runtime medium est=8s lane=default
func TestV41EngramStepLookAhead(t *testing.T) {
	for _, mode := range []string{"default-off", "overlap", "ineligible", "hold-refused", "deferred-read-error", "cancel-read", "earlier-panic", "earlier-goexit"} {
		t.Run(mode, func(t *testing.T) {
			s, r := v41StepLookAheadFixture(t)
			st := s.v41State()
			before := captureV41ForwardSnapshot(st)
			requests := v41IncrementalEngramRequests(s.M)
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			primary, secondary := errors.New("earlier layer fault"), errors.New("later row read fault")
			blocked := mode == "overlap" || mode == "cancel-read" || mode == "earlier-panic" || mode == "earlier-goexit"
			r[1].block.Store(blocked)
			if mode == "ineligible" {
				r[1].eligible.Store(false)
			}
			if mode == "hold-refused" {
				r[1].accept.Store(false)
			}
			if mode == "deferred-read-error" || mode == "earlier-panic" || mode == "earlier-goexit" {
				r[1].fault = secondary
			}
			computeEntered, resumeCompute := make(chan struct{}), make(chan struct{})
			var computeOnce sync.Once
			var layers atomic.Uint32
			st.setDenseProjection(func(layer int, _ string, _ []float32, _, _, _ int) ([]float32, v41DenseProjectionOutcome, error) {
				layers.Or(1 << uint(layer))
				if layer == 0 {
					computeOnce.Do(func() {
						if blocked {
							<-r[1].entered
						}
						if mode == "deferred-read-error" {
							<-r[1].readReturned
						}
						close(computeEntered)
						if blocked {
							<-resumeCompute
						}
						if mode == "earlier-panic" {
							panic(primary)
						}
						if mode == "earlier-goexit" {
							runtime.Goexit()
						}
					})
				}
				return nil, v41ProjectionDeclined, nil
			})
			done := make(chan v41StepLookAheadOutcome, 1)
			go func() {
				var result v41StepLookAheadOutcome
				defer func() { result.panicValue = recover(); done <- result }()
				if mode == "default-off" {
					result.logits = s.Step(5)
				} else {
					result.logits = s.StepWithV41EngramLookAhead(ctx, 5)
				}
				result.returned = true
			}()
			if blocked {
				<-computeEntered
				// Layer2's real ReaderAt is blocked while layer0's real projection
				// callback is running, with the prior panel already released.
				if !r[1].reading.Load() || r[1].holds.Load() != 1 || r[1].releases.Load() != 0 || r[0].releases.Load() != 1 {
					t.Error("next-layer read did not overlap earlier-layer projection dispatch with one retained panel")
				}
				if mode == "cancel-read" {
					cancel()
				}
				close(resumeCompute)
				r[1].unblock()
			}
			result := <-done
			if st.engramLookAheadContext != nil || v41EngramLookAheadCharge.Load() != 0 {
				t.Fatal("per-call context or process-wide credit escaped the Step")
			}
			for i, reader := range r {
				want := int64(1)
				if mode == "default-off" || (i == 1 && mode == "ineligible") {
					want = 0
				}
				wantReleased := want
				if i == 1 && mode == "hold-refused" {
					wantReleased = 0
				}
				if reader.holds.Load() != want || reader.releases.Load() != wantReleased || reader.releasedEarly.Load() {
					t.Fatalf("source%d holds/releases=%d/%d, want %d/%d; early=%t", i, reader.holds.Load(), reader.releases.Load(), want, wantReleased, reader.releasedEarly.Load())
				}
			}
			success := mode == "default-off" || mode == "overlap" || mode == "ineligible"
			if success {
				if !result.returned || result.panicValue != nil {
					t.Fatalf("Step failed: %+v", result)
				}
				oracle, _ := v41IncrementalEngramFixture(t, false)
				reference := oracle.NewSession()
				defer reference.Close()
				reference.Prefill([]int{1, 3, 2})
				if !reflect.DeepEqual(result.logits, reference.Step(5)) {
					t.Fatal("opt-in changed exact synchronous logits")
				}
				if got := v41IncrementalEngramRequests(s.M) - requests; got != 48 {
					t.Fatalf("one gather per declared layer requested %d rows, want48", got)
				}
				// The opt-in is per-call: a subsequent plain Step takes no hold.
				counts := make([][2]int64, len(r))
				for i, reader := range r {
					counts[i] = [2]int64{reader.holds.Load(), reader.releases.Load()}
					reader.active.Store(false)
				}
				st.setDenseProjection(nil)
				if !reflect.DeepEqual(s.Step(6), reference.Step(6)) {
					t.Fatal("subsequent synchronous Step changed")
				}
				for i, reader := range r {
					if got := [2]int64{reader.holds.Load(), reader.releases.Load()}; got != counts[i] {
						t.Fatalf("plain Step inherited opt-in for source%d: counts %v -> %v", i, counts[i], got)
					}
				}
				return
			}
			if result.returned || !reflect.DeepEqual(before, captureV41ForwardSnapshot(st)) {
				t.Fatal("failed token returned or changed committed continuation")
			}
			switch mode {
			case "earlier-panic":
				if result.panicValue != primary {
					t.Fatalf("primary panic replaced: %v", result.panicValue)
				}
			case "earlier-goexit":
				if result.panicValue != nil {
					t.Fatalf("Goexit replaced: %v", result.panicValue)
				}
			default:
				err, ok := result.panicValue.(error)
				if !ok {
					t.Fatalf("missing failure: %v", result.panicValue)
				}
				if mode == "cancel-read" {
					if !errors.Is(err, context.Canceled) {
						t.Fatalf("cancellation lost: %v", err)
					}
				} else {
					var stage *V41ForwardError
					if !errors.As(err, &stage) || stage.Layer != 2 || !errors.Is(err, ErrV41NativeUnsupported) {
						t.Fatalf("later-layer typed refusal lost: %v", err)
					}
					if layers.Load()&3 != 3 || layers.Load()&4 != 0 {
						t.Fatal("later refusal was not deferred to layer2")
					}
					if mode == "deferred-read-error" && !errors.Is(err, secondary) {
						t.Fatal("read cause lost")
					}
					if mode == "hold-refused" && r[1].reads.Load() != 0 {
						t.Fatal("refused hold replayed an unretained read")
					}
				}
			}
		})
	}
	t.Run("join-before-teardown", func(t *testing.T) {
		s, r := v41StepLookAheadFixture(t)
		st := s.v41State()
		st.engramLookAheadContext = context.Background()
		defer func() { st.engramLookAheadContext = nil }()
		r[0].block.Store(true)
		tx, err := s.M.beginV41Step(5, st, &v41ProjScratch{})
		if err != nil {
			t.Fatal(err)
		}
		<-r[0].entered
		observed := &v41StepLookAheadCloseObserved{v41EngramLookAhead: tx.engramLookAhead.future, entered: make(chan struct{})}
		tx.engramLookAhead.future = observed
		var cleaned atomic.Bool
		cleanup := tx.cleanup
		tx.cleanup = func() { cleaned.Store(true); cleanup() }
		done := make(chan struct{})
		go func() { defer close(done); tx.abort() }()
		<-observed.entered
		// close entry synchronizes this observation. The underlying read remains
		// blocked, so no valid future join can have reached resource teardown.
		if cleaned.Load() || tx.closed {
			t.Error("transaction state was torn down before joining its source read")
		}
		r[0].unblock()
		<-done
		if !cleaned.Load() || !tx.closed || r[0].releases.Load() != 1 || r[0].releasedEarly.Load() || v41EngramLookAheadCharge.Load() != 0 {
			t.Fatal("joined abort did not finish cleanup and release exactly once")
		}
	})
}
