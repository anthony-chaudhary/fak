package model

import (
	"errors"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

type v41RoutedAbortRecorder struct {
	*v41RoutedIDTestBackend
	failure                  error
	panicValue               any
	trackedStatuses          map[compute.Buffer]bool
	trackedOutputs           map[compute.Buffer]bool
	statusFrees, outputFrees int
}

func (b *v41RoutedAbortRecorder) track(out compute.Tensor, st compute.V41DeferredStatus, err error) {
	if err == nil {
		b.trackedOutputs[out.Buf()], b.trackedStatuses[st.Record.Buf()] = true, true
	}
}
func (b *v41RoutedAbortRecorder) V41RoutedSwiGLUDeferred(g, u compute.Tensor, limit, weight float32) (compute.Tensor, compute.V41DeferredStatus, error) {
	out, st, err := b.v41RoutedIDTestBackend.V41RoutedSwiGLUDeferred(g, u, limit, weight)
	b.track(out, st, err)
	return out, st, err
}
func (b *v41RoutedAbortRecorder) V41RoutedSwiGLUIDDeferred(g, u, ids compute.Tensor, picks int, limit float32) (compute.Tensor, compute.V41DeferredStatus, error) {
	out, st, err := b.v41RoutedIDTestBackend.V41RoutedSwiGLUIDDeferred(g, u, ids, picks, limit)
	b.track(out, st, err)
	return out, st, err
}
func (b *v41RoutedAbortRecorder) V41BF16PublishDeferred(x compute.Tensor, site uint32) (compute.Tensor, compute.V41DeferredStatus, error) {
	if b.panicValue != nil {
		panic(b.panicValue)
	}
	if b.failure != nil {
		return compute.Tensor{}, compute.V41DeferredStatus{}, b.failure
	}
	return b.v41RoutedIDTestBackend.V41BF16PublishDeferred(x, site)
}
func (b *v41RoutedAbortRecorder) Free(x compute.Tensor) {
	if b.trackedStatuses[x.Buf()] {
		b.statusFrees++
	}
	if b.trackedOutputs[x.Buf()] {
		b.outputFrees++
	}
	b.v41RoutedIDTestBackend.Free(x)
}

type v41RoutedCancelRecorder struct {
	*v41RoutedAbortRecorder
	cancels, accepted int
	primary           error
	cleanupPanic      bool
	complete          bool
}

func (b *v41RoutedCancelRecorder) V41CancelDeferred(statuses []compute.V41DeferredStatus, primary error) (bool, error) {
	b.cancels++
	b.primary = primary
	for _, st := range statuses {
		if !b.trackedStatuses[st.Record.Buf()] {
			panic("cancel received untracked ownership")
		}
		b.accepted++
		delete(b.pending, st.Record.Buf())
		for _, part := range b.multiPending[st.Record.Buf()] {
			delete(b.pending, part.Record.Buf())
		}
		delete(b.multiPending, st.Record.Buf())
	}
	if b.cleanupPanic {
		panic("cleanup must not replace operation panic")
	}
	return b.complete, primary
}

// fak-test:runtime fast est=3s lane=default
// Runs the actual two model paths; host recorders prove ownership/wiring only.
func TestV41RoutedAbortCancellationOwnership(t *testing.T) {
	t.Setenv(v41LayerExpertBatchEnv, "1")
	t.Setenv(v41RoutedIDEnv, "0")
	t.Setenv(v41RoutedFusedEnv, "0")
	t.Setenv(v41PrefillExpertPanelEnv, "1")
	for _, tc := range []struct {
		name                                                string
		panel, explicit, readFailure, panicFailure, unknown bool
	}{
		{name: "decode-cancel", explicit: true},
		{name: "panel-cancel", panel: true, explicit: true},
		{name: "unknown-completion", explicit: true, unknown: true},
		{name: "decode-legacy"},
		{name: "panel-legacy", panel: true},
		{name: "read-owns-status", explicit: true, readFailure: true},
		{name: "original-panic", explicit: true, panicFailure: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			primary := errors.New("selected routed failure")
			recorder := &v41RoutedAbortRecorder{v41RoutedIDTestBackend: newV41RoutedIDTestBackend(), failure: primary,
				trackedStatuses: map[compute.Buffer]bool{}, trackedOutputs: map[compute.Buffer]bool{}}
			if tc.readFailure {
				recorder.failure = nil
				recorder.readFailure = primary
			}
			if tc.panicFailure {
				recorder.failure = nil
				recorder.panicValue = primary
			}
			canceler := &v41RoutedCancelRecorder{v41RoutedAbortRecorder: recorder, cleanupPanic: tc.panicFailure, complete: !tc.unknown}
			var backend compute.Backend = recorder
			if tc.explicit {
				backend = canceler
			}
			m := v41IncrementalExpertFixture(t, true, false)
			s, err := m.NewBackendSessionChecked(backend)
			if err != nil {
				t.Fatal(err)
			}
			defer s.Close()
			picks, x := v41RoutedLayerFaultPicks(m)
			var cause error
			var outcome v41RoutedExpertOutcome
			var panicked any
			func() {
				defer func() { panicked = recover() }()
				if tc.panel {
					_, outcome, cause = s.v41RoutedExpertPanel(0, "ffn.experts.0", [][]float32{x, x}, []float32{1, 1}, nil)
				} else {
					fn := s.v41RoutedLayerFunc()
					if fn == nil {
						t.Fatal("fixture did not bind routed layer")
					}
					_, outcome, cause = fn(0, picks, x)
				}
			}()
			if len(recorder.trackedStatuses) == 0 {
				t.Fatal("fixture never issued a tracked status")
			}
			if tc.panicFailure {
				if panicked != primary {
					t.Fatalf("original panic replaced: %v", panicked)
				}
			} else if panicked != nil || outcome != v41RoutedExpertError || !errors.Is(cause, primary) {
				t.Fatalf("primary failure changed: outcome=%v cause=%v panic=%v", outcome, cause, panicked)
			}
			if tc.readFailure {
				if canceler.cancels != 0 || recorder.deferredReads != 1 {
					t.Fatal("read ownership was canceled a second time")
				}
			} else if tc.explicit {
				if canceler.cancels != 1 || canceler.accepted != len(recorder.trackedStatuses) || recorder.statusFrees != 0 {
					t.Fatalf("cancel ownership: calls=%d accepted=%d status frees=%d output frees=%d", canceler.cancels, canceler.accepted, recorder.statusFrees, recorder.outputFrees)
				}
				if tc.unknown || tc.panicFailure {
					if recorder.outputFrees != 0 {
						t.Fatal("unknown completion freed caller outputs")
					}
				} else if recorder.outputFrees != len(recorder.trackedOutputs) {
					t.Fatal("proven completion did not release caller outputs once")
				}
				if !tc.panicFailure && canceler.primary != cause {
					t.Fatal("canceler lost exact attributed primary")
				}
			} else if recorder.statusFrees == 0 || recorder.outputFrees == 0 {
				t.Fatal("legacy no-canceler cleanup changed")
			}
		})
	}
}
