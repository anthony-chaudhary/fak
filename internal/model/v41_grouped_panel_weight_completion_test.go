package model

import (
	"errors"
	"io"
	"os"
	"reflect"
	"runtime"
	"testing"
	"time"
)

type v41GroupedWeightCompletionReader struct {
	io.ReaderAt
	before func()
}

func (r *v41GroupedWeightCompletionReader) ReadAt(p []byte, offset int64) (int, error) {
	if r.before != nil {
		r.before()
	}
	return r.ReaderAt.ReadAt(p, offset)
}

// A selected checkpoint read must not become a zero weight or proceed to
// activation work. The real panel caller, lazy payload and residency lock are
// exercised here; the CPU recorder does not establish physical device behavior.
// fak-test:runtime medium est=15s lane=default
func TestV41GroupedPanelWeightCompletion(t *testing.T) {
	v41SelectedRetirementCPU(t)
	for _, mode := range []string{"0", "1"} {
		t.Run("panicnil="+mode, func(t *testing.T) {
			t.Setenv("GODEBUG", os.Getenv("GODEBUG")+",panicnil="+mode)
			for _, fault := range []string{"nil", "nonnil", "goexit", "returned-error", "healthy"} {
				t.Run(fault, func(t *testing.T) {
					m := v41GroupedFixture(t, "Q2_K", true, false)
					c := m.Cfg
					m.kqw[layerName(0, "attn.wo_b.weight")] = q3kFixtureTensor(c.HiddenSize, c.OGroups*c.OLoraRank)
					b := newV41GroupedPanelQuantRecorder()
					s, err := m.NewBackendSessionChecked(b)
					if err != nil {
						t.Fatal(err)
					}
					s.Q4K, s.Quant = m.q4kw != nil, m.q8w != nil
					const rows = 2
					input := v41GroupedPanelInput(rows, c.NumHeads*c.HeadDim)
					want := v41GroupedPanelCacheOracle(t, m, input, rows)
					readers := v41GroupedPanelCacheLazy(m)
					reader := &v41GroupedWeightCompletionReader{ReaderAt: readers[0]}
					m.kqw[layerName(0, "attn.wo_a.weight")].lazy.Reader = reader
					resident := m.immutableDeviceWeights(b)
					if resident == nil {
						t.Fatal("missing model-owned weight cache")
					}
					primary := errors.New("grouped weight checkpoint fault")
					calls, badSeam := 0, false
					reader.before = func() {
						calls++
						if resident.mu.TryLock() {
							resident.mu.Unlock()
							badSeam = true
						}
						if b.uploads != 0 || b.matmuls != 0 || b.gathers != 0 || b.deferredPublishes != 0 || len(resident.weights) != 0 {
							badSeam = true
						}
						switch fault {
						case "nil":
							panic(nil)
						case "nonnil":
							panic(primary)
						case "goexit":
							runtime.Goexit()
						}
					}
					if fault == "returned-error" {
						readers[0].err = primary
					}
					joined := false
					t.Cleanup(func() {
						if !joined {
							return // A timed-out worker still owns the fixture.
						}
						reader.before, readers[0].err = nil, nil
						s.Close()
						if err := m.CloseWeights(); err != nil {
							t.Errorf("joined model cleanup: %v", err)
						}
						for handle, created := range b.created {
							if b.freed[handle] != created {
								t.Errorf("weight fixture retirement: created=%d freed=%d", created, b.freed[handle])
							}
						}
					})
					var got [][]float32
					var outcome v41DenseProjectionOutcome
					var cause error
					var recovered any
					returned, boundaryReturned := false, false
					done := make(chan struct{})
					go func() {
						defer close(done)
						func() {
							defer func() { recovered = recover() }()
							got, outcome, cause = v41GroupedPanelCacheCall(s, input, rows)
							returned = true
						}()
						boundaryReturned = true
					}()
					select {
					case <-done:
						joined = true
					case <-time.After(5 * time.Second):
						t.Fatal("grouped weight worker did not join; teardown withheld")
					}
					diagnostic := func() {
						t.Fatalf("grouped weight fault=%q returned=%t boundaryReturned=%t rows=%d cause=%v panic=%v reads=%d uploads=%d matmuls=%d", fault, returned, boundaryReturned, len(got), cause, recovered, calls, b.uploads, b.matmuls)
					}
					if calls != 1 || badSeam {
						diagnostic()
					}
					if !resident.mu.TryLock() {
						t.Fatal("weight acquisition retained the model cache lock")
					}
					cached := len(resident.weights)
					resident.mu.Unlock()
					if fault == "healthy" {
						if !returned || !boundaryReturned || cause != nil || recovered != nil || outcome != v41ProjectionHandled || !reflect.DeepEqual(got, want) || len(got) != rows || len(got[0]) != c.HiddenSize {
							diagnostic()
						}
						if cached != c.OGroups+1 || readers[0].calls != 1 || readers[1].calls != 1 {
							t.Fatal("healthy weight staging did not publish each immutable handle once")
						}
						reader.before = nil
						again, next, err := v41GroupedPanelCacheCall(s, input, rows)
						if err != nil || next != v41ProjectionHandled || !reflect.DeepEqual(again, want) || readers[0].calls != 1 || readers[1].calls != 1 {
							t.Fatal("healthy cache hit reread or changed the panel")
						}
						return
					}
					if got != nil || cached != 0 || len(s.halW) != 0 || b.uploads != 0 || b.matmuls != 0 || b.gathers != 0 || b.deferredPublishes != 0 || readers[1].calls != 0 {
						diagnostic()
					}
					switch fault {
					case "nil":
						if returned || !boundaryReturned || cause != nil {
							diagnostic()
						}
						if mode == "1" {
							if recovered != nil {
								diagnostic()
							}
						} else if _, ok := recovered.(*runtime.PanicNilError); !ok {
							diagnostic()
						}
					case "nonnil":
						if returned || !boundaryReturned || recovered != primary || cause != nil {
							diagnostic()
						}
					case "goexit":
						if returned || boundaryReturned || recovered != nil || cause != nil {
							diagnostic()
						}
						return
					case "returned-error":
						if !returned || !boundaryReturned || recovered != nil || outcome != v41ProjectionError || !errors.Is(cause, primary) {
							diagnostic()
						}
					}
					var selected *BackendForwardOperationError
					if !s.BackendSessionClosed() || !errors.As(s.halFailure, &selected) || selected.Path != "v41-grouped-output-panel-weights" || selected.Stage != "weight upload" {
						t.Fatalf("wrong selected failure: %v", s.halFailure)
					}
					if fault != "nil" && !errors.Is(s.halFailure, primary) {
						t.Fatal("selected failure lost its original cause")
					}
					reader.before, readers[0].err = nil, nil
					var retry any
					func() {
						defer func() { retry = recover() }()
						v41GroupedPanelCacheCall(s, input, rows)
					}()
					if retry != s.halFailure || b.uploads != 0 || readers[0].calls > 1 || readers[1].calls != 0 {
						t.Fatal("closed selected weight acquisition retried")
					}
				})
			}
		})
	}
}
