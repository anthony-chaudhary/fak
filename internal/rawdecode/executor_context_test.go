package rawdecode

import (
	"context"
	"errors"
	"fmt"
	"os"
	"runtime"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/model"
)

type contextSession struct {
	name string
	call func(string)
}

func (s *contextSession) Prefill([]int) []float32 {
	s.call(s.name + " prefill")
	return []float32{0, 2, 1}
}

func (s *contextSession) Step(int) []float32 {
	s.call(s.name + " step")
	return []float32{0, 2, 1}
}

func (s *contextSession) Close() { s.call(s.name + " close") }

type contextLoadedModel struct {
	fakeLoadedModel
	call func(string)
	eos  func(int) bool
}

func (m *contextLoadedModel) IsEOS(token int) bool {
	return m.eos != nil && m.eos(token)
}

func (m *contextLoadedModel) NewCandidateSession(compute.Backend, Request) (session, error) {
	m.call("candidate create")
	return &contextSession{name: "candidate", call: m.call}, nil
}

func (m *contextLoadedModel) NewCPUSession(Request) session {
	m.call("cpu create")
	return &contextSession{name: "cpu", call: m.call}
}

type contextBackend struct {
	rawUnsupportedBackend
	call   func(string)
	endErr error
}

func (b *contextBackend) BeginBackendExecutionWindow() (compute.BackendExecutionWindow, error) {
	b.call("begin")
	return contextWindow{backend: b}, nil
}

type contextWindow struct{ backend *contextBackend }

func (w contextWindow) End() (compute.BackendExecutionObservation, error) {
	w.backend.call("end")
	return compute.BackendExecutionObservation{
		Identity: compute.BackendRuntimeIdentity{
			Backend: "context-test", Device: "fake", Driver: "fake", Runtime: "fake",
		},
		TransferCountersObserved: true,
		DeviceMemoryObserved:     true,
		DeviceMemoryTotalBytes:   1,
		DeviceAllocationObserved: true,
	}, w.backend.endErr
}

// fak-test:runtime fast est=10ms
func TestExecuteModelContextLifecycle(t *testing.T) {
	t.Parallel()
	t.Run("public API canceled before session", func(t *testing.T) {
		ctx, cancel := context.WithCancel(context.Background())
		cancel()
		// An empty model cannot execute: cancellation must avoid constructing a
		// real Session or allocating weights even through the exported seam.
		got, err := ExecuteModelContext(ctx, Request{}, &model.Model{}, nil)
		if !errors.Is(err, context.Canceled) || len(got.Runs) != 0 {
			t.Fatalf("runs=%d err=%v", len(got.Runs), err)
		}
	})

	for _, tc := range []struct {
		name        string
		cancelAt    string
		verifyCPU   bool
		repetitions int
		want        string
	}{
		{"before repetition", "before", false, 1, ""},
		{"before candidate session", "begin", false, 1, "begin,end"},
		{"before prefill", "candidate create", false, 1, "begin,candidate create,candidate close,end"},
		{"during prefill", "candidate prefill", false, 1, "begin,candidate create,candidate prefill,candidate close,end"},
		{"before step", "before step", false, 1, "begin,candidate create,candidate prefill,candidate close,end"},
		{"during EOS selection", "eos", false, 1, "begin,candidate create,candidate prefill,candidate close,end"},
		{"during final step", "candidate step", false, 1, "begin,candidate create,candidate prefill,candidate step,candidate close,end"},
		{"before CPU session", "candidate close", true, 1, "begin,candidate create,candidate prefill,candidate step,candidate close,end"},
		{"before CPU prefill", "cpu create", true, 1, "begin,candidate create,candidate prefill,candidate step,candidate close,cpu create,cpu close,end"},
		{"during CPU prefill", "cpu prefill", true, 1, "begin,candidate create,candidate prefill,candidate step,candidate close,cpu create,cpu prefill,cpu close,end"},
		{"during CPU final step", "cpu step", true, 1, "begin,candidate create,candidate prefill,candidate step,candidate close,cpu create,cpu prefill,cpu step,cpu close,end"},
		{"during final cleanup", "candidate close", false, 1, "begin,candidate create,candidate prefill,candidate step,candidate close,end"},
		{"before next repetition", "end", false, 2, "begin,candidate create,candidate prefill,candidate step,candidate close,end"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			if tc.cancelAt == "before" {
				cancel()
			}
			var events []string
			call := func(event string) {
				events = append(events, event)
				if event == tc.cancelAt {
					cancel()
				}
			}
			m := &contextLoadedModel{call: call}
			if tc.cancelAt == "before step" || tc.cancelAt == "eos" {
				m.eos = func(int) bool { cancel(); return tc.cancelAt == "eos" }
			}
			be := &contextBackend{rawUnsupportedBackend: rawUnsupportedBackend{name: "context-test"}, call: call}
			req := Request{PromptTokenIDs: []int{0}, ContextLimit: 3, GeneratedTokenLimit: 2, Repetitions: tc.repetitions, VerifyCPU: tc.verifyCPU}
			got, err := executeLoaded(ctx, req, m, be, nil, fakeClock(), time.Since)
			if !errors.Is(err, context.Canceled) || len(got.Runs) != 0 {
				t.Fatalf("canceled run published: runs=%d err=%v", len(got.Runs), err)
			}
			if _, ok := got.GenerationObservation(0); ok {
				t.Fatal("canceled run gained generation evidence")
			}
			if _, ok := got.TimingObservation(0); ok {
				t.Fatal("canceled run gained timing evidence")
			}
			if got := strings.Join(events, ","); got != tc.want {
				t.Fatalf("events=%s, want %s", got, tc.want)
			}
			if m.closeCalls != 0 {
				t.Fatal("closed caller-owned model")
			}
		})
	}

	t.Run("primary error survives cleanup panic and error", func(t *testing.T) {
		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()
		var events []string
		call := func(event string) {
			events = append(events, event)
			switch event {
			case "candidate prefill":
				cancel()
			case "candidate close":
				panic("close failed")
			}
		}
		endErr := errors.New("end failed")
		be := &contextBackend{rawUnsupportedBackend: rawUnsupportedBackend{name: "context-test"}, call: call, endErr: endErr}
		req := Request{PromptTokenIDs: []int{0}, ContextLimit: 2, GeneratedTokenLimit: 1, Repetitions: 1}
		got, err := executeLoaded(ctx, req, &contextLoadedModel{call: call}, be, nil, fakeClock(), time.Since)
		if !errors.Is(err, context.Canceled) || !errors.Is(err, endErr) || !strings.Contains(err.Error(), "close failed") || len(got.Runs) != 0 {
			t.Fatalf("lost primary or cleanup failure: runs=%d err=%v", len(got.Runs), err)
		}
		want := []string{"begin", "candidate create", "candidate prefill", "candidate close", "end"}
		if !slices.Equal(events, want) {
			t.Fatalf("events=%v, want %v", events, want)
		}
	})

	for _, primaryAt := range []string{"candidate create", "candidate prefill", "candidate step", "candidate close", "cpu prefill", "cpu step", "cpu close", "end"} {
		t.Run("panic at "+primaryAt, func(t *testing.T) {
			primary := &struct{ phase string }{primaryAt}
			var events []string
			call := func(event string) {
				events = append(events, event)
				if event == primaryAt {
					panic(primary)
				}
				// Panic in both cleanups as well: neither may replace the
				// original panic or stop the remaining cleanup from running.
				secondaryClose := event == "candidate close" && strings.HasPrefix(primaryAt, "candidate") || event == "cpu close" && strings.HasPrefix(primaryAt, "cpu")
				if secondaryClose || event == "end" && primaryAt != "end" {
					panic("secondary cleanup panic")
				}
			}
			be := &contextBackend{rawUnsupportedBackend: rawUnsupportedBackend{name: "context-test"}, call: call}
			req := Request{PromptTokenIDs: []int{0}, ContextLimit: 3, GeneratedTokenLimit: 2, Repetitions: 1, VerifyCPU: strings.HasPrefix(primaryAt, "cpu")}
			var observed any
			func() {
				defer func() { observed = recover() }()
				_, _ = executeLoaded(context.Background(), req, &contextLoadedModel{call: call}, be, nil, fakeClock(), time.Since)
			}()
			if observed != primary {
				t.Fatalf("panic=%v, want original %p (%s)", observed, primary, primaryAt)
			}
			want := []string{"begin", "candidate create"}
			if primaryAt != "candidate create" {
				want = append(want, "candidate prefill")
				if primaryAt != "candidate prefill" {
					want = append(want, "candidate step")
				}
				want = append(want, "candidate close")
			}
			if req.VerifyCPU {
				want = append(want, "cpu create", "cpu prefill")
				if primaryAt != "cpu prefill" {
					want = append(want, "cpu step")
				}
				want = append(want, "cpu close")
			}
			want = append(want, "end")
			if !slices.Equal(events, want) {
				t.Fatalf("events=%v, want %v", events, want)
			}
		})
	}

	t.Run("successful boundaries and ownership", func(t *testing.T) {
		var events []string
		call := func(event string) { events = append(events, event) }
		m := &contextLoadedModel{call: call}
		be := &contextBackend{rawUnsupportedBackend: rawUnsupportedBackend{name: "context-test"}, call: call}
		req := Request{PromptTokenIDs: []int{0}, ContextLimit: 3, GeneratedTokenLimit: 2, Repetitions: 1, VerifyCPU: true}
		got, err := executeLoaded(context.Background(), req, m, be, nil, fakeClock(), time.Since)
		if err != nil || len(got.Runs) != 1 || m.closeCalls != 0 {
			t.Fatalf("runs=%d model closes=%d err=%v", len(got.Runs), m.closeCalls, err)
		}
		want := "[begin candidate create candidate prefill candidate step candidate close cpu create cpu prefill cpu step cpu close end]"
		if fmt.Sprint(events) != want {
			t.Fatalf("events=%v, want %s", events, want)
		}
		for i, boundary := range got.Runs[0].timing.boundaries {
			if want := time.Unix(100, 0).Add(time.Duration(i+1) * time.Millisecond); boundary != want {
				t.Fatalf("boundary %d=%v, want %v", i, boundary, want)
			}
		}
	})
}

type executionCleanupOutcome struct {
	execution        Execution
	err              error
	panicValue       any
	returned         bool
	boundaryReturned bool
}

func observeExecutionCleanup(call func() (Execution, error)) executionCleanupOutcome {
	var out executionCleanupOutcome
	done := make(chan struct{})
	go func() {
		defer close(done)
		func() {
			defer func() { out.panicValue = recover() }()
			out.execution, out.err = call()
			out.returned = true
		}()
		out.boundaryReturned = true
	}()
	<-done
	return out
}

func requireExecutionCleanupUnwind(t *testing.T, out executionCleanupOutcome, mode, fault string, primary any) {
	t.Helper()
	if out.returned || out.err != nil || len(out.execution.Runs) != 0 {
		t.Fatalf("incomplete execution returned or assigned a result: %+v", out)
	}
	if _, ok := out.execution.GenerationObservation(0); ok {
		t.Fatal("incomplete execution gained generation evidence")
	}
	if _, ok := out.execution.TimingObservation(0); ok {
		t.Fatal("incomplete execution gained timing evidence")
	}
	if out.boundaryReturned != (fault != "goexit") {
		t.Fatalf("recovery boundary returned=%t, fault=%s", out.boundaryReturned, fault)
	}
	switch fault {
	case "panic":
		if out.panicValue != primary {
			t.Fatalf("panic=%v, want original %v", out.panicValue, primary)
		}
	case "nil":
		if mode == "0" {
			if _, ok := out.panicValue.(*runtime.PanicNilError); !ok {
				t.Fatalf("panic=%T, want modern nil-panic payload", out.panicValue)
			}
		} else if out.panicValue != nil {
			t.Fatalf("panic=%v, want legacy nil payload", out.panicValue)
		}
	case "goexit":
		if out.panicValue != nil {
			t.Fatalf("Goexit acquired a panic payload: %v", out.panicValue)
		}
	}
}

func executionCleanupFault(fault string, primary any) {
	switch fault {
	case "nil":
		panic(nil)
	case "panic":
		panic(primary)
	case "goexit":
		runtime.Goexit()
	}
}

// fak-test:runtime fast est=10ms
func TestExecuteModelContextCleanupCompletion(t *testing.T) {
	for _, mode := range []string{"0", "1"} {
		t.Run("panicnil="+mode, func(t *testing.T) {
			t.Setenv("GODEBUG", os.Getenv("GODEBUG")+",panicnil="+mode)
			for _, phase := range []string{"candidate close", "cpu close", "end"} {
				for _, fault := range []string{"nil", "goexit"} {
					t.Run(phase+"/"+fault, func(t *testing.T) {
						var events []string
						call := func(event string) {
							events = append(events, event)
							if event == phase {
								executionCleanupFault(fault, nil)
							}
						}
						m := &contextLoadedModel{call: call}
						be := &contextBackend{rawUnsupportedBackend: rawUnsupportedBackend{name: "context-test"}, call: call}
						req := Request{PromptTokenIDs: []int{0}, ContextLimit: 2, GeneratedTokenLimit: 1, Repetitions: 1, VerifyCPU: phase == "cpu close"}
						out := observeExecutionCleanup(func() (Execution, error) {
							return executeLoaded(context.Background(), req, m, be, nil, fakeClock(), time.Since)
						})
						requireExecutionCleanupUnwind(t, out, mode, fault, nil)
						want := []string{"begin", "candidate create", "candidate prefill", "candidate close"}
						if req.VerifyCPU {
							want = append(want, "cpu create", "cpu prefill", "cpu close")
						}
						want = append(want, "end")
						if !slices.Equal(events, want) || m.closeCalls != 0 {
							t.Fatalf("events=%v, want %v; model closes=%d", events, want, m.closeCalls)
						}
					})
				}
			}
			for _, returnedError := range []bool{false, true} {
				t.Run(fmt.Sprintf("secondary nil cleanup/returned error=%t", returnedError), func(t *testing.T) {
					ctx, cancel := context.WithCancel(context.Background())
					defer cancel()
					var events []string
					call := func(event string) {
						events = append(events, event)
						switch event {
						case "candidate prefill":
							if returnedError {
								cancel()
							} else {
								panic(nil)
							}
						case "candidate close", "end":
							panic(nil)
						}
					}
					m := &contextLoadedModel{call: call}
					be := &contextBackend{rawUnsupportedBackend: rawUnsupportedBackend{name: "context-test"}, call: call}
					req := Request{PromptTokenIDs: []int{0}, ContextLimit: 2, GeneratedTokenLimit: 1, Repetitions: 1}
					out := observeExecutionCleanup(func() (Execution, error) {
						return executeLoaded(ctx, req, m, be, nil, fakeClock(), time.Since)
					})
					if returnedError {
						if !out.returned || !out.boundaryReturned || out.panicValue != nil || !errors.Is(out.err, context.Canceled) || len(out.execution.Runs) != 0 {
							t.Fatalf("lost ordinary error return: %+v", out)
						}
						for _, label := range []string{"raw decode: close candidate session: panic:", "raw decode: end backend observation for run 1: panic:"} {
							if !strings.Contains(out.err.Error(), label) {
								t.Fatalf("cleanup label %q missing from %v", label, out.err)
							}
						}
					} else {
						requireExecutionCleanupUnwind(t, out, mode, "nil", nil)
					}
					want := []string{"begin", "candidate create", "candidate prefill", "candidate close", "end"}
					if !slices.Equal(events, want) || m.closeCalls != 0 {
						t.Fatalf("events=%v, want %v; model closes=%d", events, want, m.closeCalls)
					}
				})
			}
		})
	}
}

type executionCleanupBackend struct {
	compute.Backend
	call    func(string)
	kvFreed bool
}

func (b *executionCleanupBackend) NewKV(cfg compute.KVConfig) compute.KVStore {
	return &executionCleanupKV{KVStore: b.Backend.NewKV(cfg), cfg: cfg, backend: b}
}

func (b *executionCleanupBackend) Recycle() {
	if b.kvFreed {
		// Token execution also recycles; only KV Free identifies Close here.
		b.call("close recycle")
	}
}

func (b *executionCleanupBackend) BeginBackendExecutionWindow() (compute.BackendExecutionWindow, error) {
	b.call("begin")
	return executionCleanupWindow{backend: b}, nil
}

type executionCleanupKV struct {
	compute.KVStore
	cfg     compute.KVConfig
	backend *executionCleanupBackend
}

func (k *executionCleanupKV) KVConfig() compute.KVConfig { return k.cfg }

func (k *executionCleanupKV) Free() {
	k.KVStore.Free()
	k.backend.kvFreed = true
	k.backend.call("kv free")
}

type executionCleanupWindow struct{ backend *executionCleanupBackend }

func (w executionCleanupWindow) End() (compute.BackendExecutionObservation, error) {
	w.backend.call("end")
	return compute.BackendExecutionObservation{
		Identity: compute.BackendRuntimeIdentity{
			Backend: w.backend.Name(), Device: "test", Driver: "test", Runtime: "test",
		},
		TransferCountersObserved: true,
		DeviceMemoryObserved:     true,
		DeviceMemoryTotalBytes:   1,
		DeviceAllocationObserved: true,
	}, nil
}

// fak-test:runtime fast est=30ms
func TestExecuteModelContextPublicCleanupCompletion(t *testing.T) {
	for _, mode := range []string{"0", "1"} {
		t.Run("panicnil="+mode, func(t *testing.T) {
			t.Setenv("GODEBUG", os.Getenv("GODEBUG")+",panicnil="+mode)
			for _, tc := range []struct{ phase, fault string }{
				{"close recycle", "nil"}, {"close recycle", "panic"}, {"close recycle", "goexit"},
				{"end", "nil"}, {"end", "panic"}, {"end", "goexit"}, {"healthy", ""},
			} {
				t.Run(tc.phase+"/"+tc.fault, func(t *testing.T) {
					m := model.NewSynthetic(model.Config{
						ModelType: "llama", HiddenSize: 16, NumLayers: 1, NumHeads: 4,
						NumKVHeads: 2, HeadDim: 4, IntermediateSize: 32, VocabSize: 16,
						RMSNormEps: 1e-5, RopeTheta: 10000, EOSTokenID: 15,
					})
					primary := errors.New("injected public cleanup panic")
					var events []string
					be := &executionCleanupBackend{Backend: compute.Pick("cpu-ref"), call: func(event string) {
						events = append(events, event)
						if event == tc.phase {
							executionCleanupFault(tc.fault, primary)
						}
					}}
					if be.Backend == nil || be.Name() != "cpu-ref" {
						t.Fatal("real cpu-ref backend unavailable")
					}
					req := Request{PromptTokenIDs: []int{1}, ContextLimit: 2, GeneratedTokenLimit: 1, Repetitions: 1, IgnoreEOS: true, VerifyCPU: true}
					out := observeExecutionCleanup(func() (Execution, error) {
						return ExecuteModelContext(context.Background(), req, m, be)
					})
					if tc.fault == "" {
						if !out.returned || !out.boundaryReturned || out.panicValue != nil || out.err != nil || len(out.execution.Runs) != 1 {
							t.Fatalf("healthy public execution failed: %+v", out)
						}
						if out.execution.CPUModelParity == nil || !*out.execution.CPUModelParity || out.execution.Runs[0].BackendExecution == nil {
							t.Fatal("healthy run lost CPU parity or completed observation")
						}
						if _, ok := out.execution.GenerationObservation(0); !ok {
							t.Fatal("healthy run lost generation evidence")
						}
						if _, ok := out.execution.TimingObservation(0); !ok {
							t.Fatal("healthy run lost timing evidence")
						}
					} else {
						requireExecutionCleanupUnwind(t, out, mode, tc.fault, primary)
					}
					// These are directly observed attempts. An interrupted Recycle
					// does not prove that all later model retirement work completed.
					want := []string{"begin", "kv free", "close recycle", "end"}
					if !slices.Equal(events, want) {
						t.Fatalf("public cleanup events=%v, want %v", events, want)
					}
				})
			}
		})
	}
}
