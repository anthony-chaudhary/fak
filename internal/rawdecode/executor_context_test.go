package rawdecode

import (
	"context"
	"errors"
	"fmt"
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
