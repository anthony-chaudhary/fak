package model

import (
	"errors"
	"math"
	"reflect"
	"runtime"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

type v41RefoldFailureBackend struct {
	compute.Backend
	begins, ends int
}

func (b *v41RefoldFailureBackend) BeginV41StepBatch() bool {
	b.begins++
	return true
}

func (b *v41RefoldFailureBackend) EndV41StepBatch() error {
	b.ends++
	return nil
}

// fak-test:runtime medium est=5s lane=default
func TestV41SuffixRefoldFailureRetiresSession(t *testing.T) {
	t.Setenv(v41StepBatchEnv, "1")
	t.Setenv(v41SuffixUbatchEnv, "")
	t.Setenv(v41SuffixRefoldMaxHistoryEnv, "")
	t.Setenv(v41ActivationTraceEnvVar, "")
	backend, ok := compute.Lookup("cpu-ref")
	if !ok {
		t.Fatal("production cpu-ref backend is not registered")
	}
	type outcome struct {
		returned, unwound bool
		payload           any
	}
	invoke := func(f func()) outcome {
		var out outcome
		done := make(chan struct{})
		go func() {
			defer close(done)
			func() {
				defer func() { out.payload = recover() }()
				f()
				out.returned = true
			}()
			out.unwound = true
		}()
		<-done
		return out
	}
	for _, tc := range []struct {
		name, fault    string
		host, noLogits bool
	}{
		{"returned_error", "error", false, false},
		{"host_error_no_logits", "error", true, true},
		{"panic", "panic", false, false},
		{"goexit_no_logits", "goexit", false, true},
		{"healthy", "", false, false},
		{"healthy_host_no_logits", "", true, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			m := v41IncrementalPlainModel(t, 2)
			defer m.CloseWeights()
			b := &v41RefoldFailureBackend{Backend: backend}
			newSession := func() *Session {
				var s *Session
				if tc.host {
					s = m.NewSession()
				} else {
					var err error
					s, err = m.NewBackendSessionChecked(b)
					if err != nil {
						t.Fatal(err)
					}
				}
				t.Cleanup(s.Close)
				return s
			}
			s := newSession()
			// The ring retains 128 PRIOR rows; the current KV is separate.
			// Positions 127 and 128 commit before position 129 first refuses.
			prefix := make([]int, 127)
			for i := range prefix {
				prefix[i] = i % m.Cfg.VocabSize
			}
			s.Prefill(prefix)
			prior, err := s.PrefixSnapshot()
			if err != nil {
				t.Fatal(err)
			}
			defer prior.Close()
			priorState := prior.v41.clone()
			st := s.v41State()
			if st.layerState(0).windowSize != 128 {
				t.Fatal("fixture must reach the real 128-row full-causal refusal")
			}
			gather := st.gatherOps
			sentinel := errors.New("full refold embedding failed")
			calls, panels, panelHistory, panelCursor := 0, 0, -1, -1
			st.gatherOps = &v41DeviceGatherOps{embedGather: func(ids []int) ([]float32, error) {
				calls++
				if len(ids) > 1 {
					panels++
					panelHistory, panelCursor = len(st.history), st.layerState(0).nextWindowPos
					switch tc.fault {
					case "error":
						return nil, sentinel
					case "panic":
						panic(sentinel)
					case "goexit":
						runtime.Goexit()
					}
				}
				return m.v41EmbeddingPanel(ids)
			}}
			b.begins, b.ends = 0, 0
			suffix := []int{4, 5, 6}
			var logits []float32
			out := invoke(func() {
				if tc.noLogits {
					s.PrefillNoLogits(suffix)
				} else {
					logits = s.Prefill(suffix)
				}
			})
			st.gatherOps = gather
			if calls != 4 || panels != 1 || panelHistory != 130 || panelCursor != 129 {
				t.Fatalf("not a refold after two real commits: calls/panels=%d/%d history/cursor=%d/%d", calls, panels, panelHistory, panelCursor)
			}
			if !tc.host && (b.begins != 4 || b.ends != 4) {
				t.Fatalf("real batch boundaries changed: %d/%d", b.begins, b.ends)
			}
			if out.returned != (tc.fault == "") || out.unwound != (tc.fault != "goexit") {
				t.Fatalf("completion changed: %+v", out)
			}
			if tc.fault == "error" {
				e, ok := out.payload.(error)
				if !ok || !errors.Is(e, sentinel) {
					t.Fatalf("returned error lost original cause: %v", out.payload)
				}
			} else if tc.fault == "panic" {
				if out.payload != sentinel {
					t.Fatalf("original panic changed: %v", out.payload)
				}
			} else if out.payload != nil {
				t.Fatalf("healthy/Goexit became %v", out.payload)
			}
			if tc.fault != "" {
				if !s.BackendSessionClosed() || s.halFailure == nil || logits != nil {
					t.Fatalf("failed refold left reusable mixed state: closed=%v history=%d cursor=%d failure=%v", s.BackendSessionClosed(), len(st.history), st.layerState(0).nextWindowPos, s.halFailure)
				}
				if tc.fault == "error" && !errors.Is(s.halFailure, sentinel) {
					t.Fatalf("sticky refusal lost returned error: %v", s.halFailure)
				}
				for _, retry := range []func(){func() { s.Step(7) }, func() { s.Prefill([]int{7}) }, func() { s.PrefillNoLogits([]int{7}) }} {
					if got := invoke(retry); got.returned || !got.unwound || got.payload != s.halFailure {
						t.Fatalf("retired refold accepted retry: %+v", got)
					}
				}
				for _, snapshot := range []func() (*PrefixSnapshot, error){s.PrefixSnapshot, s.PrefixSnapshotHostOnly} {
					p, err := snapshot()
					if p != nil {
						p.Close()
					}
					if p != nil || err != s.halFailure {
						t.Fatal("failed refold published a reusable prefix")
					}
				}
				if calls != 4 || !tc.host && (b.begins != 4 || b.ends != 4) {
					t.Fatal("retired refold performed new forward work")
				}
			} else {
				ref := newSession()
				want := ref.Prefill(append(append([]int(nil), prefix...), suffix...))
				if s.BackendSessionClosed() || !tc.noLogits && !reflect.DeepEqual(logits, want) ||
					!reflect.DeepEqual(captureV41ForwardSnapshot(st), captureV41ForwardSnapshot(ref.v41Forward)) {
					t.Fatal("healthy refold changed its complete result/state")
				}
				got, want := s.Step(7), ref.Step(7)
				if len(got) != m.Cfg.VocabSize || !reflect.DeepEqual(got, want) {
					t.Fatal("healthy refold cannot continue")
				}
				for _, v := range got {
					if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
						t.Fatal("nonfinite healthy continuation")
					}
				}
			}
			// Retirement belongs only to the failed live owner. An already
			// published independent prefix remains exact and restorable.
			if !reflect.DeepEqual(prior.v41, priorState) || prior.Tokens != len(prefix) {
				t.Fatal("earlier independent prefix was erased")
			}
			branch, control := newSession(), newSession()
			if err := prior.Restore(branch); err != nil {
				t.Fatal(err)
			}
			control.Prefill(prefix)
			if got, want := branch.Step(4), control.Step(4); len(got) != m.Cfg.VocabSize || !reflect.DeepEqual(got, want) {
				t.Fatal("independent earlier prefix lost usable continuation")
			}
		})
	}
}
