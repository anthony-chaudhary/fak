package model

import (
	"errors"
	"math"
	"os"
	"reflect"
	"runtime"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// The wrapper observes the real public batch boundary; tensor/KV operations
// remain the CPU-reference implementation and forward math remains production.
type v41UbatchHistoryBackend struct {
	compute.Backend
	begins, ends int
}

func (b *v41UbatchHistoryBackend) BeginV41StepBatch() bool {
	b.begins++
	return true
}

func (b *v41UbatchHistoryBackend) EndV41StepBatch() error {
	b.ends++
	return nil
}

// fak-test:runtime medium est=5s lane=default
func TestV41SuffixUbatchConstructionRestoresHistory(t *testing.T) {
	t.Setenv(v41StepBatchEnv, "1")
	t.Setenv(v41SuffixUbatchEnv, "2")
	t.Setenv(v41SuffixRefoldMaxHistoryEnv, "")
	t.Setenv(v41ActivationTraceEnvVar, "")
	backend, ok := compute.Lookup("cpu-ref")
	if !ok {
		t.Fatal("production cpu-ref backend is not registered")
	}
	for _, tc := range []struct {
		name, fault string
		noLogits    bool
		laterChunk  bool
	}{
		{"panic", "panic", false, false},
		{"goexit", "goexit", false, false},
		{"legacy_nil_no_logits", "nil", true, false},
		{"error_no_logits", "error", true, false},
		{"healthy", "", false, false},
		{"healthy_no_logits", "", true, false},
		{"later_chunk_panic", "panic", false, true},
		{"later_chunk_goexit_no_logits", "goexit", true, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if tc.fault == "nil" {
				t.Setenv("GODEBUG", os.Getenv("GODEBUG")+",panicnil=1")
			}
			m := v41IncrementalPlainModel(t, 2)
			defer m.CloseWeights()
			m.Cfg.Window = []int{64, 64}
			newSession := func() (*Session, *v41UbatchHistoryBackend) {
				b := &v41UbatchHistoryBackend{Backend: backend}
				s, err := m.NewBackendSessionChecked(b)
				if err != nil {
					t.Fatal(err)
				}
				t.Cleanup(s.Close)
				return s, b
			}
			s, b := newSession()
			ref, _ := newSession()
			prefix := []int{1, 2, 3}
			s.Prefill(prefix)
			ref.Prefill(prefix)
			sourcePrefix, err := s.PrefixSnapshot()
			if err != nil {
				t.Fatal(err)
			}
			defer sourcePrefix.Close()
			sourceState := sourcePrefix.v41.clone()
			suffix := []int{4, 5, 6}
			committed, target := 0, 2
			if tc.laterChunk {
				// The public partition is 513+512. Fail construction of the
				// second row in chunk two, after chunk one genuinely committed.
				suffix = make([]int, v41SuffixUbatchMaxRows+1)
				for i := range suffix {
					suffix[i] = (i + 4) % m.Cfg.VocabSize
				}
				committed = (len(suffix) + 1) / 2
				target += committed
				ref.PrefillNoLogits(suffix[:committed])
			}
			var wantLogits []float32
			if tc.fault == "" {
				if tc.noLogits {
					ref.PrefillNoLogits(suffix)
				} else {
					wantLogits = ref.Prefill(suffix)
				}
			}
			wantState := captureV41ForwardSnapshot(ref.v41Forward)
			st := s.v41State()
			priorGather := st.gatherOps
			sentinel := errors.New("second ubatch embedding did not complete")
			calls, faultHistory, faultCursor := 0, -1, -1
			st.gatherOps = &v41DeviceGatherOps{embedGather: func(ids []int) ([]float32, error) {
				calls++
				if tc.fault != "" && calls == target {
					faultHistory = len(st.history)
					faultCursor = st.layerState(0).nextWindowPos
					switch tc.fault {
					case "panic":
						panic(sentinel)
					case "nil":
						panic(nil)
					case "goexit":
						runtime.Goexit()
					case "error":
						return nil, sentinel
					}
				}
				return m.v41EmbeddingPanel(ids)
			}}
			b.begins, b.ends = 0, 0
			var logits []float32
			var payload any
			returned, unwound := false, false
			done := make(chan struct{})
			go func() {
				defer close(done)
				func() {
					defer func() { payload = recover() }()
					if tc.noLogits {
						s.PrefillNoLogits(suffix)
					} else {
						logits = s.Prefill(suffix)
					}
					returned = true
				}()
				unwound = true
			}()
			<-done
			st.gatherOps = priorGather
			if returned != (tc.fault == "") || unwound != (tc.fault != "goexit") {
				t.Fatalf("completion changed: returned=%v unwound=%v payload=%v", returned, unwound, payload)
			}
			switch tc.fault {
			case "panic":
				if payload != sentinel {
					t.Fatalf("original panic changed: %v", payload)
				}
			case "error":
				e, ok := payload.(error)
				if !ok || !errors.Is(e, sentinel) {
					t.Fatalf("original returned error changed: %v", payload)
				}
			default:
				if payload != nil {
					t.Fatalf("nil/Goexit/healthy outcome became %v", payload)
				}
			}
			wantBatches := 1
			if tc.laterChunk {
				wantBatches = 2
			}
			if b.begins != wantBatches || b.ends != wantBatches || s.BackendSessionClosed() {
				t.Fatalf("healthy End changed: begin/end=%d/%d closed=%v", b.begins, b.ends, s.BackendSessionClosed())
			}
			if tc.fault != "" {
				if calls != target || faultHistory != len(prefix)+committed+1 || faultCursor != len(prefix)+committed || logits != nil {
					t.Fatalf("not a second-row construction fault: calls=%d history/cursor=%d/%d logits=%d", calls, faultHistory, faultCursor, len(logits))
				}
			} else if calls != len(suffix) || !reflect.DeepEqual(logits, wantLogits) {
				t.Fatalf("healthy ubatch changed: calls=%d logits=%d", calls, len(logits))
			}
			if got := captureV41ForwardSnapshot(st); !reflect.DeepEqual(got, wantState) {
				t.Fatalf("construction leaked uncommitted history/state: history=%v want=%v", st.history, wantState.history)
			}
			post, err := s.PrefixSnapshot()
			if err != nil {
				t.Fatal(err)
			}
			defer post.Close()
			if post.Tokens != len(wantState.history) || !reflect.DeepEqual(post.v41, wantState) {
				t.Fatal("published prefix does not match the last committed boundary")
			}
			got, want := s.Step(7), ref.Step(7)
			if len(got) != m.Cfg.VocabSize || !reflect.DeepEqual(got, want) {
				t.Fatal("real Step after construction unwind diverged")
			}
			for _, value := range got {
				if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
					t.Fatal("nonfinite continuation logits")
				}
			}
			if !reflect.DeepEqual(captureV41ForwardSnapshot(st), captureV41ForwardSnapshot(ref.v41Forward)) ||
				!reflect.DeepEqual(sourcePrefix.v41, sourceState) || !reflect.DeepEqual(post.v41, wantState) {
				t.Fatal("continuation state diverged or an independent prefix was mutated")
			}
		})
	}
}
